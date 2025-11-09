#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import uvicorn
import os

# Local imports from your repo
from server.settings import settings
import indexing  # your FAISS helpers
from search import (
    load_index, build_metadata_index, embed_query,
    _dedup_results, DEFAULT_NPROBE, DEFAULT_TOPK
)
from generation import (
    build_prompt, reference_legend, call_haiku, save_markdown
)

app = FastAPI(title="ContextMD API", version="0.1.0")

# CORS (adjust as you like)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Models ----------
class Patient(BaseModel):
    age: str | None = None
    sex: str | None = None
    complaint: str | None = None
    history: str | None = None
    meds: str | None = None
    vitals: str | None = None
    labs: str | None = None
    exam: str | None = None
    notes: str | None = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Free-text query (e.g., patient summary)")
    topk: int = Field(default=settings.topk)
    nprobe: int = Field(default=settings.nprobe)
    prefer_abstract_topk: int = Field(default=settings.prefer_abstract_topk)

class Hit(BaseModel):
    pmid: int
    score: float
    title: str | None = ""
    abstract: str | None = ""

class SearchResponse(BaseModel):
    results: List[Hit]

class CaseCreateRequest(BaseModel):
    patient: Patient
    query: str
    topk: int = Field(default=20)

class CaseCreateResponse(BaseModel):
    case_path: str
    hits_saved: int

class GenerateRequest(BaseModel):
    # Either supply a case_path (from CaseCreateResponse) or inline objects:
    case_path: str | None = None
    patient: Patient | None = None
    hits: List[Hit] | None = None
    api_key: str | None = None  # optional override

class GenerateResponse(BaseModel):
    markdown_path: str
    preview: str

# ---------- Globals loaded at startup ----------
_index = None
_meta_db: Dict[int, tuple[str, str]] = {}
_loaded_ok = False

@app.on_event("startup")
def _startup() -> None:
    global _index, _meta_db, _loaded_ok
    # Thread/env hints for FAISS stability on macOS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("FAISS_NTHREADS", "1")

    # Load FAISS index
    if not settings.index_path.exists():
        raise RuntimeError(f"FAISS index not found: {settings.index_path}")
    _index = load_index(settings.index_path, settings.nprobe)

    # Load metadata once (PMID->(title, abstract))
    _meta_db = build_metadata_index(settings.data_dir)
    _loaded_ok = True
    print("[startup] Index and metadata ready.")

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": _loaded_ok,
        "index_path": str(settings.index_path),
        "ntotal": getattr(_index, "ntotal", "NA") if _index else None,
        "metadata_size": len(_meta_db),
        "nprobe_default": settings.nprobe,
    }

# ---------- Helpers ----------
def _search_impl(query: str, topk: int, nprobe: int, prefer_abstract_topk: int) -> List[Dict[str, Any]]:
    if _index is None:
        raise HTTPException(500, "Index not loaded")

    # Embed
    qv = embed_query([query])  # [1, d]
    # Search
    D, I = indexing.search(_index, qv, topk, nprobe) if hasattr(indexing, "search") else _index.search(qv, topk)
    pmids = [int(x) for x in I[0]]
    scores = [float(x) for x in D[0]]

    # Attach metadata + dedup
    results = []
    for pmid, score in zip(pmids, scores):
        title, abstract = _meta_db.get(pmid, ("", ""))
        results.append({
            "pmid": pmid,
            "score": score,
            "title": title or "",
            "abstract": (abstract or "").strip()
        })
    results = _dedup_results(results)

    # Prefer abstracts for display subset
    with_abs = [r for r in results if r["abstract"]]
    if len(with_abs) >= prefer_abstract_topk:
        display = with_abs[:prefer_abstract_topk]
    else:
        need = prefer_abstract_topk - len(with_abs)
        fallback = [r for r in results if not r["abstract"]][:max(0, need)]
        display = with_abs + fallback

    return display

# ---------- Endpoints ----------
@app.post("/api/search", response_model=SearchResponse)
def api_search(body: SearchRequest):
    hits = _search_impl(
        query=body.query,
        topk=body.topk,
        nprobe=body.nprobe,
        prefer_abstract_topk=body.prefer_abstract_topk,
    )
    return {"results": hits}

@app.post("/api/cases", response_model=CaseCreateResponse)
def api_create_case(body: CaseCreateRequest):
    # Pull a bit deeper (20) then keep the best 5 with abstracts
    raw = _search_impl(body.query, topk=max(20, body.topk), nprobe=settings.nprobe, prefer_abstract_topk=5)
    # Save case file
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    case_path = settings.cases_dir / f"case_{ts}.json"
    case_json = {
        "patient": body.patient.dict(),
        "retrieved": raw
    }
    case_path.write_text(__import__("json").dumps(case_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"case_path": str(case_path), "hits_saved": len(raw)}

@app.post("/api/generate", response_model=GenerateResponse)
def api_generate(body: GenerateRequest):
    # Load case either from disk or from payload
    if body.case_path:
        p = Path(body.case_path)
        if not p.exists():
            raise HTTPException(404, f"Case file not found: {p}")
        case = __import__("json").loads(p.read_text(encoding="utf-8"))
        patient = case.get("patient", {}) or {}
        hits = case.get("retrieved", []) or []
        case_path = p
    else:
        if not body.patient or not body.hits:
            raise HTTPException(400, "Provide either case_path OR (patient + hits).")
        patient = body.patient.dict()
        hits = [h.dict() if hasattr(h, "dict") else h for h in body.hits]
        # Save a temporary case so outputs line up with your current workflow
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        case_path = settings.cases_dir / f"case_{ts}.json"
        case_path.write_text(__import__("json").dumps({"patient": patient, "retrieved": hits}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build prompt and call Haiku
    prompt = build_prompt(patient, hits)
    legend = reference_legend(hits)
    # Allow passing API key directly for local testing
    api_key = body.api_key or settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key  # used by call_haiku
    output = call_haiku(prompt)
    output_with_legend = output + "\n\n---\n" + legend + "\n"

    md_path = save_markdown(case_path, output_with_legend)
    return {"markdown_path": str(md_path), "preview": output_with_legend}

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)