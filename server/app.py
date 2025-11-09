#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import uvicorn
import os
import json
from datetime import datetime

# Local imports from your repo (fail fast if missing)
from server.settings import settings

import indexing
from search import (
    load_index, build_metadata_index, embed_query,
    _dedup_results, DEFAULT_NPROBE, DEFAULT_TOPK
)

from generation import (
    build_prompt, reference_legend, call_haiku, save_markdown
)

app = FastAPI(title="ContextMD API", version="0.1.0")

# CORS (honor configured origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "cors_origins", ["http://localhost:8501", "http://127.0.0.1:8501"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class Patient(BaseModel):
    age: Optional[str] = None
    sex: Optional[str] = None
    complaint: Optional[str] = None
    history: Optional[str] = None
    meds: Optional[str] = None
    vitals: Optional[str] = None
    labs: Optional[str] = None
    exam: Optional[str] = None
    notes: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Free-text query (e.g., patient summary)")
    topk: int = Field(default=20)
    nprobe: int = Field(default=32)
    prefer_abstract_topk: int = Field(default=5)

class Hit(BaseModel):
    pmid: int
    score: float
    title: Optional[str] = ""
    abstract: Optional[str] = ""

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
    case_path: Optional[str] = None
    patient: Optional[Patient] = None
    hits: Optional[List[Hit]] = None
    api_key: Optional[str] = None

class GenerateResponse(BaseModel):
    markdown_path: str
    preview: str

# --- New models for one-shot run ---
class RunRequest(BaseModel):
    patient: Patient
    query: str
    topk: int = Field(default=20)
    api_key: Optional[str] = None

class RunResponse(BaseModel):
    case_path: str
    markdown_path: str
    preview: str
    hits_saved: int

# ---------- Globals loaded at startup ----------
_index = None
_meta_db: Dict[int, tuple] = {}
_loaded_ok = False

@app.on_event("startup")
def _startup() -> None:
    global _index, _meta_db, _loaded_ok
    
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("FAISS_NTHREADS", "1")

    # Ensure cases directory exists
    settings.cases_dir.mkdir(parents=True, exist_ok=True)

    # Try to load FAISS index if it exists
    if settings.index_path.exists():
        try:
            _index = load_index(settings.index_path, settings.nprobe)
            _meta_db = build_metadata_index(settings.data_dir)
            _loaded_ok = True
            print(f"[startup] Index loaded: {settings.index_path}")
        except Exception as e:
            print(f"[startup] Warning: Could not load index: {e}")
            _loaded_ok = False
    else:
        print(f"[startup] Index file not found at {settings.index_path} - running in demo mode")
        _loaded_ok = False

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": _loaded_ok,
        "index_path": str(settings.index_path),
        "ntotal": getattr(_index, "ntotal", "N/A") if _index else "N/A",
        "metadata_size": len(_meta_db),
        "nprobe_default": settings.nprobe,
        "mode": "full" if _loaded_ok else "demo",
    }

# ---------- Helpers ----------
def _search_impl(query: str, topk: int, nprobe: int, prefer_abstract_topk: int) -> List[Dict[str, Any]]:
    """Strict search: error if index/metadata not ready."""
    if not _loaded_ok or _index is None or not _meta_db:
        raise HTTPException(status_code=503, detail="Search unavailable: FAISS index or metadata not loaded")

    # Embed
    qv = embed_query([query])  # [1, d]
    
    # Search
    if hasattr(indexing, "search"):
        D, I = indexing.search(_index, qv, topk, nprobe)
    else:
        D, I = _index.search(qv, topk)
    
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
    try:
        hits = _search_impl(
            query=body.query,
            topk=body.topk,
            nprobe=body.nprobe,
            prefer_abstract_topk=body.prefer_abstract_topk,
        )
        return {"results": hits}
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")

@app.post("/api/cases", response_model=CaseCreateResponse)
def api_create_case(body: CaseCreateRequest):
    try:
        # Search
        raw = _search_impl(
            body.query,
            topk=max(20, body.topk),
            nprobe=settings.nprobe,
            prefer_abstract_topk=settings.prefer_abstract_topk
        )
        
        # Save case file
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        case_path = settings.cases_dir / f"case_{ts}.json"
        
        case_json = {
            "patient": body.patient.dict(),
            "retrieved": raw,
            "query": body.query,
            "timestamp": ts,
        }
        
        case_path.write_text(
            json.dumps(case_json, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        return {"case_path": str(case_path), "hits_saved": len(raw)}
    
    except Exception as e:
        raise HTTPException(500, f"Case creation failed: {str(e)}")

@app.post("/api/generate", response_model=GenerateResponse)
def api_generate(body: GenerateRequest):
    try:
        # Load case either from disk or from payload
        if body.case_path:
            p = Path(body.case_path)
            if not p.exists():
                raise HTTPException(404, f"Case file not found: {p}")
            
            case = json.loads(p.read_text(encoding="utf-8"))
            patient = case.get("patient", {}) or {}
            hits = case.get("retrieved", []) or []
            case_path = p
        else:
            if not body.patient or not body.hits:
                raise HTTPException(400, "Provide either case_path OR (patient + hits).")
            
            patient = body.patient.dict()
            hits = [h.dict() if hasattr(h, "dict") else h for h in body.hits]
            
            # Save temporary case
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            case_path = settings.cases_dir / f"case_{ts}.json"
            case_path.write_text(
                json.dumps({"patient": patient, "retrieved": hits}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        # Build prompt and call Haiku
        prompt = build_prompt(patient, hits)
        legend = reference_legend(hits)
        
        # Set API key if provided or available in env
        api_key = body.api_key or settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        
        output = call_haiku(prompt)
        output_with_legend = output + "\n\n---\n" + legend + "\n"

        md_path = save_markdown(case_path, output_with_legend)
        
        return {
            "markdown_path": str(md_path),
            "preview": output_with_legend
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


# ----------- One-shot endpoint: /api/run -----------
@app.post("/api/run", response_model=RunResponse)
def api_run(body: RunRequest):
    """Create a case (search + save) and then generate the Haiku report in one call."""
    try:
        # 1) Search (prefer abstracts but get up to max for quality)
        hits = _search_impl(
            query=body.query,
            topk=max(20, body.topk),
            nprobe=settings.nprobe,
            prefer_abstract_topk=settings.prefer_abstract_topk,
        )

        # 2) Save case JSON
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        case_path = settings.cases_dir / f"case_{ts}.json"
        payload = {
            "patient": body.patient.dict(),
            "retrieved": hits,
            "query": body.query,
            "timestamp": ts,
        }
        case_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # 3) Build prompt and call Haiku
        prompt = build_prompt(payload["patient"], hits)
        legend = reference_legend(hits)

        api_key = body.api_key or settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

        output = call_haiku(prompt)
        output_with_legend = output + "\n\n---\n" + legend + "\n"
        md_path = save_markdown(case_path, output_with_legend)

        return RunResponse(
            case_path=str(case_path),
            markdown_path=str(md_path),
            preview=output_with_legend,
            hits_saved=len(hits),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Run failed: {str(e)}")