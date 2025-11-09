#!/usr/bin/env python3
"""
create_case.py
Given patient data + a search query, retrieve supporting literature and save:
  1) case_<timestamp>.json   (structured bundle)
  2) case_<timestamp>.prompt.txt   (Claude-ready prompt)
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import orjson
import numpy as np

# Import your existing search functions
import search   # <-- important: relies on your working FAISS + embed setup

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "Cases"
DATA_DIR = PROJECT_ROOT / "data" / "MedCPT"


# ---- Dedup helpers (case/whitespace-insensitive on title/abstract) ----
def _norm(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\u200b\ufeff]", "", s)  # remove zero-width/bom
    return s

def _dup_key(title: str, abstract: str):
    t = _norm(title)
    a = _norm(abstract)
    if t:
        return ("t", t)
    if a:
        return ("a", a[:160])  # stable prefix
    return ("none", "")

def _dedup_hits(hits: List[Dict]) -> List[Dict]:
    """Keep first occurrence, but if duplicate appears later with longer abstract, prefer that."""
    best = {}
    for idx, h in enumerate(hits):
        key = _dup_key(h.get("title", ""), h.get("abstract", ""))
        prev = best.get(key)
        if prev is None:
            best[key] = (idx, h)
        else:
            p_idx, p = prev
            if len((h.get("abstract") or "")) > len((p.get("abstract") or "")):
                best[key] = (p_idx, h)
    return [item for _, item in sorted(best.values(), key=lambda x: x[0])]


# --------------------- Parse Patient Inputs ---------------------

def parse_inline_kv(s: str) -> Dict:
    """Parse key=value pairs. Values may be quoted."""
    if not s:
        return {}
    parts = re.findall(r"(\w+)=('([^']*)'|\"([^\"]*)\"|[^\s]+)", s)
    out = {}
    for full, v1, vq1, vq2 in parts:
        val = vq1 or vq2 or v1
        out[full] = val
    return out


def load_patient_json(path_str: str) -> Dict:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"patient_file not found: {path}")

    txt = path.read_text(encoding="utf-8")
    # Try JSON first
    try:
        return json.loads(txt)
    except Exception:
        pass
    # If file extension suggests YAML, try PyYAML
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # optional dependency
        except Exception as e:
            raise RuntimeError("YAML file provided but PyYAML is not installed. Install with `pip install pyyaml`.") from e
        return yaml.safe_load(txt) or {}
    raise ValueError("Unsupported patient_file format. Use JSON or YAML.")


# --------------------- Build Prompt ---------------------

def build_prompt(patient: Dict, hits: List[Dict], max_refs: int = 5) -> str:
    fields = [
        ("Age", patient.get("age")),
        ("Sex", patient.get("sex")),
        ("Chief complaint", patient.get("complaint")),
        ("History", patient.get("history")),
        ("Medications", patient.get("meds")),
        ("Vitals", patient.get("vitals")),
        ("Labs", patient.get("labs")),
        ("Exam", patient.get("exam")),
        ("Imaging", patient.get("imaging")),
        ("Notes", patient.get("notes")),
    ]

    pt_lines = ["PATIENT CASE:"]
    for label, value in fields:
        if value:
            pt_lines.append(f"- {label}: {value}")
    patient_block = "\n".join(pt_lines)

    ref_lines = ["REFERENCES (top matches):"]
    for i, h in enumerate(hits[:max_refs], 1):
        title = (h.get("title") or "").strip()
        abstract = (h.get("abstract") or "").strip().replace("\n", " ")
        if len(abstract) > 600:
            abstract = abstract[:600] + "..."
        ref_lines.append(
            f"[{i}] PMID {h['pmid']} (score={h['score']:.3f}): {title}\n{abstract}"
        )
    refs_block = "\n\n".join(ref_lines)

    instructions = (
        "You are assisting with a *case briefing*.\n"
        "Summarize key differentials, relevant workup, and management *options* using the references.\n"
        "Do NOT provide medical advice or definitive diagnosis. Cite sources by [#].\n"
    )

    return f"{instructions}\n\n{patient_block}\n\n{refs_block}\n\nRESPONSE FORMAT:\n" \
           "- Ranked differentials + reasoning [#]\n" \
           "- Key tests to confirm/triage [#]\n" \
           "- Red flags / escalation triggers\n" \
           "- Management options with rationale [#]\n" \
           "- Final concise 3–5 sentence briefing\n"


# --------------------- Save Case Files ---------------------

def save_case(patient: Dict, hits: List[Dict]):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    bundle = {
        "created_utc": ts,
        "patient": patient,
        "retrieved": hits,
    }

    json_path = RUNS_DIR / f"case_{ts}.json"
    prompt_path = RUNS_DIR / f"case_{ts}.prompt.txt"

    json_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    prompt_path.write_text(build_prompt(patient, hits), encoding="utf-8")

    print(f"\n Case saved:")
    print(f"  JSON:   {json_path}")
    print(f"  Prompt: {prompt_path}")


# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str, help="Search query (usually patient summary)")
    ap.add_argument("--patient", type=str, default="", help="Inline patient fields, e.g. age=65 sex=male complaint='cough'")
    ap.add_argument("--patient_file", type=str, default="", help="JSON/YAML patient record")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--nprobe", type=int, default=64)
    args = ap.parse_args()

    patient = parse_inline_kv(args.patient)
    patient.update(load_patient_json(args.patient_file))

    # Retrieve search results (call search.py as library)
    index = search.load_index(search.INDEX_PATH, args.nprobe)
    qv = search.embed_query([args.query])
    D, I = search.faiss_search(index, qv, args.topk)

    pmids = I[0].tolist()
    scores = D[0].tolist()

    # Load metadata & package hits
    meta_db = search.build_metadata_index(DATA_DIR)
    hits = []
    for pmid, score in zip(pmids, scores):
        entry = {"pmid": pmid, "score": score}
        if meta_db and pmid in meta_db:
            t, a = meta_db[pmid]
            entry.update({"title": t, "abstract": a})
        hits.append(entry)
    # Prefer references that actually have an abstract, but fallback so we always have at least 5 if possible
    hits = _dedup_hits(hits)
    hits = [h for h in hits if (h.get("abstract") or "").strip()][:5]
    save_case(patient, hits)


if __name__ == "__main__":
    main()