#!/usr/bin/env python3
"""
RAG evaluation using Ragas on the latest case + latest Haiku output.

It builds a synthetic question from the patient fields, uses the top-5 retrieved
abstracts as context, and evaluates the saved Haiku markdown (Summary + Insights)
with common Ragas metrics.

Requirements (install once):
  pip install ragas datasets
  # Optional (if using OpenAI as the judge): export OPENAI_API_KEY=sk-...

Usage:
  python ragas_eval.py [--cases_dir ./Cases] [--max_contexts 5] [--out ./Cases]

Outputs:
  - Prints a metric table to stdout
  - Saves JSON and CSV reports next to the case (or in --out)
"""

from __future__ import annotations
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Ragas
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_utilization,
        context_entity_recall,
    )
except Exception as e:
    raise SystemExit(
        "Missing dependency: ragas\nInstall with: pip install ragas datasets"
    ) from e

try:
    from datasets import Dataset
except Exception as e:
    raise SystemExit(
        "Missing dependency: datasets\nInstall with: pip install datasets"
    ) from e

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_DIR = PROJECT_ROOT / "Cases"

# ------------------------ helpers ------------------------

def _latest(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} under {path}")
    return files[-1]

def load_latest_case(cases_dir: Path) -> Path:
    return _latest(cases_dir, "case_*.json")

def load_latest_markdown(cases_dir: Path, case_path: Path) -> Path:
    stem = case_path.stem  # case_YYYYMMDDThhmmssZ
    md = case_path.with_name(stem + ".haiku.md")
    if md.exists():
        return md
    # fallback to newest *.haiku.md if matching file not found
    try:
        return _latest(cases_dir, "case_*.haiku.md")
    except Exception:
        raise FileNotFoundError("No saved Haiku markdown found. Run generation.py first.")

def build_question(patient: Dict[str, Any], query: str) -> str:
    """Reconstruct the EXACT input the pipeline acted on.

    The clinical `query` is what drives MedCPT embedding + FAISS retrieval
    (see search.py / server.app), and the patient block is what generation.py
    feeds into the prompt alongside the retrieved references. Anchoring the
    Ragas question on `query` (plus patient context) makes answer_relevancy and
    context_utilization measure the real task instead of a fabricated one.
    """
    parts = []
    for k in ["age", "sex", "complaint", "history", "meds", "vitals", "labs", "exam", "notes"]:
        v = patient.get(k)
        if v:
            parts.append(f"{k}: {v}")
    patient_ctx = "; ".join(parts)
    query = (query or "").strip()

    if query and patient_ctx:
        return f"{query} (Patient — {patient_ctx})"
    if query:
        return query
    if patient_ctx:
        return f"Clinical guidance for patient — {patient_ctx}"
    return "Clinical question based on patient case"

def extract_answer_text(md_text: str) -> str:
    """Take ONLY the LLM-generated Summary + Insights as the model answer.

    generation.py appends a programmatically-built reference legend
    ('---' + '### Reference legend' + PMIDs/URLs) to every report. That text
    is NOT produced by the model, so it must be stripped before scoring
    faithfulness/relevancy — otherwise we'd grade Claude on boilerplate.
    """
    # Drop the appended legend (and any trailing '---' divider before it).
    lower = md_text.lower()
    cut = lower.find("### reference legend")
    if cut != -1:
        md_text = md_text[:cut].rstrip().rstrip("-").rstrip()

    # Very light parsing: grab everything after '## Summary' and '## Insights'
    # If not found, just return the whole file.
    summary = []
    insights = []
    cur = None
    for line in md_text.splitlines():
        if line.strip().lower().startswith("## summary"):
            cur = summary
            continue
        if line.strip().lower().startswith("## insights"):
            cur = insights
            continue
        if cur is not None:
            cur.append(line)
    text = "\n".join(summary + [""] + insights).strip()
    return text if text else md_text

def contexts_from_hits(hits: List[Dict[str, Any]], max_ctx: int) -> List[str]:
    ctx = []
    for h in hits[:max_ctx]:
        title = (h.get("title") or "").strip()
        abstract = (h.get("abstract") or "").strip()
        piece = title
        if abstract:
            piece = f"{title}. {abstract}" if title else abstract
        ctx.append(piece)
    return ctx

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate latest case with Ragas")
    ap.add_argument("--cases_dir", type=str, default=str(DEFAULT_CASES_DIR))
    ap.add_argument("--max_contexts", type=int, default=5, help="how many retrieved items to pass as context")
    ap.add_argument("--out", type=str, default=str(DEFAULT_CASES_DIR), help="folder to store eval reports")
    args = ap.parse_args()

    cases_dir = Path(args.cases_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_path = load_latest_case(cases_dir)
    case = json.loads(case_path.read_text(encoding="utf-8"))

    md_path = load_latest_markdown(cases_dir, case_path)
    md_text = md_path.read_text(encoding="utf-8")

    question = build_question(case.get("patient", {}) or {}, case.get("query", ""))
    answer = extract_answer_text(md_text)
    contexts = contexts_from_hits(case.get("retrieved", []) or [], args.max_contexts)

    print(f"\nEvaluating case: {case_path.name}")
    print(f"  Question (drives retrieval+generation): {question}")
    print(f"  Answer length: {len(answer)} chars | Contexts: {len(contexts)} abstracts\n")

    # Build the HuggingFace Dataset expected by ragas
    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })

    # faithfulness and answer_relevancy work without ground truth;
    # context_utilization measures how well retrieved chunks are used in the answer
    metrics = [
        faithfulness,
        answer_relevancy,
        context_utilization,
    ]

    # Run evaluation. Ragas will use your default LLM provider (e.g., OPENAI_API_KEY) if set.
    print("Running Ragas evaluation… (this uses your LLM API credits)")
    result = evaluate(data, metrics=metrics)

    # Pretty print
    print("\n=== RAGAS METRICS ===")
    df = result.to_pandas()
    # df has one row per sample; print the row with rounded scores
    row = df.iloc[0].to_dict()
    for k, v in row.items():
        if isinstance(v, float):
            row[k] = round(v, 4)
    for k in [m.name for m in metrics]:
        if k in row:
            print(f"{k:>24}: {row[k]}")

    # Save reports
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_out = out_dir / f"eval_{ts}.json"
    csv_out = out_dir / f"eval_{ts}.csv"
    json_out.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    df.to_csv(csv_out, index=False)
    print(f"\nSaved evaluation to:\n  {json_out}\n  {csv_out}")

    # Helpful tips
    print("\nTips:")
    print("- Set OPENAI_API_KEY (or configure Ragas to use your preferred LLM provider) for consistent scoring.")
    print("- For dataset-level evaluation, run multiple cases and concatenate rows before calling evaluate().")
    print("- To add ground-truth answers and use answer_correctness, extend this script with a reference column.")

if __name__ == "__main__":
    main()