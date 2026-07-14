#!/usr/bin/env python3
"""Shared helpers for the ContextMD evaluation suite.

Keeps a single source of truth for: locating case/markdown pairs, reconstructing
the exact question the pipeline acted on, extracting the model's answer (minus
the auto-generated reference legend), and parsing [#] citations.
"""
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "Cases"

PATIENT_FIELDS = ["age", "sex", "complaint", "history", "meds", "vitals", "labs", "exam", "notes"]


def find_case_pairs(cases_dir: Path = CASES_DIR) -> List[Tuple[Path, Path]]:
    """Return [(case_json, haiku_md), ...] for every case that has a report."""
    pairs = []
    for cj in sorted(cases_dir.glob("case_*.json")):
        md = cj.with_name(cj.stem + ".haiku.md")
        if md.exists():
            pairs.append((cj, md))
    return pairs


def latest_case_pair(cases_dir: Path = CASES_DIR) -> Tuple[Path, Path]:
    pairs = find_case_pairs(cases_dir)
    if not pairs:
        raise FileNotFoundError(f"No case_*.json with a matching .haiku.md in {cases_dir}")
    return pairs[-1]


def load_case(case_path: Path) -> Dict[str, Any]:
    return json.loads(case_path.read_text(encoding="utf-8"))


def build_question(patient: Dict[str, Any], query: str) -> str:
    """Reconstruct the EXACT input the pipeline acted on: the clinical `query`
    that drove retrieval, enriched with the patient context fed to generation."""
    parts = [f"{k}: {patient.get(k)}" for k in PATIENT_FIELDS if patient.get(k)]
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
    """Only the LLM-generated Summary + Insights; strips the appended legend."""
    cut = md_text.lower().find("### reference legend")
    if cut != -1:
        md_text = md_text[:cut].rstrip().rstrip("-").rstrip()
    summary: List[str] = []
    insights: List[str] = []
    cur: Optional[List[str]] = None
    for line in md_text.splitlines():
        s = line.strip().lower()
        if s.startswith("## summary"):
            cur = summary
            continue
        if s.startswith("## insights"):
            cur = insights
            continue
        if cur is not None:
            cur.append(line)
    text = "\n".join(summary + [""] + insights).strip()
    return text if text else md_text


# ---- Citation parsing ----------------------------------------------------

_CITE_RE = re.compile(r"\[(\d+)(?:\s*;[^\]]*)?\]")  # [3] or [3; PMID 123]


def claim_lines(answer_text: str) -> List[str]:
    """Substantive content lines (bullets / sentences), excluding headers/blanks."""
    out = []
    for line in answer_text.splitlines():
        s = line.strip().lstrip("-*0123456789. ").strip()
        if len(s) >= 25 and any(c.isalpha() for c in s):  # ignore stubs/headers
            out.append(s)
    return out


def citations_in(text: str) -> List[int]:
    return [int(m.group(1)) for m in _CITE_RE.finditer(text)]


def ref_context(case: Dict[str, Any], cite_idx: int) -> Optional[Dict[str, str]]:
    """Map a [#] citation to its source abstract via the case's retrieved order."""
    retrieved = case.get("retrieved", []) or []
    if 1 <= cite_idx <= len(retrieved):
        h = retrieved[cite_idx - 1]
        return {
            "pmid": str(h.get("pmid", "")),
            "title": (h.get("title") or "").strip(),
            "abstract": (h.get("abstract") or "").strip(),
        }
    return None
