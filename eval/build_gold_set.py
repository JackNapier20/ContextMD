#!/usr/bin/env python3
"""
Build a labeled retrieval gold set (query -> relevant PMIDs) for IR metrics.

Hand-labeling relevance across 7.3M abstracts is infeasible, so we use the
standard *pooling* method with an LLM annotator:

  1. For each clinical query, retrieve a DEEP pool of candidates at high nprobe
     (near-exact recall) so the pool is close to complete.
  2. Claude judges each candidate abstract as relevant / not-relevant to the query.
  3. Save {query -> relevant PMIDs} to eval/gold_set.json.

eval_ir_metrics.py then scores the *production* retrieval (nprobe=32) against
these labels with Recall@k / Precision@k / MRR / nDCG@k. Because the pool is
drawn at high nprobe, recall for lower nprobe settings is meaningful.

Limitation (documented honestly): candidates come from the retriever's own deep
pool, so a truly relevant doc the encoder never surfaces at high nprobe cannot be
labeled. This measures ranking quality + recall-relative-to-pool, not absolute
recall over the whole corpus.

Requires ANTHROPIC_API_KEY. Uses the mmap'd index — run when nothing else is
hammering the index (24 GB RAM).

Usage:
  python eval/build_gold_set.py
  python eval/build_gold_set.py --pool_nprobe 512 --pool_depth 30
"""
from __future__ import annotations

import os
import re
import sys
import json
import glob
import argparse
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import orjson  # noqa: E402
import faiss  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env", override=False)

from anthropic import Anthropic  # noqa: E402

INDEX_PATH = PROJECT_ROOT / "output" / "pubmed_ivfpq.faiss"
PUBMED_DIR = PROJECT_ROOT / "data" / "MedCPT" / "pubmed"
GOLD_PATH = Path(__file__).resolve().parent / "gold_set.json"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"

QUERIES = [
    "lung cancer with EGFR mutation, confirm testing & first-line options",
    "preoperative cardiac risk assessment in elderly patients",
    "type 2 diabetes second-line therapy after metformin",
    "anticoagulation for atrial fibrillation stroke prevention",
    "HER2-positive breast cancer targeted therapy",
    "community-acquired pneumonia antibiotic treatment",
    "biologic therapy for rheumatoid arthritis",
    "management of severe sepsis and septic shock",
]

JUDGE_INSTR = (
    "You are a biomedical librarian assessing search relevance. Given a QUERY and "
    "a numbered list of candidate abstracts, decide for each whether it is "
    "RELEVANT — i.e. it could directly help answer the clinical query (same "
    "condition/topic and useful content). Off-topic or only tangentially related "
    "abstracts are NOT relevant.\n"
    "Respond ONLY with a JSON array, one object per candidate in order:\n"
    '[{"i": <n>, "relevant": true|false}]'
)


def load_metadata_for(wanted: set, pubmed_dir: Path) -> dict:
    """Memory-safe: parse one 1.6 GB dict-of-dicts file at a time, keep only wanted PMIDs."""
    out = {}
    for pf in sorted(glob.glob(str(pubmed_dir / "pubmed_chunk_*.json"))):
        raw = orjson.loads(Path(pf).read_bytes())  # {pmid_str: {"t":.., "a":..}}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    pmid = int(k)
                except Exception:
                    continue
                if pmid in wanted and isinstance(v, dict):
                    out[pmid] = ((v.get("t") or v.get("title") or "").strip(),
                                 (v.get("a") or v.get("abstract") or "").strip())
        del raw
        if len(out) >= len(wanted):
            break
    return out


def judge_relevance(client: Anthropic, query: str, cands: list) -> list:
    blocks = []
    for i, (pmid, title, abstract) in enumerate(cands, 1):
        blocks.append(f"[{i}] {title}\n{abstract[:1200]}")
    prompt = f"{JUDGE_INSTR}\n\nQUERY: {query}\n\nCANDIDATES:\n" + "\n\n".join(blocks)
    msg = client.messages.create(model=JUDGE_MODEL, max_tokens=1500, temperature=0.0,
                                 messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    m = re.search(r"\[.*\]", text, re.DOTALL)
    verdicts = json.loads(m.group(0)) if m else []
    by_i = {int(v.get("i", idx + 1)): bool(v.get("relevant")) for idx, v in enumerate(verdicts)}
    return [by_i.get(i, False) for i in range(1, len(cands) + 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build retrieval gold set via pooling + Claude")
    ap.add_argument("--pool_nprobe", type=int, default=512, help="near-exact recall for pooling")
    ap.add_argument("--pool_depth", type=int, default=30, help="candidates per query to judge")
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set.")
    client = Anthropic(api_key=key)

    from search import embed_query
    print(f"Embedding {len(QUERIES)} queries…")
    qv = embed_query(QUERIES).astype(np.float32)
    faiss.normalize_L2(qv)

    print(f"Opening index (mmap), pooling at nprobe={args.pool_nprobe}, depth={args.pool_depth}…")
    index = faiss.read_index(str(INDEX_PATH), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    index.nprobe = args.pool_nprobe
    D, I = index.search(qv, args.pool_depth)

    # Gather all pooled PMIDs, load their metadata once (memory-safe).
    wanted = {int(p) for row in I for p in row if p >= 0}
    print(f"Loading metadata for {len(wanted)} pooled PMIDs…")
    meta = load_metadata_for(wanted, PUBMED_DIR)

    gold = {"pool_nprobe": args.pool_nprobe, "pool_depth": args.pool_depth,
            "judge_model": JUDGE_MODEL, "queries": []}
    for qi, query in enumerate(QUERIES):
        cands = []
        for p in I[qi]:
            p = int(p)
            if p < 0:
                continue
            title, abstract = meta.get(p, ("", ""))
            if abstract:  # only judge candidates that actually have text
                cands.append((p, title, abstract))
        rels = judge_relevance(client, query, cands)
        relevant = [pmid for (pmid, _, _), r in zip(cands, rels) if r]
        gold["queries"].append({
            "query": query,
            "pool_pmids": [int(p) for p in I[qi] if p >= 0],
            "judged_pmids": [c[0] for c in cands],
            "relevant_pmids": relevant,
        })
        print(f"  {query[:50]:50s}  relevant={len(relevant)}/{len(cands)}")

    GOLD_PATH.write_text(json.dumps(gold, indent=2), encoding="utf-8")
    print(f"\nSaved gold set: {GOLD_PATH}")


if __name__ == "__main__":
    main()
