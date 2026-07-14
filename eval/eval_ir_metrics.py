#!/usr/bin/env python3
"""
Retrieval IR metrics against the labeled gold set (eval/gold_set.json).

Measures the PRODUCTION retrieval configuration (nprobe from server/settings.py)
with classic, deterministic information-retrieval metrics — no LLM judge at eval
time (labels were produced once by build_gold_set.py):

  * Recall@k    — fraction of relevant docs found in the top-k
  * Precision@k — fraction of the top-k that are relevant
  * MRR         — 1 / rank of the first relevant doc
  * nDCG@k      — rank-weighted relevance (binary gains)

Usage:
  python eval/build_gold_set.py        # once, to create gold_set.json
  python eval/eval_ir_metrics.py
  python eval/eval_ir_metrics.py --nprobe 32 --ks 5,10,20
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

os.environ.setdefault("OMP_NUM_THREADS", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import faiss  # noqa: E402

INDEX_PATH = PROJECT_ROOT / "output" / "pubmed_ivfpq.faiss"
GOLD_PATH = Path(__file__).resolve().parent / "gold_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def precision_at_k(ranked: List[int], rel: set, k: int) -> float:
    top = ranked[:k]
    return sum(1 for p in top if p in rel) / k if k else float("nan")


def recall_at_k(ranked: List[int], rel: set, k: int) -> float:
    if not rel:
        return float("nan")
    top = ranked[:k]
    return sum(1 for p in top if p in rel) / len(rel)


def reciprocal_rank(ranked: List[int], rel: set) -> float:
    for i, p in enumerate(ranked, 1):
        if p in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: List[int], rel: set, k: int) -> float:
    dcg = sum((1.0 / math.log2(i + 1)) for i, p in enumerate(ranked[:k], 1) if p in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return (dcg / ideal) if ideal > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="IR metrics vs gold set")
    ap.add_argument("--nprobe", type=int, default=0, help="0 => use settings.nprobe")
    ap.add_argument("--ks", type=str, default="5,10,20")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    if not GOLD_PATH.exists():
        raise SystemExit(f"Gold set not found: {GOLD_PATH}. Run build_gold_set.py first.")
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    queries = gold["queries"]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    topk = max(args.topk, max(ks))

    try:
        from server.settings import settings
        nprobe = args.nprobe or settings.nprobe
    except Exception:
        nprobe = args.nprobe or 32

    from search import embed_query
    q_texts = [q["query"] for q in queries]
    print(f"Embedding {len(q_texts)} gold queries…")
    qv = embed_query(q_texts).astype(np.float32)
    faiss.normalize_L2(qv)

    print(f"Opening index (mmap), evaluating at PRODUCTION nprobe={nprobe}…")
    index = faiss.read_index(str(INDEX_PATH), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    index.nprobe = nprobe
    _, I = index.search(qv, topk)

    per_query = []
    agg: Dict[str, List[float]] = {}
    for qi, q in enumerate(queries):
        rel = set(int(p) for p in q["relevant_pmids"])
        ranked = [int(p) for p in I[qi] if p >= 0]
        if not rel:
            print(f"  (skip, no relevant labels) {q['query'][:50]}")
            continue
        m = {"query": q["query"], "n_relevant": len(rel)}
        for k in ks:
            m[f"precision@{k}"] = round(precision_at_k(ranked, rel, k), 4)
            m[f"recall@{k}"] = round(recall_at_k(ranked, rel, k), 4)
            m[f"ndcg@{k}"] = round(ndcg_at_k(ranked, rel, k), 4)
        m["mrr"] = round(reciprocal_rank(ranked, rel), 4)
        per_query.append(m)
        for key, val in m.items():
            if isinstance(val, float):
                agg.setdefault(key, []).append(val)

    mean = {k: round(float(np.nanmean(v)), 4) for k, v in agg.items()}

    print(f"\n=== RETRIEVAL IR METRICS (nprobe={nprobe}, {len(per_query)} queries) ===")
    for k in ks:
        print(f"  @{k:<2d}  precision={mean[f'precision@{k}']:.3f}  "
              f"recall={mean[f'recall@{k}']:.3f}  ndcg={mean[f'ndcg@{k}']:.3f}")
    print(f"  MRR = {mean['mrr']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"ir_metrics_{ts}.json"
    out.write_text(json.dumps({"nprobe": nprobe, "ks": ks, "n_queries": len(per_query),
                               "mean": mean, "per_query": per_query}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
