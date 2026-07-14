#!/usr/bin/env python3
"""
FAISS approximate-vs-exact retrieval quality sweep.

The deployed index is an IVF-Flat index queried with a fixed `nprobe`
(see server/settings.py, search.py). IVF search is *approximate*: it only
scans `nprobe` of the `nlist` clusters, so it can miss true neighbors.

This script measures exactly how much recall that costs, with NO labels and
NO LLM judge required:

  * ground truth  = the SAME index with nprobe = nlist  (scans every cluster
                    => exact inner-product KNN for an IVF-Flat index)
  * approximate   = the same index at each nprobe in the sweep

For each nprobe we report Recall@k (overlap with the exact top-k) and mean
query latency, so you can pick an nprobe on the recall/latency frontier.

Usage:
  python eval/eval_retrieval_faiss.py
  python eval/eval_retrieval_faiss.py --nprobes 1,8,32,64,128 --topk 10
  python eval/eval_retrieval_faiss.py --queries my_queries.txt --repeats 3

Notes:
  * The 21 GB index is opened with mmap (IO_FLAG_MMAP) so it does not need to
    fit in RAM. The exact baseline (nprobe=nlist) is slow (~25 s/query); keep
    the query set modest.
"""
from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # make root modules (search.py) importable
INDEX_PATH = PROJECT_ROOT / "output" / "pubmed_ivfpq.faiss"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# A diverse set of realistic clinical queries (the kind the app receives).
DEFAULT_QUERIES = [
    "lung cancer with EGFR mutation, confirm testing & first-line options",
    "preoperative cardiac risk assessment in elderly patients",
    "management of severe sepsis and septic shock in the ICU",
    "type 2 diabetes second-line therapy after metformin",
    "anticoagulation for atrial fibrillation stroke prevention",
    "HER2-positive breast cancer targeted therapy",
    "COPD exacerbation treatment guidelines",
    "acute kidney injury diagnosis and management",
    "first-line antidepressant selection for major depression",
    "biologic therapy for rheumatoid arthritis",
    "community-acquired pneumonia antibiotic treatment",
    "colorectal cancer screening recommendations",
]


def load_queries(path: str | None) -> List[str]:
    if not path:
        return DEFAULT_QUERIES
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def recall_at_k(approx_ids: np.ndarray, exact_ids: np.ndarray, k: int) -> float:
    """Fraction of the exact top-k that appear anywhere in the approx top-k."""
    a = set(int(x) for x in approx_ids[:k] if x >= 0)
    e = set(int(x) for x in exact_ids[:k] if x >= 0)
    if not e:
        return float("nan")
    return len(a & e) / len(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="FAISS exact-vs-approx recall sweep")
    ap.add_argument("--nprobes", type=str, default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--topk", type=int, default=10, help="retrieval depth to score")
    ap.add_argument("--ks", type=str, default="5,10", help="Recall@k cutoffs to report")
    ap.add_argument("--queries", type=str, default="", help="optional file, one query per line")
    ap.add_argument("--repeats", type=int, default=1, help="latency-averaging repeats per query")
    ap.add_argument("--exact_nprobe", type=int, default=0, help="0 => use nlist (true exact)")
    args = ap.parse_args()

    if not INDEX_PATH.exists():
        raise SystemExit(f"Index not found: {INDEX_PATH}. Run indexing.py first.")

    nprobes = [int(x) for x in args.nprobes.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    topk = max(args.topk, max(ks))
    queries = load_queries(args.queries or None)

    # Embed queries with the SAME MedCPT encoder the app uses.
    print(f"Embedding {len(queries)} queries with MedCPT Query Encoder…")
    from search import embed_query  # lazy: avoids torch import unless needed
    qv = embed_query(queries).astype(np.float32)
    faiss.normalize_L2(qv)

    print(f"Opening index (mmap): {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    ivf = faiss.extract_index_ivf(index)
    nlist = ivf.nlist
    exact_nprobe = args.exact_nprobe or nlist
    print(f"ntotal={index.ntotal:,}  d={index.d}  nlist={nlist}  exact_nprobe={exact_nprobe}")

    # ---- Exact ground truth (nprobe = nlist) ----
    print(f"\nComputing EXACT ground truth at nprobe={exact_nprobe} "
          f"(~slow; {len(queries)} queries)…")
    index.nprobe = exact_nprobe
    t0 = time.time()
    _, exact_I = index.search(qv, topk)
    print(f"  exact baseline done in {time.time()-t0:.1f}s")

    # ---- Sweep ----
    rows: List[Dict] = []
    print("\nSweeping nprobe…")
    for npr in nprobes:
        index.nprobe = npr
        # latency: average per-query over repeats (single-query searches = realistic)
        per_q_latencies = []
        for _ in range(max(1, args.repeats)):
            for i in range(len(queries)):
                t = time.time()
                index.search(qv[i:i + 1], topk)
                per_q_latencies.append((time.time() - t) * 1000.0)
        # recall: one batched search is fine for correctness
        _, approx_I = index.search(qv, topk)
        rec = {k: float(np.nanmean([recall_at_k(approx_I[i], exact_I[i], k)
                                    for i in range(len(queries))])) for k in ks}
        row = {
            "nprobe": npr,
            "mean_latency_ms": round(float(np.mean(per_q_latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(per_q_latencies, 95)), 2),
            **{f"recall@{k}": round(rec[k], 4) for k in ks},
        }
        rows.append(row)
        rec_str = "  ".join(f"R@{k}={row[f'recall@{k}']:.3f}" for k in ks)
        print(f"  nprobe={npr:5d}  {rec_str}  lat={row['mean_latency_ms']:.1f}ms")

    # ---- Report ----
    print("\n=== FAISS RECALL vs nprobe (ground truth = exact IVF scan) ===")
    hdr = ["nprobe"] + [f"recall@{k}" for k in ks] + ["mean_latency_ms", "p95_latency_ms"]
    print(" | ".join(f"{h:>14}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{r[h]:>14}" for h in hdr))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    meta = {"queries": len(queries), "topk": topk, "nlist": nlist,
            "exact_nprobe": exact_nprobe, "ntotal": int(index.ntotal)}
    (RESULTS_DIR / f"retrieval_sweep_{ts}.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")
    with (RESULTS_DIR / f"retrieval_sweep_{ts}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows([{h: r[h] for h in hdr} for r in rows])
    print(f"\nSaved: eval/results/retrieval_sweep_{ts}.json (+ .csv)")


if __name__ == "__main__":
    main()
