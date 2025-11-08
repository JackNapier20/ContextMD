#!/usr/bin/env python3
# search_pubmed_faiss (search script for IVF-Flat + OnDisk FAISS)

import re
import argparse
import indexing
import orjson
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, List
# NEW IMPORTS
from transformers import AutoTokenizer, AutoModel
import os

# -------- Config --------
PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT / "output/pubmed_ivfpq.faiss"
ONDISK_DIR = PROJECT_ROOT / "output/ondisk_lists"     # Must match build config
DATA_DIR = PROJECT_ROOT / "data/MedCPT"  # Root containing embeds/ pmids/ pubmed/
MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
DEFAULT_NPROBE = 64
DEFAULT_TOPK = 10

# Must match how the index was built
USE_ONDISK = False  # set True only if indexing used OnDiskInvertedLists

# -------- Embedding --------
# -------- Embedding --------
_tok = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _lazy_load_medcpt():
    global _tok, _model
    if _tok is None or _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()

def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Mean pool over valid tokens only
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts

def embed_query(texts: List[str]) -> np.ndarray:
    """Return float32 [N, d] normalized embeddings using MedCPT Query Encoder."""
    _lazy_load_medcpt()
    with torch.no_grad():
        enc = _tok(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128
        ).to(_device)
        out = _model(**enc)
        sent = _mean_pooling(out.last_hidden_state, enc["attention_mask"])  # mean pool (robust for MedCPT)
        sent = torch.nn.functional.normalize(sent, p=2, dim=1)
    arr = sent.cpu().numpy().astype(np.float32, copy=True)
    return arr
# -------- Utilities --------

def build_metadata_index(data_dir: Path) -> Dict[int, Tuple[str, str]]:
    """
    (Optional) Builds an in-memory map {pmid: (title, abstract)}
    from the pubmed_chunk_*.json files.
    """
    meta = {}
    pubmed_files = sorted((data_dir / "pubmed").glob("pubmed_chunk_*.json"))
    if not pubmed_files:
        print(f"Warning: No 'pubmed_chunk_*.json' files found in {data_dir}. Metadata will be unavailable.")
        return {}

    print(f"Loading metadata from {len(pubmed_files)} files...")
    for pf in pubmed_files:
        with pf.open("rb") as f:
            items = orjson.loads(f.read())
        
        for it in items:
            pmid = int(it.get("pmid", -1))
            if pmid > 0 and pmid not in meta:
                title = it.get("title", "")
                abstract = it.get("abstract") or it.get("contents") or ""
                meta[pmid] = (title, abstract)
    return meta

# -------- Search --------

def load_index(index_path: Path, nprobe: int) -> indexing.Index:
    """Load FAISS IVF index; ensure on‑disk lists are attached; set nprobe; print diagnostics."""
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    # macOS stability: avoid READ_ONLY + MMAP for IVFPQ on some builds
    index = indexing.read_index(str(index_path))

    # Diagnostics
    print(f"Index diagnostics: is_trained={getattr(index, 'is_trained', True)}, ntotal={getattr(index, 'ntotal', 'NA')}")

    # If IVF-based, make sure on-disk lists are wired up (if used at build time)
    ivf = None
    try:
        ivf = indexing.extract_index_ivf(index)
    except Exception:
        ivf = None
    if ivf is not None:
        if USE_ONDISK:
            try:
                from faiss import OnDiskInvertedLists as _OD
                on_disk_already = isinstance(ivf.invlists, _OD)
            except Exception:
                on_disk_already = False
            if not on_disk_already:
                list_prefix = str((ONDISK_DIR / "ivf.lists").resolve())
                prefix_exists = any(str(p).startswith(list_prefix) for p in ONDISK_DIR.glob("ivf.lists*"))
                if prefix_exists:
                    print(f"Re-attaching on-disk inverted lists from prefix: {list_prefix}")
                    od = indexing.OnDiskInvertedLists(ivf.nlist, ivf.code_size, list_prefix)
                    od.thisown = False
                    ivf.replace_invlists(od, True)
                else:
                    print("Warning: USE_ONDISK=True but prefix files not found; continuing without reattach.")
        else:
            pass
        index.nprobe = max(1, int(nprobe))
        print(f"IVF index ready: nlist={ivf.nlist}, nprobe={index.nprobe}")
    else:
        print("Non-IVF index detected (nprobe not applicable).")

    if getattr(index, 'ntotal', 0) <= 0:
        raise RuntimeError("Loaded FAISS index has ntotal=0 (no vectors). Rebuild or check index path.")

    return index


def faiss_search(index: indexing.Index, query_vecs: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    if query_vecs.dtype != np.float32 or not query_vecs.flags['C_CONTIGUOUS']:
        query_vecs = np.ascontiguousarray(query_vecs.astype(np.float32, copy=False))
    indexing.normalize_L2(query_vecs)
    if not np.isfinite(query_vecs).all():
        raise ValueError("Query embedding contains NaN/Inf after normalization. Check the embedding pipeline.")
    if getattr(index, 'ntotal', 0) <= 0:
        raise RuntimeError("FAISS index is empty (ntotal=0).")
    return index.search(query_vecs, topk)

def main():
    # macOS stability settings (before importing faiss)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("FAISS_NTHREADS", "1")

    import faiss  # import after env hints
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Search a FAISS index built with on-disk inverted lists.")
    ap.add_argument("query", type=str, help="Free-text query (e.g., patient summary)")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Number of results to retrieve.")
    ap.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE, help="Number of IVF clusters to probe.")
    ap.add_argument("--show_metadata", action="store_true", help="Print title/abstract (requires pubmed_chunk_*.json files).")
    args = ap.parse_args()

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}. Did you run indexing.py?")

    print(f"Loading index from {INDEX_PATH} (nprobe={args.nprobe})...")
    index = load_index(INDEX_PATH, args.nprobe)

    # Optional: enable verbose logging on the index (if possible)
    try:
        index.verbose = True
    except Exception:
        pass

    # 1. Embed the query
    print("Embedding query...")
    qv = embed_query([args.query])  # shape [1, d]
    # Sanity: embedding dim must match index dim
    try:
        if qv.shape[1] != index.d:
            raise RuntimeError(f"Embedding dim {qv.shape[1]} != index dim {getattr(index, 'd', 'unknown')}. Use the same encoder family used for the index (MedCPT Query).")
    except Exception:
        pass

    # 2. Search
    print("Searching...")
    D, I = faiss_search(index, qv, args.topk)

    pmids = [int(x) for x in I[0]]
    scores = [float(x) for x in D[0]]

    # 3. Load metadata if requested
    meta_db = None
    if args.show_metadata:
        # Note: This is inefficiently loading *all* metadata.
        # A real system would use a DB or KV store.
        meta_db = build_metadata_index(DATA_DIR)
        if not meta_db:
            print("Proceeding without metadata.")

    # 4. Display results
    print("\n--- Top Results ---")
    for rank, (pmid, score) in enumerate(zip(pmids, scores), 1):
        print(f"\n{rank:2d}. PMID: {pmid}  (Score: {score:.4f})")
        
        if meta_db:
            title, abstract = meta_db.get(pmid, ("<Title not found>", "<Abstract not found>"))
            print(f"   Title: {title}")
            
            snippet = abstract.strip().replace("\n", " ")
            if len(snippet) > 350:
                snippet = snippet[:350] + "..."
            print(f"   Abstract: {snippet}")

    if all(p <= 0 for p in pmids):
        print("\nNote: All returned PMIDs are <= 0. Common causes:\n"
              "  • Query embeddings not aligned with index embeddings (use the MedCPT *Query* encoder).\n"
              "  • Index not loaded correctly (ntotal=0) or wrong index file.\n"
              "  • Metric mismatch (we use inner-product with L2 normalization for cosine).\n"
              "Try rebuilding the index with the same model, confirm ntotal>0, and re-run.")

    mode = "IVF-PQ" if hasattr(index, "pq") else "IVF-Flat"
    print(f"\n[search] Ready. Index type: {mode}; OnDisk={'yes' if USE_ONDISK else 'no'}; nprobe={getattr(index, 'nprobe', 'NA')}")

if __name__ == "__main__":
    main()