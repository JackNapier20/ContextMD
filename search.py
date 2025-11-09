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
DEFAULT_TOPK = 20
DISPLAY_TOPK_WITH_ABSTRACT = 5

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

# -------- Dedup helpers (prefer unique titles/abstracts, case/whitespace-insensitive) --------
def _normalize_for_dup(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # light punctuation collapse to avoid cosmetic diffs
    s = re.sub(r"[\u200b\ufeff]", "", s)  # zero-width & BOM
    return s

def _dup_key(title: str, abstract: str):
    t = _normalize_for_dup(title)
    a = _normalize_for_dup(abstract)
    if t:
        return ("t", t)
    if a:
        # use a prefix so very long abstracts with tiny tail diffs still match
        return ("a", a[:160])
    return ("none", "")

def _dedup_results(results: List[dict]) -> List[dict]:
    """Keep first occurrence by (normalized) title/abstract; if a later duplicate
    has a strictly longer abstract, prefer it instead. Preserve original ranking otherwise."""
    best = {}
    for idx, r in enumerate(results):
        key = _dup_key(r.get("title", ""), r.get("abstract", ""))
        prev = best.get(key)
        if prev is None:
            best[key] = (idx, r)
        else:
            p_idx, p = prev
            if len(r.get("abstract", "")) > len(p.get("abstract", "")):
                best[key] = (p_idx, r)
    # back to list in original order
    return [item for _, item in sorted(best.values(), key=lambda x: x[0])]

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
    Builds {pmid: (title, abstract)} from pubmed_chunk_*.json files.
    Tolerates:
      - JSON array of dicts
      - JSON dict-of-dicts
      - JSON array of JSON strings (each string is a record)
      - JSONL (one JSON object per line)
    """
    def _yield_records(path: Path):
        # Try full-file JSON first
        try:
            raw = path.read_bytes()
            parsed = orjson.loads(raw)
            if isinstance(parsed, list):
                for x in parsed:
                    if isinstance(x, dict):
                        yield x
                    elif isinstance(x, str):
                        # list of stringified JSON objects
                        try:
                            j = orjson.loads(x)
                            if isinstance(j, dict):
                                yield j
                        except Exception:
                            continue
                return
            if isinstance(parsed, dict):
                # Case A: array-columns format {"pmid": [...], "title": [...], "abstract": [...]} (or "contents")
                lower_keys = {k.lower(): k for k in parsed.keys()}
                pmid_key = lower_keys.get("pmid")
                title_key = lower_keys.get("title") or lower_keys.get("article_title") or lower_keys.get("articletitle")
                abs_key = lower_keys.get("abstract") or lower_keys.get("abstracttext") or lower_keys.get("abstract_text") or lower_keys.get("contents")
                if pmid_key and (title_key or abs_key):
                    pmids = parsed.get(pmid_key)
                    titles = parsed.get(title_key) if title_key else None
                    abstracts = parsed.get(abs_key) if abs_key else None
                    if isinstance(pmids, list) and ((titles is None or isinstance(titles, list)) and (abstracts is None or isinstance(abstracts, list))):
                        L = len(pmids)
                        for i in range(L):
                            rec = {
                                "pmid": pmids[i],
                                "title": titles[i] if titles and i < len(titles) else "",
                                "abstract": abstracts[i] if abstracts and i < len(abstracts) else "",
                            }
                            yield rec
                        return
                # Case B: dict-of-dicts {pmid: {...}}
                any_yielded = False
                for k, v in parsed.items():
                    if isinstance(v, dict):
                        if "pmid" not in v:
                            v = {**v, "pmid": k}
                        yield v
                        any_yielded = True
                if any_yielded:
                    return
                # Case C: dict-of-strings {pmid: "abstract text"}
                for k, v in parsed.items():
                    if isinstance(v, str):
                        yield {"pmid": k, "title": "", "abstract": v}
                return
        except Exception:
            pass
        # Fallback: JSONL (one JSON object per line)
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    j = orjson.loads(s)
                    if isinstance(j, dict):
                        yield j
                except Exception:
                    continue

    meta: Dict[int, Tuple[str, str]] = {}
    pubmed_dir = data_dir / "pubmed"
    pubmed_files = sorted(pubmed_dir.glob("pubmed_chunk_*.json"))
    if not pubmed_files:
        print(f"Warning: No 'pubmed_chunk_*.json' files found in {pubmed_dir}. Metadata will be unavailable.")
        return {}

    print(f"Loading metadata from {len(pubmed_files)} files...")
    for pf in pubmed_files:
        added_here = 0
        for rec in _yield_records(pf):
            # normalize keys to lower-case for robustness
            if not isinstance(rec, dict):
                continue
            r = {str(k).lower(): v for k, v in rec.items()}
            # pmid may be under different keys or as string
            pmid_val = r.get("pmid") or r.get("id") or r.get("pmid_int")
            try:
                pmid = int(str(pmid_val)) if pmid_val is not None else -1
            except Exception:
                pmid = -1
            if pmid <= 0 or pmid in meta:
                continue
            # title fallbacks (now include abbreviated key 't')
            title = (
                r.get("title")
                or r.get("t")  # abbreviated title
                or r.get("article_title")
                or r.get("articletitle")
                or r.get("paper_title")
                or r.get("name")
                or ""
            )
            # abstract/content fallbacks (now include abbreviated key 'a')
            abstract = (
                r.get("abstract")
                or r.get("a")  # abbreviated abstract
                or r.get("abstracttext")
                or r.get("abstract_text")
                or r.get("contents")
                or r.get("content")
                or r.get("summary")
                or r.get("text")
                or r.get("body")
                or r.get("document")
                or r.get("passage")
                or r.get("chunk")
                or r.get("full_text")
                or ""
            )
            # coerce to str and trim
            if not isinstance(title, str):
                title = str(title or "")
            if not isinstance(abstract, str):
                abstract = str(abstract or "")
            title = title.strip()
            abstract = abstract.strip()
            # If title still empty, synthesize a short one from abstract/text
            if not title and abstract:
                # first sentence-ish, up to 120 chars
                t = abstract.split(". ")[0][:120].strip()
                title = t if t else "Untitled"
            # If abstract empty but title present, keep as empty string (don't invent)
            meta[pmid] = (title, abstract)
            added_here += 1
        if added_here == 0:
            print(f"  (info) No usable records found in {pf.name}; check field names.")
    print(f"Loaded metadata for {len(meta):,} PMIDs.")
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

    # 3. Load metadata (needed to prefer abstracts); we still keep --show_metadata for printing
    meta_db = build_metadata_index(DATA_DIR)
    if not meta_db:
        print("Proceeding without metadata.")

    # Prefer results that have an abstract; keep original ranking order
    results = []
    for pmid, score in zip(pmids, scores):
        title, abstract = meta_db.get(pmid, ("", "")) if meta_db else ("", "")
        results.append({
            "pmid": pmid,
            "score": score,
            "title": title,
            "abstract": abstract.strip() if isinstance(abstract, str) else ""
        })

    # Deduplicate adjacent/similar items that differ only by case/minor formatting
    results = _dedup_results(results)

    with_abs = [r for r in results if r["abstract"]]
    # If we don't have enough, fall back to top results (even if abstract is empty)
    if len(with_abs) >= DISPLAY_TOPK_WITH_ABSTRACT:
        display = with_abs[:DISPLAY_TOPK_WITH_ABSTRACT]
    else:
        need = DISPLAY_TOPK_WITH_ABSTRACT - len(with_abs)
        fallback = [r for r in results if not r["abstract"]][:need]
        display = with_abs + fallback

    # 4. Display results
    print("\n--- Top Results (preferring abstracts, deduped by title/abstract) ---")
    for rank, r in enumerate(display, 1):
        print(f"\n{rank:2d}. PMID: {r['pmid']}  (Score: {r['score']:.4f})")
        if args.show_metadata or r.get("title") or r.get("abstract"):
            title = r.get("title") or "<Title not found>"
            print(f"   Title: {title}")
            snippet = (r.get("abstract") or "<Abstract not found>").strip().replace("\n", " ")
            if len(snippet) > 350:
                snippet = snippet[:350] + "..."
            print(f"   Abstract: {snippet}")

    print(f"\nSelected {len(display)} results (requested top {DEFAULT_TOPK}); preferred those with non-empty abstracts.")

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