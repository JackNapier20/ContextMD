#!/usr/bin/env python3
import re
import math
import os
import faiss
import orjson
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Iterator

# --------------------------
# CONFIG
# --------------------------
# (Adjust these paths and parameters)

# Resolve project root as the folder containing this file
PROJECT_ROOT = Path(__file__).resolve().parent

#
# Data source root: contains subfolders `embeds/`, `pmids/`, `pubmed/`
# You can override via the MEDCPT_DIR environment variable.
env_data_dir = os.environ.get("MEDCPT_DIR")
if env_data_dir:
    DATA_DIR = Path(env_data_dir).expanduser().resolve()
else:
    DATA_DIR = (PROJECT_ROOT / "data/MedCPT").resolve()

EMBEDS_DIR = DATA_DIR / "embeds"
PMIDS_DIR = DATA_DIR / "pmids"
PUBMED_DIR = DATA_DIR / "pubmed"

print(f"[indexing] Using DATA_DIR = {DATA_DIR}")

# Which chunks to process
CHUNK_RANGE = range(30, 38)  # 30..37

# Build toggles (use these to quickly isolate issues)
USE_PQ = False        # False => IVF-Flat (no quantization). True => IVF-PQ
USE_ONDISK = False    # False => keep lists in-memory; True => write OnDiskInvertedLists

# Output (stored under ./output relative to this repo)
INDEX_OUT = PROJECT_ROOT / "output/pubmed_ivfpq.faiss"
# FAISS will write many invlist files here
ONDISK_DIR = PROJECT_ROOT / "output/ondisk_lists"

# Sanity checks: make sure subfolders exist
missing = [p for p in [EMBEDS_DIR, PMIDS_DIR] if not p.exists()]
if missing:
    raise FileNotFoundError(
        "\n".join([
            "Required subfolders not found:",
            *[str(m) for m in missing],
            "\nExpected layout:",
            f"  {DATA_DIR}/embeds/embeds_chunk_30.npy",
            f"  {DATA_DIR}/pmids/pmids_chunk_30.json",
            f"  {DATA_DIR}/pubmed/pubmed_chunk_30.json (optional for metadata)",
        ])
    )

# Indexing parameters
TRAIN_VEC_TARGET = 1_000_000  # vectors to sample for IVF training
NLIST = 8192                # nb of coarse clusters
BATCH = 8_000              # add() batch size (smaller to avoid mmap/OS SIGBUS on macOS)
METRIC = faiss.METRIC_INNER_PRODUCT  # use IP; we will L2-normalize vectors for cosine

# ---- FAISS re-exports for convenience (so search.py can `import indexing`) ----
Index = faiss.Index
IndexFlatIP = faiss.IndexFlatIP
IndexIVFPQ = faiss.IndexIVFPQ
IO_FLAG_MMAP = faiss.IO_FLAG_MMAP
IO_FLAG_READ_ONLY = faiss.IO_FLAG_READ_ONLY
read_index = faiss.read_index
write_index = faiss.write_index
extract_index_ivf = faiss.extract_index_ivf
OnDiskInvertedLists = faiss.OnDiskInvertedLists
normalize_L2 = faiss.normalize_L2

RANDOM_SEED = 123
# PQ quantization parameters (minimal change from IVF-Flat -> IVF-PQ)
PQ_M = 16        # number of subquantizers
PQ_NBITS = 8     # bits per sub-vector (8 = 256 codewords)

# --------------------------
# HELPERS
# --------------------------

# Pre-compile regex for performance
_chunk_pat = re.compile(r"embeds_chunk_(\d+)\.npy$")

def list_chunks(data_dir: Path, wanted: range) -> List[Tuple[int, Path, Path]]:
    """Finds all (chunk_id, embed_path, pmid_path) tuples matching the range."""
    out = []
    for p in (EMBEDS_DIR).glob("embeds_chunk_*.npy"):
        m = _chunk_pat.match(p.name)
        if not m:
            continue
        cid = int(m.group(1))
        if cid in wanted:
            pmids = PMIDS_DIR / f"pmids_chunk_{cid}.json"
            if not pmids.exists():
                raise FileNotFoundError(f"Missing PMIDs file for chunk {cid}: {pmids}")
            out.append((cid, p, pmids))
    out.sort(key=lambda x: x[0])
    if not out:
        raise RuntimeError(f"No chunks found in {EMBEDS_DIR} for range {wanted}")
    return out

def load_pmids(pmids_path: Path) -> np.ndarray:
    """Load PMIDs as int64 array (FAISS IDs)."""
    with pmids_path.open("rb") as f:
        arr = orjson.loads(f.read())
    # Ensure int64 for FAISS
    return np.asarray(arr, dtype=np.int64)

def normalize_ip(x: np.ndarray):
    """L2-normalize rows in-place for IP -> cosine similarity."""
    normalize_L2(x)

def sample_for_training(chunks: List[Tuple[int, Path, Path]], target: int, seed: int) -> np.ndarray:
    """Randomly sample up to `target` vectors across chunks for IVF training."""
    rng = np.random.default_rng(seed)
    sampled = []
    remaining = target

    # Get all chunk sizes first to do proportional sampling
    sizes = []
    for _, emb_path, _ in chunks:
        # Use mmap_mode to avoid loading all data into RAM
        X = np.load(emb_path)
        sizes.append(X.shape[0])
    total = sum(sizes)
    if total == 0:
        raise RuntimeError("No vectors present in selected chunks.")

    print(f"Sampling from {total:,} total vectors...")
    for (cid, emb_path, _), n in zip(chunks, sizes):
        if remaining <= 0:
            break
        # Proportional sampling
        take = min(n, math.ceil(target * (n / total)))
        if take <= 0:
            continue
            
        X = np.load(emb_path) # Load fully into RAM
        idx = rng.choice(n, size=take, replace=False)
        # We must copy from the mmap'd array into memory
        sampled.append(np.array(X[idx], dtype=np.float32, copy=True))
        remaining -= take
        del X

    Xtrain = np.vstack(sampled)
    if Xtrain.shape[0] > target:
        Xtrain = Xtrain[:target]
        
    normalize_ip(Xtrain) # Must normalize training data too
    return Xtrain

# --------------------------
# BUILD INDEX
# --------------------------
def main():
    """Builds and saves the on-disk IVF-Flat index."""
    # Be conservative with threads on macOS to avoid rare SIGBUS issues
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    ONDISK_DIR.mkdir(parents=True, exist_ok=True)

    chunks = list_chunks(DATA_DIR, CHUNK_RANGE)
    print(f"Found {len(chunks)} chunks to process.")

    # Get vector dimension from first chunk
    first_embeds = np.load(chunks[0][1], mmap_mode="r")
    d = first_embeds.shape[1]
    del first_embeds
    print(f"Vector dimension: {d}")

    # 1. Init Index (IVF-Flat by default; flip USE_PQ=True for IVF-PQ)
    quantizer = IndexFlatIP(d)
    if USE_PQ:
        index = faiss.IndexIVFPQ(quantizer, d, NLIST, PQ_M, PQ_NBITS, METRIC)
    else:
        index = faiss.IndexIVFFlat(quantizer, d, NLIST, METRIC)
    try:
        index.verbose = False
    except Exception:
        pass

    # 2. Train
    print("Sampling training vectors...")
    Xtrain = sample_for_training(chunks, TRAIN_VEC_TARGET, RANDOM_SEED)
    if USE_PQ:
        print(f"Training IVF-PQ index with {Xtrain.shape[0]:,} vectors (M={PQ_M}, nbits={PQ_NBITS})...")
    else:
        print(f"Training IVF-Flat index with {Xtrain.shape[0]:,} vectors...")
    index.train(Xtrain)
    del Xtrain # free memory
    print("Training complete.")

    # 3. (Optional) Configure On-Disk Storage
    ivf = extract_index_ivf(index)
    if USE_ONDISK:
        abs_ondisk_dir = ONDISK_DIR.resolve()
        list_prefix = str(abs_ondisk_dir / "ivf.lists")
        ondisk = OnDiskInvertedLists(ivf.nlist, ivf.code_size, list_prefix)
        ivf.replace_invlists(ondisk, True)
        ondisk.thisown = False
        print(f"Set index to use on-disk inverted lists in: {abs_ondisk_dir}")
    else:
        print("Using in-memory inverted lists (no on-disk files).")

    # 4. Add
    total_added = 0
    for cid, emb_path, pmids_path in chunks:
        print(f"\nProcessing chunk {cid}...")
        X = np.load(emb_path) # mmap
        ids = load_pmids(pmids_path)
        
        if X.shape[0] != ids.shape[0]:
            raise ValueError(f"Row count mismatch for chunk {cid}: {X.shape[0]} vs {ids.shape[0]}")

        n = X.shape[0]
        # Stream in batches
        for s in tqdm(range(0, n, BATCH), desc=f"chunk {cid}", ncols=80):
            e = min(s + BATCH, n)
            # Copy batch from mmap into a C-contiguous array in RAM
            xb = np.asarray(X[s:e], dtype=np.float32, order="C")
            normalize_ip(xb)
            idb = ids[s:e]
            
            index.add_with_ids(xb, idb)
            total_added += (e - s)
        
        del X, ids # close mmap handle

    print(f"\nTotal vectors added: {total_added:,}")

    # 5. Save
    # This just saves the index *structure*. The invlists are already in ONDISK_DIR.
    print(f"Writing index metadata to: {INDEX_OUT}")
    write_index(index, str(INDEX_OUT))
    mode = "IVF-PQ" if hasattr(index, "pq") else "IVF-Flat"
    print(f"\n[search] Ready. Index type: {mode}; OnDisk={'yes' if USE_ONDISK else 'no'}; nprobe={getattr(index, 'nprobe', 'NA')}")
    print("Done.")

if __name__ == "__main__":
    main()