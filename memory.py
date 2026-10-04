import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MEMORY_DIR = "memory_store"
INDEX_PATH = os.path.join(MEMORY_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(MEMORY_DIR, "metadata.json")
EMBEDDING_DIM = 384  # output size of the all-MiniLM-L6-v2 model

_model = None


def _get_model():
    """Load the local embedding model once and reuse it (loading it is slow, using it is fast)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _embed(texts):
    model = _get_model()
    vectors = np.asarray(model.encode(texts), dtype="float32")
    faiss.normalize_L2(vectors)  # required so inner product = cosine similarity
    return vectors


def _load_metadata() -> list:
    """
    metadata.json is the SOURCE OF TRUTH: it holds every remembered goal and outcome.
    The FAISS index is just a fast lookup built from it, so it can always be rebuilt.
    """
    if not os.path.exists(METADATA_PATH):
        return []
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict) and "goal" in m and "final_answer" in m]
    except (json.JSONDecodeError, OSError):
        pass
    # Unreadable file: keep a copy for inspection instead of silently destroying it, then start fresh
    try:
        os.replace(METADATA_PATH, METADATA_PATH + ".corrupt")
    except OSError:
        pass
    return []


def _write_atomic(path: str, writer):
    """Write to a temporary file first, then swap it in, so a crash can't leave a half-written file."""
    tmp_path = path + ".tmp"
    writer(tmp_path)
    os.replace(tmp_path, path)


def _save_metadata(metadata: list):
    def write(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    _write_atomic(METADATA_PATH, write)


def _save_index(index):
    _write_atomic(INDEX_PATH, lambda p: faiss.write_index(index, p))


def _rebuild_index(metadata: list):
    """Build a fresh FAISS index from the remembered goals (IndexFlatIP + normalized vectors = cosine)."""
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    if metadata:
        index.add(_embed([m["goal"] for m in metadata]))
    return index


def _load_index_and_metadata():
    """
    Load both, and SELF-HEAL: if the index and the metadata disagree (for example a run was
    interrupted between the two file writes, or one file was edited or deleted), rebuild
    the index from the metadata so every vector lines up with its goal again.
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)
    metadata = _load_metadata()

    index = None
    if os.path.exists(INDEX_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
        except Exception:
            index = None

    if index is None or index.ntotal != len(metadata) or index.d != EMBEDDING_DIM:
        index = _rebuild_index(metadata)
        _save_index(index)

    return index, metadata


def add_memory(goal: str, final_answer: str):
    """Store a completed run's goal + outcome so future similar goals can recall it."""
    index, metadata = _load_index_and_metadata()
    index.add(_embed([goal]))
    metadata.append({"goal": goal, "final_answer": final_answer})
    _save_metadata(metadata)   # the source of truth first...
    _save_index(index)         # ...then the lookup (it self-heals on the next load if this step is skipped)


def search_memory(goal: str, k: int = 2, min_similarity: float = 0.5):
    """
    Find past goals similar to the current one (cosine similarity, 0-1,
    higher = more similar). Returns a list of {"goal":..., "final_answer":...}
    for past runs similar enough to be worth recalling. Empty list if no
    memory exists yet, or nothing is similar enough.
    """
    index, metadata = _load_index_and_metadata()
    if index.ntotal == 0:
        return []

    similarities, indices = index.search(_embed([goal]), min(k, index.ntotal))

    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx < 0 or idx >= len(metadata):   # never trust an index blindly
            continue
        if sim >= min_similarity:
            results.append(metadata[idx])
    return results