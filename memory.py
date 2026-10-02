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


def _load_index_and_metadata():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        # IndexFlatIP + normalized vectors = cosine similarity search
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = []
    return index, metadata


def _save_index_and_metadata(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _embed(texts):
    model = _get_model()
    vectors = model.encode(texts).astype("float32")
    faiss.normalize_L2(vectors)  # required so inner product = cosine similarity
    return vectors


def add_memory(goal: str, final_answer: str):
    """Store a completed run's goal + outcome so future similar goals can recall it."""
    index, metadata = _load_index_and_metadata()
    vector = _embed([goal])
    index.add(vector)
    metadata.append({"goal": goal, "final_answer": final_answer})
    _save_index_and_metadata(index, metadata)


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

    vector = _embed([goal])
    similarities, indices = index.search(vector, min(k, index.ntotal))

    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        if sim >= min_similarity:
            results.append(metadata[idx])
    return results