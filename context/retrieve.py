"""
context/retrieve.py — query the local Laws-of-the-Game vector index.

The index is built by context/ingest_laws.py (Docling chunks + Granite embeddings,
stored under context/index/). ``retrieve`` embeds the question with the SAME Granite
embedder and returns the most similar Laws chunks, so the tactical explainer can
ground its answer in the actual rule. Degrades to [] if the index isn't built yet,
so the explainer can still answer from match data alone.
"""
from __future__ import annotations

import json
from pathlib import Path

INDEX_DIR = Path(__file__).resolve().parent / "index"
_CHUNKS_FILE = "laws_chunks.json"
_VECTORS_FILE = "laws_vectors.npy"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings via the Granite embeddings endpoint (OpenAI-compatible).

    Shared by the ingest (chunks) and retrieval (query) paths so both use the same
    model and vector space.
    """
    from agent.granite_client import build_granite_client, embed_model_id
    client = build_granite_client()
    resp = client.embeddings.create(model=embed_model_id(), input=list(texts))
    return [d.embedding for d in resp.data]


def _load_index(index_dir: Path = INDEX_DIR):
    """Return (chunks, vectors) from disk, or (None, None) if the index isn't built."""
    import numpy as np
    index_dir = Path(index_dir)
    chunks_path = index_dir / _CHUNKS_FILE
    vectors_path = index_dir / _VECTORS_FILE
    if not (chunks_path.exists() and vectors_path.exists()):
        return None, None
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))["chunks"]
    return chunks, np.load(vectors_path)


def retrieve(question: str, k: int = 4, index_dir: Path = INDEX_DIR) -> list[dict]:
    """Return the top-k Laws chunks most similar to `question` (cosine similarity).

    Each result is {"text", "score"}. Returns [] when the index hasn't been built.
    """
    import numpy as np
    chunks, vectors = _load_index(index_dir)
    if not chunks:
        return []
    q = np.asarray(embed_texts([question])[0], dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-9)
    mat = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    sims = mat @ q
    top = np.argsort(-sims)[: max(1, k)]
    return [{"text": chunks[int(i)], "score": float(sims[int(i)])} for i in top]
