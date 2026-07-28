"""
Vector store — facade (aage ka darwaza).

Baqi poora project sirf isi file se baat karta hai. Ye faisla karti hai ke
neeche ChromaDB chalega ya Pinecone, config ke mutabiq:

    VECTOR_BACKEND=chroma     (default — local development)
    VECTOR_BACKEND=pinecone   (deployment)

Faida: `retriever.py`, `ingest.py`, aur `app.py` mein ek line bhi nahi
badalti jab aap backend switch karte hain. Yahi achhi software design ka
maqsad hai — jo cheez badalne wali hai use ek jagah qaid kar do.
"""

from __future__ import annotations

from . import config
from .chunker import Chunk
from .stores.base import SearchHit, VectorStore

# `SearchHit` yahan se re-export ho raha hai taake baqi code
# `from .vector_store import SearchHit` likh sake.
__all__ = [
    "SearchHit",
    "VectorStore",
    "get_store",
    "backend_name",
    "reset_collection",
    "add_chunks",
    "search",
    "stats",
]

_store: VectorStore | None = None


def get_store() -> VectorStore:
    """Configured backend lo (ek dafa banao, phir reuse karo)."""
    global _store
    if _store is None:
        if config.VECTOR_BACKEND == "pinecone":
            from .stores.pinecone_store import PineconeStore

            _store = PineconeStore()
        elif config.VECTOR_BACKEND == "chroma":
            from .stores.chroma_store import ChromaStore

            _store = ChromaStore()
        else:
            raise ValueError(
                f"VECTOR_BACKEND '{config.VECTOR_BACKEND}' pehchana nahi gaya. "
                "'chroma' ya 'pinecone' istemal karein."
            )
    return _store


def reset_store() -> None:
    """Cached backend bhool jao — backend switch karne ke baad kaam aata hai."""
    global _store
    _store = None


def backend_name() -> str:
    """UI mein dikhane ke liye — konsa backend chal raha hai."""
    try:
        return get_store().name
    except Exception:
        return config.VECTOR_BACKEND


# ---------------------------------------------------------- module API -----
# Ye patle wrappers hain. Inki wajah se caller ko `get_store()` yaad nahi
# rakhna parta — aur agar kal interface badle to sirf yahan haath lagega.


def reset_collection() -> None:
    get_store().reset()


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    get_store().add(chunks, embeddings)


def search(query_embedding: list[float], top_k: int = config.TOP_K) -> list[SearchHit]:
    return get_store().search(query_embedding, top_k)


def stats() -> dict:
    return get_store().stats()
