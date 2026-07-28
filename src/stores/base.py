"""
Common interface jo har vector store backend implement karta hai.

Ye file khud koi kaam nahi karti — ye sirf "contract" define karti hai.
ChromaStore aur PineconeStore dono isi contract ko poora karte hain, isliye
unhein aapas mein badla ja sakta hai bina baqi code chhue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..chunker import Chunk


@dataclass
class SearchHit:
    """
    Ek search result.

    `score` hamesha COSINE SIMILARITY hai — 0.0 (bilkul mukhtalif) se
    1.0 (bilkul same) tak.

    Ye ahem hai kyunki dono databases alag cheez wapas karte hain:
      - ChromaDB  -> distance deta hai (kam = behtar)
      - Pinecone  -> similarity score deta hai (zyada = behtar)

    Har backend apne andar conversion kar leta hai, taake bahar wale code ko
    hamesha ek hi paimana mile. Agar ye normalization na ho to threshold
    (0.50) ka matlab dono backends par ulta ho jata.
    """

    text: str
    source: str
    chunk_index: int
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Har backend ko ye chaar cheezein deni hain."""

    name: str

    def reset(self) -> None:
        """Poora index khaali karo (re-ingest se pehle)."""
        ...

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Chunks aur unke vectors store karo."""
        ...

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        """Sab se milte-julte top-K chunks wapas karo (similarity ke saath)."""
        ...

    def stats(self) -> dict:
        """UI ke liye: {"chunks": int, "sources": list[str], "backend": str}."""
        ...
