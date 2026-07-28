"""
ChromaDB backend — local disk par chalta hai.

Kab istemal karein: LOCAL DEVELOPMENT.
  - Koi signup nahi, koi API key nahi
  - Bohot tez (network round-trip nahi)
  - Internet ke baghair kaam karta hai

Kab NA karein: Streamlit Cloud deployment.
  Wahan filesystem ephemeral hai — restart par chroma_db/ folder mit jata
  hai aur index khaali ho jata hai. Us ke liye PineconeStore use karein.
"""

from __future__ import annotations

# ChromaDB ko sqlite3 >= 3.35 chahiye. Kuch Linux hosts par purana sqlite
# hota hai, isliye wahan pysqlite3 ko sqlite3 ki jagah swap kar dete hain.
# Ye import chromadb se PEHLE hona zaroori hai.
try:  # pragma: no cover
    import sys

    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:  # Windows / local — yahan zaroorat nahi
    pass

import chromadb

from .. import config
from ..chunker import Chunk
from .base import SearchHit


class ChromaStore:
    name = "ChromaDB (local)"

    def __init__(self) -> None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # ------------------------------------------------------------ helpers --
    @property
    def _collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine distance use karo
        )

    # -------------------------------------------------------------- api ----
    def reset(self) -> None:
        try:
            self._client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass  # collection thi hi nahi — koi baat nahi
        _ = self._collection  # dobara bana do

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.index} for c in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        collection = self._collection
        count = collection.count()
        if count == 0:
            return []

        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for text, meta, distance in zip(documents, metadatas, distances):
            meta = meta or {}
            hits.append(
                SearchHit(
                    text=text,
                    source=str(meta.get("source", "unknown")),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    # Chroma DISTANCE deta hai; hum SIMILARITY chahte hain.
                    # cosine space mein: similarity = 1 - distance
                    score=round(1.0 - float(distance), 4),
                )
            )
        return hits

    def stats(self) -> dict:
        collection = self._collection
        total = collection.count()
        if total == 0:
            return {"chunks": 0, "sources": [], "backend": self.name}

        got = collection.get(include=["metadatas"])
        sources = sorted(
            {str((m or {}).get("source", "?")) for m in got.get("metadatas", [])}
        )
        return {"chunks": total, "sources": sources, "backend": self.name}
