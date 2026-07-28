"""
Pinecone backend — managed cloud vector database.

Kab istemal karein: DEPLOYMENT (Streamlit Cloud, Render, waghera).
  - Index cloud mein rehta hai, app restart hone par mitta nahi
  - Ek dafa ingest karo, har deployment usi index ko parhta hai
  - Free (Starter) tier is project ke liye kaafi hai

ChromaDB se do ahem farq:

  1. SCORE ka matlab.
     Chroma DISTANCE deta hai (kam = behtar), Pinecone `metric="cosine"` par
     seedha SIMILARITY deta hai (zyada = behtar). Isliye yahan koi conversion
     nahi karni parti — `match.score` pehle se wohi paimana hai jo hamara
     threshold expect karta hai.

  2. Text kahan rehta hai.
     Chroma `documents` alag rakhta hai. Pinecone sirf vectors store karta
     hai, isliye chunk ka text hum METADATA mein rakhte hain. Metadata ki
     limit 40 KB per vector hai — hamare ~900 character chunks aaram se
     fit ho jate hain.
"""

from __future__ import annotations

import time

from .. import config
from ..chunker import Chunk
from .base import SearchHit

# Ek dafa mein kitne vectors upsert karein. Pinecone ki request size limit
# hai (~2 MB), isliye batch karna zaroori hai.
UPSERT_BATCH = 100


class PineconeStoreError(RuntimeError):
    """Pinecone setup ya connection ka masla."""


class PineconeStore:
    name = "Pinecone (cloud)"

    def __init__(self) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:  # pragma: no cover
            raise PineconeStoreError(
                "pinecone package install nahi hai.\n"
                "  pip install -r requirements.txt"
            ) from exc

        api_key = config.get_pinecone_key()
        if not api_key:
            raise PineconeStoreError(
                "PINECONE_API_KEY nahi mili.\n"
                "  - Local: .env mein PINECONE_API_KEY=... daalein\n"
                "  - Streamlit Cloud: Settings > Secrets mein daalein\n"
                "  - Free key: https://app.pinecone.io"
            )

        self._pc = Pinecone(api_key=api_key)
        self._spec = ServerlessSpec(
            cloud=config.PINECONE_CLOUD,
            region=config.PINECONE_REGION,
        )
        self._namespace = config.PINECONE_NAMESPACE
        self._index_name = config.PINECONE_INDEX
        self._index = None

    # ------------------------------------------------------------ helpers --
    def _ensure_index(self):
        """
        Index mojood hai to use lo, warna bana kar ready hone ka intezaar karo.

        Index banana ASYNCHRONOUS hai — create_index() foran wapas aa jata
        hai lekin index kuch seconds baad ready hota hai. Us se pehle upsert
        karne par error aata hai, isliye poll karte hain.
        """
        if self._index is not None:
            return self._index

        existing = {i["name"] for i in self._pc.list_indexes()}

        if self._index_name not in existing:
            self._pc.create_index(
                name=self._index_name,
                dimension=config.EMBEDDING_DIM,  # 768 — embedder se match hona ZAROORI hai
                metric="cosine",
                spec=self._spec,
            )
            # ready hone ka intezaar (max ~60 second)
            for _ in range(60):
                try:
                    if self._pc.describe_index(self._index_name).status["ready"]:
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise PineconeStoreError(
                    f"Index '{self._index_name}' 60 second mein ready nahi hua."
                )

        self._index = self._pc.Index(self._index_name)
        return self._index

    # -------------------------------------------------------------- api ----
    def reset(self) -> None:
        index = self._ensure_index()
        try:
            index.delete(delete_all=True, namespace=self._namespace)
        except Exception:
            # Namespace mojood hi nahi tha — pehli dafa chal raha hai.
            pass

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        index = self._ensure_index()

        vectors = [
            {
                "id": chunk.chunk_id,  # "leave_policy.md::chunk-003"
                "values": embedding,
                "metadata": {
                    "source": chunk.source,
                    "chunk_index": chunk.index,
                    "text": chunk.text,  # Pinecone text alag nahi rakhta
                },
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        for start in range(0, len(vectors), UPSERT_BATCH):
            index.upsert(
                vectors=vectors[start : start + UPSERT_BATCH],
                namespace=self._namespace,
            )

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        index = self._ensure_index()

        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self._namespace,
            include_metadata=True,
        )

        hits: list[SearchHit] = []
        for match in response.get("matches", []):
            meta = match.get("metadata") or {}
            hits.append(
                SearchHit(
                    text=str(meta.get("text", "")),
                    source=str(meta.get("source", "unknown")),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    # metric="cosine" par Pinecone seedha SIMILARITY deta hai.
                    # Chroma ke barkhilaf yahan 1-x karne ki zaroorat NAHI.
                    score=round(float(match.get("score", 0.0)), 4),
                )
            )
        return hits

    def stats(self) -> dict:
        index = self._ensure_index()

        try:
            described = index.describe_index_stats()
            namespaces = described.get("namespaces", {}) or {}
            total = int(namespaces.get(self._namespace, {}).get("vector_count", 0))
        except Exception:
            total = 0

        if total == 0:
            return {"chunks": 0, "sources": [], "backend": self.name}

        # Source names IDs se nikal lete hain ("filename.md::chunk-003").
        # Isse vectors fetch karne ki zaroorat nahi parti — sasta aur tez.
        #
        # Ehtiyat: SDK version ke hisab se list() ya to plain string IDs deta
        # hai ya `ListItem` objects (jinme .id property hoti hai). Dono soorat
        # handle kar lete hain, warna source ka naam "ListItem(id='..." jaisa
        # ganda aa jata hai.
        sources: set[str] = set()
        try:
            for id_page in index.list(namespace=self._namespace):
                for entry in id_page:
                    vector_id = getattr(entry, "id", entry)
                    sources.add(str(vector_id).split("::")[0])
        except Exception:
            pass  # list() har plan par available nahi — count phir bhi sahi hai

        return {"chunks": total, "sources": sorted(sources), "backend": self.name}
