"""
Embeddings — text ko numbers mein badalna.

Embedding kya hai?
  "annual leave kitni hai"  ->  [0.021, -0.884, 0.310, ... ]  (768 numbers)

Khaas baat: milte-julte MATLAB wale text ke vectors bhi paas paas hote hain.
Isliye "chuttiyan kitni milti hain" aur "annual leave entitlement" —
alag alfaaz, lekin vector space mein qareeb. Yahi RAG ka core hai, aur yahi
wajah hai ke plain keyword search se ye behtar kaam karta hai.

Ek important detail: task_type
  RETRIEVAL_DOCUMENT -> jab hum knowledge base index kar rahe hain
  RETRIEVAL_QUERY    -> jab user ka sawal embed kar rahe hain
Gemini in dono ke liye thora mukhtalif vector banata hai, jisse match
behtar hota hai. Ye ek chhoti si detail hai jo accuracy kaafi barha deti hai.
"""

from __future__ import annotations

import math
import time

from google import genai
from google.genai import types

from . import config

_client: genai.Client | None = None

# Free tier par rate limit hai, isliye ek dafa mein 20 texts bhejte hain
BATCH_SIZE = 20
BATCH_PAUSE_SECONDS = 0.6


class EmbeddingError(RuntimeError):
    """Embedding banate waqt koi masla aaya."""


def get_client() -> genai.Client:
    """Gemini client banao (ek dafa) aur reuse karo."""
    global _client
    if _client is None:
        api_key = config.get_api_key()
        if not api_key:
            raise EmbeddingError(
                "GOOGLE_API_KEY nahi mili.\n"
                "  - Local: .env.example ko .env banayein aur key daalein\n"
                "  - Streamlit Cloud: Settings > Secrets mein key daalein\n"
                "  - Free key: https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _l2_normalize(vector: list[float]) -> list[float]:
    """
    Vector ki length 1 kar do.

    Kyun? Hum cosine similarity use karte hain. Agar sab vectors normalized
    hon to cosine similarity = simple dot product ban jati hai — tez aur
    numerically stable. Gemini truncated dimensions par normalized vector
    nahi deta, isliye hum khud karte hain.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    client = get_client()

    cfg = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=config.EMBEDDING_DIM,
    )

    try:
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=texts,
            config=cfg,
        )
    except Exception as exc:
        # Kuch purane embedding models output_dimensionality support nahi
        # karte — us soorat mein bina us parameter ke dobara try karo.
        try:
            response = client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        except Exception:
            raise EmbeddingError(
                f"Embedding fail hui (model: {config.EMBEDDING_MODEL}). "
                f"Asal error: {exc}"
            ) from exc

    return [_l2_normalize(list(e.values)) for e in response.embeddings]


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Kai texts ko batches mein embed karo."""
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors.extend(_embed_batch(batch, task_type))

        if start + BATCH_SIZE < len(texts):
            time.sleep(BATCH_PAUSE_SECONDS)  # rate limit ka ehtiram

    return vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Knowledge base ke chunks ke liye (indexing ke waqt)."""
    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """User ke sawal ke liye (search ke waqt)."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
