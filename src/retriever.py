"""
Retrieval — sawal ke liye relevant chunks dhoondna, aur FALLBACK ka faisla.

Ye file assignment ke Quality Bar ka sabse ahem hissa hai:

    "Fallback must clearly guide the user when the answer is not found."

Tareeqa:
  1. Sawal ka embedding banao
  2. Vector DB se top-K chunks lao (har ek ka similarity score)
  3. Score ko threshold se compare karo:
       - best score >= threshold  -> chunks LLM ko bhejo
       - best score <  threshold  -> LLM ko call hi MAT karo, fallback do

Point 3 ka doosra hissa khaas hai. Bohot se RAG demos LLM ko har haal mein
call kar dete hain aur ummeed karte hain ke woh khud mana kar de. Hum us se
pehle hi rok dete hain — is se do faide:
    - Hallucination ka mauqa hi nahi milta (grounding guarantee)
    - Har out-of-scope sawal par API cost bachti hai
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, embedder, vector_store
from .vector_store import SearchHit


@dataclass
class RetrievalResult:
    """Retrieval ka natija — UI aur generator dono isko parhte hain."""

    question: str
    hits: list[SearchHit]  # threshold pass karne wale chunks
    all_hits: list[SearchHit]  # sab chunks (debugging / transparency ke liye)
    threshold: float

    @property
    def has_context(self) -> bool:
        """Kya hamare paas kaafi relevant context hai?"""
        return len(self.hits) > 0

    @property
    def best_score(self) -> float:
        return self.all_hits[0].score if self.all_hits else 0.0


def retrieve(
    question: str,
    top_k: int = config.TOP_K,
    threshold: float = config.SIMILARITY_THRESHOLD,
) -> RetrievalResult:
    """Sawal ke liye relevant chunks nikalo."""
    question = question.strip()
    if not question:
        return RetrievalResult(question, [], [], threshold)

    query_vector = embedder.embed_query(question)
    all_hits = vector_store.search(query_vector, top_k=top_k)

    # Sirf woh chunks rakho jo threshold se upar hain
    hits = [h for h in all_hits if h.score >= threshold]

    return RetrievalResult(
        question=question,
        hits=hits,
        all_hits=all_hits,
        threshold=threshold,
    )


def build_context(hits: list[SearchHit]) -> str:
    """
    Chunks ko ek numbered context string mein badlo.

    Har chunk par [Source N: filename] ka label lagate hain taake model
    apne jawab mein source ka hawala de sake — aur hum verify kar sakein
    ke uska hawala sach mein us chunk mein tha.
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Source {i}: {hit.source} | chunk {hit.chunk_index} | "
            f"relevance {hit.score:.2f}]\n{hit.text}"
        )
    return "\n\n---\n\n".join(blocks)
