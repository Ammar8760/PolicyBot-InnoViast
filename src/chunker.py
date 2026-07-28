"""
Chunking strategy.

Sawal: poori document LLM ko kyun nahi bhej dete?
Jawab: teen wajah —
  1. Cost — 50 page ki PDF har sawal par bhejna mehenga hai
  2. Accuracy — model ko 50 pages mein se relevant line dhoondni pare to
     woh "lost in the middle" ho jata hai
  3. Retrieval — chote tukde precise match dete hain, poori file nahi

Hamari strategy: "structure-aware greedy packing with overlap"
  - Pehle document ko paragraphs mein toro (natural boundaries)
  - Phir paragraphs ko jama kar ke CHUNK_SIZE tak ke chunks banao
  - Har naye chunk ki shuruaat mein pichle chunk ka aakhri hissa (overlap)
    daal do, taake boundary par toota hua matlab bacha rahe
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config


@dataclass
class Chunk:
    """Ek chunk + uski metadata (source file, index)."""

    text: str
    source: str
    index: int

    @property
    def chunk_id(self) -> str:
        return f"{self.source}::chunk-{self.index:03d}"


# ------------------------------------------------------------------ split --
def _split_paragraphs(text: str) -> list[str]:
    """Text ko paragraphs mein toro (khaali line = boundary)."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Ek lambe paragraph ko sentences mein toro."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """Aakhri sahara — agar ek sentence bhi CHUNK_SIZE se bara ho."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def _to_units(paragraphs: list[str], size: int) -> list[str]:
    """
    Har paragraph ko aisa "unit" banao jo CHUNK_SIZE mein fit ho jaye.
    Bare paragraphs ko sentences mein, phir zaroorat par characters mein toro.
    """
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            units.append(para)
            continue

        for sentence in _split_sentences(para):
            if len(sentence) <= size:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, size))
    return units


def _overlap_tail(units: list[str], overlap: int) -> list[str]:
    """
    Pichle chunk ke aakhir se utne units wapas lo jitne `overlap`
    characters ban jayein. Yahi agle chunk ka "context bridge" hai.
    """
    if overlap <= 0:
        return []

    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        tail.insert(0, unit)
        total += len(unit)
        if total >= overlap:
            break
    return tail


# ------------------------------------------------------------------- api ---
def chunk_text(
    text: str,
    source: str,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Ek document ka poora text lo, Chunk objects ki list wapas do."""
    units = _to_units(_split_paragraphs(text), size)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        extra = len(unit) + (2 if current else 0)  # 2 = "\n\n" separator

        # Agar ye unit add karne se chunk bara ho jayega, to chunk band karo
        if current and current_len + extra > size:
            chunks.append("\n\n".join(current))

            # Naya chunk pichle ke overlap tail se shuru hota hai.
            #
            # Lekin ehtiyat: tail khud kaafi bara ho sakta hai. Agar hum
            # tail + naya unit bina check kiye jor dein to chunk `size` se
            # bara ho jayega. Isliye tail ko aage se chhota karte hain jab
            # tak naya unit fit na ho jaye.
            #
            # (`_to_units` guarantee karta hai ke koi bhi unit akela `size`
            #  se bara nahi hota, isliye ye loop hamesha khatam hota hai —
            #  bad-tareen soorat mein tail bilkul khaali ho jata hai.)
            tail = _overlap_tail(current, overlap)
            while tail:
                tail_len = sum(len(u) for u in tail) + 2 * (len(tail) - 1)
                if tail_len + len(unit) + 2 <= size:
                    break
                tail.pop(0)

            current = tail
            current_len = sum(len(u) for u in current) + 2 * max(len(current) - 1, 0)
            extra = len(unit) + (2 if current else 0)

        current.append(unit)
        current_len += extra

    if current:
        chunks.append("\n\n".join(current))

    return [Chunk(text=c, source=source, index=i) for i, c in enumerate(chunks)]
