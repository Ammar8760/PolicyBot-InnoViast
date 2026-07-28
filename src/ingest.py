"""
Ingestion pipeline — knowledge base ko searchable banana.

    knowledge_base/*.md,*.txt,*.pdf
            |
            v  (1) text nikalo
            v  (2) chunks mein toro
            v  (3) har chunk ka embedding banao
            v  (4) ChromaDB mein save karo
        chroma_db/

Ye ek dafa chalti hai (ya jab bhi documents badlein). Har sawal par nahi.

Chalane ka tareeqa:
    python -m src.ingest            # poora index dobara banao
    python -m src.ingest --keep     # purana rakh kar naye docs add karo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, embedder, vector_store
from .chunker import Chunk, chunk_text

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


# ------------------------------------------------------------ file reading --
def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_files(directory: Path) -> list[Path]:
    """Knowledge base folder se sab supported files lo (recursive)."""
    if not directory.exists():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


# ---------------------------------------------------------------- pipeline --
def build_chunks(files: list[Path]) -> list[Chunk]:
    """Har file ko parh kar chunks banao."""
    all_chunks: list[Chunk] = []

    for path in files:
        text = read_document(path).strip()
        if not text:
            print(f"  [skip] {path.name} — khaali file")
            continue

        chunks = chunk_text(text, source=path.name)
        all_chunks.extend(chunks)
        print(f"  [ok]   {path.name:<38} {len(text):>6} chars -> {len(chunks)} chunks")

    return all_chunks


def ingest(keep_existing: bool = False) -> dict:
    """Poori ingestion pipeline chalao."""
    print(f"\n[1/4] Documents dhoond rahe hain: {config.KNOWLEDGE_BASE_DIR}")
    files = collect_files(config.KNOWLEDGE_BASE_DIR)
    if not files:
        print("  !! Koi document nahi mila. knowledge_base/ mein files daalein.")
        return {"files": 0, "chunks": 0}
    print(f"  {len(files)} file(s) mili")

    print("\n[2/4] Chunking...")
    chunks = build_chunks(files)
    if not chunks:
        print("  !! Koi chunk nahi bana.")
        return {"files": len(files), "chunks": 0}
    print(f"  Total {len(chunks)} chunks")

    print(f"\n[3/4] Embeddings bana rahe hain ({config.EMBEDDING_MODEL})...")
    print("  (yeh thora waqt le sakta hai — API calls ho rahi hain)")
    vectors = embedder.embed_documents([c.text for c in chunks])
    print(f"  {len(vectors)} vectors bane, har ek {len(vectors[0])} dimensions ka")

    print(f"\n[4/4] {vector_store.backend_name()} mein save kar rahe hain...")
    if not keep_existing:
        vector_store.reset_collection()
        print("  Purana index clear kar diya")
    vector_store.add_chunks(chunks, vectors)

    info = vector_store.stats()
    print(f"\n  DONE — index mein {info['chunks']} chunks hain")
    print(f"  Sources: {', '.join(info['sources'])}\n")
    return {"files": len(files), "chunks": len(chunks)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Knowledge base ko index karein")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="purana index na mitayein, sirf naye chunks add karein",
    )
    args = parser.parse_args()

    try:
        ingest(keep_existing=args.keep)
    except Exception as exc:
        print(f"\n  ERROR: {exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
