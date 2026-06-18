"""
context/ingest_laws.py — build the Laws-of-the-Game knowledge base.

Docling parses the IFAB Laws of the Game PDF and HybridChunker splits it into
structure-aware, contextualized chunks; each chunk is embedded with the Granite
embedder and the vectors + texts are written to context/index/ as a small local
index that context/retrieve.py queries. Run once (or whenever the PDF changes):

    python -m context.ingest_laws                  # uses context/laws/laws_of_the_game.pdf
    python -m context.ingest_laws --pdf path/to/laws.pdf

Place the IFAB Laws of the Game PDF (publicly available) at context/laws/. The
generated index under context/index/ is disposable — git-ignore it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from context.retrieve import INDEX_DIR, embed_texts

HERE = Path(__file__).resolve().parent
DEFAULT_PDF = HERE / "laws" / "laws_of_the_game.pdf"


def parse_chunks(pdf_path: Path) -> list[str]:
    """Docling: PDF -> structure-aware, contextualized text chunks."""
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    doc = DocumentConverter().convert(str(pdf_path)).document
    chunker = HybridChunker()
    return [chunker.contextualize(chunk=c) for c in chunker.chunk(dl_doc=doc)]


def build_index(pdf_path: Path = DEFAULT_PDF, index_dir: Path = INDEX_DIR) -> Path:
    """Parse -> embed -> persist the local Laws index. Returns the index directory."""
    import numpy as np
    from agent.granite_client import embed_model_id

    pdf_path, index_dir = Path(pdf_path), Path(index_dir)
    if not pdf_path.exists():
        raise SystemExit(
            f"Laws PDF not found at {pdf_path}. Download the IFAB Laws of the Game PDF "
            f"(https://www.theifab.com/laws-of-the-game-documents/) and place it there."
        )
    chunks = parse_chunks(pdf_path)
    if not chunks:
        raise SystemExit("Docling produced no chunks from the PDF.")

    vectors = np.asarray(embed_texts(chunks), dtype="float32")
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "laws_chunks.json").write_text(
        json.dumps(
            {"chunks": chunks, "source": pdf_path.name,
             "model": embed_model_id(), "dims": int(vectors.shape[1])},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    np.save(index_dir / "laws_vectors.npy", vectors)
    print(f"Indexed {len(chunks)} chunks ({vectors.shape[1]}-dim) -> {index_dir}")
    return index_dir


def main(argv=None) -> int:
    """Run the ingest from the command line."""
    parser = argparse.ArgumentParser(
        description="Ingest the Laws of the Game into a local vector index (Docling + Granite)."
    )
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="Path to the Laws of the Game PDF.")
    parser.add_argument("--out", default=str(INDEX_DIR), help="Output index directory.")
    args = parser.parse_args(argv)

    try:  # local convenience so GRANITE_* is available; not needed on a configured host
        from dotenv import load_dotenv
        load_dotenv(HERE.parent / ".env")
    except ImportError:
        pass

    build_index(args.pdf, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
