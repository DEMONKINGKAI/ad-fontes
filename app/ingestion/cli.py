"""``ad-fontes-ingest`` — build or inspect the Chroma index from the corpus.

    python -m app.ingestion.cli --rebuild      # drop + re-embed + re-upsert
    python -m app.ingestion.cli --stats        # index size + corpus version
    python -m app.ingestion.cli --dry-run      # chunk only; print the plan

``--rebuild`` is what the Docker build step calls to bake the index into the
image; it is idempotent (``store.reset()`` first) and deterministic given the
corpus and the embedding model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import Settings, get_settings
from app.ingestion.loader import Chunk, load_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ad-fontes-ingest", description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Drop and rebuild the index.")
    parser.add_argument("--stats", action="store_true", help="Print index size and corpus version.")
    parser.add_argument("--dry-run", action="store_true", help="Chunk only; do not embed or write.")
    return parser


def rebuild_index(settings: Settings) -> tuple[list[Chunk], int]:
    """Re-embed every chunk and replace the collection. Returns (chunks, count)."""
    from app.retrieval.embedder import load_embedder
    from app.retrieval.vector_store import ChromaVectorStore

    chunks = load_corpus(Path(settings.corpus_dir))
    embedder = load_embedder(settings.embed_model)
    store = ChromaVectorStore(Path(settings.index_dir), settings.chroma_collection)
    store.reset()

    embeddings = embedder.embed_documents([c.embed_text for c in chunks])
    store.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[c.to_chroma_metadata() for c in chunks],
    )
    return chunks, store.count()


def _print_stats(settings: Settings) -> None:
    print(f"corpus_dir      : {settings.corpus_dir}")
    print(f"index_dir       : {settings.index_dir}")
    print(f"corpus_version  : {settings.corpus_version}")
    print(f"embed_model     : {settings.embed_model}")
    chunks = load_corpus(Path(settings.corpus_dir))
    print(f"corpus chunks   : {len(chunks)}")
    try:
        from app.retrieval.vector_store import ChromaVectorStore

        store = ChromaVectorStore(Path(settings.index_dir), settings.chroma_collection)
        print(f"index size      : {store.count()}")
    except Exception as exc:
        print(f"index size      : unavailable ({type(exc).__name__})")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.dry_run:
        chunks = load_corpus(Path(settings.corpus_dir))
        by_type: dict[str, int] = {}
        for c in chunks:
            by_type[c.doc_type] = by_type.get(c.doc_type, 0) + 1
        print(f"{len(chunks)} chunks from {settings.corpus_dir}")
        for dtype, n in sorted(by_type.items()):
            print(f"  {dtype:12s} {n}")
        for c in chunks:
            print(f"  {c.chunk_id}")
        return 0

    if args.rebuild:
        chunks, count = rebuild_index(settings)
        print(f"rebuilt index: {count} chunks embedded with {settings.embed_model}")
        if count != len(chunks):
            print(f"WARNING: embedded {count} but chunked {len(chunks)}", file=sys.stderr)
            return 1
        return 0

    if args.stats:
        _print_stats(settings)
        return 0

    build_parser().print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
