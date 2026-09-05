"""``ad-fontes-ingest`` — build or rebuild the Chroma index from the corpus.

Usage (Phase 1 onward)::

    python -m app.ingestion.cli --rebuild
    ad-fontes-ingest --stats

Phase 0: the command exists and parses args so the Docker build step and
``docker compose`` can call it; it exits with a clear message until Phase 1.
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ad-fontes-ingest", description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Drop and rebuild the index.")
    parser.add_argument("--stats", action="store_true", help="Print index size and corpus version.")
    parser.add_argument("--dry-run", action="store_true", help="Chunk only; do not embed or write.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    if args.stats:
        print(f"corpus_dir      : {settings.corpus_dir}")
        print(f"index_dir       : {settings.index_dir}")
        print(f"corpus_version  : {settings.corpus_version}")
        print(f"embed_model     : {settings.embed_model}")
        return 0
    print(
        "ingestion pipeline lands in Phase 1; "
        "re-run once app/ingestion/loader.py and app/retrieval/* are implemented.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
