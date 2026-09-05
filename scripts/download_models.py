"""Pre-fetch models into the image at build time (brief §2: warm at build).

Enabled from Phase 1 (embedder) / Phase 2 (NLI, GGUFs). Kept as a stub now so the
Dockerfile's build contract is visible. Downloads into ``HF_HOME`` /
``settings.model_cache_dir``; the runtime stage copies that directory.

    python -m scripts.download_models --embed --nli --base-gguf [--tuned-gguf]
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embed", action="store_true")
    p.add_argument("--nli", action="store_true")
    p.add_argument("--base-gguf", action="store_true")
    p.add_argument("--tuned-gguf", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print(
        "model warm-up is wired in Phase 1 (embedder) and Phase 2 (NLI + GGUFs).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
