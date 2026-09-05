"""Pre-fetch models into the local HF cache / model dir (brief §2: warm at build).

    python -m scripts.download_models --embed --nli --base-gguf [--tuned-gguf]

Called by the Docker builder stage so the runtime image needs no network. Safe to
run repeatedly — Hugging Face caches by content hash.
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings


def _embed(model_name: str) -> None:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model_name, trust_remote_code=True, device="cpu")


def _nli(model_name: str) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)


def _gguf(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embed", action="store_true")
    p.add_argument("--nli", action="store_true")
    p.add_argument("--base-gguf", action="store_true")
    p.add_argument("--tuned-gguf", action="store_true")
    p.add_argument("--all", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = get_settings()
    did = False
    if args.embed or args.all:
        print(f"embed: {s.embed_model}")
        _embed(s.embed_model)
        did = True
    if args.nli or args.all:
        print(f"nli: {s.nli_model}")
        _nli(s.nli_model)
        did = True
    if args.base_gguf or args.all:
        print(f"base gguf: {s.base_gguf_repo}/{s.base_gguf_file}")
        print("  ->", _gguf(s.base_gguf_repo, s.base_gguf_file))
        did = True
    if args.tuned_gguf or args.all:
        try:
            print(f"tuned gguf: {s.tuned_gguf_repo}/{s.tuned_gguf_file}")
            print("  ->", _gguf(s.tuned_gguf_repo, s.tuned_gguf_file))
        except Exception as exc:
            print(
                f"  tuned GGUF not available yet ({type(exc).__name__}); skipping", file=sys.stderr
            )
        did = True
    if not did:
        build_parser().print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
