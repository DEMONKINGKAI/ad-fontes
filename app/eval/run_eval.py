"""Drive the pipeline over ``questions.jsonl`` and emit a metrics report.

    python -m app.eval.run_eval --stage retrieval      # Phase 1
    python -m app.eval.run_eval --stage generation --model base   # Phase 2
    python -m app.eval.run_eval --stage compare        # Phase 5 (base vs tuned)

Phase 0: argument parsing + question loading only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_QUESTIONS = Path(__file__).with_name("questions.jsonl")


def load_questions(path: Path = _QUESTIONS) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ad-fontes-eval", description=__doc__)
    p.add_argument("--stage", choices=["retrieval", "generation", "compare"], default="retrieval")
    p.add_argument("--model", choices=["base", "tuned"], default="base")
    p.add_argument("--out", type=Path, default=Path("data/eval"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    questions = load_questions()
    print(f"loaded {len(questions)} questions; stage={args.stage}")
    print("eval execution lands with the stage it measures (Phase 1 / 2 / 5).")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
