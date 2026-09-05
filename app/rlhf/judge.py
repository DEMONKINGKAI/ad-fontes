"""Score candidates for preference-pair construction. (Impl: Phase 3.)

Three inputs combined into a weighted scalar with a hard veto:
  1. this repo's verification labels (structural + NLI);
  2. the numeric guard;
  3. an LLM-judge rubric: faithfulness, humility (no overclaiming), audience fit,
     concision, correct third-person voice.

Hard veto: any candidate with a ``contradicted`` or ``fabricated_citation`` claim
cannot be 'chosen'. Kai hand-labels ~100 pairs; if judge-human agreement < ~80%
the rubric is revised before training.
"""

from __future__ import annotations

RUBRIC_WEIGHTS = {
    "faithfulness": 0.40,
    "humility": 0.25,
    "audience_fit": 0.15,
    "concision": 0.10,
    "third_person_voice": 0.10,
}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - Phase 3
    raise NotImplementedError("Implemented in Phase 3 (preference data pipeline).")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
