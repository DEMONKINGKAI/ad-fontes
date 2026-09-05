"""Produce 3-4 answer candidates per question. (Impl: Phase 3.)

Sources: base local model at two temperatures, a larger hosted model, and
``perturb.py`` edits of a faithful answer. Each candidate records its origin and
any perturbation labels so the judge's detection rate is measurable.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - Phase 3
    raise NotImplementedError("Implemented in Phase 3 (preference data pipeline).")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
