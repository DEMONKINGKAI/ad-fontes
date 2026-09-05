"""Held-out base-vs-tuned comparison and plots. (Impl: Phase 5.)

Runs both generators over the held-out questions + the adversarial overclaiming
set, applies full verification, and reports: unsupported+fabricated per 100,
numeric violations, judge win rate, length, latency, and decline correctness on
negative controls — each with a bootstrap CI. Emits the base-vs-tuned figure used
in the README.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - Phase 5
    raise NotImplementedError("Implemented in Phase 5 (evaluation and the story).")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
