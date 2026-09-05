"""Assemble (chosen, rejected) pairs in TRL DPO format. (Impl: Phase 3.)

A pair is kept when 'chosen' beats 'rejected' by a score margin and 'chosen'
passes the veto. Length is explicitly de-correlated from preference (balance so a
length-only classifier cannot predict the label) and the residual length bias is
reported. Output: ``data/rlhf/pairs.jsonl`` with fields ``prompt``, ``chosen``,
``rejected`` (+ provenance metadata).
"""

from __future__ import annotations

MIN_SCORE_MARGIN = 0.15


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - Phase 3
    raise NotImplementedError("Implemented in Phase 3 (preference data pipeline).")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
