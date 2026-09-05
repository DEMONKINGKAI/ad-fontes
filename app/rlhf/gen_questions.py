"""Generate ~600 stratified questions from the corpus. (Impl: Phase 3.)

Templates x personas (recruiter, HR screener, ML lead, skeptical CTO) over every
project and section. Stratified by project, section type, and persona; 20% held
out for eval and never used to build training pairs.
"""

from __future__ import annotations

PERSONAS = ("recruiter", "hr_screener", "ml_lead", "skeptical_cto")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - Phase 3
    raise NotImplementedError("Implemented in Phase 3 (preference data pipeline).")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
