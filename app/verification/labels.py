"""Fuse the layers into one ``ClaimLabel`` (+ numeric flag).

Decision order (fons-iuris semantics, brief §4):
  1. Structural fail                       -> ``fabricated_citation`` (NLI skipped).
  2. NLI contradiction confident + dominant -> ``contradicted``.
  3. NLI entailment clears the threshold    -> ``supported``.
  4. NLI neutral BUT the claim's content words are almost entirely present in the
     cited text (and nothing is contradicted) -> ``supported`` (lexical backstop).
  5. Otherwise                              -> ``unsupported`` (neutral).

**Why step 4:** ``DeBERTa-v3-base`` rates *aggregate* claims neutral even when
every element is in the source — "Kai's Bayesian-network projects are Threadfall,
evidentia and Causeway" vs a chunk that lists exactly those three. That is an NLI
recall gap, not an unfaithful claim, so a high-precision lexical check rescues it.
It is deliberately strict (``LEXICAL_COVERAGE_THRESHOLD`` = 0.9 of the non-stop
content words) and never overrides a contradiction. Measured effect is recorded
in ARCHITECTURE.md.

The numeric guard is orthogonal — it sets ``numeric_flag`` but does not change the
label (a supported claim with a mis-stated number is more useful shown as
supported-but-flagged). Thresholds are module constants, tuned against the eval.
"""

from __future__ import annotations

from app.api.schemas import ClaimLabel
from app.verification.nli import NLIScore
from app.verification.numeric import NumericResult
from app.verification.structural import StructuralResult

ENTAILMENT_THRESHOLD = 0.55
CONTRADICTION_THRESHOLD = 0.50
LEXICAL_COVERAGE_THRESHOLD = 0.9


def fuse_label(
    structural: StructuralResult,
    nli: NLIScore | None,
    numeric: NumericResult,
    *,
    lexical_coverage: float = 0.0,
) -> tuple[ClaimLabel, bool, bool]:
    """Return ``(label, numeric_flag, lexical_backstop_used)``."""
    flag = numeric.flagged
    if structural.fabricated:
        return ClaimLabel.fabricated_citation, flag, False
    if nli is None:
        return ClaimLabel.unsupported, flag, False
    if nli.contradiction >= CONTRADICTION_THRESHOLD and nli.contradiction >= nli.entailment:
        return ClaimLabel.contradicted, flag, False
    if nli.entailment >= ENTAILMENT_THRESHOLD:
        return ClaimLabel.supported, flag, False
    if (
        lexical_coverage >= LEXICAL_COVERAGE_THRESHOLD
        and nli.contradiction < CONTRADICTION_THRESHOLD
    ):
        return ClaimLabel.supported, flag, True
    return ClaimLabel.unsupported, flag, False
