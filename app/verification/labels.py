"""Fuse the three layers into one ``ClaimLabel`` (+ numeric flag).

Decision order (fons-iuris semantics, brief §4):
  1. Structural fail  -> ``fabricated_citation`` (NLI never runs).
  2. NLI contradiction dominates and is confident -> ``contradicted``.
  3. NLI entailment clears the threshold -> ``supported``.
  4. Otherwise -> ``unsupported`` (neutral).

The numeric guard is orthogonal: it sets ``numeric_flag`` on the result but does
not by itself change the label (a ``supported`` claim with a mis-stated number is
still shown as supported-but-flagged, which is the more useful signal). Thresholds
are module constants so Phase 2 can tune them against the eval and record the
values in ARCHITECTURE.md.
"""

from __future__ import annotations

from app.api.schemas import ClaimLabel
from app.verification.nli import NLIScore
from app.verification.numeric import NumericResult
from app.verification.structural import StructuralResult

ENTAILMENT_THRESHOLD = 0.55
CONTRADICTION_THRESHOLD = 0.50


def fuse_label(
    structural: StructuralResult,
    nli: NLIScore | None,
    numeric: NumericResult,
) -> tuple[ClaimLabel, bool]:
    """Return ``(label, numeric_flag)``."""
    if structural.fabricated:
        return ClaimLabel.fabricated_citation, numeric.flagged
    if nli is None:
        return ClaimLabel.unsupported, numeric.flagged
    if nli.contradiction >= CONTRADICTION_THRESHOLD and nli.contradiction >= nli.entailment:
        return ClaimLabel.contradicted, numeric.flagged
    if nli.entailment >= ENTAILMENT_THRESHOLD:
        return ClaimLabel.supported, numeric.flagged
    return ClaimLabel.unsupported, numeric.flagged
