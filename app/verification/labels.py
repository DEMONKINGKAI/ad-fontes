"""Fuse the layers into one ``ClaimLabel`` (+ numeric flag).

Decision order (fons-iuris semantics, brief §4):
  1. Structural fail                        -> ``fabricated_citation`` (NLI skipped).
  2. NLI contradiction confident + dominant  -> ``contradicted``.
  3. NLI entailment clears the threshold     -> ``supported``.
  4. Otherwise                               -> ``unsupported`` (neutral).

The numeric guard is orthogonal — it sets ``numeric_flag`` but does not change the
label (a supported claim with a mis-stated number is more useful shown as
supported-but-flagged). Thresholds are the fons-iuris defaults; the base-model
eval (Phase 2.5) did not justify moving them.

``unsupported`` means *NLI could not confirm it*, not *it is false* — DeBERTa-base
has real recall limits on portfolio prose (see ``app.verification.nli``). The
Phase 3 LLM judge is the stronger faithfulness signal for the preference data.
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
    flag = numeric.flagged
    if structural.fabricated:
        return ClaimLabel.fabricated_citation, flag
    if nli is None:
        return ClaimLabel.unsupported, flag
    if nli.contradiction >= CONTRADICTION_THRESHOLD and nli.contradiction >= nli.entailment:
        return ClaimLabel.contradicted, flag
    if nli.entailment >= ENTAILMENT_THRESHOLD:
        return ClaimLabel.supported, flag
    return ClaimLabel.unsupported, flag
