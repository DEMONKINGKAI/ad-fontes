"""Layer 1: structural grounding — did the claim cite chunks that were retrieved?

Cheap, deterministic, and runs before the NLI layer (mirrors fons-iuris). A claim
that cites a ``chunk_id`` never in the retrieved set is a ``fabricated_citation``
and NLI is skipped for it — there is no premise to check against.

A claim with an empty citation list is also structurally invalid: the generator
was told every claim must cite at least one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class StructuralResult:
    ok: bool
    valid_cites: tuple[str, ...]
    invalid_cites: tuple[str, ...]

    @property
    def fabricated(self) -> bool:
        """True when the claim has no usable citation (all invalid, or none given)."""
        return not self.valid_cites


def check_citations(cited: list[str], retrieved_ids: set[str]) -> StructuralResult:
    """Partition a claim's citations into those in the retrieved set and those not."""
    seen: set[str] = set()
    valid: list[str] = []
    invalid: list[str] = []
    for cid in cited:
        norm = cid.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        (valid if norm in retrieved_ids else invalid).append(norm)
    return StructuralResult(
        ok=not invalid and bool(valid),
        valid_cites=tuple(valid),
        invalid_cites=tuple(invalid),
    )
