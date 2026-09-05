"""Layer 3: numeric guard — every number/%/date in the answer must be in a cited chunk.

Motivation (brief §1): the failure mode this project targets is *rounded-up
metrics* and invented figures. Retrieval + NLI catch unsupported prose, but NLI is
weak at "97.2 vs 98" precision. So any numeric token in a claim is checked
literally against the text of the chunks that claim cites.

A token passes if it appears in a cited chunk:
  * verbatim ("97.2%", "422,458", "2024"), or
  * as the same value written differently (``422458`` == ``422,458``,
    ``1.29M`` == ``1,290,000`` == ``1290000``), or
  * as an exact unit conversion from a small known table (GB/MB, %, x-multiplier).

Anything else sets ``numeric_flag`` with a note naming the offending token. False
positives are acceptable here — the flag is advisory and shown, not fatal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches: 97.2%  |  1,290,000  |  1.29M  |  $51.25  |  2024  |  15 GB  |  99.4 %
_NUM_RE = re.compile(
    r"""
    (?P<currency>\$)?
    (?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)
    \s*
    (?P<suffix>%|k|K|M|B|bn|GB|MB|KB|TB|x|×)?
    """,
    re.VERBOSE,
)

_SCALE = {
    "k": 1_000,
    "K": 1_000,
    "M": 1_000_000,
    "m": 1_000_000,
    "B": 1_000_000_000,
    "bn": 1_000_000_000,
}
_BYTE_SCALE = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

# Standalone years / ISO-ish dates get looser treatment: match the year substring.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass(slots=True)
class NumericToken:
    raw: str
    value: float | None
    kind: str  # "number" | "percent" | "bytes" | "multiplier" | "year"


@dataclass(slots=True)
class NumericResult:
    flagged: bool
    detail: str | None = None
    unmatched: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> list[NumericToken]:
    """Pull numeric tokens out of a claim. Years are captured separately."""
    tokens: list[NumericToken] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0).strip()
        num = m.group("num").replace(",", "")
        suffix = m.group("suffix") or ""
        try:
            base = float(num)
        except ValueError:  # pragma: no cover - regex guarantees a number
            continue
        if suffix == "%":
            tokens.append(NumericToken(raw, base, "percent"))
        elif suffix in _BYTE_SCALE:
            tokens.append(NumericToken(raw, base * _BYTE_SCALE[suffix], "bytes"))
        elif suffix in _SCALE:
            tokens.append(NumericToken(raw, base * _SCALE[suffix], "number"))
        elif suffix in ("x", "×"):
            tokens.append(NumericToken(raw, base, "multiplier"))
        else:
            kind = "year" if _YEAR_RE.fullmatch(num) else "number"
            tokens.append(NumericToken(raw, base, kind))
    return tokens


def _value_present(tok: NumericToken, haystack: str) -> bool:
    """Is this token's value expressed anywhere in the cited text?"""
    if tok.raw in haystack:
        return True
    if tok.raw.replace(",", "") in haystack.replace(",", ""):
        return True
    if tok.value is None:
        return False
    for cand in extract_numbers(haystack):
        if cand.kind != tok.kind or cand.value is None:
            continue
        if abs(cand.value - tok.value) <= 1e-6 * max(1.0, abs(tok.value)):
            return True
    return False


def check_numbers(claim_text: str, cited_texts: list[str]) -> NumericResult:
    """Flag any numeric token in ``claim_text`` absent from every cited chunk."""
    haystack = "\n".join(cited_texts)
    unmatched = [
        tok.raw for tok in extract_numbers(claim_text) if not _value_present(tok, haystack)
    ]
    if not unmatched:
        return NumericResult(flagged=False)
    return NumericResult(
        flagged=True,
        detail=f"not found in cited text: {', '.join(dict.fromkeys(unmatched))}",
        unmatched=list(dict.fromkeys(unmatched)),
    )
