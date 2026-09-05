from __future__ import annotations

from app.api.schemas import ClaimLabel
from app.verification.labels import fuse_label
from app.verification.nli import NLIScore
from app.verification.numeric import check_numbers, extract_numbers
from app.verification.structural import check_citations

# --- layer 1: structural --------------------------------------------------


def test_structural_all_valid():
    res = check_citations(["a#one", "b#two"], {"a#one", "b#two", "c#three"})
    assert res.ok is True
    assert res.fabricated is False
    assert res.valid_cites == ("a#one", "b#two")


def test_structural_flags_unretrieved_citation():
    res = check_citations(["a#one", "ghost#x"], {"a#one"})
    assert res.ok is False
    assert res.invalid_cites == ("ghost#x",)
    assert res.fabricated is False  # still has one valid cite


def test_structural_empty_is_fabricated():
    assert check_citations([], {"a#one"}).fabricated is True
    assert check_citations(["ghost#x"], {"a#one"}).fabricated is True


def test_structural_dedupes():
    res = check_citations(["a#one", "a#one"], {"a#one"})
    assert res.valid_cites == ("a#one",)


# --- layer 3: numeric ---------------------------------------------------


def test_extract_numbers_kinds():
    kinds = {t.raw: t.kind for t in extract_numbers("97.2% over 422,458 cases in 2024, 15 GB, 3x")}
    assert kinds["97.2%"] == "percent"
    assert kinds["422,458"] == "number"
    assert kinds["2024"] == "year"
    assert kinds["15 GB"] == "bytes"
    assert kinds["3x"] == "multiplier"


def test_numeric_guard_passes_on_verbatim():
    res = check_numbers(
        "97.2% article-level hit rate", ["... 97.2% article-level retrieval hit rate ..."]
    )
    assert res.flagged is False


def test_numeric_guard_matches_thousands_separator():
    res = check_numbers("processed 422458 cases", ["feature engineering on 422,458 rows"])
    assert res.flagged is False


def test_numeric_guard_flags_inflated_number():
    res = check_numbers("over 99% hit rate", ["most recent run: 97.2% article-level hit rate"])
    assert res.flagged is True
    assert "99%" in res.detail


# --- fusion ----------------------------------------------------------------


def _struct(ok=True, fabricated=False):
    from app.verification.structural import StructuralResult

    return StructuralResult(
        ok=ok,
        valid_cites=() if fabricated else ("a#one",),
        invalid_cites=() if ok else ("ghost#x",),
    )


def test_fuse_fabricated_short_circuits():
    label, _flag, _lex = fuse_label(
        _struct(ok=False, fabricated=True), None, check_numbers("x", ["x"])
    )
    assert label is ClaimLabel.fabricated_citation


def test_fuse_supported():
    label, _, _ = fuse_label(_struct(), NLIScore(0.9, 0.08, 0.02), check_numbers("x", ["x"]))
    assert label is ClaimLabel.supported


def test_fuse_contradicted():
    label, _, _ = fuse_label(_struct(), NLIScore(0.1, 0.1, 0.8), check_numbers("x", ["x"]))
    assert label is ClaimLabel.contradicted


def test_fuse_unsupported_when_neutral():
    label, _, _ = fuse_label(_struct(), NLIScore(0.3, 0.6, 0.1), check_numbers("x", ["x"]))
    assert label is ClaimLabel.unsupported


def test_lexical_backstop_rescues_neutral_when_coverage_is_high():
    label, _, lex = fuse_label(
        _struct(), NLIScore(0.2, 0.75, 0.05), check_numbers("x", ["x"]), lexical_coverage=0.95
    )
    assert label is ClaimLabel.supported
    assert lex is True


def test_lexical_backstop_never_overrides_contradiction():
    label, _, lex = fuse_label(
        _struct(), NLIScore(0.05, 0.15, 0.8), check_numbers("x", ["x"]), lexical_coverage=1.0
    )
    assert label is ClaimLabel.contradicted
    assert lex is False


def test_numeric_flag_does_not_override_label():
    label, flag, _ = fuse_label(
        _struct(),
        NLIScore(0.9, 0.05, 0.05),
        check_numbers("99%", ["the figure was 40%"]),
    )
    assert label is ClaimLabel.supported
    assert flag is True
