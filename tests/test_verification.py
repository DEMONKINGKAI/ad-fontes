from __future__ import annotations

from app.api.schemas import ClaimLabel
from app.verification.labels import fuse_label
from app.verification.nli import NLIScore, _windows
from app.verification.numeric import check_numbers, extract_numbers
from app.verification.structural import check_citations

# --- NLI windowing (latency fix, Phase 2.6) -----------------------------


def test_windows_hard_splits_punctuationless_premise():
    # a markdown-table chunk: long, only "|" and newlines, no . ! ? ;
    table = "tech-stack-map\n" + " | ".join(f"tech{i} project role" for i in range(200))
    assert len(table) > 3000
    wins = _windows(table)
    # every window stays small enough that the NLI batch won't pad to max length
    assert wins and all(len(w) <= 700 for w in wins)


def test_windows_short_premise_is_one_window():
    assert _windows("A short grounded sentence about the project.") == [
        "A short grounded sentence about the project."
    ]


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
    label, _flag = fuse_label(_struct(ok=False, fabricated=True), None, check_numbers("x", ["x"]))
    assert label is ClaimLabel.fabricated_citation


def test_fuse_supported():
    label, _ = fuse_label(_struct(), NLIScore(0.9, 0.08, 0.02), check_numbers("x", ["x"]))
    assert label is ClaimLabel.supported


def test_fuse_contradicted():
    label, _ = fuse_label(_struct(), NLIScore(0.1, 0.1, 0.8), check_numbers("x", ["x"]))
    assert label is ClaimLabel.contradicted


def test_fuse_unsupported_when_neutral():
    label, _ = fuse_label(_struct(), NLIScore(0.3, 0.6, 0.1), check_numbers("x", ["x"]))
    assert label is ClaimLabel.unsupported


def test_fuse_contradiction_only_when_it_dominates_entailment():
    # a framed-true claim: high entailment via a stripped variant, moderate
    # contradiction from the framed form — must NOT be labelled contradicted
    label, _ = fuse_label(_struct(), NLIScore(0.98, 0.0, 0.6), check_numbers("x", ["x"]))
    assert label is ClaimLabel.supported


def test_fuse_conflation_is_unsupported_not_contradicted():
    # a conflation of two source statements: low entailment, moderate
    # contradiction that clears the threshold but does not clear it *by the
    # margin* -> "NLI could not confirm it", not "the source contradicts it"
    label, _ = fuse_label(_struct(), NLIScore(0.42, 0.0, 0.52), check_numbers("x", ["x"]))
    assert label is ClaimLabel.unsupported


def test_fuse_clear_contradiction_still_fires():
    label, _ = fuse_label(_struct(), NLIScore(0.20, 0.1, 0.70), check_numbers("x", ["x"]))
    assert label is ClaimLabel.contradicted


def test_numeric_flag_does_not_override_label():
    label, flag = fuse_label(
        _struct(),
        NLIScore(0.9, 0.05, 0.05),
        check_numbers("99%", ["the figure was 40%"]),
    )
    assert label is ClaimLabel.supported
    assert flag is True
