"""Phase 3 pipeline report — question mix, candidate quality by source, the
judge's and the verification layer's **perturbation detection rate**, pair
counts, and the length-bias check.

    python -m app.rlhf.report            # -> data/rlhf/phase3-report.md + stdout
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from app.rlhf._io import RLHF_DIR, read_jsonl


def _mean(xs) -> float:
    xs = list(xs)
    return round(statistics.mean(xs), 3) if xs else 0.0


def build_report() -> str:
    questions = read_jsonl(RLHF_DIR / "questions.jsonl")
    cand_rows = read_jsonl(RLHF_DIR / "candidates.jsonl")
    judged = read_jsonl(RLHF_DIR / "judged.jsonl")
    pairs_full = read_jsonl(RLHF_DIR / "pairs_full.jsonl")

    lines: list[str] = ["# Phase 3 — preference-data pipeline report", ""]

    # -- questions -----------------------------------------------
    lines += [
        "## Questions",
        "",
        f"- total **{len(questions)}** · holdout {sum(q['holdout'] for q in questions)} "
        f"· train {sum(not q['holdout'] for q in questions)}",
        f"- buckets: {dict(Counter(q['bucket'] for q in questions))}",
        f"- personas: {dict(Counter(q['persona'] for q in questions))}",
        "",
    ]

    if not cand_rows:
        lines.append("_(no candidates yet — run `python -m app.rlhf.gen_candidates`)_")
        return "\n".join(lines)

    cands = [c for r in cand_rows for c in r["candidates"]]
    lines += [
        "## Candidates",
        "",
        f"- {len(cand_rows)} questions x ~{len(cands) / len(cand_rows):.1f} candidates = {len(cands)}",
        "",
        "| source | n | supported/ans | unsup+fab/ans | contra/ans | unverified prose/ans |",
        "|--|--|--|--|--|--|",
    ]
    by_src: dict[str, list] = defaultdict(list)
    for c in cands:
        by_src[c["source"].split(":")[0] if c["perturbation"] is None else "perturb"].append(c)
    for src, cs in sorted(by_src.items()):
        v = [c["verification"] for c in cs]
        lines.append(
            f"| {src} | {len(cs)} | {_mean(x['supported'] for x in v)} | "
            f"{_mean(x['unsupported_plus_fabricated'] for x in v)} | "
            f"{_mean(x['contradicted'] for x in v)} | {_mean(x['unverified_prose_count'] for x in v)} |"
        )

    # -- perturbation detection --------------------------------
    if judged:
        lines += ["", "## Perturbation detection rate", ""]
        jd = {j["candidate_id"]: j for j in judged}
        perts = [c for c in cands if c["perturbation"]]
        lines += [
            f"{len(perts)} perturbed candidates. 'Caught' = the layer flagged it "
            "(verification: any contradicted/fabricated/numeric/>=1 unverified-prose; "
            "judge: veto or faithfulness <= 3).",
            "",
            "| perturbation | n | verification caught | judge caught | either |",
            "|--|--|--|--|--|",
        ]
        by_type: dict[str, list] = defaultdict(list)
        for c in perts:
            by_type[c["perturbation"]].append(c)
        for ptype, cs in sorted(by_type.items()):
            vcaught = jcaught = either = 0
            for c in cs:
                vf = c["verification"]
                v_hit = (
                    vf["contradicted"]
                    or vf["fabricated"]
                    or vf["numeric_flags"]
                    or vf["unverified_prose_count"] >= 1
                )
                j = jd.get(c["candidate_id"])
                j_hit = bool(j and (j["veto"] or j["rubric"].get("faithfulness", 5) <= 3))
                vcaught += bool(v_hit)
                jcaught += bool(j_hit)
                either += bool(v_hit or j_hit)
            n = len(cs)
            lines.append(f"| {ptype} | {n} | {vcaught}/{n} | {jcaught}/{n} | {either}/{n} |")

        # -- judge score by source --------------------------
        lines += [
            "",
            "## Judge scalar by candidate source",
            "",
            "| source | n | mean judge_scalar | vetoed |",
            "|--|--|--|--|",
        ]
        jby: dict[str, list] = defaultdict(list)
        for j in judged:
            jby[j["source"].split(":")[0] if j["perturbation"] is None else "perturb"].append(j)
        for src, js in sorted(jby.items()):
            lines.append(
                f"| {src} | {len(js)} | {_mean(x['judge_scalar'] for x in js)} | "
                f"{sum(1 for x in js if x['veto'])}/{len(js)} |"
            )

    # -- pairs ---------------------------------------------------
    if pairs_full:
        lines += ["", "## Pairs", ""]
        deltas = [p["len_delta"] for p in pairs_full]
        z = _mean(deltas) / (statistics.pstdev(deltas) or 1)
        lines += [
            f"- **{len(pairs_full)}** pairs "
            f"({sum(not p['holdout'] for p in pairs_full)} train / "
            f"{sum(p['holdout'] for p in pairs_full)} holdout)",
            f"- chosen source mix: {dict(Counter(p['chosen_source'] for p in pairs_full))}",
            f"- rejected source mix: {dict(Counter(p['rejected_source'] for p in pairs_full))}",
            f"- mean score margin: {_mean(p['score_margin'] for p in pairs_full)}",
            f"- length bias: mean(len_delta rejected-chosen) = {_mean(deltas)} chars, "
            f"z ~ {z:.2f} (near 0 => length not predictive of preference)",
        ]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=RLHF_DIR / "phase3-report.md")
    args = p.parse_args(argv)
    md = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md.encode("ascii", "replace").decode())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
