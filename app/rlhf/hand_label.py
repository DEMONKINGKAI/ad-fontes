"""Export ~100 preference pairs for Kai to hand-label, then measure judge–human
agreement (brief §3: "if agreement < ~80%, fix the rubric before training").

    python -m app.rlhf.hand_label export        # -> data/rlhf/hand_label.jsonl
    # ... Kai fills in "human_choice": "A" | "B" | "tie" for each row ...
    python -m app.rlhf.hand_label score

The export shuffles which side (A/B) is the judge's ``chosen`` per row and hides
it, so the label is blind. Sampling is stratified by (bucket, rejected_source) so
perturbations are well represented and the per-type detection rate is meaningful.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from app.rlhf._io import RLHF_DIR, read_jsonl, write_jsonl

_LABEL_FILE = RLHF_DIR / "hand_label.jsonl"


def _answer_text(completion: str) -> str:
    try:
        obj = json.loads(completion)
        prose = obj.get("prose", "")
        cites = "; ".join(
            f"{c['text']}  [{', '.join(c.get('cite', []))}]" for c in obj.get("claims", [])
        )
        return f"{prose}\n\nClaims:\n{cites}" if cites else prose
    except (ValueError, json.JSONDecodeError):
        return completion


def export(n: int, seed: int) -> None:
    pairs = read_jsonl(RLHF_DIR / "pairs_full.jsonl")
    if not pairs:
        raise SystemExit("run build_pairs first (data/rlhf/pairs_full.jsonl is empty)")
    rng = random.Random(seed)

    strata: dict[tuple, list] = defaultdict(list)
    for p in pairs:
        strata[(p["bucket"], p["rejected_source"].split(":")[0])].append(p)
    picked: list[dict] = []
    keys = list(strata)
    rng.shuffle(keys)
    while len(picked) < n and any(strata.values()):
        for k in keys:
            if strata[k]:
                picked.append(strata[k].pop(rng.randrange(len(strata[k]))))
            if len(picked) >= n:
                break

    rows = []
    for i, p in enumerate(picked):
        flip = rng.random() < 0.5
        a, b = (p["rejected"], p["chosen"]) if flip else (p["chosen"], p["rejected"])
        rows.append(
            {
                "row": i,
                "question": p["prompt"].split("Question: ")[-1].split("\n")[0],
                "answer_A": _answer_text(a),
                "answer_B": _answer_text(b),
                "human_choice": "",  # <- Kai fills: "A" | "B" | "tie"
                "_judge_choice": "B" if flip else "A",  # hidden until scoring
                "_rejected_source": p["rejected_source"],
                "_perturbation": p["rejected_perturbation"],
                "_bucket": p["bucket"],
            }
        )
    write_jsonl(_LABEL_FILE, rows)
    print(f"wrote {len(rows)} pairs -> {_LABEL_FILE}")
    print("Fill 'human_choice' (A/B/tie) for each row, then: python -m app.rlhf.hand_label score")


def score() -> None:
    rows = [r for r in read_jsonl(_LABEL_FILE) if r.get("human_choice")]
    if not rows:
        raise SystemExit(f"no labelled rows in {_LABEL_FILE}")

    agree = sum(1 for r in rows if r["human_choice"] == r["_judge_choice"])
    ties = sum(1 for r in rows if r["human_choice"] == "tie")
    decided = [r for r in rows if r["human_choice"] in ("A", "B")]
    strict = sum(1 for r in decided if r["human_choice"] == r["_judge_choice"])

    print(f"labelled: {len(rows)}  (ties: {ties})")
    print(f"agreement incl. ties-as-disagree: {agree / len(rows):.1%}")
    print(
        f"agreement on decided pairs only:  {strict / len(decided):.1%}  ({strict}/{len(decided)})"
    )

    per_pert: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        if r["_perturbation"]:
            # judge rejected the perturbed side; did the human agree it's worse?
            human_agrees = r["human_choice"] == r["_judge_choice"]
            per_pert[r["_perturbation"]].append(human_agrees)
    if per_pert:
        print("\nper-perturbation: human agrees the perturbed answer is worse")
        for k, v in sorted(per_pert.items()):
            print(f"  {k:22s} {sum(v)}/{len(v)}")

    by_bucket = Counter(r["_bucket"] for r in rows)
    print(f"\nby bucket: {dict(by_bucket)}")

    rate = strict / len(decided) if decided else 0
    if rate < 0.8:
        print(
            f"\n⚠  agreement {rate:.0%} < 80% — revise the judge rubric before training (brief §3)."
        )
    else:
        print(f"\n✓  agreement {rate:.0%} ≥ 80% — the judge is a usable proxy.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("-n", type=int, default=100)
    e.add_argument("--seed", type=int, default=0)
    sub.add_parser("score")
    args = p.parse_args(argv)
    if args.cmd == "export":
        export(args.n, args.seed)
    else:
        score()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
