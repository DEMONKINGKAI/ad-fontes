"""Assemble (chosen, rejected) preference pairs in TRL DPO format (brief §3).

For each question, rank its judged candidates by a combined score
(judge scalar − verification penalties); the top non-vetoed candidate is
``chosen`` and it is paired with every candidate it beats by ``--margin``.

**Length control** (brief): after building all pairs we measure whether "chosen
is shorter" predicts preference, drop the pairs most responsible for any bias,
and report the residual point-biserial correlation.

    python -m app.rlhf.build_pairs                 # -> data/rlhf/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from app.config import get_settings
from app.generation.prompts import format_context, system_prompt, user_prompt
from app.retrieval.retriever import RetrievedChunk
from app.rlhf._io import RLHF_DIR, iter_jsonl, write_jsonl


def _combined(j: dict) -> float:
    v = j["verification"]
    penalty = (
        0.10 * v["unsupported_plus_fabricated"]
        + 0.35 * v["contradicted"]
        + 0.30 * v["fabricated"]
        + 0.20 * v["unverified_prose_count"]
    )
    return j["judge_scalar"] - min(penalty, 1.0)


def _prompt_for(cand_row: dict) -> str:
    retrieved = [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            text=r["text"],
            title=r["title"],
            section="",
            doc_type="",
            source_path=r["source_path"],
            project_id=None,
            repo_url=None,
            score=0.0,
            base_score=0.0,
        )
        for r in cand_row["retrieved"]
    ]
    ctx = format_context(retrieved)
    from app.api.schemas import Audience

    aud = Audience(cand_row["audience"])
    return system_prompt(aud) + "\n\n" + user_prompt(cand_row["question"], ctx)


def _completion(cand: dict) -> str:
    claims = [
        {"text": c["text"], "cite": c["cite"]} for c in cand["verification"].get("claims", [])
    ]
    return json.dumps({"prose": cand["prose"], "claims": claims}, ensure_ascii=False)


def build(margin: float, seed: int) -> dict:
    cand_rows = {r["question_id"]: r for r in iter_jsonl(RLHF_DIR / "candidates.jsonl")}
    judged: dict[str, list[dict]] = {}
    for j in iter_jsonl(RLHF_DIR / "judged.jsonl"):
        judged.setdefault(j["question_id"], []).append(j)

    pairs: list[dict] = []
    for qid, js in judged.items():
        cand_row = cand_rows.get(qid)
        if not cand_row or len(js) < 2:
            continue
        by_id = {c["candidate_id"]: c for c in cand_row["candidates"]}
        ranked = sorted(js, key=_combined, reverse=True)
        # 'chosen' must be a real generation that passes the veto — a perturbed
        # candidate is degraded *by construction* and can only be 'rejected'.
        top = next((j for j in ranked if j["veto"] is None and j["perturbation"] is None), None)
        if top is None:
            continue
        chosen_cand = by_id[top["candidate_id"]]
        prompt = _prompt_for(cand_row)
        chosen_text = _completion(chosen_cand)
        for j in ranked:
            if j["candidate_id"] == top["candidate_id"]:
                continue
            if _combined(top) - _combined(j) < margin:
                continue
            rej_cand = by_id[j["candidate_id"]]
            pairs.append(
                {
                    "question_id": qid,
                    "bucket": cand_row["bucket"],
                    "holdout": cand_row["holdout"],
                    "prompt": prompt,
                    "chosen": chosen_text,
                    "rejected": _completion(rej_cand),
                    "chosen_source": top["source"],
                    "rejected_source": j["source"],
                    "rejected_perturbation": j["perturbation"],
                    "score_margin": round(_combined(top) - _combined(j), 3),
                    "len_delta": len(rej_cand["prose"]) - len(chosen_cand["prose"]),
                }
            )

    # --- length-bias control ---------------------------------------
    def biserial(rows: list[dict]) -> float:
        # chosen-is-shorter (1/0) vs a constant "chosen preferred" -> use len_delta
        deltas = [p["len_delta"] for p in rows]
        if len(deltas) < 3 or statistics.pstdev(deltas) == 0:
            return 0.0
        mean = statistics.mean(deltas)
        return round(mean / statistics.pstdev(deltas), 3)

    for i, p in enumerate(pairs):
        p["_i"] = i
    before = biserial(pairs)
    # Balance the sign of len_delta so "chosen is shorter" is not predictive of
    # preference: down-sample the majority sign toward the minority, dropping the
    # smallest-margin pairs first (least informative).
    longer = sorted((p for p in pairs if p["len_delta"] > 0), key=lambda p: p["score_margin"])
    shorter = sorted((p for p in pairs if p["len_delta"] <= 0), key=lambda p: p["score_margin"])
    keep_n = min(len(longer), len(shorter))
    slack = int(keep_n * 0.35) + 1  # allow a modest imbalance to keep volume
    keep_ids = {p["_i"] for p in longer[-(keep_n + slack) :] + shorter[-(keep_n + slack) :]}
    kept = [p for p in pairs if p["_i"] in keep_ids] or pairs
    for p in pairs:
        p.pop("_i", None)
    after = biserial(kept)

    write_jsonl(
        RLHF_DIR / "pairs.jsonl",
        [
            {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
            for p in kept
            if not p["holdout"]
        ],
    )
    write_jsonl(RLHF_DIR / "pairs_full.jsonl", kept)

    train = [p for p in kept if not p["holdout"]]
    return {
        "pairs_built": len(pairs),
        "pairs_kept": len(kept),
        "train_pairs": len(train),
        "holdout_pairs": len(kept) - len(train),
        "length_bias_len_delta_z_before": before,
        "length_bias_len_delta_z_after": after,
        "rejected_source_mix": dict(Counter(p["rejected_source"] for p in kept)),
        "chosen_source_mix": dict(Counter(p["chosen_source"] for p in kept)),
        "mean_score_margin": round(statistics.mean(p["score_margin"] for p in kept), 3)
        if kept
        else 0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--margin", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    _ = get_settings()
    stats = build(args.margin, args.seed)
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {RLHF_DIR / 'pairs.jsonl'} (TRL DPO format) + pairs_full.jsonl (with metadata)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
