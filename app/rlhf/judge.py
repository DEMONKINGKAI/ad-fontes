"""Score every candidate: this repo's verification + an LLM-judge rubric (brief §3).

Per candidate:
  * **verification** (already computed by ``gen_candidates``): contradicted /
    fabricated claims, unsupported+fabricated count, numeric flags, unverified
    prose sentences;
  * **LLM-judge rubric** (hosted model, JSON-constrained): faithfulness, humility
    (no overclaiming), audience fit, concision, third-person voice — each 1-5;
  * a **weighted scalar** in [0, 1];
  * a **hard veto** — a vetoed candidate can never be the 'chosen' side of a pair.

Resumable: appends to ``data/rlhf/judged.jsonl`` keyed by candidate_id.

    python -m app.rlhf.judge --limit 40
"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.config import get_settings
from app.rlhf._io import RLHF_DIR, append_jsonl, done_ids, iter_jsonl

RUBRIC_WEIGHTS = {
    "faithfulness": 0.40,
    "humility": 0.25,
    "audience_fit": 0.12,
    "concision": 0.10,
    "third_person_voice": 0.13,
}

_JUDGE_SYSTEM = """You are a strict evaluator of a portfolio assistant's answers about Archit "Kai" Sharma.
You are given the source passages the assistant was allowed to use, the question, and the answer.
Rate the answer 1-5 on each axis (5 = best):

- faithfulness: every statement is supported by the passages; no invented facts, dates, URLs, or deployments. Check every NUMBER and PERCENTAGE against the passages digit for digit — "over 99%" when the passage says "97.2%" is a faithfulness failure (score 1-2).
- humility: does not overclaim — keeps the source's verbs (not "led" where it says "contributed"), does not round numbers up or say "over N" for a specific N, does not imply scope the passages don't state, does not drop a limitation the passage records.
- audience_fit: right depth and tone for the audience.
- concision: no padding, no unrequested biography.
- third_person_voice: talks about Kai in the third person; never writes as Kai ("I built…").

Respond with ONE JSON object: {"faithfulness": n, "humility": n, "audience_fit": n, "concision": n, "third_person_voice": n, "worst_problem": "<=12 words"}"""


def _weighted(scores: dict) -> float:
    total = sum(RUBRIC_WEIGHTS[k] * (scores.get(k, 1) - 1) / 4 for k in RUBRIC_WEIGHTS)
    return round(total, 4)


def _veto(cand: dict, scores: dict) -> str | None:
    v = cand["verification"]
    if v["contradicted"] > 0:
        return "contradicted_claim"
    if v["fabricated"] > 0:
        return "fabricated_citation"
    # >=3 (was 2): Phase 2.6 made verify_prose stricter and noisier — DeBERTa-base
    # false-positives on reworded-but-true prose mean 1-2 flags is common on a
    # genuinely fine answer. build_pairs._combined still penalises every flag.
    if v["unverified_prose_count"] >= 3:
        return "prose_hallucination"
    if scores.get("faithfulness", 5) <= 2:
        return "judge_faithfulness"
    if scores.get("third_person_voice", 5) <= 2:
        return "first_person"
    return None


async def _judge_one(client_call, question: str, audience: str, passages: str, answer: str) -> dict:
    prompt = f"Audience: {audience}\n\nPassages:\n{passages}\n\nQuestion: {question}\n\nAnswer:\n{answer}"
    raw = await client_call(_JUDGE_SYSTEM, prompt)
    from app.generation.base import parse_answer  # tolerant JSON extraction

    try:
        obj = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        try:
            obj = parse_answer(raw).model_dump()
        except Exception:
            obj = {}
    out = {}
    for k in RUBRIC_WEIGHTS:
        try:
            out[k] = max(1, min(5, int(round(float(obj.get(k, 3))))))
        except (TypeError, ValueError):
            out[k] = 3
    out["worst_problem"] = str(obj.get("worst_problem", ""))[:120]
    return out


def run(limit: int | None, seed: int) -> None:
    from huggingface_hub import InferenceClient

    s = get_settings()
    if not s.hf_token:
        raise SystemExit("HF_TOKEN required for the LLM judge.")
    client = InferenceClient(
        model=s.hosted_model,
        token=s.hf_token,
        provider=None if s.hosted_provider == "auto" else s.hosted_provider,
    )

    async def call(system: str, user: str) -> str:
        loop = asyncio.get_running_loop()

        def _do():
            r = client.chat_completion(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return r.choices[0].message.content or "{}"

        return await loop.run_in_executor(None, _do)

    src = RLHF_DIR / "candidates.jsonl"
    out = RLHF_DIR / "judged.jsonl"
    done = done_ids(out, key="candidate_id")

    rows = list(iter_jsonl(src))
    pending = [(q, c) for q in rows for c in q["candidates"] if c["candidate_id"] not in done]
    if limit:
        pending = pending[:limit]
    print(f"{len(pending)} candidates to judge ({len(done)} done) -> {out}")

    async def _main():
        for n, (q, c) in enumerate(pending, start=1):
            passages = "\n\n".join(
                f"[{i + 1}] {r['title']}\n{r['text']}" for i, r in enumerate(q["retrieved"])
            )
            scores = await _judge_one(call, q["question"], q["audience"], passages, c["prose"])
            veto = _veto(c, scores)
            append_jsonl(
                out,
                {
                    "candidate_id": c["candidate_id"],
                    "question_id": q["question_id"],
                    "bucket": q["bucket"],
                    "holdout": q["holdout"],
                    "answerable": q["answerable"],
                    "source": c["source"],
                    "perturbation": c["perturbation"],
                    "rubric": scores,
                    "verification": {
                        k: c["verification"][k]
                        for k in (
                            "supported",
                            "unsupported_plus_fabricated",
                            "contradicted",
                            "fabricated",
                            "numeric_flags",
                            "unverified_prose_count",
                            "n_claims",
                        )
                    },
                    "judge_scalar": _weighted(scores),
                    "veto": veto,
                    "prose_len": len(c["prose"]),
                },
            )
            if n % 20 == 0 or n == len(pending):
                print(f"  {n}/{len(pending)}", flush=True)

    asyncio.run(_main())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run(args.limit, args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
