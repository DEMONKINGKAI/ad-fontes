"""Produce 3-5 answer candidates per question (brief §3).

For each question: retrieve once, then generate

  * ``base-t0.3`` / ``base-t0.9`` — the local base GGUF at two temperatures
  * ``hosted``                    — the larger hosted model
  * ``perturb:<type>``            — 1-2 deliberate degradations of the best
    faithful candidate (``app.rlhf.perturb``), each labelled by type

Every candidate is run through this repo's verification layers (claims + prose)
so ``judge.py`` and ``build_pairs.py`` have the labels. Resumable: appends to
``data/rlhf/candidates.jsonl`` and skips question ids already done.

    python -m app.rlhf.gen_candidates --limit 40          # pilot
    python -m app.rlhf.gen_candidates --holdout train     # full training set
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from collections import Counter
from pathlib import Path

from app.api.schemas import Audience, GeneratorKind
from app.config import get_settings
from app.generation.prompts import format_context
from app.guardrails.audience import resolve_audience
from app.rlhf._io import RLHF_DIR, append_jsonl, done_ids, iter_jsonl
from app.rlhf.perturb import PerturbationType, apply
from app.verification.verify import verify_answer, verify_prose

_PERTURBATIONS = list(PerturbationType)


def _verdict(prose: str, claim_dicts: list[dict], retrieved, nli) -> dict:
    from app.generation.schema import AnswerDraft, ClaimDraft

    draft = AnswerDraft(
        prose=prose or "(empty)",
        claims=[
            ClaimDraft(text=c["text"][:500], cite=list(c.get("cite") or [])[:6])
            for c in claim_dicts
            if c.get("text")
        ],
    )
    claims = verify_answer(draft, retrieved, nli)
    unverified = verify_prose(prose, claims, retrieved, nli)
    labels = Counter(c.verification.label.value for c in claims)
    return {
        "n_claims": len(claims),
        "labels": dict(labels),
        "supported": labels.get("supported", 0),
        "unsupported_plus_fabricated": labels.get("unsupported", 0)
        + labels.get("fabricated_citation", 0),
        "contradicted": labels.get("contradicted", 0),
        "fabricated": labels.get("fabricated_citation", 0),
        "numeric_flags": sum(1 for c in claims if c.verification.numeric_flag),
        "unverified_prose": unverified,
        "unverified_prose_count": len(unverified),
        "claims": [
            {"text": c.text, "cite": c.cite, "label": c.verification.label.value} for c in claims
        ],
    }


async def _gen_local(gen, question, ctx, audience) -> tuple[str, list[dict]]:
    ga = await gen.collect(question, ctx, audience, deadline=time.monotonic() + 90)
    return ga.draft.prose, [{"text": c.text, "cite": c.cite} for c in ga.draft.claims]


async def _gen_hosted(hosted, question, ctx, audience) -> tuple[str, list[dict]]:
    ga = await hosted.collect(question, ctx, audience)
    return ga.draft.prose, [{"text": c.text, "cite": c.cite} for c in ga.draft.claims]


def run(limit: int | None, holdout: str, seed: int) -> None:
    from app.bootstrap import load_generation, load_retrieval
    from app.generation.local_llm import load_local_generator
    from app.pipeline import PipelineComponents

    s = get_settings()
    comps = PipelineComponents()
    load_retrieval(comps, s)
    load_generation(comps, s, load_tuned=False)  # base + nli + hosted
    base_lo = comps.base_generator
    base_hi = load_local_generator(
        GeneratorKind.local_base,
        s.base_gguf_repo,
        s.base_gguf_file,
        n_ctx=s.llm_ctx,
        max_tokens=s.llm_max_tokens,
        temperature=0.9,
        grammar_mode=s.llm_grammar_mode,
    )
    hosted = comps.hosted_generator
    if hosted is None:
        raise SystemExit("HF_TOKEN required for the hosted candidate.")

    questions = list(iter_jsonl(RLHF_DIR / "questions.jsonl"))
    if holdout != "all":
        want = holdout == "eval"
        questions = [q for q in questions if q["holdout"] == want]
    out = RLHF_DIR / "candidates.jsonl"
    already = done_ids(out, key="question_id")
    todo = [q for q in questions if q["id"] not in already]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} questions to do ({len(already)} already) -> {out}")

    rng = random.Random(seed)
    for n, q in enumerate(todo, start=1):
        audience = resolve_audience(Audience.auto, q["question"])
        retrieved = comps.retriever.retrieve(q["question"], top_k=s.retrieval_top_k)
        ctx = format_context(retrieved)
        cands: list[dict] = []

        async def _all(qtext: str, ctx: str, audience) -> tuple:
            lo = await _gen_local(base_lo, qtext, ctx, audience)
            hi = await _gen_local(base_hi, qtext, ctx, audience)
            ho = await _gen_hosted(hosted, qtext, ctx, audience)
            return lo, hi, ho

        lo, hi, ho = asyncio.run(_all(q["question"], ctx, audience))
        for src, (prose, cl) in {"base-t0.3": lo, "base-t0.9": hi, "hosted": ho}.items():
            cands.append({"source": src, "perturbation": None, "prose": prose, "claim_seed": cl})

        # perturb the most-faithful real candidate
        scored = [(c, _verdict(c["prose"], c["claim_seed"], retrieved, comps.nli)) for c in cands]
        best = min(
            scored,
            key=lambda cv: (
                cv[1]["unsupported_plus_fabricated"] + 2 * cv[1]["contradicted"],
                -cv[1]["supported"],
            ),
        )[0]
        for ptype in rng.sample(_PERTURBATIONS, 2):
            p_prose, p_claims, applied = apply(
                ptype,
                best["prose"],
                best["claim_seed"],
                rng,
                project_slug=q.get("project_id") or "project",
            )
            if applied:
                cands.append(
                    {
                        "source": f"perturb:{ptype.value}",
                        "perturbation": ptype.value,
                        "prose": p_prose,
                        "claim_seed": p_claims,
                    }
                )

        row = {
            "question_id": q["id"],
            "question": q["question"],
            "bucket": q["bucket"],
            "persona": q["persona"],
            "answerable": q["answerable"],
            "holdout": q["holdout"],
            "audience": audience.value,
            "retrieved": [
                {
                    "chunk_id": r.chunk_id,
                    "title": r.title,
                    "text": r.text,
                    "source_path": r.source_path,
                }
                for r in retrieved
            ],
            "candidates": [],
        }
        for i, c in enumerate(cands):
            v = _verdict(c["prose"], c["claim_seed"], retrieved, comps.nli)
            row["candidates"].append(
                {
                    "candidate_id": f"{q['id']}-c{i}",
                    "source": c["source"],
                    "perturbation": c["perturbation"],
                    "prose": c["prose"],
                    "verification": v,
                }
            )
        append_jsonl(out, row)
        print(
            f"  [{n}/{len(todo)}] {q['id']:22s} {len(cands)} cands "
            f"({', '.join(c['source'] for c in cands)})",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--holdout", choices=["train", "eval", "all"], default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=RLHF_DIR / "candidates.jsonl")
    args = p.parse_args(argv)
    run(args.limit, args.holdout, args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
