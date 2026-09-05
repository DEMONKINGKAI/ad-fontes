"""Generation + verification eval (brief §2 baseline, §5 comparison).

Runs the full pipeline (retrieval + generator + NLI + numeric guard + scope gate
+ hosted fallback) over ``questions.jsonl`` and reports:

  * citation hit rate — claims whose citations are all in the retrieved set;
  * label distribution — supported / unsupported / contradicted / fabricated;
  * **unsupported + fabricated per 100 answers** — the headline metric;
  * contradicted + numeric-guard violations per 100;
  * answers with zero claims (a base-model failure mode);
  * decline correctness on the 10 negative controls (and false-decline rate on
    the answerable set);
  * hosted-fallback rate;
  * p50 / p95 latency (total and generation-only).

``--runs N`` repeats the whole set (temperature > 0 → real variance) and reports
mean ± spread. Results -> ``data/eval/generation-<ts>-<model>.{json,md}``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.api.schemas import AskRequest, ClaimLabel, ModelChoice
from app.config import get_settings
from app.eval.metrics import per_100
from app.eval.run_eval import _git_sha, load_questions


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[idx]


async def _answer_all(
    pipeline, questions: list[dict], model: ModelChoice, *, verbose: bool = False
) -> list[dict]:
    rows = []
    for n, q in enumerate(questions, start=1):
        t0 = time.monotonic()
        resp = await pipeline.answer_sync(
            AskRequest(question=q["question"], audience="auto", model=model)
        )
        wall = time.monotonic() - t0
        if verbose:
            labs = "/".join(c.verification.label.value[:4] for c in resp.claims) or (
                "decline" if resp.declined else "no-claims"
            )
            print(
                f"    [{n:>2}/{len(questions)}] {wall:5.1f}s {resp.meta.generator.value:14s} "
                f"{labs}  {q['id']}",
                flush=True,
            )
        claim_labels = [c.verification.label.value for c in resp.claims]
        numeric_flags = sum(1 for c in resp.claims if c.verification.numeric_flag)
        fab = sum(1 for c in resp.claims if c.verification.label is ClaimLabel.fabricated_citation)
        valid_cite_claims = sum(
            1 for c in resp.claims if c.verification.label is not ClaimLabel.fabricated_citation
        )
        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "answerable": q["answerable"],
                "question": q["question"],
                "declined": resp.declined,
                "generator": resp.meta.generator.value,
                "n_claims": len(resp.claims),
                "labels": claim_labels,
                "numeric_flags": numeric_flags,
                "fabricated": fab,
                "citation_hits": valid_cite_claims,
                "prose_chars": len(resp.prose),
                "prose": resp.prose,
                "unverified_prose": resp.unverified_prose,
                "latency_ms": resp.meta.latency_ms,
                "generation_ms": resp.meta.generation_ms,
                "verification_ms": resp.meta.verification_ms,
                "wall_s": round(wall, 2),
                "retrieved": resp.meta.retrieved_chunk_ids,
                "claims": [
                    {
                        "text": c.text,
                        "cite": c.cite,
                        "label": c.verification.label.value,
                        "entailment": c.verification.entailment,
                        "contradiction": c.verification.contradiction,
                        "numeric_flag": c.verification.numeric_flag,
                    }
                    for c in resp.claims
                ],
            }
        )
    return rows


def _summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    negatives = [r for r in rows if not r["answerable"]]
    all_labels = Counter(lbl for r in rows for lbl in r["labels"])
    n_claims = sum(r["n_claims"] for r in rows)
    n_answers = len(rows)

    unsupported = all_labels.get("unsupported", 0)
    fabricated = all_labels.get("fabricated_citation", 0)
    contradicted = all_labels.get("contradicted", 0)
    supported = all_labels.get("supported", 0)
    numeric_violations = sum(r["numeric_flags"] for r in rows)
    citation_hits = sum(r["citation_hits"] for r in rows)
    prose_unverified = sum(len(r.get("unverified_prose", [])) for r in rows)
    answers_with_unverified_prose = sum(1 for r in rows if r.get("unverified_prose"))

    lat = [r["latency_ms"] for r in rows]
    gen = [r["generation_ms"] for r in rows]

    return {
        "n_answers": n_answers,
        "n_claims": n_claims,
        "claims_per_answer": round(n_claims / n_answers, 2) if n_answers else 0.0,
        "answers_with_no_claims": sum(1 for r in rows if r["n_claims"] == 0 and not r["declined"]),
        "label_distribution": dict(all_labels),
        "supported_rate": (supported / n_claims) if n_claims else 0.0,
        "citation_hit_rate": (citation_hits / n_claims) if n_claims else 0.0,
        "unsupported_plus_fabricated_per_100": per_100(unsupported + fabricated, n_answers),
        "contradicted_per_100": per_100(contradicted, n_answers),
        "bad_claims_per_100": per_100(unsupported + fabricated + contradicted, n_answers),
        "numeric_violations_per_100": per_100(numeric_violations, n_answers),
        "prose_unverified_sentences_per_100": per_100(prose_unverified, n_answers),
        "answers_with_unverified_prose_pct": (
            answers_with_unverified_prose / n_answers if n_answers else 0.0
        ),
        "hosted_fallback_rate": sum(1 for r in rows if r["generator"] == "hosted-fallback")
        / n_answers,
        "decline_on_negatives": (
            sum(1 for r in negatives if r["declined"]) / len(negatives) if negatives else None
        ),
        "false_decline_on_answerable": (
            sum(1 for r in answerable if r["declined"]) / len(answerable) if answerable else None
        ),
        "latency_ms": {
            "p50": _percentile(lat, 50),
            "p95": _percentile(lat, 95),
            "mean": round(statistics.mean(lat)) if lat else 0,
        },
        "generation_ms": {
            "p50": _percentile(gen, 50),
            "p95": _percentile(gen, 95),
        },
        "mean_prose_chars": round(statistics.mean(r["prose_chars"] for r in rows)) if rows else 0,
    }


def run_generation(model: str, *, runs: int, limit: int | None) -> dict:
    from app.bootstrap import load_all

    settings = get_settings()
    components = load_all(settings, load_tuned=(model == "tuned"))
    from app.pipeline import Pipeline

    pipeline = Pipeline(settings, components)

    questions = load_questions()
    if limit:
        # keep the stratification: take the first `limit` after interleaving categories
        by_cat: dict[str, list] = {}
        for q in questions:
            by_cat.setdefault(q["category"], []).append(q)
        interleaved = [q for grp in zip(*by_cat.values(), strict=False) for q in grp]
        questions = interleaved[:limit]

    model_choice = ModelChoice(model)
    per_run: list[dict] = []
    all_rows: list[list[dict]] = []
    for i in range(runs):
        print(f"  run {i + 1}/{runs} ...", flush=True)
        rows = asyncio.run(_answer_all(pipeline, questions, model_choice, verbose=True))
        all_rows.append(rows)
        per_run.append(_summarise(rows))
        print(
            f"  run {i + 1}/{runs}: "
            f"unsup+fab/100={per_run[-1]['unsupported_plus_fabricated_per_100']:.1f} "
            f"p50={per_run[-1]['latency_ms']['p50']}ms "
            f"fallback={_pct(per_run[-1]['hosted_fallback_rate'])}"
        )

    def spread(key_path):
        vals = []
        for s in per_run:
            v = s
            for k in key_path:
                v = v[k]
            vals.append(v)
        return {"mean": round(statistics.mean(vals), 2), "min": min(vals), "max": max(vals)}

    return {
        "stage": "generation",
        "model": model,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "corpus_version": settings.corpus_version,
        "base_gguf": settings.base_gguf_file,
        "hosted_model": settings.hosted_model if settings.hf_token else None,
        "llm_temperature": settings.llm_temperature,
        "local_timeout_s": settings.local_timeout_s,
        "n_questions": len(questions),
        "runs": runs,
        "variance": {
            "unsupported_plus_fabricated_per_100": spread(["unsupported_plus_fabricated_per_100"]),
            "prose_unverified_sentences_per_100": spread(["prose_unverified_sentences_per_100"]),
            "citation_hit_rate": spread(["citation_hit_rate"]),
            "supported_rate": spread(["supported_rate"]),
            "latency_p50_ms": spread(["latency_ms", "p50"]),
            "latency_p95_ms": spread(["latency_ms", "p95"]),
            "hosted_fallback_rate": spread(["hosted_fallback_rate"]),
        },
        "runs_detail": per_run,
        "last_run_rows": all_rows[-1],
    }


def write_reports(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["generated"].replace(":", "").replace("-", "")
    base = out_dir / f"generation-{stamp}-{result['model']}"
    base.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    v = result["variance"]
    s0 = result["runs_detail"][0]
    lines = [
        f"# Generation eval — {result['model']} — {result['generated']}",
        "",
        f"- commit `{result['git_sha']}` · corpus `{result['corpus_version']}` · "
        f"model `{result['base_gguf']}` · hosted fallback `{result['hosted_model']}`",
        f"- {result['n_questions']} questions · {result['runs']} run(s) · "
        f"temp {result['llm_temperature']} · local timeout {result['local_timeout_s']}s",
        "",
        "## Headline (mean across runs)",
        "",
        "| metric | mean | min | max |",
        "|--|--|--|--|",
        f"| unsupported + fabricated / 100 answers | **{v['unsupported_plus_fabricated_per_100']['mean']}** | {v['unsupported_plus_fabricated_per_100']['min']} | {v['unsupported_plus_fabricated_per_100']['max']} |",
        f"| unverified prose sentences / 100 answers | **{v['prose_unverified_sentences_per_100']['mean']}** | {v['prose_unverified_sentences_per_100']['min']} | {v['prose_unverified_sentences_per_100']['max']} |",
        f"| citation hit rate | {v['citation_hit_rate']['mean']} | {v['citation_hit_rate']['min']} | {v['citation_hit_rate']['max']} |",
        f"| supported rate | {v['supported_rate']['mean']} | {v['supported_rate']['min']} | {v['supported_rate']['max']} |",
        f"| latency p50 (ms) | {v['latency_p50_ms']['mean']} | {v['latency_p50_ms']['min']} | {v['latency_p50_ms']['max']} |",
        f"| latency p95 (ms) | {v['latency_p95_ms']['mean']} | {v['latency_p95_ms']['min']} | {v['latency_p95_ms']['max']} |",
        f"| hosted-fallback rate | {v['hosted_fallback_rate']['mean']} | {v['hosted_fallback_rate']['min']} | {v['hosted_fallback_rate']['max']} |",
        "",
        "## Run 1 detail",
        "",
        f"- label distribution: {s0['label_distribution']}",
        f"- claims/answer {s0['claims_per_answer']} · answers with no claims {s0['answers_with_no_claims']}",
        f"- contradicted/100 {s0['contradicted_per_100']:.1f} · numeric violations/100 {s0['numeric_violations_per_100']:.1f}",
        f"- answers with ≥1 unverified prose sentence: {s0['answers_with_unverified_prose_pct']:.0%}",
        f"- decline on negatives {s0['decline_on_negatives']} · false-decline on answerable {s0['false_decline_on_answerable']}",
        f"- mean prose length {s0['mean_prose_chars']} chars",
        "",
        "## Sample overclaims caught (run 1)",
        "",
    ]
    bad = [
        (r, c)
        for r in result["last_run_rows"]
        for c in r["claims"]
        if c["label"] in ("unsupported", "contradicted", "fabricated_citation")
    ][:12]
    for r, c in bad:
        lines.append(f"- `{r['id']}` [{c['label']}] {c['text'][:140]}")

    prose_flags = [
        (r["id"], s) for r in result["last_run_rows"] for s in r.get("unverified_prose", [])
    ][:12]
    lines += ["", "## Unverified prose (asserted in prose, not backed by a claim or chunk)", ""]
    for qid, sent in prose_flags:
        lines.append(f"- `{qid}` {sent[:160]}")

    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".md")
