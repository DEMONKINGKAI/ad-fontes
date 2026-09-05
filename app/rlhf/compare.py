"""Phase 5 — base vs. tuned comparison (brief §5).

Runs two generator "arms" over a held-out set (RLHF holdout questions +
`app/eval/adversarial.jsonl` + the negative controls from
`app/eval/questions.jsonl`) through the full pipeline, then reports:

  * unsupported+fabricated / 100, contradicted / 100, numeric violations / 100,
    unverified-prose sentences / 100, supported rate;
  * decline correctness on the unanswerable questions (and false-decline rate);
  * mean prose length; latency p50 / p95;
  * **LLM judge win rate** (blind A/B, faithful-and-humble);
  * **bootstrap 95% CIs** on the tuned−base deltas for the headline metrics;
  * a base-vs-tuned bar chart (PNG).

Before the tuned GGUF exists, use `--a base --b hosted` as a stand-in — it
exercises every code path and produces a real comparison.

    python -m app.rlhf.compare --a base --b tuned            # the real run
    python -m app.rlhf.compare --a base --b hosted --limit 20   # dry run now
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.api.schemas import AskRequest, Audience
from app.config import get_settings
from app.eval.generation import answer_row
from app.eval.metrics import per_100
from app.eval.run_eval import _git_sha, load_questions

_EVAL = Path("app/eval")
_OUT = Path("data/eval")

_JUDGE_AB = """You compare two answers from a portfolio assistant about Archit "Kai" Sharma.
Given the source passages, the question, and answers A and B, pick the one that is more
FAITHFUL to the passages (no invented facts, numbers, dates, URLs, deployments; numbers
match digit-for-digit) and more HUMBLE (keeps the source's verbs and stated limitations,
no overclaiming, third person). A shorter answer that is fully supported beats a longer
one that reaches. Respond with ONE JSON object:
{"winner": "A" | "B" | "tie", "why": "<=15 words"}"""


# --------------------------------------------------------------------------- #
# question set
# --------------------------------------------------------------------------- #


def _eval_questions() -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()

    rlhf_q = Path("data/rlhf/questions.jsonl")
    if rlhf_q.exists():
        for q in load_questions(rlhf_q):
            if q.get("holdout"):
                out.append(
                    {
                        "id": q["id"],
                        "category": q["bucket"],
                        "question": q["question"],
                        "answerable": q["answerable"],
                        "set": "rlhf_holdout",
                    }
                )
                seen.add(q["question"].lower())

    for q in load_questions(_EVAL / "adversarial.jsonl"):
        out.append({**q, "set": "adversarial"})
        seen.add(q["question"].lower())

    for q in load_questions(_EVAL / "questions.jsonl"):
        if q["category"] == "negative_control" and q["question"].lower() not in seen:
            out.append({**q, "set": "negative_control"})

    return out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    labels = Counter(x for r in rows for x in r["labels"])
    n_claims = sum(r["n_claims"] for r in rows)
    unanswerable = [r for r in rows if not r["answerable"]]
    answerable = [r for r in rows if r["answerable"]]
    lat = [r["latency_ms"] for r in rows]
    return {
        "n": n,
        "n_claims": n_claims,
        "unsupported_plus_fabricated_per_100": round(
            per_100(labels.get("unsupported", 0) + labels.get("fabricated_citation", 0), n), 1
        ),
        "contradicted_per_100": round(per_100(labels.get("contradicted", 0), n), 1),
        "numeric_violations_per_100": round(per_100(sum(r["numeric_flags"] for r in rows), n), 1),
        "unverified_prose_per_100": round(
            per_100(sum(len(r["unverified_prose"]) for r in rows), n), 1
        ),
        "supported_rate": round((labels.get("supported", 0) / n_claims) if n_claims else 0.0, 3),
        "decline_on_unanswerable": (
            sum(r["declined"] for r in unanswerable) / len(unanswerable) if unanswerable else None
        ),
        "false_decline_on_answerable": (
            sum(r["declined"] for r in answerable) / len(answerable) if answerable else None
        ),
        "mean_prose_chars": round(statistics.mean(r["prose_chars"] for r in rows)) if rows else 0,
        "latency_p50_ms": _pct(lat, 50),
        "latency_p95_ms": _pct(lat, 95),
        "generator_mix": dict(Counter(r["generator"] for r in rows)),
    }


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def _metric(rows: list[dict], key: str) -> float:
    """Recompute one scalar over an arbitrary row subset (for bootstrapping)."""
    n = len(rows) or 1
    labels = Counter(x for r in rows for x in r["labels"])
    if key == "unsupported_plus_fabricated_per_100":
        return per_100(labels.get("unsupported", 0) + labels.get("fabricated_citation", 0), n)
    if key == "contradicted_per_100":
        return per_100(labels.get("contradicted", 0), n)
    if key == "unverified_prose_per_100":
        return per_100(sum(len(r["unverified_prose"]) for r in rows), n)
    if key == "numeric_violations_per_100":
        return per_100(sum(r["numeric_flags"] for r in rows), n)
    if key == "mean_prose_chars":
        return statistics.mean(r["prose_chars"] for r in rows) if rows else 0.0
    if key == "decline_on_unanswerable":
        u = [r for r in rows if not r["answerable"]]
        return (sum(r["declined"] for r in u) / len(u)) if u else 0.0
    raise KeyError(key)


def _bootstrap_delta(rows_a: list[dict], rows_b: list[dict], key: str, n_boot: int, seed: int):
    """95% CI on metric(B) − metric(A), resampling the *paired* question index."""
    rng = random.Random(seed)
    m = len(rows_a)
    by_id_a = {r["id"]: r for r in rows_a}
    by_id_b = {r["id"]: r for r in rows_b}
    ids = [r["id"] for r in rows_a if r["id"] in by_id_b]
    point = _metric([by_id_b[i] for i in ids], key) - _metric([by_id_a[i] for i in ids], key)
    deltas = []
    for _ in range(n_boot):
        sample = [ids[rng.randrange(len(ids))] for _ in range(m)]
        da = _metric([by_id_a[i] for i in sample], key)
        db = _metric([by_id_b[i] for i in sample], key)
        deltas.append(db - da)
    deltas.sort()
    return {
        "delta": round(point, 2),
        "ci95": [round(deltas[int(0.025 * n_boot)], 2), round(deltas[int(0.975 * n_boot)], 2)],
    }


# --------------------------------------------------------------------------- #
# LLM judge win rate
# --------------------------------------------------------------------------- #


def _call_judge(client, user: str) -> str:
    r = client.chat_completion(
        [{"role": "system", "content": _JUDGE_AB}, {"role": "user", "content": user}],
        max_tokens=120,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content or "{}"


async def _win_rate(questions, cand_rows_a, rows_b, retrieved_by_id, seed) -> dict:
    from huggingface_hub import InferenceClient

    s = get_settings()
    if not s.hf_token:
        return {"skipped": "no HF_TOKEN"}
    client = InferenceClient(
        model=s.hosted_model,
        token=s.hf_token,
        provider=None if s.hosted_provider == "auto" else s.hosted_provider,
    )
    rng = random.Random(seed)
    a_by = {r["id"]: r for r in cand_rows_a}
    b_by = {r["id"]: r for r in rows_b}
    loop = asyncio.get_running_loop()
    tally = Counter()

    for q in questions:
        ra, rb = a_by.get(q["id"]), b_by.get(q["id"])
        if not ra or not rb:
            continue
        passages = "\n\n".join(
            f"[{i + 1}] {c['title']}\n{c['text']}"
            for i, c in enumerate(retrieved_by_id.get(q["id"], []))
        )
        flip = rng.random() < 0.5
        first, second = (rb, ra) if flip else (ra, rb)
        user = (
            f"Passages:\n{passages}\n\nQuestion: {q['question']}\n\n"
            f"Answer A:\n{first['prose']}\n\nAnswer B:\n{second['prose']}"
        )
        raw = await loop.run_in_executor(None, _call_judge, client, user)
        try:
            w = json.loads(raw[raw.index("{") : raw.rindex("}") + 1]).get("winner", "tie")
        except (ValueError, json.JSONDecodeError):
            w = "tie"
        if w == "tie":
            tally["tie"] += 1
        else:
            winner_is_b = (w == "A") == flip
            tally["b" if winner_is_b else "a"] += 1

    decided = tally["a"] + tally["b"]
    return {
        "a_wins": tally["a"],
        "b_wins": tally["b"],
        "ties": tally["tie"],
        "b_win_rate_decided": round(tally["b"] / decided, 3) if decided else None,
    }


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #


def _plot(name_a, name_b, sa, sb, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    keys = [
        ("unsupported_plus_fabricated_per_100", "unsup+fab /100"),
        ("contradicted_per_100", "contra /100"),
        ("unverified_prose_per_100", "prose /100"),
        ("numeric_violations_per_100", "numeric /100"),
    ]
    labels = [lbl for _, lbl in keys]
    va = [sa[k] for k, _ in keys]
    vb = [sb[k] for k, _ in keys]
    x = range(len(keys))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - 0.2 for i in x], va, width=0.4, label=name_a, color="#94a3b8")
    ax.bar([i + 0.2 for i in x], vb, width=0.4, label=name_b, color="#2563eb")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("per 100 answers")
    ax.set_title(f"ad fontes — faithfulness: {name_a} vs {name_b}")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def _resolve_generator(components, arm: str):
    from app.generation.hosted_llm import HostedGenerator

    if arm == "base":
        return components.base_generator
    if arm == "tuned":
        if components.tuned_generator is None:
            raise SystemExit(
                "tuned generator not loaded — set AD_FONTES_TUNED_GGUF_PATH or upload the GGUF, "
                "or use --b hosted for a stand-in run."
            )
        return components.tuned_generator
    if arm == "hosted":
        if not isinstance(components.hosted_generator, HostedGenerator):
            raise SystemExit("hosted generator not available (needs HF_TOKEN).")
        return components.hosted_generator
    raise SystemExit(f"unknown arm {arm!r}")


async def _run_arm(pipeline, generator, questions, tag: str) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, start=1):
        t0 = time.monotonic()
        resp = await pipeline.answer_sync(
            AskRequest(question=q["question"], audience=Audience.auto), force_generator=generator
        )
        rows.append({**answer_row(q, resp, time.monotonic() - t0), "set": q.get("set", "")})
        if i % 10 == 0 or i == len(questions):
            print(f"  {tag}: {i}/{len(questions)}", flush=True)
    return rows


def run(a: str, b: str, limit: int | None, n_boot: int, seed: int) -> dict:
    from app.bootstrap import load_all
    from app.pipeline import Pipeline

    s = get_settings()
    components = load_all(s, load_tuned=(a == "tuned" or b == "tuned"))
    pipeline = Pipeline(s, components)
    gen_a = _resolve_generator(components, a)
    gen_b = _resolve_generator(components, b)

    questions = _eval_questions()
    if limit:
        questions = questions[:limit]
    retrieved_by_id = {
        q["id"]: [
            {"title": r.title, "text": r.text}
            for r in components.retriever.retrieve(q["question"], top_k=s.retrieval_top_k)
        ]
        for q in questions
    }
    print(f"{len(questions)} questions · arm A = {a} · arm B = {b}")

    rows_a = asyncio.run(_run_arm(pipeline, gen_a, questions, a))
    rows_b = asyncio.run(_run_arm(pipeline, gen_b, questions, b))
    sa, sb = _summary(rows_a), _summary(rows_b)

    boot = {
        k: _bootstrap_delta(rows_a, rows_b, k, n_boot, seed)
        for k in (
            "unsupported_plus_fabricated_per_100",
            "contradicted_per_100",
            "unverified_prose_per_100",
            "numeric_violations_per_100",
            "mean_prose_chars",
            "decline_on_unanswerable",
        )
    }
    win = asyncio.run(_win_rate(questions, rows_a, rows_b, retrieved_by_id, seed))

    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%S")
    png = _plot(a, b, sa, sb, _OUT / f"compare-{stamp}-{a}-vs-{b}.png")

    result = {
        "generated": now.isoformat(timespec="seconds"),
        "stamp": stamp,
        "git_sha": _git_sha(),
        "corpus_version": s.corpus_version,
        "arm_a": a,
        "arm_b": b,
        "n_questions": len(questions),
        "bootstrap_n": n_boot,
        "summary_a": sa,
        "summary_b": sb,
        "deltas_b_minus_a": boot,
        "judge_win_rate": win,
        "plot": str(png) if png else None,
        "rows_a": rows_a,
        "rows_b": rows_b,
    }
    _write(result)
    return result


def _write(r: dict) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    base = _OUT / f"compare-{r['stamp']}-{r['arm_a']}-vs-{r['arm_b']}"
    base.with_suffix(".json").write_text(json.dumps(r, indent=2), encoding="utf-8")

    sa, sb, d = r["summary_a"], r["summary_b"], r["deltas_b_minus_a"]

    def line(key, label, better="lower"):
        bd = d.get(key, {})
        arrow = ""
        if bd:
            lo, hi = bd["ci95"]
            sig = (hi < 0) if better == "lower" else (lo > 0)
            arrow = "  ✓" if sig and ((bd["delta"] < 0) == (better == "lower")) else ""
        return (
            f"| {label} | {sa.get(key)} | {sb.get(key)} | "
            f"{bd.get('delta', '')} [{', '.join(map(str, bd.get('ci95', [])))}]{arrow} |"
        )

    lines = [
        f"# Compare — {r['arm_a']} vs {r['arm_b']} — {r['generated']}",
        "",
        f"- commit `{r['git_sha']}` · corpus `{r['corpus_version']}` · "
        f"{r['n_questions']} held-out/adversarial/negative questions · {r['bootstrap_n']} bootstrap resamples",
        f"- generator mix — A: {sa['generator_mix']} · B: {sb['generator_mix']}",
        "",
        "## Headline (lower is better; ✓ = 95% CI excludes 0 in the good direction)",
        "",
        f"| metric | {r['arm_a']} | {r['arm_b']} | Δ (B−A) [95% CI] |",
        "|--|--|--|--|",
        line("unsupported_plus_fabricated_per_100", "**unsupported + fabricated / 100**"),
        line("contradicted_per_100", "contradicted / 100"),
        line("unverified_prose_per_100", "unverified prose / 100"),
        line("numeric_violations_per_100", "numeric violations / 100"),
        line("mean_prose_chars", "mean prose chars", better="lower"),
        "",
        "## Behaviour",
        "",
        f"| metric | {r['arm_a']} | {r['arm_b']} |",
        "|--|--|--|",
        f"| supported rate | {sa['supported_rate']:.2f} | {sb['supported_rate']:.2f} |",
        f"| decline on unanswerable | {sa['decline_on_unanswerable']} | {sb['decline_on_unanswerable']} |",
        f"| false-decline on answerable | {sa['false_decline_on_answerable']} | {sb['false_decline_on_answerable']} |",
        f"| latency p50 / p95 ms | {sa['latency_p50_ms']} / {sa['latency_p95_ms']} "
        f"| {sb['latency_p50_ms']} / {sb['latency_p95_ms']} |",
        "",
        "## LLM judge win rate (blind A/B, faithful-and-humble)",
        "",
        f"`{r['arm_b']}` wins **{r['judge_win_rate'].get('b_win_rate_decided')}** of decided pairs "
        f"({r['judge_win_rate']})"
        if "b_win_rate_decided" in r["judge_win_rate"]
        else str(r["judge_win_rate"]),
    ]
    if r["plot"]:
        lines += ["", f"![base vs tuned]({Path(r['plot']).name})"]
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {base.with_suffix('.md')}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", default="base", choices=["base", "tuned", "hosted"])
    p.add_argument("--b", default="tuned", choices=["base", "tuned", "hosted"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    r = run(args.a, args.b, args.limit, args.bootstrap, args.seed)
    d = r["deltas_b_minus_a"]["unsupported_plus_fabricated_per_100"]
    line = (
        f"\n{args.b} vs {args.a}: unsup+fab/100  delta {d['delta']}  CI {d['ci95']}  "
        f"| judge win {r['judge_win_rate'].get('b_win_rate_decided')}"
    )
    print(line.encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
