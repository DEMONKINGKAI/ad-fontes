"""Drive the pipeline over ``questions.jsonl`` and emit a metrics report.

    python -m app.eval.run_eval --stage retrieval               # Phase 1
    python -m app.eval.run_eval --stage retrieval --no-boosts    # ablation
    python -m app.eval.run_eval --stage generation --model base  # Phase 2
    python -m app.eval.run_eval --stage compare                  # Phase 5

Phase 1 implements ``--stage retrieval``: it builds the real retriever (nomic
embedder + the persisted Chroma index), runs every question, and reports hit@k at
chunk and file granularity plus MRR, broken down by category. Negative controls
are reported separately (top-1 similarity, not hit@k). Results are written to
``data/eval/retrieval-<timestamp>.{json,md}`` and summarised to stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from app.config import get_settings
from app.eval import metrics

_QUESTIONS = Path(__file__).with_name("questions.jsonl")
_K_VALUES = (1, 3, 5, 6, 10)


def load_questions(path: Path = _QUESTIONS) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def run_retrieval(*, enable_boosts: bool, top_k: int) -> dict:
    from app.retrieval.retriever import RetrievalConfig, build_retriever

    settings = get_settings()
    retriever = build_retriever(settings)
    retriever.config = RetrievalConfig(
        top_k=max(_K_VALUES),
        enable_boosts=enable_boosts,
        candidate_pool=RetrievalConfig().candidate_pool,
    )
    health = retriever.index_health()
    if health["indexed"] == 0:
        raise SystemExit(
            "Chroma index is empty. Build it first: python -m app.ingestion.cli --rebuild"
        )

    questions = load_questions()
    per_question: list[dict] = []
    for q in questions:
        retrieved = retriever.retrieve(q["question"], top_k=max(_K_VALUES))
        ids = [r.chunk_id for r in retrieved]
        files = [r.source_path for r in retrieved]
        gold_ids = q.get("gold_chunk_ids", [])
        gold_files = q.get("gold_files", [])
        per_question.append(
            {
                "id": q["id"],
                "category": q["category"],
                "answerable": q["answerable"],
                "question": q["question"],
                "retrieved": ids,
                "retrieved_files": files,
                "scores": [round(r.score, 4) for r in retrieved],
                "boosts": [list(r.boosts) for r in retrieved],
                "gold_chunk_ids": gold_ids,
                "gold_files": gold_files,
                "chunk_rank": metrics.first_rank(ids, gold_ids),
                "file_rank": metrics.first_rank(files, gold_files),
                "chunk_hits": {k: metrics.hit_at_k(ids, gold_ids, k) for k in _K_VALUES},
                "file_hits": {k: metrics.hit_at_k(files, gold_files, k) for k in _K_VALUES},
                "top_score": round(retrieved[0].score, 4) if retrieved else None,
            }
        )

    answerable = [r for r in per_question if r["answerable"]]
    negatives = [r for r in per_question if not r["answerable"]]

    summary = {
        "chunk_hit_at_k": {
            k: metrics.rate([r["chunk_hits"][k] for r in answerable]) for k in _K_VALUES
        },
        "file_hit_at_k": {
            k: metrics.rate([r["file_hits"][k] for r in answerable]) for k in _K_VALUES
        },
        "chunk_mrr": metrics.mrr(
            [metrics.reciprocal_rank(r["retrieved"], r["gold_chunk_ids"]) for r in answerable]
        ),
        "file_mrr": metrics.mrr(
            [metrics.reciprocal_rank(r["retrieved_files"], r["gold_files"]) for r in answerable]
        ),
        "n_answerable": len(answerable),
        "n_negative": len(negatives),
        "negative_top_score_mean": (
            round(mean(r["top_score"] for r in negatives), 4) if negatives else None
        ),
        "answerable_top_score_mean": (
            round(mean(r["top_score"] for r in answerable), 4) if answerable else None
        ),
    }

    by_cat: dict[str, dict] = {}
    cats = defaultdict(list)
    for r in answerable:
        cats[r["category"]].append(r)
    for cat, rows in sorted(cats.items()):
        by_cat[cat] = {
            "n": len(rows),
            "chunk_hit_at_6": metrics.rate([x["chunk_hits"][6] for x in rows]),
            "file_hit_at_6": metrics.rate([x["file_hits"][6] for x in rows]),
        }

    return {
        "stage": "retrieval",
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "corpus_version": settings.corpus_version,
        "embed_model": settings.embed_model,
        "index_size": health["indexed"],
        "enable_boosts": enable_boosts,
        "top_k": top_k,
        "summary": summary,
        "by_category": by_cat,
        "per_question": per_question,
    }


def _write_reports(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["generated"].replace(":", "").replace("-", "")
    tag = "boosts" if result["enable_boosts"] else "noboosts"
    base = out_dir / f"retrieval-{stamp}-{tag}"
    base.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    s = result["summary"]
    lines = [
        f"# Retrieval eval — {result['generated']}",
        "",
        f"- commit `{result['git_sha']}` · corpus `{result['corpus_version']}` · "
        f"index {result['index_size']} chunks · embedder `{result['embed_model']}`",
        f"- boosts: **{'on' if result['enable_boosts'] else 'off'}** · "
        f"{s['n_answerable']} answerable + {s['n_negative']} negative controls",
        "",
        "## Overall (answerable questions)",
        "",
        "| k | chunk hit@k | file hit@k |",
        "|--|--|--|",
    ]
    for k in _K_VALUES:
        lines.append(f"| {k} | {_pct(s['chunk_hit_at_k'][k])} | {_pct(s['file_hit_at_k'][k])} |")
    lines += [
        "",
        f"chunk MRR **{s['chunk_mrr']:.3f}** · file MRR **{s['file_mrr']:.3f}**",
        "",
        f"mean top-1 similarity: answerable **{s['answerable_top_score_mean']}** vs "
        f"negative-control **{s['negative_top_score_mean']}**",
        "",
        "## By category (hit@6)",
        "",
        "| category | n | chunk | file |",
        "|--|--|--|--|",
    ]
    for cat, c in result["by_category"].items():
        lines.append(
            f"| {cat} | {c['n']} | {_pct(c['chunk_hit_at_6'])} | {_pct(c['file_hit_at_6'])} |"
        )

    misses = [r for r in result["per_question"] if r["answerable"] and not r["chunk_hits"][6]]
    lines += ["", f"## Chunk misses @6 ({len(misses)})", ""]
    for r in misses:
        lines.append(
            f"- `{r['id']}` — {r['question']}  \n"
            f"  gold {r['gold_chunk_ids']} · got {r['retrieved'][:6]}"
        )

    neg = [r for r in result["per_question"] if not r["answerable"]]
    lines += ["", "## Negative controls (top-3 retrieved)", ""]
    for r in neg:
        lines.append(
            f"- `{r['id']}` — {r['question']}  \n"
            f"  top score {r['top_score']} · {r['retrieved'][:3]}"
        )

    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".md")


def _print_summary(result: dict) -> None:
    s = result["summary"]
    print(
        f"\nretrieval eval · boosts={'on' if result['enable_boosts'] else 'off'} · "
        f"index={result['index_size']} · {s['n_answerable']} answerable\n"
    )
    print(f"{'k':>3} | {'chunk hit@k':>12} | {'file hit@k':>11}")
    print("-" * 32)
    for k in _K_VALUES:
        print(f"{k:>3} | {_pct(s['chunk_hit_at_k'][k]):>12} | {_pct(s['file_hit_at_k'][k]):>11}")
    print(f"\nchunk MRR {s['chunk_mrr']:.3f} · file MRR {s['file_mrr']:.3f}")
    print(
        f"top-1 sim: answerable {s['answerable_top_score_mean']} "
        f"vs negative {s['negative_top_score_mean']}"
    )
    print("\nby category (chunk hit@6):")
    for cat, c in result["by_category"].items():
        print(f"  {cat:18s} n={c['n']:<3} {_pct(c['chunk_hit_at_6'])}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ad-fontes-eval", description=__doc__)
    p.add_argument("--stage", choices=["retrieval", "generation", "compare"], default="retrieval")
    p.add_argument("--model", choices=["base", "tuned"], default="base")
    p.add_argument("--no-boosts", action="store_true", help="Disable retrieval metadata boosts.")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--runs", type=int, default=1, help="Repeat the set N times (variance).")
    p.add_argument("--limit", type=int, default=None, help="Only the first N questions (dev).")
    p.add_argument("--out", type=Path, default=Path("data/eval"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.stage == "retrieval":
        result = run_retrieval(enable_boosts=not args.no_boosts, top_k=args.top_k)
        report_path = _write_reports(result, args.out)
        _print_summary(result)
        print(f"\nwrote {report_path}")
        return 0

    if args.stage == "generation":
        from app.eval.generation import run_generation, write_reports

        result = run_generation(args.model, runs=args.runs, limit=args.limit)
        path = write_reports(result, args.out)
        v = result["variance"]
        print(
            f"\n{args.model}: unsup+fab/100 = {v['unsupported_plus_fabricated_per_100']['mean']} "
            f"· citation hit {v['citation_hit_rate']['mean']} "
            f"· p50 {v['latency_p50_ms']['mean']}ms / p95 {v['latency_p95_ms']['mean']}ms "
            f"· fallback {v['hosted_fallback_rate']['mean']}"
        )
        print(f"wrote {path}")
        return 0

    print(f"stage '{args.stage}' lands in Phase 5.", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
