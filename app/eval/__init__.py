"""Evaluation harness: retrieval hit@k (Phase 1), generation + verification
metrics (Phase 2), and the base-vs-tuned comparison (Phase 5).

``questions.jsonl`` is the shared gold set; ``run_eval.py`` drives the pipeline
over it; ``metrics.py`` computes the numbers that go into ``report.md`` and,
ultimately, the README results section. No fabricated numbers: every figure in
docs traces to a run of this harness.
"""
