"""RLAIF preference-data pipeline (Phase 3) and DPO training/export (Phase 4).

Flow (every stage appends to a resumable JSONL under ``data/rlhf/`` and skips ids
already done)::

    gen_questions   -> questions.jsonl      (~378 stratified, 20% holdout)
    gen_candidates  -> candidates.jsonl     (base×2 temps + hosted + perturb, verified)
       + perturb                            (labelled faithfulness degradations)
    judge           -> judged.jsonl         (verification + LLM rubric + veto)
    build_pairs     -> pairs.jsonl          (TRL DPO format; length-bias controlled)
                       pairs_full.jsonl     (same + provenance metadata)
    report          -> phase3-report.md     (perturbation detection rate, length bias)
    hand_label      -> hand_label.jsonl     (blind sample for Kai) -> judge–human agreement

    train_dpo.ipynb (Colab, Phase 4) -> merge -> export_gguf.md -> compare.py (Phase 5)

The judge reuses this repo's own verification layer plus an LLM-judge rubric, so
the preference labels are self-generated — a stated limitation (ARCHITECTURE.md).
``_io.py`` holds the shared JSONL helpers.
"""
