"""RLAIF preference-data pipeline (Phase 3) and DPO training/export (Phase 4).

Flow: gen_questions -> gen_candidates (+ perturb) -> judge -> build_pairs ->
data/rlhf/pairs.jsonl (TRL DPO format) -> train_dpo.ipynb (Colab) -> export_gguf
-> compare. The judge reuses this repo's own verification layer plus an LLM-judge
rubric; ARCHITECTURE.md records that the preference labels are therefore
self-generated, which is a stated limitation.
"""
