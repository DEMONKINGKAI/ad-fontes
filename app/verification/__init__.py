"""Per-claim verification: structural (layer 1), NLI (layer 2), numeric (layer 3).

Layers 1 and 3 are pure, deterministic, and implemented now (Phase 0) so they can
be tested without a model. Layer 2 (the DeBERTa NLI cross-encoder) and the label
fusion that combines all three land in Phase 2.
"""
