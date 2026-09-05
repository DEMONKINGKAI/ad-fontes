---
doc_type: project
project_id: neumf
name: "NeuMF Genre-Aware Movie Recommender"
tagline: "Describe a mood. Get a list that fits."
status: complete
repo: https://github.com/DEMONKINGKAI/NeuMF-Movie-Recommendation-Engine
demo: null
domain: [recommender systems, deep learning, NLP intent understanding]
stack: [Python 3.9+, PyTorch, FastAPI, Uvicorn, sentence-transformers (all-MiniLM-L6-v2), faiss-cpu, numpy, pandas, scikit-learn, YAML config, React, Vite, Anime.js, Axios, MovieLens 100K and 25M]
---

# NeuMF Genre-Aware Movie Recommender

## One-line summary
A Neural Matrix Factorization recommender extended with genre awareness and a natural-language intent layer: a user describes a mood ("something exciting and funny") and gets recommendations that stay personalised to their taste.

## The problem it solves
Conventional recommenders force users into rigid genre filters or rely on opaque collaborative signals. This bridges natural-language intent and collaborative filtering.

## Model
**NeuMF** fuses two branches:
- **GMF** (Generalized Matrix Factorization): elementwise product of user and item embeddings — linear interactions.
- **MLP**: concatenation of user embedding, item embedding, a projected multi-hot **genre vector** (`genre_proj(g) = W·g + b`), and an optional **intent tower** (`ReLU(W2·Dropout(ReLU(W1·v + b1)) + b2)` over a 384-dim sentence embedding), passed through fully-connected layers (e.g. [128, 64]).
- Final: concat GMF and MLP outputs → linear → sigmoid. Trained with binary cross-entropy and negative sampling (4 negatives per positive by default), Adam optimiser, leave-one-out train/val/test splits, evaluated with HR@10 and NDCG@10.

Supports MovieLens 100K (tab-separated, 19 fixed genres) and MovieLens 25M (CSV, dynamic pipe-separated genres, movies to 2019) with automatic format detection; 25M is the recommended default.

## NLP intent system (serving time)
1. **Embed** the query with all-MiniLM-L6-v2 and L2-normalise.
2. **Genre centroids**: mean normalised embedding of all movies in each genre (from "title + genres" text).
3. **Query steering**: pull the query toward the top-k most similar genre centroids and away from the bottom-k: `q = q + α_pos·c_top − α_neg·c_bot`, renormalise.
4. **Affect detection**: cosine similarity against precomputed anchors for {sad, funny, scary, romantic, exciting, inspiring, family, dark}, with **keyword-aware boosting** — action keywords ("pumping", "adrenaline", "thrilling") boost "exciting" and damp "scary". This fixed a concrete bug where "make my blood pumping" scored scary 0.47 vs exciting 0.36 and returned horror.
5. **Genre weights**: clipped centroid similarities plus affect-conditioned genre priors P(g|a), thresholding, conflict suppression (e.g. Horror suppressed for "exciting"), normalisation.
6. **Candidate retrieval and scoring**: top candidates by item-embedding similarity, scored by the NeuMF model, then `final = base + α_genre·genre_bonus + α_pop·pop_bonus + α_embed·embed_bonus` with user-tunable weights (defaults 0.35 / 0.05 / 0.60).

## Engineering highlight: the memory fix
The original implementation stored every interaction's intent vector in memory — about 86.8 GB for the 25M dataset. Item embeddings are now keyed by item with lazy lookup in the data loader, storing only item indices (8 bytes vs 1,536 bytes each): a 99.4% reduction to ~485 MB with no performance loss. A `max_ratings` sampling parameter enables faster experiments.

## System components
- `recsys/` — data.py (loading, splits, sampling), model.py, train.py, eval.py.
- `backend/` — FastAPI with `/recommendations` (genre-based, strict or soft filtering), `/intent_recommendations`, `/genres`, `/users`; hyperparameters loaded from `configs/starter.yaml` to match training exactly; genre centroids, popularity scores, and affect anchors precomputed at startup; CORS enabled.
- `frontend/` — React with GenreSelector, PromptSearch (alpha tuning), Recommendations, StrictToggle; Axios interceptors, 30s timeout, loading states, request/response logging.
- `scripts/` — dataset download, item-embedding build (`item_embeddings.npy`, [num_items, 384]).

## Key decisions and why
- **NeuMF over plain MF** — captures both linear and non-linear user-item interactions.
- **Genre as a learned projection inside the MLP** rather than a post-filter — the model learns how genre interacts with taste.
- **Intent handled mostly at serving time** (steering, affects, bonuses) with an optional trained intent tower — keeps the base model reusable and the intent logic tunable without retraining.
- **Config-driven serving** — the API reads the same YAML as training so architecture mismatches are caught at load.

## Skills demonstrated
Deep-learning recommender design (NeuMF, negative sampling, HR/NDCG evaluation), sentence embeddings and semantic steering, serving-time ranking with multiple signals, memory-efficient data loading at 25M-row scale, FastAPI + React full stack, debugging model behaviour from concrete failure cases.
