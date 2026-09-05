---
doc_type: stack_map
description: Cross-project map of technologies — which project uses what, and for what purpose. Use for questions like "where has Kai used FastAPI?" or "which projects use a vector database?"
---

# Technology → project map

| Technology | Projects | Role |
|---|---|---|
| Python | all | primary language |
| FastAPI | Threadfall, fons iuris, evidentia, NeuMF, EffiGO | backend APIs, SSE streaming (Threadfall) |
| React | Threadfall (18), fons iuris (19 + TypeScript), evidentia (18), NeuMF, portfolio site, EffiGO | frontends |
| Tailwind CSS | Threadfall, fons iuris (v4), evidentia, portfolio | styling |
| Vite | all React frontends | build tool |
| Cytoscape.js | Threadfall, evidentia (react-cytoscapejs / cytoscape), Causeway (dash-cytoscape) | interactive graph rendering |
| Dash | Causeway | Python-only interactive app |
| pgmpy | Threadfall, evidentia, Causeway | Bayesian networks, CPTs, Variable Elimination |
| causal-learn | pharmacausal, Causeway | PC / FCI causal discovery |
| DoWhy, EconML, lingam, statsmodels | Causeway | identification, effect estimation, LiNGAM, IV |
| networkx | Causeway | DAG representation, d-separation |
| PyTorch | NeuMF (model), Causeway (encoders), fons iuris (NLI model runtime) | deep learning |
| torchvision | Causeway | ResNet image features |
| transformers | fons iuris (NLI cross-encoder), Causeway | local transformer models |
| sentence-transformers | fons iuris (nomic-embed-text-v1.5), Threadfall (all-MiniLM-L6-v2), NeuMF (all-MiniLM-L6-v2), Causeway (SBERT text encoder) | embeddings |
| ChromaDB | fons iuris (primary index), Threadfall (default store) | vector database |
| Qdrant | Threadfall (optional production store) | vector database |
| FAISS | NeuMF | similarity search |
| HuggingFace Inference (InferenceClient / Router / Providers) | Threadfall (Qwen2.5-7B narration), fons iuris (Qwen3-30B-A3B generation), Causeway (Qwen3-VL image-to-DAG) | hosted LLM/VLM inference |
| NLI cross-encoder (DeBERTa-v3-base-mnli-fever-anli) | fons iuris | claim entailment verification |
| JSON-schema-constrained generation | fons iuris | structured, citable outputs |
| BeautifulSoup / lxml | fons iuris | EUR-Lex HTML parsing |
| pydantic v2 | Threadfall, fons iuris | schemas |
| pandas / numpy / scipy | pharmacausal, evidentia, NeuMF, Causeway, Loan project | data processing |
| pyarrow | pharmacausal | columnar storage for FAERS |
| scikit-learn | Causeway, evidentia, Loan project | modelling, evaluation |
| XGBoost | Loan project | gradient boosting |
| matplotlib / seaborn | evidentia, Causeway, Loan project | plots, reliability diagrams |
| Docker / docker-compose | Threadfall (multi-stage + nginx + chroma service), fons iuris (build-time corpus + model warm-up) | deployment |
| nginx | Threadfall | SPA serving, SSE-safe API proxy |
| GitHub Actions | fons iuris | backend test CI |
| pytest | Threadfall (64 tests), pharmacausal, Causeway, fons iuris | Python tests |
| vitest + Testing Library | evidentia | frontend tests |
| Hugging Face Spaces (Docker SDK) | fons iuris (recommended backend host) | hosting |
| Vercel | portfolio site, fons iuris frontend | hosting |
| Gemini, Amazon Titan | EffiGO | production generative AI |
| Vertex AI, Bedrock, SageMaker (GCP/AWS/Azure) | EffiGO | ML pipelines, multi-cloud |
| Spring Boot, Java | EffiGO | backend services |
| OCR | EffiGO internship | invoice extraction |
| Kaggle | Loan project | publication |

# Datasets and external sources used
- **EUR-Lex** consolidated GDPR and EU AI Act HTML — fons iuris.
- **FDA FAERS** quarterly extracts (2026Q2, 422k cases) and **SIDER 4.1** — pharmacausal.
- **DDXPlus** (Figshare, English version; ~1.29M synthetic patients) — evidentia.
- **MovieLens 100K and 25M** — NeuMF.
- Custom-engineered **loan approval dataset** — Kaggle.
- Synthetic SCM datasets (smoking/cancer, education/career, confounded) — Causeway.

# Recurring architectural patterns
- **Deterministic core + LLM at the edge**: Threadfall (engine decides, LLM narrates), fons iuris (retrieval and verification bound the LLM), evidentia (BN reasoning, no LLM needed).
- **Two-layer verification**: structural check before semantic check (fons iuris).
- **Measured tradeoffs documented in the repo**: reverted BM25 hybrid and reranker (fons iuris), bounded PC and decoupled FCI sample size (pharmacausal), lazy embedding lookup (NeuMF).
- **Full-stack delivery**: FastAPI + React + Docker appears in four projects.
