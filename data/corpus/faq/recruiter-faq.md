---
doc_type: faq
description: Anticipated recruiter and hiring-manager questions with answers grounded in the corpus. Each answer cites the source document. Useful both as retrieval targets and as seed material for preference-pair construction.
---

# Recruiter FAQ (grounded)

**Q: What is Kai's strongest area?**
Causal inference and probabilistic graphical models applied to real systems, combined with production LLM/RAG engineering. Four of six repos are causal/PGM projects (Causeway, Threadfall, pharmacausal, evidentia); fons iuris is a production-grade RAG system with hallucination verification. *Sources: kai-profile.md, skills.md.*

**Q: Has Kai shipped anything to production, or is it all academic?**
Yes — at EffiGO Global (Jun 2024–Jul 2025) Kai built production generative-AI document pipelines on Gemini and Amazon Titan, deployed on Vertex AI, Bedrock, and SageMaker across GCP/AWS/Azure, with an MLOps lifecycle for versioning, monitoring, and continuous training. *Source: experience.md.*

**Q: Which project best shows RAG skills?**
fons iuris: paragraph-granularity chunking of EUR-Lex law, nomic-embed-text-v1.5 chosen after measuring truncation in a 512-token model, breadcrumb context and recital de-prioritisation added after measured failures, LLM query decomposition, JSON-schema-forced citations, and a two-layer verification stack (structural + NLI). 97.2% article-level retrieval hit rate on a 40-question eval with negative controls. *Source: fons-iuris.md.*

**Q: Does Kai just take techniques from papers or actually evaluate them?**
Evaluates them — and reverts what doesn't work. In fons iuris, BM25 hybrid retrieval dropped article hit rate from 100% to 80.6% on definitional questions and a MiniLM cross-encoder reranker was inconsistent on legal text; both were removed. In evidentia the README explicitly argues the 98% accuracy headline partly reflects the synthetic generator's regularities. *Sources: fons-iuris.md, evidentia.md.*

**Q: What's the most technically demanding thing in the portfolio?**
Arguably pharmacausal: running PC and FCI on 422k FDA cases required forking causal-learn's PC with a conditioning-depth cap, discovering FCI's row-count scaling wall empirically, and decoupling FCI's sample size to run it as a confirmatory pass — while reasoning correctly about causal sufficiency, selection-as-collider, and faithfulness. Causeway is broader in theory coverage (do-calculus, counterfactuals, PN/PS/PNS, three discovery algorithms, DR estimation). *Sources: pharmacausal.md, causeway.md.*

**Q: Frontend ability?**
Solid for an ML engineer: React (18 and 19 with TypeScript), Tailwind, Vite, Cytoscape graph UIs, SSE streaming consumption, mobile-responsive layouts, per-panel error boundaries, vitest tests (evidentia), plus a custom portfolio site with an interactive SVG DAG. *Sources: threadfall.md, evidentia.md, fons-iuris.md, kai-profile.md.*

**Q: Testing and reliability practices?**
Threadfall: 64 unit tests, Docker multi-stage builds, token auth, streaming disconnect safety. pharmacausal: tests aimed at silent-corruption logic (name normalisation, unit conversion, tier assignment). fons iuris: eval suite with negative controls, CI via GitHub Actions, rate limiting and input caps on the public endpoint. *Sources: threadfall.md, pharmacausal.md, fons-iuris.md.*

**Q: German language?**
A2. Kai discloses this honestly in applications where German is required. *Source: kai-profile.md.*

**Q: What kind of role is Kai looking for?**
Full-time AI/ML engineering, causal inference, or applied generative AI roles; also Werkstudent roles compatible with writing a master's thesis. Based in Germany (Hessen). *Source: kai-profile.md.*

**Q: Is there a live demo I can try?**
The portfolio site is live on Vercel. Project repos are public on GitHub; fons iuris and Threadfall are Dockerised and documented for deployment (fons iuris recommends Hugging Face Spaces for the backend). Hosted demo links should be checked on the portfolio site. *Sources: fons-iuris.md, threadfall.md.*

**Q: Which projects use Bayesian networks specifically?**
Threadfall (belief update over world state, CPTs auto-generated from edge weights, Variable Elimination), evidentia (895-node diagnostic BN with CPTs learned from ~1M DDXPlus patients), and Causeway (pgmpy as a dependency for graph tooling). *Sources: threadfall.md, evidentia.md, tech-stack-map.md.*

**Q: Recommender systems experience?**
NeuMF Genre-Aware Movie Recommender: GMF + MLP fusion with genre projection and an optional intent tower, trained on MovieLens 25M with negative sampling and HR@10/NDCG@10 evaluation, plus a serving-time NLP intent layer. Notable engineering fix: 99.4% memory reduction via lazy embedding lookup. *Source: neumf-recommender.md.*
