---
doc_type: skills
person: Archit Sharma (Kai)
source: portfolio site "Focus / Expertise" section, cross-checked against project repos
---

# Skills and expertise

## Causal inference & probabilistic graphical models
DAGs and d-separation · do-calculus (Pearl) · backdoor / front-door criteria · structural causal models · PC / FCI / GES algorithms · NOTEARS / DirectLiNGAM · Markov equivalence classes · Halpern–Pearl causality · CausalVAE / CITRIS / BISCUIT · Bayesian networks, CPT learning, Variable Elimination · treatment-effect estimation (doubly-robust AIPW, IPW, G-formula, T/S/X-learners, instrumental variables) · counterfactuals (abduction–action–prediction, but-for test, PN/PS/PNS) · calibration (Brier, ECE, reliability diagrams).
*Evidence:* Causeway, Threadfall, pharmacausal, evidentia.

## Generative AI & NLP
LLM application architecture · RAG pipelines (chunking, embedding-model selection, dense retrieval, query decomposition, ranking corrections, retrieval evaluation) · grounded generation with JSON-schema-constrained outputs · NLI-based hallucination/entailment verification · structured-output prompting with fallback chains · streaming (SSE) · multimodal LLMs (vision + language, image-to-graph extraction) · document generation · visual question answering · VAEs, GANs, diffusion models · HuggingFace ecosystem (Inference API/Providers, transformers, sentence-transformers) · Gemini, Amazon Titan, GPT-4o · prompt engineering.
*Evidence:* fons iuris, Threadfall, Causeway, EffiGO work.

## Data science & analytics
Feature engineering and selection · class-imbalance handling · supervised and unsupervised ML · XGBoost, Random Forest, SVM · synthetic data generation · Kaggle and reproducible pipelines · pandas, numpy, matplotlib, seaborn · statistical modelling · large-scale ETL over messy real-world data (FAERS, DDXPlus, MovieLens 25M).
*Evidence:* Loan Approval dataset/model, pharmacausal, evidentia, NeuMF.

## Computer vision
CNNs, ResNet, ViT · image classification and embedding · feature extraction (torchvision) · multimodal grounding · document understanding · image-text retrieval.
*Evidence:* Causeway image encoders and image-to-DAG, EffiGO OCR pipeline.

## Reinforcement learning
Policy gradient (PPO, A3C) · Q-learning, DQN · model-based RL · causal RL and planning · reward shaping · Gymnasium / OpenAI Gym.

## Recommender systems
Neural matrix factorization (NeuMF: GMF + MLP) · negative sampling · HR@K / NDCG@K evaluation · semantic intent steering · hybrid ranking with tunable signals.
*Evidence:* NeuMF recommender.

## MLOps & engineering
GCP, AWS, Azure · Vertex AI, Bedrock, SageMaker · multi-cloud and serverless deployments · ML pipeline design · FastAPI, Python · PyTorch · pgmpy, DoWhy, causal-learn, EconML · Docker (multi-stage builds, compose, nginx) · GitHub Actions CI · vector databases (ChromaDB, Qdrant, FAISS) · React, TypeScript, Tailwind, Vite · Dash / Cytoscape · Spring Boot, Java · pytest, vitest · rate limiting and token auth for public demos · Hugging Face Spaces, Vercel deployment.

## Ways of working (visible across repos)
Measure before adopting — retrieval tweaks in fons iuris and model choices are backed by numbers, and reverted when they don't hold. Limitations stated as first-class content (pharmacausal, evidentia). Tests aimed at silent-failure logic. Reproducibility (seeded sessions, committed snapshots, config-driven serving).
