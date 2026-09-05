# Generation eval — base — 2026-09-05T21:06:07+00:00

- commit `f89e135` · corpus `2026-09-05` · model `qwen2.5-1.5b-instruct-q4_k_m.gguf` · hosted fallback `Qwen/Qwen2.5-7B-Instruct`
- 76 questions · 1 run(s) · temp 0.3 · local timeout 25.0s

## Headline (mean across runs)

| metric | mean | min | max |
|--|--|--|--|
| unsupported + fabricated / 100 answers | **44.74** | 44.73684210526316 | 44.73684210526316 |
| unverified prose sentences / 100 answers | **35.53** | 35.526315789473685 | 35.526315789473685 |
| citation hit rate | 0.94 | 0.9367088607594937 | 0.9367088607594937 |
| supported rate | 0.51 | 0.5063291139240507 | 0.5063291139240507 |
| latency p50 (ms) | 16406 | 16406 | 16406 |
| latency p95 (ms) | 32750 | 32750 | 32750 |
| hosted-fallback rate | 0.07 | 0.06578947368421052 | 0.06578947368421052 |

## Run 1 detail

- label distribution: {'supported': 40, 'unsupported': 29, 'fabricated_citation': 5, 'contradicted': 5}
- claims/answer 1.04 · answers with no claims 1
- contradicted/100 6.6 · numeric violations/100 3.9
- answers with ≥1 unverified prose sentence: 29%
- decline on negatives 0.4 · false-decline on answerable 0.0
- mean prose length 266 chars

## Sample overclaims caught (run 1)

- `prof-location` [unsupported] Kai is based in Germany, specifically in the state of Hessen.
- `prof-study` [unsupported] Kai is currently studying causal inference and probabilistic graphical models applied to real systems, combined with production LLM/RAG engi
- `prof-strongest` [unsupported] Kai's strongest area is causal inference and probabilistic graphical models applied to real systems, combined with production-level language
- `exp-effigo` [fabricated_citation] Kai built NLP-based document-generation pipelines for automated RFQs, contracts, and procurement documents at EffiGO Global.
- `exp-effigo` [fabricated_citation] Kai improved document generation speed and scaled serverless/containerised deployments at EffiGO Global.
- `exp-production` [unsupported] Kai has deployed machine learning to production.
- `exp-cloud` [unsupported] Kai has worked with GCP, AWS, and Azure.
- `exp-axisray` [unsupported] Kai worked on document generation pipelines at Axisray, replacing manual drafting of RFQs, contracts, and vendor documents with NLP-based do
- `exp-intern` [fabricated_citation] Kai interned at EffiGO Global in Bengaluru, India from February 2024 to May 2024.
- `exp-intern-what` [fabricated_citation] Kai built a chatbot for automated query resolution using intent classification and response generation, which deflected a significant share 
- `exp-ocr` [unsupported] Kai has collaborated on NLP-based document-generation pipelines that produce structured RFQs, contracts, and procurement documents from unst
- `skill-genai` [unsupported] Kai has extensive experience in building systems that reason, not just generate.

## Unverified prose (asserted in prose, not backed by a claim or chunk)

- `prof-identity` Kai's core technical identity revolves around causal inference and probabilistic graphical models, with a strong focus on their application in generative AI.
- `exp-effigo` At EffiGO Global, Kai built NLP-based document-generation pipelines for automated RFQs, contracts, and procurement documents, deployed on multiple cloud platfor
- `exp-axisray` This initiative streamlined the procurement and supply-chain teams, reducing the time spent on document drafting from hours to seconds and enabling scalable dep
- `exp-intern-what` He also developed an OCR-based invoice-processing pipeline that extracted line items, amounts, and vendor details from scanned documents and matched them agains
- `exp-ocr` Kai also built production generative-AI document pipelines on Gemini and Amazon Titan, deployed on Vertex AI, Bedrock, and SageMaker across GCP/AWS/Azure, with 
- `skill-causal` Kai is strong in causal inference and probabilistic graphical models, with a focus on real-world applications.
- `skill-causal` He has contributed to several projects that utilize these techniques, including Causeway, Threadfall, pharmacausal, and evidentia, which demonstrate his experti
- `skill-mlops` He has also built production generative-AI document pipelines on Gemini and Amazon Titan, deployed on Vertex AI, Bedrock, and SageMaker across GCP/AWS/Azure, wi
- `proj-fons` It solves the problem of LLM hallucinating specific clause numbers and legal requirements when asked about GDPR/AI Act compliance by using a deterministic causa
- `proj-neumf` It uses Neural Matrix Factorization (NeuMF) with a genre-aware projection inside the MLP to capture both linear and non-linear user-item interactions.
- `proj-loan` Kai built a dataset and model for loan approval prediction, which included real-world loan application data with features like applicant demographics, income, c
- `proj-loan` The dataset was engineered to handle missing values, outliers, and class imbalance, earning Kai a Kaggle Silver Medal.
