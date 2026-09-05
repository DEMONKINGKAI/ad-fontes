# Generation eval — base — 2026-09-05T20:03:06+00:00

- commit `e17fa62` · corpus `2026-09-05` · model `qwen2.5-1.5b-instruct-q4_k_m.gguf` · hosted fallback `Qwen/Qwen2.5-7B-Instruct`
- 76 questions · 1 run(s) · temp 0.3 · local timeout 25.0s

## Headline (mean across runs)

| metric | mean | min | max |
|--|--|--|--|
| unsupported + fabricated / 100 answers | **56.58** | 56.578947368421055 | 56.578947368421055 |
| citation hit rate | 0.94 | 0.9375 | 0.9375 |
| supported rate | 0.42 | 0.425 | 0.425 |
| latency p50 (ms) | 14219 | 14219 | 14219 |
| latency p95 (ms) | 28014 | 28014 | 28014 |
| hosted-fallback rate | 0.07 | 0.06578947368421052 | 0.06578947368421052 |

## Run 1 detail

- label distribution: {'supported': 34, 'unsupported': 38, 'fabricated_citation': 5, 'contradicted': 3}
- claims/answer 1.05 · answers with no claims 0
- contradicted/100 3.9 · numeric violations/100 3.9
- decline on negatives 0.4 · false-decline on answerable 0.0
- mean prose length 269 chars

## Sample overclaims caught (run 1)

- `prof-identity` [unsupported] Kai's core technical identity is centered on causal inference and probabilistic graphical models, with an emphasis on their application in g
- `prof-philosophy` [unsupported] Kai's design philosophy emphasizes a clean separation of concerns, where deterministic logic owns state and decisions, while learned models 
- `prof-site` [unsupported] Kai's portfolio website is built with Vite + React + Tailwind, and it showcases his projects through an interactive causal DAG.
- `prof-location` [unsupported] Kai is based in Germany, specifically in the state of Hessen.
- `prof-study` [unsupported] Kai is currently studying causal inference and probabilistic graphical models applied to real systems, combined with production LLM/RAG engi
- `prof-strongest` [unsupported] Kai's strongest area is causal inference and probabilistic graphical models applied to real systems, combined with production-level language
- `exp-effigo` [fabricated_citation] Kai built NLP-based document-generation pipelines at EffiGO Global, automating RFQs, contracts, and procurement documents.
- `exp-effigo` [fabricated_citation] Kai deployed these pipelines on Google Cloud Vertex AI, Amazon Bedrock, AWS SageMaker, and integrated them with MLOps for continuous trainin
- `exp-production` [unsupported] Kai has deployed machine learning to production.
- `exp-cloud` [unsupported] Kai has worked with GCP, AWS, and Azure.
- `exp-axisray` [contradicted] Kai worked on document generation pipelines at Axisray, replacing manual drafting of RFQs, contracts, and vendor documents with NLP-based do
- `exp-intern` [fabricated_citation] Kai interned at EffiGO Global in Bengaluru, India from February 2024 to May 2024.
