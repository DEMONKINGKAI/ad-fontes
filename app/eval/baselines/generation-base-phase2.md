# Generation eval — base — 2026-09-06T01:24:00+00:00

- commit `e4d41ea` · corpus `2026-09-05` · model `qwen2.5-1.5b-instruct-q4_k_m.gguf` · hosted fallback `Qwen/Qwen2.5-7B-Instruct`
- 76 questions · 1 run(s) · temp 0.3 · local timeout 25.0s

## Headline (mean across runs)

| metric | mean | min | max |
|--|--|--|--|
| unsupported + fabricated / 100 answers | **48.68** | 48.68421052631579 | 48.68421052631579 |
| unverified prose sentences / 100 answers | **68.42** | 68.42105263157895 | 68.42105263157895 |
| citation hit rate | 0.94 | 0.9367088607594937 | 0.9367088607594937 |
| supported rate | 0.48 | 0.4810126582278481 | 0.4810126582278481 |
| latency p50 (ms) | 15343 | 15343 | 15343 |
| latency p95 (ms) | 32360 | 32360 | 32360 |
| hosted-fallback rate | 0.07 | 0.06578947368421052 | 0.06578947368421052 |

## Run 1 detail

- label distribution: {'supported': 38, 'unsupported': 32, 'fabricated_citation': 5, 'contradicted': 4}
- claims/answer 1.04 · answers with no claims 1
- contradicted/100 5.3 · numeric violations/100 3.9
- answers with ≥1 unverified prose sentence: 49%
- decline on negatives 0.4 · false-decline on answerable 0.0
- mean prose length 263 chars

## Sample overclaims caught (run 1)

- `prof-location` [unsupported] Kai is based in Germany, specifically in the state of Hessen.
- `prof-study` [unsupported] Kai is currently studying causal inference and probabilistic graphical models applied to real systems, combined with production LLM/RAG engi
- `prof-strongest` [unsupported] Kai's strongest area is causal inference and probabilistic graphical models applied to real systems, combined with production-level language
- `exp-effigo` [fabricated_citation] Kai built NLP-based document-generation pipelines for automated RFQs, contracts, and procurement documents at EffiGO Global.
- `exp-effigo` [fabricated_citation] Kai developed a chatbot for automated query resolution and an OCR-based invoice-processing pipeline at EffiGO Global.
- `exp-production` [unsupported] Kai has deployed machine learning to production.
- `exp-cloud` [unsupported] Kai has worked with GCP, AWS, and Azure.
- `exp-axisray` [unsupported] Kai worked on document generation pipelines at Axisray, replacing manual drafting of RFQs, contracts, and vendor documents with NLP-based do
- `exp-intern` [fabricated_citation] Kai interned at EffiGO Global in Bengaluru, India from February 2024 to May 2024.
- `exp-intern-what` [fabricated_citation] Kai built a chatbot for automated query resolution using intent classification and response generation, which deflected a significant share 
- `exp-ocr` [unsupported] Kai has collaborated on NLP-based document-generation pipelines that produce structured RFQs, contracts, and procurement documents from unst
- `skill-genai` [unsupported] Kai has extensive experience in building systems that reason, not just generate.

## Unverified prose (asserted in prose, not backed by a claim or chunk)

- `prof-identity` Kai's core technical identity revolves around causal inference and probabilistic graphical models, with a strong focus on their application in generative AI.
- `prof-philosophy` This principle is visible across Kai's portfolio, including Threadfall (engine decides, LLM narrates), fons iuris (retrieval + verification constrain the LLM), 
- `prof-study` Kai is currently studying causal inference and probabilistic graphical models applied to real systems, combined with production LLM/RAG engineering.
- `prof-strongest` Kai's strongest area is causal inference and probabilistic graphical models applied to real systems, combined with production-level language model (LLM) and ret
- `exp-effigo` At EffiGO Global, Kai built NLP-based document-generation pipelines for automated RFQs, contracts, and procurement documents, deployed on multiple cloud platfor
- `exp-effigo` Kai also developed a chatbot for automated query resolution and an OCR-based invoice-processing pipeline.
- `exp-axisray` Kai worked on document generation pipelines at Axisray, replacing manual drafting of RFQs, contracts, and vendor documents with NLP-based document-generation pi
- `exp-axisray` This initiative streamlined the procurement and supply-chain teams, reducing the time spent on document drafting from hours to seconds and enabling scalable dep
- `exp-intern` Yes, Kai interned at EffiGO Global in Bengaluru, India from February 2024 to May 2024.
- `exp-intern-what` During his EffiGO internship, Kai built a chatbot for automated query resolution using intent classification and response generation, which deflected a signific
- `exp-intern-what` He also developed an OCR-based invoice-processing pipeline that extracted line items, amounts, and vendor details from scanned documents and matched them agains
- `exp-intern-what` The stack used was Python, Spring Boot, OCR, NLP, and RESTful APIs, with Java.
