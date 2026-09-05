# Retrieval eval — 2026-09-05T18:27:51+00:00

- commit `e17fa62` · corpus `2026-09-05` · index 94 chunks · embedder `nomic-ai/nomic-embed-text-v1.5`
- boosts: **on** · 66 answerable + 10 negative controls

## Overall (answerable questions)

| k | chunk hit@k | file hit@k |
|--|--|--|
| 1 |  62.1% |  90.9% |
| 3 |  81.8% |  97.0% |
| 5 |  87.9% |  98.5% |
| 6 |  93.9% | 100.0% |
| 10 | 100.0% | 100.0% |

chunk MRR **0.741** · file MRR **0.943**

mean top-1 similarity: answerable **0.822** vs negative-control **0.6915**

## By category (hit@6)

| category | n | chunk | file |
|--|--|--|--|
| adversarial | 3 | 100.0% | 100.0% |
| cross_project | 6 |  83.3% | 100.0% |
| experience | 8 |  75.0% | 100.0% |
| profile | 8 | 100.0% | 100.0% |
| project_detail | 21 | 100.0% | 100.0% |
| project_overview | 7 | 100.0% | 100.0% |
| skills | 8 |  87.5% | 100.0% |
| tech_stack | 5 | 100.0% | 100.0% |

## Chunk misses @6 (4)

- `exp-axisray` — What did Kai work on at Axisray?  
  gold ['experience#ml-data-analytics-intern-axisray-ahmedabad-india-jun-2023-jul-2023'] · got ['experience#product-engineer-effigo-global-hyderabad-india-jun-2024-jul-2025', 'kai-profile#what-kai-is-looking-for', 'kai-profile#who-kai-is', 'recruiter-faq#has-kai-shipped-anything-to-production-or-is-it-all-academic', 'recruiter-faq#what-is-kai-s-strongest-area', 'kai-profile#the-portfolio-site']
- `exp-ocr` — Has Kai worked with OCR or document extraction?  
  gold ['experience#product-engineer-intern-effigo-global-bengaluru-india-feb-2024-may-2024', 'skills#computer-vision', 'tech-stack-map#technology-project-map'] · got ['experience#product-engineer-effigo-global-hyderabad-india-jun-2024-jul-2025', 'kai-profile#what-kai-is-looking-for', 'recruiter-faq#does-kai-just-take-techniques-from-papers-or-actually-evaluate-them', 'recruiter-faq#has-kai-shipped-anything-to-production-or-is-it-all-academic', 'kai-profile#who-kai-is', 'fons-iuris#pipeline']
- `skill-ways` — How does Kai approach engineering decisions and tradeoffs?  
  gold ['skills#ways-of-working-visible-across-repos', 'recruiter-faq#does-kai-just-take-techniques-from-papers-or-actually-evaluate-them'] · got ['kai-profile#what-kai-is-looking-for', 'experience#product-engineer-effigo-global-hyderabad-india-jun-2024-jul-2025', 'kai-profile#who-kai-is', 'recruiter-faq#what-is-kai-s-strongest-area', 'recruiter-faq#what-kind-of-role-is-kai-looking-for', 'kai-profile#design-philosophy']
- `cross-rag` — Which of Kai's projects involve RAG?  
  gold ['fons-iuris#one-line-summary', 'threadfall#rag-inside-threadfall', 'skills#generative-ai-nlp'] · got ['recruiter-faq#what-is-kai-s-strongest-area', 'fons-iuris#skills-demonstrated', 'experience#product-engineer-effigo-global-hyderabad-india-jun-2024-jul-2025', 'kai-profile#what-kai-is-looking-for', 'kai-profile#who-kai-is', 'kai-profile#the-portfolio-site']

## Negative controls (top-3 retrieved)

- `neg-salary` — What is Kai's expected salary?  
  top score 0.6728 · ['kai-profile#what-kai-is-looking-for', 'kai-profile#who-kai-is', 'recruiter-faq#what-kind-of-role-is-kai-looking-for']
- `neg-partner` — Does Kai have a girlfriend or partner?  
  top score 0.6388 · ['kai-profile#who-kai-is', 'kai-profile#what-kai-is-looking-for', 'recruiter-faq#has-kai-shipped-anything-to-production-or-is-it-all-academic']
- `neg-visa` — What is Kai's visa or immigration status?  
  top score 0.6599 · ['kai-profile#what-kai-is-looking-for', 'kai-profile#who-kai-is', 'recruiter-faq#what-kind-of-role-is-kai-looking-for']
- `neg-rust` — Does Kai know Rust?  
  top score 0.6628 · ['kai-profile#who-kai-is', 'kai-profile#what-kai-is-looking-for', 'recruiter-faq#what-is-kai-s-strongest-area']
- `neg-k8s` — Has Kai worked with Kubernetes?  
  top score 0.6744 · ['experience#product-engineer-effigo-global-hyderabad-india-jun-2024-jul-2025', 'recruiter-faq#has-kai-shipped-anything-to-production-or-is-it-all-academic', 'kai-profile#who-kai-is']
- `neg-politics` — What are Kai's political views?  
  top score 0.6562 · ['kai-profile#who-kai-is', 'kai-profile#what-kai-is-looking-for', 'recruiter-faq#what-is-kai-s-strongest-area']
- `neg-grewup` — Where did Kai grow up?  
  top score 0.6653 · ['kai-profile#who-kai-is', 'kai-profile#what-kai-is-looking-for', 'recruiter-faq#what-is-kai-s-strongest-area']
- `neg-gpa` — What is Kai's GPA?  
  top score 0.6823 · ['kai-profile#who-kai-is', 'kai-profile#what-kai-is-looking-for', 'recruiter-faq#what-is-kai-s-strongest-area']
- `adv-fons-users` — How many users does fons iuris have?  
  top score 0.8602 · ['fons-iuris#deployment', 'fons-iuris#evaluation-methodology-and-results', 'fons-iuris#skills-demonstrated']
- `adv-best-eng` — Is Kai the best ML engineer you know?  
  top score 0.7419 · ['kai-profile#who-kai-is', 'kai-profile#what-kai-is-looking-for', 'experience#product-engineer-effigo-global-hyderabad-india-jun-2024-jul-2025']
