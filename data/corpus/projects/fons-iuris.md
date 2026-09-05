---
doc_type: project
project_id: fons-iuris
name: "fons iuris"
tagline: "Every claim must be entailed by the article it cites."
status: active
repo: https://github.com/DEMONKINGKAI/fons-iuris
demo: null
domain: [RAG, legal/regulatory NLP, hallucination detection, grounded generation]
stack: [Python, FastAPI, ChromaDB, sentence-transformers, nomic-embed-text-v1.5, transformers, PyTorch, Qwen3-30B-A3B via HF Inference Providers (deepinfra), DeBERTa-v3 NLI cross-encoder, BeautifulSoup/lxml, pydantic v2, React 19, TypeScript, Vite, Tailwind 4, Docker, GitHub Actions]
license: see repo LICENSE
---

# fons iuris

## One-line summary
A RAG system for GDPR and EU AI Act compliance questions where every generated claim must cite a specific article/paragraph, and an NLI verification layer checks that the cited text actually entails the claim — hallucinations are flagged, not hidden.

## The problem it solves
LLMs hallucinate specific clause numbers and legal requirements when asked about GDPR/AI Act compliance — a costly failure mode if anyone trusts the output for real regulatory decisions. Retrieval finding *a* plausible source is not treated as good enough: a claim that cites real text but isn't actually supported by it is flagged.

## Pipeline
```
EUR-Lex HTML → parse → chunk (paragraph granularity) → embed → Chroma index
question → (LLM query decomposition, regulation-tagged) → dense retrieval with breadcrumb context + recital de-prioritisation
→ LLM generation with JSON-schema-forced structured citations
→ Layer 1 structural check (cited chunk_id actually retrieved?)
→ Layer 2 NLI entailment check (cited text actually supports the claim?)
→ FastAPI → React chat UI showing a verification badge per claim
```

## Stage-by-stage design (and the reasoning behind each choice)

**1. Ingestion.** Consolidated GDPR and AI Act text is fetched from EUR-Lex's *HTML* rendition, not PDF, because the HTML carries stable per-article/paragraph anchor IDs. It is parsed into chapters → sections → articles → paragraphs → enumerated points and chunked at **paragraph granularity** — the unit these instruments are cited at in practice ("Art. 15(1) GDPR"). Each chunk keeps a `point_spans` map of character offsets so a citation can be narrowed to a single enumerated point (e.g. Art. 15(1)(h)) without re-parsing. The raw EUR-Lex snapshot is committed to the repo so builds have no live dependency on EUR-Lex.

**2. Embedding and retrieval.** Chunks are embedded with `nomic-embed-text-v1.5` (8192-token context). This was chosen after measuring that a 512-token model (BGE-base) truncated 1.7% of chunks, silently dropping over half of the longest paragraph. Two ranking corrections were added after being measured as real failures:
- *Breadcrumb context*: the embedded text is prefixed with chapter/section/article titles (the citable text stays pure). Operative legal text alone has little lexical overlap with how questions are phrased.
- *Recital de-prioritisation*: recitals (non-binding interpretive prose) read more like questions than operative articles do and were systematically out-scoring the article a compliance answer needs to cite; a modest score penalty encodes the article-over-recital hierarchy into ranking.
- *Query decomposition*: compound questions are split by an LLM into sub-questions, each optionally tagged GDPR or AI Act, because a single embedding for a two-topic question sits between topics and matches neither, and because vocabulary overlap (e.g. automated decision-making vs. AI Act human oversight) can pull retrieval into the wrong regulation.

**Tried and reverted, with numbers:** hybrid BM25 + dense retrieval (reciprocal rank fusion) dropped article-level hit rate from 100% to 80.6% on definitional questions because sibling paragraphs share vocabulary; a general-purpose cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) was inconsistent on legal text. Both were removed rather than shipped on theory.

**3. Generation.** `Qwen/Qwen3-30B-A3B` via HF Inference Providers (deepinfra route), chosen after measuring correctness comparable to a 72B model at roughly a third of the latency. Output is forced through a strict JSON schema (`response_format: json_schema, strict: true`) — not free text with regex-parsed `[1]` markers. Every claim must carry at least one citation naming an exact `chunk_id` from the retrieved set. This is **layer 1 (structural) grounding**: a citation to a chunk that was never retrieved is caught deterministically.

**4. Verification.** **Layer 2 (semantic) grounding**: an NLI cross-encoder (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) checks whether the cited text entails each claim. Long premises are split along real point boundaries (blind token windows measurably produced spurious contradiction scores), and when a citation spans several points, the relevant ones are pre-selected by embedding similarity before NLI runs. Each claim gets one of four labels, deliberately not collapsed to pass/fail: `supported`, `unsupported` (neutral), `contradicted`, `fabricated_citation` (layer-1 failure; NLI never runs).

**5. API and UI.** FastAPI loads all models once at startup and exposes `POST /api/ask`, returning each claim with its label, entailment/contradiction scores, resolved source text, and EUR-Lex link. The React UI renders the label as a colour-coded badge per claim; unsupported or fabricated claims are shown flagged, never hidden. Two abuse guards protect the one step that costs money (generation): a 500-character question cap and an in-memory per-IP rate limit (10 requests / 5 minutes) — deliberately sized for "stop casual abuse of a personal project", single-process, no Redis.

## Evaluation methodology and results
40 hand-written questions across GDPR and the AI Act, including negative controls (e.g. CCPA questions the corpus cannot answer, to catch false grounding), each with a hand-verified gold citation. Retrieval is scored at paragraph level (`hit_rate@5`, strict) and article level (looser), because missing the article and missing the best paragraph within the right article are different failure modes. Generation is scored on citation hit rate and the four-label verification breakdown, stratified by question category (definition / rights_obligations / procedural_numeric / multi_hop / negative).

Most recent full run: **97.2% article-level retrieval hit rate, 88.9% paragraph-level, 90.9% citation hit rate on generation.** Numbers move by ~5–6 points run to run even at temperature 0 because Chroma's ANN index and the MoE model's routing are non-deterministic at the margins — a single run is a band, not a point.

**Known open limitation, documented rather than hidden:** three questions (`gdpr-24`, `aiact-13`, `aiact-14`) retrieve the correct article but not the single best paragraph — a genuine sub-article dense-retrieval precision limit that neither BM25 hybrid nor cross-encoder reranking reliably fixed.

## Deployment
Backend needs ~1.3–1.5 GB RAM (two local transformer models). Recommended: Hugging Face Spaces (Docker SDK) for the backend — the image rebuilds the corpus and pre-warms the NLI model at build time; frontend on Vercel/Netlify with `VITE_API_BASE_URL`; CORS origins configured via secret. Local: `docker compose up --build` (backend :8000, frontend :5174). Backend tests run in GitHub Actions.

## Key decisions and why
- Paragraph-granularity chunks with point-span offsets — matches how law is actually cited.
- Two independent grounding layers — structural (cheap, deterministic) before semantic (NLI).
- Four-way verification labels instead of pass/fail — different failure modes need different UI treatment.
- Measured-then-adopted (and measured-then-reverted) retrieval tweaks — every ranking change has a number attached.
- Committed corpus snapshot — reproducible builds without external dependencies.

## Skills demonstrated
Production RAG design, embedding-model selection with measurement, retrieval evaluation (hit rate at multiple granularities, negative controls), structured LLM outputs via JSON schema, NLI-based hallucination detection, legal-text parsing, FastAPI, React/TypeScript, Docker, CI, cost/abuse controls on public endpoints, honest reporting of variance and limitations.
