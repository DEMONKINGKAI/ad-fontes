# Portfolio RAG corpus — Archit "Kai" Sharma

Curated knowledge base about Kai's projects, skills, and experience, built from the public GitHub repos (READMEs, ARCHITECTURE.md, CHANGELOG.md, requirements/package manifests) and the portfolio site source (Work, About, Experience, Focus components). Excluded by request: the `portfolio` repo as a project (used only for skills/positioning) and `ProjectZeus`.

## Layout
```
projects/   one file per project — consistent sections so chunks are comparable
profile/    who Kai is, experience, skills
stack/      technology → project map, datasets, recurring patterns
faq/        grounded recruiter FAQ with source citations
manifest.json   machine-readable index (ids, paths, tags, stack) for ingestion
```

## Project file schema
Every `projects/*.md` has YAML frontmatter (`project_id`, `name`, `repo`, `stack`, `domain`, `status`) and these sections where applicable:
One-line summary · The problem it solves · How it works / pipeline · Architecture or structure · Results / evaluation · Key decisions and why · Skills demonstrated · (Limitations, References).

## Ingestion notes
- Chunk on `##` headings (roughly 150–400 tokens each); prefix each chunk with `project name › section` as breadcrumb context for the embedding, keep the raw text for citation (this mirrors the fons iuris approach).
- Frontmatter `stack` and `domain` are good metadata filters ("which projects use X?").
- `stack/tech-stack-map.md` answers cross-project questions directly; retrieve it whenever a query names a technology rather than a project.
- `faq/recruiter-faq.md` is intentionally answer-shaped; useful as gold answers for evaluation and as seed material for preference pairs.

## Provenance and caveats
- All numbers (eval scores, dataset sizes, edge counts) are copied from repo documentation as of 2026-09-05; verify before quoting to a recruiter if the repos change.
- Two inconsistencies noticed between the portfolio site and the repos, left as-is in the corpus but worth fixing on the site: (1) the site lists **pgmpy** in pharmacausal's stack, but the repo's requirements.txt states pgmpy/networkx are intentionally not used — the discovery library is **causal-learn**; (2) Threadfall's README badge says HuggingFace "Qwen2.5" while the site says "HuggingFace" generically — consistent, but the narrator model is Qwen2.5-7B-Instruct, not Qwen3.
- The portfolio repo README still contains placeholder-checklist text (YOUR_GITHUB_URL etc.) even though the components are filled in.
