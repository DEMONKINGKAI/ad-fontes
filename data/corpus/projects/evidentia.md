---
doc_type: project
project_id: evidentia
name: "evidentia"
tagline: "Every probability update is traceable to the evidence that caused it."
status: complete (phases 1–6)
repo: https://github.com/DEMONKINGKAI/evidentia
demo: null
domain: [explainable AI, medical diagnosis, Bayesian networks, calibration]
stack: [Python, pgmpy 1.1.0, pandas, numpy, matplotlib, FastAPI, React 18, Vite, Tailwind, cytoscape, vitest, Testing Library, DDXPlus dataset]
---

# evidentia

## One-line summary
An explainable symptom checker built as a Bayesian network over the DDXPlus dataset, where every posterior update is attributed to the specific evidence that caused it, with calibration measured by Brier score, ECE, and reliability diagrams.

## The problem it solves
Medical diagnostic tools are classified high-risk under the EU AI Act, which mandates explainability. A black-box classifier cannot satisfy this; a Bayesian network's reasoning trace can. Evidentia makes each probability shift traceable to the evidence that produced it.

## Data
DDXPlus — a public dataset (CC BY 4.0, Figshare) of synthetic patient profiles with symptoms/antecedents and ground-truth differential diagnoses: 1,025,602 train, 132,448 validation, 134,529 test patients. The repo's `data/README.md` documents a genuine gotcha: the main Figshare article's `release_conditions.json` uses a different schema version from its `release_evidences.json` (zero key overlap); the dedicated "English version" article has a self-consistent pair with `E_`-coded keys, and that is what the project uses. Other handled quirks: int-vs-string possible-values, five antecedents with stale `code_question` references (fallback: parent them directly on Disease), and the fact that `EVIDENCES` records only positives so absent evidence must be filled with `default_value`.

## How it works
1. **Structure** (`structure.py`): a BN topology of 895 nodes (grown from an initial 235-node model once the four body-location evidences were modelled). Disease is the root; evidences and antecedents hang off it, with grouped/follow-up evidences where the dataset's `code_question` links apply.
2. **CPT fitting** (`build_bn.py`, `encode.py`): raw patient CSV rows are encoded into an integer-coded table and CPTs are learned from the training split; the model is saved to `models/ddx_bn.pkl`.
3. **Inference** (`inference.py`): `DiagnosisSession` supports sequential evidence entry (`add_evidence("E_91", True)`) and exact posterior queries (`ranked_posterior(k)`).
4. **Explainability** (`explain.py`): `Explainer` attributes each posterior shift to the evidence that caused it, both as a sequential delta ("moved from X% to Y%") and as an order-independent weight of evidence.
5. **API** (`api/`): FastAPI endpoints for evidence/condition catalogs and session CRUD (`POST /sessions`, `POST /sessions/{id}/evidence`, `GET /sessions/{id}`), in-memory session store with the stateful-vs-stateless tradeoff documented in module docstrings.
6. **Frontend**: React (Vite + Tailwind), responsive two-column layout. `IntakeFlow` asks one suggested question at a time (skip-able); `BeliefBars` shows ranked probabilities with animated transitions; a tabbed panel offers **Graph** (cytoscape DAG of top-K diagnoses and evidence, edges coloured supports/opposes, incrementally updated so it doesn't jump), **Explain** (per-evidence trace and weight of evidence), and **Browse** (searchable full catalog). Per-panel `ErrorBoundary`. Frontend tests with vitest + Testing Library.

## Calibration evaluation (full held-out test split, 134,529 patients)
| | top-1 accuracy | Brier (0 = perfect) | ECE (0 = perfect) |
|---|---|---|---|
| Initial evidence only | 33.98% | 0.7158 | 0.0195 |
| Full evidence | 98.57% | 0.0156 | 0.0036 |
| DDXPlus differential (baseline) | 73.21% | 0.6665 | 0.4243 |

The README insists on reading these correctly:
- Low ECE with low accuracy at initial evidence is a *good* sign — with one finding the model is right to be uncertain; low confidence tracking low accuracy is what calibration means.
- The full-evidence vs. baseline comparison is *not* a fully fair fight: DDXPlus's synthetic generator produces fairly deterministic symptom sets per pathology, and the BN was fit on exactly that generative structure, so 98% partly reflects recovering the generator's regularities rather than real-world diagnostic skill. The baseline differential is doing a different, harder job. The comparison is informative about calibration quality (0.01 vs 0.42 ECE is a real gap), not a clean accuracy contest.

## Key decisions and why
- **Bayesian network over a discriminative classifier** — explainability is the requirement, and BN inference produces an inspectable trace.
- **Two attribution views** — sequential deltas match how a clinician experiences the intake; order-independent weight of evidence avoids order artefacts.
- **Incremental graph updates in the UI** — avoids the layout jumping on every answer.
- **Honest calibration reporting** — separates what the numbers show about calibration from what they cannot show about clinical validity.

## Skills demonstrated
Bayesian network construction and CPT learning at scale (895 nodes, ~1M rows), exact inference, explainability by evidence attribution, probabilistic calibration (Brier, ECE, reliability diagrams), careful dataset forensics, FastAPI session APIs, React/cytoscape UI, frontend testing, EU AI Act awareness.
