---
doc_type: project
project_id: pharmacausal
name: "pharmacausal"
tagline: "Causal discovery over FDA adverse-event reports."
status: complete (v1; v2 improvements listed)
repo: https://github.com/DEMONKINGKAI/pharmacausal
demo: interactive HTML report in repo (reports/pc_fci_sider_viz.html)
domain: [causal discovery, pharmacovigilance, epidemiology, data engineering]
stack: [Python, pandas, pyarrow, numpy, requests, tqdm, causal-learn (PC/FCI), pytest, FAERS, SIDER, self-contained HTML/SVG visualisation]
license: MIT
---

# pharmacausal

## One-line summary
Constraint-based causal discovery (PC and FCI) over 422k FDA FAERS adverse-event reports to surface drug→event hypotheses that survive confounder adjustment, validated against SIDER, and explicit about exactly where FAERS breaks the algorithms' assumptions.

## The problem it solves
Naive drug–adverse-event co-occurrence in spontaneous-reporting data is confounded: sicker patients get more drugs *and* have more adverse events. Standard association mining cannot separate pharmacological signal from that bias. The project's thesis: the confounding structure, missing-data handling, PC-vs-FCI choice, and validation strategy matter as much as the discovery algorithm, and honesty about assumption violations is part of doing it correctly.

## Pipeline
| Step | Module | What it does |
|---|---|---|
| ETL | download.py, parse.py | Download a FAERS quarterly ASCII extract, dedupe by caseversion, drop retracted cases, filter child tables to canonical primaryids |
| Confounder audit | explore.py | Inventory which confounder fields exist and how populated they are |
| Features | features.py | Case-level matrix: drug exposure flags, adverse-event flags, indication flags, demographic/reporting confounders |
| Discovery | discovery.py, bounded_pc.py | PC (full graph) and FCI (candidate-edge subset) with tiered background knowledge |
| Validation | validate.py | Cross-reference discovered drug→event edges against SIDER |
| Visualisation | reports/pc_fci_sider_viz.html | Side-by-side PC/FCI graphs, edges coloured by SIDER status |

Run: 2026Q2 extract, **422,458 canonical cases** after dropping 4,217 retracted IDs.

## Data realities (FAERS)
Confounder coverage: report type / country / drug count 100%; sex 81%; age 63% (needs unit normalisation); indication 93.5% have a record but 38% of rows are "unknown"; reporter occupation 28% and weight 17% too sparse. Two structural limits bound what any analysis here can mean:
1. **No unexposed comparison population** — FAERS has no denominator of patients who took a drug and had no event, so no population-level causal effect is estimable.
2. **Selection into the dataset is a collider** — whether a case is reported depends on both the drug (media, litigation, Weber effect) and outcome severity; no covariate represents "was this reported", so no conditioning fixes it.

## Feature engineering (422,458 × 143)
50 drug flags, 75 adverse-event flags, 10 indication flags, plus age, sex, n_drugs, n_suspect_drugs, reporter_us, report_expedited. Notable choices:
- Drug identity = active ingredient (`prod_ai`), not brand; combination products exploded; salt/ester suffixes stripped — required for the SIDER join.
- **Exposure flags are role-agnostic**: primary suspect, secondary suspect, and concomitant all count. `role_cod` is the reporter's causal *opinion*; using only "primary suspect" would bake the reporter's judgment into the input before discovery runs.
- Indication kept as a node (top-10 known categories) rather than "adjusted and forgotten", because it can legitimately sit upstream of the drug.

## Discovery: assumptions, stated plainly
- **Missing data**: listwise deletion (232,233 complete rows, 55%) rather than missingness indicators, which would mostly encode which report types fill in demographics and become a spurious hub.
- **PC assumes causal sufficiency, which is false for FAERS** (disease severity, comorbidity burden, reporting stimulus are unmeasured). PC output is read as candidates under a known-violated assumption.
- **FCI relaxes causal sufficiency** (latent confounders shown as `o->` / `<->` in the PAG) **but does not solve selection bias**.
- **Faithfulness**: one mechanical near-violation named — `n_suspect_drugs ≤ n_drugs` by construction.
- **Tiered background knowledge** fixes orientation the data cannot: demographics/report metadata → indication/polypharmacy → drug → event. An early run had oriented `Headache → ABALOPARATIDE`.
- **Tractability as a finding**: causal-learn's `pc()` has no cap on conditioning-set size, infeasible at 141 variables, so `bounded_pc.py` forks the skeleton search with a depth cap. FCI scales badly with node count *and* row count (48-var subset: 4.5s at N=3,000, did not finish in 30 min at N=10,000), which led to decoupling FCI's sample size from PC's and running FCI as a confirmatory check on PC's candidate edges (64-node subset).

## Results
- PC (N=10,000, α=0.001, depth ≤ 2): 489 edges, **36 drug→event edges**.
- FCI (N=5,000, 64-node subset): 85 edges, **22 drug→event edges**; 20 of 22 match a PC edge exactly.
- FCI's more interesting output: 16 edges marked as sharing a latent common cause — symptom clusters (Dizziness ↔ Headache, Pruritus ↔ Rash) and co-prescription clusters (ALBUTEROL o→ MONTELUKAST) that naive co-occurrence would misread as direct causation.

## Validation against SIDER (not DrugBank)
DrugBank holds drug–drug interactions, a different relation type; SIDER holds drug→side-effect pairs mined from labels, the correct match.
| | Edges | SIDER coverage | Precision on covered subset |
|---|---|---|---|
| PC | 36 | 33% (12/36) | 58% (7/12) |
| FCI | 22 | 27% (6/22) | 67% (4/6) |
Coverage and precision are reported separately on purpose: SIDER 4.1 dates from 2015 and half the candidate drugs (dupilumab, semaglutide, tirzepatide, bimekizumab, abaloparatide) postdate it or are biologics it covers poorly.

**Standout recovered signals**: ROSUVASTATIN → Myalgia (textbook statin effect), MEDROXYPROGESTERONE → Meningioma (an active, real pharmacovigilance signal), APIXABAN → Anaemia (mechanistically sensible). **Stated failure**: MONTELUKAST → Asthma is indication bleeding into the reaction field; several unmatched edges are administrative MedDRA process terms (Therapy interrupted, Off label use, Incorrect dose administered) that a v2 should stop-list.

## Testing and reproducibility
Unit tests target logic where a silent bug would corrupt results without raising: drug-name normalisation, age-unit conversion, background-knowledge tier assignment, PC edge parsing. Everything downstream of download is deterministic given the extract and `random_state=0`; the quarter is a CLI parameter.

## Skills demonstrated
Real-world causal discovery (PC, FCI, PAGs), epidemiological reasoning about confounding, colliders, and selection bias, large-scale ETL over messy government data, feature engineering with domain justification, algorithm tractability engineering (bounded PC, decoupled sample sizes), external validation with honest coverage/precision separation, scientific writing that treats limitations as first-class.
