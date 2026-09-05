---
doc_type: project
project_id: causeway
name: "Causeway — Multimodal Causal Inference Framework"
tagline: "A structured path from raw data to true causes."
status: flagship
repo: https://github.com/DEMONKINGKAI/Causeway
demo: null
domain: [causal inference, causal discovery, counterfactual reasoning, data science tooling]
stack: [Python 3.9+, numpy, scipy, pandas, networkx, scikit-learn, statsmodels, PyTorch, torchvision, transformers, pgmpy, DoWhy, EconML, causal-learn, lingam, Dash, dash-cytoscape, matplotlib, seaborn, HuggingFace Router API, pytest]
---

# Causeway

## One-line summary
A complete Python framework for structural causal modelling, causal discovery, counterfactual reasoning, effect estimation, and interactive graph visualisation — grounded in Pearl's do-calculus and the Potential Outcomes framework.

## The problem it solves
Most machine learning finds correlations, not causes. There was no single Python framework that took raw data all the way through causal discovery → identification via do-calculus → structural equation fitting → effect estimation → counterfactual reasoning → interactive intervention, all in one place and grounded in Pearl's formal theory.

## What it does, end to end
Given raw data (tabular CSV, images, text, or time series), Causeway:
1. **Discovers** the causal DAG (PC, NOTEARS, or DirectLiNGAM).
2. **Identifies** whether causal effects are estimable from observational data (backdoor, front-door, instrumental variables, Pearl's three do-calculus rules).
3. **Fits** structural equations from data via `SCMLearner` — every coefficient is learned, nothing hardcoded; each node's fitted equation, R², and noise are inspectable.
4. **Estimates** Average Treatment Effects (regression adjustment / G-formula, IPW, doubly-robust AIPW) and Conditional ATE (T-, S-, X-learners).
5. **Answers counterfactuals** via the Abduction–Action–Prediction procedure.
6. **Tests "but-for" causation** (the legal standard of necessary causation) and computes Probability of Necessity, Sufficiency, and PNS with bounds.
7. **Visualises** everything in a two-tab Dash app with live do(X=v) interventions.

## Theory implemented (as documented in the README)
- **Pearl's Ladder of Causation** — association / intervention / counterfactual, with explicit notes on why level 1 cannot answer level 2 questions.
- **SCM as a 4-tuple** ⟨U, V, F, P(U)⟩ with structural equations V_i = f_i(Pa(V_i), U_i); worked 7-variable Education/Career example (SES, Health, Education, Skill, Income, JobSat, Wellbeing).
- **Do-calculus rules 1–3** with the graph mutilations G_X̄, G_X̄Z̄, G_X̄Z(W), and an `identify()` routine that reports the method ("backdoor", "frontdoor", "failed") and a human-readable adjustment formula.
- **Backdoor and front-door criteria** with adjustment formulas; **instrumental variables** with Wald estimator and 2SLS (GMM also listed in `iv.py`).
- **Counterfactuals**: abduction of exogenous noise (for additive noise, U_i = observed V_i − f_i(parents)), action (mutilate the SCM), prediction (propagate). `ButForTest` returns caused / factual / counterfactual outcomes; `population_pn()` returns PN, PS, PNS.
- **Causal discovery**: PC (partial-correlation CI tests, v-structure orientation, Meek rules), NOTEARS (acyclicity constraint h(W) = tr(e^{W∘W}) − d = 0 solved with augmented Lagrangian), DirectLiNGAM (ICA-based, non-Gaussian noise, unique causal order via kurtosis/HSIC). The README includes a comparison table: PC returns a CPDAG, NOTEARS a DAG, LiNGAM a unique DAG; none handle hidden confounders.
- **Effect estimation**: doubly-robust AIPW recommended (consistent if either outcome or propensity model is correct); Granger causality for time series.

## Multimodal support
Four encoders implementing a `BaseEncoder` ABC: `TabularEncoder` (StandardScaler + OHE + PCA), `TextEncoder` (SBERT, TF-IDF+SVD fallback), `ImageEncoder` (ResNet18/50 features, histogram fallback), `TimeSeriesEncoder` (ROCKET, statistical-feature fallback). `MultimodalFusion` supports concat / mean / attention strategies. Fused features feed the same discovery and estimation pipeline.

## The interactive app (Dash + dash-cytoscape, `python visualize.py`, port 8050)
**Tab 1 — Discover & Analyze**: upload CSV, upload an image (graph image → vision-LLM extraction; data image → per-row RGB/brightness/contrast/saturation features), build a graph manually, or load built-in examples (Smoking/Cancer, Education/Career, Confounded). Choose PC / NOTEARS / LiNGAM and alpha, pick treatment and outcome, run discovery, read the identification result (method, adjustment set, formula).

**Tab 2 — Visualize & Intervene**: Cytoscape graph with node roles (exogenous root, mediator, treatment, outcome, intervened). Click a node → Node Inspector with fitted equation, coefficients, R², noise. Add do(X=v) interventions and "Apply All" (intervened nodes gold, cut edges dashed). d-separation query panel. But-for test panel. Sample the SCM under active interventions and see bar charts.

**Image-to-DAG extraction**: a hand-drawn or printed causal diagram is resized (max 1024px JPEG) and sent to the HuggingFace Router API (`router.huggingface.co/v1/chat/completions`); supported vision models include Qwen3-VL-8B (recommended), Qwen3-VL-30B-A3B, Qwen2.5-VL-72B, Llama-4-Scout-17B. The model returns `{"nodes": [...], "edges": [[src, tgt], ...]}` and the DAG is loaded into the visualiser.

## Sample datasets (1,000 rows each)
- `smoking_cancer.txt` — Genetics → Smoking → Tar → Cancer with a Genetics→Cancer confounding path; front-door mediator Tar; true ATE ≈ 0.26 vs naive OLS ≈ 0.42.
- `education_career.txt` — the 7-variable SCM above.
- `confounded.txt`.

## Project structure
```
causality/
  core/         dag.py, scm.py, do_calculus.py, counterfactual.py
  discovery/    pc.py, notears.py, lingam.py
  inference/    identification.py, estimators.py (ATE/CATE), iv.py
  learning/     scm_learner.py
  multimodal/   base.py, tabular.py, text.py, image.py, timeseries.py, fusion.py
  visualization/app.py
examples/       smoking_cancer.py, education_career.py, but_for_test.py, multimodal_causal.py
sample_data/    three CSVs
tests/          test_dag.py, test_do_calculus.py, test_counterfactual.py, test_pipeline.py
visualize.py
```

## Key decisions and why
- **Fit equations from data rather than hardcode them** — makes the framework applicable to arbitrary uploaded datasets and keeps every coefficient inspectable in the UI.
- **Report identification method and formula explicitly** — users see *why* an effect is estimable, not just a number.
- **Doubly-robust estimation as the default recommendation** — robustness to misspecification of either nuisance model.
- **Multimodal encoders with graceful fallbacks** — optional heavy dependencies (sentence-transformers, torchvision, sktime) degrade to lighter methods rather than failing.
- **Dash rather than a separate JS frontend** — keeps the entire project in Python for a research-tooling audience.

## Skills demonstrated
Pearl's causal framework (do-calculus, identification, SCMs, counterfactuals, PN/PS/PNS), causal discovery algorithms, treatment-effect estimation (DR/IPW/G-formula, meta-learners, IV), multimodal representation learning, vision-LLM integration, Dash/Cytoscape interactive tooling, library design with ABCs and fallbacks, pytest.

## Relationship to other projects
Causeway is the general framework; **Threadfall** applies the do-operator to interactive fiction, **pharmacausal** applies constraint-based discovery (PC/FCI) to real FDA data, and **evidentia** applies Bayesian-network inference to diagnosis. The portfolio site's hero graphic is an interactive causal DAG built in the same spirit.
