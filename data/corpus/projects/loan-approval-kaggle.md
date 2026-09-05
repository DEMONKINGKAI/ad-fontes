---
doc_type: project
project_id: loan-approval
name: "Loan Approval Prediction — Dataset and Model (Kaggle)"
tagline: "A clean, reproducible benchmark for credit risk ML."
status: published
repo: null (private code; public Kaggle artifacts)
links:
  dataset: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
  notebook: https://www.kaggle.com/code/architsharma01/loan-approval-prediction
recognition: Kaggle Silver Medal (dataset)
domain: [data science, credit risk, classical ML, dataset engineering]
stack: [Python, pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn, Kaggle]
origin: ML & Data Analytics internship at Axisray (Jun–Jul 2023)
---

# Loan Approval Prediction (Kaggle)

## Two artifacts
**1. Dataset.** Most public loan datasets are poorly documented, synthetic, or lack feature diversity. This one was engineered from real-world loan application data across applicant demographics, income, credit history, debt ratios, and collateral, with missing-value imputation, outlier removal, and class balancing, then published on Kaggle with full documentation, feature descriptions, and a reproducible preprocessing pipeline. It earned a Kaggle Silver Medal.

**2. Model notebook.** Six classifiers — Logistic Regression, Random Forest, XGBoost, SVM, KNN, Decision Tree — benchmarked on the dataset, each tuned by cross-validation over hyperparameter grids, with feature-importance analysis, precision/recall/F1 comparison, confusion matrices, and ROC curves, so it is transparent where each algorithm wins and fails on credit data. Emphasis on recall for the minority (rejection) class.

## Context
Built during the Axisray internship, where the broader work included VAEs and GANs for synthetic augmentation of under-represented applicant profiles and a speech-recognition prototype for voice-driven data entry.

## Skills demonstrated
Dataset engineering and documentation, class-imbalance handling, supervised model benchmarking and tuning, model transparency and evaluation, publishing reproducible work.
