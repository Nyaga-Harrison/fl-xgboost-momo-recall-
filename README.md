# FL-XGBoost: Focal-Loss XGBoost for Minority-Class Default Recall

An engineered focal-loss objective function fabricated natively into XGBoost's
gradient-boosting loop, designed to improve minority-class default recall on
extremely imbalanced mobile-money credit datasets in sub-Saharan Africa.

This repository accompanies the research proposal *"An Engineered Focal-Loss
XGBoost Model (FL-XGBoost) for Enhancing Minority-Class Default Recall in Mobile
Money Datasets under Extreme Class Imbalance"* (Nyaga Harrison Mawutor, KNUST,
Computer Science).

## Problem

Automated mobile-money credit-scoring systems optimise global accuracy on data
with extreme class skew (roughly 98% non-default to 2% default). Standard
log-loss lets the large non-default majority dominate the gradient, so rare
default events are treated as background noise and thin-file borrowers are
wrongly denied access. FL-XGBoost addresses this at the loss level rather than
by distorting the data with resampling.

## Engineering contribution

The core contribution is a **fabricated focal-loss objective** integrated
directly into XGBoost via a custom first derivative (gradient, gᵢ) and second
derivative (Hessian, hᵢ). A modulating factor (1 − pₜ)^γ down-weights
easy-to-classify, non-default operations and forces the trees to focus on hard,
underrepresented default patterns — without external sampling such as SMOTE.

Focal loss:

    L = −y·α·(1−p)^γ·log(p) − (1−y)·(1−α)·p^γ·log(1−p)

Default hyperparameters: γ = 2.0, α = 0.25, learning rate η = 0.05.

## Datasets

- **PaySim** (simulated mobile-money transaction logs, Kaggle) — large-scale
  structural baseline.
- **African Credit Scoring Challenge** (Zindi, 2025) — real sub-Saharan loan
  records with genuine default outcomes, accessed under the challenge's
  data-use terms.

> Note: dataset files are not redistributed in this repository. PaySim is
> available on Kaggle; the Zindi dataset must be obtained from the competition
> page under its own terms of use.

## Experimental design

1. Build and **lock** an unmodified XGBoost baseline first, on fixed data
   splits (seed = 422026).
2. Train the engineered FL-XGBoost in a separate notebook on identical splits.
3. Compare formally on the same folds: Accuracy, Macro F1, AUC-ROC, AUC-PR,
   training time — with a Wilcoxon signed-rank test and reported delta.

Baselines: standard unengineered XGBoost, SMOTE + XGBoost, Balanced Random
Forest.

## Repository structure (planned)

    ├── README.md
    ├── requirements.txt
    ├── data/                # not tracked — see dataset notes above
    ├── notebooks/
    │   ├── 01_baseline_xgboost.ipynb
    │   ├── 02_fl_xgboost.ipynb
    │   └── 03_comparison_and_shap.ipynb
    └── src/
        ├── focal_objective.py   # custom gradient/Hessian focal-loss objective
        └── evaluate.py          # metrics + Wilcoxon test

## Setup

    git clone https://github.com/Nyaga-Harrison/fl-xgboost-momo-recall.git
    cd fl-xgboost-momo-recall
    pip install -r requirements.txt

## Reproducibility

All runs use a fixed seed (422026), 5-fold stratified cross-validation, and the
pinned library versions in `requirements.txt`. The custom objective computes gᵢ
and hᵢ analytically with Hessian clipping to keep second derivatives positive
for stable tree splitting.

## Citation

A Zenodo DOI will be registered for the released codebase via the
GitHub–Zenodo integration. Citation details will be added here once minted.

## License

To be added (MIT recommended for academic code release).
