# ANNITIA — MASLD Risk Stratification through Longitudinal NIT Trajectories

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-survival 0.27](https://img.shields.io/badge/scikit--survival-0.27.0-green.svg)](https://scikit-survival.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Abdourahamane Ide Salifou  
**Challenge:** ANNITIA Data Challenge — IHU ICAN × Trustii.io · France 2030 · Health Data Hub  
**Endpoint:** `0.7 × C-index(hepatic event) + 0.3 × C-index(death)`  
**Private leaderboard:** **0.9479**

---

## The Challenge

Metabolic dysfunction-associated steatotic liver disease (MASLD) affects ~30% of the global population and is the leading cause of chronic liver disease. Its progression from simple steatosis through fibrosis (F0–F4) to cirrhosis and liver-related death varies widely across patients — making early, personalised risk stratification a critical clinical need.

The **ANNITIA challenge** asks: given repeated measurements of non-invasive tests (NITs — FibroScan, FibroTest, Aixplorer) and biochemical markers collected over up to 22 visits per patient, can we build an AI model that correctly ranks patients by their risk of (1) major hepatic events and (2) all-cause mortality?

The evaluation metric is the **weighted concordance index**: `Score = 0.7 × C-index(hepatic) + 0.3 × C-index(death)`, applied to 1,676 synthetic patients derived from real clinical records at Pitié-Salpêtrière Hospital, Paris (2001–2025).

---

## Our Solution

The central clinical hypothesis is that **the rate of NIT change is more prognostic than any single cross-sectional reading**. A patient whose FibroTest rises from 0.3 to 0.65 in 2 years is fundamentally higher-risk than one who has been stable at 0.65 for a decade — yet a snapshot model sees the same value.

We operationalise this through three tightly coupled layers:

### Layer 1 — Trajectory Feature Engineering (~174 features)

Every repeated NIT and biochemical marker is summarised not just by its latest value, but by its full dynamic profile:

- **Velocity descriptors** — OLS slope (kPa/year), early vs. late acceleration, recent 3-visit slope
- **Cumulative exposure** — trapezoidal AUC integrals (e.g. cirrhosis burden ∫ max(LSM − 13, 0) dt)
- **Rate interactions** — `FibroTest_max × (1/follow-up)` and `GGT_max × (1/follow-up)`, encoding biological activity per unit time. The former achieves C = 0.715 as a *single feature*, nearly matching static FibroScan max (C = 0.766).
- **Fibrosis-stage crossings** — F2/F3/F4 ever, at last visit, and rapid F0→F4 progression flags
- **Validated composite scores** — FIB-4, APRI, NFS, FAST, and Agile 3+ computed per visit and as longitudinal trajectories

### Layer 2 — Diverse Survival Ensemble

Four complementary survival learners are trained on each endpoint with 5-fold stratified cross-validation, then blended via rank-normalised, OOF-tuned weights:

| Model | Family | Role |
|-------|--------|------|
| **GBSA** — Gradient Boosting Survival Analysis | Additive trees | Captures non-linear NIT × covariate interactions |
| **RSF** — Random Survival Forest | Bagged trees | Robust to noisy features and irregular missingness |
| **CoxNet** | Sparse L1/L2 Cox | Produces a directly interpretable, drift-resistant coefficient vector |
| **XGBoost-Cox** | Gradient boosting (survival:cox) | High-capacity ranking with SHAP-native interpretability |

Blend weights are tuned by exhaustive grid search (~6,000 combinations) on held-out OOF predictions, directly optimising the C-index objective. This is leakage-free by design.

### Layer 3 — Phenotype Risk Composer (PRC)

A cross-sectional ML ensemble may rank a **stable compensated-F4 patient** (12 years, no events, low GGT) *above* a **rapid F2→F4 inflammatory progressor** (2 years, high GGT, ascites at year 3) — simply because absolute LSM is higher in Patient A. The PRC corrects this **stage-velocity discordance** through fixed, auditable clinical phenotype terms:

```
risk_hepatic = 0.70 × base_ML + 0.18 × PRC_score
             + 0.08 × rank(F0-F2 × GGT_max) + 0.04 × rank(F3 / follow-up)

risk_death   = 0.95 × rank(1/follow-up) + 0.05 × base_ML
```

The PRC sub-scores (short follow-up, few visits, early-fibrosis-with-high-GGT, F3-rapid-progression, FT/FibroScan discordance, stable-F4 discount) are each grounded in EASL-ALEH clinical guidelines and carry fixed, inspectable weights — no black box.

The death composer weights `1/follow-up` at 0.95 because with 94% censoring, a patient observed 15 years without dying is with near-certainty lower-risk than one who died at year 2. `1/follow-up` alone achieves **C = 0.969** on the training set — a legitimate, organiser-confirmed signal.

---

## Results

| Metric | Value |
|--------|-------|
| **Private leaderboard composite** | **0.9479** |
| Public leaderboard composite | 0.9511 |
| Hepatic OOF C-index (5-fold) | 0.852 ± 0.080 |
| Death OOF C-index (5-fold) | 0.967 ± 0.009 |
| `FibroTest × 1/follow-up` — single feature C-index (train) | 0.715 |
| `1/follow-up` alone — death C-index | 0.969 |

---

## Repository Structure

```
├── ANNITIA_survival_ensemble.ipynb               ← main solution notebook (run this)
├── ANNITIA_Final_Report.pdf             ← 22-page methodology report
├── submission_upgrade_with_composer.csv ← best private LB submission (0.9479)
├── requirements.txt
├── README.md
└── analysis/                            ← optional: scripts to reproduce figures
    ├── analysis_suite.py                #   SHAP, OOF stability, leakage, SurvSHAP(t)
    └── clinical_figures.py             #   Kaplan-Meier, Cox forest plot, calibration
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/SalifouAbdourahamane/annitia_challenge.git
cd annitia_challenge
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place data files

Copy `train_data.csv` and `test_data.csv` into the repository root (same directory as the notebook).

### 4. Run the notebook

```bash
jupyter notebook Annitia_survival_ensemble.ipynb
```

Then in the browser: **Kernel → Restart & Run All**.

Alternatively with JupyterLab:

```bash
jupyter lab Annitia_survival_ensemble.ipynb
```

Or execute headlessly from the terminal:

```bash
jupyter nbconvert --to notebook --execute Annitia_survival_ensemble.ipynb --output ANNITIA_solution_executed.ipynb
```

The notebook will engineer features, train all four survival models, apply the PRC, and write:

```
submission_upgrade_with_composer.csv   (423 rows · trustii_id · risk_hepatic_event · risk_death)
```

> **Runtime:** ~20–35 min on a standard CPU · 8 GB RAM recommended · `RANDOM_STATE = 42` throughout

### 5. (Optional) Reproduce interpretability figures

```bash
python analysis/analysis_suite.py    # SHAP beeswarm, OOF stability, leakage breakdown
python analysis/clinical_figures.py  # Kaplan-Meier, Cox forest plot, calibration plot
```

Figures are written to `figures/` and `figures_clinical/`. Both scripts require the same `train_data.csv` and `test_data.csv` in the working directory.

---

## Requirements

```
matplotlib==3.10.0
pandas==2.2.2
scikit-learn==1.8.0
scikit-survival==0.27.0
scipy==1.16.3
shap==0.51.0
xgboost==3.2.0
lifelines>=0.29.0
```

---

## Reproducibility

- Global seed: `RANDOM_STATE = 42` — governs NumPy, scikit-learn, scikit-survival, and XGBoost
- All preprocessing transformers fitted on training folds only; identical schema applied to test
- No outcome information enters feature construction at any stage
- Running all cells top to bottom deterministically regenerates the submission from scratch

---

## Documentation

The full 22-page **`ANNITIA_Final_Report.pdf`** covers:

- Dataset exploration — cohort overview, biomarker distributions, missing data structure
- Complete feature engineering specification with clinical literature references
- Survival model architecture, ensemble strategy, and OOF validation stability
- Phenotype Risk Composer design with per-sub-score hepatology rationales
- Clinical interpretability — SHAP, SurvSHAP(t), Kaplan-Meier, univariable Cox forest plot, calibration
- Post-event leakage analysis (26/47 event patients, mean 60% post-event follow-up) and informed censoring treatment
- 10 peer-reviewed MASLD references (EASL-ALEH 2021, Sterling 2006, Newsome 2020, Agarwal 2024, and others)
