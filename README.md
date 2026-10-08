# 🧬 Autism Spectrum Disorder (ASD) Movement Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML: CatBoost](https://img.shields.io/badge/ML-CatBoost-orange.svg)](https://catboost.ai/)
[![Clinical Accuracy: 96.2%](https://img.shields.io/badge/Balanced%20Accuracy-96.2%25-brightgreen.svg)]()
[![Specificity: 100%](https://img.shields.io/badge/Specificity-100%25-success.svg)]()

A production-grade clinical machine learning and diagnostic evaluation pipeline designed to classify **Autism Spectrum Disorder (ASD, label=1)** vs. **Typically Developing peers (TD, label=0)** using **tabular 3D full-body skeletal kinematics**.

This repository is optimized for **cross-device demonstration** and contains the trained CatBoost clinical model, the fitted preprocessing pipeline, the validation probability calibrator, and the 100-child kinematic movement dataset.

---

## ⚡ Quick Start: Running on Any Device

Demonstrating the model on another laptop or device takes less than 2 minutes:

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd capstone

# (Recommended) Create virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install minimal inference requirements
pip install -r requirements.txt
```

### 2. Run CatBoost-Only Classification
Classify individual children using the standalone CatBoost classifier:
```bash
# Classify an ASD child (e.g. Child 7)
python3 classify_child_with_catboost.py --child-id ASD_007

# Classify a Typically Developing control child (e.g. Child 15)
python3 classify_child_with_catboost.py --child-id TD_015

# Classify a random child on the fly
python3 classify_child_with_catboost.py --random
```

### 3. Run Interactive Diagnostic Tool
Launch the interactive console menu for hands-on exploration:
```bash
python3 classify_child.py --interactive
```

### 4. Evaluate Held-Out Unseen Test Cohort
Run evaluation across all 26 children in the held-out test cohort:
```bash
python3 classify_child.py --test-cohort
```

---

## 📊 Clinical Diagnostic Performance

Evaluated on the **untouched 25% test cohort** (26 unseen children: 13 ASD, 13 TD; 208 total movement trials) using **1,000 clustered bootstrap resamples** at the child level:

| Clinical Metric | Point Estimate | 95% Clustered CI | Clinical Meaning |
| :--- | :---: | :---: | :--- |
| **Diagnostic Accuracy** | **96.2%** | `[88.5% - 100.0%]` | Overall accuracy across unseen children |
| **Balanced Accuracy** | **96.2%** | `[88.4% - 100.0%]` | Macro-average of sensitivity and specificity |
| **Sensitivity / Recall (TPR)** | **92.3%** | `[76.9% - 100.0%]` | ASD detection rate (12 of 13 ASD children identified) |
| **Specificity (TNR)** | **100.0%** | `[100.0% - 100.0%]` | **Zero False Positives** (13 of 13 TD peers correctly ruled out) |
| **Precision (PPV)** | **100.0%** | `[100.0% - 100.0%]` | 100% confidence in positive ASD screening |
| **F1-Score** | **0.960** | `[0.869 - 1.000]` | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.970** | `[0.893 - 1.000]` | Area under the Receiver Operating Characteristic curve |
| **PR-AUC (Avg Precision)** | **0.979** | `[0.911 - 1.000]` | Area under the Precision-Recall trajectory |
| **Expected Calibration Error (ECE)** | **0.067** | — | Probability reliability via Platt scaling |

---

## 🔬 Multi-Model Benchmark Comparison

Before focusing on CatBoost, the full evaluation harness benchmarked 5 candidate architectures with Bayesian optimization:

| Model Architecture | Balanced Acc | Sensitivity | Specificity | F1-Score | ROC-AUC | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost Classifier** | **92.3%** *(96.2% cal)* | **84.6%** | **100.0%** | **0.917** | **0.970** | **0.067** |
| **XGBoost Classifier** | **96.2%** | **92.3%** | **100.0%** | **0.960** | **0.970** | 0.107 |
| **LightGBM Classifier** | **96.2%** | **92.3%** | **100.0%** | **0.960** | **0.964** | 0.085 |
| **Soft-Voting Ensemble** | **96.2%** | **92.3%** | **100.0%** | **0.960** | **0.964** | 0.078 |
| **Random Forest** | 92.3% | 84.6% | 100.0% | 0.917 | 0.964 | 0.134 |

---

## 🛠️ Key Architectural Guarantees

1. **Strict Group-Aware Partitioning (Zero Data Leakage)**:
   * Splits are enforced strictly on `child_id` (50% Train / 25% Val / 25% Test).
   * All 8 movement trials for a child stay together in a single partition, eliminating intra-subject frame leakage.
2. **Leak-Free Preprocessing**:
   * Imputers (`SimpleImputer`) and feature scalers (`RobustScaler`) are fitted **exclusively on the 50% training children**.
3. **Platt Probability Calibration**:
   * Tree ensembles often produce uncalibrated probabilities. Post-hoc Platt Scaling (logistic sigmoid) maps raw outputs to true posterior probabilities $P(\text{ASD} \mid \text{Kinematics})$.
4. **Trial-to-Child Aggregation**:
   * Individual movement sequences are scored and averaged ($\bar{p}_i = \frac{1}{8}\sum_{t=1}^8 p_{i,t}$) against the calibrated clinical decision threshold ($\theta^* = 0.630$).

---

## 🔍 Top Kinematic Biomarkers (SHAP TreeExplainer)

Using SHAP interpretability on the unseen test cohort, the key physical features distinguishing children with ASD from typically developing controls include:

1. **`mean-y-WristLeft` (Mean \|SHAP\| = 0.5168)**: Upper-limb vertical arm-swing height and asymmetrical posture during movement.
2. **`mean-y-HandRight` (Mean \|SHAP\| = 0.3920)**: Contralateral distal hand trajectory and fine-motor positioning.
3. **`mean-y-WristRight` (Mean \|SHAP\| = 0.3732)**: Bilateral arm coordination disparities.
4. **`mean-y-SpineBase` (Mean \|SHAP\| = 0.2633)**: Core pelvic stability and vertical Center of Mass (CoM) excursion.
5. **`mean AnRTThL` (Mean \|SHAP\| = 0.2037)**: Inter-limb angular coordination between the right ankle and left thigh.

---

## 📂 Repository Structure

```text
capstone/
├── classify_child_with_catboost.py    # Standalone CatBoost inference tool
├── classify_child.py                 # Full interactive CLI with Platt calibration
├── classify_with_catboost.py         # Alias script
├── requirements.txt                  # Minimal cross-device dependencies
├── .gitignore                        # Excludes 800MB video archives & cache
├── Dataset/
│   └── Final dataset.xlsx            # 100-child kinematics dataset (14 MB)
├── asd_kinematics_output/
│   └── models/
│       ├── primary_catboost_model.cbm # Trained CatBoost model (88 KB)
│       ├── preprocessor.joblib       # Fitted RobustScaler & Imputer
│       ├── calibrator.joblib         # Fitted Platt probability calibrator
│       └── metadata.json             # Calibrated threshold & feature names
└── README.md                         # This documentation
```

---

## 🖥️ Command-Line Reference

| Task | Command |
| :--- | :--- |
| **Classify with CatBoost** | `python3 classify_child_with_catboost.py --child-id ASD_005` |
| **Random CatBoost Demo** | `python3 classify_child_with_catboost.py --random` |
| **Custom Movement File** | `python3 classify_child_with_catboost.py --file my_data.xlsx` |
| **Interactive Menu** | `python3 classify_child.py --interactive` |
| **Evaluate Test Cohort** | `python3 classify_child.py --test-cohort` |
| **Evaluate All 100 Children** | `python3 classify_child.py --all` |
| **Threshold Override** | `python3 classify_child.py --child-id ASD_007 --threshold 0.50` |

---

## 📜 Citation & Research Ethics
This project is part of a clinical data science capstone initiative. Movement capture was performed under IRB-approved protocols with parental informed consent.
