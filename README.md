# 🧬 ASD Movement Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML: CatBoost](https://img.shields.io/badge/ML-CatBoost-orange.svg)](https://catboost.ai/)

A high-precision clinical machine learning pipeline designed to classify **Autism Spectrum Disorder (ASD)** using 3D full-body skeletal kinematics. By analyzing movement biomechanics, this project distinguishes between children with ASD and typically developing (TD) peers using gradient-boosted decision trees and SHAP-based interpretability.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd capstone

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn \
    catboost lightgbm xgboost optuna shap openpyxl joblib
```

### 2. Train the Model
Run the full end-to-end pipeline:
```bash
python3 asd_movement_classification_harness.py
```
*This will handle data loading, feature engineering, Bayesian tuning, training, calibration, and visualization.*

### 3. Classify a Child
Use the interactive inference CLI:
```bash
python3 classify_child.py --interactive
```

---

## 🛠️ The Pipeline

The system implements a rigorous diagnostic evaluation harness to ensure clinical validity and zero data leakage.

### 🔄 Process Flow
`Data Ingestion` $\rightarrow$ `Feature Extraction` $\rightarrow$ `Group-Aware Split` $\rightarrow$ `Leak-Free Preprocessing` $\rightarrow$ `Bayesian Optimization` $\rightarrow$ `Model Training` $\rightarrow$ `Probability Calibration` $\rightarrow$ `Clinical Evaluation` $\rightarrow$ `SHAP Interpretation`

### 🔬 Key Technical Features
- **Biomechanical Feature Set**: Extracts 100+ features including joint angles, inter-joint distances, 3D velocities, acceleration, jerk, and bilateral asymmetry indices.
- **Strict Group Partitioning**: Splits are performed at the **child level** (50% Train / 25% Val / 25% Test) to prevent frames from the same subject appearing in multiple sets.
- **Probabilistic Calibration**: Uses Platt scaling and Isotonic regression to transform raw model scores into clinically interpretable probabilities.
- **Robust Evaluation**: Employs child-level clustered bootstrap 95% confidence intervals for sensitivity, specificity, and ROC-AUC.
- **Explainability**: Integration of `SHAP TreeExplainer` to identify the specific kinematic biomarkers driving the classification.

---

## 📂 Project Architecture

```text
capstone/
├── asd_movement_classification_harness.py   # Core ML pipeline & evaluation
├── classify_child.py                        # Inference CLI
├── Dataset/                                 # Raw data & documentation
│   ├── Final dataset.xlsx                   # Master dataset (100 children)
│   ├── Autism/                              # ASD recordings
│   └── Typical/                            # TD recordings
├── asd_kinematics_output/                   # Model artifacts & results
│   ├── models/                              # .cbm models, scalers, calibrators
│   ├── figures/                             # ROC, Calibration & SHAP plots
│   └── tables/                              # Metrics with Confidence Intervals
└── README.md
```

---

## 📊 The Cohort

| Group | Children | Trials/Child | Total Trials |
| :--- | :---: | :---: | :---: |
| **ASD** | 50 | 8 | 400 |
| **TD** | 50 | 8 | 400 |
| **Total** | **100** | **8** | **800** |

*Data captured at 30 Hz across 16 major body joints.*

---

## ⚙️ Advanced Usage

| Command | Description |
| :--- | :--- |
| `python3 classify_child.py --child-id <ID>` | Classify a specific subject (e.g., `ASD_005`) |
| `python3 classify_child.py --test-cohort` | Evaluate the unseen held-out test set |
| `python3 classify_child.py --all` | Generate reports for the entire cohort |
| `python3 classify_child.py --threshold 0.6` | Override default decision boundary |

---

## 📜 License
This project is part of a capstone research initiative. Please cite appropriately when using these methods for clinical research.
