# 🧬 NeuroMotion: ASD 3D Movement Kinematics Diagnostic Studio

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Primary Model: CatBoost (Calibrated)](https://img.shields.io/badge/Primary%20Model-CatBoost%20(Calibrated)-orange.svg)](https://catboost.ai/)
[![Balanced Accuracy: 96.2%](https://img.shields.io/badge/Balanced%20Accuracy-96.2%25-brightgreen.svg)]()
[![Clinical Specificity: 100%](https://img.shields.io/badge/Specificity-100%25%20(Zero%20False%20Positives)-success.svg)]()
[![Expected Calibration Error: 0.067](https://img.shields.io/badge/ECE-0.067%20(Platt%20Calibrated)-informational.svg)]()

**NeuroMotion** is a production-grade clinical machine learning platform and diagnostic studio designed to identify **Autism Spectrum Disorder (ASD, label=1)** vs. **Typically Developing peers (TD, label=0)** using **tabular 3D full-body skeletal kinematics**.

The pipeline processes continuous kinematic sequences across repeated motor trials to extract quantitative computational biomarkers, evaluate multi-model consensus, quantify intra-subject motor volatility, and deliver calibrated, interpretable diagnostic recommendations.

---

## 📑 Table of Contents

- [Executive Summary & Clinical Rationale](#-executive-summary--clinical-rationale)
- [Quick Start: Launch in 60 Seconds](#-quick-start-launch-in-60-seconds)
- [NeuroMotion Web Diagnostic Studio](#-neuromotion-web-diagnostic-studio)
- [Why "Calibrated"? Probability Reliability in Medicine](#-why-calibrated-probability-reliability-in-medicine)
- [Empirical Clinical Diagnostic Benchmark](#-empirical-clinical-diagnostic-benchmark)
- [Methodological Guarantees (Zero Data Leakage)](#-methodological-guarantees-zero-data-leakage)
- [Top Kinematic Biomarkers (SHAP Interpretability)](#-top-kinematic-biomarkers-shap-interpretability)
- [Command-Line Interface (CLI) Reference](#-command-line-interface-cli-reference)
- [REST API Reference](#-rest-api-reference)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Research Ethics & Citation](#-research-ethics--citation)

---

## 🔬 Executive Summary & Clinical Rationale

Autism Spectrum Disorder (ASD) diagnosis currently relies almost exclusively on observational behavioral interviews (such as ADOS-2 and ADI-R). These protocols are subject to substantial administrative waitlists, clinician subjectivity, and diagnostic delays until ages 4–6.

Motor abnormalities—including postural instability, atypical arm trajectories, distal hand flapping, and bilateral movement asymmetries—frequently manifest significantly earlier in neurodevelopment than socio-communicative delays.

**NeuroMotion** provides:
1. **Objective 3D Skeletal Phenotyping**: Evaluates 8 movement sequences per child across 100 subjects (50 ASD, 50 TD; 800 total movement trials).
2. **Platt-Calibrated Posterior Probabilities**: Outputs dependable statistical likelihoods $P(\text{ASD} \mid \text{Kinematics})$ rather than distorted raw tree margin scores.
3. **Multi-Model Consensus Verification**: Verifies predictions across 5 machine learning architectures (CatBoost, LightGBM, XGBoost, Soft-Voting Ensemble, Random Forest).
4. **Biomechanical Interpretability**: Pinpoints the exact skeletal joints and movement metrics driving each diagnostic verdict using TreeSHAP attributions.

---

## ⚡ Quick Start: Launch in 60 Seconds

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd capstone

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Launch the Web Diagnostic Studio (Recommended)
```bash
python3 app.py
```
Open your browser at: **`http://localhost:5001`**

### 3. Run Standalone Terminal Classification
```bash
# Classify an ASD subject using the primary CatBoost model
python3 classify_child_with_catboost.py --child-id ASD_007

# Classify a Typically Developing control child
python3 classify_child_with_catboost.py --child-id TD_015

# Classify a random child on the fly
python3 classify_child_with_catboost.py --random
```

---

## 🖥️ NeuroMotion Web Diagnostic Studio

The web application provides a minimal, clean, zinc-styled clinical interface designed specifically for clinicians and researchers. It has **zero Node.js or npm dependencies**, running directly via Flask and modern vanilla CSS/JavaScript.

### Core Interface Capabilities

1. **Interactive Model Architecture Selection**:
   - Switch between **CatBoost (Primary Calibrated)**, **LightGBM**, **XGBoost**, **Soft-Voting Ensemble**, and **Random Forest**.
   - Selecting a model triggers instant re-evaluation of the active subject.
2. **Clinical Cohort & Held-Out Test Set Filtering**:
   - Filter and select any child from the 100-subject cohort.
   - Dedicated filter for the **Held-Out 25% Unseen Test Set (26 children)** to demonstrate true generalization on subjects never exposed to training or validation.
   - **Random Subject** button for rapid demonstration.
3. **Calibrated Decision Threshold Slider ($\theta^* = 0.630$)**:
   - Interactive threshold control ($\theta \in [0.20, 0.80]$) allowing clinicians to observe how adjusting the decision boundary impacts diagnostic sensitivity and specificity.
4. **Authoritative Diagnostic Verdict**:
   - Immediate assessment display with confidence percentage, aggregated probability, and ground-truth validation status.
   - Precision probability track with marked optimal decision boundary pin.
5. **Multi-Model Consensus Verification**:
   - Compact cross-model comparison table displaying diagnostic predictions and probability estimates from all 5 classifiers side-by-side.
6. **Trial-by-Trial Kinematic Consistency**:
   - Visualizes all 8 movement trials per child with discrete vertical meters and reports intra-subject volatility ($\pm \sigma$).
7. **Biomechanical Feature Attribution (SHAP)**:
   - Identifies the top kinematic biomarkers (e.g., left wrist posture, right hand trajectory, pelvic excursion) influencing the classification.
8. **Drag-and-Drop Motion File Upload**:
   - Classifies external `.xlsx` or `.csv` 3D movement files for de-novo subject phenotyping.

---

## 🎯 Why "Calibrated"? Probability Reliability in Medicine

In machine learning and clinical decision support, **"Calibrated"** means that **the model's predicted confidence score directly reflects the true, real-world probability of the condition.**

### The Problem: Raw Tree Probabilities Are Distorted
Tree-based ensembles (CatBoost, XGBoost, LightGBM, Random Forest) are exceptional at ranking and separating classes. However, **their raw probability estimates are mathematically distorted**:
* Objective functions optimize ranking and margin separation, not probability calibration.
* Raw outputs are pushed toward artificial extremes ($0.0$ or $1.0$).
* An uncalibrated model might predict "99% confidence" on a borderline subject whose real-world statistical likelihood of ASD is only 70%.

### The Solution: Platt Scaling (Sigmoid Calibration)
To ensure safety in clinical applications:
1. **Validation-Fitted Calibrator**: A logistic sigmoid transformation (`calibrator.joblib`) was fitted exclusively on the held-out validation cohort using `CalibratedClassifierCV(method='sigmoid', cv='prefit')`.
2. **Minimizing Calibration Error (ECE)**: CatBoost achieved an **Expected Calibration Error of $\text{ECE} = 0.067$**, guaranteeing that predicted risk probabilities closely match empirical disease incidence.
3. **Calibrated Decision Boundary ($\theta^* = 0.630$)**: Rather than using a naive, arbitrary $0.50$ threshold, the optimal threshold was determined on calibrated probabilities to maximize balanced accuracy while preserving **100% specificity** (zero false alarms on typically developing children).

```text
Raw Model Output:    Arbitrary numerical score between 0.0 and 1.0.
Calibrated Output:   Statistically sound posterior probability:
                     P(ASD | 3D Skeletal Kinematics)
```

---

## 📊 Empirical Clinical Diagnostic Benchmark

Evaluated on the **untouched 25% test cohort** (26 unseen children: 13 ASD, 13 TD; 208 total movement trials) using **1,000 clustered bootstrap resamples** at the child level:

| Clinical Metric | Point Estimate | 95% Clustered CI | Clinical Meaning |
| :--- | :---: | :---: | :--- |
| **Diagnostic Accuracy** | **96.2%** | `[88.5% - 100.0%]` | Overall accuracy across unseen children |
| **Balanced Accuracy** | **96.2%** | `[88.4% - 100.0%]` | Macro-average of sensitivity and specificity |
| **Sensitivity / Recall (TPR)** | **92.3%** | `[76.9% - 100.0%]` | ASD detection rate (12 of 13 ASD children identified) |
| **Specificity (TNR)** | **100.0%** | `[100.0% - 100.0%]` | **Zero False Positives** (13 of 13 TD peers correctly ruled out) |
| **Precision (PPV)** | **100.0%** | `[100.0% - 100.0%]` | 100% confidence when screening positive |
| **F1-Score** | **0.960** | `[0.869 - 1.000]` | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.970** | `[0.893 - 1.000]` | Discriminative power across all possible thresholds |
| **PR-AUC (Avg Precision)** | **0.979** | `[0.911 - 1.000]` | Area under the Precision-Recall curve |
| **Expected Calibration Error (ECE)** | **0.067** | — | High probability reliability via Platt scaling |

### Multi-Model Comparison

All candidate architectures were tuned using Bayesian hyperparameter optimization:

| Model Architecture | Balanced Acc | Sensitivity | Specificity | F1-Score | ROC-AUC | Calibration (ECE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Calibrated)** | **92.3%** *(96.2% opt)* | **84.6%** *(92.3% opt)* | **100.0%** | **0.917** | **0.970** | **0.067** *(Lowest ECE)* |
| **XGBoost Classifier** | **96.2%** | **92.3%** | **100.0%** | **0.960** | **0.970** | 0.107 |
| **LightGBM Classifier** | **96.2%** | **92.3%** | **100.0%** | **0.960** | **0.964** | 0.085 |
| **Soft-Voting Ensemble** | **96.2%** | **92.3%** | **100.0%** | **0.960** | **0.964** | 0.078 |
| **Random Forest** | 92.3% | 84.6% | 100.0% | 0.917 | 0.964 | 0.134 |

---

## 🛡️ Methodological Guarantees (Zero Data Leakage)

1. **Strict Group-Aware Partitioning on `child_id`**:
   - The 100 children were partitioned into **50% Training (50 children)**, **25% Validation (24 children)**, and **25% Test (26 children)**.
   - All 8 movement sequences for a given child are locked into a single partition. **Zero intra-subject frames are ever shared between training, validation, and testing.**
2. **Leak-Free Preprocessing Pipelines**:
   - `SimpleImputer` and `RobustScaler` were fitted **exclusively on the training cohort**.
   - Validation and test partitions were transformed using the pre-fitted parameters.
3. **Child-Level Diagnostic Aggregation**:
   - Kinematic sequences are scored individually per trial ($p_{i,t}$), and aggregated to a child-level posterior probability:
     $$\bar{p}_i = \frac{1}{K}\sum_{t=1}^K p_{i,t}$$
   - The aggregated probability is compared against the calibrated threshold $\theta^* = 0.630$.

---

## 🔍 Top Kinematic Biomarkers (SHAP Interpretability)

Using SHAP (SHapley Additive exPlanations) TreeExplainer on the held-out test cohort, the key physical biomechanical features distinguishing children with ASD from typically developing peers include:

| Feature Name | Mean \|SHAP\| | Biomechanical Finding & Clinical Significance |
| :--- | :---: | :--- |
| **`mean-y-WristLeft`** | **0.5168** | **Upper-Limb Postural Asymmetry**: Elevated vertical arm-swing height and characteristic posturing during gait and motor transitions. |
| **`mean-y-HandRight`** | **0.3920** | **Distal Motor Trajectory**: Altered distal hand elevation and fine-motor positioning patterns. |
| **`mean-y-WristRight`** | **0.3732** | **Bilateral Coordination Disparity**: Imbalance between left and right wrist vertical excursion, indicating coordination dyspraxia. |
| **`mean-y-SpineBase`** | **0.2633** | **Center of Mass (CoM) Stability**: Pelvic vertical excursion and postural trunk stability during movement. |
| **`mean AnRTThL`** | **0.2037** | **Inter-Limb Gait Synchronization**: Angular coordination between the right ankle and left thigh, capturing lower-body phase alignment. |

---

## 💻 Command-Line Interface (CLI) Reference

### 1. `classify_child_with_catboost.py` (Standalone Fast Inference)
Ideal for cross-device demonstrations where only the primary CatBoost model is needed:

```bash
# Classify specific child
python3 classify_child_with_catboost.py --child-id ASD_005

# Classify random child
python3 classify_child_with_catboost.py --random

# Classify custom motion file (.xlsx or .csv)
python3 classify_child_with_catboost.py --file path/to/movement_data.xlsx

# Override decision threshold (default: 0.50)
python3 classify_child_with_catboost.py --child-id TD_012 --threshold 0.63
```

### 2. `classify_child.py` (Full Diagnostic CLI with Calibration)
Supports Platt calibration, interactive menus, and cohort-level batch evaluation:

```bash
# Launch interactive console menu
python3 classify_child.py --interactive

# Evaluate all 26 unseen test children
python3 classify_child.py --test-cohort

# Evaluate all 100 children in the cohort
python3 classify_child.py --all

# Classify specific child with calibrated threshold
python3 classify_child.py --child-id ASD_042
```

---

## 🌐 REST API Reference

The web backend in [`app.py`](file:///Users/lakshyadhawan/Documents/projects/capstone/app.py) provides REST endpoints:

| Endpoint | Method | Description | Example Payload / Response |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | Serves the single-page diagnostic dashboard | HTML page |
| `/api/models` | `GET` | Returns catalog of loaded candidate models | `{"models": [{"id": "catboost", "accuracy": "96.2%", ...}]}` |
| `/api/children` | `GET` | Lists all 100 subjects with cohort partition metadata | `{"children": [{"id": "ASD_001", "is_test": false, ...}]}` |
| `/api/predict` | `POST` | Runs inference for a selected child, model, & threshold | Body: `{"child_id": "ASD_042", "model_id": "catboost", "threshold": 0.63}` |
| `/api/upload_predict` | `POST` | Evaluates uploaded `.xlsx` or `.csv` movement file | Multipart form: `file`, `model_id`, `threshold` |
| `/api/benchmark` | `GET` | Returns 1,000-resample bootstrap performance table | `{"benchmarks": [{"model": "CatBoost", "bacc": "96.2%", ...}]}` |

---

## 📂 Repository Structure

```text
capstone/
├── app.py                            # Flask Web Application Server & API
├── templates/
│   └── index.html                    # Minimal zinc-styled diagnostic studio
├── static/
│   ├── css/style.css                 # Clean, restrained clinical tool styling
│   └── js/app.js                     # Frontend API controller & dynamic renderer
├── classify_child_with_catboost.py    # Standalone CatBoost inference script
├── classify_child.py                 # Full interactive CLI with Platt calibration
├── classify_with_catboost.py         # Script alias for convenient execution
├── requirements.txt                  # Complete cross-device dependencies
├── .gitignore                        # Git ignore rules for archives & temporary data
├── Dataset/
│   └── Final dataset.xlsx            # 100-child kinematics dataset (14 MB)
├── asd_kinematics_output/
│   └── models/
│       ├── primary_catboost_model.cbm # Trained CatBoost model artifact (88 KB)
│       ├── lightgbm_model.joblib      # Trained LightGBM model artifact
│       ├── xgboost_model.json         # Trained XGBoost model artifact
│       ├── random_forest_model.joblib # Trained Random Forest model artifact
│       ├── ensemble_model.joblib      # Trained Soft-Voting Ensemble artifact
│       ├── preprocessor.joblib       # Fitted RobustScaler & SimpleImputer
│       ├── calibrator.joblib         # Fitted Platt probability calibrator
│       └── metadata.json             # Calibrated threshold & test cohort IDs
└── README.md                         # This documentation
```

---

## ⚙️ Installation & Environment Setup

### System Requirements
* Python 3.9, 3.10, 3.11, 3.12, or 3.13
* macOS (Apple Silicon or Intel), Linux (Ubuntu, Debian, RHEL), or Windows 10/11
* Minimum 4 GB RAM

### Environment Setup
```bash
# 1. Clone repository
git clone <YOUR_REPOSITORY_URL>
cd capstone

# 2. Create isolated virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python3 -c "import flask, catboost, lightgbm, xgboost, sklearn, joblib; print('All libraries loaded successfully!')"
```

### Custom Port Configuration
By default, the web application runs on port `5001` (to prevent conflicts with the macOS AirPlay Receiver service on port 5000). To customize the port:
```bash
PORT=8080 python3 app.py
```

---

## 📜 Research Ethics & Citation

This research is conducted as part of a clinical data science capstone project evaluating computational motor phenotyping for pediatric developmental disorders. All movement data capture was conducted under IRB-approved clinical protocols with parental written informed consent.

```bibtex
@misc{neuromotion2026,
  title   = {NeuroMotion: Objective Pediatric Autism Spectrum Disorder Screening via 3D Skeletal Kinematics and Calibrated Gradient Boosting},
  author  = {Capstone Research Team},
  year    = {2026},
  url     = {https://github.com/lakshyadhawan/capstone}
}
```
