# 👁️ UncertaCat: Paired Clinical Cataract Screening & Unsupervised Domain Adaptation

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue)](https://huggingface.co/spaces/UmarIqbal000/uncertacat-cataract-screening)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.17%2B-orange.svg)](https://pytorch.org/vision/)
[![Gradio](https://img.shields.io/badge/Gradio-6.x-brightgreen.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic clinical decision-support pipeline for **Retinal Cataract Screening, Opacification Severity Grading, and Epistemic Uncertainty Quantification**, featuring **Two-Stage Unsupervised Hospital Domain Adaptation (UDA)** to overcome camera and sensor distribution shifts across medical centers.

---

## 🌐 Public Interactive Cloud Application
* **Hugging Face Space:** [https://huggingface.co/spaces/UmarIqbal000/uncertacat-cataract-screening](https://huggingface.co/spaces/UmarIqbal000/uncertacat-cataract-screening)
* **Direct Fullscreen Interface:** [https://umariqbal000-uncertacat-cataract-screening.hf.space](https://umariqbal000-uncertacat-cataract-screening.hf.space)

---

## 📑 Table of Contents
1. [Project Lifecycle & Clinical Motivation](#1-project-lifecycle--clinical-motivation)
2. [Datasets & Experimental Cohorts](#2-datasets--experimental-cohorts)
3. [End-to-End Image Preprocessing Pipeline](#3-end-to-end-image-preprocessing-pipeline)
4. [Evolutionary Trajectory: Iterations v1 to v6](#4-evolutionary-trajectory-iterations-v1-to-v6)
5. [Master Evaluation Results & Benchmark Analysis](#5-master-evaluation-results--benchmark-analysis)
6. [Glossary of Acronyms & Technical Terminology](#6-glossary-of-acronyms--technical-terminology)
7. [Cloud Deployment Architecture on Hugging Face Spaces](#7-cloud-deployment-architecture-on-hugging-face-spaces)
8. [Repository Directory Structure](#8-repository-directory-structure)

---

## 1. Project Lifecycle & Clinical Motivation

### The Clinical Problem:
Cataract is the leading cause of preventable visual impairment and blindness globally. While cataracts can be cured with routine phacoemulsification lens surgery, millions of patients in rural or underserved clinics lack timely access to ophthalmologists with slit-lamp biomicroscopy equipment.

### The Medical AI Bottleneck (Domain Shift):
While deep convolutional neural networks achieve outstanding accuracy when trained and evaluated within a single hospital, they consistently experience **severe performance degradation when deployed to external medical centers**. Variations in fundus camera sensors (e.g., Zeiss vs. Canon vs. Topcon), optical flash illumination, patient pigmentation, and capture protocols shift the input feature distribution. 

### The UncertaCat Solution:
1. **Multi-Task DenseNet-121 Architecture:** Jointly estimates binary cataract presence alongside continuous lens opacity severity ($0\%$ to $100\%$).
2. **Epistemic Uncertainty Quantification (Monte Carlo Dropout):** Executes $T=10$ stochastic forward passes per scan to calculate the standard deviation ($\sigma$). Scans exhibiting high epistemic doubt ($\sigma \ge 8\%$) trigger automated **Safety Referrals**, preventing overconfident diagnostic mistakes.
3. **Two-Stage Unsupervised Domain Adaptation (UDA):** Adapts the baseline model to target hospital cameras using feature covariance alignment (CORAL) and target-domain pseudo-labeling with entropy minimization—**without requiring manual annotations from target ophthalmologists**.

---

## 2. Datasets & Experimental Cohorts

| Dataset Cohort | Primary Role | Source / Modality | Sample Size ($n$) | Clinical Annotation |
| :--- | :--- | :--- | :---: | :--- |
| **OIA-ODIR** (Source Hospital) | Model Training & In-Domain Validation | Multi-center retinal fundus photography (Zeiss, Canon, Topcon) | 3,400 scans | Expert double-annotated binary cataract label & continuous lens opacification grade |
| **RFMiD** (Target Hospital) | Unsupervised Adaptation & External Holdout | Retinal Fundus Multi-Disease Open Benchmark (Topcon TRC-NW8) | 350 holdout scans (CC BY 4.0) | Independent ophthalmology validation labels for external zero-shot vs. adapted comparison |

### Partitioning Scheme:
* **Source Training Split:** 2,400 scans
* **Source Validation Split:** 600 scans
* **Source In-Domain Test Split:** 400 scans
* **External Target Hospital Holdout Split:** 350 scans

---

## 3. End-to-End Image Preprocessing Pipeline

To eliminate extraneous optical artifacts and normalize cross-center illumination discrepancies, all scans undergo an automated 6-stage standardization pipeline at $288 \times 288$ resolution:

```
Raw Fundus Photo 
  │
  ├── 1. Green-Channel Circular FOV Detection & Boundary Crop
  ├── 2. Aspect-Ratio Preserving Resize to 288 x 288 (INTER_AREA)
  ├── 3. Ben-Graham Local Contrast Illumination Normalization
  ├── 4. Green-Channel CLAHE (Contrast-Limited Adaptive Histogram Equalization)
  ├── 5. Circular Perimeter Mask Boundary Smoothing
  └── 6. PyTorch Tensor Normalization (ImageNet Statistics)
```

### Detailed Preprocessing Stages:
1. **Circular FOV (Field of View) Crop:**  
   Extracts the green color channel, performs Otsu binary thresholding, computes the minimum enclosing circle of the dominant contour, and crops away redundant black outer regions.
2. **Resolution Standardization ($288 \times 288$):**  
   Resizes cropped fundus discs using area interpolation (`cv2.INTER_AREA`), preserving fine retinal vascular structures while standardizing memory footprint.
3. **Ben-Graham Local Contrast Normalization:**  
   Balances sensor flash glare and exposure inconsistencies:
   $$\text{Fundus}_{\text{norm}} = 4 \cdot I - 4 \cdot \text{GaussianBlur}(I, \sigma=10) + 128$$
4. **Green-Channel CLAHE:**  
   Enhances subtle lens opacities and optic disc vascular transitions using CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`).
5. **Circular Perimeter Mask:**  
   Smooths zero-intensity pixels outside the retinal diameter to prevent spurious convolutional border responses.
6. **Tensor Normalization:**  
   Converts RGB pixel intensities to tensors normalized by standard ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## 4. Evolutionary Trajectory: Iterations v1 to v6

```
 v1: Single-Task Baseline (224x224) ──► Low Sensitivity, Overfitting
  │
 v2: Multi-Task Learning (224x224) ──► Added Continuous Severity Head
  │
 v3: Epistemic Uncertainty (MC-Dropout) ──► Added Doubt Sigma (T=10)
  │
 v4: Domain Shift Discovery ──► External Test Sensitivity Collapsed (53.8%)
  │
 v5: Prior Shift Trap ──► Target Imbalance Distorted Thresholds (Documented Failure)
  │
 v6: Two-Stage Clean Adaptation (288x288) ──► v6_1.0 (Zero-Shot) vs v6_2.0 (Adapted) (+19.8% Recall Recovery!)
```

### 🔬 Iteration Breakdown:

#### 1. Version 1 (v1) — Baseline Classification
* **Objective:** Initial single-task binary cataract classification using standard ResNet and DenseNet backbones at $224 \times 224$.
* **Challenge Faced:** High in-domain training accuracy but severe overfitting and poor clinical sensitivity ($< 72\%$). Binary labels lacked nuanced insight into early vs. late-stage cataract progression.

#### 2. Version 2 (v2) — Multi-Task Learning & Severity Grading
* **Objective:** Formulate a multi-task objective with dual outputs: binary classification logit + continuous lens opacification severity ($0–100\%$).
* **Challenge Faced:** Gradient scale imbalance between Binary Cross-Entropy loss ($\mathcal{L}_{\text{BCE}}$) and Mean Squared Error loss ($\mathcal{L}_{\text{MSE}}$).
* **Solution:** Applied calibrated loss weighting ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + 0.35 \cdot \mathcal{L}_{\text{MSE}}$).

#### 3. Version 3 (v3) — Epistemic Uncertainty via Monte Carlo Dropout
* **Objective:** Enable AI safety referrals by quantifying epistemic doubt using Monte Carlo Dropout ($T=10$ passes).
* **Challenge Faced:** Standard inference without dropout produced overconfident false negative predictions on low-contrast scans.
* **Solution:** Activated dropout layers during evaluation, calculating $\sigma = \text{std}(\hat{y})$. Calibrated the safety referral threshold at $\sigma \ge 0.08$ to automatically refer high-doubt cases for in-person slit-lamp exam.

#### 4. Version 4 (v4) — Domain Shift Discovery
* **Objective:** First evaluation of the trained multi-task model on external target hospital data (RFMiD).
* **Critical Finding:** While in-domain AUC was $0.9336$, **external sensitivity collapsed to $53.8\%$**—missing almost half of all cataract patients due to target camera illumination and sensor variations. First experiments with basic adversarial gradient reversal showed instability.

#### 5. Version 5 (v5) — The Prior Shift Failure Analysis
* **Objective:** Attempted end-to-end domain adaptation, but encountered **prior probability shift** (the target hospital cohort had a significantly different prevalence of cataracts than the source hospital).
* **Failure Mode Documented:** The model adapted feature distributions but shifted decision boundaries inappropriately, leading to excessive false positive rates. Checkpoints stored nested states and required cleaner architectural isolation.

#### 6. Version 6 (v6) — Two-Stage Clean Adaptation & Canonical Resolution
* **Architectural Upgrade:** Elevated resolution from $224 \times 224$ to $288 \times 288$ with green-channel CLAHE and Ben-Graham filtering.
* **Separation into Two Explicit Models:**
  * **v6_1.0 (Zero-Shot Baseline):** Pure source hospital training. Demonstrates unadapted performance drop on external clinics.
  * **v6_2.0 (Hospital-Adaptive Model):** Two-Stage Unsupervised Domain Adaptation:
    * *Stage 1:* **CORAL Loss** aligns second-order feature covariance between hospital distributions.
    * *Stage 2:* **Confidence-Filtered Pseudo-Labeling** with entropy minimization on unlabelled target hospital scans.
* **Breakthrough:** Recovered external sensitivity from **$53.8\%$ to $73.6\%$ ($+19.8\%$ absolute recall gain)** with statistically significant discrimination improvement ($p = 0.0180$) and a **$10.3\%$ reduction in Brier calibration error**.

---

## 5. Master Evaluation Results & Benchmark Analysis

*All metrics are verified from canonical evaluation artifacts (`master_table.md`, `results_v6_1.0.json`, and `results_v6_2.0.json`).*

| Metric | Source In-Domain Test (OIA-ODIR) | External Holdout: Zero-Shot (v6_1.0) | External Holdout: Hospital-Adapted (v6_2.0) | Adaptation Gain ($\Delta$) | Clinical Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ROC-AUC** | 0.9336 | 0.8113 (Ensemble: 0.8021) | **0.8144** (Ensemble: **0.8172**) | **+0.0154** ($p = 0.0180$) | Statistically significant discrimination improvement |
| **Sensitivity (Recall)** | 85.2% | 53.8% | **73.6%** | **+19.8% Absolute** | **Recovers ~20% of cataracts missed by the baseline** |
| **Specificity** | 90.8% | 87.6% | 77.3% | -10.3% | Balanced operating trade-off |
| **Brier Score (Calibration)** | 0.0912 | 0.1764 | **0.1583** | **-10.3% Error** | Significantly better probability calibration |
| **Expected Calibration Error (ECE)** | 0.0782 | 0.1412 | **0.1085** | **-23.2% Relative** | Substantially reduced overconfidence |
| **Balanced Accuracy** | 88.0% | 70.7% | **75.5%** | **+4.8% Absolute** | Higher cross-class diagnostic balance |

---

## 6. Glossary of Acronyms & Technical Terminology

| Acronym / Term | Full Form | Plain-Language Definition & Purpose |
| :--- | :--- | :--- |
| **FOV** | Field of View | The circular boundary of the eye visible through the camera lens. Border cropping isolates the FOV. |
| **CLAHE** | Contrast-Limited Adaptive Histogram Equalization | A local contrast enhancement algorithm that amplifies faint cataract media opacities without overamplifying sensor noise. |
| **MC Dropout** | Monte Carlo Dropout | Running multiple forward passes with dropout active during inference to approximate Bayesian neural network uncertainty. |
| **Epistemic Uncertainty** | Model Doubt / Knowledge Uncertainty | Uncertainty caused by a lack of training data on unfamiliar camera sensors; reducible through domain adaptation. |
| **Aleatoric Uncertainty** | Data Noise / Inherent Ambiguity | Inherent visual ambiguity in an image (e.g. motion blur, corneal scars) that cannot be eliminated by adding more data. |
| **UDA** | Unsupervised Domain Adaptation | Adapting an algorithm to a new deployment environment using unlabelled target data, eliminating the need for expensive physician annotations. |
| **CORAL** | Correlation Alignment | A domain adaptation loss that minimizes the Frobenius distance between source and target second-order feature covariance matrices. |
| **AUROC / ROC-AUC** | Area Under Receiver Operating Characteristic | A threshold-independent measure of classification capability. 1.0 is perfect; 0.5 is random chance. |
| **Brier Score** | Brier Calibration Metric | The mean squared difference between predicted probabilities and actual binary outcomes. Lower is better. |
| **ECE** | Expected Calibration Error | Measures the gap between predicted confidence and observed empirical accuracy across risk bins. |
| **OIA-ODIR** | Ocular Disease Intelligent Recognition | Multi-hospital clinical dataset of 10,000 patient fundus images used as our source hospital training cohort. |
| **RFMiD** | Retinal Fundus Multi-Disease Dataset | Independent open-access benchmark (CC BY 4.0) used as our target hospital external holdout cohort. |
| **SaMD** | Software as a Medical Device | Regulatory designation for software intended to be used for diagnostic or clinical purposes. |
| **PHI** | Protected Health Information | Individually identifiable healthcare data protected by privacy legislation (e.g. HIPAA, GDPR). |

---

## 7. Cloud Deployment Architecture on Hugging Face Spaces

The deployment package is hosted live on **Hugging Face Spaces** running on **Free CPU-Basic (2 vCPU, 16 GB RAM)**:

```
User Fundus Photo 
       │
       ▼
 [ Preprocessing: 288x288 FOV + Graham Normalization + CLAHE ]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
Zero-Shot Model (v6_1.0)        Hospital-Adaptive (v6_2.0)
 (10 MC Stochastic Passes)       (10 MC Stochastic Passes)
       │                                 │
       ▼                                 ▼
 [ Prob ± Doubt | Severity ]     [ Prob ± Doubt | Severity ]
       │                                 │
       └────────────────► ◄──────────────┘
                         │
                         ▼
   [ Side-by-Side Model Diagnostic Comparison ]
   [ Adaptation Delta (Δ) & Disagreement Banner ]
```

### Local Execution:
```bash
git clone https://github.com/UmarIqbal000/UncertaCat-Modules.git
cd UncertaCat-Modules
pip install -r requirements.txt
python app.py
```

---

## 8. Repository Directory Structure

```text
UncertaCat-Modules/
├── README.md                      # Comprehensive project documentation & lifecycle report
├── notebooks/                     # All executed research notebooks (v1 through v6)
│   ├── Major_Project_image_preprocessing.ipynb
│   ├── Major_Project_model_training_v1.ipynb
│   ├── Major_Project_model_training_v2.ipynb
│   ├── Major_Project_model_training_v3.ipynb
│   ├── Major_Project_model_training_v4.ipynb
│   ├── Major_Project_model_training_v5.ipynb
│   ├── Major_Project_model_training_v6_1.0.ipynb
│   ├── Major_Project_model_training_v6_2.0.ipynb
│   └── Major_Project_v1_to_v6_comparison_and_reports.ipynb
├── results/                       # Empirical CSVs, JSON evaluation files, and figures
│   ├── master_table.md            # Canonical cross-version benchmark markdown table
│   ├── evaluation_results_v1/     # Metrics, ROC plots, and predictions for v1
│   ├── evaluation_results_v2/     # Metrics, ROC plots, and predictions for v2
│   ├── evaluation_results_v3/     # Metrics, ROC plots, and predictions for v3
│   ├── evaluation_results_v4/     # Metrics, ROC plots, and predictions for v4
│   ├── evaluation_results_v5/     # Metrics, ROC plots, and predictions for v5
│   └── evaluation_results_v6/     # Final comparative figures and holdout predictions
│       ├── report_figures/        # Evolutionary trajectory, domain gap, and recovery curves
│       ├── adaptation_comparative_visuals_v6.png
│       ├── predictions_external_holdout_v6_1.0.csv
│       ├── predictions_external_holdout_v6_2.0.csv
│       ├── results_v6_1.0.json
│       └── results_v6_2.0.json
└── docs/                          # Future reports, defense slides, and thesis documentation
    └── README.md
```

---

## ⚠️ Academic & Ethical Disclaimer
**RESEARCH DEMO ONLY:** This software is an academic paired-model comparison prototype designed for research and educational purposes. It is **not** an approved medical device, software as a medical device (SaMD), or diagnostic aid. It must not be used to diagnose, triage, treat, or prevent any ocular condition or pathology. Do not upload real patient records or Protected Health Information (PHI).
