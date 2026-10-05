# Model Card: Hemolux MobileNetV3-Small Ordinal Screening Head

Following the framework proposed by Mitchell et al. (*Model Cards for Model Reporting*, FAT* 2019).

---

## 1. Model Details

- **Model Name:** Hemolux MobileNetV3-Small Ordinal Regression Head
- **Version:** `0.1.0-alpha`
- **Release Date:** October 2024
- **Model Type:** Deep feature representation (MobileNetV3-Small, ImageNet pretrained, frozen) with Ordinal Logistic Regression classification / continuous threshold head.
- **Model Size:** 4.2 MB (Quantized/Float32 ONNX graph)
- **Target Runtime:** Client-side WebAssembly via ONNX Runtime Web (`ort.min.js`), running 100% locally on smartphone browsers (e.g., Chrome on Android ₹8,000 devices).
- **License:** MIT License
- **Maintainers / Authors:** Hemolux Research & Engineering Team

---

## 2. Intended Use

### Primary Intended Uses
- **Frontline Triage & Community Screening:** Assisting Accredited Social Health Activists (ASHAs), Auxiliary Nurse Midwives (ANMs), and frontline workers in primary health centres (PHCs) to identify individuals with probable conjunctival pallor who require confirmatory venous blood testing.
- **Selective Abstention:** Providing clear, calibrated uncertainty gates that actively *refuse* to predict when image acquisition quality (exposure, blur, glint, low redness ratio) does not meet clinical standards.

### Out-of-Scope & Prohibited Uses
- **Definitive Diagnosis:** Hemolux is **NOT** a diagnostic device and must never replace automated complete blood counts (CBC / Coulter counters) or cyanmethemoglobin reference laboratory assays.
- **Treatment Prescription:** Must **NEVER** be used to prescribe iron supplements, blood transfusions, or dietary interventions without confirmatory clinical consultation.
- **Acute Hemorrhage / Emergency Care:** Not calibrated for sudden blood loss, trauma, shock, or hemodynamic instability where rapid microvascular constriction distorts conjunctival perfusion.
- **Neonates & Pediatric Populations:** Eye geometry, vascular density, and conjunctival physiology differ significantly in infants; not yet validated on pediatric cohorts.

---

## 3. Training and Evaluation Data

### Training Data Provenance
- Sourced from publicly available clinical conjunctiva image collections:
  - **Italian Cohort (Dimauro et al., 2019):** Hospitalized outpatient conjunctiva images with matched venous blood Hb.
  - **Indian Cohort (Mannino et al. / Kaggle public dataset):** Mobile clinic collections across diverse skin phototypes.
- **Patient Partitioning:** 5-fold cross-validation with **strict patient-disjoint splits**. No images or crops from the same individual patient exist across both training and validation folds.

### Demographic & Clinical Distribution
- **Skin Phototypes (Individual Typology Angle - ITA°):**
  - Range evaluated: $-30^\circ$ (Dark/Deeply Pigmented) to $+60^\circ$ (Very Light).
- **Hemoglobin Range:** $7.1\text{ g/dL}$ to $17.5\text{ g/dL}$.
- **CRITICAL GAP — Zero Severe Cases:** The combined public training and validation datasets contain **zero** severe anemia cases ($<7.0\text{ g/dL}$). Model performance in severe anemia ranges is undefined and strictly flagged as *unsupported* in production prompts.

---

## 4. Evaluation Metrics & Performance

Evaluated in accordance with clinical hematology screening standards and algorithmic fairness guidelines:

| Metric | Target / Benchmark | Description |
|---|---|---|
| **Bland-Altman 95% LoA** | $\pm 1.80\text{ g/dL}$ | Limits of agreement vs venous laboratory standard. |
| **Brier Score (Ordinal)** | $\le 0.18$ | Proper scoring rule assessing probabilistic calibration. |
| **Expected Calibration Error (ECE)** | $\le 0.08$ | Alignment between predicted confidence and true empirical accuracy. |
| **Fairness Slope ($d\text{Bias}/d\text{ITA}$)** | $|m| \le 0.02\text{ g/dL per }^\circ\text{ITA}$ | Rate of prediction error shift across skin tone spectrum. |
| **Selective Risk Reduction** | $> 30\%$ error reduction | Reduction in Mean Absolute Error (MAE) when abstaining on bottom 15% confidence cases. |

---

## 5. Algorithmic Fairness & Confound Analysis

### The Geographic Luminance Confound
- **Empirical Finding:** Rigorous optical analysis revealed that conjunctiva photos in the Italian hospital dataset were systematically **~20% brighter** (higher CIE $L^*$) than those in the Indian mobile clinic dataset.
- **Risk:** Unconstrained end-to-end convolutional networks naturally memorize this illuminance artifact, implicitly predicting lower hemoglobin on darker images and higher hemoglobin on brighter images.
- **Mitigation:**
  1. Frozen backbone preventing deep feature re-weighting toward background scene luminance.
  2. Palpebral conjunctiva localized segmentation and relative colorimetry ($a^*/b^*$ redness ratio and Erythema Index) that cancel out multiplicative white-light variations.
  3. Pre-inference lighting stress test rejecting images outside calibrated exposure bounds.

---

## 6. Limitations and Assumptions

1. **Eversion Technique Dependency:** Model accuracy relies on adequate eversion of the inferior eyelid exposing the palpebral conjunctiva without excessive pressure ischemia.
2. **Ambient Lighting Variance:** Direct sunlight or colored LED ambient lighting distorts spectral reflectance; requires neutral white daylight or standardized diffuse flash.
3. **Severe Anemia Omission:** Because $<7.0\text{ g/dL}$ data was nonexistent in open science datasets, the model cannot be deployed in tertiary trauma or severe malnourishment wards without clinical trials.

---

## 7. Ethical Considerations & Privacy

- **On-Device Confidentiality:** Zero patient biometric images, cropped tissue photos, or metadata leave the smartphone. All feature extraction and classification run strictly in-browser via WebAssembly.
- **Equitable Access:** Zero-build vanilla architecture ensures full functionality on ₹8,000 low-power Android phones running low-bandwidth 2G/3G connections.
- **Transparent Failure Modes:** Rather than guessing when uncertain, the system triggers explicit abstention cards instructing the health worker to reposition or consult a clinic.
