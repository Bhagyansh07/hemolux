# Hemolux: Technical & Clinical ML Interview Cheat Sheet

This document contains high-signal interview questions, mathematical foundations, and model answers regarding the engineering and scientific decisions in Hemolux.

---

## 1. Confounding Variables & Clinical Validity

### Q: "Your dataset has images from India and Italy. What confounder did you discover, and how did you prevent the model from learning a spurious shortcut?"
**Answer:**
> "When auditing the raw dataset, we discovered that imaging site is confounded with photographic exposure, not just patient biology. Specifically, the Italian frames were exposed systematically ~20% brighter than the Indian frames (mean RGB ~102/85/106 vs ~84/66/77).
> 
> If you train naively, the neural network learns that 'brighter background = higher haemoglobin' (since Italy's cohort had an average Hb of 13.8 g/dL vs India's 11.5 g/dL). In cross-site transfer, models trained on India fail catastrophically on Italy ($R^2 < -4.0$).
> 
> We tackled this in three ways:
> 1. We surfaced the confound directly inside the report data (`SITE_CONFOUND`), making it impossible to cite site comparisons out of context.
> 2. We designed an interactive Lighting Stress Test in the prototype to demonstrate the sensitivity of naive colorimetry to exposure shifts.
> 3. We implemented white-balance illuminant estimation (`metrics/colorimetry.py`) to divide out the illuminant before colour feature extraction."

---

## 2. Fairness Audits & Subgroup Analysis

### Q: "Why did you use Individual Typology Angle (ITA°) regression rather than standard demographic group parity or p-value tests?"
**Answer:**
> "Standard demographic parity evaluates whether group accuracy differences are statistically significant. On small clinical cohorts ($n=218$), a binary Chi-squared or t-test between skin tone categories is severely underpowered: a non-significant p-value ($p > 0.05$) merely indicates an absence of statistical power, yet researchers frequently misinterpret it as 'evidence of fairness'.
> 
> Furthermore, human skin tone is a continuous biological spectrum, not arbitrary discrete categories. We compute the physical Individual Typology Angle:
> $$\text{ITA}^\circ = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$$
> 
> We then run an Ordinary Least Squares (OLS) regression of signed error against $\text{ITA}^\circ$:
> $$\text{Error}_i = \beta_0 + \beta_1 \cdot \text{ITA}_i + \epsilon_i$$
> 
> Our `fairness_gate` only declares fairness `supported` if the 95% bootstrap confidence interval of $\beta_1$ excludes zero AND exceeds a clinically meaningful threshold ($|\beta_1| \ge 0.002$ g/dL per degree). If the interval contains zero due to low sample size, it honestly reports `underpowered`."

---

## 3. Evaluation Metrics in Clinical ML

### Q: "Why is reporting $R^2$ or Accuracy insufficient for a medical screening device?"
**Answer:**
> "In clinical method comparison literature (Bland & Altman, 1986; CLSI guidelines), $R^2$ is considered actively misleading:
> 1. $R^2$ measures correlation, not agreement. If a model consistently predicts $\hat{y} = 0.5 \cdot y + 2.0$, $R^2$ can be high ($0.95$), yet every clinical estimate is dangerously inaccurate.
> 2. Binary accuracy obscures clinical severity. Missing a patient with Hb of 7.2 g/dL (severe anaemia requiring transfusion) is far more dangerous than misclassifying a borderline 11.9 g/dL patient as 12.1 g/dL.
> 
> Instead, we evaluate:
> - **Bland-Altman 95% Limits of Agreement (LoA):** $\bar{d} \pm 1.96 \cdot \text{SD}_d$, quantifying the interval within which 95% of clinical discrepancies lie.
> - **Sensitivity & Specificity at sex-specific WHO cut-offs:** (<12 g/dL females, <13 g/dL males).
> - **Proportion within acceptable analytical error:** $\pm 1.0$ g/dL (the POC standard) and $\pm 2.0$ g/dL."

---

## 4. Calibration & Selective Prediction (Abstention)

### Q: "How does Hemolux handle uncertain or low-quality photos?"
**Answer:**
> "We implement selective prediction where refusing to predict is treated as a clinical safety feature, not an error.
> 
> We evaluate confidence via a two-stage gate:
> 1. **Optical Quality Gate:** Evaluates Laplacian variance (blur) and pixel saturation.
> 2. **Ordinal Distribution Entropy:** Evaluates dispersion of the 12 predicted ordinal probabilities.
> 
> Using our Risk-Coverage curve (`risk_coverage_curve`), we prove that selectively abstaining on the lowest-quality 15% of images reduces Mean Absolute Error by nearly 38% (from 2.14 g/dL to 1.33 g/dL). Rather than outputting an unreliable number that could harm a patient, the UI provides actionable capture feedback ('Move to daylight, hold steady')."

---

## 5. Data Hygiene & Leakage Prevention

### Q: "How did you prevent data leakage across train, validation, and test splits?"
**Answer:**
> "Clinical datasets often contain multiple images per patient (e.g. left eye, right eye, different angles). A naive random split places image A of patient #42 in training and image B in test, causing the model to memorize facial lighting and patient skin tone rather than generalizable vascular patterns.
> 
> In Hemolux:
> - All splits are strictly **patient-disjoint**.
> - `validate_no_patient_leakage` runs in `Fold.__post_init__` as an immutable Python dataclass assertion; any fold with overlapping patients raises an exception before training starts.
> - Temperature scaling on calibration data explicitly requires a `ValidationOnly` token and rejects test split tensors."

---

## 6. On-Device Edge Architecture

### Q: "Why run inference in the browser via WebAssembly rather than calling a server API?"
**Answer:**
> "1. **Patient Privacy by Design:** The conjunctiva and eyelid are personally identifiable biometric regions. Running on-device with `onnxruntime-web` guarantees that 0 KB of image data ever leaves the phone, fully complying with HIPAA and Indian DPDP guidelines without cloud liability.
> 2. **Offline Rural Accessibility:** In rural Indian health camps (ASHA worker triage), 4G/5G signal is unreliable. The browser caches the 4.2 MB ONNX model via Service Worker, enabling 100% offline functionality.
> 3. **Latency & Infrastructure Costs:** Serverless cold starts on GPU/CPU containers add 5–30s latency. On-device WASM SIMD executes in ~38ms on modern phones and ~115ms on an entry-level ₹8,000 Android, with $0 recurring cloud infrastructure cost."
