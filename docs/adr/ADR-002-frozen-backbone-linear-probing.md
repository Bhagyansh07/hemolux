# ADR 002: Frozen Backbone vs End-to-End Fine-Tuning

- **Status:** Accepted
- **Date:** 2026-10-02
- **Deciders:** Core Engineering Team

---

## Context & Problem Statement

The Eyes Defy Anemia dataset contains 217 usable patient records across two clinical sites (India and Italy). Pre-trained convolutional architectures (e.g. MobileNetV3-Small, EfficientNet-B0) contain 1.5M to 5.3M parameters.

Training millions of parameters on 217 samples risks catastrophic overfitting. Additionally, the Italian site is photographed approximately 20% brighter than the Indian site. Does fine-tuning learn true physiological features, or does it memorize site-specific lighting artifacts?

---

## Decision

We freeze the entire convolutional backbone (ImageNet pre-trained weights) and train **only a linear probe / shallow classification head** (`unfreeze_param_fraction = 0.0`).

Feature representations are extracted once and cached as deterministic NumPy arrays (`hemolux features`), making hyperparameter sweeps, cross-validation, and site-holdout audits 100× faster and mathematically reproducible.

---

## Rationale & Empirical Findings

### 1. Feature Memorization vs Physiological Generalisation
When backbones are fine-tuned end-to-end on $n=217$:
- Training loss drops to near-zero within 10 epochs.
- Cross-site transfer performance collapses catastrophically: models trained on Indian frames fail completely on Italian frames ($R^2 < -4.0$).
- Feature representations latch onto background skin illumination, sensor noise characteristics, and white-balance differences rather than conjunctival microvascular redness.

### 2. The Linear Probe as an Audit Anchor
- Freezing the backbone preserves general-purpose spatial and colour texture filters.
- A linear head cannot manipulate convolutional feature maps to overfit spurious correlations.
- This creates an honest baseline: if a frozen feature space cannot predict haemoglobin, fine-tuning's apparent success is proven to be memorisation.

### 3. Edge Computation & Verification Speed
- Caching frozen 1024-dimensional feature vectors allows 40-epoch cross-validation sweeps across all 5 heads to run in seconds rather than hours.
- Deterministic feature hashing (`feature_fingerprint`) guarantees that code changes affecting preprocessing are detected automatically.

---

## Consequences

### Positive:
- Drastically reduced risk of overfitting to the 217 patients.
- Fast, reproducible experiments without GPU dependencies.
- Models generalize better across patient-disjoint validation folds.

### Negative / Trade-offs:
- Upper bound on in-domain accuracy: The backbone cannot adapt its lower-level convolutional filters specifically to conjunctival microvasculature.
- Mitigation: When larger multicentric cohorts ($n > 5,000$) become available, staged unfreezing of the top 10% of layers can be audited via `hemolux train --unfreeze 0.1`.
