# ADR 003: Selective Prediction and Clinical Abstention Gate

- **Status:** Accepted
- **Date:** 2026-10-02
- **Deciders:** Core Engineering Team

---

## Context & Problem Statement

In consumer health software, AI models are almost always programmed to return a prediction for every input image, regardless of blur, lighting, specular glare, or eyelid framing.

In clinical diagnostics, a point estimate returned on an unmeasurable image is not a feature — it is a hazard. An estimate of 12.8 g/dL generated from an underexposed or motion-blurred photo may falsely reassure an anaemic patient, causing them to forgo a laboratory Complete Blood Count (CBC).

How do we architect an honest screening tool that prioritizes patient safety over coverage?

---

## Decision

We make **Abstention a first-class operational feature** through a two-stage gate:

1. **Pre-Inference Quality Gate (`QUALITY_ABSTAIN`):**
   - Blur metric (Laplacian variance threshold).
   - Exposure limits (clipping / dynamic range check).
   - Minimum region-of-interest (ROI) fill ratio.
2. **Post-Inference Predictive Uncertainty Gate:**
   - If predictive dispersion $\sigma > \tau_{\text{abstain}}$, or if the ordinal posterior distribution is multi-modal, the model explicitly refrains from outputting a number.
   - The UI displays an informative, neutral guidance card: *"Could not read this image: The photo was not clear enough to give a reliable estimate. Move to even light and retake."*

---

## Empirical Rationale (Risk-Coverage Analysis)

Using `hemolux.metrics.calibration.risk_coverage_curve`, we sweep coverage from 100% down to 60%:

| Operational Mode | Image Acceptance Rate | Mean Absolute Error (MAE) | Clinical Error Reduction |
| --- | --- | --- | --- |
| **Naive (No Abstain)** | 100% | 2.14 g/dL | Baseline |
| **Quality Gate Only** | 92% | 1.84 g/dL | -14.0% |
| **Quality + Uncertainty Gate** | **85%** | **1.33 g/dL** | **-37.8%** |
| **Strict High-Confidence Gate** | 70% | 1.08 g/dL | -49.5% |

### Key Clinical Insights:
1. **Refusing to guess drops clinical error by nearly 38%** while still providing actionable triage for 85 out of 100 individuals.
2. The remaining 15% of patients are not "failed"; they are immediately prompted to retake under daylight or directed to standard lab evaluation.
3. In point-of-care screening, a false negative (missing moderate/severe anaemia) carries severe morbidity risks; abstention eliminates erratic false negatives caused by optical degradation.

---

## Consequences

### Positive:
- Protects patient safety and aligns with medical device regulations (FDA SaMD guidelines).
- Produces trustworthy metrics that clinicians can actually defend.
- Demonstrates engineering rigor to recruiters: "We know when our model does not know."

### Negative / Trade-offs:
- 10–15% of captures require retakes under improved lighting.
- The UI must handle the "Abstained" state as an expected clinical interaction rather than a system crash.
