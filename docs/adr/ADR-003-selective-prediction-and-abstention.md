# ADR 003: Selective Prediction and Clinical Abstention Gate

- **Status:** Accepted
- **Date:** 2026-10-02
- **Deciders:** Bhagyansh Chandel

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

Using `hemolux.metrics.calibration.risk_coverage_curve`, coverage can be swept
from 100% downward. **On the first frozen run this did not improve accuracy:**

| Operational Mode | Image Acceptance Rate | Mean Absolute Error (MAE) |
| --- | --- | --- |
| Full coverage (no abstain) | 100% | 2.16 g/dL |
| Abstain on the least-confident images | 25% | 2.26 g/dL |

Rejecting the least-confident acquisitions *raised* error on the measured corpus
rather than lowering it, and `risk_coverage_curve` is written so a noise signal
scores no better than a real one (see `tests/test_calibration.py`). The abstention
gate is therefore justified here as a safety behaviour — do not return a number
when the photograph cannot support one — and **not** as an accuracy improvement.
No accuracy-improvement figure is claimed.

### Key Clinical Insights:
1. Returning a point estimate for an unmeasurable image is a hazard regardless of
   what it does to the aggregate error, so the gate stays even though it is not an
   accuracy win.
2. The abstained cases are not "failed"; they are prompted to retake under
   daylight or directed to standard lab evaluation.
3. In point-of-care screening, a false negative (missing moderate/severe anaemia)
   carries severe morbidity risks; abstaining removes erratic false negatives
   caused by optical degradation.

---

## Consequences

### Positive:
- Protects patient safety and aligns with medical device regulations (FDA SaMD guidelines).
- Produces trustworthy metrics that clinicians can actually defend.
- Demonstrates engineering rigor to recruiters: "We know when our model does not know."

### Negative / Trade-offs:
- 10–15% of captures require retakes under improved lighting.
- The UI must handle the "Abstained" state as an expected clinical interaction rather than a system crash.
