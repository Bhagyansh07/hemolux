# ADR 001: Ordinal Classification Head vs Direct Regression and Binary Classification

- **Status:** Accepted
- **Date:** 2026-10-02
- **Deciders:** Core Engineering Team

---

## Context & Problem Statement

In non-invasive haemoglobin estimation from smartphone tissue photographs, there are three common architectural approaches:
1. **Binary Classification:** Classify patient as anaemic vs non-anaemic ($Hb < 12$ or $< 13$ g/dL).
2. **Direct Scalar Regression:** Single continuous output trained with Mean Squared Error (MSE) or Huber loss.
3. **Ordinal Classification (Label Distribution Learning):** Discretise continuous Hb into ordered bins, predict a probability distribution constrained by ordinality, and compute expected value $\hat{y} = \sum p_i \cdot c_i$.

Which architecture maximizes clinical triage utility, numerical stability, and robustness under small sample sizes ($n=217$)?

---

## Decision

We adopt an **Ordinal Head with Soft-Target Gaussian Label Distribution Learning** as the primary screening architecture, accompanied by Huber regression as a baseline comparison.

```
Input Image (224x224) ───> Frozen MobileNetV3 ───> 1024-dim Features
                                                           │
                                                           ▼
                                            Linear Layer (1024 -> 12 bins)
                                                           │
                                                           ▼
                                            Softmax Probability Vector (p)
                                                           │
                                ┌──────────────────────────┴──────────────────────────┐
                                ▼                                                     ▼
                  Point Estimate: E[Hb] = Σ p_i * c_i                  Posterior Variance: σ²
```

### Key Technical Specifications:
- **12 Ordinal Bins:** Centered from 5.0 g/dL to 17.0 g/dL ($c_i$).
- **Soft Targets:** During training, true Hb is converted to a normalised Gaussian distribution centered at $y$ with $\sigma = 0.6$ g/dL.
- **Cross-Entropy Loss:** Penalises predictions based on divergence from the continuous distribution, preserving metric distances between neighbouring categories.

---

## Rationale & Clinical Justification

### 1. Binary Classification Destroys Critical Clinical Urgency
- In clinical practice, $Hb = 7.1$ g/dL (severe anaemia requiring urgent medical intervention) and $Hb = 11.8$ g/dL (borderline mild anaemia treatable with dietary consultation) are both labelled "positive / anaemic" by a binary threshold.
- A binary classifier cannot rank clinical risk or monitor response to treatment.

### 2. Direct MSE Regression is Vulnerable to Outlier Gradient Shock
- Direct scalar regression with MSE squares large errors: an error of $3.0$ g/dL exerts 9× the gradient of $1.0$ g/dL. On a small clinical dataset ($n=217$), outliers skew linear weights dramatically.
- While Huber loss mitigates extreme gradients, it does not provide an intrinsic probability distribution.

### 3. Ordinal Heads Provide Uncalibrated Uncertainty for Free
- The ordinal distribution outputs a probability vector $p \in \Delta^{11}$.
- The variance $\sigma^2 = \sum p_i (c_i - \hat{y})^2$ serves as an immediate predictive uncertainty estimate without requiring Monte Carlo Dropout or deep ensembles.
- Samples where the distribution is multi-modal or wide can trigger the **Abstain Gate**, protecting patient safety.

---

## Consequences

### Positive:
- Clinically meaningful continuous output in g/dL with associated confidence bounds.
- Smoother optimization landscape than direct regression on small, class-imbalanced datasets.
- Enables temperature scaling and Expected Calibration Error (ECE) auditing.

### Negative / Trade-offs:
- Slightly larger head parameter count (1024×12 vs 1024×1) — negligible for edge deployment (<0.05 MB).
- Bounded extrapolation: Predictions cannot fall outside $[4.0, 18.0]$ g/dL, which matches human physiological limits.
