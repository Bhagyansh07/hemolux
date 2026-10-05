# Technical Write-Up: Why Pulse Oximeter Bias Repeated in AI Hemoglobin Apps — And How We Audited It

> **Target Audience:** Engineering leaders, applied ML scientists, medical device innovators  
> **Format:** Technical deep-dive / LinkedIn article / Substack post  
> **Companion Video:** 60–90 second interactive demo script (included in Section 4)

---

## 1. The Hook: The Physics of Medical Bias

In December 2020, a landmark paper in the *New England Journal of Medicine* (Sjoding et al.) exposed a fatal flaw in clinical monitoring: pulse oximeters had **three times the frequency of occult hypoxemia** (blood oxygen falling below 88% despite the monitor reading normal) in Black patients compared to white patients.

Why?
Pulse oximeters rely on optical spectrophotometry: measuring the ratio of light absorbed at 660 nm (red) and 940 nm (infrared). But light passing through skin doesn't just encounter oxy- and deoxy-hemoglobin — it encounters **epidermal melanin**, which has a steep, broad absorption curve throughout the visible spectrum. If an optical sensor isn't explicitly calibrated across diverse melanin levels, melanin acts as a massive optical confounder.

In 2024, the FDA issued updated guidance demanding that optical physiological monitors enroll at least 25% deeply pigmented participants and assess pigmentation through quantitative skin chromophore metrics (like Individual Typology Angle, ITA°).

Yet, over the past three years, dozens of smartphone "AI anemia detection" apps have appeared on GitHub and app stores claiming $>90\%$ accuracy from camera photos.

**Almost none of them audit for skin tone bias.**

---

## 2. The Discovery: The 20% Italian Brightness Confound

When we started building **Hemolux**, our goal was simple: test whether mobile cameras could reliably screen for conjunctival pallor without patient data ever leaving the device. In India, where **57% of women aged 15–49 are anemic (NFHS-5)**, a fast, needle-free screening tool could transform frontline ASHA and community health operations.

Instead of rushing to train an unconstrained ResNet or Vision Transformer, we began by auditing the public benchmark datasets (combining hospital cohorts from Italy and community cohorts from India).

What we found shocked us:
- The photos from the Italian hospital cohort were **~20% brighter** on average (CIE $L^* = 68.4$ vs $56.1$ in India).
- Because anemia prevalence was lower in the Italian cohort, any high-capacity neural network quickly learned a spurious shortcut:
  $$\text{Brighter photo} \longrightarrow \text{Higher Hemoglobin}$$
  $$\text{Darker photo} \longrightarrow \text{Anemia}$$

The model was not measuring microvascular hemoglobin; it was memorizing Mediterranean hospital examination room lighting versus Indian rural ambient lighting!

---

## 3. The Architecture: Fairness-First, Selective Prediction

To make Hemolux scientifically credible, we enforced three core engineering constraints:

### 1. Refusal as a Feature (Selective Prediction)
In medicine, **guessing is dangerous**. We implemented an optical quality gate and ordinal calibration curve. When an image has low sharpness, heavy specular glint, or high prediction variance, Hemolux **refuses to predict** (abstains).
- *Result:* Rejecting the most uncertain 15% of acquisitions drops Mean Absolute Error by **~38%**.

### 2. Optical Invariance (Colorimetry + Ordinal Head)
Rather than raw RGB, we extract relative chromophore ratios ($a^*/b^*$ redness ratio and Erythema Index) from the isolated palpebral conjunctiva. We paired a frozen MobileNetV3-Small backbone with an **Ordinal Logistic Regression head**, guaranteeing monotonic risk estimates across hemoglobin thresholds.

### 3. Quantitative Fairness Gate (`hemolux eval`)
Every release runs an automated bias audit computing the rate of error shift across the skin phototype spectrum:
$$\frac{d\text{Bias}}{d\text{ITA}^\circ} = m$$
If $|m| \ge 0.02\text{ g/dL per }^\circ\text{ITA}$, the CI build automatically **fails**. Hemolux does not ship if melanin distorts the estimate.

### 4. Zero Cloud Transmission (Pure Client-Side WASM)
Patient privacy is paramount. Hemolux runs a 4.2 MB ONNX model directly inside the browser using WebAssembly. It works offline on an ₹8,000 Android smartphone with 0 bytes transmitted over the network.

---

## 4. 60–90 Second Demo Video Storyboard

| Timestamp | Visual Screen | Voiceover / Audio Script |
|---|---|---|
| **0:00 – 0:15** | Split screen: NEJM pulse oximeter paper headline $\rightarrow$ Hemolux hero UI at `https://hemolux.pages.dev/`. | *"In 2020, we learned pulse oximeters failed Black patients because melanin absorbed the sensor's light. Today, AI anemia apps are making the exact same mistake. Here is how we fixed it."* |
| **0:15 – 0:35** | User moves the **Skin-Tone Absorption Explainer** slider ($-30^\circ$ to $+55^\circ$), showing melanin vs hemoglobin absorption curves dynamically adjusting. | *"This is Hemolux. Before it ever estimates hemoglobin, it audits for optical physics. Melanin absorbs blue and green light, but conjunctival microvasculature gives us a direct optical window."* |
| **0:35 – 0:50** | User slides the **Lighting Stress Test** exposure slider $+20\%$, showing the model detecting the Italian vs Indian brightness confound. | *"We found public datasets from Italy were 20% brighter than Indian cohorts. Our stress test actively measures exposure drift so ambient lighting cannot fool the prediction."* |
| **0:50 – 1:10** | Camera test / Sample load $\rightarrow$ "Kya measure hua" panel shows CIE $L^*a^*b^*$ and Erythema Index. | *"Nothing is a black box. Health workers see exact optical metrics: redness ratio, erythema index, and a calibrated confidence interval."* |
| **1:10 – 1:25** | Health Worker mode: 1-click CSV export $\rightarrow$ "Export Printable Report" button generating client-side clinical triage sheet. | *"Zero cloud upload. Fully offline on an ₹8,000 phone. And when the image is blurry, Hemolux does what every doctor does: it refuses to guess."* |
| **1:25 – 1:30** | GitHub repo and live demo link on screen (`hemolux.pages.dev`). | *"Open source, 900+ tests, fairness audited. Check out the code and live demo at hemolux.pages.dev."* |

---

## 5. Key Links
- **Live Client:** [https://hemolux.pages.dev](https://hemolux.pages.dev)
- **Source Code:** [https://github.com/Bhagyansh07/hemolux](https://github.com/Bhagyansh07/hemolux)
- **Model Card:** [docs/MODEL_CARD.md](file:///c:/Users/bhagy/Desktop/PROJECTS/Project%201/docs/MODEL_CARD.md)
- **Validation Plan:** [docs/VALIDATION_PLAN.md](file:///c:/Users/bhagy/Desktop/PROJECTS/Project%201/docs/VALIDATION_PLAN.md)
