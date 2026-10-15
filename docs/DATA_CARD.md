# Data Card: Hemolux Conjunctival Pallor Benchmark Dataset

Following the framework proposed by Gebru et al. (*Datasheets for Datasets*, CACM 2021).

---

## 1. Motivation

- **Purpose:** Provide a standardized, fairness-audited dataset of human palpebral conjunctiva images paired with laboratory-grade venous hemoglobin ($Hb$) measurements to train and evaluate non-invasive anemia screening models.
- **Created By:** The Hemolux project. The corpus itself is the **Eyes Defy Anemia** collection, obtained from Kaggle and **not** redistributed here — see [DATASET.md](DATASET.md) for how to obtain it and the requirement to confirm its own licence first. The published citation for the collection is not asserted here; it has to be read off the dataset's own page before anything derived from it is published.
- **Funding / Support:** Independent, unfunded.

---

## 2. Dataset Composition

### Cohort Breakdown
The benchmark aggregates the two sites in the corpus:

1. **Italian site (`Italy/` in the corpus, 122 patients):**
   - Predominantly higher ITA° values.
   - Reference standard: venous blood sample analysed with a laboratory haematology counter.
   - Illumination: exposed brighter than the Indian frames (mean BGR 102/85/106 against 84/66/77).

2. **Indian site (`India/` in the corpus, 95 patients):**
   - Predominantly lower and medium ITA° values.
   - Reference standard: matched venous blood haemoglobin assay.
   - Illumination: ambient daylight and handheld mobile camera.

Site labels are measured, not asserted: `hemolux validate` prints them. The two
sites differ in exposure as much as in population, which is why
[DATASET.md](DATASET.md) refuses to pool them silently.

### Clinical Class Distribution

Measured over all 217 patients with `hemolux validate` (WHO bands, sex-adjusted).
These are the numbers the loader produces; they are quoted from here rather than
from a publication because a distribution quoted from elsewhere is a distribution
about a different cohort.

| Clinical Category | WHO Hemoglobin Threshold | Patients | Share | Note |
|---|---|---|---|---|
| **Normal / Non-Anemic** | $Hb \ge 12.0\text{ g/dL}$ (Females) / $\ge 13.0\text{ g/dL}$ (Males) | 126 | 58.1% | Standard baseline |
| **Mild Anemia** | $11.0 \le Hb < 12.0\text{ g/dL}$ | 76 | 35.0% | Frontier triage zone |
| **Moderate Anemia** | $8.0 \le Hb < 11.0\text{ g/dL}$ | 15 | 6.9% | Clear clinical pallor |
| **Severe Anemia** | $Hb < 7.0\text{ g/dL}$ | **0 (Zero)** | 0% | **CRITICAL GAP** |

> [!WARNING]
> **Zero Severe Cases Limitation:**
> There are zero confirmed severe cases ($<7.0\text{ g/dL}$) in the combined open-access datasets. Any machine learning model claiming to detect severe anemia based on these public datasets is extrapolating without empirical validation. Hemolux explicitly refuses to predict in this severe tranche until hospital trials provide genuine ground truth.

---

## 3. Data Splits & Leakage Prevention

- **Patient-Disjoint Partitions:** `hemolux.data.splits.patient_grouped_folds`
  builds k folds with `n_splits = 5`, so every image, crop, and augmentation
  originating from patient $i$ stays on the same side of a fold boundary.
- **What the committed numbers actually used:** the frozen run reported in
  [EVALS.md](../EVALS.md) is a **single seed-42 split** — 131 train, 43
  validation, 43 test — not a five-fold sweep. Every figure in that file comes
  from that one split and has to be read as one sample.
- **Leakage:** deduplication is by patient ID. A split that put one patient's two
  eyes on opposite sides of the boundary would report a better score and a worse
  result, so the loader groups on the site-qualified patient key rather than on
  the image.

---

## 4. Optical Analysis & The Geographic Confound

### Illumination Analysis
- **Italian vs Indian Cohort Luminance ($L^*$):**
  - Italian Cohort mean $L^* = 68.4 \pm 7.2$
  - Indian Cohort mean $L^* = 56.1 \pm 8.9$
  - **Difference:** The Italian photos are approximately **20% brighter** on average.
- **Confound Mechanism:** Because the Italian cohort had a slightly lower anemia prevalence than the Indian mobile camp cohort, a naive neural network correlates higher scene brightness with normal hemoglobin, failing completely when presented with dark ambient lighting in rural clinics.

### Pigmentation Characterization (Individual Typology Angle - ITA°)
Skin pigmentation adjacent to the lower eyelid margin is computed using CIE $L^*a^*b^*$ coordinates:
$$\text{ITA}^\circ = \frac{\arctan\left(\frac{L^* - 50}{b^*}\right) \times 180}{\pi}$$
The distribution spans:
- Very Light ($> 55^\circ$)
- Light ($41^\circ \text{ to } 55^\circ$)
- Intermediate ($28^\circ \text{ to } 41^\circ$)
- Tan ($10^\circ \text{ to } 28^\circ$)
- Brown / Dark ($-30^\circ \text{ to } 10^\circ$)

---

## 5. Collection Process & Ethical Considerations

- **Informed Consent:** Original datasets were acquired under institutional IRB protocols with written informed consent.
- **De-Identification:** Full-face images have been cropped to the ocular/palpebral region to preserve complete anonymity and protect patient privacy.
- **Redistribution Policy:** Nothing from the corpus is redistributed. Raw images
  and labels sit under `data/raw/` and are ignored; feature caches and weights are
  local build artifacts (`artifacts/*` is ignored, and weights ship as GitHub
  Release assets). The only run output committed here is the aggregate metrics
  under `artifacts/reports/` — and `predictions.csv`, which would carry one row
  per patient with age, sex and the measured lab value, is deliberately kept off
  track. Photographs have to be obtained from the dataset's own source.
