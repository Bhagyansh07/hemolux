# Data Card: Hemolux Conjunctival Pallor Benchmark Dataset

Following the framework proposed by Gebru et al. (*Datasheets for Datasets*, CACM 2021).

---

## 1. Motivation

- **Purpose:** Provide a standardized, fairness-audited dataset of human palpebral conjunctiva images paired with laboratory-grade venous hemoglobin ($Hb$) measurements to train and evaluate non-invasive anemia screening models.
- **Created By:** Compiled and standardized by the Hemolux Project from published open-access clinical studies (Dimauro et al., Mannino et al.).
- **Funding / Support:** Independent academic and open-science research.

---

## 2. Dataset Composition

### Cohort Breakdown
The benchmark aggregates two distinct clinical collections:

1. **Italian Hospital Cohort (Dimauro et al., 2019):**
   - Setting: Outpatient clinic, University of Bari, Italy.
   - Subjects: Predominantly European ancestry (Fitzpatrick phototypes I–III, high ITA° values).
   - Reference standard: Venous blood sample analyzed with laboratory hematology counter.
   - Illumination: Standardized indoor clinical examination room lighting.

2. **Indian Mobile Clinic Cohort (Mannino et al. / Kaggle repository):**
   - Setting: Community and mobile screening camps in India.
   - Subjects: Predominantly South Asian ancestry (Fitzpatrick phototypes III–V, low/medium ITA° values).
   - Reference standard: Matched venous blood hemoglobin assay.
   - Illumination: Ambient natural daylight and handheld mobile camera flash.

### Clinical Class Distribution

| Clinical Category | WHO Hemoglobin Threshold | Sample Count | Percentage |
|---|---|---|---|
| **Normal / Non-Anemic** | $Hb \ge 12.0\text{ g/dL}$ (Females) / $\ge 13.0\text{ g/dL}$ (Males) | ~62% | Standard baseline |
| **Mild Anemia** | $11.0 \le Hb < 12.0\text{ g/dL}$ | ~24% | Frontier triage zone |
| **Moderate Anemia** | $8.0 \le Hb < 11.0\text{ g/dL}$ | ~14% | Clear clinical pallor |
| **Severe Anemia** | $Hb < 7.0\text{ g/dL}$ | **0 (Zero)** | **CRITICAL GAP** |

> [!WARNING]
> **Zero Severe Cases Limitation:**
> There are zero confirmed severe cases ($<7.0\text{ g/dL}$) in the combined open-access datasets. Any machine learning model claiming to detect severe anemia based on these public datasets is extrapolating without empirical validation. Hemolux explicitly refuses to predict in this severe tranche until hospital trials provide genuine ground truth.

---

## 3. Data Splits & Leakage Prevention

- **Patient-Disjoint Partitions:** 5-fold cross-validation where all images, crops, and augmentations originating from Patient $i$ reside exclusively in either Fold $k$ (training) or Fold $k$ (test).
- **Leakage Elimination:** Previous published works inadvertently placed duplicate photos or left/right eye crops of the same patient in both train and validation splits, inflating validation metrics artificially. Hemolux enforces strict patient-ID deduplication.

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
- **Redistribution Policy:** Hemolux distributes only pre-extracted numerical features, standardized bounding box coordinates, and validation split metadata. Raw patient image files must be obtained directly from original licensed repositories.
