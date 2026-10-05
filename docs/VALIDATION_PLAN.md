# Clinical Validation Protocol & Prospective Study Plan

> **Status:** `PLANNED` | **Target Phase:** Prospective Phase II Observational Study  
> **Document Version:** 1.0.0 | **Author:** Hemolux Clinical Research & Fairness Working Group

---

## 1. Executive Summary & Clinical Rationale

According to India's National Family Health Survey (NFHS-5), **57% of women aged 15–49 years and 67% of children under 5 suffer from anemia**, up from 53% in NFHS-4. Traditional venous blood sampling via automated hematology counters remains the diagnostic gold standard, but frontline community settings face profound bottlenecks:
1. Shortage of phlebotomists and disposable needles.
2. Cold-chain and specimen transport degradation.
3. Patient reluctance (fear of needles, cultural hesitation).

Non-invasive screening via palpebral conjunctiva photography offers high potential, but existing academic models suffer from severe uncalibrated demographic bias, optical illumination confounds (e.g., Italy +20% luminance), and complete absence of severe anemia ground truth. This protocol outlines a prospective observational validation study designed to address these gaps under rigorous clinical governance.

---

## 2. Institutional Framework & Ethics Governance

### Ethics Approval (IEC)
- **Oversight:** Application to be submitted to an ICMR-registered Institutional Ethics Committee (IEC).
- **Guidelines:** Adherence to the *National Ethical Guidelines for Biomedical and Health Research Involving Human Participants* (ICMR, 2017) and the *Declaration of Helsinki*.
- **Patient Consent:** Signed informed consent obtained from all literate participants; witnessed thumbprint and verbal assent obtained for non-literate participants. All patient information sheets provided in bilingual English and Hindi.
- **Data Privacy & De-Identification:**
  - Zero facial images stored. Only cropped ocular bounding boxes ($224 \times 224\text{ px}$) retained.
  - Strict de-identification assigning cryptographic alphanumeric IDs (e.g., `HX-DEL-001`).

### Collaborating Partner Architecture (Planned)
- **Primary Clinical Site:** Secondary/Tertiary District Hospital or Medical College Department of Hematology/Pathology (e.g., Northern/Central India).
- **Community Outreach Partner:** Rural health NGO operating mobile screening units and ASHA/ANM field workers in high-prevalence rural districts.

---

## 3. Study Design & Cohort Stratification

- **Study Type:** Prospective, single-blind, observational cross-sectional diagnostic accuracy study.
- **Reference Standard:** Venous blood draw (2 mL K2-EDTA tube) analyzed within 30 minutes on a calibrated laboratory 5-part automated hematology analyzer (Sysmex XN-series or Beckman Coulter DxH).
- **Index Test:** Hemolux mobile conjunctiva capture performed immediately prior to phlebotomy.

### Target Cohort Size ($N = 500$)
To ensure statistical power for demographic fairness and the critical severe anemia tranche:

| Stratum | Target Sub-cohort | Key Inclusion Criteria |
|---|---|---|
| **Severe Anemia ($Hb < 7.0\text{ g/dL}$)** | $N \ge 60$ | Dedicated recruitment from inpatient hematology & maternity triage. |
| **Moderate Anemia ($8.0 \le Hb < 11.0\text{ g/dL}$)** | $N \ge 150$ | Antenatal clinics, general medicine OPD. |
| **Mild Anemia ($11.0 \le Hb < 12.0\text{ g/dL}$)** | $N \ge 100$ | Community screening camps. |
| **Normal / Non-Anemic ($\ge 12.0\text{ g/dL}$)** | $N \ge 190$ | Healthy controls and routine health checkups. |

### Pigmentation (ITA°) Quota
Following FDA guidance for optical physiological sensors (e.g., pulse oximeters), at least **35% of the total cohort** must fall within Fitzpatrick phototypes IV–VI (ITA° $< 28^\circ$, Deeply Pigmented / Dark Brown).

---

## 4. Optical Acquisition & Standardization Protocol

1. **Patient Positioning:** Seated comfortably in neutral, non-glare illumination.
2. **Conjunctival Eversion:** Gently pulling downward on the inferior eyelid margin using a clean gloved thumb or sterile gauze to expose the palpebral conjunctiva without blanching microvasculature.
3. **Lighting Quality Gate:** Hemolux client-side automated coach verifies:
   - CIE $L^*$ luminance between 45 and 80.
   - Glint / specular reflection $< 3\%$ of tissue area.
   - Sharpness gradient (Laplacian variance $> 100$).
4. **Selective Abstention:** Any acquisition flagged with optical instability is rejected, and health workers are guided to reposition.

---

## 5. Statistical Endpoints & Acceptance Criteria

1. **Agreement (Primary Endpoint):**
   - Bland-Altman Mean Difference (Bias) within $\pm 0.4\text{ g/dL}$.
   - 95% Limits of Agreement (LoA) within $\pm 1.50\text{ g/dL}$ vs reference analyzer.
2. **Screening Efficacy (Secondary Endpoint):**
   - Sensitivity $\ge 90.0\%$ for detecting any anemia ($Hb < 11.0\text{ g/dL}$).
   - Specificity $\ge 80.0\%$.
   - Positive Predictive Value (PPV) and Negative Predictive Value (NPV) reported with 95% Clopper-Pearson confidence intervals.
3. **Fairness Gate:**
   - Linear regression of bias against skin tone:
   $$\text{Bias} = m \cdot \text{ITA}^\circ + c$$
   - Protocol pass criterion: $|m| < 0.02\text{ g/dL per }^\circ\text{ITA}$ with $p > 0.05$ (no statistically significant racial/pigmentation disparity).

---

## 6. Medical Advisor & Hematologist Collaboration Framework

- **Transparent Attribution Policy:** In accordance with scientific integrity rules, clinical advisors and hematologists will only be named upon written institutional consent following formal protocol review. No unauthorized or unverified endorsements are permitted.
- **Advisory Role:** Reviewing eversion ergonomic safety, evaluating false negative clinical risks, and confirming laboratory quality control procedures (daily three-level QC on Sysmex/Coulter analyzers).
