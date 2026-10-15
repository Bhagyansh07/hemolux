# Evaluations

**One measured run. The numbers are bad, and they are here because they are what
the model actually produced.**

`artifacts/reports/results.json` is a single frozen run — seed 42, 43 held-out
patients, patient-disjoint split — over the real corpus. The training code writes
that file and this page transcribes it, so the committed number and the run that
produced it cannot drift apart. One seed is a sample, not a measurement, and
nothing here should be quoted as a headline result.

## The measured run

| Head | MAE (g/dL) | RMSE | R² | Bias |
| --- | --- | --- | --- | --- |
| binary | 2.142 | 2.493 | 0.048 | +0.334 |
| regression | 2.139 | 2.607 | −0.042 | +0.928 |
| ordinal | 2.156 | 2.538 | 0.013 | +0.620 |
| multitask | 2.158 | 2.544 | 0.009 | +0.654 |
| severity | 2.458 | 2.804 | −0.205 | −1.076 |

- All five heads sit at MAE ≈ 2.1–2.5 g/dL. Point-of-care acceptability is
  conventionally around 1.0 g/dL (`clinical_acceptable_error_gdl`), so every head
  is roughly twice the acceptable error.
- R² is at or below zero for every head but the binary one. On this corpus the
  model explains almost none of the between-patient haemoglobin variance.
- Every head but `severity` carries a positive bias: the model tends to
  over-estimate haemoglobin on this split.

### Selective prediction did not help on this run

The ordinal head's risk-coverage curve runs from **MAE 2.156 at full coverage to
2.264 at 25% coverage** (`coverage.ordinal` in the same file). Rejecting the
least-confident acquisitions *raised* mean absolute error rather than lowering it.
The other four heads expose no ranking signal at all, so their curves are marked
`unavailable` rather than drawn. The abstention gate therefore stays in the
product as a safety behaviour — do not guess when the photograph is poor — and
not as an accuracy feature, until a curve shows otherwise.

## How to read anything added here

Every reported figure carries:

| Field | Why it is required |
| --- | --- |
| MAE with an interval | A bare MAE from one run is not a result |
| Split | Patient-disjoint, and stated by name |
| Seeds | A single seed is a sample, not a measurement |
| Selection basis | Read on validation, or the number is not a result at all |
| n | 217 patients, and fewer in any particular split |

The last row is the one that catches people. A configuration chosen by looking at
test scores produces a better test number and a worse paper. Hemolux selects on
the validation split and reads test once.

## Reference points from the literature

These are what a result would be compared against. They are compiled in
`hemolux.config.PUBLISHED_REFERENCE` and are **not** thresholds this project
asserts or has reproduced.

| Quantity | Value | Key |
| --- | --- | --- |
| R², erythema-index method (2016) | 0.27 | `erythema_index_2016_r2` |
| R², forniceal-only (2016) | 0.09 | `forniceal_only_2016_r2` |
| R², label-distribution learning (2022) | 0.512 | `label_distribution_2022_r2` |
| R², Mask R-CNN segmentation (2023) | 0.503 | `mask_rcnn_2023_r2` |
| R², same, unsegmented frames | 0.306 | `mask_rcnn_2023_unsegmented_r2` |
| MAE, Hemo-ConViT (2025) | 0.870 | `hemoconvit_2025_mae` |
| R², Hemo-ConViT (2025) | 0.761 | `hemoconvit_2025_r2` |
| Smartphone pallor method (2021) | 4.73 | `smartphone_2021_loa` |

Two entries need care when they are eventually used for comparison:

- The 2016 forniceal-only result (R² 0.09) is the reason `dataset.py` defaults to
  the **palpebral** ROI rather than the forniceal one. That choice is experiment
  C2, and it is a choice, not a settled fact.
- `smartphone_2021_loa` is on a different scale from everything else here. It
  must not appear in the same table without its unit.

Clinical acceptability for point-of-care haemoglobin devices is conventionally
around 1.0 g/dL (`clinical_acceptable_error_gdl` in `config.py`). No Hemolux
result should be presented as meeting it until an interval says so.

Full citations are still to be added here. The table is not a bibliography, and
until it is one it should not be quoted as one.

## C1 — continuous estimate versus a binarised flag

**Measured, once.** Five heads trained over one frozen backbone. On the 43-patient
test fold the ranking by MAE is regression (2.139) < binary (2.142) < ordinal
(2.156) < multitask (2.158) < severity (2.458); the regression and binary heads
are within 0.003 g/dL of each other, which is a tie at one seed. The stated
decision rule is the best MAE *among heads whose fairness audit does not refute
them*, and C3 is unmeasured, so **no head has cleared the decision rule yet**. The
default export is the ordinal head because it is the only one that also emits a
distribution, which is what the abstention gate needs — not because it won on MAE.

## C2 — masked ROI versus whole frame

**Not measured as a controlled comparison.** The unsegmented condition is
available by passing `None` to `extract_colour_features`, and
`hemolux.data.dataset.ConjunctivaDataset` can emit it, off by default. The frozen
sweep varies the ROI but does not run a whole-frame control beside a masked one,
so the value of the segmentation is still unquantified. Decision rule: report
both, and the gap between them is what the segmentation is worth.

## C3 — pigmentation-dependent error

**Not measured in the frozen run.** `fairness_gate` returns one of `supported`,
`not_supported`, or `underpowered`, decided by the confidence interval on the
slope of signed bias regressed on Individual Typology Angle. The frozen report
writes no bias-vs-ITA slope, so C3 has no number attached yet.

The most likely outcome from 217 patients is `underpowered`, because that verdict
is what a slope returns when fewer than 20 samples have a finite ITA. If that is
what comes out, it is what gets written here, with the interval.

## C4 — multi-site fusion and cross-site transfer

**Measured, once, and the answer is negative.** Training on one site and testing
on the other:

| Fold | Head | MAE (g/dL) | R² |
| --- | --- | --- | --- |
| India → Italy | regression | 4.357 | −4.836 |
| India → Italy | ordinal | 2.913 | −1.494 |
| Italy → India | regression | 3.300 | −2.358 |
| Italy → India | ordinal | 3.151 | −2.110 |

Cross-site transfer is worse than the pooled split on every axis, with strongly
negative R², and the two sites differ in exposure as well as population (Italy is
about 20% brighter), so part of that gap is lighting rather than biology.
Multi-site fusion does not close the gap on this run. Decision rule: a positive
transfer number with a worse pooled MAE is not a win; here there is no positive
transfer number to argue about.

## Figures

Published figures land in `artifacts/figures/`, which is tracked. Anything
derived from the real dataset is derived under the terms in
[docs/DATASET.md](docs/DATASET.md); no image crop is committed.

## Adding a result

Do not edit this file by hand from console output. The training code writes
`artifacts/reports/results.json` and `artifacts/reports/results.csv`, and the
figures below are transcribed from that file so the committed number and the run
that produced it cannot drift apart.