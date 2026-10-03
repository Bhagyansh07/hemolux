# Hemolux

[![CI](https://github.com/Bhagyansh07/hemolux/actions/workflows/ci.yml/badge.svg)](https://github.com/Bhagyansh07/hemolux/actions/workflows/ci.yml)
[![Last commit](https://img.shields.io/github/last-commit/Bhagyansh07/hemolux)](https://github.com/Bhagyansh07/hemolux/commits/main)
[![License](https://img.shields.io/github/license/Bhagyansh07/hemolux)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Torch](https://img.shields.io/badge/torch-2.x%20%2B%20onnxruntime-informational)](pyproject.toml)

Non-invasive haemoglobin screening from smartphone images of the palpebral
conjunctiva, with a skin-tone fairness audit attached to every number it reports.

Haemoglobin absorbs light between roughly 490 and 577 nm and reflects it between
630 and 760 nm. A smartphone sensor sees three broad channels and cannot separate
that signal from melanin, which absorbs in an overlapping band. That is the
central problem, and it is also why the fairness audit is not an appendix to this
project — it is the same measurement problem seen from the other side.

**This is a research prototype for screening and triage support. It is not a
diagnostic device and must not be used to diagnose anaemia, to rule it out, or to
make treatment decisions.** See [ETHICS.md](ETHICS.md).

## Status

The measurement, evaluation and training library is complete and tested, and the
ONNX graph the browser will run is exported and checked against PyTorch. The web
front end and the telemetry Worker are not yet in this tree, so nothing is claimed
about a screen anyone can look at.

| Component | State |
| --- | --- |
| Dataset loader, 217 patients, two sites | implemented, 114 tests |
| Colourimetry, L\*a\*b\*, erythema index, ITA | implemented, 62 tests |
| Regression metrics, Bland-Altman with bootstrap CI | implemented, 46 tests |
| Fairness audit, subgroup error and bias-vs-ITA | implemented, 94 tests |
| Calibration, ECE, temperature scaling, risk-coverage | implemented, 75 tests |
| Patient-disjoint and site-holdout splitting | implemented, 64 tests |
| Image quality gate | implemented, 41 tests |
| Losses | implemented, 54 tests |
| Backbone builder and five regression heads | implemented, 26 tests |
| Corpus validator | implemented, 29 tests |
| Synthetic corpus fixture | implemented, 12 tests |
| Image metadata warning filter | implemented, 17 tests |
| Feature-cache fingerprint | implemented, 18 tests |
| Training loop and feature cache | implemented, 53 tests |
| Validation-only configuration sweep | implemented, 17 tests |
| ONNX export, verified against PyTorch | implemented, 21 tests |
| Command line, all six subcommands end to end | implemented, 27 tests |
| Web front end and telemetry Worker | not yet in this tree |
| Measured results | not yet published; [EVALS.md](EVALS.md) is a stub |

770 tests, all passing, `ruff check .` clean.

## The claims this project tests

Each is falsifiable, and each has code that decides it rather than asserting it.

- **C1 — Does a continuous haemoglobin estimate beat a binarised flag?**
  Five heads share one frozen backbone so the comparison is controlled:
  binary, four-way severity, direct regression, ordinal over twelve Hb bins, and
  multitask. The answer decides what ships.
- **C2 — Does masking the conjunctiva beat measuring the whole frame?**
  `extract_colour_features` accepts `None`, which measures the entire image. That
  path is deliberately worse; C2 measures by how much, because it is the control
  that says the segmentation is doing work.
- **C3 — Is the error dependent on the patient's pigmentation?**
  An OLS regression of signed bias on Individual Typology Angle.
  `fairness_gate` returns one of three verdicts. `supported` needs an interval
  that excludes zero *and* an effect large enough to act on. `not_supported`
  means the interval contains zero or the effect is negligible, and that is an
  honest negative rather than an absence of evidence. `underpowered` means the
  interval cannot be computed at all, which is a statement about 217 patients and
  not about the model.
- **C4 — Does multi-site fusion close the India to Italy gap?**
  Cross-site transfer is the hardest thing to fake here, because the two sites
  differ in exposure as well as in population. `site_holdout_folds` trains on one
  country and evaluates on the other.

## Why the evaluation code is the bulk of this repository

Reporting a haemoglobin estimate is easy. Reporting one that can be defended is
where most of the work goes, and most of that work is in refusing to report.

- **No bare numbers.** Every metric leaves `metrics/regression.py` carrying a
  spread and, where the method-comparison literature requires one, a confidence
  interval. `format_mean_std([0.87])` returns the string `single run`, not a
  number.
- **Splits are patient-disjoint, and the check is not optional.**
  `validate_no_patient_leakage` runs in `Fold.__post_init__`, so a leaking fold
  cannot be constructed without the error firing at the call site.
- **Calibration refuses to touch the test split.** `temperature_scale` requires a
  `ValidationOnly` token and raises `TypeError` otherwise. A temperature fitted on
  the data it is reported against invalidates every calibration number downstream
  of it, and nothing else in the code would notice.
- **Baselines are stated in their original units.** Cross-site MAE is printed as
  two numbers, India and Italy, because a pooled figure hides a site effect that
  is the most important thing in the result.
- **Formulas are either exact or labelled.** `metrics/colorimetry.py`
  implements documented colour statistics and says so. An invented formula that
  happens to score well would defeat the purpose of having a floor.

## Repository layout

```
src/hemolux/
  config.py              seeds, paths, input size, published reference points
  losses.py              Huber, soft-target cross-entropy, focal, class weights
  training.py            five heads, the training loop, the feature cache
  validation.py          the corpus validator behind `hemolux validate`
  fingerprint.py         what a cached feature set must notice about the code
  sweep.py               configuration selection on validation, test read once
  export.py              ONNX graph assembly, and the check against PyTorch
  cli.py                 the six subcommands
  data/
    dataset.py           workbook and mask discovery, preprocessing spec
    splits.py            patient-disjoint folds, site holdout, leakage guards
    quality.py           the image quality gate and its thresholds
  metrics/
    colorimetry.py       CIELAB, redness ratios, erythema index, ITA, b* proxy
    regression.py        MAE, RMSE, R2, EVS, Bland-Altman with bootstrap CI
    fairness.py          subgroup error, bias-vs-ITA slope, verdicts
    calibration.py       posterior mean, ECE, Brier, temperature, risk-coverage
  models/
    backbone.py          frozen-backbone builder, staged unfreezing, browser set
    heads.py             binary, severity, regression, ordinal, multitask
scripts/
  quality_sweep.py          measures the corpus to justify QUALITY_ABSTAIN
  make_synthetic_fixture.py writes a corpus in the real layout, for CI
tests/                      770 tests
```

## Dataset

The Eyes Defy Anemia corpus, from Kaggle. See `docs/DATASET.md` for provenance,
licensing, and the full description of what is and is not in it.

| | India | Italy |
| --- | --- | --- |
| Patients | 95 | 122 |
| Hb mean (g/dL) | 11.47 | 13.83 |
| Hb SD (g/dL) | 2.08 | 2.04 |
| Hb range (g/dL) | 7.6 to 17.1 | 7.0 to 17.4 |

217 of 218 workbook rows are usable. `Italy/93` is dropped because the authors
marked it `ELIMINATO`; the loader reports every drop with its reason rather than
silently shrinking the corpus. WHO severity bands across all 217 patients are 126
normal, 76 mild, 15 moderate, and **zero severe**, which is why every severity head
in this repository takes an explicit class count rather than inferring one.

The two sites differ in exposure as well as in population — Italy is uniformly
about 20% brighter — so a pooled metric is not a metric. Anything that crosses
sites goes through `site_holdout_folds`.

## Installation

```bash
git clone https://github.com/Bhagyansh07/hemolux.git
cd hemolux
python -m venv .venv
.venv\Scripts\activate            # or: source .venv/bin/activate
pip install -e ".[dev]"
```

PyTorch must come from the CPU index or pip will pull multi-gigabyte CUDA
wheels that this project never uses. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Development

```bash
pytest                                  # 770 tests
ruff check .                            # lint
hemolux validate                        # what is actually in the corpus
hemolux train                           # train every head, report, write checkpoints
hemolux export --head ordinal           # the graph the browser runs
hemolux report                          # reprint the last run's numbers
python scripts/quality_sweep.py         # corpus quality distribution
```

## Licence

MIT. See [LICENSE](LICENSE).

The dataset is not included and is not redistributed here. It contains
identifiable body images and laboratory values, which is why `data/raw/` is
ignored and no crop from it is committed.