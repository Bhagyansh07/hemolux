# Ethics

This document is a constraint on the software, not a disclaimer appended to it.
Several of the rules below are enforced in code, and the tests that enforce them
are named so you can check.

## Intended use

Hemolux is a research prototype for **non-invasive haemoglobin screening**. Its
purpose is triage support: helping someone decide whether a clinical test is
worth their time.

It is not a diagnostic device. It does not diagnose anaemia, it does not rule
anaemia out, and it does not support a treatment decision. A haemoglobin
estimate from this software is not a laboratory result and must never be
presented as one.

## Language

The project says **screen**, **estimate**, and **triage**. It does not say
**detect**, **diagnose**, or **identify** a condition. This is not stylistic.

The word "detect" claims that the condition is present or absent. This software
cannot make that claim: it produces a number with an interval, it abstains when
its own uncertainty is too large, and it is wrong on out-of-distribution
photographs in ways the user cannot see. Language that implies a binary answer
would be a claim the code does not support.

## What a user is shown

- An estimate with an interval, never a bare number.
- An explicit **abstention** when the model is uncertain or the photograph is
  poor, in preference to a guess.
- A plain-language reason whenever a photograph is rejected, naming what to fix.
- A route to a real test. The screen exists to route people to clinical care, not
  to substitute for it.

`QualityReport.message` carries a remediation message for every rejection
reason, and `data/quality.py` guarantees each one names the specific thing to
fix, because a rejection a user cannot act on is a rejection that will be
retried identically.

## Fairness

Skin tone is not an edge case. Melanin and haemoglobin absorb in overlapping
spectral bands, so a three-channel sensor cannot cleanly separate them, and a
model trained without regard to this will have **systematic,
pigmentation-dependent error** — either worse tracking, or a bias whose sign
flips across pigmentation strata.

Therefore:

- Every reported number is accompanied by a fairness analysis, not offered
  instead of one.
- Pigmentation is measured from the image (Individual Typology Angle, conjunctival
  `b*` proxy), never from self-report. There is no demographic field to ask for.
- The fairness verdict is three-way: `supported`, `not_supported`, or
  `underpowered`. `underpowered` is an acceptable and honest outcome; quietly
  reporting a null result from an underpowered subgroup is not.
- The verdict is decided by the **confidence interval on the slope of bias
  against ITA**, not by the point estimate. An interval containing zero is an
  absence of evidence, and `metrics/fairness.py` is built so that it cannot be
  read as evidence.

Two mechanisms exist because neither is trusted alone: the hard quality gate
rejects photographs outright, and the abstention gate refuses predictions on
marginal images. An image can be caught by both.

## Known limitations, stated up front

- **Zero severe cases.** The corpus contains 15 moderate and no severe patients.
  Sensitivity in the severe band cannot be computed from this dataset at all, and
  no number in this repository should be read as covering it.
- **Two sites, two populations.** India and Italy differ in exposure as well as in
  demography. Performance in a third country is unmeasured.
- **Retrospective, controlled acquisition.** The corpus is not photographs taken
  in the conditions where this tool would actually be used. Reported performance
  is an upper bound on field performance.
- **Small sample.** 217 patients. Confidence intervals are wide, and the code is
  built to show that rather than hide it.
- **No prospective validation.** Nothing here has been tested for clinical
  outcome impact.

## Data ethics

The corpus contains identifiable body images and clinical laboratory values.
Consequently:

- `data/raw/`, `data/interim/`, `data/processed/` and `data/synthetic/` are
  ignored. No image and no patient-level value is committed to this repository,
  not even a crop.
- Derived artifacts under `artifacts/` are ignored too, because a per-patient
  prediction file is re-identifiable against a public dataset.
- Only aggregate results — never per-patient rows — are published here.

## Refusing to optimise a number

If an experiment produces a worse MAE but a fairer model, **the fairer model
ships and the worse MAE is reported**. Publishing the fairer number as the
headline result even when it is not the best one is the entire point of running
the audit alongside the model rather than after it.

## Reporting a null result

C3 may well come out refuted or underpowered. If so, this repository says so in
[EVALS.md](EVALS.md) with the interval that decided it. A benchmark that only
ever reports the configurations that worked is not a benchmark.

## Contact

Security issues or a concern about how patient data is handled should go through
GitHub's private vulnerability reporting on this repository.