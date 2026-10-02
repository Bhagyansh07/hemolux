<!--
Keep this file plain. What a reviewer needs from a template is a short list of
the things that are easy to forget, not a form to fill in before saying anything.
-->

## What this changes

<!-- The problem, not the implementation. If the motivation does not fit here,
     the change is probably premature. -->

## Why this way

<!-- The decision a reader could not otherwise infer. "What alternatives did you
     consider and why did they lose?" is usually the useful part of a PR. -->

## How it was verified

<!-- Required. A claim without a command that reproduces it is not a claim.

     Paste the exact invocation and the relevant output. If the verification was
     a test, name the test. -->

```
```

## Checklist

- [ ] `pytest` passes.
- [ ] `ruff check .` is clean.
- [ ] New behaviour has a test that would fail without the change.
- [ ] Expected values in new tests are derived from a closed form or an explicit
      source, not copied from a run.
- [ ] Any fixed defect says what the old code did and how the test catches it.
- [ ] No new bare numbers: a reported metric carries a spread and, where the
      method requires one, an interval.
- [ ] No new threshold without the measurement that justifies it, in a comment and
      in a script that reproduces it.
- [ ] No duplicated literal. Shared values are imported from one place.
- [ ] If this touches a measurement path, `python scripts/quality_sweep.py` still
      reproduces its distribution.
- [ ] Nothing from `data/raw/` is committed, including crops.
- [ ] If a result changed, `EVALS.md` says so — including when the result got
      worse.

## Screening language

- [ ] The wording in code, comments, docs, and commit message says **screen**,
      **estimate**, or **triage**.
- [ ] It does not say **detect**, **diagnose**, or **identify** a condition.