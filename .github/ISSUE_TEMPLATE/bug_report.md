---
name: Bug report
about: Something in the library gives a wrong or surprising answer
labels: bug
---

## What happened

<!-- The observed behaviour. What you expected is useful but secondary. -->

## What you expected instead

## Reproduction

The command, or the smallest code that shows it:

```python
```

If it depends on the dataset, say which split and which seed. If it depends on a
mask or an ROI, say which one.

## Environment

```
python -V
pip freeze | grep -Ei 'torch|timm|numpy|opencv|scikit|albument'
```

## Anything else

Output that looks wrong is usually more useful than a description of it.
Include it verbatim rather than summarising it.

<!--
Please do not include patient images, crops, or per-patient haemoglobin values in
an issue. The dataset is not public and this repository does not carry it.
-->