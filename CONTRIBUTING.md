# Contributing

## Setup

PyTorch must be installed from the CPU index. PyPI's default `torch` wheel
bundles CUDA and pulls several gigabytes of libraries this project never uses;
the `--index-url` below is not optional.

```bash
git clone https://github.com/Bhagyansh07/hemolux.git
cd hemolux

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[dev]"
```

Verified on Windows and Python 3.13.1 with the versions pinned in
`pyproject.toml`. The pins are exact on purpose: this project reports numbers,
and a silent version change is a silent change to the numbers.

If `pip install -e ".[dev]"` reports a torch resolution error, the first command
did not take effect. Check `pip show torch` before continuing.

## Running the checks

```bash
pytest                    # the full suite
ruff check .              # lint
pytest tests/test_fairness.py -v
python scripts/validate_dataset.py      # what is actually in the corpus
python scripts/quality_sweep.py         # the quality distribution
```

`pyproject.toml` sets `testpaths = ["tests"]` and `--strict-markers`. Tests
needing the real dataset are marked `slow` and skip cleanly without it.

## The dataset

`data/raw/` is ignored and nothing from it is committed. Get the Eyes Defy Anemia
corpus from Kaggle and unpack it to the path in
`hemolux.data.dataset.DEFAULT_ROOT`. See [docs/DATASET.md](docs/DATASET.md).

`python scripts/make_synthetic_fixture.py` writes a small synthetic corpus in the
same layout for tests and CI, and `python scripts/validate_dataset.py --synthetic`
checks it. Every artifact the generator writes is prefixed `SYNTHETIC_`, so a
synthetic result can never be mistaken for a real one.

Run the validator against the real corpus before trusting a number from it. It
exits non-zero on anything that would make a reported figure wrong rather than
merely imprecise, and it prints what it found about the corpus either way: the
mask resolution mismatch, the patients with no forniceal mask, and the empty severe
band are all facts about the data that a result has to respect.

## Rules this repository holds itself to

These are the review criteria. A change that violates one of them is not ready
regardless of how well it scores.

**Every behaviour gets a test.** A new function without a test is unfinished, not
unfinished-and-drafted. Tests assert properties, not recordings — a test whose
expected values were copied from a run detects a change, it does not verify a
correctness.

**Expected values are derived, not captured.** Where a number has a closed form,
work it out in the test and put the derivation in a comment. If there is no
closed form, say why the expected value is what it is.

**A defect fix names the defect.** Say what the old code did, why it was wrong,
and how the new test catches it. If the fix has no test, it is not fixed.

**No bare numbers.** A reported metric carries a spread and, where the
method-comparison literature requires one, a confidence interval. Code that would
let a single run escape as a result is a bug.

**Thresholds are measured, not chosen.** A constant that gates behaviour needs the
corpus measurement that justifies it in a comment, and a script that reproduces
the measurement. `scripts/quality_sweep.py` and `QUALITY_ABSTAIN` are the
worked example: the original 0.35 sat below every score in the corpus, so the
abstention gate's image-quality branch could never fire.

**Duplicated literals are bugs.** If two places need the same value, one of them
imports it. `should_abstain` once compared quality against a hand-typed `0.35`
while the module defining the threshold carried a different value; raising the
constant changed nothing, and the branch was inert.

**Preserve a check rather than deleting it.** If a check is wrong, fix the code
it caught, and keep the check.

**Prefer rejecting to guessing.** Where an input is ambiguous, raise. The loader
reports every dropped patient with a reason instead of silently returning fewer
rows.

## Commit messages

Describe what changed and why, in prose. The body earns its place by explaining
a decision a reader could not otherwise infer — particularly, why a number is
what it is, and what measurement established it.

Reference the claim an experiment tests (`C1`, `C2`, `C3`, `C4`) when one
applies. Those labels are defined in [README.md](README.md).

Do not add `Co-Authored-By` trailers. This repository has a single author and the
commit history is intended to reflect that.

## Adding a metric or a claim

1. Write the decision rule first: what result would support it, what would refute
   it, and what would leave it undecided.
2. Implement it so that the rule is a function returning a verdict, not a number
   the reader has to interpret.
3. Test the rule at the boundary, not just the typical case.
4. Add the result to [EVALS.md](EVALS.md) even when it is null or unfavourable.