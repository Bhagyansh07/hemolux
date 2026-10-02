# Dataset

## What this project uses

The **Eyes Defy Anemia** corpus: smartphone photographs of the palpebral
conjunctiva with paired laboratory haemoglobin values, collected at two sites.

It is obtained from Kaggle and is **not** redistributed in this repository. It
contains identifiable body images and clinical laboratory values, so `data/raw/`
is ignored and no crop from it is committed anywhere in the history.

**Confirm the dataset's own licence and terms of use before publishing anything
derived from it.** That check is a prerequisite for any release from this
project, and it is not something this repository can do on the author's behalf.

## Layout on disk

```
dataset anemia/
  India/
    India.xlsx                     label table, one row per patient
    <patient>/                     one directory per patient
      <image>.jpg                  the photograph
      <image>_palpebral.png        conjunctival mask
      <image>_forniceal.png        deeper fornix mask
  Italy/
    Italy.xlsx
    <patient>/
      ...
```

The expected root is `hemolux.data.dataset.DEFAULT_ROOT`, relative to the
repository root. It is defined in one place because two tools need it and a
duplicated path literal is how two programs end up disagreeing about which
dataset produced a number.

## What is in it

Measured by `build_records`, not quoted from the publication:

| | India | Italy |
| --- | --- | --- |
| Patients | 95 | 122 |
| Hb mean (g/dL) | 11.47 | 13.83 |
| Hb SD (g/dL) | 2.08 | 2.04 |
| Hb range (g/dL) | 7.6 to 17.1 | 7.0 to 17.4 |

218 rows appear in the workbooks. **217 are usable.** `Italy/93` is dropped
because the authors marked it `ELIMINATO`; the loader lists every dropped patient
with its reason rather than quietly returning fewer rows.

Download size is roughly 650 MB across 865 files, of which 862 images decode.

## Severity bands

WHO bands, sex-adjusted, over all 217 patients:

| Band | Patients |
| --- | --- |
| Normal | 126 |
| Mild | 76 |
| Moderate | 15 |
| Severe | **0** |

**There are no severe cases.** This is the single most consequential fact about
the corpus. Sensitivity in the severe band cannot be computed from this dataset,
and no aggregate number here should be read as covering it.

It also has a direct engineering consequence. Because the highest band is empty,
`InverseFrequencyWeights` derives its class count from the largest index actually
present, so a four-band head silently gets a three-element weight vector. Passing
that to a cross-entropy raises rather than training on a truncated weighting:

```
RuntimeError: weight tensor should be defined either for all 4 classes or no
classes but got weight tensor of shape: [3]
```

Every severity head in this repository therefore takes an explicit class count.
The absent class is not dropped, either: it receives the smallest weight rather
than being skipped.

## The two sites are not interchangeable

India and Italy differ in more than demography. Mean frame brightness is roughly
20% higher at the Italian site, which is an exposure offset, and it behaves like
a confound: it shifts channel ratios in the same direction the haemoglobin signal
does.

Consequences baked into the code:

- Nothing is reported as a single pooled number across sites unless it is
  labelled as one. Cross-site MAE is printed per site.
- `site_holdout_folds` trains on one country and evaluates on the other. A
  patient with a missing or blank site label is **rejected**, because such a
  patient silently vanishes from both directions of a holdout and quietly
  shrinks the population a reported MAE was computed over.

## Patient identifiers

Subjects are site-qualified: `India/12`, `Italy/93`. The site prefix is part of
the identifier rather than a directory detail, because `12` in one country is not
`12` in another and a patient key built on the bare number collides.

## Masks

Masks are matched to their photograph by suffix, longest suffix first, so
`forniceal_palpebral` is never mistaken for `forniceal`. That ordering is in
`MASK_SUFFIX_ORDER` and is tested.

Filenames in the corpus contain typos. `ROI_TYPOS` holds the known ones, and a
bounded Levenshtein fallback catches the rest, called with a cap of a single
edit so an unrelated filename can never be matched by a long accidental
similarity. Both mechanisms exist because an unmatched mask silently degrades a
measurement instead of failing, which is the worst possible failure mode for a
mask.

**Masks are not stored at the frame resolution.** Every frame is 2988×3984, and
211 of the 217 palpebral masks are 800×1067 — roughly a quarter of the area. The
loader resizes with nearest-neighbour interpolation, because a mask is a region
label and interpolating one invents boundary pixels nobody segmented.

That is why mask area is reported below as a **fraction** of the frame rather than
in pixels. A fraction is resolution-independent; a pixel count is not, and mixing
the two produced a wrong number here before it was measured properly.

Six patients have no forniceal mask at all: `Italy/1`, `Italy/35`, `Italy/54`,
`Italy/58`, `Italy/75` and `Italy/109`. The default ROI is palpebral, which all
217 patients have, so the default path is unaffected.

One palpebral mask *is* stored at full frame resolution, which makes it the
outlier in every table below and is the sole reason the maximum area looks like it
does. It is not a failure of matching — its fill fraction of 0.9999 is a fact about
the mask.

### Measured mask geometry

Measured over the 217 patients with `scripts/validate_dataset.py`, which prints
this table rather than having it transcribed.

| ROI | matched | area fraction: min | p10 | median | max | below 1.5% | above 85% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| palpebral | 217/217 | 0.000001 | 0.0013 | 0.0975 | 0.9999 | 77 | 31 |
| forniceal | 211/217 | 0.000986 | 0.0353 | 0.1411 | 0.9996 | 8 | 30 |
| forniceal_palpebral | 211/217 | 0.050677 | 0.1258 | 0.2046 | 0.9989 | 0 | 30 |

In pixels, the palpebral mask runs from **1 px** to 11 865 803 px, median 83 190.

**108 of the 217 palpebral masks fall outside the 1.5%–85% band** the quality gate
applies — 77 too small, 31 too large. `India/3` is the extreme small case at a
single pixel; `Italy/2` is the extreme large one at 99.99% of the frame, which is
to say its mask is the photograph.

This is the most consequential thing in this file and it was invisible until the
validator measured it. Roughly half the default ROI is unusable as drawn, so any
result computed over the palpebral ROI is computed over a population that is about
half degenerate, and the figure moves depending on how the gate treats those
patients. Whatever that treatment is, it has to be stated next to the number
rather than left to the reader of `data/quality.py`.

## Preprocessing

`PREPROCESS_SPEC` in `hemolux.data.dataset` is the single source of truth for
what happens to a frame, and the emitted ONNX graph carries the same
specification so the browser transformer can be checked against the Python one
rather than reimplemented from memory.

ROI selection is masking and cropping, and it happens **before**
`build_preprocess()` is called. `build_preprocess()` takes no `roi` argument; an
earlier version did, and passing a region to it meant the crop happened twice
with the mask applied at the wrong stage.

## Synthetic fixture

`python scripts/make_synthetic_fixture.py` writes a corpus in this same layout
for tests and CI. Everything it produces is prefixed `SYNTHETIC_` so a synthetic
result can never be mistaken for a real one.