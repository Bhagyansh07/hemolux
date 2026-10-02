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
`_forniceal` wins over a hypothetical shorter match rather than the other way
round. That ordering is in `MASK_SUFFIX_ORDER` and is tested.

Filenames in the corpus contain typos. `ROI_TYPOS` holds the known ones, and a
bounded Levenshtein fallback catches the rest, called with a cap of a single
edit so an unrelated filename can never be matched by a long accidental
similarity. Both mechanisms exist because an unmatched mask silently degrades a
measurement instead of failing, which is the worst possible failure mode for a
mask.

Mask areas vary enormously — the palpebral mask is 0 px at minimum, median 73 822,
maximum 11 890 760. A mask that matched nothing reads as an empty mask and is
caught by the framing check in `data/quality.py`, which rejects an ROI outside
1.5% to 85% of the frame.

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