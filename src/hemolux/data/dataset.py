"""Dataset loading for the Eyes Defy Anemia corpus.

On-disk layout, as released
---------------------------
Verified against the real download, not inferred. The tree is two levels deep and
the site is a **directory**, not a column::

    dataset anemia/
      India.xlsx                       <- 95 rows, one per patient
      Italy.xlsx                       <- 123 rows
      India/1/20200118_164733.jpg
              20200118_164733_forniceal.png
              20200118_164733_palpebral.png
              20200118_164733_forniceal_palpebral.png
      Italy/1/1.jpg
             001_palpebral.png

Three properties of the release that a loader has to be built around:

**There is one workbook per site, not one workbook.** ``site`` comes from the
path. Neither sheet has a site column.

**Filenames are matched by suffix, never by prefix.** India uses a capture
timestamp (``20200118_164733``); Italy patient 1 uses the folder number
zero-padded to three digits (``001_palpebral.png``). Parsing the prefix to
reconstruct a name would break on one of the two sites, so masks are found by
suffix and the prefix is never interpreted.

**Some folders contain download artefacts.** ``India/8`` holds
``..._palpebral(1).png`` alongside ``..._palpebral.png`` -- byte-identical
duplicates from the uploader's browser (SHA-256 confirmed). Matching by suffix
alone would return two candidates and silently pick one at random.

**Some filenames are misspelled.** ``India/7`` ships
``20200124_202058_papebral.png`` and ``Italy/95`` ships
``T_64_20190612_092742_palplebral.png``. A strict suffix check drops the default-ROI
mask for both patients, which trains them on a whole frame while the metadata
claims a palpebral ROI -- and errors on nothing. See :data:`ROI_TYPOS` and
:func:`mask_matches`.

**The document is wrong about the missing masks.** ``Dataset anemia.docx`` and the
``Note`` column say 30 Italian patients need their fornix segmented
(``"da segmentare la forniceale"``), but only 6 folders actually lack a forniceal
mask: numbers 1, 35, 54, 58, 75, 109. The free-text column is stale. Mask
availability is therefore read from disk, never from the note.

The Hb column needs a dedicated parser
--------------------------------------
``Italy.xlsx`` stores 15 of its 123 haemoglobin values as **text with a European
decimal comma** (``'15,1'``, ``'13,7'``), which makes the whole column ``object``
dtype, and one patient (number 93) as ``'_'`` with the note "Hgb not available".

This is the most dangerous cell in the dataset. ``pd.to_numeric(errors="coerce")``
turns those 15 good measurements into NaN and drops 12% of one site -- and not
randomly: they are patients 53 to 80, one contiguous block. The surviving subset
is therefore biased *by construction*, and it is exactly the per-site Hb
distribution that the fairness analysis compares. A silently truncated Italy
column would make the site comparison wrong in a way that looks like a finding.

So :func:`parse_hb` understands the decimal comma, and :func:`build_records`
**counts and reports** every value it had to drop.

One patient is withdrawn by the authors
--------------------------------------
``Italy.xlsx`` patient 93 holds ``Hgb = '_'``, the note "Hgb not available", and
``ELIMINATO`` in an *unnamed* column. It is the only such record, so the loader
returns **217 of 218** folders: 95 India, 122 Italy.

What the corpus actually looks like, after all of the above
----------------------------------------------------------
============================  =====  =====================  ==================
Site                          n     Hb mean +/- sd (g/dL)   Age mean
============================  =====  =====================  ==================
India (Karapakkam, Chennai)    95    11.47 +/- 2.08         33.7
Italy (Bari)                  122    13.83 +/- 2.04         49.3
============================  =====  =====================  ==================

Two properties of this table shape the whole project, and both are easy to lose.

**Anaemia prevalence differs 3.2x between sites.** At the WHO adult threshold of
12 g/dL, 58.9% of the Indian cohort is anaemic against 18.7% of the Italian one.
A model trained on the pooled set is therefore mostly learning the Indian
condition, and a site holdout is a genuinely hard test rather than a formality.

**Site is confounded with age and sex.** The Italian cohort is 15.6 years older
on average and 67% male against 52%. A site holdout changes skin tone *and* age
*and* sex at once, so a cross-site gap cannot be attributed to skin tone alone.
This has to be stated wherever a cross-site number is reported.

Both figures are measured by :func:`build_records` on every run and printed, so a
silent data change would show up as a changed line in the log.

On not guessing the spreadsheet schema
--------------------------------------
Column names are still resolved through accepted aliases and still **raise** when
the haemoglobin column cannot be identified, rather than guessing. A loader that
silently picks the wrong column produces a training run that trains on noise and
reports a plausible R², which is the single most expensive failure mode in this
project.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hemolux.config import HB_VALID_MAX, HB_VALID_MIN, IMAGE_SIZE
from hemolux.data.splits import SITES

__all__ = [
    "DEFAULT_ROI",
    "PREPROCESS_SPEC",
    "ConjunctivaDataset",
    "PatientRecord",
    "apply_mask",
    "build_preprocess",
    "build_records",
    "crop_to_mask_bbox",
    "discover_patient_dirs",
    "load_mask",
    "load_rgb",
    "read_label_table",
    "resolve_label_columns",
]

FloatArr = NDArray[np.float64]


# --------------------------------------------------------------------------- #
# ROI configuration
# --------------------------------------------------------------------------- #

#: Mask filename **suffix** per ROI, matched against the stem.
#:
#: Suffixes rather than exact names, because the release is inconsistent: India
#: prefixes every mask with a capture timestamp (``20200118_164733_palpebral``)
#: while ``Italy/1`` uses the folder number zero-padded to three digits
#: (``001_palpebral``). See the module docstring.
MASK_SUFFIXES: dict[str, str] = {
    "forniceal": "forniceal",
    "palpebral": "palpebral",
    "forniceal_palpebral": "forniceal_palpebral",
}

#: Longest first, so ``forniceal_palpebral`` is never mistaken for ``forniceal``.
#: Without this ordering a stem ending ``forniceal_palpebral`` matches both.
MASK_SUFFIX_ORDER: tuple[str, ...] = (
    "forniceal_palpebral",
    "forniceal",
    "palpebral",
)

#: This project trains on the palpebral ROI. Published work on this corpus puts
#: R2 at 0.27 for forniceal against 0.09 for the forniceal-only condition, so
#: the ROI choice is not cosmetic. The other ROIs stay loadable because comparing
#: them is experiment C2, not because they are expected to win.
DEFAULT_ROI = "palpebral"

#: Where the Kaggle archive unpacks, relative to the repository root.
#:
#: Lives here rather than in the CLI because a second consumer needs it:
#: ``scripts/quality_sweep.py`` measures the same corpus to justify the abstention
#: threshold, and a duplicated path literal is how two tools end up disagreeing
#: about which dataset produced a number.
DEFAULT_ROOT = "data/raw/eyes-defy-anemia/dataset anemia"

#: Emitted alongside the ONNX graph so the browser transformer can be checked
#: against the Python one rather than reimplemented from memory.
PREPROCESS_SPEC: dict[str, object] = {
    "color_space": "RGB",
    "input_size": [IMAGE_SIZE, IMAGE_SIZE],
    "resize_interpolation": "bilinear",
    "scale": 1.0 / 255.0,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "roi": DEFAULT_ROI,
    "mask_threshold": 127,
    "layout": "NCHW",
    "dtype": "float32",
}


# --------------------------------------------------------------------------- #
# Label table
# --------------------------------------------------------------------------- #

#: Accepted spellings for each logical field, lower-cased and stripped of
#: non-alphanumerics before comparison so ``Hb (g/dL)`` and ``hb_gdl`` match.
#: ``Number`` is the actual header in both released workbooks, added after the
#: real ``India.xlsx`` / ``Italy.xlsx`` rejected an alias list that lacked it.
#: The other spellings are kept for the secondary sources.
_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "patient_id": (
        "number",
        "patientid",
        "patient",
        "id",
        "subject",
        "subjectid",
        "code",
        "casename",
        "patientnumber",
        "recnumber",
        "n",
    ),
    "hb": ("hb", "hgb", "hbgdl", "hemoglobin", "haemoglobin", "hbvalue", "hbconc"),
    "sex": ("sex", "gender", "s", "sexo"),
    "age": ("age", "ageyears", "years", "eta"),
    "site": ("site", "country", "location", "centre", "center", "institution"),
    "note": ("note", "notes", "comment", "comments", "remark", "remarks"),
}

_HB_ALIAS_KEYS = _LABEL_ALIASES["hb"]


def _normalise_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


#: Cell values that mean "this measurement does not exist" rather than a number.
#: ``Italy.xlsx`` patient 93 carries ``'_'`` with the note "Hgb not available",
#: and is also flagged ``ELIMINATO`` in an unnamed column. Anything in this set is
#: a missing measurement, never a zero.
HB_MISSING_MARKERS: frozenset[str] = frozenset(
    {"", "_", "-", "--", "n/a", "na", "nan", "none", "null"}
)


def parse_hb(value: object) -> float | None:
    """Parse one haemoglobin cell, returning ``None`` when it is not a number.

    Handles the European decimal comma, because ``Italy.xlsx`` really does store
    15 of its values that way and pandas therefore types the entire column as
    ``object``. See the module docstring for why dropping them silently would
    corrupt the fairness analysis.

    Accepted forms: ``13.7``, ``"13.7"``, ``"13,7"``, ``" 13,7 "``, ``13``.
    Ambiguous input raises rather than guessing, because a swapped decimal point
    is a 10x error in a clinical range:

    * both a dot and a comma (``"13.7,1"``) -- genuinely undecidable;
    * a comma with more than one digit after it (``"1,234"``) -- this is a
      thousands separator, not a decimal, and is rejected as out of range rather
      than read as ``1.234``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return None if isinstance(value, float) and not np.isfinite(value) else float(value)

    text = str(value).strip()
    if text.lower() in HB_MISSING_MARKERS:
        return None

    has_dot, has_comma = "." in text, "," in text
    if has_dot and has_comma:
        raise ValueError(
            f"haemoglobin {text!r} contains both '.' and ','; the decimal "
            "separator is ambiguous and guessing could shift the value by 10x"
        )
    if has_comma:
        head, _, tail = text.partition(",")
        if len(tail) != 1 and head.lstrip("+-").isdigit() and tail.isdigit():
            # '1,234' / '1,234,567' -- a grouped thousands separator.
            raise ValueError(
                f"haemoglobin {text!r} looks like a thousands separator, not a decimal"
            )
        text = f"{head}.{tail}"

    try:
        out = float(text)
    except ValueError:
        return None
    return out if np.isfinite(out) else None


#: Free-text markers in any column meaning the record was withdrawn by the
#: authors. ``Italy.xlsx`` flags patient 93 ``ELIMINATO``; the same row is also
#: the one with no Hb, so the two signals agree here. Checked anyway, because a
#: dataset can mark a record withdrawn while keeping its measurement, and
#: training on a withdrawn record is worse than training without its label.
EXCLUDED_MARKERS: tuple[str, ...] = (
    "eliminato",
    "eliminata",
    "excluded",
    "exclude",
    "removed",
    "drop",
)


def is_excluded(row: dict[str, object]) -> str | None:
    """Return the marker that withdraws this record, or ``None`` if it is kept.

    Scans every column, because the flag in this release sits in an *unnamed*
    column (``Unnamed: 6``) and a lookup by name would miss it.
    """
    for column, value in row.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        text = str(value).strip().lower()
        for marker in EXCLUDED_MARKERS:
            if marker in text:
                return f"{marker!r} in column {column!r}"
    return None


def resolve_label_columns(columns: Iterable[str]) -> dict[str, str]:
    """Map logical field names onto actual spreadsheet columns.

    Raises
    ------
    ValueError
        If the haemoglobin column cannot be identified. Guessing here would
        produce a run that trains on the wrong numbers and reports a confident,
        wrong R².
    """
    normalised = {_normalise_key(c): c for c in columns}
    out: dict[str, str] = {}
    for field_name, aliases in _LABEL_ALIASES.items():
        for alias in aliases:
            if alias in normalised:
                out[field_name] = normalised[alias]
                break

    if "hb" not in out:
        raise ValueError(
            "cannot identify the haemoglobin column. "
            f"Columns seen: {sorted(normalised.values())}. "
            f"Add one of {sorted(_HB_ALIAS_KEYS)} to the workbook header, or pass "
            "an explicit column mapping."
        )
    return out


def read_label_table(path: Path) -> pd.DataFrame:
    """Read the Hb spreadsheet, whatever of xlsx/xls/csv it happens to be."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".xls":
        return pd.read_excel(path, engine="xlrd")
    # Unknown extension: let pandas try, and report its own error if it cannot.
    return pd.read_excel(path)


# --------------------------------------------------------------------------- #
# Patient records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PatientRecord:
    """One patient's image, its ROI masks and its ground-truth Hb.

    ``patient_id`` is **site-qualified** (``"India/12"``), which is not
    decoration. Both sites number their folders from 1, so there are two patients
    called ``1``. Using the bare folder number as the identity would let
    ``India/5`` and ``Italy/5`` collide, and the consequence lands exactly where
    this project must be trustworthy: a patient-disjoint split could place one
    in train and the other in test, and ``ConjunctivaDataset.subset`` could
    return the Italian patient when asked for the Indian one. The site prefix
    makes the collision impossible rather than unlikely.
    """

    patient_id: str
    image: Path
    masks: dict[str, Path]
    hb: float
    sex: str
    site: str
    age: float | None = None
    note: str = ""

    @property
    def mask(self) -> Path | None:
        """The mask for :data:`DEFAULT_ROI`, if that patient has one."""
        return self.masks.get(DEFAULT_ROI)

    @property
    def local_id(self) -> str:
        """The site's own number for this patient, e.g. ``"12"``."""
        return self.patient_id.split("/", 1)[-1]


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


#: Real misspellings of the ROI names, present in the release, mapped to the
#: correct ROI. Found by listing the actual filenames rather than by assuming the
#: documented ones:
#:
#: * ``India/7``  -> ``20200124_202058_papebral.png``  (letters transposed)
#: * ``Italy/95`` -> ``T_64_20190612_092742_palplebral.png``  ("lpe" for "lpe")
#:
#: A strict suffix check silently dropped the default-ROI mask for both of these
#: patients, which would have trained them on a whole frame while the metadata
#: claimed a palpebral ROI. Silently is the operative word: nothing errored.
#:
#: The explicit table covers the typos actually observed. The bounded fuzzy
#: fallback in :func:`mask_matches` handles anything else, but it compares whole
#: tails only, so it cannot see that ``..._forniceal_palplebral`` is a typo of the
#: *combined* ROI -- its tail is ``palplebral``, which is one edit from
#: ``palpebral`` and four from ``forniceal_palpebral``. It resolves to
#: ``palpebral``, which is a real ROI, so nothing errors and the combined mask is
#: silently mislabelled. The table entry below is what prevents that.
ROI_TYPOS: dict[str, str] = {
    "papebral": "palpebral",
    "palpebaral": "palpebral",
    "palpebal": "palpebral",
    "palplebral": "palpebral",
    "palpebralal": "palpebral",
    "fornicial": "forniceal",
    "forniceel": "forniceal",
    "fornical": "forniceal",
    "fornical_palpebral": "forniceal_palpebral",
    "forniceal_palpebaral": "forniceal_palpebral",
    "forniceal_palplebral": "forniceal_palpebral",
    "forniceal_palpebra": "forniceal_palpebral",
}


#: A trailing ``(n)``, left when a browser downloaded the same file twice.
_VARIANT_RE = re.compile(r"\(\d+\)$")


def _strip_variant_suffix(stem: str) -> str:
    """Drop a trailing ``(n)`` left by a browser re-download."""
    return _VARIANT_RE.sub("", stem.lower())


def _edit_distance(a: str, b: str, cap: int = 2) -> int:
    """Levenshtein distance, abandoned once it provably exceeds ``cap``.

    Only ever called on short strings (mask tails), and the cap keeps a
    pathological input from turning into an O(n*m) scan.
    """
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        best = i
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
            best = min(best, current[-1])
        if best > cap:
            return cap + 1
        previous = current
    return previous[-1]


def mask_matches(stem: str) -> str | None:
    """Return the ROI a mask filename stem belongs to, or ``None``.

    Four things this has to get right, every one of them forced by a filename in
    the real corpus rather than by theory.

    **Longest suffix first.** ``20200118_164733_forniceal_palpebral`` must resolve
    to the combined ROI, not be truncated to ``forniceal``. A plain
    ``endswith("forniceal")`` check gets this wrong on every combined mask.

    **Strip a trailing ``(n)`` first.** ``India/8`` holds
    ``..._palpebral(1).png`` next to ``..._palpebral.png``. On the raw stem
    ``endswith("palpebral")`` is false, so the duplicate was not recognised as a
    mask -- and because :func:`_pick_image` excludes masks using this same
    function, the duplicate could have been picked as the patient's photograph.

    **Known typos.** ``_papebral`` and ``_palplebral`` are both in the release.

    **A bounded edit-distance fallback** for anything else, so an unseen typo
    still resolves instead of being dropped. The bound is one edit, and ties are
    broken toward the longer ROI name so ``forniceal`` cannot absorb a combined
    mask.
    """
    lowered = _strip_variant_suffix(stem)

    for roi in MASK_SUFFIX_ORDER:
        if lowered.endswith(roi):
            return roi

    # A stem can name **two** ROIs, joined by an underscore:
    # ``..._forniceal_palpebral``. So before falling back to the text after the
    # last underscore, try correcting the whole tail against each ROI name --
    # longest first. Without this, ``..._forniceal_palplebral`` has tail
    # ``palplebral``, which is one edit from ``palpebral`` and four from
    # ``forniceal_palpebral``, so it would resolve to ``palpebral`` and the
    # combined mask would be silently mislabelled.
    corrected = lowered
    for roi in MASK_SUFFIX_ORDER:
        tail = corrected.rsplit("_", 1)[-1] if "_" in corrected else corrected
        if tail in ROI_TYPOS:
            corrected = corrected[: len(corrected) - len(tail)] + ROI_TYPOS[tail]
            break
        if len(tail) > 3 and _edit_distance(tail, roi, cap=1) <= 1:
            corrected = corrected[: len(corrected) - len(tail)] + roi
            break
    for roi in MASK_SUFFIX_ORDER:
        if corrected.endswith(roi):
            return roi

    tail = lowered.rsplit("_", 1)[-1] if "_" in lowered else lowered
    if tail in ROI_TYPOS:
        return ROI_TYPOS[tail]

    # Bounded fuzzy tail. Compared against the whole lowered stem as well, since
    # some folders have no underscore before the ROI name.
    # Lowest distance wins; on a tie the longer ROI name wins, so `forniceal`
    # cannot absorb a combined mask. Sorting by (distance, -len(name)) makes the
    # comparison explicit rather than relying on dict order.
    scored: list[tuple[int, int, str]] = []
    for roi in MASK_SUFFIX_ORDER:
        for candidate in (tail, lowered):
            if not candidate:
                continue
            distance = _edit_distance(candidate, roi, cap=1)
            if distance <= 1:
                scored.append((distance, -len(roi), roi))
    if scored:
        scored.sort()
        return scored[0][2]

    return None


def _dedup_key(path: Path) -> tuple[str, int]:
    """Identity of a mask for duplicate-collapse purposes.

    ``India/8`` contains ``..._palpebral.png`` and ``..._palpebral(1).png``, which
    are byte-identical (verified by SHA-256) copies left behind by a re-download.

    The key is the **resolved ROI** plus file size, not the stem: a stem-based key
    would fail to collapse the typo variant ``_papebral.png`` against
    ``_palpebral.png`` if a folder ever contained both spellings of one mask,
    since they are the same ROI at one edit distance apart.
    """
    return (mask_matches(path.stem) or path.stem.lower(), path.stat().st_size)


#: Rank of a mask filename: lower wins. Deliberately **not** dependent on
#: iteration order, because ``'('`` is 0x28 and ``'.'`` is 0x2E, so
#: ``sorted()`` puts ``..._palpebral(1).png`` *before* ``..._palpebral.png``. A
#: rule phrased as "prefer the name I saw second" silently loses; ranking by an
#: explicit property cannot.
def _mask_rank(path: Path) -> tuple[int, int, str]:
    """Sort key for choosing between two files claiming the same ROI.

    1. **Unsuffixed first.** ``_papebral.png`` is an original the uploader
       mistyped; ``_papebral(1).png`` is a re-download of it. That distinction is
       about provenance, not about sort order.
    2. **Shorter name first**, then lexicographic, so the choice is total and
       reproducible rather than depending on which file the OS listed first.
    """
    return (1 if _VARIANT_RE.search(path.name) else 0, len(path.name), path.name)


def discover_masks(folder: Path) -> dict[str, Path]:
    """Every ROI mask in a patient folder, one per ROI.

    Longest suffix wins, duplicate ``(n)`` copies collapse to one, and a
    documented typo is mapped to its ROI. When two genuinely different files claim
    one ROI, :func:`_mask_rank` picks deterministically.
    """
    by_roi: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        roi = mask_matches(path.stem)
        if roi is None:
            continue
        existing = by_roi.get(roi)
        if existing is None:
            by_roi[roi] = path
            continue
        if _dedup_key(path) == _dedup_key(existing) or _mask_rank(path) < _mask_rank(existing):
            by_roi[roi] = path
    return by_roi


def discover_patient_dirs(root: Path) -> list[Path]:
    """Every immediate subdirectory of ``root``, sorted for determinism.

    Directories with no image inside are skipped, because a stray ``__MACOSX``
    or a backup folder should not become a patient with a missing label.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"patient root does not exist: {root}")

    found: list[Path] = []
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(f.suffix.lower() in _IMAGE_SUFFIXES for f in entry.iterdir() if f.is_file()):
            found.append(entry)
    return found


def _pick_image(folder: Path, mask_paths: dict[str, Path]) -> Path | None:
    """Choose the patient's photograph, excluding every file that is a mask.

    Decided by asking :func:`mask_matches` rather than by comparing against a
    list of literal names, so it stays correct for ``001_palpebral.png`` and for
    the ``(1)`` duplicates alike.
    """
    mask_set = set(mask_paths.values())
    candidates = [
        f
        for f in sorted(folder.iterdir())
        if f.is_file()
        and f.suffix.lower() in _IMAGE_SUFFIXES
        and f not in mask_set
        and mask_matches(f.stem) is None
    ]
    return candidates[0] if candidates else None


def _normalise_sex(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("m"):
        return "M"
    if text.startswith("f"):
        return "F"
    return "unknown"


def _normalise_site(value: object) -> str:
    text = str(value).strip().lower()
    for site in SITES:
        if site.lower() in text:
            return site
    # Fall back on the two countries this corpus actually contains.
    if "india" in text:
        return "India"
    if "ital" in text:
        return "Italy"
    return "unknown"


def _read_workbook(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """One site's workbook, keyed by the folder number as written on disk.

    Returns the rows *and* the logical-name -> actual-column map, because the
    rows themselves keep the workbook's own headers. An earlier version returned
    rows only and :func:`build_records` then asked for ``row["hb"]``, which is
    absent when the column is called ``Hgb`` -- so every patient was reported as
    having no haemoglobin and the loader returned an empty list rather than
    raising. Carrying the map removes the need for the two sides to agree on a
    name by coincidence.

    The key is the *integer* number, not the folder name, because the two sides
    of the join disagree on padding: ``Italy.xlsx`` has ``Number = 1`` while one
    folder's mask prefix is ``001`` and another is ``109``. Comparing as integers
    makes padding irrelevant.
    """
    table = read_label_table(path)
    cols = resolve_label_columns(table.columns)
    if "patient_id" not in cols:
        raise ValueError(
            f"cannot identify the patient-id column in {path.name}. "
            f"Columns seen: {sorted(table.columns)}."
        )

    rows: dict[str, dict[str, object]] = {}
    for record in table.to_dict("records"):
        number = record.get(cols["patient_id"])
        try:
            key = str(int(float(number)))
        except (TypeError, ValueError):
            continue
        rows[key] = record
    return rows, cols


def build_records(
    root: Path,
    *,
    require_masks: bool = False,
    verbose: bool = True,
) -> list[PatientRecord]:
    """Join the released two-level tree to the per-site workbooks.

    Parameters
    ----------
    root
        The ``dataset anemia`` directory: one subdirectory per site, one
        subdirectory per patient inside that, and ``<Site>.xlsx`` inside the
        site directory.
    require_masks
        When true, a patient with no :data:`DEFAULT_ROI` mask is dropped and
        reported rather than trained on with a whole-frame ROI. Six Italian
        patients have no forniceal mask, but the default ROI is palpebral, so
        this costs nothing on the default path.
    verbose
        Print the drop report. Dropped patients are reported rather than merely
        skipped, because a count that changes silently between runs is how 15
        Italian haemoglobin values get lost without anybody noticing.

    Returns
    -------
    list[PatientRecord]
        Site-qualified ids, ascending by site then patient number, so the order
        is reproducible without depending on directory iteration order.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    site_dirs = [
        d
        for d in sorted(p for p in root.iterdir() if p.is_dir())
        if _normalise_site(d.name) in SITES
    ]
    if not site_dirs:
        raise FileNotFoundError(
            f"no site directory under {root}; expected one of {list(SITES)}. "
            f"Saw: {[d.name for d in sorted(root.iterdir()) if d.is_dir()]}"
        )

    records: list[PatientRecord] = []
    dropped: list[str] = []
    counts = {
        "no label row": 0,
        "no haemoglobin": 0,
        "excluded by authors": 0,
        "no image": 0,
        "no mask": 0,
    }

    for site_dir in site_dirs:
        site = _normalise_site(site_dir.name)
        workbook = site_dir / f"{site}.xlsx"
        if not workbook.is_file():
            raise FileNotFoundError(f"no label workbook for {site}: expected {workbook}")

        rows, cols = _read_workbook(workbook)
        hb_col = cols["hb"]
        age_col = cols.get("age")
        sex_col = cols.get("sex")
        note_col = cols.get("note")

        for folder in discover_patient_dirs(site_dir):
            pid = f"{site}/{folder.name}"
            row = rows.get(str(int(folder.name)) if folder.name.isdigit() else folder.name)

            if row is None:
                dropped.append(f"{pid}: no matching row in {workbook.name}")
                counts["no label row"] += 1
                continue

            exclusion = is_excluded(row)
            if exclusion is not None:
                dropped.append(f"{pid}: excluded ({exclusion})")
                counts["excluded by authors"] += 1
                continue

            raw_hb = row.get(hb_col)
            hb = parse_hb(raw_hb)
            if hb is None:
                dropped.append(f"{pid}: no haemoglobin value ({raw_hb!r})")
                counts["no haemoglobin"] += 1
                continue

            if not (HB_VALID_MIN <= hb <= HB_VALID_MAX):
                dropped.append(f"{pid}: haemoglobin {hb} outside [{HB_VALID_MIN}, {HB_VALID_MAX}]")
                counts["no haemoglobin"] += 1
                continue

            masks = discover_masks(folder)
            image = _pick_image(folder, masks)
            if image is None:
                dropped.append(f"{pid}: no photograph among {len(list(folder.iterdir()))} files")
                counts["no image"] += 1
                continue

            if require_masks and DEFAULT_ROI not in masks:
                dropped.append(f"{pid}: no {DEFAULT_ROI} mask")
                counts["no mask"] += 1
                continue

            raw_age = row.get(age_col) if age_col else None
            try:
                age = float(raw_age)  # type: ignore[arg-type]
                if not np.isfinite(age):
                    age = None
            except (TypeError, ValueError):
                age = None

            note = row.get(note_col) if note_col else None
            records.append(
                PatientRecord(
                    patient_id=pid,
                    image=image,
                    masks=masks,
                    hb=hb,
                    sex=_normalise_sex(row.get(sex_col, "") if sex_col else ""),
                    site=site,
                    age=age,
                    note="" if note is None or pd.isna(note) else str(note).strip(),
                )
            )

    records.sort(key=lambda r: (r.site, int(r.local_id) if r.local_id.isdigit() else 0))

    if verbose:
        print(f"[dataset] {root}")
        print(f"[dataset] {len(records)} patients kept")
        for site in sorted({r.site for r in records}):
            hb = np.asarray([r.hb for r in records if r.site == site])
            print(
                f"[dataset]   {site:<6} n={len(hb):<4} Hb mean={hb.mean():.2f} "
                f"sd={hb.std(ddof=1):.2f} range=[{hb.min():.1f}, {hb.max():.1f}]"
            )
        if dropped:
            print(f"[dataset] {len(dropped)} patients dropped: {counts}")
            for line in dropped:
                print(f"[dataset]   {line}")

    return records


# --------------------------------------------------------------------------- #
# Image and mask IO
# --------------------------------------------------------------------------- #


def load_rgb(path: Path) -> NDArray[np.uint8]:
    """Read an image as RGB uint8.

    OpenCV returns BGR, so the channels are reversed explicitly. Getting this
    backwards swaps the red and blue channels and produces a model that looks
    trained while reading the wrong reflectance.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_mask(path: Path, *, threshold: int = 127) -> NDArray[np.bool_] | None:
    """Read a binary mask as a boolean array, or ``None`` if the file is absent.

    Accepts grayscale or colour PNGs, and either 0/255 or 0/1 valued data, since
    the released masks were drawn by hand in more than one tool.
    """
    if not path.is_file():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw[..., 0] if raw.shape[2] == 4 else cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if raw.dtype == np.bool_:
        return raw
    if raw.dtype != np.uint8:
        raw = raw.astype(np.float32)
        return raw > 0.5
    return raw > threshold


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def apply_mask(rgb: NDArray[np.uint8], mask: NDArray[np.bool_] | None) -> NDArray[np.uint8]:
    """Zero out everything outside ``mask``.

    With no mask the frame is returned untouched. That is not a neutral default,
    it is the unsegmented condition that C2 measures against, so it is recorded
    rather than treated as equivalent.
    """
    if mask is None:
        return rgb
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    out = rgb.copy()
    out[~mask] = 0
    return out


def build_preprocess(
    *,
    train: bool,
    size: int = IMAGE_SIZE,
) -> Callable[[NDArray[np.uint8]], FloatArr]:
    """Return the exact preprocessing function, shared by training and export.

    Augmentation is deliberately geometric-only. Colour jitter would change the
    very signal being measured -- haemoglobin absorbance and melanin
    concentration are both *colour* -- so a colour-jittered model learns
    invariance to the target itself. Mild rotation and translation do not touch
    the spectrum.

    The output is CHW float32, matching :data:`PREPROCESS_SPEC`, which is what
    the browser transformer is written against.

    Note there is deliberately no ``roi`` argument. ROI selection happens in
    :class:`ConjunctivaDataset` via masking and cropping, before this function
    runs, so the tensor maths is identical for every ROI. An earlier version took
    ``roi`` here and ignored it, which made the signature imply a dependency
    that did not exist. The ROI a model was trained on is still recorded, in
    :data:`PREPROCESS_SPEC`, because the browser side needs to reproduce the
    masking step too.
    """
    mean = np.asarray(PREPROCESS_SPEC["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(PREPROCESS_SPEC["std"], dtype=np.float32).reshape(3, 1, 1)

    def _preprocess(rgb: NDArray[np.uint8]) -> FloatArr:
        if train:
            rgb = _augment(rgb)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        chw = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
        chw *= 1.0 / 255.0
        chw -= mean
        chw /= std
        return chw

    return _preprocess


def _augment(rgb: NDArray[np.uint8], rng: np.random.Generator | None = None) -> NDArray[np.uint8]:
    """Rotation and translation only, and only mildly.

    See :func:`build_preprocess` for why nothing here touches colour.
    """
    rng = rng or np.random.default_rng()
    angle = float(rng.uniform(-12.0, 12.0))
    shift_x, shift_y = (int(rng.integers(-16, 17)), int(rng.integers(-16, 17)))

    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    return cv2.warpAffine(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def crop_to_mask_bbox(
    rgb: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]] | None:
    """Crop to the mask's bounding box, keeping the conjunctiva filling the frame.

    The ROI is small in the original frame, so resizing the whole frame to
    224x224 leaves the conjunctiva at a few dozen pixels. Cropping first is what
    makes the published numbers reachable. Returns ``None`` for an empty mask.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # A little context around the box; a mask that exactly fills the crop tends
    # to be a boundary artefact rather than an anatomical ROI.
    pad = max(2, int(0.10 * max(y1 - y0, x1 - x0)))
    y0, y1 = max(0, y0 - pad), min(rgb.shape[0], y1 + pad)
    x0, x1 = max(0, x0 - pad), min(rgb.shape[1], x1 + pad)

    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #


class ConjunctivaDataset:
    """Maps :class:`PatientRecord` objects to cropped, normalised tensors.

    Implemented against ``torch.utils.data.Dataset``'s protocol without
    subclassing it, so that importing this module for its helpers (as
    ``scripts/validate_dataset.py`` does) does not pull in torch.

    Each item is ``(tensor, hb, patient_id)``. The tensor is CHW float32 and
    matches :data:`PREPROCESS_SPEC` exactly, which is what keeps the browser
    transformer and the Python one comparable.

    ``use_mask=False`` measures the whole frame instead of the conjunctival ROI.
    That is the unsegmented condition of experiment C2 and is off by default,
    because the ROI is the reason the published R² is 0.503 and not 0.306.
    """

    def __init__(
        self,
        records: Sequence[PatientRecord],
        *,
        roi: str = DEFAULT_ROI,
        train: bool = False,
        size: int = IMAGE_SIZE,
        use_mask: bool = True,
        crop: bool = True,
    ) -> None:
        if roi not in MASK_SUFFIXES:
            raise ValueError(f"unknown roi {roi!r}; expected one of {sorted(MASK_SUFFIXES)}")
        self.records = list(records)
        self.roi = roi
        self.use_mask = use_mask
        self.crop = crop
        self._train = train
        self._preprocess = build_preprocess(train=train, size=size)
        self._size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[NDArray[np.float32], float, str]:
        record = self.records[index]
        rgb = load_rgb(record.image)

        if self.use_mask:
            mask = load_mask(record.masks.get(self.roi, Path()))
            if mask is not None:
                rgb = apply_mask(rgb, mask)
                if self.crop:
                    cropped = crop_to_mask_bbox(rgb, mask)
                    if cropped is not None:
                        rgb = cropped[0]

        tensor = self._preprocess(rgb)
        return tensor, float(record.hb), record.patient_id

    # Convenience for the training loop, which needs the aligned label vectors
    # rather than the tuples.
    @property
    def hb(self) -> NDArray[np.float64]:
        return np.asarray([r.hb for r in self.records], dtype=np.float64)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(r.patient_id for r in self.records)

    @property
    def site_of(self) -> dict[str, str]:
        return {r.patient_id: r.site for r in self.records}

    @property
    def sex_of(self) -> dict[str, str]:
        return {r.patient_id: r.sex for r in self.records}

    def subset(self, patient_ids: Iterable[str]) -> ConjunctivaDataset:
        """A new dataset over a subset, preserving this one's settings."""
        wanted = set(patient_ids)
        kept = [r for r in self.records if r.patient_id in wanted]
        missing = wanted - {r.patient_id for r in kept}
        if missing:
            raise KeyError(
                f"{len(missing)} patient ids are not in this dataset, e.g. {sorted(missing)[0]}"
            )
        return ConjunctivaDataset(
            kept,
            roi=self.roi,
            train=self._train,
            size=self._size,
            use_mask=self.use_mask,
            crop=self.crop,
        )
