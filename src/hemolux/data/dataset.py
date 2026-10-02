"""Dataset loading for the Eyes Defy Anemia corpus.

On-disk layout this module expects
----------------------------------
One directory per patient, named after the patient, containing:

* one JPEG of the eye region (the exact filename varies per patient),
* ``forniceal.png`` -- hand-drawn mask of the fornix conjunctiva,
* ``palpebral.png`` -- hand-drawn mask of the palpebral conjunctiva,
* ``forniceal_palpebral.png`` -- mask of the two combined.

Haemoglobin, sex and site live in a spreadsheet (``labels.xlsx``) next to the
patient tree, one row per patient.

Why the masks matter to the result
----------------------------------
Published work on this corpus puts R² at 0.27 for the forniceal ROI and 0.09
for forniceal alone, so the ROI choice is not cosmetic. ``DEFAULT_ROI`` is
therefore ``palpebral``. The other ROIs are loadable because comparing them is
experiment C2, not because they are expected to win.

On not guessing the spreadsheet schema
--------------------------------------
The exact column names in the released workbook are not documented, so
:func:`resolve_label_columns` looks for a set of accepted aliases and **raises**
when it cannot find the haemoglobin column, rather than guessing. A loader that
silently picks the wrong column produces a training run that trains on noise and
reports a plausible R², which is the single most expensive failure mode in this
project. ``scripts/validate_dataset.py`` prints the real header so the mapping
can be corrected against the actual file in one pass.
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

#: Mask filename per ROI. Kept as a module constant so the training config, the
#: ONNX export and the browser-side preprocessing all read the same names.
MASK_FILENAMES: dict[str, str] = {
    "forniceal": "forniceal.png",
    "palpebral": "palpebral.png",
    "forniceal_palpebral": "forniceal_palpebral.png",
}

#: The ROI this project trains on. Palpebral, because that is where the
#: published signal is roughly three times stronger than forniceal.
DEFAULT_ROI = "palpebral"

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
_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "patient_id": ("patientid", "patient", "id", "subject", "subjectid", "code", "casename"),
    "hb": ("hb", "hbgdl", "hemoglobin", "haemoglobin", "hgb", "hbvalue", "hbconc"),
    "sex": ("sex", "gender", "s"),
    "site": ("site", "country", "location", "centre", "center", "institution"),
}

_HB_ALIAS_KEYS = _LABEL_ALIASES["hb"]


def _normalise_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


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
    """One patient's image, its ROI masks and its ground-truth Hb."""

    patient_id: str
    image: Path
    masks: dict[str, Path]
    hb: float
    sex: str
    site: str

    @property
    def mask(self) -> Path | None:
        """The mask for :data:`DEFAULT_ROI`, if that patient has one."""
        return self.masks.get(DEFAULT_ROI)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def _pick_image(folder: Path) -> Path | None:
    """Choose the patient's JPEG, ignoring the mask PNGs."""
    candidates = [
        f
        for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES and f.name.lower() not in MASK_FILENAMES.values()
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


def build_records(
    patient_root: Path,
    label_table: Path,
    *,
    require_masks: bool = False,
) -> list[PatientRecord]:
    """Join the patient tree to the label table.

    Parameters
    ----------
    require_masks
        When true, a patient missing the :data:`DEFAULT_ROI` mask is dropped and
        reported, rather than being trained on with a whole-frame ROI. Dropping
        is the default only when ``require_masks`` is set, because an early run
        wants to know how many patients lack masks rather than silently having
        them counted as "no segmentation".
    """
    table = read_label_table(label_table)
    cols = resolve_label_columns(table.columns)
    table = table.rename(columns={v: k for k, v in cols.items()})
    if "patient_id" not in cols:
        raise ValueError(
            "cannot identify the patient-id column. "
            f"Columns seen: {sorted(table.columns)}. Rename one of the "
            f"patient directories to match a value in the workbook."
        )

    by_id: dict[str, dict[str, object]] = {}
    for row in table.to_dict("records"):
        key = _normalise_key(row["patient_id"])
        by_id[key] = row

    records: list[PatientRecord] = []
    skipped: list[str] = []

    for folder in discover_patient_dirs(patient_root):
        image = _pick_image(folder)
        if image is None:
            continue

        row = by_id.get(_normalise_key(folder.name))
        if row is None:
            skipped.append(f"{folder.name}: no matching row in the label table")
            continue

        raw_hb = row.get("hb")
        try:
            hb = float(raw_hb)
        except (TypeError, ValueError):
            skipped.append(f"{folder.name}: haemoglobin is not a number ({raw_hb!r})")
            continue

        if not np.isfinite(hb) or not (HB_VALID_MIN <= hb <= HB_VALID_MAX):
            skipped.append(f"{folder.name}: haemoglobin {hb} outside [{HB_VALID_MIN}, {HB_VALID_MAX}]")
            continue

        masks = {
            roi: folder / filename
            for roi, filename in MASK_FILENAMES.items()
            if (folder / filename).is_file()
        }
        if DEFAULT_ROI not in masks:
            if require_masks:
                skipped.append(f"{folder.name}: no {DEFAULT_ROI} mask")
                continue
            masks[DEFAULT_ROI] = folder / MASK_FILENAMES[DEFAULT_ROI]  # absent, guarded on load

        records.append(
            PatientRecord(
                patient_id=folder.name,
                image=image,
                masks=masks,
                hb=hb,
                sex=_normalise_sex(row.get("sex", "")),
                site=_normalise_site(row.get("site", "")),
            )
        )

    if skipped:
        detail = "\n  ".join(skipped[:10])
        more = f"\n  ... and {len(skipped) - 10} more" if len(skipped) > 10 else ""
        print(f"[dataset] skipped {len(skipped)} patients:\n  {detail}{more}")

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
        if roi not in MASK_FILENAMES:
            raise ValueError(f"unknown roi {roi!r}; expected one of {sorted(MASK_FILENAMES)}")
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
            raise KeyError(f"{len(missing)} patient ids are not in this dataset, e.g. {sorted(missing)[0]}")
        return ConjunctivaDataset(
            kept,
            roi=self.roi,
            train=self._train,
            size=self._size,
            use_mask=self.use_mask,
            crop=self.crop,
        )
