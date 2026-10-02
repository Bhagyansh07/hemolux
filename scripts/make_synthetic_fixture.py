"""Generate a synthetic stand-in for the Eyes Defy Anemia corpus.

Why this exists
---------------
The real corpus cannot be redistributed and its licence acceptance is a manual
step, so CI cannot download it and the training code would otherwise be untested
until a human finished signing up. This script writes a small corpus with the
**same on-disk layout** -- one folder per patient, one JPEG, the three mask PNGs,
and an Hb spreadsheet -- so the loader, the split logic, the training loop and
the ONNX export can all be exercised end to end without it.

What it is not
--------------
This is **not** a substitute for the real data and no number produced from it may
appear in ``EVALS.md``. Every artefact it writes is prefixed ``SYNTHETIC_`` and
lands under ``data/synthetic/``, which is excluded from experiment code paths.

The generator is physically motivated rather than random, so that a smoke test
can assert the model actually learns something. Conjunctival redness is set
inversely by Hb and attenuated by melanin, which mirrors the confound this
project is about:

    redness  = base * f(hb) * (1 - k * melanin)

so a model trained on the fixture has to resolve two overlapping absorbers,
which is the actual difficulty of the real task in miniature.

Usage
-----
    python scripts/make_synthetic_fixture.py --patients 60
    python scripts/make_synthetic_fixture.py --patients 60 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemolux.config import PATHS

PREFIX = "SYNTHETIC_"
IMAGE_SIZE_PX = 512  # deliberately not 224, so the crop-then-resize path is real

_BANNER = """
================================================================
  SYNTHETIC FIXTURE -- generated data, not patient data.
  Nothing produced from this corpus is a reportable result.
================================================================
"""


def _synth_rgb(
    hb: float, melanin: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Render one fake eye ROI. Returns ``(rgb_image, ellipse_mask)``.

    Parameters
    ----------
    hb
        Haemoglobin in g/dL. Drives how red the conjunctiva is.
    melanin
        Conjunctival melanin fraction in ``[0, 1]``. Absorbs short wavelengths,
        so it lowers blue and very slightly lowers green.
    """
    size = IMAGE_SIZE_PX

    # --- background: surrounding skin, darkened by melanin -----------------
    skin_l = 78.0 - 46.0 * melanin
    skin = np.array([skin_l + 14.0, skin_l - 4.0, skin_l - 22.0], dtype=np.float32)
    img = np.clip(rng.normal(skin, 4.0, (size, size, 3)), 0, 255)

    # --- the conjunctival ROI ----------------------------------------------
    # Redness falls as Hb falls. 4 g/dL is nearly colourless, 16 g/dL is deep.
    redness = np.clip(0.25 + 0.048 * hb, 0.0, 1.0)  # 0.25 at 4, 0.94 at ~15
    base = np.array([206.0, 96.0, 92.0], dtype=np.float32)
    pale = np.array([224.0, 168.0, 160.0], dtype=np.float32)
    colour = pale + (base - pale) * redness

    # Melanin absorbs the short-wavelength end hardest.
    absorb = 1.0 - 0.55 * melanin
    colour[1] *= absorb
    colour[2] *= absorb**1.35

    centre = (int(size * (0.42 + 0.16 * rng.random())), int(size * (0.44 + 0.12 * rng.random())))
    axes = (int(size * (0.20 + 0.07 * rng.random())), int(size * (0.11 + 0.05 * rng.random())))
    angle = float(rng.uniform(0.0, 180.0))

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, centre, axes, angle, 0, 360, 255, -1)
    mask_bool = mask > 127

    # Vessel texture, which is what the hue-based baselines actually key on.
    for _ in range(int(rng.integers(3, 8))):
        p0 = (
            centre[0] + int(rng.integers(-axes[0], axes[0])),
            centre[1] + int(rng.integers(-axes[1], axes[1])),
        )
        p1 = (
            centre[0] + int(rng.integers(-axes[0], axes[0])),
            centre[1] + int(rng.integers(-axes[1], axes[1])),
        )
        vessel = colour.copy()
        vessel[0] = min(255.0, vessel[0] * 1.35)
        vessel[1] *= 0.72
        vessel[2] *= 0.74
        cv2.line(mask, p0, p1, 0, int(rng.integers(2, 5)), cv2.LINE_AA)

    region = rng.normal(colour, 5.0, (size, size, 3))
    img[mask_bool] = np.clip(region[mask_bool], 0, 255)

    # Palpebral sits in the lower part of the fornix; model them as nested
    # ellipses so the two ROIs are genuinely different populations.
    return np.clip(img, 0, 255).astype(np.uint8), mask_bool


def _roi_masks(mask_bool: np.ndarray) -> dict[str, np.ndarray]:
    """Split one blob into palpebral, forniceal and combined masks."""
    ys, xs = np.nonzero(mask_bool)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

    # Palpebral is the lower portion of the exposed conjunctiva.
    split = y0 + int(0.45 * (y1 - y0))
    palpebral = mask_bool.copy()
    palpebral[:split, :] = False

    # Forniceal is a shallower band at the top.
    fornix_top = y0 + int(0.22 * (y1 - y0))
    fornix_bottom = y0 + int(0.45 * (y1 - y0))
    forniceal = np.zeros_like(mask_bool)
    forniceal[fornix_top:fornix_bottom, x0 : x1 + 1] = mask_bool[
        fornix_top:fornix_bottom, x0 : x1 + 1
    ]

    return {
        "palpebral": palpebral,
        "forniceal": forniceal,
        "forniceal_palpebral": mask_bool,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing fixture instead of exiting",
    )
    args = parser.parse_args()

    print(_BANNER)

    root = PATHS.synthetic
    patients_dir = root / "patients"
    if patients_dir.exists() and any(patients_dir.iterdir()):
        if not args.force:
            print(f"[fixture] {patients_dir} already exists. Use --force to regenerate.")
            return 1
        import shutil

        shutil.rmtree(patients_dir)

    patients_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # The real corpus is roughly 95 India / 123 Italy. Mirror the imbalance so
    # a per-site metric on the fixture is not accidentally balanced by accident.
    n_india = round(args.patients * 0.436)

    rows: list[dict[str, object]] = []
    truth: dict[str, dict[str, float]] = {}

    for i in range(args.patients):
        site = "India" if i < n_india else "Italy"
        patient_id = f"{PREFIX}P{i:04d}"

        # Hb: broad adult range with a anaemic tail, per site. Italian subjects
        # skew higher in the real corpus, which the site metric must notice.
        hb = float(rng.normal(13.4 if site == "Italy" else 11.8, 2.3))
        hb = float(np.clip(hb, 5.2, 17.2))
        melanin = float(rng.beta(2.0, 2.6))  # darker in the India cohort
        if site == "India":
            melanin = float(np.clip(melanin * 1.35, 0.0, 1.0))
        sex = "M" if rng.random() < 0.42 else "F"

        image, blob = _synth_rgb(hb, melanin, rng)
        folder = patients_dir / patient_id
        folder.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(folder / f"{patient_id}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        for roi, roi_mask in _roi_masks(blob).items():
            cv2.imwrite(str(folder / f"{roi}.png"), (roi_mask.astype(np.uint8)) * 255)

        rows.append(
            {
                "Patient ID": patient_id,
                "Hb (g/dL)": round(hb, 2),
                "Sex": sex,
                "Site": site,
            }
        )
        truth[patient_id] = {"hb": hb, "melanin": melanin}

    table = pd.DataFrame(rows)
    labels_path = root / f"{PREFIX}labels.xlsx"
    table.to_excel(labels_path, index=False)

    truth_path = root / f"{PREFIX}ground_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

    print(f"[fixture] wrote {len(rows)} patients to {patients_dir}")
    print(f"[fixture]   India : {n_india}")
    print(f"[fixture]   Italy : {args.patients - n_india}")
    print(
        f"[fixture]   Hb    : mean {table['Hb (g/dL)'].mean():.2f}, "
        f"range {table['Hb (g/dL)'].min():.2f}-{table['Hb (g/dL)'].max():.2f}"
    )
    print(f"[fixture] labels : {labels_path.name}")
    print(f"[fixture] truth  : {truth_path.name} (generator parameters, not a model)")
    print("\n[fixture] Exercise it with:")
    print("            python scripts/validate_dataset.py --synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
