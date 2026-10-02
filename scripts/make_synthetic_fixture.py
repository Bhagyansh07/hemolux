"""Generate a synthetic stand-in for the Eyes Defy Anemia corpus.

Why this exists
---------------
The real corpus cannot be redistributed and accepting its terms is a manual step,
so CI cannot download it. Without a stand-in, the loader, the split logic, the
training loop and the ONNX export would stay untested until a human finished
signing up.

What it reproduces, and why each detail is deliberate
----------------------------------------------------
An earlier version of this script wrote ``data/synthetic/patients/P0000/`` with a
single combined spreadsheet. The loader rejected it: it wants one directory per
site, a patient directory inside that, and a workbook *named after its site*. The
script printed a cheerful summary for a corpus nothing could read, which is the
worst kind of fixture -- one that looks like coverage and provides none.

So the layout is now the real one, and the awkward parts of the real release are
reproduced on purpose, because each one is a branch the loader has to survive:

- **Frames are portrait** (``W x H``), as the released frames are.
- **Masks are stored at a smaller resolution than their frame.** In the release
  a frame is 2988x3984 and its mask is 800x1067, roughly a quarter of the area.
  A fixture whose masks already matched would leave the resize path untested.
- **The two sites name their masks differently.** India prefixes every mask with
  a capture timestamp (``20200118_164733_palpebral.png``); Italy uses the folder
  number zero-padded to three digits (``001_palpebral.png``). Reproducing both
  means CI exercises both arms of the mask matcher rather than only the easy one.
- **One patient is withdrawn.** The release marks it ``ELIMINATO``, in a column
  with no header, and the loader has to notice and report it.
- **Some patients have no forniceal mask.** Six Italians do not in the release,
  so the fixture withholds a couple as well, and the default ROI (palpebral) is
  unaffected.

What it is not
--------------
This is **not** a substitute for the real data and no number produced from it may
appear in ``EVALS.md``. Every artefact it writes is prefixed ``SYNTHETIC_`` and
lands under ``data/synthetic/``, which is excluded from experiment code paths.

The generator is physically motivated rather than random, so a smoke test can
assert the model actually learns something. Conjunctival redness is set inversely
by Hb and attenuated by melanin, which mirrors the confound this project is about:

    redness  = base * f(hb) * (1 - k * melanin)

so a model trained on the fixture has to resolve two overlapping absorbers, which
is the actual difficulty of the real task in miniature.

Usage
-----
    python scripts/make_synthetic_fixture.py --patients 60
    python scripts/make_synthetic_fixture.py --patients 60 --force
    python scripts/validate_dataset.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemolux.config import PATHS

PREFIX = "SYNTHETIC_"

#: Frame size as ``(width, height)``. Portrait, and not 224, so the crop-then-resize
#: path is exercised rather than short-circuited by a frame that is already the
#: network's input size.
FRAME_W, FRAME_H = 288, 384

#: Integer divisor between a frame and its masks. The release is nearer 3.73; an
#: integer keeps the fixture honest about there being *two* resolutions without
#: making every expectation in a test a float comparison.
MASK_DOWNSCALE = 3

#: Sites, in the order the release lists them.
SITES = ("India", "Italy")

#: Fraction of the corpus that is Indian. Mirrors the real 95/218 so that a
#: per-site metric on the fixture is not accidentally balanced by accident.
INDIA_FRACTION = 0.436

#: How many Indian patients are generated without a forniceal mask. The real
#: corpus withholds six Italian ones; the default ROI is palpebral, so this costs
#: nothing on the default path and exercises the reporting path.
WITHHELD_FORNIX = 2

_BANNER = """
============================================================
  SYNTHETIC FIXTURE -- generated data, not patient data.
  Nothing produced from this corpus is a reportable result.
============================================================
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
    width, height = FRAME_W, FRAME_H

    # --- background: surrounding skin, darkened by melanin -----------------
    skin_l = 78.0 - 46.0 * melanin
    skin = np.array([skin_l + 14.0, skin_l - 4.0, skin_l - 22.0], dtype=np.float32)
    img = np.clip(rng.normal(skin, 4.0, (height, width, 3)), 0, 255)

    # --- the conjunctival ROI ----------------------------------------------
    # Redness falls as Hb falls. 4 g/dL is nearly colourless, 16 g/dL is deep.
    redness = float(np.clip(0.25 + 0.048 * hb, 0.0, 1.0))
    base = np.array([206.0, 96.0, 92.0], dtype=np.float32)
    pale = np.array([224.0, 168.0, 160.0], dtype=np.float32)
    colour = pale + (base - pale) * redness

    # Melanin absorbs the short-wavelength end hardest.
    absorb = 1.0 - 0.55 * melanin
    colour[1] *= absorb
    colour[2] *= absorb**1.35

    centre = (
        int(height * (0.42 + 0.16 * rng.random())),
        int(width * (0.44 + 0.12 * rng.random())),
    )
    axes = (
        int(height * (0.20 + 0.07 * rng.random())),
        int(width * (0.11 + 0.05 * rng.random())),
    )
    angle = float(rng.uniform(0.0, 180.0))

    mask = np.zeros((height, width), dtype=np.uint8)
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
        cv2.line(mask, p0, p1, 0, int(rng.integers(2, 5)), cv2.LINE_AA)

    region = rng.normal(colour, 5.0, (height, width, 3))
    img[mask_bool] = np.clip(region[mask_bool], 0, 255)

    return np.clip(img, 0, 255).astype(np.uint8), mask_bool


def _roi_masks(mask_bool: np.ndarray) -> dict[str, np.ndarray]:
    """Split one blob into palpebral, forniceal and combined masks.

    Palpebral and forniceal are different regions of the same exposure, so a model
    that reports one of them well and the other badly is saying something about
    anatomy rather than about colour. Modelling them as nested bands preserves
    that distinction.
    """
    ys, xs = np.nonzero(mask_bool)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    # Palpebral is the lower portion of the exposed conjunctiva.
    split = y0 + int(0.45 * (y1 - y0))
    palpebral = mask_bool.copy()
    palpebral[:split, :] = False

    # Forniceal is a shallower band at the top.
    fornix_top = y0 + int(0.22 * (y1 - y0))
    fornix_bottom = split
    forniceal = np.zeros_like(mask_bool)
    forniceal[fornix_top:fornix_bottom, x0 : x1 + 1] = mask_bool[
        fornix_top:fornix_bottom, x0 : x1 + 1
    ]

    return {
        "palpebral": palpebral,
        "forniceal": forniceal,
        "forniceal_palpebral": mask_bool,
    }


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a mask at the fixture's *smaller* resolution, as the release does.

    Nearest-neighbour, because a mask is a region label and interpolating one
    invents boundary pixels that no one segmented.
    """
    height, width = mask.shape[:2]
    small = cv2.resize(
        mask.astype(np.uint8) * 255,
        (width // MASK_DOWNSCALE, height // MASK_DOWNSCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(path), small)


def _stem_for(site: str, number: int, rng: np.random.Generator) -> str:
    """The filename stem each site uses in the real release.

    India stamps a capture time; Italy zero-pads the folder number. Reproducing
    both is the point, so CI meets the same two shapes the loader meets in
    production.
    """
    if site == "India":
        return (
            f"20{rng.integers(20, 24):02d}{rng.integers(1, 13):02d}{rng.integers(1, 29):02d}_"
            f"{rng.integers(0, 24):02d}{rng.integers(0, 60):02d}{rng.integers(0, 60):02d}"
        )
    return f"{number:03d}"


def _clean(root: Path, sites: Iterable[str]) -> None:
    for site in sites:
        target = root / site
        if target.exists():
            shutil.rmtree(target)
    for stale in root.glob(f"{PREFIX}*"):
        stale.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    root.mkdir(parents=True, exist_ok=True)
    if any((root / site).exists() for site in SITES):
        if not args.force:
            print(f"[fixture] {root} already holds a fixture. Use --force to regenerate.")
            return 1
        _clean(root, SITES)

    rng = np.random.default_rng(args.seed)

    n_india = round(args.patients * INDIA_FRACTION)
    # Patient folders are numbered from 1 within each site, and both sites number
    # from 1, so the fixture reproduces the folder-number collision that
    # site-qualified identifiers exist to solve.
    plan = [("India", i) for i in range(1, n_india + 1)]
    plan += [("Italy", i) for i in range(1, args.patients - n_india + 1)]

    rows: dict[str, list[dict[str, object]]] = {site: [] for site in SITES}
    truth: dict[str, dict[str, float]] = {}
    withheld = 0

    for site, number in plan:
        # Hb: broad adult range with an anaemic tail, per site. Italian subjects
        # skew higher in the real corpus, which a per-site metric must notice.
        hb = float(rng.normal(13.4 if site == "Italy" else 11.8, 2.3))
        hb = float(np.clip(hb, 5.2, 17.2))
        melanin = float(rng.beta(2.0, 2.6))  # darker in the India cohort
        if site == "India":
            melanin = float(np.clip(melanin * 1.35, 0.0, 1.0))
        sex = "M" if rng.random() < 0.42 else "F"
        age = int(rng.integers(19, 76))

        image, blob = _synth_rgb(hb, melanin, rng)
        folder = root / site / str(number)
        folder.mkdir(parents=True, exist_ok=True)

        stem = _stem_for(site, number, rng)
        cv2.imwrite(str(folder / f"{stem}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        masks = _roi_masks(blob)
        # Withhold a forniceal mask from the first few Italian patients, mirroring
        # the six the release omits, and leaving the palpebral default intact.
        omit_fornix = site == "Italy" and number <= WITHHELD_FORNIX
        if omit_fornix:
            withheld += 1
            masks.pop("forniceal")
            masks.pop("forniceal_palpebral")
        for roi, roi_mask in masks.items():
            _write_mask(folder / f"{stem}_{roi}.png", roi_mask)

        rows[site].append(
            {
                "Number": number,
                "Hb (g/dL)": round(hb, 2),
                "Sex": sex,
                "Age": age,
                "Site": site,
                "Note": "",
            }
        )
        truth[f"{site}/{number}"] = {"hb": hb, "melanin": melanin}

    # One withdrawn patient, exactly as the release does it: a marker with no
    # header, which a lookup by column name would never find.
    withdrawn_site = "Italy"
    withdrawn = max(n for s, n in plan if s == withdrawn_site)
    rows[withdrawn_site].append(
        {
            "Number": withdrawn + 1,
            "Hb (g/dL)": 12.4,
            "Sex": "F",
            "Age": 34,
            "Site": withdrawn_site,
            "Note": "ELIMINATO",
        }
    )

    truth_path = root / f"{PREFIX}ground_truth.json"
    written: list[str] = []
    for site in SITES:
        site_dir = root / site
        site_dir.mkdir(parents=True, exist_ok=True)
        labels_path = site_dir / f"{site}.xlsx"
        pd.DataFrame(rows[site]).to_excel(labels_path, index=False)
        kept = sum(1 for r in rows[site] if r["Note"] == "")
        written.append(f"{site}/{labels_path.name} ({kept} patients)")

    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

    all_rows = pd.DataFrame([r for site in SITES for r in rows[site]])
    kept = all_rows[all_rows["Note"] != "ELIMINATO"]
    print(f"[fixture] root : {root}")
    print(f"[fixture]   India : {n_india}")
    print(f"[fixture]   Italy : {args.patients - n_india}")
    print(
        f"[fixture]   Hb    : mean {kept['Hb (g/dL)'].mean():.2f}, "
        f"range {kept['Hb (g/dL)'].min():.2f}-{kept['Hb (g/dL)'].max():.2f}"
    )
    print(f"[fixture] frame  : {FRAME_W}x{FRAME_H}, masks at 1/{MASK_DOWNSCALE} of that")
    print(f"[fixture] masks  : {withheld} patient(s) withheld a forniceal mask")
    print(f"[fixture] labels : {', '.join(written)}, 1 withdrawn as ELIMINATO")
    print(f"[fixture] truth  : {truth_path.name} (generator parameters, not a model)")
    print("\n[fixture] Exercise it with:")
    print("            python scripts/validate_dataset.py --synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
