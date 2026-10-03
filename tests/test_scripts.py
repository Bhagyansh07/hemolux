"""Tests for the synthetic fixture generator.

The generator exists because of a specific failure. ``scripts/make_synthetic_fixture.py``
wrote ``data/synthetic/patients/P0000/`` behind one combined spreadsheet.
``build_records`` wants one directory per site, a patient directory inside that, and
a workbook named after its site, so it rejected the fixture outright with a message
naming the two directory names it had expected. The generator had been printing a
cheerful summary for a corpus that nothing in the repository could read.

The awkward parts of the real release are reproduced on purpose, and each is asserted
here, because an unasserted reproduction is a comment:

* masks are stored at a smaller resolution than their frame, so the resize path is
  exercised rather than assumed;
* the two sites name their masks differently, so both arms of the mask matcher run;
* one patient is withdrawn with a marker in a column that has no header;
* some patients have no forniceal mask while the palpebral default is unaffected.

The round-trip test is the load-bearing one: it generates a fixture into a temporary
directory and then requires ``hemolux.validation`` to accept it, which is the only way
to catch a layout that only the generator believes in. That half lives in
``test_validation.py``, next to the validator it exercises.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hemolux.metrics.colorimetry import high_hue_ratio

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    """Import a script by path, since ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture_mod = _load("make_synthetic_fixture")


# --------------------------------------------------------------------------- #
# Fixture geometry
# --------------------------------------------------------------------------- #


def test_roi_masks_are_three_regions_and_two_of_them_disjoint() -> None:
    """Palpebral and forniceal must not overlap, or C2 compares a region with itself."""
    rng = np.random.default_rng(7)
    _, blob = fixture_mod._synth_rgb(13.0, 0.3, rng)
    masks = fixture_mod._roi_masks(blob)

    assert set(masks) == {"palpebral", "forniceal", "forniceal_palpebral"}
    palpebral, forniceal = masks["palpebral"], masks["forniceal"]
    assert (palpebral & forniceal).sum() == 0, "the two ROIs overlap"
    assert (palpebral | forniceal).sum() <= blob.sum()
    assert (palpebral | forniceal).sum() > 0, "the split produced nothing"


def test_roi_masks_are_all_non_empty_on_a_typical_patient() -> None:
    """An empty forniceal mask would make the ROI comparison vacuous."""
    rng = np.random.default_rng(11)
    _, blob = fixture_mod._synth_rgb(9.0, 0.2, rng)
    for roi, mask in fixture_mod._roi_masks(blob).items():
        assert mask.any(), f"{roi} mask is empty"


def test_masks_are_written_at_a_smaller_resolution_than_the_frame(tmp_path: Path) -> None:
    """The release stores an 800x1067 mask for a 2988x3984 frame, and the loader
    resizes. A fixture whose masks already matched would leave that path untested,
    so the downscale is asserted rather than assumed."""
    mask = np.zeros((fixture_mod.FRAME_H, fixture_mod.FRAME_W), dtype=bool)
    mask[100:300, 60:200] = True

    target = tmp_path / "roi.png"
    fixture_mod._write_mask(target, mask)

    written = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
    assert written is not None, "the mask was not written"
    expected = (
        fixture_mod.FRAME_W // fixture_mod.MASK_DOWNSCALE,
        fixture_mod.FRAME_H // fixture_mod.MASK_DOWNSCALE,
    )
    assert (written.shape[1], written.shape[0]) == expected
    assert fixture_mod.MASK_DOWNSCALE > 1, "masks match the frame, which is the thing avoided"


def test_written_mask_is_still_binary_after_downscaling(tmp_path: Path) -> None:
    """Nearest-neighbour is required: interpolating a region label invents pixels
    that nobody segmented, and the validator's area fraction is computed from them."""
    rng = np.random.default_rng(3)
    _, blob = fixture_mod._synth_rgb(12.0, 0.4, rng)
    target = tmp_path / "roi.png"
    fixture_mod._write_mask(target, fixture_mod._roi_masks(blob)["palpebral"])

    written = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    assert written is not None
    assert set(np.unique(written).tolist()) <= {0, 255}


# --------------------------------------------------------------------------- #
# Filename conventions
# --------------------------------------------------------------------------- #


def test_the_two_sites_name_their_masks_differently() -> None:
    """Both shapes occur in the release, so both arms of the matcher need running.

    Asserted rather than left to chance: a fixture that quietly standardised the
    naming would still load, and would have stopped covering the branch it was
    written to cover.
    """
    rng = np.random.default_rng(5)
    indian = fixture_mod._stem_for("India", 1, rng)
    italian = fixture_mod._stem_for("Italy", 7, rng)

    # India: 8-digit capture timestamp, e.g. 20200118_164733.
    assert indian[:8].isdigit(), indian
    assert indian[8] == "_", indian

    # Italy: the folder number zero-padded to three digits.
    assert italian == "007", italian


def test_the_indian_stem_looks_like_a_timestamp_not_a_number() -> None:
    """The two are easy to confuse if only one is asserted, so check the segments.

    A capture stamp starts with a plausible year and the padded folder number never
    contains an underscore, so the separator alone is enough to tell them apart.
    """
    rng = np.random.default_rng(0)
    for seed in (0, 1, 2):
        stem = fixture_mod._stem_for("India", seed, rng)
        head = stem.split("_", 1)[0]
        assert len(head) == 8 and head.isdigit(), stem
        assert 2020 <= int(head[:4]) <= 2023, stem


# --------------------------------------------------------------------------- #
# Physical premise
# --------------------------------------------------------------------------- #


def _roi_patch(hb: float, melanin: float, seed: int) -> np.ndarray:
    """The masked conjunctival pixels of one rendered patient, as uint8 RGB."""
    rng = np.random.default_rng(seed)
    image, blob = fixture_mod._synth_rgb(hb, melanin, rng)
    return image[blob]


def test_conjunctival_redness_rises_with_haemoglobin() -> None:
    """This is the whole premise in miniature: a low Hb must read as paler.

    Measured with the project's own ``high_hue_ratio`` rather than with a raw
    channel, because the sign of the raw red channel is the wrong question. Pale
    conjunctiva is a *lighter* pink, so its red channel is not necessarily lower;
    what rises with Hb is saturation, and saturation is what the published
    smartphone estimators key on. Asserting on the raw channel would have passed a
    fixture whose premise was inverted.

    If this inverts, a smoke test can still pass while training on a corpus where
    the target is anti-correlated with the signal, which hides a broken pipeline
    rather than revealing one.
    """
    poor = high_hue_ratio(_roi_patch(6.0, 0.2, seed=1))
    rich = high_hue_ratio(_roi_patch(16.0, 0.2, seed=1))
    assert rich > poor, f"expected higher hue ratio at 16 g/dL, got {rich:.3f} vs {poor:.3f}"


def test_melanin_lowers_the_blue_channel_at_fixed_haemoglobin() -> None:
    """The second absorber, and the reason the fairness audit exists.

    A fixture where melanin changed nothing would let a model look fair for the
    wrong reason.
    """
    fair = _roi_patch(12.0, 0.05, seed=2).mean(axis=0)
    dark = _roi_patch(12.0, 0.85, seed=2).mean(axis=0)
    assert dark[2] < fair[2], (
        f"expected lower blue at high melanin, got {dark[2]:.1f} vs {fair[2]:.1f}"
    )


# --------------------------------------------------------------------------- #
# Round trip: generate, then read back through the real loader
# --------------------------------------------------------------------------- #


def _run_fixture_main(patients: int) -> int:
    """Invoke the generator's ``main`` with a redirected output root."""
    argv = sys.argv
    sys.argv = ["make_synthetic_fixture.py", "--patients", str(patients), "--force"]
    try:
        return fixture_mod.main()
    finally:
        sys.argv = argv


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A small fixture generated into ``tmp_path``."""
    root = tmp_path / "synthetic"
    root.mkdir()
    monkeypatch.setattr(fixture_mod, "PATHS", SimpleNamespace(synthetic=root), raising=True)
    assert _run_fixture_main(8) == 0
    return root


def test_the_generator_produces_a_corpus_the_loader_can_read(corpus: Path) -> None:
    """The load-bearing property of the generator.

    The round trip is exercised end to end in ``test_validation.py``, where the
    validator runs over this output. What is asserted here is the narrower thing:
    the generator does not quietly drop a share of its own patients, because the
    survivors are still learnable and a smoke test would still pass.
    """
    from hemolux.data.dataset import build_records

    records = build_records(corpus, verbose=False)
    assert len(records) == 8, f"expected 8 patients, loader returned {len(records)}"
    assert {r.site for r in records} == {"India", "Italy"}


def test_the_withdrawn_patient_is_dropped_and_reported(corpus: Path) -> None:
    """``ELIMINATO`` sits in a column with no header, so a lookup by name misses it.

    Asserted by count: the row is written, so the only reason it can be absent from
    the records is that the loader recognised the marker.
    """
    import pandas as pd

    from hemolux.data.dataset import build_records

    italian = pd.read_excel(corpus / "Italy" / "Italy.xlsx")
    indian = pd.read_excel(corpus / "India" / "India.xlsx")
    assert (italian["Note"] == "ELIMINATO").sum() == 1, "fixture no longer writes a withdrawn row"

    # Compared against both workbooks, because build_records reads the whole tree.
    # Comparing against one site's table would pass or fail for the wrong reason.
    total_rows = len(italian) + len(indian)
    assert len(build_records(corpus, verbose=False)) == total_rows - 1


def test_both_site_mask_naming_conventions_survive_the_loader(corpus: Path) -> None:
    """The point of reproducing both conventions: each site must still match."""
    from hemolux.data.dataset import DEFAULT_ROI, build_records

    records = build_records(corpus, verbose=False)
    assert records, "no records"
    assert all(DEFAULT_ROI in r.masks for r in records), "a palpebral mask went unmatched"


def test_the_mask_resolution_differs_from_the_frame_in_the_generated_corpus(
    corpus: Path,
) -> None:
    """Guards the property the fixture exists to provide.

    Without this, a future edit that stores masks at frame resolution would still
    load and still train, while quietly removing the resize path from CI.
    """
    from hemolux.data.dataset import build_records

    record = build_records(corpus, verbose=False)[0]
    frame = cv2.imread(str(record.image), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(record.mask), cv2.IMREAD_GRAYSCALE)
    assert mask.shape[:2] != frame.shape[:2], "fixture masks now match the frame resolution"
    # Aspect ratio is preserved, so the mask still belongs to this frame.
    assert abs((mask.shape[1] / mask.shape[0]) - (frame.shape[1] / frame.shape[0])) < 0.02


# --------------------------------------------------------------------------- #
# Social card
# --------------------------------------------------------------------------- #


def test_social_card_is_drawn_at_the_size_the_meta_tags_declare() -> None:
    """``og:image:width/height`` say 1200x630, and nothing else couples them.

    The tag lives in ``app/web/index.html`` and the image in
    ``scripts/make_og_image.py``, so a size that drifts is only caught by a test
    that draws the card and measures it -- every platform crops or letterboxes a
    mismatch, and neither looks like broken markup.
    """
    mod = _load("make_og_image")
    card = mod.draw_card()
    assert card.size == (1200, 630)
    # A blank card of the right size would pass the line above.
    colours = card.getcolors(maxcolors=1_000_000)
    assert colours is not None and len(colours) > 2
