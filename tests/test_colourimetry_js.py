"""Browser/Python parity for the ported colourimetry.

The shipped inference path computes the 13 ``COLOUR_COLUMNS`` in JavaScript, but the
model was trained against :mod:`hemolux.metrics.colorimetry`. A silent divergence between
the two does not throw or go out of range; it just feeds the model numbers with a
different meaning than the ones it learned, which is the failure mode this module exists
to catch.

The test builds several small deterministic frames with numpy, computes the expected
13-vectors with the **real** Python functions (``white_balance`` +
``extract_colour_features``, plus the ``roi_mean`` rule from ``extract_colour_rows``),
runs the Node module once over all of them, and asserts the per-column maximum absolute
difference.

Tolerances
----------
Two tiers, because the columns are not all the same kind of quantity.

* Exact-float columns -- ``lab_*``, ``redness_ratio``, ``erythema_index``,
  ``high_hue_ratio``, ``roi_*`` -- are asserted to ``1e-9``. They are short floating-point
  pipelines (a 3x3 matrix product, an ``hypot``, a percentile) with no lookup tables, and
  the observed error is around ``1e-13``; ``1e-9`` is tight enough that a changed matrix
  constant or a swapped percentile would fail well outside it, while leaving room for the
  last-ulp differences in the order numpy sums a frame.
* OpenCV-integer columns -- ``hsv_*`` and ``otsu_vessel_redness`` -- are asserted to
  ``1e-6``. ``COLOR_RGB2HSV`` is a fixed-point algorithm over two ``cvRound`` division
  tables and Otsu is an integer-histogram loop, so these are transcribed rather than
  derived and can only differ by integer-rounding noise; ``otsu_vessel_redness``
  additionally rounds through ``cv2.normalize``, whose SIMD FMA can differ from a plain
  multiply-add by one ulp before the uint8 truncation. ``1e-6`` absorbs that and nothing
  else -- a real algorithmic error in a lookup table or the Otsu loop moves these values
  by whole units. The observed error is 0 for the HSV medians.

``shutil.which("node")`` guards the whole module so a machine without Node **skips**
rather than fails; the Python parity still has meaning without the browser artefact.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hemolux.metrics.colorimetry import extract_colour_features, extract_colour_features_balanced
from hemolux.training import COLOUR_COLUMNS

#: Exact-float columns: a short arithmetic pipeline, no integer tables.
_EXACT = 1e-9
#: OpenCV-integer columns: fixed-point/table or FMA-sensitive round, hence looser.
_OPENCV = 1e-6

#: Per-column tolerance, keyed by ``COLOUR_COLUMNS``.
TOLERANCE: dict[str, float] = {
    "lab_l": _EXACT,
    "lab_a": _EXACT,
    "lab_b": _EXACT,
    "redness_ratio": _EXACT,
    "erythema_index": _EXACT,
    "high_hue_ratio": _EXACT,
    "otsu_vessel_redness": _OPENCV,
    "hsv_hue": _OPENCV,
    "hsv_sat": _OPENCV,
    "hsv_val": _OPENCV,
    "roi_r": _EXACT,
    "roi_g": _EXACT,
    "roi_b": _EXACT,
}

_RUNNER = Path(__file__).resolve().parents[1] / "app" / "web" / "colorimetry_parity.mjs"


def _expected_row(rgb: np.ndarray, mask: np.ndarray | None, *, balance: bool) -> np.ndarray:
    """One 13-vector as ``extract_colour_rows`` would produce it for a single frame.

    Reproducing the row here rather than reading a cached dataset keeps the test
    self-contained and makes the source of the expected values auditable: every entry is
    the real colourimetry function's output, and the trailing three are the raw ROI means
    from the *original* frame even when the measured ten came from the balanced one.
    """
    measured = (
        extract_colour_features_balanced(rgb, mask)
        if balance
        else extract_colour_features(rgb, mask)
    )
    analysed = rgb if mask is None else rgb[mask]
    roi_mean = analysed.reshape(-1, 3).mean(axis=0) / 255.0
    return np.array(
        [
            measured.lab_l,
            measured.lab_a,
            measured.lab_b,
            measured.redness_ratio,
            measured.erythema_index,
            measured.high_hue_ratio,
            measured.otsu_vessel_redness,
            measured.hsv_hue,
            measured.hsv_sat,
            measured.hsv_val,
            *roi_mean,
        ],
        dtype=np.float64,
    )


def _build_frames() -> tuple[list[dict], list[np.ndarray]]:
    """Deterministic frames covering shape, mask geometry, darkness and saturation.

    The seeds and shapes are fixed so the tolerances above are a property of the port and
    not of a lucky draw. At least one frame carries fewer than 100 ROI pixels, which
    switches ``high_hue_ratio`` from percentiles to a true min/max, so both branches are
    exercised rather than one being assumed.
    """
    rng = np.random.default_rng(20240607)
    cases: list[dict] = []
    expected: list[np.ndarray] = []

    def add(rgb: np.ndarray, mask: np.ndarray | None, *, balance: bool) -> None:
        height, width = rgb.shape[:2]
        cases.append(
            {
                "width": width,
                "height": height,
                "rgb": rgb.ravel().tolist(),
                "mask": None if mask is None else mask.ravel().astype(np.uint8).tolist(),
                "balance": balance,
            }
        )
        expected.append(_expected_row(rgb, mask, balance=balance))

    # A random frame with a circular ROI, under both exposure paths and with no mask.
    height, width = 40, 48
    rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    blob = ((yy - height / 2) ** 2 + (xx - width / 2) ** 2) < (min(height, width) / 3) ** 2
    add(rgb, blob, balance=True)
    add(rgb, blob, balance=False)
    add(rgb, None, balance=True)
    # A mask that covers every pixel leaves no reference outside it, so white_balance
    # must fall back to the whole frame rather than divide by an empty estimate.
    add(rgb, np.ones((height, width), dtype=bool), balance=True)

    # A warm-skewed surround with a tight ROI, so grey-world has a real illuminant to
    # divide out rather than a frame whose channels are already balanced.
    warm = rgb.copy()
    warm[..., 0] = np.clip(warm[..., 0].astype(np.int32) + 40, 0, 255)
    warm[..., 2] = np.clip(warm[..., 2].astype(np.int32) - 40, 0, 255)
    add(warm.astype(np.uint8), blob, balance=True)

    # The degenerate all-dark frame: every measured column is zero or NaN, and the
    # pipeline must not divide by zero or raise.
    add(np.zeros((20, 24, 3), dtype=np.uint8), None, balance=True)

    # A dark-but-nonzero frame, where ``srgb_to_linear``'s data-driven peak must pick 255.
    add(np.full((20, 24, 3), 6, dtype=np.uint8), np.ones((20, 24), dtype=bool), balance=True)

    # Clipped / saturated: half black, half white, with a mask over one half only.
    clipped = np.zeros((16, 18, 3), dtype=np.uint8)
    clipped[:8] = 255
    band = np.zeros((16, 18), dtype=bool)
    band[:8] = True
    add(clipped, band, balance=True)

    # Fewer than 100 ROI pixels: high_hue_ratio takes the min/max branch.
    tiny = rng.integers(30, 220, size=(12, 10, 3), dtype=np.uint8)
    tiny_mask = np.zeros((12, 10), dtype=bool)
    tiny_mask[2:6, 2:7] = True  # 20 pixels
    add(tiny, tiny_mask, balance=True)

    # A large, non-square frame with a mask covering most of it, so the mask-excluded
    # illuminant estimate has only a sliver of reference tissue to work from.
    big = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    most = np.ones((48, 64), dtype=bool)
    most[:, :6] = False
    add(big, most, balance=True)

    return cases, expected


@pytest.fixture(scope="module")
def parity() -> dict:
    """Run the Node runner once and return its rows beside the Python expectation."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot exercise the browser colourimetry module")

    cases, expected = _build_frames()
    payload = {"cases": cases}
    # A module-scoped fixture outlives pytest's function-scoped ``tmp_path``, so the
    # payload gets its own short-lived file next to the runner.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        json_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [node, str(_RUNNER), str(json_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        json_path.unlink(missing_ok=True)

    parsed = json.loads(completed.stdout)
    columns = parsed["columns"]
    assert columns == list(COLOUR_COLUMNS), (
        f"the JS emitted columns {columns}, expected {list(COLOUR_COLUMNS)}"
    )
    rows = [
        np.array([np.nan if value is None else value for value in row], dtype=np.float64)
        for row in parsed["rows"]
    ]
    assert len(rows) == len(expected)
    return {"columns": columns, "rows": rows, "expected": expected}


@pytest.mark.parametrize("column", COLOUR_COLUMNS)
def test_column_matches_python(column: str, parity: dict) -> None:
    """The per-column maximum absolute difference must sit inside its stated tolerance."""
    index = parity["columns"].index(column)
    # NaN on both sides is parity; NaN on one side is a mismatch the tolerance cannot
    # absorb, so it is asserted separately rather than folded into a numeric difference.
    differences = []
    for expected, row in zip(parity["expected"], parity["rows"], strict=True):
        python_value = expected[index]
        js_value = row[index]
        if np.isnan(python_value) or np.isnan(js_value):
            assert np.isnan(python_value) and np.isnan(js_value), (
                f"{column}: NaN on one side only (python={python_value}, js={js_value})"
            )
            differences.append(0.0)
        else:
            differences.append(abs(python_value - js_value))

    worst = max(differences)
    tolerance = TOLERANCE[column]
    assert worst <= tolerance, (
        f"{column}: max abs diff {worst:.3e} exceeds tolerance {tolerance:.0e}"
    )
