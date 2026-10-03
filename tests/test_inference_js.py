"""Browser/Python parity for the deep preprocessing.

The exported graph owns colour, so the browser owns geometry: crop, resize to
224x224, divide by 255, lay out NCHW. Only the resize can drift, and a drift of
one pixel in the sampling grid is a small image-space error that becomes a
silent shift in every prediction. So this module:

* regenerates a deterministic frame on both sides from the same LCG,
* compares the JavaScript resize against ``cv2.resize`` (``INTER_LINEAR``),
  which is the call :func:`hemolux.data.dataset.build_preprocess` makes,
* checks the tensor layout and the ``/255`` scale against the resized image,
* checks the ROI box maths and the decode's use of the validated residual.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

_HARNESS = Path(__file__).resolve().parent / "inference_harness.mjs"


def _lcg_frame(width: int, height: int, seed: int) -> np.ndarray:
    """Rebuild exactly the frame ``inference_harness.mjs`` generated."""
    state = seed & 0xFFFFFFFF
    frame = np.empty((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            for c in range(3):
                state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                frame[y, x, c] = state >> 24
    return frame


@pytest.fixture(scope="module")
def payload() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot exercise the browser preprocessing")
    completed = subprocess.run([node, str(_HARNESS)], capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


@pytest.mark.parametrize("index", [0, 1], ids=["upsample", "downsample"])
def test_resize_matches_opencv_within_one_level(payload: dict, index: int) -> None:
    case = payload["resizeCases"][index]
    size = payload["size"]
    frame = _lcg_frame(case["w"], case["h"], case["seed"])

    expected = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    actual = np.asarray(case["resized"], dtype=np.uint8).reshape(size, size, 3)

    # cv2 interpolates in fixed point, so an exact match is not the contract;
    # a one-level (1/255) agreement is, and is far tighter than any model cares.
    worst = np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()
    assert worst <= 1, f"{case['name']}: resize differs from cv2 by {worst} levels"


def test_tensor_is_scaled_and_channel_first(payload: dict) -> None:
    for case in payload["resizeCases"]:
        plane = payload["size"] ** 2
        samples = np.asarray(case["resized"], dtype=np.float32).reshape(-1, 3)
        for flat_index, value in case["tensorSamples"]:
            channel = flat_index // plane
            position = flat_index % plane
            assert value == pytest.approx(samples[position][channel] / 255.0, abs=1e-6)
        assert case["tensorLayoutError"] < 1e-6


def test_roi_box_is_clamped_to_the_frame(payload: dict) -> None:
    roi = payload["roi"]
    assert roi["inside"] == {"x": 5, "y": 12, "w": 10, "h": 16}
    assert roi["clipped"] == {"x": 0, "y": 0, "w": 7, "h": 7}
    assert roi["onEdge"] == {"x": 88, "y": 46, "w": 12, "h": 8}


def test_decode_prefers_the_validated_residual(payload: dict) -> None:
    validated = payload["decode"]["withResidual"]
    assert validated["hb"] == pytest.approx(11.5)
    assert validated["sigma"] == pytest.approx(1.4)
    assert validated["lo"] < validated["hb"] < validated["hi"]
    assert validated["posteriorSigma"] == pytest.approx(0.9)

    # Without a report there is no validated interval, so the posterior width is
    # used and labelled as such rather than presented as calibrated.
    fallback = payload["decode"]["fallback"]
    assert fallback["sigma"] == pytest.approx(0.9)
