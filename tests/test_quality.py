"""Image-quality gate tests.

The gate is the only thing standing between a bad photograph and a confident
wrong haemoglobin number, so the tests care about two things: each rejection
fires on the image it is meant to catch, and the rejections fire in a useful
order.

That ordering is asserted, not assumed. ``check_quality`` checks brightness
before sharpness before framing, because a user retaking a dark photo will fix
the darkness and the blur may go with it -- telling them "it's blurry" first
would send them away with an unchanged, still-dark image. The same reasoning
applies to the code comment: a dark image's Laplacian variance is not a
trustworthy focus reading in the first place.

The thresholds are named constants in the module, so the tests build images that
clear or miss them by a wide margin rather than nudging an edge. A test that sits
one grey level from a boundary only pins the constant it already knows.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hemolux.data.quality import (
    BLUR_VARIANCE,
    BRIGHT_LUMA,
    CLIPPED_FRACTION,
    DARK_LUMA,
    ROI_MAX_AREA,
    ROI_MIN_AREA,
    QualityReport,
    check_quality,
    laplacian_sharpness,
)
from hemolux.metrics.calibration import should_abstain

# --------------------------------------------------------------------------- #
# Synthetic images
# --------------------------------------------------------------------------- #


def _grey(value: int, size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _noisy(luma: int = 128, size: int = 128, seed: int = 0, spread: int = 32) -> np.ndarray:
    """Uniform noise centred on ``luma``: correct exposure, high sharpness.

    Uniform rather than Gaussian because the Laplacian of white noise has a
    predictable, high variance, which is what keeps these fixtures clear of
    ``BLUR_VARIANCE`` by a margin rather than by luck.
    """
    rng = np.random.default_rng(seed)
    lo = max(0, luma - spread // 2)
    return rng.integers(lo, lo + spread, size=(size, size, 3), dtype=np.uint8)


def _mask(area: float, size: int = 128) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    n = max(1, round(area * size * size))
    flat = m.reshape(-1)
    flat[:n] = 255
    return m


# --------------------------------------------------------------------------- #
# Fixture preconditions
# --------------------------------------------------------------------------- #


class TestFixturesClearTheThresholds:
    def test_noise_image_is_exposed_correctly(self) -> None:
        r = check_quality(_noisy(128))
        assert DARK_LUMA < r.luma < BRIGHT_LUMA

    def test_noise_image_is_sharp(self) -> None:
        assert laplacian_sharpness(_noisy()[:, :, 0]) > 10 * BLUR_VARIANCE

    def test_flat_image_is_blurred(self) -> None:
        """A constant image has a zero Laplacian everywhere. This is the one
        focus measure that is exact rather than heuristic."""
        assert laplacian_sharpness(_grey(128)[:, :, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_flat_image_is_correctly_exposed(self) -> None:
        """So the flat fixture fails on blur alone and not on brightness."""
        assert DARK_LUMA < 128.0 < BRIGHT_LUMA


# --------------------------------------------------------------------------- #
# The focus measure
# --------------------------------------------------------------------------- #


class TestLaplacianSharpness:
    def test_is_the_variance_of_the_laplacian(self) -> None:
        import cv2

        gray = _noisy()[:, :, 0]
        assert laplacian_sharpness(gray) == pytest.approx(
            float(cv2.Laplacian(gray, cv2.CV_64F).var())
        )

    def test_noise_beats_a_gradient(self) -> None:
        """A smooth ramp has a near-zero second derivative; noise does not. This
        is the discrimination the threshold is making."""
        ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
        assert laplacian_sharpness(_noisy()[:, :, 0]) > laplacian_sharpness(ramp)

    def test_a_single_bright_spot_raises_it(self) -> None:
        flat = _grey(128)[:, :, 0].copy()
        boosted = flat.copy()
        boosted[60:68, 60:68] = 255
        assert laplacian_sharpness(boosted) > laplacian_sharpness(flat)

    def test_returned_value_is_a_plain_float(self) -> None:
        assert isinstance(laplacian_sharpness(_grey(128)[:, :, 0]), float)


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


class TestQualityVerdicts:
    def test_a_good_photo_passes(self) -> None:
        r = check_quality(_noisy(128), _mask(0.10))
        assert r.passed is True
        assert r.reason is None
        assert r.message is None

    def test_a_good_photo_passes_without_a_mask(self) -> None:
        """The web path may not have a mask if the segmenter abstained. Framing
        simply contributes full marks rather than failing the photo."""
        r = check_quality(_noisy(128))
        assert r.passed is True
        assert r.roi_area is None

    def test_unreadable_image_is_a_decode_failure_not_a_quality_failure(self) -> None:
        """A different remediation entirely -- re-upload, do not retake."""
        for bad in (None, np.zeros((0, 0, 3), dtype=np.uint8)):
            r = check_quality(bad)  # type: ignore[arg-type]
            assert r.passed is False
            assert r.reason == "DECODE_FAIL"
            assert r.score == 0.0

    def test_dark_photo_is_rejected_as_too_dark(self) -> None:
        r = check_quality(_grey(10))
        assert r.reason == "TOO_DARK"
        assert r.luma == pytest.approx(10.0)

    def test_overexposed_photo_is_rejected(self) -> None:
        r = check_quality(_grey(250))
        assert r.reason == "TOO_BRIGHT"

    def test_blowing_a_small_fraction_of_pixels_is_enough_to_reject(self) -> None:
        """Not the mean that matters but the highlights: a specular reflection on
        the sclera destroys the channel ratio this project measures. 5% blown
        pixels at an otherwise acceptable luma of 205."""
        img = _grey(200).copy()
        img[:16] = 255  # 16 of 128 rows = 12.5%
        r = check_quality(img)
        assert r.reason == "TOO_BRIGHT"
        assert r.luma < BRIGHT_LUMA, "the rejection must come from clipping, not luma"
        assert r.clipped_fraction > CLIPPED_FRACTION

    def test_a_clipped_fraction_just_under_the_limit_is_accepted(self) -> None:
        img = _noisy(140, spread=8)
        img[:3] = 255  # 3/128 = 2.34%, marginally over
        assert check_quality(img).reason == "TOO_BRIGHT"

        img2 = _noisy(140, spread=8)
        img2[:2] = 255  # 2/128 = 1.56%, under the 2% limit
        assert check_quality(img2).reason != "TOO_BRIGHT"

    def test_blurry_photo_is_rejected(self) -> None:
        r = check_quality(_grey(128))
        assert r.reason == "TOO_BLUR"
        assert r.sharpness == pytest.approx(0.0)

    def test_a_tiny_roi_is_a_framing_failure(self) -> None:
        r = check_quality(_noisy(128), _mask(0.004))
        assert r.reason == "WRONG_FRAMING"
        assert r.roi_area is not None and r.roi_area < ROI_MIN_AREA

    def test_a_roi_filling_the_frame_is_a_framing_failure(self) -> None:
        """Segmentation that leaks to the whole image produces a confidently
        wrong colour average, so it has to be caught."""
        r = check_quality(_noisy(128), _mask(1.0))
        assert r.reason == "WRONG_FRAMING"
        assert r.roi_area is not None and r.roi_area > ROI_MAX_AREA

    @pytest.mark.parametrize("area", [0.02, 0.05, 0.10, 0.40, 0.80])
    def test_roi_areas_inside_the_band_are_accepted(self, area: float) -> None:
        assert check_quality(_noisy(128), _mask(area)).passed is True

    def test_a_boolean_mask_is_accepted_as_well_as_a_0_255_mask(self) -> None:
        b = _mask(0.10) > 0
        assert check_quality(_noisy(128), b).roi_area == pytest.approx(0.10, abs=0.01)

    def test_an_all_zero_mask_is_a_framing_failure(self) -> None:
        """A mask that matched nothing must never read as "no mask given"."""
        r = check_quality(_noisy(128), np.zeros((128, 128), dtype=np.uint8))
        assert r.reason == "WRONG_FRAMING"
        assert r.roi_area == pytest.approx(0.0)

    def test_every_rejection_carries_a_message_for_the_user(self) -> None:
        """A rejection the UI cannot explain is an infuriating one."""
        cases = [
            (_grey(10), None),
            (_grey(250), None),
            (_grey(128), None),
            (_noisy(128), _mask(0.001)),
        ]
        for img, mask in cases:
            r = check_quality(img, mask)
            assert r.passed is False, f"{img.shape} with mask={mask is not None} passed"
            assert r.reason and r.reason.isupper()
            assert r.message and len(r.message) > 20


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


class TestCheckOrder:
    def test_dark_and_blurry_reports_dark_first(self) -> None:
        """A flat dark image fails both. Telling the user it is blurry first
        sends them away to stabilise the phone while the photo is still
        unreadably dark."""
        assert check_quality(_grey(5)).reason == "TOO_DARK"

    def test_dark_and_misframed_reports_dark_first(self) -> None:
        assert check_quality(_grey(5), _mask(0.001)).reason == "TOO_DARK"

    def test_blurry_and_misframed_reports_blur_first(self) -> None:
        """Exposure is fine, so the next most visible problem is focus."""
        assert check_quality(_grey(128), _mask(0.001)).reason == "TOO_BLUR"

    def test_blurred_and_clipped_reports_brightness_first(self) -> None:
        img = _grey(250)
        assert check_quality(img).reason == "TOO_BRIGHT"


# --------------------------------------------------------------------------- #
# Score
# --------------------------------------------------------------------------- #


class TestQualityScore:
    def test_is_bounded_for_every_fixture(self) -> None:
        for img, mask in [
            (_noisy(128), None),
            (_noisy(128), _mask(0.10)),
            (_grey(0), None),
            (_grey(255), None),
            (_grey(128), _mask(0.5)),
            (_grey(20), _mask(0.9)),
        ]:
            r = check_quality(img, mask)
            assert 0.0 <= r.score <= 1.0, f"score {r.score} outside [0, 1]"

    def test_exposure_terme_is_worst_furthest_from_mid_grey(self) -> None:
        """Exposure peaks at luma 130 and decays with absolute distance, which is
        the only reason a dark image and a bright image both score badly."""
        mid = check_quality(_noisy(130, spread=8)).score
        dark = check_quality(_noisy(80, spread=8)).score
        bright = check_quality(_noisy(190, spread=8)).score
        assert mid > dark
        assert mid > bright

    def test_framing_saturates_at_ten_percent_of_the_frame(self) -> None:
        """Documented behaviour, and it is worth pinning because it is
        surprising.

        ``framing = clip(roi_area / 0.10, 0, 1)``, so a mask covering 10% of the
        frame and one covering 100% receive identical framing credit and score
        identically. A leaked full-frame mask is therefore *not* penalised by the
        score -- it is caught by the hard ``ROI_MAX_AREA`` verdict instead. The
        two mechanisms are not redundant, and conflating them would be the bug.
        """
        ideal = check_quality(_noisy(128, size=160), _mask(0.10, size=160)).score
        full = check_quality(_noisy(128, size=160), _mask(1.0, size=160)).score
        assert full == pytest.approx(ideal, abs=1e-9)

    def test_a_thin_but_valid_roi_scores_below_an_ideal_one(self) -> None:
        """2% of the frame clears ``ROI_MIN_AREA`` yet earns only 0.2 framing."""
        thin = check_quality(_noisy(128), _mask(0.02)).score
        ideal = check_quality(_noisy(128), _mask(0.10)).score
        assert thin < ideal

    def test_no_mask_scores_at_least_as_high_as_a_bad_mask(self) -> None:
        assert check_quality(_noisy(128)).score > check_quality(_noisy(128), _mask(0.001)).score

    def test_a_blurred_photo_is_caught_by_both_paths_at_once(self) -> None:
        """Redundancy rather than independence, and deliberately so.

        A fully blurred photograph fails the hard gate *and* scores 0.594, which
        is below ``QUALITY_ABSTAIN``, so the score would refuse it even if the
        hard verdict were bypassed. Neither condition is trusted alone.

        The failure this guards against is the opposite one: a photograph that
        sails through both. That is what the reachable-threshold assertion in
        ``test_calibration.py`` covers, and it is the reason the constant moved
        off 0.35 -- where it sat below every score in the corpus and could never
        fire at all.
        """
        blurred = check_quality(_grey(128))
        assert blurred.passed is False
        assert blurred.reason == "TOO_BLUR"
        assert should_abstain(0.1, 10.0, quality=blurred.score) is True

    def test_a_good_photo_passes_the_score_gate(self) -> None:
        """The other side: the gate must not refuse an image the hard gate has
        just approved, or every accepted photograph would abstain."""
        good = check_quality(_noisy(128), _mask(0.10))
        assert good.passed is True
        assert should_abstain(0.1, 10.0, quality=good.score) is False


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


class TestQualityReportSerialisation:
    def test_to_dict_carries_every_field_the_ui_needs(self) -> None:
        d = check_quality(_noisy(128), _mask(0.10)).to_dict()
        assert set(d) == {
            "passed",
            "reason",
            "message",
            "score",
            "luma",
            "clipped_fraction",
            "sharpness",
            "roi_area",
        }

    def test_to_dict_is_json_ready(self) -> None:
        """These dicts go straight into the telemetry row and the API response."""
        import json

        json.dumps(check_quality(_grey(10)).to_dict())

    def test_absent_roi_stays_none_rather_than_becoming_zero(self) -> None:
        """None and 0.0 mean different things: no mask versus an empty mask. The
        UI distinguishes them and so must the payload."""
        assert check_quality(_noisy(128)).to_dict()["roi_area"] is None
        empty = np.zeros((128, 128), dtype=np.uint8)
        assert check_quality(_noisy(128), empty).to_dict()["roi_area"] == 0.0

    def test_report_is_frozen(self) -> None:
        r = QualityReport(True, None, None, 1.0, 128.0, 0.0, 500.0, 0.1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.passed = False  # type: ignore[misc]
