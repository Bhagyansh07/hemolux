"""Colour-science tests, against analytically known values.

Every assertion here is derived from the standard, not from running the code and
recording what it produced. A test that asserts the implementation agrees with
itself is worthless.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hemolux.metrics.colorimetry import (
    _as_uint8,
    balanced_rgb,
    conjunctival_pigmentation,
    erythema_index,
    extract_colour_features,
    extract_colour_features_balanced,
    high_hue_ratio,
    ita_angle,
    ita_band,
    mask_mean_lab,
    pigmentation_band,
    redness_ratio,
    rgb_to_lab,
    srgb_to_linear,
    white_balance,
)


class TestSrgbLinearisation:
    def test_endpoints_are_identity(self):
        assert srgb_to_linear(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
        assert srgb_to_linear(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-12)

    def test_8bit_input_is_normalised_then_linearised(self):
        # A common bug: handing 0-255 data to a transfer function defined on
        # 0-1, which crushes every channel onto the near-linear segment. The
        # function must both detect the 8-bit range and still apply the EOTF.
        # 128/255 = 0.50196 encoded; its linear value is 0.21586, which is the
        # standard linear-light figure for 8-bit mid grey.
        assert srgb_to_linear(np.array([255.0]))[0] == pytest.approx(1.0, abs=1e-9)
        assert srgb_to_linear(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
        assert srgb_to_linear(np.array([128.0]))[0] == pytest.approx(0.2158605, abs=1e-6)

    def test_unit_input_is_linearised_too(self):
        # 0.50196 already in 0-1 must give the identical answer as 128/255,
        # i.e. range detection must not change the result.
        assert srgb_to_linear(np.array([128.0 / 255.0]))[0] == pytest.approx(
            srgb_to_linear(np.array([128.0]))[0], abs=1e-12
        )

    def test_monotonic_increasing(self):
        vals = np.linspace(0, 255, 64)
        out = srgb_to_linear(vals)
        assert np.all(np.diff(out) > 0)


class TestRgbToLab:
    def test_white_is_L100_neutral(self):
        lab = rgb_to_lab(np.full((1, 1, 3), 255, dtype=np.uint8))[0, 0]
        assert lab[0] == pytest.approx(100.0, abs=1e-2)
        assert abs(lab[1]) < 1e-2
        assert abs(lab[2]) < 1e-2

    def test_black_is_L0(self):
        lab = rgb_to_lab(np.zeros((1, 1, 3), dtype=np.uint8))[0, 0]
        assert lab[0] == pytest.approx(0.0, abs=1e-9)

    def test_mid_grey_matches_published_value(self):
        # #808080 -> L* ~= 53.585, a* ~= 0, b* ~= 0. Standard reference point.
        lab = rgb_to_lab(np.full((1, 1, 3), 128, dtype=np.uint8))[0, 0]
        assert lab[0] == pytest.approx(53.585, abs=0.05)
        assert abs(lab[1]) < 0.5
        assert abs(lab[2]) < 0.5

    def test_pure_srgb_primaries_match_published_cielab(self):
        # Canonical CIELAB (D65) values for the sRGB primaries. Worth stating
        # plainly because the naive expectation is wrong: blue has a* = +79,
        # not negative. The a* axis is green(-)/red(+), and blue is not green.
        expected = {
            "red": ((255, 0, 0), (53.241, 80.092, 67.203)),
            "green": ((0, 255, 0), (87.735, -86.183, 83.179)),
            "blue": ((0, 0, 255), (32.297, 79.188, -107.860)),
            "yellow": ((255, 255, 0), (97.137, -21.554, 94.478)),
        }
        for name, (rgb, (l_exp, a_exp, b_exp)) in expected.items():
            lab = rgb_to_lab(np.full((1, 1, 3), rgb, dtype=np.uint8))[0, 0]
            assert lab[0] == pytest.approx(l_exp, abs=0.02), name
            assert lab[1] == pytest.approx(a_exp, abs=0.02), name
            assert lab[2] == pytest.approx(b_exp, abs=0.02), name

    def test_only_green_is_negative_on_the_a_axis(self):
        # The sign convention conjunctival_pigmentation depends on is on b*,
        # not a*. Assert it explicitly so a channel swap cannot pass.
        green = rgb_to_lab(np.full((1, 1, 3), (0, 255, 0), dtype=np.uint8))[0, 0]
        assert green[1] < 0.0

    def test_b_axis_separates_blue_from_yellow(self):
        # This is the sign convention the melanin proxy is built on.
        blue = rgb_to_lab(np.full((1, 1, 3), (0, 0, 255), dtype=np.uint8))[0, 0]
        yellow = rgb_to_lab(np.full((1, 1, 3), (255, 255, 0), dtype=np.uint8))[0, 0]
        assert blue[2] < -80.0
        assert yellow[2] > 80.0

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="H, W, 3"):
            rgb_to_lab(np.zeros((10, 10), dtype=np.uint8))

    def test_accepts_float_input(self):
        lab = rgb_to_lab(np.ones((1, 1, 3), dtype=np.float32))
        assert lab[0, 0, 0] == pytest.approx(100.0, abs=1e-2)


class TestMaskedMean:
    def test_mask_restricts_the_average(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:5] = 255
        full = mask_mean_lab(img, None)
        top = mask_mean_lab(img, np.arange(10)[:, None] < 5)
        # Half white, half black, so L* averages to 50. L* is bounded to
        # [0, 100] by the standard, so it can never exceed 100 here.
        assert full[0] == pytest.approx(50.0, abs=0.05)
        assert top[0] > 99.0  # white half only

    def test_empty_mask_is_nan_not_a_crash(self):
        # An all-False mask must not silently produce the full-frame mean, which
        # would make a failed segmentation look like a valid measurement.
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        assert np.isnan(mask_mean_lab(img, np.zeros((4, 4), dtype=bool))).all()


class TestRednessAndErythema:
    def test_redness_ratio_is_bounded(self):
        for lab in ([50, 40, 10], [50, -40, 10], [50, 0, 30], [50, 0, -30]):
            r = redness_ratio(np.array(lab, dtype=float))
            assert -1.0 <= r <= 1.0

    def test_pure_red_axis_is_one(self):
        assert redness_ratio(np.array([50.0, 30.0, 0.0])) == pytest.approx(1.0)

    def test_pure_green_axis_is_minus_one(self):
        assert redness_ratio(np.array([50.0, -30.0, 0.0])) == pytest.approx(-1.0)

    def test_achromatic_is_nan_not_zero(self):
        # a* = b* = 0 has no hue, so redness is undefined. Returning 0.0 would
        # silently claim "no redness", which is a measurement, not an absence of one.
        assert np.isnan(redness_ratio(np.array([50.0, 0.0, 0.0])))

    def test_erythema_index_polarity_is_inverted(self):
        # This is a sign trap. Higher haemoglobin -> redder tissue -> lower EI.
        redder = erythema_index(np.array([50.0, 40.0, 10.0]))
        paler = erythema_index(np.array([50.0, 5.0, 30.0]))
        assert redder < paler

    def test_erythema_index_is_finite_for_valid_input(self):
        assert np.isfinite(erythema_index(np.array([50.0, 20.0, 15.0])))


class TestHighHueRatio:
    def test_grey_patch_has_ratio_one(self):
        assert high_hue_ratio(np.full((2, 2, 3), 128, dtype=np.uint8)) == pytest.approx(1.0)

    def test_ratio_increases_with_chromaticity(self):
        grey = high_hue_ratio(np.full((2, 2, 3), 128, dtype=np.uint8))
        colourful = high_hue_ratio(
            np.array(
                [[[200, 30, 40], [180, 60, 90], [210, 20, 30], [150, 70, 110]]], dtype=np.uint8
            )
        )
        assert colourful > grey

    def test_near_black_channel_is_nan_not_infinity(self):
        assert np.isnan(high_hue_ratio(np.array([[[0, 10, 10]]], dtype=np.uint8)))


class TestItaAngle:
    def test_matches_formula_on_known_points(self):
        # arctan2((L*-50)/b*) * 180/pi, computed by hand for these vectors.
        assert ita_angle(np.array([50.0, 0.0, 25.0])) == pytest.approx(0.0, abs=1e-9)
        # L*=100, b*=+20 -> arctan(50/20) = 68.20 deg
        assert ita_angle(np.array([100.0, 0.0, 20.0])) == pytest.approx(68.1986, abs=1e-3)
        # L*=20, b*=+20 -> arctan(-30/20) = -56.31 deg
        assert ita_angle(np.array([20.0, 0.0, 20.0])) == pytest.approx(-56.3099, abs=1e-3)

    def test_b_star_zero_does_not_divide_by_zero(self):
        # The published formula divides by b*, which is undefined here. arctan2
        # makes this a well-defined answer instead of a crash or an inf.
        assert np.isfinite(ita_angle(np.array([70.0, 0.0, 0.0])))
        assert np.isfinite(ita_angle(np.array([30.0, 0.0, 0.0])))

    def test_darker_skin_gives_lower_ita(self):
        light = ita_angle(np.array([75.0, 5.0, 18.0]))
        dark = ita_angle(np.array([28.0, 10.0, 14.0]))
        assert dark < light

    def test_bands_follow_the_six_category_ita_scale(self):
        # Chardon 1991 / Del Bino & Bernerd 2013, as tabulated in Ly et al.,
        # J Invest Dermatol 140:3 (2020). Six categories, not seven: there is
        # no "very_dark". Boundaries are 55 / 41 / 28 / 10 / -30.
        assert ita_band(80.0) == "very_light"
        assert ita_band(55.0) == "very_light"  # boundary is inclusive upward
        assert ita_band(50.0) == "light"
        assert ita_band(41.0) == "light"
        assert ita_band(35.0) == "intermediate"
        assert ita_band(28.0) == "intermediate"
        assert ita_band(20.0) == "tan"
        assert ita_band(10.0) == "tan"

    def test_brown_band_exists_and_is_not_labelled_dark(self):
        # Regression test. The band was previously missing, which merged
        # ITA -30..10 into "dark" and so mislabelled most South Asian skin
        # tones as dark, in the one subgroup this project is actually about.
        assert ita_band(0.0) == "brown"
        assert ita_band(-29.9) == "brown"
        assert ita_band(-30.0) == "brown"

    def test_dark_is_the_floor_band(self):
        assert ita_band(-31.0) == "dark"
        assert ita_band(-90.0) == "dark"  # arctan2 lower bound of ITA

    def test_bands_cover_the_whole_ita_domain(self):
        # Every finite ITA must land in exactly one band, with no gaps, and the
        # ordering must never go backwards as ITA decreases.
        order = ["very_light", "light", "intermediate", "tan", "brown", "dark"]
        seen = [ita_band(v) for v in np.linspace(90.0, -90.0, 721)]
        for value, band in zip(np.linspace(90.0, -90.0, 721), seen, strict=True):
            assert band in order, value
        ranks = [order.index(b) for b in seen]
        assert all(np.diff(ranks) >= 0), "band sequence is not monotone"

    def test_nonfinite_ita_is_unknown_not_a_band(self):
        # A missing measurement must never be silently counted as a
        # pigmentation subgroup.
        assert ita_band(float("nan")) == "unknown"
        assert ita_band(float("inf")) == "unknown"
        assert ita_band(float("-inf")) == "unknown"


class TestPigmentation:
    def test_higher_b_star_means_higher_pigment_score(self):
        low = conjunctival_pigmentation(np.array([50.0, 20.0, 8.0]))
        high = conjunctival_pigmentation(np.array([50.0, 20.0, 30.0]))
        assert high > low

    def test_score_is_clipped_to_unit_interval(self):
        assert conjunctival_pigmentation(np.array([50.0, 0.0, -50.0])) == 0.0
        assert conjunctival_pigmentation(np.array([50.0, 0.0, 500.0])) == 1.0

    def test_nan_in_b_star_propagates(self):
        # The proxy reads b* only, so NaN must be in the channel it actually
        # uses. NaN in L* or a* is irrelevant by design and must not mask a
        # perfectly good b* reading.
        assert np.isnan(conjunctival_pigmentation(np.array([50.0, 0.0, np.nan])))
        assert np.isnan(conjunctival_pigmentation(np.array([np.nan, 0.0, np.nan])))
        assert not np.isnan(conjunctival_pigmentation(np.array([np.nan, np.nan, 20.0])))

    def test_pigmentation_band_uses_ink_not_absorption_direction(self):
        # b* rises with melanin concentration within the conjunctival ROI, so
        # a more pigmented ROI scores higher. Pinned so a sign flip in the
        # proxy cannot pass silently.
        low = conjunctival_pigmentation(np.array([50.0, 20.0, 8.0]))
        high = conjunctival_pigmentation(np.array([50.0, 20.0, 30.0]))
        assert high > low

    def test_bands_partition_the_unit_interval(self):
        assert pigmentation_band(0.1) == "low"
        assert pigmentation_band(0.3) == "medium"
        assert pigmentation_band(0.9) == "high"
        assert pigmentation_band(float("nan")) == "unknown"


class TestFeatureExtraction:
    def test_all_fields_populated_on_a_real_patch(self):
        rng = np.random.default_rng(0)
        img = rng.integers(60, 200, size=(64, 64, 3), dtype=np.uint8)
        feats = extract_colour_features(img)
        for name in ("lab_l", "lab_a", "lab_b", "hsv_hue", "hsv_sat", "hsv_val"):
            assert np.isfinite(getattr(feats, name)), name

    def test_mask_changes_the_result(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:16] = 220
        assert (
            extract_colour_features(img).lab_l
            != extract_colour_features(img, np.arange(32)[:, None] < 16).lab_l
        )

    def test_richer_photograph_is_redder(self):
        # Sanity on the direction of the signal: a warm patch must score higher
        # on redness than a neutral one of the same lightness.
        red = np.full((32, 32, 3), (200, 60, 55), dtype=np.uint8)
        neutral = np.full((32, 32, 3), (128, 128, 128), dtype=np.uint8)
        assert extract_colour_features(red).lab_a > extract_colour_features(neutral).lab_a


# --------------------------------------------------------------------------- #
# Exposure normalisation
# --------------------------------------------------------------------------- #

#: A 60x60 synthetic eye under a warm illuminant. The surround is a *gradient* rather
#: than a flat colour, which matters: on a uniform surround the grey-world estimate
#: and the white-patch estimate are both that same constant, so the two methods would
#: be indistinguishable and a test asserting they differ would be measuring the
#: fixture rather than the code.
SURROUND_LO = np.array([135.0, 105.0, 70.0])
SURROUND_HI = np.array([165.0, 125.0, 90.0])
CONJUNCTIVA = np.array([165.0, 85.0, 78.0])
#: A perfectly neutral surround, for the fixed-point test.
NEUTRAL = np.array([130.0, 130.0, 130.0])


def eye_crop(
    surround_lo: np.ndarray = SURROUND_LO,
    surround_hi: np.ndarray = SURROUND_HI,
    patch: np.ndarray = CONJUNCTIVA,
    size: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """A frame and its ROI mask. The patch is the middle square of the frame.

    The surround ramps left to right, so its per-channel mean differs measurably from
    its per-channel maximum and the two white-balance methods have something to
    disagree about.
    """
    ramp = np.linspace(0.0, 1.0, size)[:, None]
    frame = (surround_lo + ramp * (surround_hi - surround_lo))[None, :, :]
    frame = np.repeat(frame, size, axis=0)
    lo, hi = size // 4, size - size // 4
    frame[lo:hi, lo:hi] = patch
    mask = np.zeros((size, size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return frame.round().astype(np.uint8), mask


def roi_ratio(balanced: np.ndarray, mask: np.ndarray, high: int = 0, low: int = 1) -> float:
    """A channel ratio inside the ROI. The quantity haemoglobin is supposed to move."""
    roi = balanced[mask].mean(axis=0)
    return float(roi[high] / roi[low])


def test_white_balance_leaves_a_neutrally_lit_frame_alone() -> None:
    """A grey illuminant is the fixed point, and asserting it pins the whole design.

    There are two multiplications in ``white_balance`` -- a per-channel division by
    the estimated illuminant and a single scalar that restores overall brightness.
    When the reference is neutral those two cancel *exactly*, and the function
    returns its input. That is not a special case bolted on; it is the property that
    says the correction only removes what is actually there.

    Without this test the two factors could drift apart and the function would quietly
    start shifting neutrally-lit frames, which is the failure mode a "normalisation"
    routine is most prone to.

    The reference is the surround **excluding** the mask. Passing ``None`` here would
    fold the red patch into the estimate, making the estimated illuminant red and the
    frame's mean non-neutral -- correct behaviour, and the reason this test has to be
    explicit about which region it is balancing against.
    """
    frame, mask = eye_crop(surround_lo=NEUTRAL, surround_hi=NEUTRAL)

    balanced = white_balance(frame, mask)

    assert np.allclose(balanced, frame / 255.0, atol=1e-6)


def test_white_balance_removes_a_per_channel_gain() -> None:
    """The property the whole function exists for.

    Haemoglobin absorbs in the green and reflects in the red, so the quantity that
    carries the signal is the *ratio* of those two channels. An illuminant scales all
    three channels by an unknown gain, which leaves an absolute level meaningless and
    a ratio intact. So the test is not "the output looks nicer" but "the output does
    not move when the lighting does".
    """
    frame, mask = eye_crop()
    gain = np.array([1.30, 0.95, 0.80])  # a warm-to-cool swing, well past plausible
    relit = np.clip((frame * gain).round(), 0, 255).astype(np.uint8)

    before = roi_ratio(white_balance(frame, mask), mask)
    after = roi_ratio(white_balance(relit, mask), mask)

    assert after == pytest.approx(before, rel=0.02), "the balanced ratio moved with the gain"

    # And the uncorrected ratio moved a great deal, so the test is not vacuous: it
    # would also pass if `white_balance` returned a constant.
    assert (
        roi_ratio(relit.astype(np.float64) / 255.0, mask)
        > roi_ratio(frame.astype(np.float64) / 255.0, mask) * 1.3
    )


def test_the_raw_roi_ratio_matches_the_fixture() -> None:
    """The uncorrected value, computed on paper, so the corrected ones are anchored.

    The patch is exactly (165, 85, 78), so the raw ROI red-to-green ratio is 165/85.
    Asserting it here is what stops the corrected-ratio test below from being a
    number copied out of a run with nothing to compare it against.
    """
    frame, mask = eye_crop()
    assert roi_ratio(frame.astype(np.float64) / 255.0, mask) == pytest.approx(165 / 85, abs=1e-3)


def test_the_balanced_ratio_is_reproducible_and_moved() -> None:
    """Asserted against a measured constant, with its tolerance stated.

    The corrected ratio has no closed form, because the scalar that restores lightness
    is a whole-frame mean rather than a per-channel one. What is assertable is that it
    is stable, reproducible, and far from the raw value. The constant was recorded
    from a run; if it moves by more than the tolerance the arithmetic changed and the
    sweep that compares methods against it has to be re-read.
    """
    frame, mask = eye_crop()
    raw = roi_ratio(frame.astype(np.float64) / 255.0, mask)
    balanced = roi_ratio(white_balance(frame, mask), mask)

    assert balanced < raw, "the warm surround should have been pushed down"
    assert balanced == pytest.approx(1.49, abs=0.02), f"the corrected ratio moved to {balanced:.3f}"


def test_the_reference_excludes_the_roi() -> None:
    """The substantive decision in the function, and it is not neutral.

    A grey-world estimate taken *over* the conjunctiva is pulled toward red by the
    very signal being measured, so dividing by it partly undoes the correction. The
    reference must therefore be the tissue around the ROI, which is what grey-world
    actually assumes to be neutral.

    The fixture makes the direction measurable. Including the red patch raises the
    estimated red illuminant, and dividing by a larger red illuminant strips *more*
    red than it should. So the corrected ratio is **higher** with the mask than
    without, and an assertion that only checked "they differ" would pass whichever
    way the sign went.
    """
    frame, mask = eye_crop()

    excluded = roi_ratio(white_balance(frame, mask), mask)
    inclusive = roi_ratio(white_balance(frame, None), mask)

    assert excluded > inclusive, (
        "excluding the ROI did not weaken the correction, so the mask is not reaching "
        "the illuminant estimate"
    )
    assert abs(excluded - inclusive) > 0.05, "the difference is too small to matter"


def test_a_mask_covering_the_whole_frame_falls_back_to_the_frame() -> None:
    """A mask that leaves nothing behind has no reference, and this corpus has one.

    ``Italy/2``'s mask covers 99.99% of the frame. With no reference pixels the only
    available estimate is the whole frame, which is biased -- but biased beats
    dividing by a substituted constant, which would invent an illumination.
    """
    frame, mask = eye_crop()
    everything = np.ones(mask.shape, dtype=bool)

    balanced = white_balance(frame, everything)

    assert np.isfinite(balanced).all()
    assert balanced.max() <= 1.0


def test_a_misshapen_mask_falls_back_rather_than_raising() -> None:
    """A mask from a different crop must not take the whole run down.

    Masks in this corpus come from a separate file and one at a different resolution
    from its frame, so a shape mismatch is a real possibility rather than a
    hypothetical. Raising would abort a 217-patient sweep over one patient.
    """
    frame, _ = eye_crop()

    balanced = white_balance(frame, np.ones((7, 7), dtype=bool))

    assert balanced.shape == frame.shape
    assert np.isfinite(balanced).all()


def test_an_all_black_frame_does_not_divide_by_zero() -> None:
    """``eps`` earns its keep here: the reference mean is exactly zero.

    There is no illuminant to remove from a black frame, so the function returns the
    frame untouched. Substituting ``eps`` instead would divide by 1e-6 and turn a
    black frame into values up to a million, which then propagate into a mean over
    217 patients.
    """
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    balanced = white_balance(frame)

    assert np.isfinite(balanced).all(), "a black frame produced NaN or inf"
    assert balanced.max() == 0.0
    assert balanced.min() == 0.0


def test_an_all_white_frame_does_not_saturate_into_nonsense() -> None:
    """The other degenerate input, and the one ``eps`` alone does not fix.

    Every channel mean is 1.0, so the ratio is 1.0 and the output is the input. What
    has to be asserted is that it does *not* become a field of NaNs from a
    division of ``mean / mean`` where the mean is read after clipping.
    """
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)

    balanced = white_balance(frame)

    assert np.isfinite(balanced).all()
    assert balanced.min() == pytest.approx(1.0)
    assert balanced.max() == pytest.approx(1.0)


def test_a_single_channel_black_frame_is_survivable() -> None:
    """Per-channel guards, not a whole-frame one.

    A frame whose green channel is zero has no green illuminant to estimate but does
    have red and blue. Guarding the frame as a whole would skip the correction
    entirely; guarding per channel fixes the two channels that can be estimated.
    """
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[..., 0] = 200
    frame[..., 2] = 180

    balanced = white_balance(frame)

    assert np.isfinite(balanced).all()
    assert balanced[..., 1].max() == 0.0


def test_the_output_is_float_in_the_unit_range() -> None:
    """A declared contract, so the caller's assumptions are checkable."""
    frame, mask = eye_crop(surround_lo=SURROUND_HI * 1.4, surround_hi=SURROUND_HI * 1.4)
    balanced = white_balance(frame, mask)

    assert balanced.dtype == np.float64
    assert balanced.shape == frame.shape
    assert balanced.min() >= 0.0
    assert balanced.max() <= 1.0


def test_uint8_and_float_input_agree() -> None:
    """The two accepted input scales must not be two different functions.

    ``rgb_to_lab`` accepts either, and a caller reaching this module through the
    dataset loader gets uint8 while a caller testing a hypothesis writes floats. If
    the two disagreed, every comparison between them would be a comparison of scale
    conventions.
    """
    frame, mask = eye_crop()

    from_uint8 = white_balance(frame, mask)
    from_float = white_balance(frame.astype(np.float64) / 255.0, mask)

    assert np.allclose(from_uint8, from_float, atol=1e-6)


def test_white_balance_is_pure() -> None:
    """No mutation of the caller's array, and the same answer every time.

    The input is the only copy of a 12-megapixel frame; a function that modified it in
    place would corrupt the caller's mask alignment for every later use.
    """
    frame, mask = eye_crop()
    original = frame.copy()

    first = white_balance(frame, mask)
    second = white_balance(frame, mask)

    assert np.array_equal(frame, original), "the input was modified in place"
    assert np.array_equal(first, second)


def test_the_white_patch_method_differs_and_survives() -> None:
    """The alternative is implemented and reachable, not just described.

    White-patch takes the brightest reference pixel as the illuminant, a weaker
    assumption than grey-world in one direction and stronger in another. It has to be
    callable so the sweep can measure it rather than argue about it.

    Asserted as a magnitude rather than with ``allclose`` because the two methods'
    outputs are *similar* by construction -- both are re-scaled to the frame's mean
    brightness, which pulls them back together -- and only the residual difference
    distinguishes them. A tolerance loose enough to describe "different" would also
    swallow the whole gap, which on this fixture is under two percent.
    """
    frame, mask = eye_crop()

    grey = white_balance(frame, mask, method="grey_world")
    patch = white_balance(frame, mask, method="white_patch")

    assert np.isfinite(patch).all()
    assert patch.max() <= 1.0
    assert np.abs(grey - patch).max() > 5e-3, (
        f"the two methods differ by only {np.abs(grey - patch).max():.5f}, which is "
        "not enough to tell a max-based estimate from a mean-based one"
    )


def test_an_unknown_method_is_refused() -> None:
    """A typo must not fall through to grey-world and look like it worked."""
    frame, _ = eye_crop()
    with pytest.raises(ValueError, match="unknown method"):
        white_balance(frame, None, method="gray_world")


def test_a_misshapen_image_is_refused() -> None:
    """This one does raise: a wrong-shaped image is a caller bug, not a data quirk."""
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        white_balance(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        white_balance(np.zeros((10, 10, 4), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# The uint8 bridge
# --------------------------------------------------------------------------- #


def test_a_float_image_is_scaled_up_not_truncated() -> None:
    """The regression: casting a ``[0, 1]`` float to uint8 zeroes it.

    OpenCV's HSV conversion takes uint8 only, and the previous code reached it by
    ``astype(np.uint8)``. Every pixel below 1.0 -- which is all of them -- became 0,
    so the hue and saturation medians came out as zeros. That reads as "this
    conjunctiva is grey", which is a plausible answer rather than an obvious
    failure, which is why it needs a test.
    """
    frame = np.full((10, 10, 3), 0.5, dtype=np.float64)

    as_uint8 = _as_uint8(frame)

    assert as_uint8.dtype == np.uint8
    assert as_uint8.max() == pytest.approx(128, abs=1)


def test_uint8_input_passes_through_untouched() -> None:
    """No rescaling of an array that is already 0-255, in either direction."""
    frame = np.array([[[0, 128, 255]]], dtype=np.uint8)
    assert np.array_equal(_as_uint8(frame), frame)


def test_a_float_image_already_in_0_255_is_not_scaled_again() -> None:
    """The ``> 1.5`` heuristic, which is the same one ``srgb_to_linear`` uses."""
    frame = np.full((4, 4, 3), 200.0, dtype=np.float64)
    assert _as_uint8(frame).max() == 200


def test_the_hsv_medians_of_a_float_image_are_not_all_zero() -> None:
    """The user-visible consequence, asserted where it is observed.

    A saturated red patch in ``[0, 1]`` has a hue near 0 on OpenCV's scale and a
    saturation near 255. Truncated, both read 0 -- indistinguishable from a grey
    pixel, and from a bug.
    """
    patch = np.zeros((8, 8, 3), dtype=np.float64)
    patch[..., 0] = 0.8
    patch[..., 1] = 0.1

    features = extract_colour_features(patch)

    assert features.hsv_sat > 100, "saturation was truncated away"
    assert features.lab_a > 20, "the same image read as neutral"


# --------------------------------------------------------------------------- #
# The balanced extractor
# --------------------------------------------------------------------------- #


def test_the_balanced_extractor_is_the_unbalanced_one_on_the_corrected_image() -> None:
    """The wrapper is thin on purpose, and this is what keeps it thin.

    Balancing is a change to *what is measured*, not a new kind of measurement. If the
    balanced path ever grows its own copy of the ten feature calculations, the two
    would drift and a difference between their results could no longer be attributed
    to the division -- which is the only reason to have both.

    Compared field by field with ``equal_nan`` rather than with ``==`` on the
    dataclass: ``high_hue_ratio`` returns NaN on a near-uniform patch, because its
    guard rejects a channel below 1.0, and ``nan != nan`` would fail the comparison
    for a reason that has nothing to do with the wrapper.
    """
    frame, mask = eye_crop()

    via_wrapper = dataclasses.asdict(extract_colour_features_balanced(frame, mask))
    via_explicit = dataclasses.asdict(extract_colour_features(balanced_rgb(frame, mask), mask))

    for key, value in via_wrapper.items():
        assert np.allclose(value, via_explicit[key], equal_nan=True), f"{key} diverged"


def test_balancing_changes_the_features_on_a_warm_frame() -> None:
    """Otherwise the two paths would be indistinguishable and the sweep pointless."""
    frame, mask = eye_crop()

    raw = extract_colour_features(frame, mask)
    balanced = extract_colour_features_balanced(frame, mask)

    assert raw.lab_a != pytest.approx(balanced.lab_a, abs=0.5), (
        "a* did not move, so the illuminant was not removed"
    )


def test_balancing_accepts_the_alternative_method() -> None:
    """The sweep needs both, so both are reachable through the public entry point."""
    frame, mask = eye_crop()

    features = extract_colour_features_balanced(frame, mask, method="white_patch")

    assert np.isfinite(features.lab_l)
    assert np.isfinite(features.lab_a)
