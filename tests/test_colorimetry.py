"""Colour-science tests, against analytically known values.

Every assertion here is derived from the standard, not from running the code and
recording what it produced. A test that asserts the implementation agrees with
itself is worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemolux.metrics.colorimetry import (
    conjunctival_pigmentation,
    erythema_index,
    extract_colour_features,
    high_hue_ratio,
    ita_angle,
    ita_band,
    mask_mean_lab,
    pigmentation_band,
    redness_ratio,
    rgb_to_lab,
    srgb_to_linear,
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
            np.array([[[200, 30, 40], [180, 60, 90], [210, 20, 30], [150, 70, 110]]], dtype=np.uint8)
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
        assert extract_colour_features(img).lab_l != extract_colour_features(
            img, np.arange(32)[:, None] < 16
        ).lab_l

    def test_richer_photograph_is_redder(self):
        # Sanity on the direction of the signal: a warm patch must score higher
        # on redness than a neutral one of the same lightness.
        red = np.full((32, 32, 3), (200, 60, 55), dtype=np.uint8)
        neutral = np.full((32, 32, 3), (128, 128, 128), dtype=np.uint8)
        assert extract_colour_features(red).lab_a > extract_colour_features(neutral).lab_a
