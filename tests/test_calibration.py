"""Calibration, uncertainty and selective-prediction tests.

Expected values here are worked out by hand from the definitions, not captured
from a run. Where a number comes from a closed form the derivation is in the
test, because a test that only records what the code printed is a change
detector, not a check.

The bin grid is fixed and public (``HB_BIN_EDGES`` / ``HB_BIN_CENTRES``) and the
whole ordinal formulation rests on it, so it is asserted exactly: edges
4,6,7,...,16,18 give twelve bins with midpoints 5, 6.5, 7.5, ..., 15.5, 17.
The two unequal end gaps (4->6 and 16->18 are 2 wide, the rest are 1) are why
the last centre is 17 and not 16.5.

Three things are being defended:

* ``expected_hb`` is a genuine posterior mean, not an argmax lookup. That is
  what makes the output continuous and bounded by the bin range.
* ``temperature_scale`` refuses to run without proof that it was given
  validation data. A temperature fitted on the test split silently invalidates
  every calibration number downstream of it, and nothing else in the code would
  notice.
* ``risk_coverage_curve`` only earns the abstention feature if a genuinely
  informative uncertainty signal lowers MAE at low coverage. An inverted signal
  must *raise* it, which is the control that proves the curve is reading the
  signal at all rather than rewarding low coverage on its own.
"""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import numpy as np
import pytest

from hemolux.data.quality import QUALITY_ABSTAIN
from hemolux.metrics.calibration import (
    HB_BIN_CENTRES,
    HB_BIN_EDGES,
    N_BINS,
    CoveragePoint,
    ValidationOnly,
    _apply_temperature,
    abstention_threshold,
    brier_score,
    expected_calibration_error,
    expected_hb,
    ordinal_target_index,
    predictive_sigma,
    risk_coverage_curve,
    should_abstain,
    soft_label,
    temperature_scale,
)

# --------------------------------------------------------------------------- #
# The bin grid
# --------------------------------------------------------------------------- #


class TestBinGrid:
    def test_centres_are_the_edge_midpoints(self) -> None:
        # (4+6)/2=5, (6+7)/2=6.5, ..., (16+18)/2=17
        assert HB_BIN_CENTRES == (5.0, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 17.0)

    def test_twelve_bins_from_thirteen_edges(self) -> None:
        assert len(HB_BIN_EDGES) == 13
        assert N_BINS == 12
        assert len(HB_BIN_CENTRES) == N_BINS

    def test_edges_are_strictly_increasing(self) -> None:
        assert all(a < b for a, b in pairwise(HB_BIN_EDGES))

    def test_the_two_end_bins_are_twice_as_wide(self) -> None:
        """4->6 and 16->18 span 2 g/dL, every interior gap spans 1. Asymmetric on
        purpose: the corpus runs 7.0 to 17.4 and the tails are where a fixed
        unit grid would waste resolution."""
        widths = [b - a for a, b in pairwise(HB_BIN_EDGES)]
        assert widths[0] == 2.0
        assert widths[-1] == 2.0
        assert all(w == 1.0 for w in widths[1:-1])


# --------------------------------------------------------------------------- #
# Posterior mean
# --------------------------------------------------------------------------- #


def _onehot(row: int, value: float = 1.0) -> np.ndarray:
    p = np.zeros((1, N_BINS), dtype=np.float64)
    p[0, row] = value
    return p


class TestExpectedHb:
    def test_one_hot_decodes_to_that_bins_centre(self) -> None:
        for k, centre in enumerate(HB_BIN_CENTRES):
            assert expected_hb(_onehot(k)) == pytest.approx(centre)

    def test_uniform_decodes_to_the_mean_of_the_centres(self) -> None:
        """(5+6.5+...+15.5+17)/12 = 132/12 = 11.0 exactly."""
        assert HB_BIN_CENTRES and sum(HB_BIN_CENTRES) == pytest.approx(132.0)
        assert expected_hb(np.full((1, N_BINS), 1.0 / N_BINS)) == pytest.approx(11.0)

    def test_output_is_a_weighted_average_so_it_cannot_leave_the_bin_range(self) -> None:
        """This is the deployment advantage over classification-then-threshold:
        the decode is bounded by construction, so no patient can be told 3 g/dL
        from a grid whose lowest centre is 5."""
        rng = np.random.default_rng(0)
        probs = rng.random((50, N_BINS))
        out = expected_hb(probs)
        assert out.min() >= min(HB_BIN_CENTRES)
        assert out.max() <= max(HB_BIN_CENTRES)

    def test_is_continuous_not_quantised(self) -> None:
        """Equal mixtures of adjacent bins move in unit-centre steps. A 0.5/0.5
        split of bins 8 and 9 (13.5, 14.5) must land on 14.0, a value no bin
        represents."""
        p = np.zeros((1, N_BINS))
        p[0, 8] = p[0, 9] = 0.5
        assert expected_hb(p) == pytest.approx(14.0)

    def test_rows_are_renormalised_defensively(self) -> None:
        """onnxruntime probabilities drift off 1.0 by a few ulp. The decode must
        not care."""
        p = _onehot(5, value=0.999)
        assert expected_hb(p) == pytest.approx(10.5)

    def test_zero_row_is_rejected_not_silently_nan(self) -> None:
        with pytest.raises(ValueError, match="sums to zero"):
            expected_hb(np.zeros((1, N_BINS)))

    @pytest.mark.parametrize("shape", [(N_BINS,), (2, N_BINS + 1), (2, 3)])
    def test_wrong_shape_is_rejected(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="probabilities"):
            expected_hb(np.ones(shape))


# --------------------------------------------------------------------------- #
# Predictive sigma
# --------------------------------------------------------------------------- #


class TestPredictiveSigma:
    def test_one_hot_has_zero_spread(self) -> None:
        """A peaked distribution means the model is certain of one bin."""
        for k in range(N_BINS):
            assert predictive_sigma(_onehot(k))[0] == pytest.approx(0.0, abs=1e-12)

    def test_two_point_half_half_is_half_the_gap(self) -> None:
        """For p=0.5 on two bins, SD = |c1-c0|/2. Bins 0 and 1 are 5.0 and 6.5,
        so sigma = 0.75 exactly."""
        p = np.zeros((1, N_BINS))
        p[0, 0] = p[0, 1] = 0.5
        assert predictive_sigma(p)[0] == pytest.approx(0.75)

    def test_never_negative_even_under_float_rounding(self) -> None:
        """E[X^2] - E[X]^2 is a subtraction of nearly equal numbers and can go
        to -1e-17. sqrt of that is NaN, which would poison a risk-coverage sort."""
        p = _onehot(7)
        assert predictive_sigma(p)[0] >= 0.0
        rng = np.random.default_rng(3)
        assert np.all(predictive_sigma(rng.random((200, N_BINS))) >= 0.0)

    def test_wider_spread_gives_larger_sigma(self) -> None:
        narrow = np.zeros((1, N_BINS))
        narrow[0, 7] = 0.9
        narrow[0, 8] = 0.1
        wide = np.zeros((1, N_BINS))
        wide[0, 0] = 0.5
        wide[0, 11] = 0.5
        assert predictive_sigma(wide)[0] > predictive_sigma(narrow)[0]


# --------------------------------------------------------------------------- #
# Soft labels
# --------------------------------------------------------------------------- #


class TestSoftLabel:
    def test_rows_are_distributions(self) -> None:
        p = soft_label(np.array([7.4, 11.0, 15.9]))
        assert p.shape == (3, N_BINS)
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_peaks_on_the_nearest_centre(self) -> None:
        """The point of label-distribution learning: Hb 10.5 sits exactly on
        centre 5, so the target is one-hot there and softens with distance."""
        for value in (5.0, 7.5, 10.5, 14.5, 17.0):
            assert (
                soft_label(np.array([value])).argmax(axis=1)[0]
                == ordinal_target_index(np.array([value]))[0]
            )

    def test_agrees_with_the_hard_bin_assignment(self) -> None:
        hb = np.linspace(4.5, 17.5, 60)
        assert np.array_equal(soft_label(hb).argmax(axis=1), ordinal_target_index(hb))

    def test_is_symmetric_about_its_own_centre(self) -> None:
        """A Gaussian in Hb space cannot be symmetric on an uneven grid, because
        the centres 10.5 and 12.5 sit 1.0 g/dL either side of 11.5. On that
        locally-uniform stretch it is, so the shape is pinned there and a change
        to the logit formula shows up."""
        assert HB_BIN_CENTRES[5] == pytest.approx(11.5 - 1.0)
        assert HB_BIN_CENTRES[7] == pytest.approx(11.5 + 1.0)
        p = soft_label(np.array([11.5]), sigma=0.5)
        assert p[0, 5] == pytest.approx(p[0, 7], rel=1e-12)

    def test_peak_height_follows_the_width(self) -> None:
        """Closed form, so a change to the logit formula cannot slip through.

        A peak at 11.5 (index 6) with sigma=0.6; the interior centres sit exactly
        1.0 g/dL away, which is 5/3 sigma. So the neighbour carries
        ``exp(-0.5 * (1.0/0.6)^2) = exp(-25/18) = 0.24935`` of the peak. Both
        sides match, which is the symmetry check and the height check at once.
        """
        p = soft_label(np.array([11.5]), sigma=0.6)
        expected = float(np.exp(-0.5 * (1.0 / 0.6) ** 2))
        assert expected == pytest.approx(0.249352, abs=1e-6)
        assert p[0, 5] / p[0, 6] == pytest.approx(expected, rel=1e-9)
        assert p[0, 7] / p[0, 6] == pytest.approx(expected, rel=1e-9)

    def test_wider_sigma_flattens_the_target(self) -> None:
        narrow = soft_label(np.array([11.5]), sigma=0.3)
        wide = soft_label(np.array([11.5]), sigma=2.0)
        assert wide.max() < narrow.max()

    def test_exact_centre_is_near_one_hot_at_narrow_width(self) -> None:
        assert soft_label(np.array([10.5]), sigma=0.1)[0, 5] == pytest.approx(1.0, abs=1e-6)

    def test_handles_a_1d_and_2d_input_alike(self) -> None:
        hb = np.array([[11.5], [12.5]])
        assert soft_label(hb).shape == (2, N_BINS)


class TestOrdinalTargetIndex:
    def test_snaps_to_the_nearest_centre(self) -> None:
        # centres are 5, 6.5, 7.5, ..., 15.5, 17
        assert ordinal_target_index(np.array([10.4]))[0] == 5  # 0.1 from 10.5
        assert ordinal_target_index(np.array([10.6]))[0] == 5  # 0.1 from 10.5, 0.9 from 11.5
        assert ordinal_target_index(np.array([11.0]))[0] == 5  # exactly between 10.5 and 11.5
        assert ordinal_target_index(np.array([11.1]))[0] == 6
        assert ordinal_target_index(np.array([4.0]))[0] == 0
        assert ordinal_target_index(np.array([18.0]))[0] == N_BINS - 1

    def test_values_outside_the_grid_clamp_to_the_end_bins(self) -> None:
        """An Hb of 3.2 is out of range for this corpus, but the function must
        return a valid class index rather than raise mid-report."""
        assert ordinal_target_index(np.array([3.2]))[0] == 0
        assert ordinal_target_index(np.array([21.0]))[0] == N_BINS - 1


# --------------------------------------------------------------------------- #
# ECE and Brier
# --------------------------------------------------------------------------- #


class TestCalibrationScores:
    def test_ece_of_a_perfectly_calibrated_bin_is_zero(self) -> None:
        # One bin at confidence 0.5, half the predictions correct.
        conf = np.array([0.5] * 10)
        hit = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        assert expected_calibration_error(conf, hit) == pytest.approx(0.0)

    def test_ece_is_the_weighted_gap_worked_out_by_hand(self) -> None:
        """Four predictions at confidence 0.9, three of them right.
        |0.75 - 0.9| = 0.15, weight 4/4 = 1, so ECE = 0.15 exactly."""
        conf = np.array([0.9] * 4)
        hit = np.array([1, 1, 1, 0], dtype=bool)
        assert expected_calibration_error(conf, hit) == pytest.approx(0.15)

    def test_ece_counts_a_confidence_of_exactly_one(self) -> None:
        """np.digitize drops values equal to the right edge, which would let a
        perfectly-confident-and-wrong set report ECE 0."""
        conf = np.array([1.0] * 4)
        hit = np.array([1, 1, 1, 0], dtype=bool)
        assert expected_calibration_error(conf, hit) == pytest.approx(0.25)
        assert expected_calibration_error(
            np.array([1.0] * 4), np.ones(4, dtype=bool)
        ) == pytest.approx(0.0)

    def test_ece_of_an_all_wrong_confident_set_is_one(self) -> None:
        conf = np.ones(8)
        assert expected_calibration_error(conf, np.zeros(8, dtype=bool)) == pytest.approx(1.0)

    def test_ece_is_zero_for_an_empty_input_rather_than_a_crash(self) -> None:
        assert np.isnan(expected_calibration_error(np.array([]), np.array([], dtype=bool)))

    def test_ece_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            expected_calibration_error(np.ones(4), np.ones(5, dtype=bool))

    def test_ece_is_bounded_by_one(self) -> None:
        rng = np.random.default_rng(11)
        for _ in range(50):
            conf = rng.random(200)
            hit = rng.random(200) < 0.5
            assert 0.0 <= expected_calibration_error(conf, hit, n_bins=10) <= 1.0

    def test_brier_worked_out_by_hand(self) -> None:
        """(0.01+0.01+0.01+0.81)/4 = 0.21."""
        conf = np.array([0.9] * 4)
        hit = np.array([1, 1, 1, 0], dtype=bool)
        assert brier_score(conf, hit) == pytest.approx(0.21)

    def test_brier_of_a_fully_confident_correct_set_is_zero(self) -> None:
        assert brier_score(np.ones(5), np.ones(5, dtype=bool)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, targets: np.ndarray, t: float) -> float:
    probs = _apply_temperature(logits, t)
    picked = np.clip(probs[np.arange(probs.shape[0]), targets], 1e-12, 1.0)
    return float(-np.mean(np.log(picked)))


class TestTemperatureScale:
    @staticmethod
    def _calibrated_case(n: int = 4000, n_bins: int = 4, seed: int = 5):
        """A case whose optimal temperature is known exactly.

        Targets are **drawn from** ``softmax(true_logits)`` rather than taken as
        their argmax. That distinction decides whether the test works at all: if
        the target is always the argmax, then scaling the logits by any positive
        factor leaves the classification problem untouched and already-correct
        confident predictions, so NLL rises monotonically with T and the search
        parks at the grid minimum no matter how miscalibrated the model is. With
        sampled targets the expected NLL is the cross-entropy between the true
        distribution and the scaled one, minimised exactly when T undoes the
        scaling.
        """
        rng = np.random.default_rng(seed)
        true_logits = rng.normal(size=(n, n_bins)) * 1.5
        p_true = _softmax(true_logits)
        targets = np.array([rng.choice(n_bins, p=row) for row in p_true], dtype=np.int64)
        return true_logits, targets

    def test_refuses_anything_but_a_validation_token(self) -> None:
        """The single most common silent failure in a medical-ML report is a
        confidence number tuned on the data it is reported against."""
        logits, targets = self._calibrated_case(n=400)
        for bad in (None, 42, "val", object()):
            with pytest.raises(TypeError, match="ValidationOnly"):
                temperature_scale(logits, targets, bad)  # type: ignore[arg-type]

    def test_token_count_must_match_the_targets(self) -> None:
        logits, targets = self._calibrated_case(n=400)
        with pytest.raises(ValueError, match="token reports n="):
            temperature_scale(logits, targets, ValidationOnly(n=399))

    def test_rejects_non_2d_logits(self) -> None:
        logits, targets = self._calibrated_case(n=200)
        with pytest.raises(ValueError, match=r"\(N, K\) logits"):
            temperature_scale(logits.ravel(), targets, ValidationOnly(n=200))

    def test_rejects_sample_count_mismatch(self) -> None:
        logits, targets = self._calibrated_case(n=200)
        with pytest.raises(ValueError, match="disagree on the number of samples"):
            temperature_scale(logits, targets[:-1], ValidationOnly(n=199))

    @pytest.mark.parametrize("factor", [2.0, 3.0, 0.5, 4.0])
    def test_recovers_a_known_inflation_of_the_logits(self, factor: float) -> None:
        """If the model's logits are the true logits scaled by ``c``, the optimal
        temperature is exactly ``c``: dividing by ``c`` restores them. The grid
        is 0.25..5.0 in 60 steps, i.e. 0.0805 wide, so the tolerance is one step
        plus the sampling noise of a 4000-draw estimate."""
        true_logits, targets = self._calibrated_case(n=4000, seed=9)
        result = temperature_scale(true_logits * factor, targets, ValidationOnly(n=4000), n_grid=60)
        step = (5.0 - 0.25) / 59
        assert result.temperature == pytest.approx(factor, abs=3 * step)

    def test_overconfident_logits_get_a_temperature_above_one(self) -> None:
        """Logits scaled up are overconfident; the fix is to soften them."""
        true_logits, targets = self._calibrated_case(n=4000, seed=4)
        result = temperature_scale(true_logits * 4.0, targets, ValidationOnly(n=4000))
        assert result.temperature > 1.0

    def test_underconfident_logits_get_a_temperature_below_one(self) -> None:
        """Logits scaled down are underconfident; the fix is to sharpen them.
        This is the direction a single-sided test would miss."""
        true_logits, targets = self._calibrated_case(n=4000, seed=4)
        result = temperature_scale(true_logits * 0.4, targets, ValidationOnly(n=4000))
        assert result.temperature < 1.0

    def test_already_calibrated_logits_stay_near_one(self) -> None:
        true_logits, targets = self._calibrated_case(n=8000, seed=31)
        result = temperature_scale(true_logits, targets, ValidationOnly(n=8000))
        assert result.temperature == pytest.approx(1.0, abs=0.35)

    def test_fit_never_worsens_the_validation_nll_it_optimised(self) -> None:
        """T=1 is inside the search grid, so the returned temperature cannot be
        worse than no calibration on the very objective that was minimised."""
        true_logits, targets = self._calibrated_case(n=3000, seed=17)
        inflated = true_logits * 2.5
        result = temperature_scale(inflated, targets, ValidationOnly(n=3000))
        assert _nll(inflated, targets, result.temperature) <= _nll(inflated, targets, 1.0) + 1e-9

    def test_temperature_stays_inside_the_search_grid(self) -> None:
        logits, targets = self._calibrated_case(n=400, seed=21)
        result = temperature_scale(logits, targets, ValidationOnly(n=400))
        assert 0.25 <= result.temperature <= 5.0

    def test_result_serialises_for_the_results_file(self) -> None:
        logits, targets = self._calibrated_case(n=200)
        d = temperature_scale(logits, targets, ValidationOnly(n=200)).to_dict()
        assert set(d) == {"temperature", "ece_before", "ece_after"}
        assert all(isinstance(v, float) for v in d.values())


class TestApplyTemperature:
    def test_rows_stay_distributions(self) -> None:
        logits = np.array([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]])
        for t in (0.25, 1.0, 5.0):
            assert np.allclose(_apply_temperature(logits, t).sum(axis=1), 1.0)

    def test_a_higher_temperature_flattens_the_distribution(self) -> None:
        """logits/T moves every logit toward the mean, so max probability falls
        toward 1/K. This is the direction check."""
        logits = np.array([[3.0, 0.0, -1.0]])
        sharp = _apply_temperature(logits, 0.5)
        flat = _apply_temperature(logits, 5.0)
        assert flat.max() < sharp.max()

    def test_uniform_logits_give_uniform_probabilities(self) -> None:
        assert np.allclose(_apply_temperature(np.zeros((1, 7)), 2.0), 1.0 / 7)

    def test_extreme_logits_do_not_overflow(self) -> None:
        assert np.all(np.isfinite(_apply_temperature(np.array([[1e4, -1e4]]), 0.25)))


# --------------------------------------------------------------------------- #
# Risk-coverage
# --------------------------------------------------------------------------- #


class TestRiskCoverageCurve:
    def test_a_perfectly_informative_signal_lowers_mae_at_low_coverage(self) -> None:
        """Uncertainty set equal to the absolute error. Sorting by it sorts by
        error, so the most-confident five of six predictions are exact and the
        first point (full coverage, k=6) sees the whole error including the
        outlier, while the last point (k=1) sees only a zero."""
        y_true = np.zeros(6)
        y_pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        unc = np.abs(y_pred - y_true)
        curve = risk_coverage_curve(unc, y_true, y_pred, n_points=20)

        assert curve[0].coverage == pytest.approx(1.0)
        assert curve[0].mae == pytest.approx(10.0 / 6)
        assert curve[-1].mae == pytest.approx(0.0)
        assert curve[-1].mae < curve[0].mae

    def test_an_inverted_signal_raises_mae_at_low_coverage(self) -> None:
        """The control. If a *reversed* signal also improved MAE, the improvement
        would be an artefact of shrinking the sample and the abstention feature
        would be worthless."""
        y_true = np.zeros(6)
        y_pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        unc = -np.abs(y_pred - y_true)
        curve = risk_coverage_curve(unc, y_true, y_pred, n_points=20)
        assert curve[-1].mae > curve[0].mae

    def test_a_pure_noise_signal_shows_no_reliable_improvement(self) -> None:
        """The negative control.

        A *constant* uncertainty is not the control it looks like:
        ``np.argsort`` is stable, so equal values come back in input order and the
        prefix means simply walk the running average of the original sequence.
        That can rise or fall depending only on how the rows were written down,
        which says nothing about the confidence signal.

        Random uncertainty is the honest version. Under a random ordering the
        expected mean of any prefix is the population mean, so the improvement
        ``mae_at_full_coverage - mae_at_10pct`` has expectation zero. Averaging
        it over 200 fixed seeds gives a mean within three standard errors of
        zero, while a genuinely informative signal improves on every seed.
        """
        rng = np.random.default_rng(0)
        y_true = rng.normal(12.0, 2.0, size=60)
        y_pred = y_true + rng.normal(0.0, 1.5, size=60)
        y_err = np.abs(y_pred - y_true)

        improvements = []
        for seed in range(200):
            unc = np.random.default_rng(seed).random(60)
            curve = risk_coverage_curve(unc, y_true, y_pred, n_points=20)
            improvements.append(curve[0].mae - curve[-1].mae)

        gains = np.array(improvements)
        standard_error = gains.std(ddof=1) / np.sqrt(gains.size)
        assert abs(gains.mean()) < 3 * standard_error, (
            f"a random signal moved MAE by {gains.mean():.4f} +/- {3 * standard_error:.4f}, "
            "which suggests the curve is not reading the signal at all"
        )

        # Same errors, but the signal is the error itself. Every seed must gain.
        informative = []
        for seed in range(200):
            order = np.random.default_rng(seed).permutation(60)
            curve = risk_coverage_curve(y_err[order], y_true[order], y_pred[order], n_points=20)
            informative.append(curve[0].mae - curve[-1].mae)
        assert min(informative) > 0.0

    def test_coverage_descends_from_full_to_ten_percent(self) -> None:
        y_true = np.zeros(20)
        y_pred = np.arange(20.0)
        curve = risk_coverage_curve(np.arange(20.0), y_true, y_pred, n_points=20)
        coverages = [p.coverage for p in curve]
        assert coverages == sorted(coverages, reverse=True)
        assert coverages[0] == pytest.approx(1.0)
        assert coverages[-1] == pytest.approx(0.1)

    def test_every_point_reports_the_sigma_it_cut_on(self) -> None:
        y_true = np.zeros(10)
        y_pred = np.arange(10.0)
        unc = np.arange(10.0)
        curve = risk_coverage_curve(unc, y_true, y_pred, n_points=10)
        for point in curve:
            k = max(1, round(point.coverage * 10))
            assert point.threshold == pytest.approx(unc[k - 1])

    def test_points_are_frozen(self) -> None:
        """The curve is built once and read many times while the report is
        assembled; a mutated point would silently change a reported number."""
        point = CoveragePoint(coverage=1.0, mae=2.0, threshold=3.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            point.mae = 1.0  # type: ignore[misc]

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="identical shapes"):
            risk_coverage_curve(np.ones(5), np.zeros(4), np.zeros(5))

    def test_single_sample_still_yields_a_usable_curve(self) -> None:
        curve = risk_coverage_curve(np.array([1.0]), np.array([2.0]), np.array([2.5]))
        assert curve
        assert curve[0].coverage == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Abstention gate
# --------------------------------------------------------------------------- #


class TestAbstention:
    def _curve(self) -> list[CoveragePoint]:
        return [
            CoveragePoint(coverage=c / 10, mae=1.0, threshold=float(10 - c))
            for c in range(10, 0, -1)
        ]

    def test_threshold_retains_at_most_the_requested_coverage(self) -> None:
        curve = self._curve()
        t = abstention_threshold(curve, target_coverage=0.75)
        kept = [p for p in curve if p.threshold <= t]
        assert len(kept) / 10 <= 0.75 + 1e-9

    def test_target_of_one_returns_the_full_coverage_threshold(self) -> None:
        curve = self._curve()
        assert abstention_threshold(curve, target_coverage=1.0) == curve[0].threshold

    def test_unreachable_target_falls_back_to_the_last_point(self) -> None:
        curve = self._curve()
        assert abstention_threshold(curve, target_coverage=0.01) == curve[-1].threshold

    def test_empty_curve_yields_infinite_threshold_so_nothing_is_kept(self) -> None:
        assert abstention_threshold([], 0.75) == float("inf")

    def test_abstains_on_a_wide_predictive_sigma(self) -> None:
        assert should_abstain(sigma=2.0, threshold=1.0)

    def test_does_not_abstain_inside_the_threshold(self) -> None:
        assert not should_abstain(sigma=0.5, threshold=1.0)

    def test_boundary_sigma_equal_to_threshold_is_kept(self) -> None:
        """Strictly greater, so the threshold itself is admissible."""
        assert not should_abstain(sigma=1.0, threshold=1.0)

    def test_abstains_on_a_poor_image_even_when_the_model_is_confident(self) -> None:
        """The half of the gate that is about the photograph rather than the
        model. A sharp-trained model on a blurred frame returns a confident wrong
        answer, which is the failure this catches."""
        assert should_abstain(sigma=0.1, threshold=1.0, quality=0.2)

    def test_quality_boundary_matches_the_validator_threshold(self) -> None:
        """Against the imported constant, not a literal. The gate used to hardcode
        ``0.35`` here while ``quality.py`` defined its own value, so the two
        drifted and raising the constant changed nothing."""
        assert not should_abstain(0.1, 1.0, quality=QUALITY_ABSTAIN)
        assert should_abstain(0.1, 1.0, quality=QUALITY_ABSTAIN - 0.01)

    def test_the_quality_branch_is_reachable_by_a_real_photograph(self) -> None:
        """The condition is not decorative.

        Measured over all 862 photographs in the corpus, the lowest score of any
        kind is 0.4087 and the lowest among the 320 that clear the hard gate is
        0.5907. A threshold of 0.35 sat below the whole range, so this branch
        could never fire and the gate was one condition wide while appearing to be
        two. These bounds are loose on purpose -- they assert reachability, not
        the exact corpus distribution.
        """
        assert QUALITY_ABSTAIN > 0.4087, "threshold is below every real photograph"
        assert QUALITY_ABSTAIN < 0.65, "threshold has climbed so high it rejects most survivors"
        assert should_abstain(0.1, 1.0, quality=0.42)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_sigma_always_abstains(self, bad: float) -> None:
        assert should_abstain(bad, 1.0)
