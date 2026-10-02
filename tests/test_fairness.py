"""Fairness-audit tests, against published statistical tables.

Every expected number here comes from a standard t-distribution table or from
algebra, not from running the code and recording its output. A test that only
asserts the implementation agrees with itself is worthless.

These tests exist because two defects in this module were found by inspection
and one by comparison rather than by any test:

* ``confidence`` was accepted and then ignored, so every interval was 95%.
* the ``n < 3`` path multiplied a dataclass by a tuple and raised ``TypeError``
  instead of returning NaN, on exactly the underpowered subgroups an audit
  spends its time in.
* the hand-rolled incomplete beta disagreed with ``scipy`` by up to four orders
  of magnitude on the critical value.

If the t critical value is ever wrong again, ``TestTCritMatchesPublishedTable``
fails without needing scipy to say so.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hemolux.metrics.fairness import (
    BiasSlope,
    FusionComparison,
    _t_crit,
    _two_sided_t_pvalue,
    bias_vs_ita,
    fairness_gate,
)

# --------------------------------------------------------------------------- #
# The t critical value, from the printed two-sided t table.
# (alpha, df) -> t. These are textbook values and are the contract the C3 claim
# rests on: if the interval moves, the claim moves with it.
# --------------------------------------------------------------------------- #

T_TABLE_95 = [
    (1, 12.706),
    (2, 4.303),
    (3, 3.182),
    (4, 2.776),
    (5, 2.571),
    (6, 2.447),
    (8, 2.306),
    (10, 2.228),
    (12, 2.179),
    (15, 2.131),
    (20, 2.086),
    (25, 2.060),
    (30, 2.042),
    (40, 2.021),
    (60, 2.000),
    (120, 1.980),
]

T_TABLE_99 = [
    (1, 63.657),
    (2, 9.925),
    (3, 5.841),
    (4, 4.604),
    (5, 4.032),
    (6, 3.707),
    (8, 3.355),
    (10, 3.169),
    (12, 3.055),
    (15, 2.947),
    (20, 2.845),
    (25, 2.787),
    (30, 2.750),
    (40, 2.704),
    (60, 2.660),
    (120, 2.617),
]

T_TABLE_90 = [(1, 6.314), (5, 2.015), (10, 1.812), (20, 1.725), (30, 1.697), (60, 1.671)]

T_TABLE_80 = [(10, 1.372), (30, 1.310), (60, 1.296)]


class TestTCritMatchesPublishedTable:
    @pytest.mark.parametrize(("dof", "expected"), T_TABLE_95)
    def test_95_percent(self, dof, expected):
        # Printed tables round to 3 decimals; agreement must be at least that good.
        assert _t_crit(0.95, dof) == pytest.approx(expected, abs=5e-4)

    @pytest.mark.parametrize(("dof", "expected"), T_TABLE_99)
    def test_99_percent(self, dof, expected):
        assert _t_crit(0.99, dof) == pytest.approx(expected, abs=5e-4)

    @pytest.mark.parametrize(("dof", "expected"), T_TABLE_90)
    def test_90_percent(self, dof, expected):
        assert _t_crit(0.90, dof) == pytest.approx(expected, abs=5e-4)

    @pytest.mark.parametrize(("dof", "expected"), T_TABLE_80)
    def test_80_percent(self, dof, expected):
        assert _t_crit(0.80, dof) == pytest.approx(expected, abs=5e-4)

    @pytest.mark.parametrize("confidence", [0.80, 0.90, 0.95, 0.99, 0.999])
    def test_strictly_increasing_in_confidence(self, confidence):
        # A wider confidence must never produce a narrower critical value.
        lo, hi = _t_crit(confidence - 0.01, 30), _t_crit(confidence, 30)
        assert hi > lo

    def test_approaches_normal_limit_as_dof_grows(self):
        # t -> z: 1.959964 at alpha=0.05.
        assert _t_crit(0.95, 10_000_000) == pytest.approx(1.959964, abs=1e-4)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.4])
    def test_rejects_confidence_outside_open_unit_interval(self, bad):
        assert math.isnan(_t_crit(bad, 30))

    def test_rejects_nonpositive_dof(self):
        assert math.isnan(_t_crit(0.95, 0))
        assert math.isnan(_t_crit(0.95, -3))


class TestTwoSidedTPvalue:
    def test_zero_statistic_gives_unity(self):
        assert _two_sided_t_pvalue(0.0, 30) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("dof", [3, 30, 198])
    def test_symmetric_in_sign(self, dof):
        assert _two_sided_t_pvalue(2.1, dof) == pytest.approx(
            _two_sided_t_pvalue(-2.1, dof), abs=1e-12
        )

    @pytest.mark.parametrize("dof", [3, 30, 198])
    def test_decreasing_in_magnitude(self, dof):
        vals = [_two_sided_t_pvalue(t, dof) for t in (0.5, 1.0, 1.96, 2.5, 4.0)]
        assert all(np.diff(vals) < 0)

    def test_matches_table_crosscheck(self):
        # t=2.042 at df=30 is the 95% critical value, so its two-sided
        # p-value must be 0.05 to three decimals.
        assert _two_sided_t_pvalue(2.042, 30) == pytest.approx(0.05, abs=1e-3)

    @pytest.mark.parametrize("dof", [0, -1])
    def test_nonpositive_dof_is_nan(self, dof):
        assert math.isnan(_two_sided_t_pvalue(1.0, dof))

    def test_nonfinite_statistic_is_nan(self):
        assert math.isnan(_two_sided_t_pvalue(float("nan"), 30))
        assert math.isnan(_two_sided_t_pvalue(float("inf"), 30))


class TestBiasVsItaRecoversKnownSlope:
    def test_recovers_slope_exactly_on_noiseless_data(self):
        # Constructed so the signed error is exactly 0.02 * ITA + 0.5. OLS on
        # noiseless data must return slope 0.02 and intercept 0.5 exactly.
        ita = np.linspace(-60.0, 70.0, 200)
        true = np.full(200, 13.0)
        pred = true + (0.02 * ita + 0.5)
        got = bias_vs_ita(true, pred, ita)
        assert got.slope == pytest.approx(0.02, abs=1e-10)
        assert got.intercept == pytest.approx(0.5, abs=1e-8)
        assert got.n == 200

    def test_sign_convention_underprediction_is_negative_slope(self):
        # The regression is on (predicted - true), so a model that reads low at
        # high ITA must produce a negative slope. This is the sign the report
        # has to state correctly, so it is pinned rather than assumed.
        ita = np.linspace(0.0, 80.0, 150)
        true = np.full(150, 12.0)
        pred = true - 0.03 * ita
        assert bias_vs_ita(true, pred, ita).slope == pytest.approx(-0.03, abs=1e-10)

    def test_constant_error_has_zero_slope(self):
        ita = np.linspace(-30.0, 60.0, 120)
        true = np.full(120, 11.0)
        pred = true + 0.7
        got = bias_vs_ita(true, pred, ita)
        assert got.slope == pytest.approx(0.0, abs=1e-12)
        assert got.intercept == pytest.approx(0.7, abs=1e-10)

    def test_zero_variance_ita_falls_back_to_mean_error(self):
        # ITA is constant, so slope is undefined and must be reported as 0 with
        # the intercept carrying the mean signed error.
        got = bias_vs_ita(
            np.array([12.0, 13.0, 14.0]),
            np.array([12.1, 13.1, 14.1]),
            np.array([40.0, 40.0, 40.0]),
        )
        assert got.slope == 0.0
        assert got.intercept == pytest.approx(0.1, abs=1e-9)
        assert got.n == 3

    def test_ignores_nonfinite_pairs(self):
        ita = np.array([10.0, 20.0, np.nan, 40.0, 50.0, 60.0])
        true = np.array([12.0, 12.0, 12.0, 12.0, 12.0, 12.0])
        pred = np.array([12.0, 12.0, 12.0, 12.0, 12.0, np.inf])
        got = bias_vs_ita(true, pred, ita)
        assert got.n == 4


class TestBiasVsItaConfidenceIsHonoured:
    @staticmethod
    def _data():
        rng = np.random.default_rng(20261002)
        n = 220
        true = rng.uniform(7.0, 16.5, n)
        pred = true + rng.normal(0.0, 1.1, n)
        ita = rng.uniform(-60.0, 70.0, n)
        return true, pred, ita

    def test_wider_confidence_gives_wider_interval(self):
        # This is the regression test for the ignored-argument defect.
        true, pred, ita = self._data()
        widths = []
        for conf in (0.80, 0.90, 0.95, 0.99):
            got = bias_vs_ita(true, pred, ita, confidence=conf)
            widths.append(got.ci_high - got.ci_low)
        assert all(np.diff(widths) > 0), widths

    def test_interval_contains_the_point_estimate(self):
        true, pred, ita = self._data()
        for conf in (0.80, 0.95, 0.99):
            got = bias_vs_ita(true, pred, ita, confidence=conf)
            assert got.ci_low <= got.slope <= got.ci_high

    def test_default_is_95_percent(self):
        true, pred, ita = self._data()
        assert bias_vs_ita(true, pred, ita) == bias_vs_ita(true, pred, ita, confidence=0.95)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.2, 100.0])
    def test_raises_on_impossible_confidence(self, bad):
        true, pred, ita = self._data()
        with pytest.raises(ValueError, match="confidence"):
            bias_vs_ita(true, pred, ita, confidence=bad)

    def test_raises_when_ita_length_mismatches(self):
        true = np.full(10, 12.0)
        with pytest.raises(ValueError, match="shape mismatch"):
            bias_vs_ita(true, true, np.zeros(9))

    def test_raises_when_y_true_and_y_pred_differ_in_length(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            bias_vs_ita(np.full(10, 12.0), np.full(8, 12.0), np.zeros(10))

    def test_nonfinite_ita_does_not_realign_other_patients(self):
        # If ITA were filtered independently of the Hb columns, this would
        # silently pair each surviving ITA with the wrong patient's error.
        ita = np.array([10.0, 20.0, np.nan, 40.0, 50.0, 60.0])
        true = np.array([12.0, 12.0, 12.0, 12.0, 12.0, 12.0])
        pred = np.array([12.0, 12.0, 12.0, 12.0, 12.0, np.inf])
        got = bias_vs_ita(true, pred, ita)
        assert got.n == 4
        # Survivors are ita 10/20/40/50 with zero error, so the slope is exactly 0.
        assert got.slope == pytest.approx(0.0, abs=1e-12)


class TestBiasVsItaSmallSampleDoesNotCrash:
    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_returns_nan_rather_than_raising(self, n):
        # Regression test for the TypeError on the n<3 path. A fairness audit
        # meets underpowered strata constantly; they must be reportable.
        true = np.linspace(10.0, 14.0, n)
        pred = true + 0.1
        ita = np.linspace(10.0, 60.0, n)
        got = bias_vs_ita(true, pred, ita)
        assert isinstance(got, BiasSlope)
        assert got.n == n

    def test_n_below_three_is_all_nan(self):
        got = bias_vs_ita(
            np.array([12.0, 13.0]),
            np.array([12.2, 13.1]),
            np.array([30.0, 45.0]),
        )
        assert got.n == 2
        for value in (got.slope, got.ci_low, got.ci_high, got.intercept, got.r2, got.p_value):
            assert math.isnan(value)


class TestFairnessGate:
    @staticmethod
    def _slope(**kw) -> BiasSlope:
        base = {
            "slope": 0.01,
            "ci_low": 0.004,
            "ci_high": 0.016,
            "intercept": 0.0,
            "r2": 0.2,
            "p_value": 0.001,
            "n": 200,
        }
        base.update(kw)
        return BiasSlope(**base)

    def test_supported_when_interval_excludes_zero(self):
        assert fairness_gate(self._slope()) == "supported"

    def test_not_supported_when_interval_spans_zero(self):
        assert fairness_gate(self._slope(ci_low=-0.002, ci_high=0.02)) == "not_supported"

    def test_not_supported_when_point_estimate_is_negligible(self):
        # Interval clears zero but the effect is below the minimum slope, which
        # is a real finding and must not be reported as bias.
        assert fairness_gate(self._slope(slope=0.0005)) == "not_supported"

    def test_underpowered_below_twenty_samples(self):
        assert fairness_gate(self._slope(n=19)) == "underpowered"

    def test_underpowered_when_slope_is_nan(self):
        got = bias_vs_ita(np.array([12.0, 13.0]), np.array([12.0, 13.0]), np.array([1.0, 2.0]))
        assert fairness_gate(got) == "underpowered"

    def test_negative_slope_is_still_a_supported_finding(self):
        assert fairness_gate(self._slope(slope=-0.01, ci_low=-0.02, ci_high=-0.004)) == "supported"


class TestFusionComparison:
    def test_spread_reduction_positive_when_fusion_helps_worst_group(self):
        # Single-site model: Italy is much worse than the India/Italy average.
        # Fused model: the two sites converge. C4 is the claim that fusion helps
        # the worst-served group specifically, so this must be positive.
        cmp = FusionComparison(
            site="IT",
            single_site_mae={"IN": 0.6, "IT": 2.4},
            fused_mae={"IN": 0.8, "IT": 1.1},
        )
        # single: worst 2.4 - pooled 1.5 = 0.9 ; fused: worst 1.1 - pooled 0.95 = 0.15
        assert cmp.worst_group_single == pytest.approx(2.4)
        assert cmp.pooled_single == pytest.approx(1.5)
        assert cmp.spread_reduction == pytest.approx(0.75)

    def test_spread_reduction_zero_when_nothing_changes(self):
        mae = {"IN": 0.7, "IT": 1.9}
        cmp = FusionComparison(site="IN", single_site_mae=dict(mae), fused_mae=dict(mae))
        assert cmp.spread_reduction == pytest.approx(0.0)

    def test_spread_reduction_negative_when_fusion_hurts_worst_group(self):
        cmp = FusionComparison(
            site="IT",
            single_site_mae={"IN": 1.0, "IT": 1.4},
            fused_mae={"IN": 1.0, "IT": 3.0},
        )
        assert cmp.spread_reduction < 0

    def test_empty_group_maps_are_nan_not_zero(self):
        # An empty dict averaging to 0.0 would understate the gap and quietly
        # flatter fusion; nan propagates the absence instead.
        cmp = FusionComparison(site="IT")
        assert math.isnan(cmp.pooled_single)
        assert math.isnan(cmp.worst_group_single)

    def test_to_dict_reports_every_headline(self):
        keys = FusionComparison(
            site="IT", single_site_mae={"IN": 0.6}, fused_mae={"IN": 0.7}
        ).to_dict()
        assert set(keys) == {
            "site",
            "single_site_mae",
            "fused_mae",
            "worst_group_single",
            "worst_group_fused",
            "pooled_single",
            "pooled_fused",
            "spread_reduction",
        }
