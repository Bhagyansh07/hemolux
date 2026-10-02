"""Tests for regression and agreement metrics.

Why sklearn is the reference
----------------------------
Every formula here has an independent implementation in ``sklearn.metrics``, so
these tests do not check hemolux against hemolux (E2). Agreement is asserted to
10 decimal places, which is tight enough to catch a wrong divisor and loose
enough not to fail on a float32 round-trip in a third-party build.

Real defect these lock in
--------------------------
``bland_altman()`` could not return at all. ``BlandAltman._diffs`` was declared
as a ``@property`` that raised ``NotImplementedError``, and the factory tried to
populate it with ``object.__setattr__``. A property is a data descriptor, so it
takes precedence over the instance dictionary, and a frozen dataclass has no way
to assign one:

    AttributeError: property '_diffs' of 'BlandAltman' object has no setter

Because ``regression_report()`` calls ``bland_altman()``, the crash took out the
entire reporting path -- the project's headline metrics -- while 159 other tests
passed, because this module had never been executed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hemolux.metrics.regression import (
    BlandAltman,
    bias_ci95,
    bland_altman,
    explained_variance,
    format_mean_std,
    mae,
    pearson_r,
    r2,
    regression_report,
    rmse,
)

# A deliberately imperfect predictor: a small positive bias plus noise, over a
# realistic adult Hb range spanning mild to severe anaemia.
YT = np.array([8.1, 9.4, 10.2, 11.0, 11.8, 12.4, 13.1, 14.0, 14.6, 15.2, 6.9, 10.9])
YP = np.array([8.6, 9.1, 10.5, 10.7, 12.2, 12.1, 13.4, 13.6, 15.0, 14.8, 7.5, 10.6])


# --------------------------------------------------------------------------- #
# Error magnitudes, against sklearn
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("fn", "ref"),
    [
        (mae, mean_absolute_error),
        (rmse, lambda a, b: mean_squared_error(a, b) ** 0.5),
        (r2, r2_score),
    ],
)
def test_error_metrics_match_sklearn(fn, ref) -> None:  # type: ignore[no-untyped-def]
    assert fn(YT, YP) == pytest.approx(ref(YT, YP), abs=1e-10)


def test_rmse_never_below_mae_for_the_same_errors() -> None:
    """A definitional consequence, asserted because it is easy to invert."""
    rng = np.random.default_rng(7)
    a = rng.normal(12.0, 2.0, 300)
    b = a + rng.normal(0.0, 1.4, 300)
    assert rmse(a, b) >= mae(a, b) - 1e-12


def test_mae_and_rmse_are_zero_for_a_perfect_predictor() -> None:
    assert mae(YT, YT) == 0.0
    assert rmse(YT, YT) == 0.0
    assert r2(YT, YT) == pytest.approx(1.0)


def test_mae_is_invariant_to_shifting_both_series() -> None:
    """Shifting truth and prediction together leaves every difference untouched.

    An earlier draft of this test asserted ``mae(y, p + 3) == mae(y, p) + 3``.
    That is false, and not by a rounding margin: adding 3 to the *predictions*
    only adds 3 where the error was already positive and subtracts 3 where it was
    negative. On this fixture MAE goes 0.375 -> 3.042, not -> 3.375, because the
    signed errors straddle zero. Asserting the false version would have pinned a
    bug rather than a property.
    """
    assert mae(YT + 3.0, YP + 3.0) == pytest.approx(mae(YT, YP), abs=1e-12)


def test_mae_does_not_track_a_prediction_offset_uniformly() -> None:
    """Pins the asymmetry the previous test walked into.

    Because the errors here straddle zero, a prediction shift moves MAE by less
    than its own magnitude. A "MAE grows by exactly the shift" test suite would
    have caught a units error; this one catches the opposite mistake.
    """
    base = mae(YT, YP)
    shifted = mae(YT, YP + 3.0)
    assert 0.0 < shifted - base < 3.0


# --------------------------------------------------------------------------- #
# Degenerate inputs: NaN rather than a fabricated number
# --------------------------------------------------------------------------- #


def test_r2_is_nan_when_truth_is_constant() -> None:
    """R^2 is undefined against a zero-variance reference. 0.0 would be a lie."""
    assert math.isnan(r2(np.full(10, 12.0), np.linspace(10, 14, 10)))


def test_evs_is_nan_when_truth_is_constant() -> None:
    assert math.isnan(explained_variance(np.full(10, 12.0), np.linspace(10, 14, 10)))


def test_pearson_is_nan_for_a_constant_series() -> None:
    assert math.isnan(pearson_r(np.full(10, 12.0), np.linspace(10, 14, 10)))
    assert math.isnan(pearson_r(np.linspace(10, 14, 10), np.full(10, 12.0)))


def test_non_finite_pairs_are_dropped_and_counted_via_n() -> None:
    """NaN predictions must not be silently averaged away."""
    yt = np.array([10.0, 11.0, 12.0, 13.0])
    yp = np.array([10.5, np.nan, 11.5, 12.5])
    ba = bland_altman(yt, yp)
    assert ba.n == 3, "the NaN pair must be dropped, not counted"
    assert math.isfinite(ba.bias)


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        mae(np.zeros(5), np.zeros(4))


def test_too_few_pairs_raise() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        mae(np.array([1.0]), np.array([1.0]))


# --------------------------------------------------------------------------- #
# Bland-Altman
# --------------------------------------------------------------------------- #


def test_bland_altman_returns_instead_of_raising() -> None:
    """The regression. If this raises, the whole reporting path is dead."""
    ba = bland_altman(YT, YP)
    assert isinstance(ba, BlandAltman)


@pytest.mark.parametrize(
    ("field", "manual"),
    [
        ("bias", lambda: float(np.mean(YP - YT))),
        ("sd_diff", lambda: float(np.std(YP - YT, ddof=1))),
        ("loa_lower", lambda: float(np.mean(YP - YT) - 1.96 * np.std(YP - YT, ddof=1))),
        ("loa_upper", lambda: float(np.mean(YP - YT) + 1.96 * np.std(YP - YT, ddof=1))),
        ("n", lambda: 12),
    ],
)
def test_bland_altman_fields_match_hand_arithmetic(field, manual) -> None:  # type: ignore[no-untyped-def]
    """Checked by hand, not against this module's own output."""
    ba = bland_altman(YT, YP)
    assert getattr(ba, field) == pytest.approx(manual(), abs=1e-10)


def test_limits_of_agreement_bracket_the_bias() -> None:
    ba = bland_altman(YT, YP)
    assert ba.loa_lower < ba.bias < ba.loa_upper
    # Symmetric by construction, so the mid-point must be the bias itself.
    assert (ba.loa_lower + ba.loa_upper) / 2 == pytest.approx(ba.bias, abs=1e-12)


def test_sd_uses_the_sample_denominator() -> None:
    """ddof=1, not 0. A population SD would shrink the limits of agreement by
    sqrt(n/(n-1)) and understate the spread, which is the conservative
    direction this project cannot take."""
    ba = bland_altman(YT, YP)
    assert ba.sd_diff == pytest.approx(float(np.std(YP - YT, ddof=1)), abs=1e-12)
    assert ba.sd_diff != pytest.approx(float(np.std(YP - YT, ddof=0)), abs=1e-6)


def test_covers_matches_a_direct_count() -> None:
    ba = bland_altman(YT, YP)
    assert ba.covers(1.0) == pytest.approx(float(np.mean(np.abs(YP - YT) <= 1.0)))
    assert ba.covers(2.0) == pytest.approx(float(np.mean(np.abs(YP - YT) <= 2.0)))


def test_covers_is_monotone_in_the_margin() -> None:
    ba = bland_altman(YT, YP)
    margins = [0.0, 0.5, 1.0, 2.0, 5.0]
    covers = [ba.covers(m) for m in margins]
    assert covers == sorted(covers), f"wider margin must never cover fewer: {covers}"
    assert ba.covers(0.0) < ba.covers(5.0)


def test_covers_is_nan_without_differences() -> None:
    """A fraction that was never computed must not read as 0.0 or 1.0."""
    empty = BlandAltman(bias=0.0, sd_diff=0.0, loa_lower=0.0, loa_upper=0.0,
                        proportional_slope=float("nan"), n=0)
    assert math.isnan(empty.covers(1.0))


def test_perfect_agreement_has_zero_width_limits() -> None:
    ba = bland_altman(YT, YT)
    assert ba.bias == 0.0
    assert ba.sd_diff == 0.0
    assert ba.loa_lower == ba.loa_upper == 0.0
    assert ba.covers(0.0) == 1.0


@pytest.mark.parametrize("a", [0.8, 1.2, 1.6, 2.0])
def test_proportional_bias_slope_uses_the_pair_mean(a: float) -> None:
    """The Bland-Altman slope is regressed on the *mean* of each pair, and the
    correction matters.

    For a linear predictor ``pred = a*truth + b``:

        d = pred - truth = (a-1)*truth + b
        m = (pred + truth)/2 = (a+1)*truth/2 + b/2

    so ``d`` as a function of ``m`` has slope ``2(a-1)/(a+1)``, not ``a-1``. An
    earlier draft of this test expected ``a-1`` and failed at 0.6 vs 0.4615 --
    the code was right and the expectation was naive.

    Asserting the corrected value is what actually locks the behaviour: an
    implementation that regressed the difference against the truth would return
    ``a-1`` here and fail, which is the mistake this test exists to catch.
    """
    truth = np.linspace(6.0, 16.0, 40)
    pred = a * truth - 0.6 * 11.0
    ba = bland_altman(truth, pred)
    assert ba.proportional_slope == pytest.approx(2 * (a - 1.0) / (a + 1.0), abs=1e-9)
    assert ba.proportional_slope != pytest.approx(a - 1.0, abs=1e-6)


def test_proportional_bias_is_detected_when_present() -> None:
    """Error growing with Hb level is the failure a single global accuracy
    number hides, so the slope must come out clearly non-zero."""
    truth = np.linspace(6.0, 16.0, 40)
    pred = truth + 0.6 * (truth - 11.0)  # systematically worse at high Hb
    ba = bland_altman(truth, pred)
    assert ba.proportional_slope == pytest.approx(0.6 / 1.3, abs=1e-9)


def test_no_proportional_bias_gives_a_near_zero_slope() -> None:
    rng = np.random.default_rng(3)
    truth = rng.uniform(6.0, 16.0, 300)
    pred = truth + rng.normal(0.0, 0.6, 300)
    ba = bland_altman(truth, pred)
    assert abs(ba.proportional_slope) < 0.05


def test_proportional_slope_is_nan_when_the_mean_is_flat() -> None:
    """All pairs at the same average: a slope through one point is undefined."""
    ba = bland_altman(np.full(10, 10.0), np.full(10, 10.5))
    assert math.isnan(ba.proportional_slope)


def test_to_dict_carries_only_scalars() -> None:
    """It is written into a results CSV; an ndarray there would break the load."""
    ba = bland_altman(YT, YP)
    d = ba.to_dict()
    assert set(d) == {"bias", "sd_diff", "loa_lower", "loa_upper", "proportional_slope", "n"}
    assert all(isinstance(v, (int, float)) for v in d.values())


def test_diffs_are_not_in_the_repr_or_equality() -> None:
    ba = bland_altman(YT, YP)
    assert "diffs" not in repr(ba)
    assert ba == bland_altman(YT, YP), "equality must ignore the derived array"


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def test_bias_ci_is_deterministic_for_a_fixed_seed() -> None:
    """Two runs must produce identical output, so a number in the report can
    be reproduced rather than merely believed."""
    a = bias_ci95(YT, YP, seed=11)
    b = bias_ci95(YT, YP, seed=11)
    assert a == b


def test_bias_ci_brackets_the_point_estimate() -> None:
    ba = bland_altman(YT, YP)
    lo, hi = bias_ci95(YT, YP, seed=5, n_boot=500)
    assert lo <= ba.bias <= hi


def test_bias_ci_narrows_with_more_bootstrap_samples() -> None:
    """A wider bootstrap should not produce a wider interval. Catches a
    percentile calculation that ignores the sample count."""
    widths = [bias_ci95(YT, YP, seed=2, n_boot=n)[1] - bias_ci95(YT, YP, seed=2, n_boot=n)[0]
              for n in (100, 2000)]
    assert widths[1] <= widths[0] + 1e-9


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_regression_report_agrees_with_sklearn() -> None:
    """The crash took this out too; it is the function EVALS.md is built from."""
    rep = regression_report(YT, YP)
    assert rep.n == 12
    assert rep.mae == pytest.approx(mean_absolute_error(YT, YP), abs=1e-10)
    assert rep.rmse == pytest.approx(mean_squared_error(YT, YP) ** 0.5, abs=1e-10)
    assert rep.r2 == pytest.approx(r2_score(YT, YP), abs=1e-10)


def test_report_within_matches_the_public_covers_method() -> None:
    """One implementation, so the table and the API cannot disagree."""
    rep = regression_report(YT, YP)
    ba = bland_altman(YT, YP)
    assert rep.within_1 == pytest.approx(ba.covers(1.0))
    assert rep.within_2 == pytest.approx(ba.covers(2.0))


def test_within_1_is_always_at_least_within_2() -> None:
    """A +/-1 g/dL band is contained in a +/-2 g/dL band."""
    rep = regression_report(YT, YP)
    assert rep.within_1 <= rep.within_2


def test_report_row_is_a_markdown_table_row() -> None:
    row = regression_report(YT, YP).to_row()
    assert row.startswith("|") and row.endswith("|")
    assert row.count("|") == 11, "header/divider expect 11 pipes for 10 columns"
    assert "%" in row


def test_report_to_dict_is_all_scalars() -> None:
    d = regression_report(YT, YP).to_dict()
    assert all(isinstance(v, (int, float)) for v in d.values())


# --------------------------------------------------------------------------- #
# Reporting discipline
# --------------------------------------------------------------------------- #


def test_a_single_run_is_labelled_as_such() -> None:
    """A bare number from a single run is the failure mode this guards: it is
    indistinguishable from a real measurement."""
    assert "single run" in format_mean_std([0.87])


def test_mean_std_reports_the_run_count() -> None:
    out = format_mean_std([0.80, 0.90, 1.00])
    assert "n=3" in out
    assert "+/-" in out
    assert "95% CI" in out


def test_mean_std_drops_non_finite_values_and_counts_the_rest() -> None:
    out = format_mean_std([0.8, float("nan"), 1.0, float("inf")])
    assert "n=2" in out


def test_mean_std_with_nothing_usable_says_so() -> None:
    assert format_mean_std([float("nan"), float("inf")]) == "n/a"
    assert format_mean_std([]) == "n/a"
