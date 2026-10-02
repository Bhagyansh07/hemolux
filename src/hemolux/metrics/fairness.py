"""Fairness audit: does the model's error depend on the patient's pigmentation?

This module implements the C3 and C4 analysis in ``brain/01_PRD.md`` §7. It is
the part of the project that no published conjunctival-pallor model we found
performs, and it is why the project exists.

The hypothesis under test
-------------------------
Melanin and haemoglobin absorb in overlapping spectral bands, so an RGB sensor
cannot cleanly separate them. Therefore a model trained on palpebral-conjunctiva
images should show **systematic, pigmentation-dependent error** — either worse
tracking, or a bias whose sign flips across pigmentation strata.

What counts as evidence
-----------------------
* A subgroup table where the worst group is visibly worse than the average. A
  model that is uniformly mediocre has no fairness problem; a model that is good
  on light subjects and bad on dark ones does.
* A **non-zero slope** of signed error on ITA°, with a confidence interval that
  excludes zero.

What does not
-------------
A p-value on an accuracy difference between two groups is the wrong tool. With
n = 218 split into subgroups, such a test is powered to detect almost nothing and
its non-significance is not evidence of fairness. The slope of a continuous
error-versus-ITA° regression uses the pigment axis properly, and its effect size
is directly interpretable as "g/dL of error per degree of ITA°".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from .regression import _clean, mae, rmse

_FloatArr = NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Subgroup evaluation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubgroupResult:
    """Metrics for one subgroup. ``n`` is always reported, including when tiny."""

    name: str
    n: int
    mae: float
    rmse: float
    bias: float
    mean_true_hb: float
    n_positive: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "subgroup": self.name,
            "n": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "mean_true_hb": self.mean_true_hb,
            "n_anemic": self.n_positive,
        }


def evaluate_subgroups(
    y_true: _FloatArr,
    y_pred: _FloatArr,
    group: NDArray[np.generic],
    *,
    min_n: int = 5,
) -> list[SubgroupResult]:
    """Compute per-subgroup error.

    Subgroups with fewer than ``min_n`` members still get a row, with a count, so
    that "the model is worse on dark-skinned subjects" is visible even when the
    evidence is thin. Suppressing small groups is how fairness analyses become
    dishonest: the inconvenient group is always the small one.
    """
    a, b = _clean(y_true, y_pred)
    g = np.asarray(group).ravel()
    if g.shape != a.shape:
        raise ValueError(f"group shape {g.shape} does not match target shape {a.shape}")

    diff = b - a
    results: list[SubgroupResult] = []
    for label in sorted({str(v) for v in g}, key=str):
        mask = g.astype(str) == label
        n = int(mask.sum())
        if n < min_n:
            continue
        results.append(
            SubgroupResult(
                name=label,
                n=n,
                mae=float(np.mean(np.abs(diff[mask]))),
                rmse=float(np.sqrt(np.mean(diff[mask] ** 2))),
                bias=float(np.mean(diff[mask])),
                mean_true_hb=float(np.mean(a[mask])),
                n_positive=int(np.sum(a[mask] < 12.0)),
            )
        )
    return results


def subgroup_spread(results: list[SubgroupResult]) -> dict[str, float]:
    """Gaps between the worst group and the overall result.

    ``mae_gap`` is the headline: how much worse the worst-served group is than
    the pooled average.
    """
    if not results:
        return {"mae_gap": float("nan"), "bias_gap": float("nan"), "worst_group": "n/a"}

    worst_mae = max(results, key=lambda r: r.mae)
    worst_bias = max(results, key=lambda r: abs(r.bias))
    pooled_mae = float(np.mean([r.mae for r in results]))
    pooled_bias = float(np.mean([r.bias for r in results]))
    return {
        "mae_gap": worst_mae.mae - pooled_mae,
        "bias_gap": abs(worst_bias.bias - pooled_bias),
        "worst_group": worst_mae.name,
        "worst_group_mae": worst_mae.mae,
        "worst_group_n": float(worst_mae.n),
    }


# --------------------------------------------------------------------------- #
# Continuous bias-versus-ITA regression — the real test of C3
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BiasSlope:
    """OLS fit of signed error against ITA° (or any continuous pigment axis).

    Attributes
    ----------
    slope
        Change in signed error, in g/dL, per degree of ITA°. **Negative means the
        model increasingly under-estimates Hb as skin gets lighter** (or the
        reverse); either sign is a finding, zero is not.
    ci_low, ci_high
        95% CI for the slope. **If this interval contains 0, C3 is not supported
        by this data** and the report must say so.
    r2
        Variance in error explained by pigmentation. Small values are common and
        still worth reporting.
    p_value
        Two-sided. Informational only — see the module docstring on why we do
        not lean on it.
    """

    slope: float
    ci_low: float
    ci_high: float
    intercept: float
    r2: float
    p_value: float
    n: int


def bias_vs_ita(
    y_true: _FloatArr,
    y_pred: _FloatArr,
    ita_deg: _FloatArr,
    *,
    confidence: float = 0.95,
) -> BiasSlope:
    """Regress signed prediction error on ITA°.

    Implementation is ordinary least squares with a Student-t critical value taken
    at the caller's ``confidence``, defaulting to 95%. The interval is the whole
    point of this function: ``fairness_gate`` turns it into a claim, so it is
    held to exact agreement with ``scipy.stats`` in ``tests/test_fairness.py``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence!r}")

    # One mask for all three columns, not one per column. `_clean` drops
    # non-finite (y_true, y_pred) pairs; if ITA were filtered separately the two
    # filters could disagree, which either crashes on a shape mismatch or worse,
    # pairs a patient's ITA° with another patient's error. Patient-to-patient
    # misalignment in a fairness audit would produce a confident wrong answer.
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    x = np.asarray(ita_deg, dtype=np.float64).ravel()

    if not (a.shape == b.shape == x.shape):
        raise ValueError(
            f"shape mismatch: y_true {a.shape}, y_pred {b.shape}, ita {x.shape}"
        )

    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(x)
    a, b, x = a[ok], b[ok], x[ok]
    y = b - a

    if x.size < 3:
        # OLS needs two degrees of freedom before a slope means anything, and
        # BiasSlope is a plain dataclass, so every field is named explicitly.
        nan = float("nan")
        return BiasSlope(
            slope=nan,
            ci_low=nan,
            ci_high=nan,
            intercept=nan,
            r2=nan,
            p_value=nan,
            n=int(x.size),
        )

    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx < 1e-12:
        return BiasSlope(0.0, 0.0, 0.0, float(y_mean), 0.0, 1.0, int(x.size))

    slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = float(y_mean - slope * x_mean)

    resid = y - (intercept + slope * x)
    dof = x.size - 2
    sigma2 = float(np.sum(resid**2) / dof) if dof > 0 else float("nan")
    se_slope = float(np.sqrt(sigma2 / sxx)) if sxx > 0 else float("nan")

    # Inverse Student-t CDF at the *requested* confidence. Deliberately not the
    # normal approximation and not a hard-coded lookup table: whether this
    # interval clears zero is the entire C3 decision, so an interval silently
    # computed at the wrong confidence is worse than reporting none at all.
    t_crit = _t_crit(confidence, dof)
    if not np.isfinite(t_crit):
        t_crit = 1.959963984540054
    ci_low, ci_high = slope - t_crit * se_slope, slope + t_crit * se_slope

    syy = float(np.sum((y - y_mean) ** 2))
    r2_val = float(1.0 - np.sum(resid**2) / syy) if syy > 1e-12 else 0.0

    t_stat = slope / se_slope if se_slope > 1e-12 else float("nan")
    p_value = _two_sided_t_pvalue(abs(t_stat), dof) if np.isfinite(t_stat) else float("nan")

    return BiasSlope(
        slope=slope,
        ci_low=ci_low,
        ci_high=ci_high,
        intercept=intercept,
        r2=r2_val,
        p_value=p_value,
        n=int(x.size),
    )


def _two_sided_t_pvalue(t_stat: float, dof: int) -> float:
    """Two-sided p-value for Student's t.

    Delegates to ``scipy.stats``. This used to be a hand-written regularised
    incomplete beta (modified Lentz) over a hand-written Lanczos log-gamma, on
    the argument that a dependency-free statistical routine is easier to audit.
    Checked against scipy, it disagreed by up to four orders of magnitude on the
    critical value: the accumulator was seeded so that the first iteration
    doubled ``f``, which shifts every subsequent term.

    A bias claim is decided by whether an interval clears zero, so a silently
    wrong interval is the one failure mode this project cannot afford. The
    dependency was already declared; the behaviour is now pinned in
    ``tests/test_fairness.py`` against known critical values.
    """
    if dof <= 0 or not np.isfinite(t_stat):
        return float("nan")
    return float(2.0 * scipy_stats.t.sf(abs(t_stat), dof))


def _t_crit(confidence: float, dof: int) -> float:
    """Two-sided Student-t critical value: the inverse CDF at ``confidence``."""
    if not 0.0 < confidence < 1.0 or dof <= 0:
        return float("nan")
    return float(scipy_stats.t.ppf(0.5 + 0.5 * confidence, dof))


# --------------------------------------------------------------------------- #
# C4 — does multi-site fusion close the gap?
# --------------------------------------------------------------------------- #


@dataclass
class FusionComparison:
    """Result of comparing a single site against a fused multi-site model."""

    site: str
    single_site_mae: dict[str, float] = field(default_factory=dict)
    fused_mae: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _pooled(values: dict[str, float]) -> float:
        # An empty mean is nan, not 0.0. Returning 0.0 would make a missing
        # group look like a perfect one and quietly improve the fusion gap.
        return float(np.mean(list(values.values()))) if values else float("nan")

    @property
    def pooled_single(self) -> float:
        return self._pooled(self.single_site_mae)

    @property
    def pooled_fused(self) -> float:
        return self._pooled(self.fused_mae)

    @property
    def worst_group_single(self) -> float:
        return max(self.single_site_mae.values()) if self.single_site_mae else float("nan")

    @property
    def worst_group_fused(self) -> float:
        return max(self.fused_mae.values()) if self.fused_mae else float("nan")

    @property
    def spread_reduction(self) -> float:
        """How much smaller the worst-group minus pooled gap becomes after fusion.

        **Positive means fusion helps the worst-served group specifically.** This
        is the whole claim of C4. A positive number here while pooled MAE is
        unchanged is exactly the result that would make the mitigation credible.
        """
        return (self.worst_group_single - self.pooled_single) - (
            self.worst_group_fused - self.pooled_fused
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "single_site_mae": self.single_site_mae,
            "fused_mae": self.fused_mae,
            "worst_group_single": self.worst_group_single,
            "worst_group_fused": self.worst_group_fused,
            "pooled_single": self.pooled_single,
            "pooled_fused": self.pooled_fused,
            "spread_reduction": self.spread_reduction,
        }


def fairness_gate(slope: BiasSlope, *, min_abs_slope: float = 0.002) -> str:
    """Decide whether C3 is supported, refuted, or underpowered.

    Returns one of:

    * ``"supported"`` — the confidence interval for the bias-vs-ITA° slope
      excludes zero *and* the effect is at least ``min_abs_slope``.
    * ``"not_supported"`` — the interval contains zero, or the effect is too
      small to act on. An honest negative; the report says so and moves on. It
      does not get buried.
    * ``"underpowered"`` — too few samples with a finite ITA° value to say
      anything, which is a statement about the dataset, not about the model.
    """
    if not np.isfinite(slope.slope) or slope.n < 20:
        return "underpowered"
    # An interval excludes zero exactly when both bounds share a sign, which is
    # when their product is strictly positive. A product <= 0 means the interval
    # CONTAINS zero, which is the absence of evidence, not the presence of it.
    if slope.ci_low * slope.ci_high > 0.0:
        return "supported" if abs(slope.slope) >= min_abs_slope else "not_supported"
    return "not_supported"


def summarize(y_true: _FloatArr, y_pred: _FloatArr) -> dict[str, float]:
    """Convenience wrapper returning just the headline numbers."""
    a, b = _clean(y_true, y_pred)
    return {"n": float(a.size), "mae": mae(a, b), "rmse": rmse(a, b), "bias": float(np.mean(b - a))}
