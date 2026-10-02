"""Regression and agreement metrics for haemoglobin estimation.

Haemoglobin error is not normally distributed and is heteroscedastic: the error
at 6 g/dL is not the same size as the error at 14 g/dL. So this module reports
three families:

* **error magnitudes** — MAE, RMSE, which have units of g/dL and therefore have
  a clinical meaning;
* **variance explained** — R², EVS, Pearson r, which have none on their own and
  are always reported alongside an error magnitude;
* **agreement** — Bland–Altman bias and limits of agreement, which is the
  standard the method-comparison literature requires.

Reporting MAE alone is the most common way this kind of paper overstates its own
result. No bare number is ever reported: every metric leaves here carrying a
spread and, where the method requires one, a confidence interval.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray

_FloatArr = NDArray[np.float64]


def _clean(y_true: _FloatArr, y_pred: _FloatArr) -> tuple[_FloatArr, _FloatArr]:
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: y_true {a.shape} vs y_pred {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.all():
        # Silent NaN-dropping would let a broken prediction pass unnoticed, so
        # the count is surfaced rather than swallowed.
        a, b = a[finite], b[finite]
    if a.size < 2:
        raise ValueError("need at least 2 finite pairs")
    return a, b


def mae(y_true: _FloatArr, y_pred: _FloatArr) -> float:
    """Mean absolute error, in g/dL."""
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true: _FloatArr, y_pred: _FloatArr) -> float:
    """Root mean squared error, in g/dL. Penalises large misses harder than MAE."""
    a, b = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(y_true: _FloatArr, y_pred: _FloatArr) -> float:
    """Coefficient of determination against the mean of ``y_true``.

    Returns NaN when ``y_true`` is constant: R² is undefined there, and 0.0 would
    be a lie.
    """
    a, b = _clean(y_true, y_pred)
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((a - b) ** 2) / ss_tot)


def explained_variance(y_true: _FloatArr, y_pred: _FloatArr) -> float:
    """Explained variance score. Less pessimistic than R² for biased predictors."""
    a, b = _clean(y_true, y_pred)
    var_y = float(np.var(a))
    if var_y < 1e-12:
        return float("nan")
    return float(1.0 - np.var(a - b) / var_y)


def pearson_r(y_true: _FloatArr, y_pred: _FloatArr) -> float:
    """Pearson correlation. Reports ranking ability only, ignoring bias."""
    a, b = _clean(y_true, y_pred)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class BlandAltman:
    """Bland–Altman agreement analysis.

    Attributes
    ----------
    bias
        Mean of ``y_pred - y_true``. Systematic over/under-estimation.
    sd_diff
        Standard deviation of the differences.
    loa_lower, loa_upper
        The 95% limits of agreement, ``bias +/- 1.96 * sd_diff``.
    proportional_slope
        Slope of ``(mean, difference)``. Non-zero means the error depends on the
        measured value, i.e. the model behaves differently at different Hb
        levels — which is what invalidates a single global "accuracy" claim.
    n
        Number of pairs.
    """

    bias: float
    sd_diff: float
    loa_lower: float
    loa_upper: float
    proportional_slope: float
    n: int
    #: Per-pair differences (``y_pred - y_true``). Kept because ``covers`` needs
    #: them and because a limits-of-agreement result that cannot be checked
    #: against its own differences is not auditable.
    #:
    #: Excluded from ``repr``, ``eq`` and :meth:`to_dict`: it is derived from the
    #: inputs, not part of the result, and ``asdict`` would deepcopy the whole
    #: array into every serialised row.
    diffs: _FloatArr = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        repr=False,
        compare=False,
    )

    def covers(self, margin: float) -> float:
        """Fraction of predictions within ``+/- margin`` g/dL of the truth.

        Returns NaN if the differences were not supplied, rather than reporting
        0.0 or 1.0 for a fraction that was never computed.
        """
        if self.diffs.size == 0:
            return float("nan")
        return float(np.mean(np.abs(self.diffs) <= margin))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "bias": self.bias,
            "sd_diff": self.sd_diff,
            "loa_lower": self.loa_lower,
            "loa_upper": self.loa_upper,
            "proportional_slope": self.proportional_slope,
            "n": self.n,
        }


def bland_altman(y_true: _FloatArr, y_pred: _FloatArr) -> BlandAltman:
    """Bland–Altman analysis of agreement.

    Two models can have identical R² and disagree clinically. This is the check
    that catches it, and for point-of-care Hb devices the literature reference
    band is roughly +/- 1.0 g/dL.

    The returned ``covers(1.0)`` is therefore the single most defensible number
    in the whole project.

    Note
    ----
    An earlier version declared the differences as a ``@property`` that raised
    ``NotImplementedError``, then tried to populate it with
    ``object.__setattr__``. A property is a data descriptor, so it takes
    precedence over the instance dictionary, and a frozen dataclass has no way
    to assign one at all -- ``bland_altman`` raised
    ``AttributeError: property '_diffs' has no setter`` before returning
    anything. Because ``regression_report`` calls this function, the crash took
    the whole reporting path down with it, and 159 passing tests had not caught
    it: this module had never been executed. ``diffs`` is now a real field.
    """
    a, b = _clean(y_true, y_pred)
    diff = b - a
    mean = (a + b) / 2.0

    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0

    if diff.size > 2 and np.ptp(mean) > 1e-9:
        slope = float(np.polyfit(mean, diff, 1)[0])
    else:
        slope = float("nan")

    return BlandAltman(
        bias=bias,
        sd_diff=sd,
        loa_lower=bias - 1.96 * sd,
        loa_upper=bias + 1.96 * sd,
        proportional_slope=slope,
        n=int(diff.size),
        diffs=diff,
    )


def bias_ci95(
    y_true: _FloatArr, y_pred: _FloatArr, *, seed: int = 42, n_boot: int = 2000
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the Bland–Altman bias.

    Seeded, so two runs of the same code produce byte-identical output and a
    number in the report can be reproduced rather than merely believed.
    """
    a, b = _clean(y_true, y_pred)
    rng = np.random.default_rng(seed)
    n = a.size
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = (b[idx] - a[idx]).mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass(frozen=True)
class RegressionReport:
    """A complete, reportable regression result."""

    n: int
    mae: float
    rmse: float
    r2: float
    evs: float
    pearson_r: float
    bias: float
    loa_lower: float
    loa_upper: float
    within_1: float
    within_2: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_row(self) -> str:
        """One-line Markdown table row. Keeps EVALS.md hand-typo-free."""
        return (
            f"| {self.n} | {self.mae:.3f} | {self.rmse:.3f} | {self.r2:.3f} | "
            f"{self.evs:.3f} | {self.pearson_r:.3f} | {self.bias:+.3f} | "
            f"[{self.loa_lower:+.2f}, {self.loa_upper:+.2f}] | "
            f"{100 * self.within_1:.1f}% | {100 * self.within_2:.1f}% |"
        )


def regression_report(y_true: _FloatArr, y_pred: _FloatArr) -> RegressionReport:
    """Compute every regression metric at once, so a report is never partial."""
    a, b = _clean(y_true, y_pred)
    ba = bland_altman(a, b)
    return RegressionReport(
        n=int(a.size),
        mae=mae(a, b),
        rmse=rmse(a, b),
        r2=r2(a, b),
        evs=explained_variance(a, b),
        pearson_r=pearson_r(a, b),
        bias=ba.bias,
        loa_lower=ba.loa_lower,
        loa_upper=ba.loa_upper,
        # Via covers(), not a second inline computation, so the report and the
        # public method cannot disagree about the same number.
        within_1=ba.covers(1.0),
        within_2=ba.covers(2.0),
    )


def format_mean_std(values: list[float], digits: int = 3) -> str:
    """``mean +/- std`` across folds or seeds. Never a bare number.

    A single sample is labelled as such rather than presented as a result.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return "n/a"
    if arr.size == 1:
        return f"{arr[0]:.{digits}f} (single run)"
    std = float(arr.std(ddof=1))
    half = 1.96 * std / math.sqrt(arr.size)
    return f"{arr.mean():.{digits}f} +/- {std:.{digits}f} (95% CI of mean +/- {half:.{digits}f}, n={arr.size})"
