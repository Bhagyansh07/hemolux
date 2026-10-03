"""Thresholded screening metrics at the WHO cutoffs, per group.

Why a regression project needs a confusion matrix
-------------------------------------------------
Everything else in this package scores a *number*: MAE, RMSE, R², Bland-Altman.
That is the right way to grade a continuous estimate, and it is not the way a
screening decision is actually made. A screening tool is used at a threshold --
"is this person below 12 g/dL?" -- and two models with identical MAE can differ
enormously in how often they miss the people who need care. Reporting only the
error axis therefore hides the failure that matters most.

The corpus makes this concrete. Cross-site transfer here has R² -4.835 with a
bias of -4.285 g/dL, and it is tempting to file that under "regression". It is
also a calibration failure: at the anaemia cutoff the sensitivity in one site can
sit far from the other while the pooled MAE barely moves. Nobody notices from a
mean. This module is how it stops being invisible.

Three bands, one asymmetry
--------------------------
The WHO cutoffs define three binary questions for a non-pregnant adult:

* **anaemia** -- below 13 g/dL for men, 12 for women. Sex-specific, so the
  threshold is per patient and not a single number.
* **moderate or worse** -- below 9 g/dL.
* **severe** -- below 7 g/dL.

The severe band is the one that must be handled honestly: this corpus contains
**zero** patients below 7 g/dL, so the band has no positives and sensitivity is
undefined rather than 0 or 1. It is reported as ``not_computable``. Folding it
into a pooled average would hide exactly the gap in the data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hemolux.config import WHO_CUTOFF_FEMALE, WHO_CUTOFF_MALE, WHO_MODERATE, WHO_SEVERE

_FloatArr = NDArray[np.float64]

#: The power floor, and it is deliberately the same 20 as
#: :func:`~hemolux.metrics.fairness.fairness_gate`. Two different floors in one
#: report would let a reader accept a subgroup on one line and reject the same
#: subgroup three lines down, with no way to tell which number was the mistake.
MIN_N = 20

#: The three binary questions, as ``(name, cutoff)``. ``None`` means sex-specific
#: -- see :func:`who_cutoff`. Written as data rather than three call sites so a
#: band cannot be added to the loop and forgotten in the CSV header.
SCREENING_BANDS: tuple[tuple[str, float | None], ...] = (
    ("anaemia", None),
    ("moderate_or_worse", WHO_MODERATE),
    ("severe", WHO_SEVERE),
)

#: Verdicts, as a closed set. ``"ok"`` is the only one that permits reading the
#: numbers; the other two are reasons not to.
VERDICTS = ("ok", "underpowered", "not_computable")


@dataclass(frozen=True)
class ThresholdMetrics:
    """A binary confusion matrix and the rates derived from it.

    Every count is stored, not just the rates, because a sensitivity without its
    denominator is not checkable and the counts are what a clinician audits.
    """

    label: str
    threshold: float | None
    n: int
    positives: int
    tp: int
    tn: int
    fp: int
    fn: int
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    verdict: str

    def to_dict(self) -> dict[str, object]:
        return {
            "band": self.label,
            "threshold": self.threshold,
            "n": self.n,
            "positives": self.positives,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ppv": self.ppv,
            "npv": self.npv,
            "verdict": self.verdict,
        }


def who_cutoff(sex: Sequence[str]) -> _FloatArr:
    """Per-patient anaemia cutoff: 13 g/dL for men, 12 for everyone else.

    The rule matches :func:`~hemolux.data.splits.severity_bin`, which decides the
    strata the split is built from. If the two disagreed, a patient could be
    "normal" for stratification and positive here, and the split would stratify
    on a label nothing else used.
    """
    return np.array(
        [
            WHO_CUTOFF_MALE if str(value).upper().startswith("M") else WHO_CUTOFF_FEMALE
            for value in sex
        ],
        dtype=np.float64,
    )


def _ratio(numerator: int, denominator: int) -> float:
    """A rate, or NaN when the denominator is empty.

    NaN rather than 0.0 on purpose. A sensitivity of 0 when there are no
    positives looks like total failure; the truth is that it was never measured.
    """
    return float(numerator / denominator) if denominator else float("nan")


def _verdict(n: int, positives: int, negatives: int, min_n: int) -> str:
    """Which of the three verdicts this cell earns.

    Ordered so the power floor is checked before the class check: a site with 12
    patients and no positives is *underpowered* -- the reason to distrust the row
    is the sample, and that is the more useful sentence to print.
    """
    if n == 0:
        return "not_computable"
    if n < min_n:
        return "underpowered"
    if positives == 0 or negatives == 0:
        return "not_computable"
    return "ok"


def threshold_metrics(
    y_true: NDArray[np.ndarray],
    y_pred: NDArray[np.ndarray],
    threshold: float | NDArray[np.ndarray],
    *,
    label: str = "",
    min_n: int = MIN_N,
) -> ThresholdMetrics:
    """Score one binary screening question.

    ``threshold`` is either a scalar or a per-patient array. The per-patient form
    is not a convenience: the anaemia definition genuinely depends on sex, so a
    single number would silently over-call anaemia in women by 1 g/dL. Lower Hb
    is the positive class, in both the truth and the prediction.

    Non-finite truth or prediction pairs are dropped before scoring, the same way
    :func:`~hemolux.metrics.regression._clean` handles them, so an abstained or
    failed row cannot contribute a spurious negative.
    """
    a_all = np.asarray(y_true, dtype=np.float64).ravel()
    b_all = np.asarray(y_pred, dtype=np.float64).ravel()
    if a_all.shape != b_all.shape:
        raise ValueError(f"y_true {a_all.shape} and y_pred {b_all.shape} disagree")

    if np.ndim(threshold) == 0:
        cut_all = np.full(a_all.shape, float(threshold), dtype=np.float64)
        recorded: float | None = float(threshold)
    else:
        cut_all = np.asarray(threshold, dtype=np.float64).ravel()
        if cut_all.shape != a_all.shape:
            raise ValueError(
                f"a per-patient threshold must match y_true, got {cut_all.shape} "
                f"against {a_all.shape}"
            )
        # Per-patient thresholds have no single value to print, and printing one
        # patient's cutoff as if it were the study cutoff is how a sex-specific
        # rule becomes a one-line footnote nobody reads.
        recorded = None

    ok = np.isfinite(a_all) & np.isfinite(b_all)
    a, b, cut = a_all[ok], b_all[ok], cut_all[ok]

    positive = a < cut
    called = b < cut
    tp = int(np.sum(positive & called))
    fn = int(np.sum(positive & ~called))
    fp = int(np.sum(~positive & called))
    tn = int(np.sum(~positive & ~called))
    positives = tp + fn
    negatives = tn + fp

    return ThresholdMetrics(
        label=label,
        threshold=recorded,
        n=int(a.size),
        positives=positives,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        sensitivity=_ratio(tp, tp + fn),
        specificity=_ratio(tn, tn + fp),
        ppv=_ratio(tp, tp + fp),
        npv=_ratio(tn, tn + fn),
        verdict=_verdict(int(a.size), positives, negatives, min_n),
    )


def screening_metrics(
    y_true: NDArray[np.ndarray],
    y_pred: NDArray[np.ndarray],
    sex: Sequence[str],
    *,
    min_n: int = MIN_N,
) -> dict[str, ThresholdMetrics]:
    """Every WHO band at once, keyed by band name."""
    sex_arr = np.asarray(sex, dtype=object).ravel()
    if sex_arr.shape != np.asarray(y_true, dtype=np.float64).ravel().shape:
        raise ValueError(
            f"sex has {sex_arr.size} entries but y_true has "
            f"{np.asarray(y_true, dtype=np.float64).size}; a misaligned sex column "
            f"would apply the wrong cutoff to the wrong patient"
        )
    sex_cut = who_cutoff(list(sex_arr))
    out: dict[str, ThresholdMetrics] = {}
    for band, fixed in SCREENING_BANDS:
        # A fixed cutoff is passed as a scalar, not broadcast to an array, so
        # ``threshold_metrics`` can record the value. Broadcasting it would make
        # every band look sex-specific and print ``threshold: null`` for the ones
        # that are not.
        if fixed is None:
            out[band] = threshold_metrics(y_true, y_pred, sex_cut, label=band, min_n=min_n)
        else:
            out[band] = threshold_metrics(y_true, y_pred, fixed, label=band, min_n=min_n)
    return out


def by_group(
    y_true: NDArray[np.ndarray],
    y_pred: NDArray[np.ndarray],
    sex: Sequence[str],
    group: NDArray[np.generic],
    *,
    min_n: int = MIN_N,
) -> dict[str, dict[str, ThresholdMetrics]]:
    """Every WHO band, computed inside each value of ``group``.

    ``group`` is a site or a severity band. Groups are visited in sorted order so
    the output is stable across runs and across numpy versions.
    """
    g = np.asarray(group).ravel()
    n = np.asarray(y_true, dtype=np.float64).ravel().size
    if g.size != n:
        raise ValueError(f"group has {g.size} entries against {n} patients")
    sex_arr = np.asarray(sex, dtype=object).ravel()
    if sex_arr.size != n:
        raise ValueError(f"sex has {sex_arr.size} entries against {n} patients")

    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    out: dict[str, dict[str, ThresholdMetrics]] = {}
    for value in sorted({str(v) for v in g}, key=str):
        mask = g.astype(str) == value
        out[value] = screening_metrics(a[mask], b[mask], list(sex_arr[mask]), min_n=min_n)
    return out


def screening_report(
    y_true: NDArray[np.ndarray],
    y_pred: NDArray[np.ndarray],
    sex: Sequence[str],
    site: NDArray[np.generic],
    *,
    min_n: int = MIN_N,
) -> dict[str, object]:
    """The block the results file carries: pooled, then per site.

    Pooled first and per site second, in that order, because the per-site numbers
    are the point and a reader who stops after the first section should have seen
    the pooled figure they are about to be disaggregated from. The bands are
    named in the output so an all-``not_computable`` severe row is legible as
    *missing data* rather than a row that failed to compute.
    """
    return {
        "bands": [band for band, _ in SCREENING_BANDS],
        "pooled": {
            band: m.to_dict()
            for band, m in screening_metrics(y_true, y_pred, sex, min_n=min_n).items()
        },
        "per_site": {
            site_value: {band: m.to_dict() for band, m in per_band.items()}
            for site_value, per_band in by_group(y_true, y_pred, sex, site, min_n=min_n).items()
        },
    }
