"""Calibration and selective prediction.

Why a confidence number needs its own module
--------------------------------------------
The obvious way to build confidence is to take the softmax over classes and
report the top probability. That number is **not** a probability, because the
network was never trained to be one. It is overconfident, sometimes severely, and
its overconfidence is not uniform across subgroups — which means a badly
calibrated model is *also* an unfair one: it is most confidently wrong on
exactly the group that was already worst served.

Two things are therefore built here:

1. :func:`temperature_scale` — one scalar fitted on the validation set, the
   cheapest possible correction, and the standard baseline that any more
   elaborate scheme has to beat.
2. :func:`risk_coverage_curve` — the mechanism behind abstention. Sort by
   confidence, keep the most confident fraction, measure the error of what
   remains. If MAE does not fall as coverage drops, confidence carries no
   information and the whole abstention feature must be withdrawn.

Fitting on the test set is made impossible by construction, not by convention:
:func:`temperature_scale` accepts a :class:`ValidationOnly` token and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from hemolux.data.quality import QUALITY_ABSTAIN

_FloatArr = NDArray[np.float64]

#: Ordinal bin centres in g/dL. Must match ``src/hemolux/models/heads.py``.
HB_BIN_EDGES = (4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0)
HB_BIN_CENTRES = tuple((lo + hi) / 2.0 for lo, hi in pairwise(HB_BIN_EDGES))
N_BINS = len(HB_BIN_CENTRES)


class ValidationOnly:
    """Token proving a calibration fit used validation data.

    ``temperature_scale`` refuses anything else. This is deliberate: the most
    common silent failure in a medical-ML report is a confidence number that was
    tuned on the very data it is reported against, which makes the reported
    calibration meaningless.
    """

    __slots__ = ("n",)

    def __init__(self, n: int) -> None:
        self.n = int(n)


def expected_hb(probs: NDArray[np.ndarray]) -> _FloatArr:
    """Decode ordinal probabilities into a haemoglobin estimate.

    ``Hb_hat = sum_k p_k * mu_k`` where ``mu_k`` are the bin centres.

    This is the expectation under the predicted distribution, so it is a
    genuine posterior mean rather than an argmax bin lookup. It also means the
    output is continuous and can never fall outside the bin range, which is a
    real deployment advantage over classification-then-threshold.

    Probabilities are renormalised defensively, so a numerically drifting
    onnxruntime output still decodes to a sensible number.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != N_BINS:
        raise ValueError(f"expected (N, {N_BINS}) probabilities, got shape {p.shape}")
    totals = p.sum(axis=1, keepdims=True)
    if not np.all(totals > 1e-12):
        raise ValueError("a row of probabilities sums to zero")
    p = p / totals
    return p @ np.asarray(HB_BIN_CENTRES)


def predictive_sigma(probs: NDArray[np.ndarray]) -> _FloatArr:
    """Predictive standard deviation, in g/dL, from the decoded distribution.

    ``sigma = sqrt(E[X^2] - E[X]^2)`` over the bin centres. Because the bins are
    ordered and the distribution is ordinal, this is a usable uncertainty signal
    without training a separate variance head — which matters, since the dataset
    has 218 patients and a second head would simply overfit.
    """
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    mu = np.asarray(HB_BIN_CENTRES)
    mean = p @ mu
    second = p @ (mu**2)
    return np.sqrt(np.maximum(second - mean**2, 0.0))


def soft_label(hb: _FloatArr, sigma: float = 0.6) -> _FloatArr:
    """Build a Gaussian label distribution over bins from a scalar Hb value.

    This is the label-distribution-learning step of experiment C1. Haemoglobin is
    an *ordered* quantity, so a target of 10.2 g/dL is far more informative about
    10.0 than about 14.0. One-hot encoding throws that ordering away; a soft
    target keeps it, and the loss then rewards getting the neighbourhood right
    rather than one exact bin.
    """
    values = np.asarray(hb, dtype=np.float64).ravel()
    centres = np.asarray(HB_BIN_CENTRES)
    logits = -0.5 * ((values[:, None] - centres[None, :]) / sigma) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def ordinal_target_index(hb: _FloatArr) -> NDArray[np.int64]:
    """Index of the bin containing each Hb value. Used only for ECE, never as a
    training target on its own."""
    values = np.asarray(hb, dtype=np.float64).ravel()
    centres = np.asarray(HB_BIN_CENTRES)
    return np.abs(values[:, None] - centres[None, :]).argmin(axis=1)


# --------------------------------------------------------------------------- #
# ECE and Brier
# --------------------------------------------------------------------------- #


def expected_calibration_error(
    confidence: _FloatArr,
    correct: NDArray[np.bool_] | _FloatArr,
    n_bins: int = 10,
) -> float:
    """Expected calibration error, binned by confidence.

    ECE = ``sum_b (n_b / N) * |acc(b) - conf(b)|``.
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    hit = np.asarray(correct).ravel().astype(np.float64)
    if conf.shape != hit.shape:
        raise ValueError(f"shape mismatch: {conf.shape} vs {hit.shape}")
    if conf.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # A confidence of exactly 1.0 belongs to the last bin; np.digitize would drop it.
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += (mask.sum() / conf.size) * abs(hit[mask].mean() - conf[mask].mean())
    return float(total)


def brier_score(confidence: _FloatArr, correct: NDArray[np.bool_] | _FloatArr) -> float:
    """Brier score for a single-label confidence vector."""
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    hit = np.asarray(correct).ravel().astype(np.float64)
    return float(np.mean((conf - hit) ** 2))


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TemperatureResult:
    temperature: float
    ece_before: float
    ece_after: float

    def to_dict(self) -> dict[str, float]:
        return {
            "temperature": self.temperature,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
        }


def _apply_temperature(logits: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    scaled = logits / max(t, 1e-3)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def temperature_scale(
    logits: NDArray[np.ndarray],
    targets: NDArray[np.int64],
    token: ValidationOnly,
    *,
    n_grid: int = 60,
) -> TemperatureResult:
    """Fit a single temperature by grid search, minimising validation NLL.

    Requires :class:`ValidationOnly`; passing anything else raises, because a
    temperature fitted on the test split invalidates every calibration number
    downstream of it. Nothing downstream would notice.

    Grid search rather than L-BFGS on purpose: the objective is smooth and
    one-dimensional, a grid is trivially reproducible, and there is no optimiser
    dependency to drift between versions.
    """
    if not isinstance(token, ValidationOnly):
        raise TypeError(
            "temperature_scale requires a ValidationOnly token; calibrating on the "
            "test split invalidates every calibration number reported downstream"
        )

    logit_arr = np.asarray(logits, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.int64).ravel()
    if logit_arr.ndim != 2:
        raise ValueError(f"expected (N, K) logits, got shape {logit_arr.shape}")
    if logit_arr.shape[0] != target_arr.size:
        raise ValueError("logits and targets disagree on the number of samples")
    if token.n != target_arr.size:
        raise ValueError(
            f"ValidationOnly token reports n={token.n} but {target_arr.size} targets were supplied"
        )

    def nll(t: float) -> float:
        probs = _apply_temperature(logit_arr, t)
        picked = np.clip(probs[np.arange(probs.shape[0]), target_arr], 1e-12, 1.0)
        return float(-np.mean(np.log(picked)))

    grid = np.linspace(0.25, 5.0, n_grid)
    losses = [nll(t) for t in grid]
    best_t = float(grid[int(np.argmin(losses))])

    probs_before = _apply_temperature(logit_arr, 1.0)
    probs_after = _apply_temperature(logit_arr, best_t)
    hit = target_arr == probs_after.argmax(axis=1)

    return TemperatureResult(
        temperature=best_t,
        ece_before=expected_calibration_error(probs_before.max(axis=1), hit),
        ece_after=expected_calibration_error(probs_after.max(axis=1), hit),
    )


# --------------------------------------------------------------------------- #
# Selective prediction / abstention
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CoveragePoint:
    coverage: float
    mae: float
    threshold: float


def risk_coverage_curve(
    uncertainty: _FloatArr,
    y_true: _FloatArr,
    y_pred: _FloatArr,
    *,
    n_points: int = 20,
) -> list[CoveragePoint]:
    """Sweep a confidence threshold and measure the error of what survives.

    ``uncertainty`` is lower-is-more-confident (predictive sigma, or the entropy
    of the ordinal distribution).

    **The property to check**
    ------------------------
    MAE at low coverage must be lower than MAE at full coverage. If it is not,
    the confidence signal carries no information about the error, and the
    abstention feature must be withdrawn from the product rather than shipped on
    faith.

    Note this is an endpoint comparison, *not* a per-step monotonicity
    requirement. An earlier version of this docstring demanded that MAE be
    non-increasing at every step, and pinned that with a test. That claim is
    false. Each point is the mean of a *prefix* of the confidence-sorted error
    sequence, and a running mean is not monotone; adding a sample with a large
    error can raise it. A concrete counterexample with n = 10, one outlier at
    mid-ranked uncertainty and four exact predictions after it, gives

        coverage  1.0  0.9  0.8  0.7  0.6  0.5  0.4  0.3  0.2  0.1
        MAE      0.9 1.00 1.13 1.29 1.50 1.80 0.0  0.0  0.0  0.0

    so MAE rises while coverage falls. Monotonicity holds only in expectation
    over orderings, which a single dataset does not supply. A test that asserted
    it would have failed on correct code.

    ``tests/test_calibration.py`` therefore asserts the endpoint property on a
    genuinely informative signal, and additionally asserts that a pure-noise
    signal shows no such improvement.
    """
    unc = np.asarray(uncertainty, dtype=np.float64).ravel()
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    if not (unc.shape == y_t.shape == y_p.shape):
        raise ValueError("uncertainty, y_true and y_pred must have identical shapes")

    order = np.argsort(unc)  # most certain first
    abs_err = np.abs(y_p - y_t)[order]
    n = abs_err.size

    points: list[CoveragePoint] = []
    for frac in np.linspace(1.0, 0.1, n_points):
        k = max(1, round(frac * n))
        points.append(
            CoveragePoint(
                coverage=round(k / n, 4),
                mae=float(abs_err[:k].mean()),
                threshold=float(unc[order][k - 1]),
            )
        )
    return points


def abstention_threshold(curve: list[CoveragePoint], target_coverage: float = 0.75) -> float:
    """Uncertainty threshold that retains ``target_coverage`` of predictions."""
    for point in curve:
        if point.coverage <= target_coverage:
            return point.threshold
    return curve[-1].threshold if curve else float("inf")


def should_abstain(sigma: float, threshold: float, quality: float = 1.0) -> bool:
    """The single gate the product uses.

    Abstain when either the model is uncertain **or** the image is poor. Both
    conditions are necessary: a well-trained model on a blurred image will happily
    return a confident wrong answer.

    The quality threshold is :data:`~hemolux.data.quality.QUALITY_ABSTAIN`, not a
    literal. It used to be spelled ``0.35`` here while the module defining it
    carried a different value, so the gate compared against a number no one was
    maintaining: raising the constant in ``quality.py`` changed nothing. That is
    the failure mode a duplicated literal always produces.
    """
    if not np.isfinite(sigma):
        return True
    return bool(sigma > threshold or quality < QUALITY_ABSTAIN)
