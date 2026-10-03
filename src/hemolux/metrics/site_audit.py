"""The cross-site half of the fairness audit: thresholds, calibration, confound.

Why cross-site *calibration* and not only cross-site error
----------------------------------------------------------
The single-split fairness audit reports error and bias per site. Neither is the
failure that harms a patient. A model can be unbiased on average across two sites
and still be miscalibrated in a way that falls on one group: at a threshold of
12 g/dL, sensitivity can be 0.94 in one site and 0.61 in the other while the
pooled MAE barely moves. That gap is invisible in a mean, and it is the number a
screening programme would be run on.

This corpus already shows the error half. Cross-site transfer is catastrophic --
R² -4.835 for India->Italy with a bias of -4.285 g/dL, and -2.358 the other way
with +3.171 -- and the *direction* of the bias is the direction of the exposure
difference between the sites. That last fact is why this module leads with the
confound rather than mentioning it in a footnote.

The confound, stated before any contrast
----------------------------------------
Site is not a biological category here. It is confounded with lighting, and in a
known direction: the Italian frames are exposed brighter than the Indian ones, and
the cross-site bias points the same way as the exposure offset. Any subgroup
comparison that omits this invites the reader to conclude something about
pigmentation that the two-site design cannot support. :data:`SITE_CONFOUND` is
therefore carried *inside* the report data and not left to the prose, so a consumer
of the JSON cannot present a site contrast without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hemolux.metrics.calibration import calibration_by_group
from hemolux.metrics.classification import screening_report
from hemolux.metrics.fairness import bias_vs_ita, fairness_gate

_FloatArr = NDArray[np.float64]

#: Printed before every site contrast, in the artefact itself. The exposure
#: numbers are the corpus means, measured in ``results.json``'s per-site block.
SITE_CONFOUND = (
    "Site is not a biological category in this corpus. It is confounded with "
    "lighting: the Italian frames are exposed brighter than the Indian ones (mean "
    "BGR 102/85/106 against 84/66/77), and the direction of the cross-site bias "
    "matches the direction of that exposure difference. Every site contrast below "
    "is therefore a contrast between two lighting setups as much as between two "
    "populations. A gap that survives this confound is evidence; one that does not "
    "is not evidence about pigmentation."
)


@dataclass(frozen=True)
class SiteAudit:
    """Thresholded screening metrics and calibration, per site, with the confound."""

    screening: dict[str, object]
    calibration: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "confound": SITE_CONFOUND,
            "screening": self.screening,
            "calibration": self.calibration,
        }


def audit_sites(
    y_true: NDArray[np.ndarray],
    y_pred: NDArray[np.ndarray],
    sex: Sequence[str],
    site: NDArray[np.generic],
    *,
    confidence: NDArray[np.ndarray] | None = None,
    correct: NDArray[np.generic] | None = None,
    n_bins: int = 10,
) -> SiteAudit:
    """Assemble the per-site screening and calibration block.

    ``confidence`` and ``correct`` are optional because a bare regression head
    emits no probability: calibration is a property of the ordinal or
    classification arms, and the block says so rather than fabricating a curve
    from a model that cannot produce one.
    """
    screening = screening_report(y_true, y_pred, sex, site)
    calibration = _calibration(confidence, correct, site, n_bins=n_bins)
    return SiteAudit(screening=screening, calibration=calibration)


def _calibration(
    confidence: NDArray[np.ndarray] | None,
    correct: NDArray[np.generic] | None,
    site: NDArray[np.generic],
    *,
    n_bins: int,
) -> dict[str, object]:
    if confidence is None or correct is None:
        return {
            "available": False,
            "reason": (
                "a bare regression head emits no probability to be calibrated, so "
                "there is no confidence to bin. This is a measurement that was not "
                "taken, not a model that failed a check."
            ),
        }

    curves = calibration_by_group(
        np.asarray(confidence, dtype=np.float64),
        np.asarray(correct),
        np.asarray(site, dtype=object),
        n_bins=n_bins,
    )
    per_site = {name: curve.to_dict() for name, curve in curves.items() if name != "pooled"}
    finite = [c.ece for name, c in curves.items() if name != "pooled" and np.isfinite(c.ece)]
    # The gap between the best- and worst-calibrated site is the number a pooled
    # ECE hides. NaN rather than 0.0 when fewer than two sites are finite: an
    # unmeasured gap is not a zero gap.
    gap = float(max(finite) - min(finite)) if len(finite) >= 2 else float("nan")
    return {
        "available": True,
        "n_bins": n_bins,
        "pooled": curves["pooled"].to_dict(),
        "per_site": per_site,
        "max_ece_gap": gap,
    }


def calibration_gate(
    confidence: NDArray[np.ndarray],
    correct: NDArray[np.generic],
    ita_deg: NDArray[np.ndarray],
    *,
    min_abs_slope: float = 0.002,
) -> dict[str, object]:
    """Does the *miscalibration* itself depend on the pigment axis?

    Runs the same OLS and the same :func:`~hemolux.metrics.fairness.fairness_gate`
    as C3, on the calibration residual ``confidence - correct`` instead of on the
    Hb error. Reusing the gate is the point: one power floor, one set of three
    words, so "underpowered" means the same thing everywhere in the report.

    Read the verdict as a claim about the *hypothesis*, matching C3:
    ``"supported"`` means the miscalibration is pigmentation-dependent, not that
    the model is fair.
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    hit = np.asarray(correct).ravel().astype(np.float64)
    ita = np.asarray(ita_deg, dtype=np.float64).ravel()
    if not (conf.shape == hit.shape == ita.shape):
        raise ValueError(
            f"confidence {conf.shape}, correct {hit.shape} and ita {ita.shape} "
            f"must be the same length; a misaligned ITA column would regress one "
            f"patient's residual on another patient's pigmentation"
        )

    # A positive residual is an overconfident call. bias_vs_ita computes
    # `y_pred - y_true`, so passing zeros as the truth leaves the residual as y.
    residual = conf - hit
    slope = bias_vs_ita(np.zeros_like(residual), residual, ita)
    return {
        "hypothesis": "calibration error is associated with ITA",
        "verdict": fairness_gate(slope, min_abs_slope=min_abs_slope),
        "slope_per_degree": slope.slope,
        "ci_low": slope.ci_low,
        "ci_high": slope.ci_high,
        "n": slope.n,
    }
