"""The quality gate that runs *before* inference.

A model handed a blurred or badly framed image will return a confident wrong
answer. Catching that costs three milliseconds; not catching it costs the
user's trust and, in the intended use, potentially a missed referral. So quality
is checked first, and every rejection message names the specific thing to fix.

The thresholds here are deliberately conservative: false rejections cost the user
one retake, false acceptances cost them a wrong number. Bias the gate toward
rejecting.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

#: Mean luma (0-255) below which an image is considered too dark.
DARK_LUMA = 40.0
#: Mean luma above which an image is considered overexposed.
BRIGHT_LUMA = 235.0
#: Fraction of pixels at 255 above which highlights are considered clipped.
CLIPPED_FRACTION = 0.02
#: Variance of the Laplacian below which an image is considered blurred. The
#: metric is the standard OpenCV focus measure: it is scale-dependent, which is
#: why it is only meaningful for a fixed resolution.
BLUR_VARIANCE = 40.0
#: Conjunctival ROI must occupy between these fractions of the frame.
ROI_MIN_AREA = 0.015
ROI_MAX_AREA = 0.85
#: Quality below which the product abstains regardless of model confidence.
#:
#: Measured, not chosen. Over all 862 photographs in the corpus the lowest score
#: of any kind is 0.4087, and among the 320 that clear the hard gate above the
#: lowest is 0.5907. The previous value of 0.35 therefore sat below the entire
#: observed range and ``should_abstain``'s quality condition could never fire on
#: a real photograph -- an inert safety path that read as a working one.
#:
#: 0.60 sits immediately above the measured floor of the passing population, so
#: the branch is live and rejects the single worst photograph that the hard gate
#: lets through (1 of 320, 0.3%). Deliberately conservative: the hard gate has
#: already rejected 542 of 862, so this score's remaining job is to catch the
#: marginal cases among survivors, not to re-litigate the ones already refused.
QUALITY_ABSTAIN = 0.60


@dataclass(frozen=True)
class QualityReport:
    """Result of the quality gate.

    ``passed`` is the single field the UI branches on. Every other field exists so
    the UI can say *why*, which is the difference between a usable rejection and
    an infuriating one.
    """

    passed: bool
    reason: str | None
    message: str | None
    score: float
    luma: float
    clipped_fraction: float
    sharpness: float
    roi_area: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "message": self.message,
            "score": round(self.score, 4),
            "luma": round(self.luma, 2),
            "clipped_fraction": round(self.clipped_fraction, 4),
            "sharpness": round(self.sharpness, 2),
            "roi_area": None if self.roi_area is None else round(self.roi_area, 4),
        }


def laplacian_sharpness(gray: NDArray[np.ndarray]) -> float:
    """Variance of the Laplacian: the standard single-image focus measure.

    High for a sharp image, near zero for a blurred one. Note it is resolution
    dependent, so it must be computed after resizing, never on the original.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_quality(
    image_bgr: NDArray[np.ndarray],
    roi_mask: NDArray[np.ndarray] | None = None,
) -> QualityReport:
    """Run every quality check and return a single verdict.

    Order is deliberate — brightness before sharpness before framing. The user
    fixes the most visible problem first, and a dark image's sharpness reading
    is unreliable anyway.
    """
    if image_bgr is None or image_bgr.size == 0:
        return QualityReport(
            False, "DECODE_FAIL", "We couldn't read that image.", 0.0, 0, 0, 0, None
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    luma = float(gray.mean())
    clipped = float(np.mean(gray >= 255))
    sharp = laplacian_sharpness(gray)
    roi_area = None if roi_mask is None else float(np.mean(np.asarray(roi_mask) > 0))

    # Score in [0, 1]: a weighted blend of exposure, sharpness and framing. Used
    # by the abstention gate, so it must be comparable across images — it is a
    # heuristic, not a calibrated probability, and is never reported as one.
    exposure = float(np.clip(1.0 - abs(luma - 130.0) / 130.0, 0.0, 1.0))
    sharpness_score = float(np.clip(np.log1p(sharp) / np.log1p(400.0), 0.0, 1.0))
    framing = 1.0 if roi_area is None else float(np.clip(roi_area / 0.10, 0.0, 1.0))
    score = 0.4 * exposure + 0.4 * sharpness_score + 0.2 * framing

    if luma < DARK_LUMA:
        return QualityReport(
            False,
            "TOO_DARK",
            "The photo is too dark to read. Move into daylight and retake it.",
            score,
            luma,
            clipped,
            sharp,
            roi_area,
        )
    if luma > BRIGHT_LUMA or clipped > CLIPPED_FRACTION:
        return QualityReport(
            False,
            "TOO_BRIGHT",
            "The photo is overexposed. Move away from direct light or the flash.",
            score,
            luma,
            clipped,
            sharp,
            roi_area,
        )
    if sharp < BLUR_VARIANCE:
        return QualityReport(
            False,
            "TOO_BLUR",
            "The photo is blurry. Hold the phone steady and tap to focus on the eye.",
            score,
            luma,
            clipped,
            sharp,
            roi_area,
        )
    if roi_area is not None and (roi_area < ROI_MIN_AREA or roi_area > ROI_MAX_AREA):
        return QualityReport(
            False,
            "WRONG_FRAMING",
            "We couldn't find the eye area. Get closer so the eye fills the frame.",
            score,
            luma,
            clipped,
            sharp,
            roi_area,
        )

    return QualityReport(True, None, None, score, luma, clipped, sharp, roi_area)
