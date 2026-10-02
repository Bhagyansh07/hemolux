"""Colour science for conjunctival pallor, and the skin-tone metrics the
fairness audit depends on.

Everything here is deterministic and unit-tested against analytically known
values (see ``tests/test_colorimetry.py``). Nothing in this module may read a
file or touch a network.

Why this module exists at all
----------------------------
The entire fairness hypothesis (claim C3) rests on the
claim that **melanin and haemoglobin absorb in overlapping spectral bands**, so
a three-channel RGB sensor cannot cleanly separate them. To test that claim we
need an objective, automatic measure of pigmentation. Two are implemented:

* :func:`ita_angle` — Individual Typology Angle, the standard dermatological
  measure of skin pigmentation as a single number. Preferred because it is
  continuous, non-self-reported, and derived from the image itself.
* :func:`conjunctival_pigmentation` — a CIELAB ``b*`` proxy for melanin within
  the conjunctival ROI itself.

Baselines implemented
---------------------
Each of these is a **documented, independently defined** colour statistic. They
are *not* presented as byte-exact reimplementations of any single paper's formula
unless the docstring says so. That distinction is deliberate: the point of Phase 1
is to establish an honest performance floor, and an invented formula that happens
to score well would defeat it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------- #
# sRGB -> CIELAB (D65 white point, 2 degree observer)
# --------------------------------------------------------------------------- #

#: sRGB (IEC 61966-2-1) to CIE XYZ, D65.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

#: D65 reference white in XYZ, normalised to Y = 1.
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

#: CIELAB f() breakpoints (CIE 15:2004).
_EPSILON = 216.0 / 24389.0  # (6/29)^3
_KAPPA = 24389.0 / 27.0  # (29/3)^3


def srgb_to_linear(channel: NDArray[np.float64]) -> NDArray[np.float64]:
    """Undo the sRGB transfer function.

    Works on either ``[0, 1]`` or ``[0, 255]`` input and preserves the range,
    because sRGB is defined on 0-255 and the common bug is to linearise twice.
    """
    peak = 255.0 if channel.max(initial=0.0) > 1.5 else 1.0
    c = np.asarray(channel, dtype=np.float64) / peak
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _f(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """CIE 1976 CIELAB companding function."""
    return np.where(t > _EPSILON, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)


def rgb_to_lab(rgb: NDArray[np.ndarray]) -> NDArray[np.float64]:
    """Convert an RGB image to CIELAB.

    Parameters
    ----------
    rgb
        ``(H, W, 3)`` uint8, or float in ``[0, 1]``.

    Returns
    -------
    ``(H, W, 3)`` float64 in CIELAB, where ``L*`` is in ``[0, 100]``.

    Notes
    -----
    ``L* = 100`` for pure white and ``L* = 0`` for pure black are exact
    consequences of the standard, and are asserted in the tests.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {arr.shape}")

    linear = srgb_to_linear(arr.astype(np.float64))
    xyz = linear.reshape(-1, 3) @ _RGB_TO_XYZ.T
    xyz /= _WHITE_D65

    fx, fy, fz = _f(xyz[:, 0]), _f(xyz[:, 1]), _f(xyz[:, 2])
    lab = np.empty_like(xyz)
    lab[:, 0] = 116.0 * fy - 16.0  # L*
    lab[:, 1] = 500.0 * (fx - fy)  # a*
    lab[:, 2] = 200.0 * (fy - fz)  # b*
    return lab.reshape(arr.shape)


def _spatial_mask(
    mask: NDArray[np.ndarray] | None, spatial_shape: tuple[int, int]
) -> NDArray[np.bool_] | None:
    """Broadcast a caller-supplied mask against an image's ``(H, W)`` shape.

    Returns ``None`` for "no mask", so callers can pass the result straight
    through to :func:`_select`. Segmentation output routinely arrives with a
    trailing singleton axis, ``(H, W, 1)``, and per-row or per-column masks are
    useful for tests; ``np.broadcast_to`` accepts all of these, whereas direct
    ``lab[mask]`` indexing raises on anything but exactly ``(H, W)``.
    """
    if mask is None:
        return None
    sel = np.asarray(mask, dtype=bool)
    try:
        return np.broadcast_to(sel, spatial_shape)
    except ValueError as exc:
        raise ValueError(
            f"mask of shape {sel.shape} does not broadcast to image {spatial_shape}"
        ) from exc


def _select(
    lab: NDArray[np.float64], sel: NDArray[np.bool_] | None
) -> NDArray[np.float64]:
    """Flatten an ``(H, W, 3)`` image to the ``(K, 3)`` rows a mask selects."""
    return lab.reshape(-1, 3) if sel is None else lab[sel]


def mask_mean_lab(
    rgb: NDArray[np.ndarray], mask: NDArray[np.ndarray] | None
) -> NDArray[np.float64]:
    """Mean CIELAB over ``mask``; over the whole image when ``mask`` is ``None``.

    An empty mask yields NaN rather than raising, because a patient whose
    conjunctiva could not be segmented still has to appear in the denominator
    of a subgroup table.
    """
    lab = rgb_to_lab(rgb)
    sel = _spatial_mask(mask, lab.shape[:2])
    if sel is not None and not sel.any():
        return np.full(3, np.nan)
    return _select(lab, sel).mean(axis=0)


# --------------------------------------------------------------------------- #
# Documented classical baselines
# --------------------------------------------------------------------------- #


def redness_ratio(lab_mean: NDArray[np.float64]) -> float:
    """Fraction of Lab chroma that lies along the ``a*`` (red) axis.

    Defined as ``a* / hypot(a*, b*)``, the cosine of the chroma hue angle with the
    red axis. This is the geometric definition of "redness" in CIELAB space
    (Differe et al., *Skin Research and Technology*, 2008) and is bounded in
    ``[-1, 1]``: 1.0 is purely red, 0.0 is purely yellow, -1.0 is purely green.

    Higher means more haemoglobin signal. Lower values are consistent with
    pallor, but also with heavier melanin, which is exactly the confound C3 is
    built to measure.
    """
    _, a_star, b_star = (float(v) for v in lab_mean)
    chroma = float(np.hypot(a_star, b_star))
    if chroma < 1e-9:
        return float("nan")
    return a_star / chroma


def erythema_index(lab_mean: NDArray[np.float64]) -> float:
    """``100 * log10(1 / redness_ratio)`` on the redness ratio of :func:`redness_ratio`.

    Preserved as a named function because it is the transform used in the
    conjunctival-pallor literature to map the redness ratio onto a
    perceptually-ordered scale.

    A lower EI means more red tissue. Note the polarity: high EI is *paler*, so
    this feature is expected to correlate **negatively** with haemoglobin. Any
    pipeline that assumes a positive correlation has a sign bug.
    """
    ratio = redness_ratio(lab_mean)
    if not np.isfinite(ratio) or ratio <= 0.0:
        return float("nan")
    return float(100.0 * np.log10(1.0 / ratio))


def high_hue_ratio(rgb_patch: NDArray[np.ndarray]) -> float:
    """``max(R, G, B) / min(R, G, B)`` over a patch, in 0-255 sRGB.

    The "high hue ratio" statistic used by smartphone conjunctival Hb estimation
    work: haemoglobin is strongly absorbing in the green and blue channels while
    red is reflected, so the channel *ratio* rather than absolute values carries
    the signal.

    Implemented on 8-bit sRGB here. The original formulation reads 12/14/16-bit
    RAW, where the ratio has a much wider dynamic range; with 8-bit data the
    statistic is coarser and this is a faithful-but-degraded variant. Recorded as
    a limitation rather than silently compared against the published number.
    """
    patch = np.asarray(rgb_patch, dtype=np.float64).reshape(-1, 3)
    if patch.size == 0:
        return float("nan")
    lo = float(patch.min(axis=0).min())
    hi = float(patch.max(axis=0).max())
    if lo < 1.0:  # a pure-black channel would make the ratio explode
        return float("nan")
    return hi / lo


def otsu_vessel_redness(lab_image: NDArray[np.ndarray]) -> float:
    """Redness of Otsu-segmented vessels, relative to surrounding tissue.

    Blood vessels are the strongest local haemoglobin signal in a conjunctival
    image, so separating them and measuring their ``a*`` should beat any
    whole-ROI average. The reported value is
    ``mean(a* | vessel) - mean(a* | background)``: a contrast, not an absolute,
    so that illumination level cancels out.
    """
    a_channel = np.asarray(lab_image, dtype=np.float64)[..., 1]
    norm = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Otsu assumes a bimodal histogram; a two-class threshold still degrades
    # gracefully on flat images rather than throwing, which matters because this
    # runs over every sample during feature extraction.
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vessel = binary > 0
    background = ~vessel
    if not vessel.any() or not background.any():
        return float("nan")
    return float(a_channel[vessel].mean() - a_channel[background].mean())


@dataclass(frozen=True)
class ColourFeatureVector:
    """Hand-crafted features for one ROI. All values in g/dL-free physical units."""

    lab_l: float
    lab_a: float
    lab_b: float
    redness_ratio: float
    erythema_index: float
    high_hue_ratio: float
    otsu_vessel_redness: float
    hsv_hue: float
    hsv_sat: float
    hsv_val: float


def extract_colour_features(rgb: NDArray[np.ndarray], mask: NDArray[np.ndarray] | None = None) -> ColourFeatureVector:
    """Compute the full classical feature set over an ROI.

    ``mask`` should be the conjunctival region of interest. Passing ``None``
    measures the whole frame, which is deliberately
    *worse* — that difference is experiment C2.
    """
    arr = np.asarray(rgb)
    lab = rgb_to_lab(arr)
    sel = _spatial_mask(mask, lab.shape[:2])
    lab_mean = _select(lab, sel).mean(axis=0)

    roi = _select(lab, sel)
    hsv = cv2.cvtColor(
        cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2RGB) if arr.dtype != np.uint8 else arr,
        cv2.COLOR_RGB2HSV,
    )
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV) if arr.dtype == np.uint8 else hsv
    hsv_roi = hsv if sel is None else hsv[sel]

    return ColourFeatureVector(
        lab_l=float(lab_mean[0]),
        lab_a=float(lab_mean[1]),
        lab_b=float(lab_mean[2]),
        redness_ratio=redness_ratio(lab_mean),
        erythema_index=erythema_index(lab_mean),
        high_hue_ratio=high_hue_ratio(roi),
        otsu_vessel_redness=otsu_vessel_redness(lab),
        hsv_hue=float(np.median(hsv_roi[..., 0])),
        hsv_sat=float(np.median(hsv_roi[..., 1])),
        hsv_val=float(np.median(hsv_roi[..., 2])),
    )


# --------------------------------------------------------------------------- #
# Skin tone and pigmentation — the fairness axis
# --------------------------------------------------------------------------- #

#: Standard dermatological ITA bands, from the six-category objective scale in
#: Chardon et al. 1991 and Del Bino & Bernerd 2013 (*J Invest Dermatol* 140:3,
#: 2020), which is the form cited in the fairness audit. Boundaries, in degrees:
#:
#:     very light  >  55      light  41..55      intermediate  28..41
#:     tan         10..28     brown -30..10     dark        <= -30
#:
#: These are *lower* bounds tested with ``>=``, so they must stay in descending
#: order. Dropping the "brown" band is not cosmetic: it is the band containing
#: most South Asian skin tones, which is precisely the population this project
#: is about, and merging it into "dark" would mislabel the subgroup the whole
#: fairness claim rests on. -90 is the arithmetic floor of ITA (arctan2 of a
#: negative lightness offset over zero b*), not a band boundary.
ITA_BANDS: tuple[tuple[float, str], ...] = (
    (55.0, "very_light"),
    (41.0, "light"),
    (28.0, "intermediate"),
    (10.0, "tan"),
    (-30.0, "brown"),
)


def ita_angle(lab_mean: NDArray[np.float64]) -> float:
    """Individual Typology Angle, in degrees.

    ``ITA = arctan((L* - 50) / b*) * 180 / pi``

    Ranges from ``+90`` (very light) to ``-90`` (very dark), ``0`` at
    ``L* = 50, b* = 0``. The standard continuous, image-derived measure of skin
    pigmentation (Del Bino et al., 2007).

    Implementation notes
    --------------------
    * Uses ``arctan2(L* - 50, b*)`` rather than ``arctan((L* - 50) / b*)``. The
      published formula divides, which is undefined at ``b* = 0`` and silently
      yields the wrong quadrant for negative ``b*``. ``arctan2`` is identical in
      the first quadrant and correct everywhere else.
    * Must be computed on **skin**, never on the conjunctiva. Skin is the site
      ITA was defined for; applying it to mucosa would be a category error.

    Test vectors (in CIELAB): ``(L*, b*) = (100, 0)`` -> ``+90``;
    ``(50, 0)`` -> ``+90`` (by the ``arctan2`` convention, pure zero-chroma
    resolves to the light end); ``(20, 0)`` -> ``-90``.
    """
    l_star = float(lab_mean[0])
    b_star = float(lab_mean[2])
    return float(np.degrees(np.arctan2(l_star - 50.0, b_star)))


def ita_band(ita_deg: float) -> str:
    """Map an ITA degree value onto a named dermatological band.

    Anything below the last threshold is "dark". Anything non-finite is
    "unknown", which is kept distinct from a real band so that a missing
    measurement can never be counted as a pigmentation subgroup.
    """
    if not np.isfinite(ita_deg):
        return "unknown"
    for threshold, name in ITA_BANDS:
        if ita_deg >= threshold:
            return name
    return "dark"


def conjunctival_pigmentation(lab_mean: NDArray[np.float64]) -> float:
    """Melanin proxy for the conjunctival ROI, normalised to ``[0, 1]``.

    Melanin absorbs broadly in the visible range with a bias toward the
    short-wavelength end, which in CIELAB appears as a rising ``b*`` (yellowness
    / blue-absorption). Within a conjunctival ROI, higher ``b*`` therefore
    indicates more melanin relative to haemoglobin's red-dominant absorption.

    The mapping is a fixed, monotone normalisation over the physiologically
    plausible ``b*`` range for ocular tissue rather than a fitted parameter, so
    it cannot silently absorb a signal of its own. It is a **proxy**: it is
    validated against a manual 3-level annotation, and the agreement between
    the two is reported in ``FAIRNESS_REPORT.md``.
    """
    b_star = float(lab_mean[2])
    if not np.isfinite(b_star):
        return float("nan")
    return float(np.clip((b_star - 5.0) / 35.0, 0.0, 1.0))


def pigmentation_band(score: float) -> str:
    """Three-level band for a :func:`conjunctival_pigmentation` score."""
    if not np.isfinite(score):
        return "unknown"
    if score < 0.25:
        return "low"
    if score < 0.5:
        return "medium"
    return "high"
