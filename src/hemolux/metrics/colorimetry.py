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

Exposure normalisation
----------------------
:func:`white_balance` exists because absolute colour is not a property of the tissue.
The two sites in this corpus differ in exposure by about 20% -- Italy's frames sit
near B102/G85/R106 against India's B84/G66/R77 -- and site is confounded with
haemoglobin distribution (mean 11.47 g/dL against 13.83). A model reading raw RGB
therefore has to spend capacity separating illumination from pigment before it can
separate anaemia from health, and it may not manage it.

The mechanism that makes this worth undoing is specific rather than generic.
Haemoglobin absorbs in the green (490-577 nm) and reflects in the red (630-760 nm),
so anaemia is a **ratio** between two channels, not a level in one. An illuminant
that scales all three channels by the same unknown gain leaves that ratio intact, so
dividing it out removes the nuisance term and keeps the signal. That is why the
division is per-channel rather than a single global gain, and why the corrected image
still carries anaemia while the raw one does not.
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


def _select(lab: NDArray[np.float64], sel: NDArray[np.bool_] | None) -> NDArray[np.float64]:
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

    The extremes are read at the 1st and 99th percentile rather than at the true
    minimum and maximum, and that is a correction rather than a refinement. A
    conjunctival ROI here holds about 12 million pixels, and a global ``min`` is
    decided by whichever single pixel is darkest: ``Italy/2`` has 159 zero-valued
    red pixels out of 11.9 million, and that was enough to return NaN for the whole
    frame. 29 of the 217 patients -- 13% of the corpus, most of them Italian --
    failed on a speck pixel or two while their channel *means* were entirely
    ordinary. Over a region this large the percentile is a better estimate of the
    tissue's darkest channel than the smallest of twelve million samples, and it is
    what makes the statistic usable at all.
    """
    patch = np.asarray(rgb_patch, dtype=np.float64).reshape(-1, 3)
    if patch.size == 0:
        return float("nan")
    if patch.shape[0] >= 100:
        lo = float(np.percentile(patch, 1.0))
        hi = float(np.percentile(patch, 99.0))
    else:
        lo = float(patch.min())
        hi = float(patch.max())
    if lo < 1.0:  # a genuinely black channel would make the ratio explode
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


def extract_colour_features(
    rgb: NDArray[np.ndarray], mask: NDArray[np.ndarray] | None = None
) -> ColourFeatureVector:
    """Compute the full classical feature set over an ROI.

    ``mask`` should be the conjunctival region of interest. Passing ``None``
    measures the whole frame, which is deliberately
    *worse* — that difference is experiment C2.
    """
    arr = np.asarray(rgb)
    lab = rgb_to_lab(arr)
    sel = _spatial_mask(mask, lab.shape[:2])
    lab_mean = _select(lab, sel).mean(axis=0)

    # ``high_hue_ratio`` is defined on 0-255 sRGB, so it gets the sRGB ROI and not the
    # CIELAB one. It was previously handed ``_select(lab, sel)``, where ``b*`` is
    # negative for any real tissue -- so its ``lo < 1.0`` guard fired on every call and
    # the feature was silently NaN for every patient. Nothing caught it because no
    # caller had run this path; the unit tests call the function directly with the
    # uint8 input it documents, and nothing had compared the two.
    roi_srgb = _select(_as_uint8(arr), sel)
    hsv_roi = cv2.cvtColor(_as_uint8(arr), cv2.COLOR_RGB2HSV)
    hsv_roi = hsv_roi if sel is None else hsv_roi[sel]

    return ColourFeatureVector(
        lab_l=float(lab_mean[0]),
        lab_a=float(lab_mean[1]),
        lab_b=float(lab_mean[2]),
        redness_ratio=redness_ratio(lab_mean),
        erythema_index=erythema_index(lab_mean),
        high_hue_ratio=high_hue_ratio(roi_srgb),
        otsu_vessel_redness=otsu_vessel_redness(lab),
        hsv_hue=float(np.median(hsv_roi[..., 0])),
        hsv_sat=float(np.median(hsv_roi[..., 1])),
        hsv_val=float(np.median(hsv_roi[..., 2])),
    )


def _as_uint8(rgb: NDArray[np.ndarray]) -> NDArray[np.uint8]:
    """Rescale any accepted input to the uint8 OpenCV's colour conversions require.

    OpenCV defines HSV in terms of 8-bit values, so a float image in ``[0, 1]`` has
    to be multiplied **up** to 255, not merely cast. The distinction is not cosmetic:
    casting a ``[0, 1]`` float straight to uint8 truncates almost every pixel to zero,
    and its hue and saturation medians come out as zeros -- a plausible-looking
    answer, since a grey conjunctiva has no hue, produced by a conversion bug.

    The same ``> 1.5`` heuristic :func:`srgb_to_linear` uses decides whether the
    array is already 0-255, so the two cannot disagree about a frame's scale.
    """
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr
    values = arr.astype(np.float64)
    if values.max(initial=0.0) <= 1.5:
        values = values * 255.0
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Exposure normalisation
# --------------------------------------------------------------------------- #


#: Rec. 709 luma weights, for picking the brightest reference pixels.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _input_peak(rgb: NDArray[np.ndarray]) -> float:
    """The value that means "fully bright" for this array, from its dtype.

    ``srgb_to_linear`` guesses this from the data; here the dtype is authoritative,
    because a dark uint8 frame and a dark float frame are both "all near zero" and a
    data-driven guess would pick the wrong one.
    """
    return 255.0 if np.asarray(rgb).dtype == np.uint8 else 1.0


def _reference_pixels(
    rgb: NDArray[np.ndarray], mask: NDArray[np.ndarray] | None
) -> NDArray[np.float64]:
    """The pixels used to estimate the illuminant.

    **The ROI mask is excluded, and that is the substantive decision here.** A
    grey-world estimate taken over a red conjunctiva is pulled toward red by the very
    signal being measured, so dividing by it partially undoes the correction. Excluding
    the ROI leaves the surrounding tissue -- skin, sclera, lid -- to stand in for the
    illumination, which is what grey-world actually assumes: that the scene average is
    neutral. The ROI is the subject, not the reference.

    If the mask covers everything, or nothing survives it, the full frame is used and
    ``None`` is returned from :func:`white_balance`'s perspective -- an estimate over
    a biased region is worse than one over the whole frame, and this corpus does
    contain masks that cover 99.99% of the frame.
    """
    arr = np.asarray(rgb, dtype=np.float64)
    if mask is None:
        return arr.reshape(-1, 3)
    sel = np.asarray(mask).astype(bool)
    if sel.shape != arr.shape[:2]:
        return arr.reshape(-1, 3)
    outside = arr[~sel]
    return outside if outside.size else arr.reshape(-1, 3)


def white_balance(
    rgb: NDArray[np.ndarray],
    mask: NDArray[np.ndarray] | None = None,
    *,
    method: str = "grey_world",
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """Divide out an estimated illuminant, returning float in ``[0, 1]``.

    Parameters
    ----------
    rgb
        ``(H, W, 3)`` uint8, or float already in ``[0, 1]``.
    mask
        The ROI. Used **only** to decide which pixels are excluded from the
        illuminant estimate, never to scale the image. See
        :func:`_reference_pixels`.
    method
        ``"grey_world"`` divides each channel by its reference mean. This is the
        default because it is the weaker assumption: it needs the scene average to be
        neutral, whereas ``"white_patch"`` needs the brightest pixel to be a white
        specular reflection, and on a phone photo of an eye that pixel is usually a
        corneal highlight covering a handful of pixels.
    eps
        Guards the two degenerate frames -- all black and all white -- which would
        otherwise divide by zero. With it, an all-black frame returns all zeros
        rather than NaN, which keeps one bad frame from poisoning a mean over 217.

    Returns
    -------
    ``(H, W, 3)`` float64 in ``[0, 1]``.

    Notes
    -----
    Two design points worth stating because both are choices rather than necessities.

    **Brightness is restored with a single scalar, not per channel.** After the
    per-channel division the frame has an unbalanced colour balance but the wrong
    overall lightness; multiplying back by one number derived from the means puts
    ``L*`` back on the same scale it had before, so a balanced and an unbalanced
    ``L*`` can be compared. Scaling per channel instead would undo the correction
    that was just applied.

    **The output is clipped to ``[0, 1]``.** Division can push a pixel above the peak
    when its channel sits above the reference mean. Clipping discards that, which is
    a real loss; it is taken because an unclipped array would change ``L*``'s ceiling
    and make the balanced and unbalanced features non-comparable. Pixels affected are
    brighter than their channel's frame average, which in these crops is largely
    sclera and specular highlight rather than conjunctiva.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {arr.shape}")
    if method not in ("grey_world", "white_patch"):
        raise ValueError(f"unknown method {method!r}; expected grey_world or white_patch")

    values = arr.astype(np.float64) / _input_peak(arr)
    reference = _reference_pixels(arr, mask)

    if method == "grey_world":
        illuminant = reference.mean(axis=0)
    else:
        # The brightest reference pixel per channel. Taken per channel rather than
        # once for the frame, because a single bright pixel is usually neutral and
        # would leave the per-channel ratios untouched.
        brightest = reference.max(axis=0)
        luminances = reference @ _LUMA
        keep = luminances >= 0.99 * luminances.max()
        if keep.any():
            brightest = reference[keep].max(axis=0)
        illuminant = brightest

    # An all-black reference has no illuminant to remove. Returning the frame
    # untouched is the only safe answer: substituting eps would amplify noise into
    # a huge number, and substituting a constant would invent an illumination.
    illuminant = np.where(illuminant > eps, illuminant, 1.0)
    balanced = values / illuminant

    # One scalar, from the means, so lightness survives.
    scale = float(values.mean() / balanced.mean()) if balanced.mean() > eps else 1.0
    return np.clip(balanced * scale, 0.0, 1.0)


def balanced_rgb(
    rgb: NDArray[np.ndarray],
    mask: NDArray[np.ndarray] | None = None,
    *,
    method: str = "grey_world",
    eps: float = 1e-6,
) -> NDArray[np.uint8]:
    """:func:`white_balance` as a uint8 image, for the paths that need one.

    OpenCV's HSV conversion is only defined for uint8, and going through a float
    array's ``astype(uint8)`` would clip every value above 1.0 to white -- which,
    after balancing, is most of the bright half of the image. Converting the
    ``[0, 1]`` float by multiplying through is the difference between a hue and a
    field of 255s.
    """
    return np.round(white_balance(rgb, mask, method=method, eps=eps) * 255.0).astype(np.uint8)


def extract_colour_features_balanced(
    rgb: NDArray[np.ndarray],
    mask: NDArray[np.ndarray] | None = None,
    *,
    method: str = "grey_world",
) -> ColourFeatureVector:
    """The same features, measured on the exposure-normalised image.

    Thin wrapper on purpose. Balancing is a change to *what is measured*, not a new
    kind of measurement, so the two paths share one implementation and a difference
    between their results is attributable to the division and nothing else. A second
    copy of the ten feature calculations would make that impossible to establish and
    would let the two drift.

    The balanced image is rounded back to uint8 before measurement, which is a real
    approximation and is taken deliberately: it means the balanced and unbalanced
    paths differ only in the pixels and not in the quantisation, so a comparison
    between them measures the correction rather than the rounding.
    """
    return extract_colour_features(balanced_rgb(rgb, mask, method=method), mask)


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
    the two is reported in the fairness section of ``EVALS.md``.
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
