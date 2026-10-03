/**
 * Colourimetry for the browser, ported from `hemolux.metrics.colorimetry`.
 *
 * The shipping path is `onnxruntime-web`, and the winning model is built on the 13
 * hand-crafted colour features in `COLOUR_COLUMNS`. Those features have to be computed
 * here exactly as Python computes them, because a model trained on one definition and
 * served another degrades silently: the numbers stay in range, only their meaning
 * changes. This module is dependency-free and runs unchanged under Node 24, so
 * `tests/test_colourimetry_js.py` can prove numerical parity against the real
 * `white_balance` + `extract_colour_features`.
 *
 * Input contract
 * --------------
 * `rgb` is a `Uint8Array` of length `width * height * 3`, interleaved RGB, values 0-255.
 * `mask` is a `Uint8Array` of length `width * height` where nonzero means "inside the
 * ROI", or `null` for the whole frame. This is the shape the browser receives from a
 * canvas and the shape the training loader hands the Python functions after
 * `_mask_at_frame_size`, so no resampling lives here.
 *
 * What is deliberately copied even where it looks wrong
 * -----------------------------------------------------
 * The port's job is byte parity, not improvement. Three Python choices are therefore
 * reproduced rather than corrected:
 *
 * * `srgb_to_linear` picks its scale from the data (`peak = 255 if max > 1.5 else 1`),
 *   not from the declared dtype. A uint8 frame whose maximum is 1 is treated as
 *   full-scale, which is a quirk; `rgbToLab` copies it so the two implementations cannot
 *   disagree about a frame's scale.
 * * `cv2.normalize(..., NORM_MINMAX).astype(uint8)` returns all zeros when the a*
 *   channel is flat (`max == min`). That is what OpenCV does, so `otsuVesselRedness`
 *   feeds an all-zero image into Otsu rather than raising.
 * * `np.round` is round-half-to-even, not JavaScript's half-up `Math.round`. The
 *   balanced frame is quantised with `roundHalfEven` or the two pipelines drift by one
 *   grey level on exact .5 boundaries.
 *
 * OpenCV integer arithmetic
 * -------------------------
 * `otsuVesselRedness` and `cvtColorRGB2HSV` cannot be expressed as portable float maths:
 * OpenCV uses fixed-point lookup tables and its own Otsu loop. Both are transcribed from
 * the OpenCV scalar source (`color_hsv.simd.hpp` `RGB2HSV_b`, `thresh.cpp`
 * `getThreshVal_Otsu`) rather than approximated. `normalize` is the one place where
 * OpenCV's SIMD FMA and this module's two-rounding multiply-add can differ by an ulp;
 * because the output is truncated to uint8, that shows up only at an exact integer
 * boundary, and it was empirically unable to move the Otsu threshold.
 */

const RGB_TO_XYZ = [
  [0.4124564, 0.3575761, 0.1804375],
  [0.2126729, 0.7151522, 0.072175],
  [0.0193339, 0.119192, 0.9503041],
];
const WHITE_D65 = [0.95047, 1.0, 1.08883];
const EPSILON = 216.0 / 24389.0; // (6/29)^3
const KAPPA = 24389.0 / 27.0; // (29/3)^3
const GREY_WORLD_EPS = 1e-6;

//: `FLT_EPSILON`, the guard OpenCV's Otsu loop compares its class masses against.
const FLT_EPSILON = 1.1920928955078125e-7;

/** CIE 1976 CIELAB companding function. */
function _f(t) {
  return t > EPSILON ? Math.cbrt(t) : (KAPPA * t + 16.0) / 116.0;
}

/** Undo the sRGB transfer function on one already-normalised channel. */
function _linearize(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/**
 * `cvRound`: round half to even.
 *
 * OpenCV's `saturate_cast<int>(double)` is a hardware convert with the default
 * round-to-nearest-even mode. `Math.round` rounds halves toward +infinity, which would
 * build a different `hdiv`/`sdiv` table on the exact .5 entries and shift a hue.
 */
function cvRound(x) {
  const fl = Math.floor(x);
  const frac = x - fl;
  if (frac > 0.5) return fl + 1;
  if (frac < 0.5) return fl;
  return fl % 2 === 0 ? fl : fl + 1;
}

/**
 * `np.round`: round half to even.
 *
 * Separate from `cvRound` for clarity even though the rule is identical; the balanced
 * frame is quantised here, and numpy's `rint` is the reference.
 */
function roundHalfEven(x) {
  const fl = Math.floor(x);
  const frac = x - fl;
  if (frac > 0.5) return fl + 1;
  if (frac < 0.5) return fl;
  return fl % 2 === 0 ? fl : fl + 1;
}

/** numpy's `astype(uint8)` on a float: truncate toward zero, then wrap modulo 256. */
function toUint8(x) {
  const t = Math.trunc(x);
  return ((t % 256) + 256) % 256;
}

/**
 * sRGB (0-255) to CIELAB, D65 / 2-degree observer.
 *
 * Returns a `Float64Array` of length `width * height * 3` in `(L*, a*, b*)` order. The
 * per-frame peak is the same data-driven guess `srgb_to_linear` makes, so a dark frame
 * and a full-scale frame are classified identically by both implementations.
 */
export function rgbToLab(rgb, width, height) {
  const n = width * height;
  let maxValue = 0;
  for (let i = 0; i < rgb.length; i++) {
    if (rgb[i] > maxValue) maxValue = rgb[i];
  }
  const peak = maxValue > 1.5 ? 255.0 : 1.0;

  const out = new Float64Array(n * 3);
  for (let p = 0; p < n; p++) {
    const r = _linearize(rgb[p * 3] / peak);
    const g = _linearize(rgb[p * 3 + 1] / peak);
    const b = _linearize(rgb[p * 3 + 2] / peak);

    const x = (RGB_TO_XYZ[0][0] * r + RGB_TO_XYZ[0][1] * g + RGB_TO_XYZ[0][2] * b) / WHITE_D65[0];
    const y = (RGB_TO_XYZ[1][0] * r + RGB_TO_XYZ[1][1] * g + RGB_TO_XYZ[1][2] * b) / WHITE_D65[1];
    const z = (RGB_TO_XYZ[2][0] * r + RGB_TO_XYZ[2][1] * g + RGB_TO_XYZ[2][2] * b) / WHITE_D65[2];

    const fx = _f(x);
    const fy = _f(y);
    const fz = _f(z);
    out[p * 3] = 116.0 * fy - 16.0;
    out[p * 3 + 1] = 500.0 * (fx - fy);
    out[p * 3 + 2] = 200.0 * (fy - fz);
  }
  return out;
}

/** The `(H, W)` boolean selection as flat indices, or `null` for the whole frame. */
function maskIndices(mask, n) {
  if (mask === null || mask === undefined) return null;
  if (mask.length !== n) {
    throw new Error(`mask length ${mask.length} does not match frame pixels ${n}`);
  }
  const idx = new Int32Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (mask[i] !== 0) idx[count++] = i;
  }
  return idx.subarray(0, count);
}

/**
 * Per-channel mean over the selected pixels, or `NaN` per channel for an empty
 * selection.
 *
 * The empty case is not an error: a patient whose conjunctiva could not be segmented
 * still has to occupy a row, and numpy returns NaN rather than raising.
 */
function meanChannels(image, idx, n) {
  const sums = [0.0, 0.0, 0.0];
  const count = idx === null ? n : idx.length;
  if (count === 0) return [Number.NaN, Number.NaN, Number.NaN];
  if (idx === null) {
    for (let p = 0; p < n; p++) {
      sums[0] += image[p * 3];
      sums[1] += image[p * 3 + 1];
      sums[2] += image[p * 3 + 2];
    }
  } else {
    for (let k = 0; k < idx.length; k++) {
      const p = idx[k];
      sums[0] += image[p * 3];
      sums[1] += image[p * 3 + 1];
      sums[2] += image[p * 3 + 2];
    }
  }
  return [sums[0] / count, sums[1] / count, sums[2] / count];
}

/**
 * Fraction of Lab chroma along the red axis: `a* / hypot(a*, b*)`.
 *
 * NaN when chroma is below `1e-9`, matching the Python guard: a grey pixel has no hue,
 * so a redness of 0.0 would be a measurement rather than the absence of one.
 */
export function rednessRatio(labA, labB) {
  const chroma = Math.hypot(labA, labB);
  if (chroma < 1e-9) return Number.NaN;
  return labA / chroma;
}

/** `100 * log10(1 / redness_ratio)`, NaN when the ratio is non-finite or non-positive. */
export function erythemaIndex(labA, labB) {
  const ratio = rednessRatio(labA, labB);
  if (!Number.isFinite(ratio) || ratio <= 0.0) return Number.NaN;
  return 100.0 * Math.log10(1.0 / ratio);
}

/** numpy's default linear-interpolation percentile over an already sorted array. */
function percentileLinear(sorted, q) {
  const n = sorted.length;
  const idx = ((n - 1) * q) / 100.0;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/** Median of a numeric list, averaging the two middle values for an even count. */
function median(values) {
  const n = values.length;
  if (n === 0) return Number.NaN;
  const sorted = Float64Array.from(values);
  sorted.sort();
  const mid = n >> 1;
  if (n % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2.0;
}

/**
 * `max(R, G, B) / min(R, G, B)` over a patch, using the 1st/99th percentile extremes
 * once the patch has at least 100 pixels.
 *
 * `patch` is the flattened sRGB ROI (three values per pixel). The percentile is a
 * correction for a global minimum decided by one dead pixel out of millions; on a small
 * patch it falls back to the true min/max, exactly as the Python does. NaN when the low
 * extreme is below 1.0, because a genuinely black channel makes the ratio explode.
 */
export function highHueRatio(patch) {
  if (patch.length === 0) return Number.NaN;
  const pixelCount = patch.length / 3;
  let lo;
  let hi;
  if (pixelCount >= 100) {
    const sorted = Float64Array.from(patch);
    sorted.sort();
    lo = percentileLinear(sorted, 1.0);
    hi = percentileLinear(sorted, 99.0);
  } else {
    lo = patch[0];
    hi = patch[0];
    for (let i = 1; i < patch.length; i++) {
      if (patch[i] < lo) lo = patch[i];
      if (patch[i] > hi) hi = patch[i];
    }
  }
  if (lo < 1.0) return Number.NaN;
  return hi / lo;
}

/**
 * OpenCV's 8-bit RGB2HSV, transcribed from `RGB2HSV_b` in `color_hsv.simd.hpp`.
 *
 * H is in `[0, 180]`, S and V in `[0, 255]`. The two division tables and the `>> 12`
 * fixed-point steps are the whole point: a float conversion rounds differently and the
 * hue column is then not the one the model was trained on. The scalar branch (not the
 * SIMD one) is the reference here; the two agree bit-for-bit on uchar input.
 */
const HSV_SHIFT = 12;
const SDIV_TABLE = new Int32Array(256);
const HDIV_TABLE = new Int32Array(256);
for (let i = 1; i < 256; i++) {
  SDIV_TABLE[i] = cvRound((255 << HSV_SHIFT) / (1.0 * i));
  HDIV_TABLE[i] = cvRound((180 << HSV_SHIFT) / (6.0 * i));
}

/** Returns `[h, s, v]` for one sRGB pixel, `rgb` ordered R, G, B. */
function rgbToHsvPixel(r, g, b) {
  let v = b;
  let vmin = b;
  if (g > v) v = g;
  if (r > v) v = r;
  if (g < vmin) vmin = g;
  if (r < vmin) vmin = r;

  const diff = v - vmin;
  const vr = v === r ? -1 : 0;
  const vg = v === g ? -1 : 0;
  const s = (diff * SDIV_TABLE[v] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
  let h =
    (vr & (g - b)) +
    (~vr & ((vg & (b - r + 2 * diff)) + (~vg & (r - g + 4 * diff))));
  h = (h * HDIV_TABLE[diff] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
  if (h < 0) h += 180;
  return [h, s, v];
}

/** OpenCV `cvtColor(..., COLOR_RGB2HSV)` over the whole frame as an `Int32Array`. */
export function cvtColorRGB2HSV(rgb, width, height) {
  const n = width * height;
  const out = new Int32Array(n * 3);
  for (let p = 0; p < n; p++) {
    const hsv = rgbToHsvPixel(rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2]);
    out[p * 3] = hsv[0];
    out[p * 3 + 1] = hsv[1];
    out[p * 3 + 2] = hsv[2];
  }
  return out;
}

/** Median of one HSV channel over the selection, or the whole frame for `null`. */
function hsvMedian(hsv, channel, idx, n) {
  const count = idx === null ? n : idx.length;
  if (count === 0) return Number.NaN;
  const values = new Float64Array(count);
  if (idx === null) {
    for (let p = 0; p < n; p++) values[p] = hsv[p * 3 + channel];
  } else {
    for (let k = 0; k < idx.length; k++) values[k] = hsv[idx[k] * 3 + channel];
  }
  return median(values);
}

/**
 * OpenCV's Otsu threshold for a 256-bin histogram, transcribed from `getThreshVal_Otsu`.
 *
 * The unusual `mu1 *= q1` before the running mass update is OpenCV's, not a typo; it
 * cancels the previous iteration's division. Returns a bin index (0-255).
 */
function otsuThreshold(histogram, total) {
  const scale = 1.0 / total;
  let mu = 0.0;
  for (let i = 0; i < 256; i++) mu += i * histogram[i];
  mu *= scale;

  let mu1 = 0.0;
  let q1 = 0.0;
  let maxSigma = 0.0;
  let maxVal = 0.0;
  for (let i = 0; i < 256; i++) {
    const pI = histogram[i] * scale;
    mu1 *= q1;
    q1 += pI;
    const q2 = 1.0 - q1;
    if (Math.min(q1, q2) < FLT_EPSILON || Math.max(q1, q2) > 1.0 - FLT_EPSILON) {
      continue;
    }
    mu1 = (mu1 + i * pI) / q1;
    const mu2 = (mu - q1 * mu1) / q2;
    const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > maxSigma) {
      maxSigma = sigma;
      maxVal = i;
    }
  }
  return maxVal;
}

/**
 * `mean(a* | Otsu vessels) - mean(a* | background)` over the **full frame** a* channel.
 *
 * The mask is intentionally not consulted: Otsu segments vessels across the whole
 * exposure-normalised frame, because a tight ROI would hide the surrounding tissue the
 * contrast is measured against. Min-max normalisation and the uint8 truncation mirror
 * `cv2.normalize(..., NORM_MINMAX).astype(np.uint8)`; a flat channel normalises to all
 * zeros exactly as OpenCV does. NaN when either class is empty.
 */
export function otsuVesselRedness(lab, width, height) {
  const n = width * height;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let p = 0; p < n; p++) {
    const a = lab[p * 3 + 1];
    if (a < min) min = a;
    if (a > max) max = a;
  }

  const flat = max === min;
  const scale = flat ? 0.0 : 255.0 / (max - min);
  const shift = flat ? 0.0 : 0.0 - min * scale;

  const histogram = new Float64Array(256);
  const normalised = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    const value = flat ? 0.0 : lab[p * 3 + 1] * scale + shift;
    const bin = toUint8(value);
    normalised[p] = bin;
    histogram[bin] += 1;
  }

  const threshold = otsuThreshold(histogram, n);
  let vesselSum = 0.0;
  let vesselCount = 0;
  let backgroundSum = 0.0;
  let backgroundCount = 0;
  for (let p = 0; p < n; p++) {
    const a = lab[p * 3 + 1];
    if (normalised[p] > threshold) {
      vesselSum += a;
      vesselCount += 1;
    } else {
      backgroundSum += a;
      backgroundCount += 1;
    }
  }
  if (vesselCount === 0 || backgroundCount === 0) return Number.NaN;
  return vesselSum / vesselCount - backgroundSum / backgroundCount;
}

/**
 * The ten measured features over an ROI or the whole frame.
 *
 * `rgb` is the frame the features are measured *on* (the balanced uint8 frame when the
 * caller balanced); the mask only selects. Order of the returned fields matches
 * `ColourFeatureVector`.
 */
export function extractColourFeatures(rgb, mask, width, height) {
  const n = width * height;
  const lab = rgbToLab(rgb, width, height);
  const idx = maskIndices(mask, n);
  const labMean = meanChannels(lab, idx, n);

  const count = idx === null ? n : idx.length;
  const roiSrgb = new Float64Array(count * 3);
  if (idx === null) {
    for (let p = 0; p < n; p++) {
      roiSrgb[p * 3] = rgb[p * 3];
      roiSrgb[p * 3 + 1] = rgb[p * 3 + 1];
      roiSrgb[p * 3 + 2] = rgb[p * 3 + 2];
    }
  } else {
    for (let k = 0; k < idx.length; k++) {
      const p = idx[k];
      roiSrgb[k * 3] = rgb[p * 3];
      roiSrgb[k * 3 + 1] = rgb[p * 3 + 1];
      roiSrgb[k * 3 + 2] = rgb[p * 3 + 2];
    }
  }

  const hsv = cvtColorRGB2HSV(rgb, width, height);

  return {
    lab_l: labMean[0],
    lab_a: labMean[1],
    lab_b: labMean[2],
    redness_ratio: rednessRatio(labMean[1], labMean[2]),
    erythema_index: erythemaIndex(labMean[1], labMean[2]),
    high_hue_ratio: highHueRatio(roiSrgb),
    otsu_vessel_redness: otsuVesselRedness(lab, width, height),
    hsv_hue: hsvMedian(hsv, 0, idx, n),
    hsv_sat: hsvMedian(hsv, 1, idx, n),
    hsv_val: hsvMedian(hsv, 2, idx, n),
  };
}

/**
 * Grey-world exposure normalisation, returning float `[0, 1]`.
 *
 * The illuminant is estimated from the pixels **outside** the mask, per channel, and the
 * frame is divided by it. Excluding the ROI is the substantive choice: a grey-world
 * estimate taken over a red conjunctiva is pulled toward red by the very signal being
 * measured, so the surrounding tissue stands in for the illumination instead. When the
 * mask covers everything, nothing survives it, or it is the wrong length, the whole
 * frame is used -- a biased estimate beats dividing by an invented constant.
 *
 * Brightness is restored with a single scalar from the frame means, never per channel,
 * or the correction just applied would be undone. Returns a `Float64Array` of length
 * `width * height * 3`.
 */
export function whiteBalance(rgb, mask, width, height) {
  const n = width * height;
  const size = n * 3;

  const shapeMatches = mask !== null && mask !== undefined && mask.length === n;
  let outsideCount = 0;
  if (shapeMatches) {
    for (let i = 0; i < n; i++) {
      if (mask[i] === 0) outsideCount += 1;
    }
  }
  const useOutside = shapeMatches && outsideCount > 0;
  const referenceCount = useOutside ? outsideCount : n;

  const referenceSum = [0.0, 0.0, 0.0];
  if (useOutside) {
    for (let i = 0; i < n; i++) {
      if (mask[i] === 0) {
        referenceSum[0] += rgb[i * 3];
        referenceSum[1] += rgb[i * 3 + 1];
        referenceSum[2] += rgb[i * 3 + 2];
      }
    }
  } else {
    for (let i = 0; i < size; i++) referenceSum[i % 3] += rgb[i];
  }

  const illuminant = [
    referenceSum[0] / referenceCount,
    referenceSum[1] / referenceCount,
    referenceSum[2] / referenceCount,
  ];
  for (let c = 0; c < 3; c++) {
    if (!(illuminant[c] > GREY_WORLD_EPS)) illuminant[c] = 1.0;
  }

  const values = new Float64Array(size);
  let valuesSum = 0.0;
  for (let i = 0; i < size; i++) {
    const v = rgb[i] / 255.0;
    values[i] = v;
    valuesSum += v;
  }
  const valuesMean = valuesSum / size;

  const balanced = new Float64Array(size);
  let balancedSum = 0.0;
  for (let i = 0; i < size; i++) {
    const b = values[i] / illuminant[i % 3];
    balanced[i] = b;
    balancedSum += b;
  }
  const balancedMean = balancedSum / size;
  const scale = balancedMean > GREY_WORLD_EPS ? valuesMean / balancedMean : 1.0;

  const out = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    const value = balanced[i] * scale;
    out[i] = value < 0.0 ? 0.0 : value > 1.0 ? 1.0 : value;
  }
  return out;
}

/** `whiteBalance` quantised back to uint8, as `balanced_rgb` does. */
export function balancedRgb(rgb, mask, width, height) {
  const balanced = whiteBalance(rgb, mask, width, height);
  const out = new Uint8Array(balanced.length);
  for (let i = 0; i < balanced.length; i++) out[i] = roundHalfEven(balanced[i] * 255.0);
  return out;
}

/**
 * The 13-vector in `COLOUR_COLUMNS` order for one frame.
 *
 * When `balance` is true the ten measured features come from the balanced uint8 frame;
 * the three `roi_*` values are **always** read from the original frame, because they are
 * the raw quantities the exposure result is reported against. `roi_mean` is the mean over
 * the analysed pixels (`rgb[mask]`, or the whole frame when there is no mask), divided by
 * 255 as in `extract_colour_rows`.
 */
export function extractColourRow(rgb, mask, balance, width, height) {
  const n = width * height;
  const measured = balance
    ? extractColourFeatures(balancedRgb(rgb, mask, width, height), mask, width, height)
    : extractColourFeatures(rgb, mask, width, height);

  const idx = maskIndices(mask, n);
  const count = idx === null ? n : idx.length;
  let rSum = 0.0;
  let gSum = 0.0;
  let bSum = 0.0;
  if (count > 0) {
    if (idx === null) {
      for (let p = 0; p < n; p++) {
        rSum += rgb[p * 3];
        gSum += rgb[p * 3 + 1];
        bSum += rgb[p * 3 + 2];
      }
    } else {
      for (let k = 0; k < idx.length; k++) {
        const p = idx[k];
        rSum += rgb[p * 3];
        gSum += rgb[p * 3 + 1];
        bSum += rgb[p * 3 + 2];
      }
    }
  }
  const roiR = count === 0 ? Number.NaN : rSum / count / 255.0;
  const roiG = count === 0 ? Number.NaN : gSum / count / 255.0;
  const roiB = count === 0 ? Number.NaN : bSum / count / 255.0;

  return [
    measured.lab_l,
    measured.lab_a,
    measured.lab_b,
    measured.redness_ratio,
    measured.erythema_index,
    measured.high_hue_ratio,
    measured.otsu_vessel_redness,
    measured.hsv_hue,
    measured.hsv_sat,
    measured.hsv_val,
    roiR,
    roiG,
    roiB,
  ];
}

/** The column order this module emits, mirrored from `COLOUR_COLUMNS`. */
export const COLOUR_COLUMNS = [
  "lab_l",
  "lab_a",
  "lab_b",
  "redness_ratio",
  "erythema_index",
  "high_hue_ratio",
  "otsu_vessel_redness",
  "hsv_hue",
  "hsv_sat",
  "hsv_val",
  "roi_r",
  "roi_g",
  "roi_b",
];

/** Mean of the selected values, or `NaN` when nothing is selected. */
export function maskedMean(values, mask) {
  const n = values.length;
  const idx = maskIndices(mask, n);
  const count = idx === null ? n : idx.length;
  if (count === 0) return Number.NaN;
  let sum = 0.0;
  if (idx === null) {
    for (let i = 0; i < n; i++) sum += values[i];
  } else {
    for (let k = 0; k < idx.length; k++) sum += values[idx[k]];
  }
  return sum / count;
}
