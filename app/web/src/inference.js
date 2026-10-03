/* Inference seam: the one place the app touches the model.
 *
 * The graph is exported by `hemolux export` and owns colour, not geometry
 * (brain/08_UI_SPEC.md, docs/DEPLOY.md). Concretely the ONNX module takes a
 * float32 `(N, 3, 224, 224)` tensor in `[0, 1]`, applies the ImageNet
 * normalisation as constants *inside* the graph, and returns the haemoglobin
 * estimate, its predictive standard deviation and the ordinal bin
 * probabilities. So this file does the only three things left to do in
 * JavaScript:
 *
 *   1. crop the guided ROI out of the frame,
 *   2. resize it to 224x224 the way the trainer did (cv2.INTER_LINEAR),
 *   3. divide by 255 and lay it out NCHW.
 *
 * Steps 2 and 3 are pure functions (`resizeBilinear`, `buildTensor`) with a
 * Python parity test, because a silent divergence here would change every
 * prediction without throwing.
 *
 * The model is gated on its metadata. `model.json` is written from the run
 * that produced the graph and carries `validated: true` plus the residual
 * spread measured on the validation fold; without it this module reports
 * `unavailable` and the UI says so, rather than showing a number no report
 * backs.
 */

export const IMAGE_SIZE = 224;

/** Where the deploy step stages the exported graph and its metadata. */
export const MODEL_URL = "/models/hemolux_screen_224.onnx";
export const MODEL_META_URL = "/models/model.json";
export const MODEL_INPUT_NAME = "image";

/** Mirrors `PREPROCESS_SPEC` in `hemolux.data.dataset`. `scale` is applied here;
 * `mean`/`std` are applied by the graph, which is the whole reason the browser
 * can hand over raw pixels. */
export const PREPROCESS = {
  scale: 1 / 255,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  size: IMAGE_SIZE,
};

const Z_95 = 1.959963984540054;

/** Pixel bounding box of the guided ROI, clamped to the frame. */
export function roiToRect(roi, width, height) {
  const x = Math.max(0, Math.floor(roi.cx - roi.rx));
  const y = Math.max(0, Math.floor(roi.cy - roi.ry));
  const right = Math.min(width, Math.ceil(roi.cx + roi.rx));
  const bottom = Math.min(height, Math.ceil(roi.cy + roi.ry));
  return { x, y, w: Math.max(1, right - x), h: Math.max(1, bottom - y) };
}

/**
 * Bilinear resize of interleaved RGBA to `size`x`size` RGB.
 *
 * Uses the half-pixel-centre mapping and edge replication that OpenCV's
 * `INTER_LINEAR` uses, so the result matches `cv2.resize` the trainer applied.
 * The output is an integer image because the Python pipeline resizes to uint8
 * before scaling; matching that rounding is part of the parity, not an
 * accident.
 */
export function resizeBilinear(rgba, srcW, srcH, size = IMAGE_SIZE) {
  const out = new Uint8ClampedArray(size * size * 3);
  const scaleX = srcW / size;
  const scaleY = srcH / size;

  for (let dy = 0; dy < size; dy += 1) {
    const sy = (dy + 0.5) * scaleY - 0.5;
    const y0raw = Math.floor(sy);
    const fy = sy - y0raw;
    const y0 = Math.min(srcH - 1, Math.max(0, y0raw));
    const y1 = Math.min(srcH - 1, Math.max(0, y0raw + 1));

    for (let dx = 0; dx < size; dx += 1) {
      const sx = (dx + 0.5) * scaleX - 0.5;
      const x0raw = Math.floor(sx);
      const fx = sx - x0raw;
      const x0 = Math.min(srcW - 1, Math.max(0, x0raw));
      const x1 = Math.min(srcW - 1, Math.max(0, x0raw + 1));

      const p00 = (y0 * srcW + x0) * 4;
      const p01 = (y0 * srcW + x1) * 4;
      const p10 = (y1 * srcW + x0) * 4;
      const p11 = (y1 * srcW + x1) * 4;
      const o = (dy * size + dx) * 3;

      for (let c = 0; c < 3; c += 1) {
        const top = rgba[p00 + c] * (1 - fx) + rgba[p01 + c] * fx;
        const bottom = rgba[p10 + c] * (1 - fx) + rgba[p11 + c] * fx;
        out[o + c] = top * (1 - fy) + bottom * fy;
      }
    }
  }
  return out;
}

/** Resize, scale to `[0, 1]` and lay out `(3, size, size)` channel-first. */
export function buildTensor(rgba, srcW, srcH, size = IMAGE_SIZE) {
  const rgb = resizeBilinear(rgba, srcW, srcH, size);
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i += 1) {
    out[i] = rgb[i * 3] * PREPROCESS.scale;
    out[plane + i] = rgb[i * 3 + 1] * PREPROCESS.scale;
    out[2 * plane + i] = rgb[i * 3 + 2] * PREPROCESS.scale;
  }
  return out;
}

/**
 * Read the graph outputs. The interval uses the residual spread the validation
 * fold measured (`meta.residual_sigma`) because that is what is calibrated;
 * the model's own posterior width is kept only as a labelled fallback and is
 * never presented as the validated interval on its own.
 */
export function decodeOutputs(outputs, meta = {}) {
  const hb = Number(outputs.hb_gdl.data[0]);
  const posteriorSigma = Number(outputs.sigma_gdl.data[0]);
  const probs = Array.from(outputs.bin_probs.data, Number);
  const sigma = Number.isFinite(meta.residual_sigma) ? Number(meta.residual_sigma) : posteriorSigma;
  return { hb, sigma, posteriorSigma, probs, lo: hb - Z_95 * sigma, hi: hb + Z_95 * sigma };
}

/* ------------------------------------------------------------------ *
 * Runtime + model loading. `ort` is a UMD global injected from the
 * self-hosted runtime (never a CDN), so the CSP can stay same-origin.
 * ------------------------------------------------------------------ */

const RUNTIME_URL = "/vendor/ort/ort.wasm.min.js";
const RUNTIME_BASE = "/vendor/ort/";

let runtimePromise = null;

function configureRuntime(ort) {
  // Self-hosted, so the wasm and its worker are resolved from our own origin
  // rather than a CDN. Threads are capped because more than a few rarely help a
  // model this small and each one costs a worker.
  ort.env.wasm.wasmPaths = RUNTIME_BASE;
  const cores = typeof navigator === "undefined" ? 1 : navigator.hardwareConcurrency || 1;
  ort.env.wasm.numThreads = Math.min(4, cores);
  return ort;
}

function ensureRuntime() {
  if (globalThis.ort && globalThis.ort.InferenceSession) {
    return Promise.resolve(configureRuntime(globalThis.ort));
  }
  if (runtimePromise) return runtimePromise;
  runtimePromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = RUNTIME_URL;
    script.async = true;
    script.onload = () =>
      globalThis.ort && globalThis.ort.InferenceSession
        ? resolve(configureRuntime(globalThis.ort))
        : reject(new Error("onnxruntime-web loaded but exposed no InferenceSession"));
    script.onerror = () => reject(new Error("onnxruntime-web failed to load"));
    document.head.append(script);
  });
  return runtimePromise;
}

let modelPromise = null;

/**
 * @returns {Promise<{ort: object, session: object, meta: object} | null>}
 */
export function loadModel() {
  if (!modelPromise) modelPromise = createModel();
  return modelPromise;
}

async function createModel() {
  try {
    const metaResponse = await fetch(MODEL_META_URL, { cache: "force-cache" });
    if (!metaResponse.ok) return null;
    const meta = await metaResponse.json();
    // A graph without a validation report behind it is not a screening model.
    if (!meta || meta.validated !== true) return null;

    const ort = await ensureRuntime();
    const graphResponse = await fetch(MODEL_URL, { cache: "force-cache" });
    if (!graphResponse.ok) return null;
    const bytes = await graphResponse.arrayBuffer();
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    return { ort, session, meta };
  } catch {
    return null;
  }
}

/**
 * @param {{ canvas: HTMLCanvasElement, roi: { cx: number, cy: number, rx: number, ry: number } }} input
 * @returns {Promise<{ status: "unavailable" | "ok", hb?: number, lo?: number, hi?: number }>}
 */
export async function analyse({ canvas, roi }) {
  const model = await loadModel();
  if (!model) return { status: "unavailable" };
  const { ort, session, meta } = model;

  const size = Number(meta.input_size) || IMAGE_SIZE;
  const rect = roiToRect(roi, canvas.width, canvas.height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const image = context.getImageData(rect.x, rect.y, rect.w, rect.h);
  const tensor = buildTensor(image.data, rect.w, rect.h, size);

  const input = new ort.Tensor("float32", tensor, [1, 3, size, size]);
  const outputs = await session.run({ [meta.input_name || MODEL_INPUT_NAME]: input });
  const decoded = decodeOutputs(outputs, meta);

  return {
    status: "ok",
    model_id: meta.model_id || "hemolux-screen",
    quality: Number.isFinite(meta.quality) ? Number(meta.quality) : 1,
    ...decoded,
  };
}
