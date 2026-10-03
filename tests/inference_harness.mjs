/* Runs the browser preprocessing over deterministic frames and prints the
 * results. tests/test_inference_js.py regenerates the same frames in Python and
 * compares against cv2.resize, which is what the trainer used.
 *
 * The input is rebuilt on the Python side from the same LCG, so only the output
 * crosses the boundary and the fixture is not pinned into the repository.
 */
import { resizeBilinear, buildTensor, roiToRect, decodeOutputs } from "../app/web/src/inference.js";

function lcgTopByte(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s >>> 24;
  };
}

function makeRgba(width, height, seed) {
  const next = lcgTopByte(seed);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    rgba[i * 4] = next();
    rgba[i * 4 + 1] = next();
    rgba[i * 4 + 2] = next();
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

const SIZE = 224;
const PLANE = SIZE * SIZE;
const SAMPLE = [0, 1, PLANE, PLANE + 1, 2 * PLANE, 2 * PLANE + 1, PLANE - 1];

const resizeCases = [
  { name: "upsample", w: 37, h: 53, seed: 12345 },
  { name: "downsample", w: 300, h: 260, seed: 6789 },
].map(({ name, w, h, seed }) => {
  const rgba = makeRgba(w, h, seed);
  const resized = resizeBilinear(rgba, w, h, SIZE);
  const tensor = buildTensor(rgba, w, h, SIZE);

  let tensorLayoutError = 0;
  for (const i of SAMPLE) {
    const channel = Math.floor(i / PLANE);
    const position = i % PLANE;
    tensorLayoutError = Math.max(
      tensorLayoutError,
      Math.abs(tensor[i] - resized[position * 3 + channel] / 255),
    );
  }

  return {
    name,
    w,
    h,
    seed,
    resized: Array.from(resized),
    tensorSamples: SAMPLE.map((i) => [i, tensor[i]]),
    tensorLayoutError,
  };
});

const fakeOutputs = {
  hb_gdl: { data: [11.5] },
  sigma_gdl: { data: [0.9] },
  bin_probs: { data: [0, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0, 0, 0, 0] },
};

process.stdout.write(
  JSON.stringify({
    size: SIZE,
    resizeCases,
    roi: {
      inside: roiToRect({ cx: 10, cy: 20, rx: 5, ry: 8 }, 100, 100),
      clipped: roiToRect({ cx: 2, cy: 2, rx: 5, ry: 5 }, 100, 100),
      onEdge: roiToRect({ cx: 98, cy: 50, rx: 10, ry: 4 }, 100, 100),
    },
    decode: {
      withResidual: decodeOutputs(fakeOutputs, { residual_sigma: 1.4 }),
      fallback: decodeOutputs(fakeOutputs, {}),
    },
  }),
);
