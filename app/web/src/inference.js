/* Inference seam.
 *
 * The model is chosen by `hemolux sweep` and exported to ONNX. Until that
 * artefact exists this module reports `unavailable`; it never fabricates a
 * number (brain/08_UI_SPEC.md §9 rule 6: every number is real or absent).
 *
 * When the winner is known, `loadModel()` fetches the ONNX file into Cache
 * Storage once, `analyse()` runs the shared preprocessing spec, and this file
 * is the only place that touches the network for model bytes — never the frame.
 *
 * The predicted interval is not invented here either: it comes from the
 * residual spread measured on the validation fold and shipped as metadata
 * alongside the ONNX file.
 */

export const MODEL_READY = false;

let session = null;

export async function loadModel() {
  if (!MODEL_READY) return null;
  // onnxruntime-web session creation goes here once the export exists.
  return session;
}

/**
 * @param {{ canvas: HTMLCanvasElement, roi: { cx: number, cy: number, rx: number, ry: number } }} _input
 * @returns {Promise<{ status: "unavailable" | "ok" | "abstain", hb?: number, lo?: number, hi?: number }>}
 */
export async function analyse(_input) {
  if (!MODEL_READY) return { status: "unavailable" };
  return { status: "unavailable" };
}
