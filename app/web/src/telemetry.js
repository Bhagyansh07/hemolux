/* Best-effort screening telemetry.
 *
 * This module can send exactly six fields, all of them non-identifying, and it
 * cannot send an image even by mistake: the payload is built from a closed
 * allow-list here, so a caller passing the frame gets it dropped. A failure is
 * swallowed — a dead endpoint must never delay or block a health estimate.
 *
 * The server enforces the same rules independently (functions/api/v1); this is
 * the client half of one contract, not the enforcement point.
 */

const BANDS = new Set(["severe", "moderate", "mild", "within_range", "abstained"]);
const MODEL_ID = /^[a-z0-9._-]{1,64}$/;

function bounded(value, min, max) {
  return typeof value === "number" && Number.isFinite(value) && value >= min && value <= max
    ? value
    : undefined;
}

/**
 * Build the request body, or `null` when the event is not reportable.
 * Unknown keys are dropped by construction.
 */
export function buildPayload(event) {
  if (!event || typeof event !== "object") return null;
  if (typeof event.model_id !== "string" || !MODEL_ID.test(event.model_id)) return null;
  if (typeof event.band !== "string" || !BANDS.has(event.band)) return null;

  const payload = { model_id: event.model_id, band: event.band };
  const hb = bounded(event.hb_hat, 3, 25);
  const sigma = bounded(event.sigma, 0, 10);
  const quality = bounded(event.quality, 0, 1);
  const latency = bounded(event.latency_ms, 0, 60000);
  if (hb !== undefined) payload.hb_hat = hb;
  if (sigma !== undefined) payload.sigma = sigma;
  if (quality !== undefined) payload.quality = quality;
  if (latency !== undefined) payload.latency_ms = Math.trunc(latency);
  return payload;
}

/**
 * POST the event, resolving to a status object and never rejecting.
 * @returns {Promise<{ok: boolean, status?: number, reason?: string}>}
 */
export async function reportScreening(event, options = {}) {
  const fetchImpl = options.fetchImpl || (typeof fetch !== "undefined" ? fetch.bind(globalThis) : null);
  const endpoint = options.endpoint || "/api/v1/telemetry";
  const payload = buildPayload(event);
  if (!payload || !fetchImpl) return { ok: false, reason: "invalid" };

  try {
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      credentials: "omit",
      keepalive: true,
    });
    return { ok: Boolean(response && response.ok), status: response ? response.status : 0 };
  } catch {
    return { ok: false, reason: "network" };
  }
}
