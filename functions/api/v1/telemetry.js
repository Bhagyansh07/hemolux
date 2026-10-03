/* POST /api/v1/telemetry — a screening happened.
 *
 * No pixels, no identifiers, no free text. An unknown key is dropped rather
 * than stored, so a client bug cannot silently widen the schema. Telemetry is
 * best effort: the client ignores failures and always shows the estimate.
 */
import { jsonError, readJson, optionalNumber, rateLimit, rowsToday } from "./_shared.js";

const MAX_DAILY = 20000;
const BANDS = new Set(["severe", "moderate", "mild", "within_range", "abstained"]);
const MODEL_ID = /^[a-z0-9._-]+$/;

export async function onRequestPost({ request, env }) {
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  if (!rateLimit(`telemetry:${ip}`, 30, 60_000)) {
    return jsonError("RATE_LIMITED", "Too many requests.", 429, undefined, { "Retry-After": "60" });
  }

  const parsed = await readJson(request);
  if (!parsed.ok) return parsed.response;
  const body = parsed.body;

  let failure = null;
  const fail = (field, message) => {
    failure = jsonError("INVALID_BODY", `${field} ${message}`, 400, { field });
    throw new Error("invalid");
  };

  let model_id;
  let band;
  let hb_hat;
  let sigma;
  let quality;
  let latency_ms;
  try {
    model_id = body.model_id;
    if (typeof model_id !== "string" || model_id.length === 0 || model_id.length > 64 || !MODEL_ID.test(model_id)) {
      fail("model_id", "must match [a-z0-9._-] and be 64 characters or fewer");
    }
    band = body.band;
    if (typeof band !== "string" || !BANDS.has(band)) {
      fail("band", "must be one of severe, moderate, mild, within_range, abstained");
    }
    hb_hat = optionalNumber(body, "hb_hat", 3, 25, fail);
    sigma = optionalNumber(body, "sigma", 0, 10, fail);
    quality = optionalNumber(body, "quality", 0, 1, fail);
    latency_ms = optionalNumber(body, "latency_ms", 0, 60000, fail);
    if (latency_ms !== undefined && !Number.isInteger(latency_ms)) {
      fail("latency_ms", "must be an integer");
    }
  } catch {
    return failure;
  }

  try {
    if ((await rowsToday(env, "telemetry")) >= MAX_DAILY) {
      return jsonError("RATE_LIMITED", "Daily telemetry cap reached.", 429, undefined, {
        "Retry-After": "3600",
      });
    }
    await env.DB.prepare(
      "INSERT INTO telemetry (created_at, model_id, band, hb_hat, sigma, quality, latency_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )
      .bind(new Date().toISOString(), model_id, band, hb_hat ?? null, sigma ?? null, quality ?? null, latency_ms ?? null)
      .run();
  } catch {
    return jsonError("INTERNAL", "Telemetry could not be recorded.", 500);
  }

  return new Response(null, { status: 204 });
}

export async function onRequest() {
  return jsonError("METHOD_NOT_ALLOWED", "Use POST.", 405, undefined, { Allow: "POST" });
}
