/* Shared helpers for the two edge endpoints.
 *
 * The contract is deliberately small: JSON in, no identifiers, no cookies. A
 * request that is malformed is answered with a stable error code and a message
 * that is safe to show a user; a request that is well formed is answered with
 * the minimum and then forgotten. Nothing here logs a body.
 */

/** Hard ceiling on a request body. A screening event is a few hundred bytes. */
export const MAX_BODY_BYTES = 1024;

const json = (status, body, headers = {}) =>
  new Response(body === null ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...headers },
  });

/** Standard error shape, per the API contract. */
export function jsonError(code, message, status, details, headers) {
  const body = { error: { code, message } };
  if (details) body.error.details = details;
  return json(status, body, headers);
}

/** The success shape used where the contract returns a resource. */
export function jsonData(data, status) {
  return json(status, { data });
}

/** 204 for telemetry: recorded, but there is no resource to hand back. */
export function noContent(headers) {
  return new Response(null, { status: 204, headers });
}

/**
 * Enforce the transport rules and parse the body.
 * @returns {Promise<{ok: true, body: object} | {ok: false, response: Response}>}
 */
export async function readJson(request) {
  const contentType = request.headers.get("Content-Type") || "";
  if (!contentType.toLowerCase().includes("application/json")) {
    return {
      ok: false,
      response: jsonError("UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/json.", 415),
    };
  }

  const declared = Number(request.headers.get("Content-Length") || "0");
  if (declared > MAX_BODY_BYTES) {
    return {
      ok: false,
      response: jsonError("PAYLOAD_TOO_LARGE", "Body must be 1024 bytes or fewer.", 413),
    };
  }

  let raw;
  try {
    raw = await request.text();
  } catch {
    return { ok: false, response: jsonError("INVALID_BODY", "Body could not be read.", 400) };
  }
  if (new TextEncoder().encode(raw).length > MAX_BODY_BYTES) {
    return {
      ok: false,
      response: jsonError("PAYLOAD_TOO_LARGE", "Body must be 1024 bytes or fewer.", 413),
    };
  }

  let body;
  try {
    body = JSON.parse(raw);
  } catch {
    return { ok: false, response: jsonError("INVALID_BODY", "Body must be valid JSON.", 400) };
  }
  if (body === null || typeof body !== "object" || Array.isArray(body)) {
    return { ok: false, response: jsonError("INVALID_BODY", "Body must be a JSON object.", 400) };
  }
  return { ok: true, body };
}

export function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/** Bounded number, or `undefined` when absent. Throws the caller's error. */
export function optionalNumber(body, field, min, max, fail) {
  const value = body[field];
  if (value === undefined) return undefined;
  if (!isFiniteNumber(value) || value < min || value > max) {
    fail(field, `must be a number between ${min} and ${max}`);
  }
  return value;
}

/* ------------------------------------------------------------------ *
 * Rate limiting. Best effort by construction: an isolate's Map is local
 * to that isolate, so the limit is a speed bump, not a wall. That is
 * acceptable because the endpoint holds nothing worth attacking, and is
 * documented rather than pretended away.
 * ------------------------------------------------------------------ */

const buckets = new Map();

export function rateLimit(key, limit, windowMs) {
  const now = Date.now();
  const bucket = buckets.get(key);
  if (!bucket || now > bucket.reset) {
    buckets.set(key, { count: 1, reset: now + windowMs });
    if (buckets.size > 5000) {
      for (const [k, v] of buckets) if (now > v.reset) buckets.delete(k);
    }
    return true;
  }
  bucket.count += 1;
  return bucket.count <= limit;
}

/** ISO-8601 start of the current UTC day, matching stored created_at strings. */
export function startOfUtcDay(now = new Date()) {
  return `${now.toISOString().slice(0, 10)}T00:00:00.000Z`;
}

/** How many rows have been written since midnight UTC, for the daily cap. */
export async function rowsToday(env, table) {
  const row = await env.DB.prepare(
    `SELECT COUNT(*) AS n FROM ${table} WHERE created_at >= ?1`,
  )
    .bind(startOfUtcDay())
    .first();
  return row && Number.isFinite(row.n) ? row.n : 0;
}
