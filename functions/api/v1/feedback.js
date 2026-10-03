/* POST /api/v1/feedback — a user reports their confirmed laboratory Hb.
 *
 * There is deliberately no way to attach an image, a session id, or a comment.
 * Without a join key we cannot compute per-image error, and that is the trade:
 * aggregate honesty over a linkage nobody consented to.
 */
import { jsonData, jsonError, readJson, isFiniteNumber, rateLimit, rowsToday } from "./_shared.js";

const MAX_DAILY = 5000;

export async function onRequestPost({ request, env }) {
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  if (!rateLimit(`feedback:${ip}`, 30, 60_000)) {
    return jsonError("RATE_LIMITED", "Too many requests.", 429, undefined, { "Retry-After": "60" });
  }

  const parsed = await readJson(request);
  if (!parsed.ok) return parsed.response;

  const hb = parsed.body.hb;
  if (!isFiniteNumber(hb) || hb < 3 || hb > 25) {
    return jsonError("INVALID_BODY", "hb must be a number between 3 and 25", 400, { field: "hb" });
  }

  try {
    if ((await rowsToday(env, "feedback")) >= MAX_DAILY) {
      return jsonError("RATE_LIMITED", "Daily feedback cap reached.", 429, undefined, {
        "Retry-After": "3600",
      });
    }
    await env.DB.prepare("INSERT INTO feedback (created_at, hb) VALUES (?1, ?2)")
      .bind(new Date().toISOString(), hb)
      .run();
  } catch {
    return jsonError("INTERNAL", "Feedback could not be recorded.", 500);
  }

  return jsonData({ recorded: true }, 202);
}

export async function onRequest() {
  return jsonError("METHOD_NOT_ALLOWED", "Use POST.", 405, undefined, { Allow: "POST" });
}
