/* Drives the two edge handlers against a mock D1 and prints the observed
 * status/body/inserts as JSON. tests/test_workers.py asserts on that output.
 *
 * The handlers are the shipped modules, imported by relative path, so a change
 * to validation or the insert column order fails a test rather than a user.
 * Node >= 18 supplies Request/Response/TextEncoder globally.
 */
import { onRequestPost as telemetryPost, onRequest as telemetryAny } from "../functions/api/v1/telemetry.js";
import { onRequestPost as feedbackPost, onRequest as feedbackAny } from "../functions/api/v1/feedback.js";

const ENDPOINT = "https://hemolux.in/api/v1/telemetry";

function makeEnv({ count = 0 } = {}) {
  const inserts = [];
  return {
    inserts,
    DB: {
      prepare(sql) {
        const stmt = {
          _args: null,
          bind(...args) {
            stmt._args = args;
            return stmt;
          },
          async run() {
            inserts.push({ sql, args: stmt._args });
            return { success: true };
          },
          async first() {
            return { n: count };
          },
        };
        return stmt;
      },
    },
  };
}

function request(body, ip, contentType = "application/json") {
  return new Request(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": contentType, "CF-Connecting-IP": ip },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

async function call(handler, req, env) {
  const res = await handler({ request: req, env, params: {} });
  const text = await res.text();
  return {
    status: res.status,
    retryAfter: res.headers.get("Retry-After"),
    allow: res.headers.get("Allow"),
    body: text ? JSON.parse(text) : null,
  };
}

const results = [];
const record = (name, value) => results.push({ name, ...value });

const valid = {
  model_id: "efficnet-b0-v4-a1b2c3d4",
  band: "within_range",
  hb_hat: 11.4,
  sigma: 1.2,
  quality: 0.86,
  latency_ms: 142,
};

// 1. A well-formed event is recorded, and extra keys are not stored.
{
  const env = makeEnv();
  const out = await call(telemetryPost, request({ ...valid, evil: "drop", nested: { a: 1 } }, "10.0.0.1"), env);
  record("valid_telemetry", { ...out, inserts: env.inserts.length, args: env.inserts[0]?.args ?? [] });
}

// 2. Transport: content type, size, method.
{
  const env = makeEnv();
  record("wrong_content_type", await call(telemetryPost, request(valid, "10.0.0.2", "text/plain"), env));
}
{
  const env = makeEnv();
  const big = "x".repeat(1100);
  record("too_large", await call(telemetryPost, request({ ...valid, pad: big }, "10.0.0.3"), env));
}
{
  const env = makeEnv();
  record("method_not_allowed", await call(telemetryAny, request(valid, "10.0.0.4"), env));
}

// 3. Validation failures name the field.
{
  const env = makeEnv();
  record("invalid_band", await call(telemetryPost, request({ ...valid, band: "cured" }, "10.0.0.5"), env));
}
{
  const env = makeEnv();
  record("invalid_model_id", await call(telemetryPost, request({ ...valid, model_id: "Bad ID" }, "10.0.0.6"), env));
}
{
  const env = makeEnv();
  record("bad_hb", await call(telemetryPost, request({ ...valid, hb_hat: 40 }, "10.0.0.7"), env));
}

// 4. The daily cap is a 429 with a retry hint, not a 500.
{
  const env = makeEnv({ count: 20000 });
  record("daily_cap", await call(telemetryPost, request(valid, "10.0.0.8"), env));
}

// 5. Feedback: accepted, validated, capped.
{
  const env = makeEnv();
  record("feedback_valid", { ...(await call(feedbackPost, request({ hb: 11.1 }, "10.0.1.1"), env)), inserts: env.inserts.length });
}
{
  const env = makeEnv();
  record("feedback_invalid", await call(feedbackPost, request({ hb: 1 }, "10.0.1.2"), env));
}
{
  const env = makeEnv({ count: 5000 });
  record("feedback_cap", await call(feedbackPost, request({ hb: 11.1 }, "10.0.1.3"), env));
}
{
  const env = makeEnv();
  record("feedback_method", await call(feedbackAny, request({ hb: 11.1 }, "10.0.1.4"), env));
}

// 6. Per-IP rate limit: 30 pass, the 31st is refused.
{
  const env = makeEnv();
  const statuses = [];
  for (let i = 0; i < 31; i += 1) {
    const out = await call(telemetryPost, request(valid, "10.9.9.9"), env);
    statuses.push(out.status);
  }
  record("rate_limit", {
    status: statuses.at(-1),
    refused: statuses.filter((s) => s === 429).length,
    allowed: statuses.filter((s) => s === 204).length,
  });
}

process.stdout.write(JSON.stringify(results));
