/* Exercises the client telemetry module against the same fields the edge
 * endpoint accepts, and against a fetch that throws. tests/test_telemetry_js.py
 * asserts on the printed JSON.
 */
import { buildPayload, reportScreening } from "../app/web/src/telemetry.js";

const results = [];
const record = (name, value) => results.push({ name, ...value });

record("valid", {
  payload: buildPayload({
    model_id: "efficnet-b0-v4-a1b2c3d4",
    band: "within_range",
    hb_hat: 11.4,
    sigma: 1.2,
    quality: 0.86,
    latency_ms: 142,
  }),
});

record("drops_unknown", {
  payload: buildPayload({ model_id: "m-v1", band: "mild", hb_hat: 10, evil: "x", image: {}, nested: { a: 1 } }),
});

record("bad_band", { payload: buildPayload({ model_id: "m-v1", band: "cured" }) });
record("bad_model", { payload: buildPayload({ model_id: "Bad ID", band: "mild" }) });
record("out_of_range", { payload: buildPayload({ model_id: "m-v1", band: "mild", hb_hat: 40 }) });

const throwing = async () => {
  throw new Error("network down");
};
record("swallows_failure", { result: await reportScreening({ model_id: "m-v1", band: "mild" }, { fetchImpl: throwing }) });

let seen = null;
const capturing = async (url, init) => {
  seen = { url, method: init.method, contentType: init.headers["Content-Type"], body: JSON.parse(init.body) };
  return { ok: true, status: 204 };
};
const good = await reportScreening({ model_id: "m-v1", band: "mild", hb_hat: 10.1 }, { fetchImpl: capturing });
record("request_seen", { seen, result: good });

process.stdout.write(JSON.stringify(results));
