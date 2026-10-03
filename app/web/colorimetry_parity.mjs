/**
 * Node runner for the Python <-> browser colourimetry parity test.
 *
 * Reads a JSON file of frame cases from `process.argv[2]`, runs `extractColourRow` on
 * each, and writes the resulting 13-vectors to stdout as JSON. NaN cannot be represented
 * in JSON, so it is serialised as `null`; `tests/test_colourimetry_js.py` reads the
 * `null`s back as NaN. This file has no logic of its own on purpose: it must not be
 * possible for the test to pass because the runner corrected something.
 */

import { readFileSync } from "node:fs";
import process from "node:process";

import { COLOUR_COLUMNS, extractColourRow } from "./colorimetry.js";

const path = process.argv[2];
if (!path) {
  process.stderr.write("usage: node colorimetry_parity.mjs <cases.json>\n");
  process.exit(2);
}

const payload = JSON.parse(readFileSync(path, "utf8"));

const rows = payload.cases.map((frame) => {
  const rgb = Uint8Array.from(frame.rgb);
  const mask = frame.mask === null ? null : Uint8Array.from(frame.mask);
  return extractColourRow(rgb, mask, frame.balance, frame.width, frame.height);
});

// JSON.stringify turns NaN into null, which the Python side maps back to NaN.
process.stdout.write(JSON.stringify({ columns: COLOUR_COLUMNS, rows }));
