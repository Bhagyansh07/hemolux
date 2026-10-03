#!/usr/bin/env node
/* Vendor the on-device inference runtime.
 *
 * The app must not call a CDN: the CSP is same-origin and the privacy claim is
 * that nothing but non-identifying telemetry leaves the device, which is hard
 * to defend if the page pulls a script from a third party at run time. So the
 * one browser dependency is copied into app/web/vendor/ort/ at deploy time and
 * served by Cloudflare Pages like any other static file.
 *
 * Only the three files the wasm backend needs are copied, not the ~80 MB dist:
 *
 *   ort.wasm.min.js              the UMD loader (global `ort`, wasm-only)
 *   ort-wasm-simd-threaded.mjs   the module worker
 *   ort-wasm-simd-threaded.wasm  the compute kernel
 *
 * The destination is git-ignored. Re-run after changing ORT_VERSION.
 *
 * Usage: node scripts/vendor_runtime.mjs
 */
import { execFileSync } from "node:child_process";
import { cp, mkdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** Pinned to the same version as the Python `onnxruntime` used to export. */
const ORT_VERSION = "1.30.0";
const FILES = ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const destination = join(root, "app", "web", "vendor", "ort");
const staging = join(tmpdir(), `hemolux-ort-${ORT_VERSION}`);
const packageDir = join(staging, "node_modules", "onnxruntime-web", "dist");

if (!existsSync(join(packageDir, "package.json")) && !existsSync(join(staging, "node_modules", "onnxruntime-web", "package.json"))) {
  await mkdir(staging, { recursive: true });
  console.log(`vendoring: npm install onnxruntime-web@${ORT_VERSION} into ${staging}`);
  // A single command string with shell:true: passing an args array to a shell
  // spawn is deprecated on Node 24, and the only variable here is a pinned
  // version constant.
  execFileSync(
    `npm install onnxruntime-web@${ORT_VERSION} --no-save --no-audit --no-fund --loglevel=error`,
    { cwd: staging, stdio: "inherit", shell: true },
  );
}

await mkdir(destination, { recursive: true });
for (const name of FILES) {
  const source = join(packageDir, name);
  if (!existsSync(source)) {
    throw new Error(`onnxruntime-web@${ORT_VERSION} did not ship ${name}; the file list changed`);
  }
  await cp(source, join(destination, name));
  const { size } = await stat(join(destination, name));
  console.log(`  ${(size / 1024).toFixed(0).padStart(7)} KB  vendor/ort/${name}`);
}
console.log(`vendored: onnxruntime-web@${ORT_VERSION} -> app/web/vendor/ort/ (git-ignored)`);
