#!/usr/bin/env node
/* Assemble the deployable static site.
 *
 * Cloudflare Pages serves app/web/ verbatim, so the two things that are
 * *generated* rather than authored have to be placed there first: the exported
 * model and the aggregate evidence the UI reads. Neither is committed — the
 * model is a binary and the reports are reproduced from the run that made
 * them. Routing the copy through this one script means the numbers on the
 * deployed site cannot drift from the ones EVALS.md cites.
 *
 * Usage:
 *   node scripts/build_site.mjs
 *   npx wrangler pages deploy app/web --project-name hemolux
 */
import { cp, mkdir, readdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const site = join(root, "app", "web");

async function stage(fromDir, toDir, test, label) {
  if (!existsSync(fromDir)) return [];
  const names = (await readdir(fromDir)).filter(test).sort();
  const copied = [];
  await mkdir(toDir, { recursive: true });
  for (const name of names) {
    const src = join(fromDir, name);
    if (!(await stat(src)).isFile()) continue;
    const dest = join(toDir, name);
    await cp(src, dest);
    copied.push({ label, name, bytes: (await stat(dest)).size });
  }
  return copied;
}

const copied = [
  ...(await stage(join(root, "artifacts", "models"), join(site, "models"), (n) => n.endsWith(".onnx"), "model")),
  ...(await stage(join(root, "artifacts", "models"), join(site, "models"), (n) => n.endsWith(".json"), "model-meta")),
  ...(await stage(join(root, "artifacts", "reports"), join(site, "data"), (n) => n.endsWith(".json"), "report")),
  ...(await stage(join(root, "artifacts", "figures"), join(site, "data", "figures"), () => true, "figure")),
];

for (const { label, name, bytes } of copied) {
  console.log(`  ${label.padEnd(10)} ${(bytes / 1024).toFixed(1).padStart(9)} KB  ${name}`);
}

if (copied.length === 0) {
  console.error("build_site: nothing to stage. Run `hemolux train` and `hemolux export` first.");
  process.exitCode = 1;
} else {
  console.log(`build_site: staged ${copied.length} file(s) into app/web/ (not committed).`);
}
