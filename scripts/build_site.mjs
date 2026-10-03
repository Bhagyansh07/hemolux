#!/usr/bin/env node
/* Assemble the deployable static site.
 *
 * Cloudflare Pages serves app/web/ verbatim, so the two things that are
 * *generated* rather than authored have to be placed there first: the exported
 * model and the aggregate evidence the UI reads. Neither is committed -- the
 * model is a binary and the reports are reproduced from the run that made them.
 * Routing the copy through this one script means the numbers on the deployed
 * site cannot drift from the ones EVALS.md cites.
 *
 * The graph and its metadata travel as a pair. The metadata is staged under the
 * fixed name the app fetches (`model.json`) and names the graph file itself, so
 * a graph can never end up next to another graph's report.
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
const copied = [];

async function stageFile(source, destination, label) {
  await mkdir(dirname(destination), { recursive: true });
  await cp(source, destination);
  const shown = destination.slice(site.length + 1).replaceAll("\\", "/");
  copied.push({ label, name: shown, bytes: (await stat(destination)).size });
}

/* Every aggregate JSON the last run wrote. /data/results.json is what the
 * Evidence tab reads; sweep.json is the cross-candidate summary. */
const reports = join(root, "artifacts", "reports");
if (existsSync(reports)) {
  const names = (await readdir(reports)).filter((name) => name.endsWith(".json")).sort();
  for (const name of names) {
    await stageFile(join(reports, name), join(site, "data", name), "report");
  }
}

const figures = join(root, "artifacts", "figures");
if (existsSync(figures)) {
  for (const name of (await readdir(figures)).sort()) {
    const source = join(figures, name);
    if ((await stat(source)).isFile()) {
      await stageFile(source, join(site, "data", "figures", name), "figure");
    }
  }
}

/* The newest exported graph and the metadata written beside it by
 * `hemolux export`. Newest wins because a re-export of the same size overwrites
 * the name, while a different size leaves two; the app fetches one model. */
const models = join(root, "artifacts", "models");
if (existsSync(models)) {
  const graphs = (await readdir(models)).filter((name) => /^hemolux_screen_\d+\.onnx$/.test(name));
  let newest = null;
  for (const name of graphs) {
    const info = await stat(join(models, name));
    if (!newest || info.mtimeMs > newest.mtimeMs) newest = { name, mtimeMs: info.mtimeMs };
  }
  if (newest) {
    await stageFile(join(models, newest.name), join(site, "models", newest.name), "model");
    const metaName = newest.name.replace(/\.onnx$/, ".json");
    if (existsSync(join(models, metaName))) {
      await stageFile(join(models, metaName), join(site, "models", "model.json"), "model-meta");
    } else {
      console.warn(`build_site: ${metaName} is missing; the app will report "unavailable".`);
    }
  } else {
    console.warn("build_site: no hemolux_screen_*.onnx under artifacts/models.");
  }
}

for (const { label, name, bytes } of copied) {
  console.log(`  ${label.padEnd(10)} ${(bytes / 1024).toFixed(1).padStart(9)} KB  ${name}`);
}

if (copied.length === 0) {
  console.error("build_site: nothing to stage. Run `hemolux train` and `hemolux export` first.");
  process.exitCode = 1;
} else {
  console.log(`build_site: staged ${copied.length} file(s) into app/web/ (not committed).`);
}
