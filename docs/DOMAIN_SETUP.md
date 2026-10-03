# Domain and clean-machine deploy

This is the long form of [DEPLOY.md](DEPLOY.md): how to reproduce the whole thing
on a machine that has never seen this repository, and how to put a custom domain
in front of it. The short version is in `DEPLOY.md`; this one exists so the
checklist item "reproduces the deploy from scratch on a clean machine" can be
ticked by following it, not by trusting it.

Everything here is free except the domain name. Inference runs on the visitor's
device, so the deploy has no inference bill.

## 0. What the deployment is

A static site (`app/web/`) served by Cloudflare Pages, plus two Pages Functions
(`functions/api/v1/`) writing to a D1 database. There is no server process to
keep alive, no container and no GPU. The model is a `.onnx` file the browser
loads once and runs through WebAssembly.

## 1. A clean machine

| Tool | Version | Why |
| --- | --- | --- |
| Python | 3.13 | the pins in `pyproject.toml` are exact |
| Node | 18 or newer | `scripts/*.mjs` and `npx wrangler` |
| git | any recent | to clone |
| Cloudflare account | free | Pages + D1 + Functions |

`wrangler` is run through `npx`, so it does not have to be installed globally.

## 2. Clone and install

```bash
git clone https://github.com/Bhagyansh07/hemolux.git
cd hemolux

python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

# CPU-only torch; the default PyPI wheel pulls multi-gigabyte CUDA libraries
# this project never uses.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[dev]"
```

If `pip install -e ".[dev]"` complains about torch, the first command did not
take effect; check `pip show torch`.

## 3. The dataset

The corpus is not redistributed here. Download the Eyes Defy Anemia dataset from
Kaggle and unpack it to the path in `hemolux.data.dataset.DEFAULT_ROOT`
(`data/raw/eyes-defy-anemia/dataset anemia`). See
[DATASET.md](DATASET.md) for provenance and licence.

Confirm the corpus before trusting any number from it:

```bash
hemolux validate
```

It exits non-zero on anything that would make a result wrong rather than merely
imprecise, and prints the mask-resolution mismatch, the patients with no
forniceal mask, and the empty severe band either way.

## 4. Reproduce the model and the reports

The order matters; the app enforces it.

```bash
hemolux sweep                          # select a configuration on validation
hemolux train                          # five heads, cross-site check, results.json
hemolux export --head ordinal          # ONNX + sibling model.json
```

- `hemolux sweep` fits every candidate with the test fold untouched, picks a
  winner on validation, and reads test exactly once. It writes
  `artifacts/reports/sweep.json` / `sweep.csv`.
- `hemolux train` writes `artifacts/reports/results.json` / `results.csv`, the
  per-patient `predictions.csv` (never committed), and one checkpoint per head.
- `hemolux export` writes `artifacts/models/hemolux_screen_224.onnx` and a
  sibling `hemolux_screen_224.json`. Its `validated` flag is true only when
  `results.json` carries a row for the exported head, and its `residual_sigma`
  comes from that row's limits of agreement. A graph exported before its head was
  measured is deliberately inert: the app reports "unavailable" rather than an
  interval it cannot defend.

## 5. Stage the site

```bash
node scripts/vendor_runtime.mjs   # onnxruntime-web -> app/web/vendor/ort/
node scripts/build_site.mjs       # model + reports -> app/web/models, app/web/data/
```

Both destinations are git-ignored. The numbers the site serves therefore always
come from a real run, never from a committed file.

## 6. Cloudflare: log in, create the database

```bash
npx wrangler login                # opens a browser to authorise
npx wrangler d1 create hemolux-telemetry
```

Paste the returned `database_id` into `wrangler.toml`, replacing
`REPLACE_WITH_D1_DATABASE_ID`, then apply the schema:

```bash
npx wrangler d1 execute hemolux-telemetry --remote --file scripts/init_db.sql
```

For CI instead of an interactive login, set `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID` as repository secrets.

## 7. Deploy

```bash
npx wrangler pages deploy app/web --project-name hemolux
```

The first deploy prints `https://hemolux.pages.dev`. That URL is complete on its
own; the custom domain is optional and can be added later without a redeploy.

## 8. Verify the deploy

```bash
# Shell, model and manifest all answer.
curl -I https://hemolux.pages.dev/
curl -I https://hemolux.pages.dev/models/model.json

# The endpoint refuses GET and accepts a minimal POST.
curl -i https://hemolux.pages.dev/api/v1/telemetry
curl -i -X POST https://hemolux.pages.dev/api/v1/telemetry \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"smoke","band":"mild"}'
```

Then, in a browser: load the page once, open DevTools → Network, and confirm no
request carries image bytes; disconnect the network and confirm a screening still
completes from the service worker cache.

## 9. A custom domain

`hemolux.in` is the intended domain. Registry and registrar pricing moves, so
check the current first-year and renewal price before buying; a `.in` domain is
usually cheap in year one and noticeably more expensive to renew, and a free
`*.is-a.dev` subdomain is a no-cost alternative if the paid name is not worth it.

Once the domain is bought and its nameservers point at Cloudflare:

1. Cloudflare dashboard → Workers & Pages → `hemolux` → **Custom domains**.
2. Add `hemolux.in` (and `www.hemolux.in` if wanted; redirect one to the other).
3. Cloudflare issues the certificate automatically; no certificate step.
4. **Update the absolute URLs.** They are the only place the domain is written
   down, and they are plain text:
   - `app/web/robots.txt` → the `Sitemap:` line
   - `app/web/sitemap.xml` → the `<loc>` value
   - `app/web/index.html` → add the `<link rel="canonical">` and Open Graph
     `og:url` once the domain is live
5. Redeploy so the updated files are served:

   ```bash
   npx wrangler pages deploy app/web --project-name hemolux
   ```

DNS and TLS are handled by Cloudflare because the zone lives there; there is no
separate certificate or load-balancer step.

## 10. Rolling back

Pages keeps every previous deployment. Dashboard → the project → **Deployments**
→ pick the last good one → **Rollback**. The database schema is additive
(`CREATE TABLE IF NOT EXISTS`), so a rollback of the site does not require a
schema change.

## 11. Cost

| Piece | Cost |
| --- | --- |
| Pages hosting + Functions | free tier |
| D1 telemetry + feedback | free tier |
| On-device inference | free |
| `hemolux.pages.dev` | free |
| Custom domain | the only paid item |
