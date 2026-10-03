# Deploy

Hemolux runs on Cloudflare Pages as a static site plus two edge functions. There
is no server to keep alive and no card on file: the free tier is the intended
deployment, not a trial. Inference never leaves the user's device.

## What costs what

| Piece | Cost |
| --- | --- |
| Pages (static hosting + functions) | free |
| D1 (telemetry, feedback) | free tier |
| Inference (WASM, on-device) | free |
| `hemolux.pages.dev` | free |
| A custom domain | optional; `hemolux.in` is paid, `*.is-a.dev` is a free subdomain |

The only paid item is the domain name, and the site is fully functional without
one.

## Prerequisites

- Node 18+ (`node --version`).
- A Cloudflare account (free).
- The generated artifacts locally: `artifacts/models/*.onnx` and
  `artifacts/reports/*.json`, from `hemolux train` / `hemolux export`.

Wrangler is used through `npx`, so it is downloaded on first run and never has
to be installed globally.

## 1. Log in

```powershell
npx wrangler login
```

This opens a browser to authorise the CLI. For CI, skip it and export
`CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` instead.

## 2. Create the database and its schema

```powershell
npx wrangler d1 create hemolux-telemetry
```

Paste the returned `database_id` into `wrangler.toml`, then apply the schema:

```powershell
npx wrangler d1 execute hemolux-telemetry --remote --file scripts/init_db.sql
```

The schema has no column that could hold an image, an identifier or free text;
the two tables are `telemetry` and `feedback`.

## 3. Vendor the runtime and stage the generated files

```powershell
node scripts/vendor_runtime.mjs   # onnxruntime-web -> app/web/vendor/ort/ (git-ignored)
node scripts/build_site.mjs       # model + reports -> app/web/ (git-ignored)
```

`vendor_runtime.mjs` copies the three files the wasm backend needs out of a
pinned `onnxruntime-web`, so the page never pulls a script from a CDN and the
CSP can stay same-origin. `build_site.mjs` copies the exported model and the
aggregate reports into `app/web/models/` and `app/web/data/`. Both destinations
are git-ignored, so the deployed numbers always come from a real run rather than
from a committed file.

## 4. Deploy

```powershell
npx wrangler pages deploy app/web --project-name hemolux
```

The first deploy prints a `https://hemolux.pages.dev` URL.

## 5. Verify

```powershell
# The shell, the model and the manifest should all answer 200.
Invoke-WebRequest https://hemolux.pages.dev/ -UseBasicParsing | Select-Object StatusCode
Invoke-WebRequest https://hemolux.pages.dev/models/ -UseBasicParsing -ErrorAction SilentlyContinue

# The endpoint must refuse a GET and accept a minimal POST.
Invoke-WebRequest https://hemolux.pages.dev/api/v1/telemetry -Method GET -UseBasicParsing
Invoke-RestMethod https://hemolux.pages.dev/api/v1/telemetry -Method POST `
  -ContentType 'application/json' -Body '{"model_id":"smoke","band":"mild"}'
```

A screening must complete with the network disabled after the first load, which
is what the service worker caches the shell, the runtime and the model for.

## 6. Attach a domain (optional)

Workers & Pages → `hemolux` → Custom domains → add `hemolux.in` (paid) or a free
`*.is-a.dev` subdomain. DNS is handled by Cloudflare; no certificate step.

## Re-deploying

`node scripts/build_site.mjs` then `npx wrangler pages deploy` again. Bump
`CACHE_VERSION` in `app/web/sw.js` whenever the shell changes, or a returning
user keeps the old one.
