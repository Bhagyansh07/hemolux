# ADR 004: Zero-Build Vanilla Web Stack with On-Device ONNX WebAssembly

- **Status:** Accepted
- **Date:** 2026-10-02
- **Deciders:** Core Engineering Team

---

## Context & Problem Statement

Most modern web applications rely on complex bundlers (Webpack, Vite, Next.js), large `node_modules` dependency graphs, and server-side API runtimes (FastAPI, Docker, cloud GPUs).

For Hemolux, we face three non-negotiable constraints:
1. **Patient Privacy:** Medical photographs of patient eyes must NEVER leave the user's phone or be transmitted across the network.
2. **Rural & Offline Resilience:** In Indian rural health camps (ASHA worker triage), internet connectivity is intermittent or non-existent.
3. **Auditability & Zero Operating Cost:** The project must run permanently without recurring server bills, dependency rot, or proprietary build pipelines.

---

## Decision

1. **Zero-Build Vanilla Architecture:**
   - Plain HTML5, Vanilla CSS, and native ES Modules (`import`/`export`).
   - Zero bundler configuration, zero compilation step, zero npm dependency bloat for deployment.
2. **On-Device Inference via ONNX Runtime Web:**
   - Pre-trained models are exported via `hemolux export` to compact ONNX graphs (~4.2 MB).
   - Inference runs inside the user's browser using WebAssembly SIMD (`onnxruntime-web`).
   - No patient photograph or pixel is ever sent over the network.
3. **Offline Progressive Web App (PWA):**
   - Service worker caches the runtime and model weights locally after first load.
   - Works 100% offline in rural clinical environments.

---

## Rationale & Engineering Comparison

| Architecture Option | Privacy Guarantee | Offline Capable | Cold Start Latency | Monthly Cloud Cost | Maintenance Complexity |
| --- | --- | --- | --- | --- | --- |
| **Server API (FastAPI / Render Free)** | Poor (Photos sent to cloud) | No | ~50s (Server sleep spin-up) | $0 until limits, then $20+/mo | High (Docker, Python versions) |
| **Next.js + Vercel Serverless** | Medium (Edge transmission) | No | ~1.5s cold start | Free tier ceiling | High (Node ecosystem churn) |
| **Hemolux On-Device Vanilla WASM** | **Absolute (0 KB image transfer)** | **Yes (100% Offline)** | **<100 ms (Cached WASM)** | **$0.00 (Permanent)** | **Minimal (Plain Web Standards)** |

### Security & Compliance Advantages:
- `_headers` enforces strict Content-Security-Policy: `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'`.
- All runtime scripts (`ort.wasm.min.js`) are self-hosted in `vendor/`, eliminating third-party CDN supply chain vulnerabilities.
- Zero server maintenance: Cloudflare Pages serves purely static assets over HTTP/3.

---

## Consequences

### Positive:
- Uncompromising patient privacy: Meets HIPAA / Indian Digital Personal Data Protection Act principles by construction.
- Permanent zero-cost hosting.
- Long-term archiveability: The code running today will still execute in standard browsers a decade from now without build-tool churn.

### Negative / Trade-offs:
- Model size is restricted to edge budgets (budget: <5 MB download).
- Preprocessing mathematics (bilinear interpolation, tensor scaling) had to be written and tested for byte-parity in pure JavaScript.
