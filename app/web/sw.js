/* Offline shell for a screening tool used where the network is worst.
 *
 * The app has to start and complete a screening with no connection: the shell,
 * the colour extractor, the ONNX runtime and the model are all cached on first
 * load and served from Cache Storage afterwards. The /api/ paths are never
 * cached — a stale telemetry response is worthless, and a cached POST is wrong.
 *
 * Bump CACHE_VERSION on any change to the shell; the activate handler drops
 * every older cache.
 */
const CACHE_VERSION = "hemolux-v2";

const SHELL = [
  "/",
  "/index.html",
  "/styles.css",
  "/icon.svg",
  "/manifest.webmanifest",
  "/colorimetry.js",
  "/src/app.js",
  "/src/i18n.js",
  "/src/inference.js",
  "/src/evidence.js",
  "/src/telemetry.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_VERSION)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_VERSION).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/")) return;

  event.respondWith(
    caches.match(request).then((hit) => {
      if (hit) return hit;
      return fetch(request)
        .then((response) => {
          if (response && response.ok && response.type === "basic") {
            const copy = response.clone();
            caches.open(CACHE_VERSION).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() => caches.match("/index.html"));
    }),
  );
});
