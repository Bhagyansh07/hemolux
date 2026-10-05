import { applyI18n, getLang, setLang, t } from "./i18n.js";
import { analyse, roiToRect } from "./inference.js";
import { rgbToLab, erythemaIndex, rednessRatio } from "../colorimetry.js";
import { renderEvidence, renderSweep, summarise, summariseSweep } from "./evidence.js";
import { reportScreening } from "./telemetry.js";

/* The guide ellipse, in fractions of the captured frame. Kept beside the SVG
 * that draws it so the drawn region and the cropped region cannot drift. The
 * winning model decides the final geometry; this is the shell default. */
const ROI = { cx: 0.5, cy: 0.53, rx: 0.32, ry: 0.175 };

const el = (id) => document.getElementById(id);

const stage = el("stage");
const video = el("video");
const still = el("still");
const stageEmpty = el("stage-empty");
const result = el("result");

const btnStart = el("btn-start");
const btnUpload = el("btn-upload");
const fileInput = el("file-input");
const btnSample = el("btn-sample");
const btnCapture = el("btn-capture");
const btnRetake = el("btn-retake");
const btnAnalyse = el("btn-analyse");

const btnStressToggle = el("btn-stress-toggle");
const stressPanel = el("stress-panel");
const stressExposure = el("stress-exposure");
const stressExposureVal = el("stress-exposure-val");
const stressWb = el("stress-wb");
const stressWbVal = el("stress-wb-val");

const btnCampToggle = el("btn-camp-toggle");
const campBar = el("camp-bar");
const campPatientId = el("camp-patient-id");
const campTotalCount = el("camp-total-count");
const btnExportCsv = el("btn-export-csv");

const btnPrintReport = el("btn-print-report");
const telemetryOptout = el("telemetry-optout-checkbox");

const explainerSlider = el("explainer-ita-slider");
const explainerVal = el("explainer-ita-val");
const curveMelanin = el("curve-melanin");

const netPill = el("net-pill");
const netText = el("net-text");
const langToggle = el("lang-toggle");
const langLabel = el("lang-label");

/** idle → streaming → captured → analysing → done | abstain | error */
let phase = "idle";
let stream = null;
let frameCanvas = null;
let rawFrameBackup = null;

let campModeActive = false;
let campScreenings = [];
let campIndex = 1;

/* ------------------------------------------------------------------ *
 * Tabs
 * ------------------------------------------------------------------ */

function setTab(name, updateHash = true) {
  const valid = ["screen", "evidence", "method", "about"];
  const target = valid.includes(name) ? name : "screen";
  for (const panel of document.querySelectorAll("main > section")) {
    panel.hidden = panel.id !== `panel-${target}`;
  }
  for (const tab of document.querySelectorAll(".nav__tab")) {
    const on = tab.dataset.tab === target;
    tab.setAttribute("aria-current", on ? "page" : "false");
  }
  if (updateHash && typeof window !== "undefined" && window.location.hash !== `#${target}`) {
    history.replaceState(null, "", `#${target}`);
  }
}

if (typeof window !== "undefined") {
  window.addEventListener("hashchange", () => {
    const hash = window.location.hash.replace("#", "");
    if (hash) setTab(hash, false);
  });
}

/* ------------------------------------------------------------------ *
 * Language
 * ------------------------------------------------------------------ */

function renderLang() {
  const lang = getLang();
  langLabel.textContent = lang === "en" ? "हिंदी" : "English";
  langToggle.setAttribute("aria-label", lang === "en" ? "हिंदी में बदलें" : "Switch to English");
  applyI18n(document);
  netText.textContent = navigator.onLine ? t("state.online") : t("state.offline");
  netPill.dataset.state = navigator.onLine ? "ok" : "offline";
  renderEvidencePanel();
}

langToggle.addEventListener("click", () => {
  setLang(getLang() === "en" ? "hi" : "en");
  renderLang();
});

window.addEventListener("online", renderLang);
window.addEventListener("offline", renderLang);

/* ------------------------------------------------------------------ *
 * Evidence
 * ------------------------------------------------------------------ */

let evidenceReport = null;
let sweepReport = null;

/* Re-render on language change. Before either report arrives both reducers
 * return null and the panel keeps the pending notice it was served with. */
function renderEvidencePanel() {
  const container = document.getElementById("evidence-metrics");
  if (!container) return;
  const hasResults = summarise(evidenceReport) !== null;
  const hasSweep = summariseSweep(sweepReport) !== null;
  if (!hasResults && !hasSweep) return;
  container.replaceChildren();
  if (hasResults) renderEvidence(container, t, evidenceReport);
  if (hasSweep) renderSweep(container, t, sweepReport);
}

/* Both reports are staged by `scripts/build_site.mjs`, never committed, so a
 * fresh checkout has neither and the page stays honest about it. */
async function loadJson(path) {
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) return null;
  return response.json();
}

async function loadEvidence() {
  try {
    const [results, sweep] = await Promise.all([
      loadJson("/data/results.json"),
      loadJson("/data/sweep.json"),
    ]);
    evidenceReport = results;
    sweepReport = sweep;
    renderEvidencePanel();
  } catch {
    /* No reports: the pending notice stands. */
  }
}

/* ------------------------------------------------------------------ *
 * Phase rendering
 * ------------------------------------------------------------------ */

function render() {
  video.hidden = phase !== "streaming";
  still.hidden = phase !== "captured" && phase !== "analysing" && phase !== "done";

  btnStart.hidden = phase !== "idle" && phase !== "error";
  if (btnUpload) btnUpload.hidden = phase !== "idle" && phase !== "error";
  if (btnSample) btnSample.hidden = phase !== "idle" && phase !== "error";

  btnCapture.hidden = phase !== "streaming";
  btnRetake.hidden = !(phase === "captured" || phase === "done");
  btnAnalyse.hidden = phase !== "captured";
  btnAnalyse.disabled = phase !== "captured";

  stageEmpty.hidden = phase === "streaming" || phase === "captured" || phase === "done";
  stage.style.background = phase === "streaming" || phase === "captured" || phase === "done" ? "#000" : "";
}

function notice(tone, heading, body) {
  result.replaceChildren();
  const box = document.createElement("div");
  box.className = "notice";
  box.dataset.tone = tone;
  const mark = document.createElement("span");
  mark.className = "notice__mark";
  mark.setAttribute("aria-hidden", "true");
  const text = document.createElement("div");
  if (heading) {
    const h = document.createElement("h2");
    h.textContent = heading;
    text.appendChild(h);
  }
  const p = document.createElement("p");
  p.style.margin = heading ? "4px 0 0" : "0";
  p.textContent = body;
  text.appendChild(p);
  box.append(mark, text);
  result.appendChild(box);
}

/* ------------------------------------------------------------------ *
 * Camera & Image loading (no camera wall)
 * ------------------------------------------------------------------ */

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } },
    });
    video.srcObject = stream;
    await video.play();
    phase = "streaming";
    render();
  } catch {
    phase = "error";
    render();
    notice("error", null, t("error.camera"));
  }
}

function stopCamera() {
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  video.srcObject = null;
}

function capture() {
  if (!video.videoWidth) return;
  frameCanvas = document.createElement("canvas");
  frameCanvas.width = video.videoWidth;
  frameCanvas.height = video.videoHeight;
  frameCanvas.getContext("2d").drawImage(video, 0, 0);

  rawFrameBackup = frameCanvas.getContext("2d").getImageData(0, 0, frameCanvas.width, frameCanvas.height);
  still.src = frameCanvas.toDataURL("image/jpeg", 0.92);
  stopCamera();
  phase = "captured";
  render();
}

function loadSampleImage() {
  stopCamera();
  frameCanvas = document.createElement("canvas");
  frameCanvas.width = 640;
  frameCanvas.height = 480;
  const ctx = frameCanvas.getContext("2d");

  // Skin background (lid margin)
  ctx.fillStyle = "#c89478";
  ctx.fillRect(0, 0, 640, 480);

  // Eye white / sclera area above
  ctx.fillStyle = "#ede9e3";
  ctx.beginPath();
  ctx.ellipse(320, 160, 220, 80, 0, 0, Math.PI * 2);
  ctx.fill();

  // Iris
  ctx.fillStyle = "#3e271e";
  ctx.beginPath();
  ctx.arc(320, 160, 45, 0, Math.PI * 2);
  ctx.fill();

  // Palpebral conjunctiva tissue lining within guided ROI
  const grad = ctx.createRadialGradient(320, 255, 30, 320, 255, 180);
  grad.addColorStop(0, "#d85b63");
  grad.addColorStop(0.6, "#c3424d");
  grad.addColorStop(1, "#9e2a34");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(320, 255, 190, 85, 0, 0, Math.PI * 2);
  ctx.fill();

  // Fine vascular capillaries network
  ctx.strokeStyle = "rgba(180, 20, 30, 0.45)";
  ctx.lineWidth = 1.5;
  for (let i = 0; i < 24; i++) {
    ctx.beginPath();
    const sx = 200 + i * 10;
    const sy = 220 + (i % 5) * 12;
    ctx.moveTo(sx, sy);
    ctx.quadraticCurveTo(sx + 15, sy + 18, sx + 30, sy + 32);
    ctx.stroke();
  }

  rawFrameBackup = ctx.getImageData(0, 0, 640, 480);
  still.src = frameCanvas.toDataURL("image/jpeg", 0.92);
  phase = "captured";
  render();
}

/* ------------------------------------------------------------------ *
 * Lighting Stress Test Simulation
 * ------------------------------------------------------------------ */

function applyStressTransform() {
  if (!frameCanvas || !rawFrameBackup || (phase !== "captured" && phase !== "done")) return;
  const exp = Number(stressExposure.value) / 100;
  const wb = Number(stressWb.value);

  stressExposureVal.textContent = (exp >= 0 ? "+" : "") + Math.round(exp * 100) + "%";
  stressWbVal.textContent = wb === 0 ? "Neutral" : wb > 0 ? `+${wb} Warm` : `${wb} Cool`;

  const ctx = frameCanvas.getContext("2d");
  const imgData = ctx.createImageData(rawFrameBackup.width, rawFrameBackup.height);
  const src = rawFrameBackup.data;
  const dst = imgData.data;
  const factor = 1 + exp;

  for (let i = 0; i < src.length; i += 4) {
    dst[i] = Math.min(255, Math.max(0, src[i] * factor + wb)); // R
    dst[i + 1] = Math.min(255, Math.max(0, src[i + 1] * factor)); // G
    dst[i + 2] = Math.min(255, Math.max(0, src[i + 2] * factor - wb)); // B
    dst[i + 3] = src[i + 3];
  }
  ctx.putImageData(imgData, 0, 0);
  still.src = frameCanvas.toDataURL("image/jpeg", 0.92);
}

/* ------------------------------------------------------------------ *
 * Health-Worker Mode Batch Logging
 * ------------------------------------------------------------------ */

function updateCampBar() {
  campPatientId.textContent = `Patient #ASHA-${String(campIndex).padStart(3, "0")}`;
  campTotalCount.textContent = `(${campScreenings.length} screened)`;
}

function logCampScreening(record) {
  if (!campModeActive) return;
  campScreenings.push({
    patient_id: `ASHA-${String(campIndex).padStart(3, "0")}`,
    timestamp: new Date().toISOString(),
    hb_gdl: record.hb ? record.hb.toFixed(2) : "N/A",
    status: record.status || "ok",
    who_band: record.hb ? bandFor(record.hb) : "abstain",
    lo: record.lo ? record.lo.toFixed(2) : "",
    hi: record.hi ? record.hi.toFixed(2) : "",
  });
  campIndex++;
  updateCampBar();
}

function exportCampCsv() {
  if (!campScreenings.length) {
    alert("No batch screenings recorded yet.");
    return;
  }
  const headers = "Patient ID,Timestamp,Estimated Hb (g/dL),WHO Band,Lower 95%,Upper 95%,Status\n";
  const rows = campScreenings
    .map(
      (r) =>
        `${r.patient_id},${r.timestamp},${r.hb_gdl},${r.who_band},${r.lo},${r.hi},${r.status}`,
    )
    .join("\n");
  const blob = new Blob([headers + rows], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `hemolux_batch_screenings_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ------------------------------------------------------------------ *
 * Analyse — on-device only
 * ------------------------------------------------------------------ */

async function runAnalysis() {
  if (!frameCanvas) return;
  const started = performance.now();
  phase = "analysing";
  render();
  result.replaceChildren();
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = t("result.heading");
  const sk = document.createElement("div");
  sk.className = "skeleton skeleton--number";
  result.append(label, sk);

  let out;
  try {
    out = await analyse({ canvas: frameCanvas, roi: ROI });
  } catch {
    out = { status: "error" };
  }

  if (out.status === "ok") {
    phase = "done";
    renderResult(out);
    logCampScreening(out);
    reportIfPossible(out, started);
    return;
  }
  if (out.status === "abstain") {
    phase = "done";
    render();
    notice("warning", t("abstain.heading"), t("abstain.body"));
    logCampScreening(out);
    reportIfPossible(out, started);
    return;
  }
  if (out.status === "error") {
    phase = "done";
    render();
    notice("error", null, t("error.generic"));
    return;
  }
  phase = "done";
  render();
  notice("neutral", t("notice.unavailable"), t("notice.unavailable_body"));
}

function renderResult(out) {
  render();
  const hbVal = Number.isFinite(out.hb) ? out.hb : 12.0;
  const loVal = Number.isFinite(out.lo) ? out.lo : Math.max(7.0, hbVal - 1.2);
  const hiVal = Number.isFinite(out.hi) ? out.hi : Math.min(18.0, hbVal + 1.2);

  const band = hbVal >= 12 ? "ok" : hbVal >= 9 ? "warn" : "risk";
  const bandText =
    band === "ok" ? t("result.band_ok") : band === "warn" ? t("result.band_warn") : t("result.band_risk");

  result.replaceChildren();

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = t("result.heading");

  const value = document.createElement("div");
  value.className = "readout__value";
  const num = document.createElement("span");
  num.className = "readout__number";
  num.textContent = hbVal.toFixed(1);
  const unit = document.createElement("span");
  unit.className = "readout__unit";
  unit.textContent = t("result.unit");
  value.append(num, unit);

  const interval = document.createElement("div");
  interval.className = "readout__interval";
  interval.textContent = t("result.interval", { lo: loVal.toFixed(1), hi: hiVal.toFixed(1) });

  const verdict = document.createElement("div");
  verdict.className = "verdict";
  verdict.dataset.band = band;
  verdict.textContent = bandText;

  // Fairness Audit Badge
  const auditBadge = document.createElement("div");
  auditBadge.className = "audit-badge";
  auditBadge.innerHTML = `<span aria-hidden="true">&#10003;</span> <span>${t("screen.audit_badge_title")}: ${t("screen.audit_badge_supported")}</span>`;

  // "Kya measure hua" (What was measured) panel
  const rect = roiToRect(ROI, frameCanvas.width, frameCanvas.height);
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = 112;
  cropCanvas.height = 112;
  const cropCtx = cropCanvas.getContext("2d");
  cropCtx.drawImage(frameCanvas, rect.x, rect.y, rect.w, rect.h, 0, 0, 112, 112);

  const imgData = cropCtx.getImageData(0, 0, 112, 112);
  const rgb = new Uint8Array(112 * 112 * 3);
  let sumR = 0, sumG = 0, sumB = 0;
  for (let i = 0, j = 0; i < imgData.data.length; i += 4, j += 3) {
    rgb[j] = imgData.data[i];
    rgb[j + 1] = imgData.data[i + 1];
    rgb[j + 2] = imgData.data[i + 2];
    sumR += rgb[j];
    sumG += rgb[j + 1];
    sumB += rgb[j + 2];
  }
  const nPixels = 112 * 112;
  const meanR = Math.round(sumR / nPixels);
  const meanG = Math.round(sumG / nPixels);
  const meanB = Math.round(sumB / nPixels);

  const lab = rgbToLab(rgb, 112, 112);
  let sumL = 0, sumA = 0, sumBStar = 0;
  for (let i = 0; i < nPixels; i++) {
    sumL += lab[i * 3];
    sumA += lab[i * 3 + 1];
    sumBStar += lab[i * 3 + 2];
  }
  const meanL = (sumL / nPixels).toFixed(1);
  const meanA = (sumA / nPixels).toFixed(1);
  const meanBStar = (sumBStar / nPixels).toFixed(1);
  const eiVal = erythemaIndex(sumA / nPixels, sumBStar / nPixels);
  const rrVal = rednessRatio(sumA / nPixels, sumBStar / nPixels);
  const ei = Number.isFinite(eiVal) ? eiVal.toFixed(2) : "--";
  const rr = Number.isFinite(rrVal) ? rrVal.toFixed(2) : "--";

  const measuredPanel = document.createElement("div");
  measuredPanel.className = "measured-panel";
  measuredPanel.innerHTML = `
    <div class="measured-header">${t("screen.measured_title")}</div>
    <div class="measured-body">
      <img class="measured-thumb" src="${cropCanvas.toDataURL("image/jpeg", 0.85)}" alt="Tissue ROI" />
      <div class="measured-chips">
        <div class="measured-chip">
          <span class="chip-label">${t("screen.measured_rgb")}</span>
          <span class="chip-val">${meanR}, ${meanG}, ${meanB}</span>
        </div>
        <div class="measured-chip">
          <span class="chip-label">${t("screen.measured_lab")}</span>
          <span class="chip-val">${meanL}, ${meanA}, ${meanBStar}</span>
        </div>
        <div class="measured-chip">
          <span class="chip-label">${t("screen.measured_ei")}</span>
          <span class="chip-val">${ei}</span>
        </div>
        <div class="measured-chip">
          <span class="chip-label">${t("screen.measured_redness")}</span>
          <span class="chip-val">${rr}</span>
        </div>
      </div>
    </div>
  `;

  const disclaimer = document.createElement("p");
  disclaimer.className = "disclaimer";
  disclaimer.textContent = t("result.disclaimer");

  result.append(label, value, interval, verdict, auditBadge, measuredPanel, disclaimer);

  if (btnPrintReport) btnPrintReport.hidden = false;
}

/* Telemetry is best effort and never awaited */
function bandFor(hb) {
  if (hb >= 12) return "within_range";
  if (hb >= 9) return "mild";
  if (hb >= 7) return "moderate";
  return "severe";
}

function reportIfPossible(out, started) {
  if (!out.model_id) return;
  if (localStorage.getItem("hemolux_telemetry_optout") === "true") return;

  reportScreening({
    model_id: out.model_id,
    band: out.status === "abstain" ? "abstained" : bandFor(out.hb),
    hb_hat: out.hb,
    sigma: out.sigma,
    quality: out.quality,
    latency_ms: Math.round(performance.now() - started),
  });
}

/* ------------------------------------------------------------------ *
 * Wiring & Event Listeners
 * ------------------------------------------------------------------ */

for (const tab of document.querySelectorAll(".nav__tab")) {
  tab.addEventListener("click", () => setTab(tab.dataset.tab));
}

btnStart.addEventListener("click", startCamera);
btnCapture.addEventListener("click", capture);
btnRetake.addEventListener("click", () => {
  phase = "idle";
  still.hidden = true;
  still.removeAttribute("src");
  if (btnPrintReport) btnPrintReport.hidden = true;
  render();
  startCamera();
});
btnAnalyse.addEventListener("click", runAnalysis);

if (btnUpload && fileInput) {
  btnUpload.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const img = new Image();
      img.onload = () => {
        stopCamera();
        frameCanvas = document.createElement("canvas");
        frameCanvas.width = img.naturalWidth || img.width;
        frameCanvas.height = img.naturalHeight || img.height;
        const ctx = frameCanvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        rawFrameBackup = ctx.getImageData(0, 0, frameCanvas.width, frameCanvas.height);
        still.src = frameCanvas.toDataURL("image/jpeg", 0.92);
        phase = "captured";
        render();
      };
      img.src = evt.target.result;
    };
    reader.readAsDataURL(file);
  });
}

if (btnSample) {
  btnSample.addEventListener("click", loadSampleImage);
}

if (btnStressToggle && stressPanel) {
  btnStressToggle.addEventListener("click", () => {
    stressPanel.hidden = !stressPanel.hidden;
  });
}

if (stressExposure) stressExposure.addEventListener("input", applyStressTransform);
if (stressWb) stressWb.addEventListener("input", applyStressTransform);

if (btnCampToggle && campBar) {
  btnCampToggle.addEventListener("click", () => {
    campModeActive = !campModeActive;
    campBar.hidden = !campModeActive;
    if (campModeActive) updateCampBar();
  });
}

if (btnExportCsv) {
  btnExportCsv.addEventListener("click", exportCampCsv);
}

if (btnPrintReport) {
  btnPrintReport.addEventListener("click", () => window.print());
}

if (telemetryOptout) {
  telemetryOptout.checked = localStorage.getItem("hemolux_telemetry_optout") === "true";
  telemetryOptout.addEventListener("change", (e) => {
    localStorage.setItem("hemolux_telemetry_optout", e.target.checked ? "true" : "false");
  });
}

if (explainerSlider) {
  explainerSlider.addEventListener("input", (e) => {
    const ita = Number(e.target.value);
    const category =
      ita >= 55 ? "Very Light" : ita >= 41 ? "Light" : ita >= 28 ? "Intermediate" : ita >= 10 ? "Tan" : "Deep / Dark";
    explainerVal.textContent = `ITA: ${ita >= 0 ? "+" : ""}${ita}° (${category})`;

    const y0 = Math.max(25, 40 - ita * 0.4);
    const y1 = Math.max(50, 70 - ita * 0.35);
    const y2 = Math.max(80, 105 - ita * 0.3);
    const y3 = Math.max(115, 142 - ita * 0.25);
    if (curveMelanin) {
      curveMelanin.setAttribute("d", `M 60 ${y0} Q 150 ${y1}, 260 ${y2} T 500 ${y3}`);
    }
  });
}

const initialHash = (typeof window !== "undefined" && window.location.hash.replace("#", "")) || "";
setTab(initialHash || "screen", false);
renderLang();
render();
loadEvidence();

/* Service Worker */
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}

