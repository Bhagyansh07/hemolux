import { applyI18n, getLang, setLang, t } from "./i18n.js";
import { analyse } from "./inference.js";

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
const btnCapture = el("btn-capture");
const btnRetake = el("btn-retake");
const btnAnalyse = el("btn-analyse");

const netPill = el("net-pill");
const netText = el("net-text");
const langToggle = el("lang-toggle");
const langLabel = el("lang-label");

/** idle → streaming → captured → analysing → done | abstain | error */
let phase = "idle";
let stream = null;
let frameCanvas = null;

/* ------------------------------------------------------------------ *
 * Tabs
 * ------------------------------------------------------------------ */

function setTab(name) {
  for (const panel of document.querySelectorAll("main > section")) {
    panel.hidden = panel.id !== `panel-${name}`;
  }
  for (const tab of document.querySelectorAll(".nav__tab")) {
    const on = tab.dataset.tab === name;
    tab.setAttribute("aria-current", on ? "page" : "false");
  }
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
}

langToggle.addEventListener("click", () => {
  setLang(getLang() === "en" ? "hi" : "en");
  renderLang();
});

window.addEventListener("online", renderLang);
window.addEventListener("offline", renderLang);

/* ------------------------------------------------------------------ *
 * Phase rendering
 * ------------------------------------------------------------------ */

function render() {
  video.hidden = phase !== "streaming";
  still.hidden = phase !== "captured" && phase !== "analysing" && phase !== "done";

  btnStart.hidden = phase !== "idle" && phase !== "error";
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
 * Camera
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

  still.src = frameCanvas.toDataURL("image/jpeg", 0.92);
  stopCamera();
  phase = "captured";
  render();
}

/* ------------------------------------------------------------------ *
 * Analyse — on-device only
 * ------------------------------------------------------------------ */

async function runAnalysis() {
  if (!frameCanvas) return;
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
    return;
  }
  if (out.status === "abstain") {
    phase = "done";
    render();
    notice("warning", t("abstain.heading"), t("abstain.body"));
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
  const band = out.hb >= 12 ? "ok" : out.hb >= 9 ? "warn" : "risk";
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
  num.textContent = out.hb.toFixed(1);
  const unit = document.createElement("span");
  unit.className = "readout__unit";
  unit.textContent = t("result.unit");
  value.append(num, unit);

  const interval = document.createElement("div");
  interval.className = "readout__interval";
  interval.textContent = t("result.interval", { lo: out.lo.toFixed(1), hi: out.hi.toFixed(1) });

  const verdict = document.createElement("div");
  verdict.className = "verdict";
  verdict.dataset.band = band;
  verdict.textContent = bandText;

  const disclaimer = document.createElement("p");
  disclaimer.className = "disclaimer";
  disclaimer.textContent = t("result.disclaimer");

  result.append(label, value, interval, verdict, disclaimer);
}

/* ------------------------------------------------------------------ *
 * Wiring
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
  render();
  startCamera();
});
btnAnalyse.addEventListener("click", runAnalysis);

setTab("screen");
renderLang();
render();
