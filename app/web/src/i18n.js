/* Bilingual strings. English is the source language; Hindi is a full
 * translation, not a machine gloss, and every user-visible string lives here —
 * there are no hard-coded labels in app.js. Adding a string means adding it to
 * both dictionaries or the missing-key guard below fires in development.
 */

export const MESSAGES = {
  en: {
    "nav.screen": "Screen",
    "nav.evidence": "Evidence",
    "nav.method": "Method & limits",
    "nav.about": "About",

    "state.online": "On device",
    "state.offline": "Offline",
    "state.offline_title": "No network. Inference runs on this device, so screening still works.",

    "screen.empty_label": "Camera off",
    "screen.empty_body":
      "Pull the lower eyelid down gently so the pink inner lining is visible, then hold the phone about 15 cm away in even light.",
    "screen.start": "Turn on camera",
    "screen.capture": "Take photo",
    "screen.retake": "Retake",
    "screen.analyse": "Analyse",
    "screen.guide_label": "Capture guidance",
    "screen.guide_body":
      "Fill the outline with the pink inner lining of the lower eyelid. Keep the phone steady; avoid flash and coloured lighting.",

    "result.heading": "Estimated haemoglobin",
    "result.unit": "g/dL",
    "result.interval": "Range {lo}–{hi} g/dL",
    "result.band_ok": "Within the normal range for this model",
    "result.band_warn": "Below the normal range",
    "result.band_risk": "Well below the normal range",
    "result.low_high": "The estimate is {band} the decision threshold for this model.",
    "result.disclaimer":
      "This is a screening estimate, not a diagnosis. Confirm with a laboratory blood test before any decision.",

    "abstain.heading": "Could not read this image",
    "abstain.body":
      "The photo was not clear enough to give a reliable estimate. Move to even light, avoid glare and shadow, and keep the phone steady. Nothing is wrong with the result — it simply was not measurable.",

    "error.camera": "Camera permission was refused or no camera is available. You can still open Settings and allow camera access.",
    "error.generic": "Something went wrong while reading the image. Try again.",

    "notice.unavailable": "This build has no model bundled yet.",
    "notice.unavailable_body":
      "The validation sweep selects the model and exports it to ONNX. Until then this screen does not produce a number.",

    "evidence.title": "Evidence",
    "evidence.intro":
      "Every figure on this page is produced by the frozen evaluation run in this repository. Nothing here is illustrative.",
    "evidence.pending":
      "Results will be published here once the validation sweep selects a model and the frozen test run completes.",
    "evidence.corpus_h": "Corpus",
    "evidence.roi": "Region",
    "evidence.backbone": "Backbone",
    "evidence.patients": "Patients",
    "evidence.sites": "Sites",
    "evidence.features": "Feature width",
    "evidence.excluded": "Excluded features",
    "evidence.none": "none",
    "evidence.heads_h": "Test metrics by head",
    "evidence.head_row": "MAE {mae} · RMSE {rmse} · R² {r2} · within ±1 g/dL {within}",
    "evidence.provenance_h": "Provenance",
    "evidence.commit": "Commit",
    "evidence.fingerprint": "Feature fingerprint",
    "evidence.generated": "Generated",
    "evidence.dirty": "Working tree dirty",
    "evidence.yes": "yes",
    "evidence.no": "no",

    "method.title": "Method & limits",
    "method.how_h": "How it works",
    "method.how_b":
      "The camera photographs the conjunctiva, the pink lining inside the lower eyelid, which carries the same capillary blood as a finger-prick. A regression model estimates haemoglobin in g/dL from that image. All computation happens on your device.",
    "method.cannot_h": "What it cannot do",
    "method.cannot_b":
      "This is a screening aid, not a diagnosis. It cannot confirm anaemia, cannot rule it out, and cannot replace a laboratory blood test. A result outside the normal range is a reason to seek a blood test, not a reason to start treatment.",
    "method.emergency_b": "Do not use it in an emergency or to make a treatment decision.",
    "method.fair_h": "Fairness",
    "method.fair_b":
      "Performance is reported separately by skin tone (ITA°) and by imaging site. Where the model is not equally reliable, that gap is stated rather than averaged away.",

    "about.title": "About",
    "about.dataset": "Dataset",
    "about.licence": "Licence",
    "about.runtime": "Runtime",
    "about.build": "Build",
    "about.privacy": "No photograph leaves this device. The app makes no network request with image data.",
  },

  hi: {
    "nav.screen": "जाँच",
    "nav.evidence": "प्रमाण",
    "nav.method": "तरीका और सीमाएँ",
    "nav.about": "परिचय",

    "state.online": "इसी डिवाइस पर",
    "state.offline": "ऑफ़लाइन",
    "state.offline_title": "इंटरनेट नहीं है। गणना इसी डिवाइस पर होती है, इसलिए जाँच चलती रहेगी।",

    "screen.empty_label": "कैमरा बंद है",
    "screen.empty_body":
      "निचली पलक को धीरे से नीचे खींचें ताकि अंदर की गुलाबी परत दिखे, फिर फ़ोन को करीब 15 सेमी दूर, एक-सी रोशनी में पकड़ें।",
    "screen.start": "कैमरा चालू करें",
    "screen.capture": "फ़ोटो लें",
    "screen.retake": "दोबारा लें",
    "screen.analyse": "विश्लेषण करें",
    "screen.guide_label": "फ़ोटो कैसे लें",
    "screen.guide_body":
      "निचली पलक की गुलाबी परत को दिख रही रेखा के अंदर भरें। फ़ोन स्थिर रखें; फ़्लैश और रंगीन रोशनी से बचें।",

    "result.heading": "अनुमानित हीमोग्लोबिन",
    "result.unit": "g/dL",
    "result.interval": "रेंज {lo}–{hi} g/dL",
    "result.band_ok": "इस मॉडल के लिए सामान्य सीमा में",
    "result.band_warn": "सामान्य सीमा से नीचे",
    "result.band_risk": "सामान्य सीमा से बहुत नीचे",
    "result.low_high": "यह अनुमान इस मॉडल की निर्णय-सीमा से {band} है।",
    "result.disclaimer":
      "यह जाँच का अनुमान है, निदान नहीं। किसी भी निर्णय से पहले लैब में खून की जाँच से पुष्टि करें।",

    "abstain.heading": "यह फ़ोटो पढ़ी नहीं जा सकी",
    "abstain.body":
      "फ़ोटो इतनी साफ़ नहीं थी कि भरोसेमंद अनुमान मिल सके। एक-सी रोशनी में जाएँ, चमक और परछाईं से बचें, फ़ोन स्थिर रखें। नतीजे में कोई खराबी नहीं — बस यह नापी नहीं जा सकी।",

    "error.camera": "कैमरे की अनुमति नहीं मिली या कोई कैमरा उपलब्ध नहीं है। सेटिंग्स में कैमरा चालू कर सकते हैं।",
    "error.generic": "छवि पढ़ते समय कुछ गड़बड़ हुई। दोबारा कोशिश करें।",

    "notice.unavailable": "इस बिल्ड में अभी मॉडल शामिल नहीं है।",
    "notice.unavailable_body":
      "वैलिडेशन स्वीप मॉडल चुनकर उसे ONNX में निर्यात करती है। तब तक यह स्क्रीन कोई संख्या नहीं देती।",

    "evidence.title": "प्रमाण",
    "evidence.intro":
      "इस पृष्ठ का हर आँकड़ा इस रिपॉज़िटरी में दर्ज फ़्रीज़ मूल्यांकन से आता है। यहाँ कुछ भी दिखावटी नहीं है।",
    "evidence.pending":
      "वैलिडेशन स्वीप द्वारा मॉडल चुने जाने और फ़्रीज़ टेस्ट पूरा होने पर नतीजे यहाँ प्रकाशित होंगे।",
    "evidence.corpus_h": "कॉर्पस",
    "evidence.roi": "क्षेत्र",
    "evidence.backbone": "बैकबोन",
    "evidence.patients": "मरीज़",
    "evidence.sites": "साइट",
    "evidence.features": "फ़ीचर चौड़ाई",
    "evidence.excluded": "हटाए गए फ़ीचर",
    "evidence.none": "कोई नहीं",
    "evidence.heads_h": "प्रति हेड टेस्ट मेट्रिक्स",
    "evidence.head_row": "MAE {mae} · RMSE {rmse} · R² {r2} · ±1 के भीतर {within}",
    "evidence.provenance_h": "स्रोत-विवरण",
    "evidence.commit": "कमिट",
    "evidence.fingerprint": "फ़ीचर फ़िंगरप्रिंट",
    "evidence.generated": "बनाया गया",
    "evidence.dirty": "वर्किंग ट्री में बदलाव",
    "evidence.yes": "हाँ",
    "evidence.no": "नहीं",

    "method.title": "तरीका और सीमाएँ",
    "method.how_h": "यह कैसे काम करता है",
    "method.how_b":
      "कैमरा कंजंक्टिवा की फ़ोटो लेता है — निचली पलक के अंदर की गुलाबी परत, जिसमें उंगली की तरह ही केशिका रक्त होता है। एक रिग्रेशन मॉडल उस छवि से हीमोग्लोबिन का अनुमान g/dL में लगाता है। पूरी गणना आपके डिवाइस पर होती है।",
    "method.cannot_h": "यह क्या नहीं कर सकता",
    "method.cannot_b":
      "यह एक स्क्रीनिंग सहायक है, निदान नहीं। यह एनीमिया की पुष्टि नहीं कर सकता, न ही इसे नकार सकता है, और लैब खून जाँच की जगह नहीं ले सकता। सामान्य सीमा से बाहर नतीजा खून की जाँच कराने का कारण है, इलाज शुरू करने का नहीं।",
    "method.emergency_b": "आपात स्थिति में या इलाज का फ़ैसला लेने के लिए इसका उपयोग न करें।",
    "method.fair_h": "निष्पक्षता",
    "method.fair_b":
      "प्रदर्शन अलग-अलग त्वचा रंग (ITA°) और अलग-अलग साइट के हिसाब से बताया जाता है। जहाँ मॉडल समान रूप से भरोसेमंद नहीं है, वहाँ अंतर को छिपाकर औसत नहीं किया जाता।",

    "about.title": "परिचय",
    "about.dataset": "डेटासेट",
    "about.licence": "लाइसेंस",
    "about.runtime": "रनटाइम",
    "about.build": "बिल्ड",
    "about.privacy": "कोई फ़ोटो इस डिवाइस से बाहर नहीं जाती। ऐप छवि डेटा के साथ कोई नेटवर्क अनुरोध नहीं करता।",
  },
};

let current = "en";

export function getLang() {
  return current;
}

export function setLang(lang) {
  current = lang === "hi" ? "hi" : "en";
  if (typeof document !== "undefined") {
    document.documentElement.lang = current;
  }
  return current;
}

/** Translate a key, interpolating `{name}` placeholders from `vars`. */
export function t(key, vars) {
  const table = MESSAGES[current] || MESSAGES.en;
  let value = table[key];
  if (value === undefined) value = MESSAGES.en[key];
  if (value === undefined) return key;
  if (vars) {
    value = value.replace(/\{(\w+)\}/g, (_, name) => (name in vars ? String(vars[name]) : `{${name}}`));
  }
  return value;
}

/** Walk `root` and fill every tagged node from the current language. */
export function applyI18n(root = document) {
  for (const node of root.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  for (const node of root.querySelectorAll("[data-i18n-aria]")) {
    node.setAttribute("aria-label", t(node.dataset.i18nAria));
  }
}
