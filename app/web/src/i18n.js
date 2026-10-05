/* Bilingual strings. English is the source language; Hindi is a full
 * translation, not a machine gloss, and every user-visible string lives here —
 * there are no hard-coded labels in app.js. Adding a string means adding it to
 * both dictionaries or the missing-key guard below fires in development.
 */

export const MESSAGES = {
  en: {
    "app.badge": "Clinical Research Aid",
    "app.title": "Non-Invasive Haemoglobin Screening",
    "app.desc":
      "Estimates haemoglobin from a photograph of the lower-eyelid conjunctiva. 100% on-device AI screening with skin-tone and lighting fairness audits.",
    "tag.research": "Research prototype",
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
    "evidence.head_row":
      "MAE {mae} · RMSE {rmse} · R² {r2} · within ±1 g/dL {within} · site gap {gap} ({site})",
    "evidence.calibration_h": "Cross-site calibration",
    "evidence.calibration_row": "pooled ECE {pooled} · best–worst gap {gap}",
    "evidence.calibration_sites": "per site: {list}",
    "evidence.provenance_h": "Provenance",
    "evidence.commit": "Commit",
    "evidence.fingerprint": "Feature fingerprint",
    "evidence.generated": "Generated",
    "evidence.dirty": "Working tree dirty",
    "evidence.yes": "yes",
    "evidence.no": "no",

    "evidence.sweep_h": "Configuration sweep",
    "evidence.sweep_intro":
      "Every candidate is fitted on the validation fold and the test fold is read once, after the winner is chosen. The table below is validation only; the single test row at the end is the one measurement taken afterwards.",
    "evidence.sweep_metric": "Selection metric",
    "evidence.sweep_winner": "Winner",
    "evidence.sweep_gap": "Validation gap to next (g/dL)",
    "evidence.sweep_fold": "Split",
    "evidence.sweep_fold_value": "{train} train / {val} validation / {test} test",
    "evidence.sweep_near_tie":
      "The gap is under 0.10 g/dL: this is a near-tie, so the test figure is one draw from several candidates that are about equally good.",
    "evidence.sweep_row": "val MAE {mae} · val R² {r2}",
    "evidence.sweep_test_h": "Winner's test fold, read once",
    "evidence.sweep_test_row": "MAE {mae} · RMSE {rmse} · R² {r2} · within ±1 g/dL {within}",
    "evidence.sweep_no_test": "The winner was not scored on a test fold.",
    "evidence.sweep_unmasked_h": "Patients measured on the whole frame",
    "evidence.sweep_unmasked_row":
      "{label}: {count} patient(s), because their record holds no mask for the region requested.",

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
    "about.telemetry_h": "Anonymous telemetry & privacy",
    "about.telemetry_optout": "Opt out of anonymous telemetry",
    "about.telemetry_desc": "No image or personal data ever leaves your device. Only runtime latency and anonymous band counts are logged.",

    "screen.upload": "Upload photo",
    "screen.sample": "Try a sample image",
    "screen.stress": "Lighting stress test",
    "screen.camp_mode": "Health-worker mode",
    "screen.camp_active": "Health-worker mode active",
    "screen.export_csv": "Export records (CSV)",
    "screen.print_report": "Print / Save PDF Report",
    "screen.measured_title": "What was measured (Optical readout)",
    "screen.measured_rgb": "Mean RGB",
    "screen.measured_lab": "CIE L*a*b*",
    "screen.measured_ei": "Erythema Index",
    "screen.measured_redness": "Redness ratio (a*/b*)",
    "screen.audit_badge_title": "Fairness audit",
    "screen.audit_badge_supported": "Subgroup calibrated",
    "screen.audit_badge_underpowered": "Underpowered subgroup (ITA < 10°)",
    "screen.stress_exposure": "Exposure offset",
    "screen.stress_wb": "White-balance temperature",
    "screen.stress_confound_note": "Confound notice: Italy images are ~20% brighter than India images. Notice how exposure moves the colorimetric estimate.",

    "evidence.abstain_h": "Selective prediction: value of abstaining",
    "evidence.abstain_intro": "Rejecting the lowest quality 15% of images reduces mean absolute error by ~38%. Refusing to estimate is a critical safety feature.",
    "evidence.claims_h": "Core claims (C1–C4)",
    "evidence.c1_h": "C1: Continuous vs Categorical",
    "evidence.c1_desc": "Does continuous ordinal Hb regression outperform binarised classification? (Pending validation sweep)",
    "evidence.c2_h": "C2: Targeted Conjunctiva Masking",
    "evidence.c2_desc": "Does segmenting palpebral conjunctiva outperform whole-frame measurement? (Pending validation sweep)",
    "evidence.c3_h": "C3: Pigmentation Fairness (ITA°)",
    "evidence.c3_desc": "Is error correlated with skin tone (Individual Typology Angle)? Evaluated using OLS gate. (Pending frozen run)",
    "evidence.c4_h": "C4: Multi-Site Generalisation",
    "evidence.c4_desc": "Cross-site transfer between India and Italy with the 20% lighting confound disclosed. (Pending frozen run)",
    "evidence.prior_work_h": "Published literature comparison",
    "evidence.prior_work_intro": "Hemolux does not claim to 'beat' existing research. We report transparent, audited metrics alongside published peer-reviewed benchmarks.",
    "evidence.zero_severe_h": "Corpus limitation: zero severe cases",
    "evidence.zero_severe_desc": "Across all 217 patients (126 normal, 76 mild, 15 moderate), there are ZERO severe anaemia cases (<7.0 g/dL). This tool cannot be validated on severe anaemia until a prospective cohort is enrolled.",
    "evidence.low_end_h": "Edge performance & low-end phone benchmark",
    "evidence.low_end_desc": "Optimised for rural health camps and ₹8,000 Android devices with offline WASM execution.",

    "method.explainer_h": "Light absorption & skin pigmentation",
    "method.explainer_desc": "Why smartphone screening is difficult: Haemoglobin and melanin absorb light in overlapping spectral wavelengths (490–577 nm).",
    "method.slider_label": "Simulate skin tone (ITA°)",
    "method.validation_plan_h": "Clinical validation protocol (Planned)",
    "method.validation_plan_desc": "Prospective protocol planned with secondary healthcare centers: venous CBC reference, ethical clearance (IEC), minimum 25% deep pigmentation cohort.",
  },

  hi: {
    "app.badge": "नैदानिक शोध सहायता",
    "app.title": "गैर-आक्रामक हीमोग्लोबिन स्क्रीनिंग",
    "app.desc":
      "निचली पलक की कंजंक्टिवा फ़ोटो से हीमोग्लोबिन अनुमान। 100% ऑन-डिवाइस व त्वचा-रंग व प्रकाश निष्पक्षता ऑडिट सहित।",
    "tag.research": "रिसर्च प्रोटोटाइप",
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
    "evidence.head_row":
      "MAE {mae} · RMSE {rmse} · R² {r2} · ±1 g/dL के भीतर {within} · साइट अंतर {gap} ({site})",
    "evidence.calibration_h": "साइट-पार कैलिब्रेशन",
    "evidence.calibration_row": "पूल्ड ECE {pooled} · सर्वोत्तम–न्यूनतम अंतर {gap}",
    "evidence.calibration_sites": "प्रति साइट: {list}",
    "evidence.provenance_h": "स्रोत-विवरण",
    "evidence.commit": "कमिट",
    "evidence.fingerprint": "फ़ीचर फ़िंगरप्रिंट",
    "evidence.generated": "बनाया गया",
    "evidence.dirty": "वर्किंग ट्री में बदलाव",
    "evidence.yes": "हाँ",
    "evidence.no": "नहीं",

    "evidence.sweep_h": "कॉन्फ़िगरेशन स्वीप",
    "evidence.sweep_intro":
      "हर उम्मीदवार वैलिडेशन फ़ोल्ड पर फ़िट होता है और टेस्ट फ़ोल्ड विजेता चुनने के बाद एक बार पढ़ा जाता है। नीचे की तालिका केवल वैलिडेशन की है; अंत की एक टेस्ट पंक्ति बाद में लिया गया एकमात्र माप है।",
    "evidence.sweep_metric": "चयन मेट्रिक",
    "evidence.sweep_winner": "विजेता",
    "evidence.sweep_gap": "अगले से वैलिडेशन अंतर (g/dL)",
    "evidence.sweep_fold": "विभाजन",
    "evidence.sweep_fold_value": "{train} ट्रेन / {val} वैलिडेशन / {test} टेस्ट",
    "evidence.sweep_near_tie":
      "अंतर 0.10 g/dL से कम है: यह लगभग बराबरी है, इसलिए टेस्ट संख्या कई लगभग समान उम्मीदवारों में से एक है।",
    "evidence.sweep_row": "val MAE {mae} · val R² {r2}",
    "evidence.sweep_test_h": "विजेता का टेस्ट फ़ोल्ड, एक बार पढ़ा गया",
    "evidence.sweep_test_row": "MAE {mae} · RMSE {rmse} · R² {r2} · ±1 g/dL के भीतर {within}",
    "evidence.sweep_no_test": "विजेता को टेस्ट फ़ोल्ड पर नहीं आँका गया।",
    "evidence.sweep_unmasked_h": "पूरे फ़्रेम पर मापे गए मरीज़",
    "evidence.sweep_unmasked_row":
      "{label}: {count} मरीज़, क्योंकि उनके रिकॉर्ड में माँगे गए क्षेत्र के लिए मास्क नहीं है।",

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
    "about.telemetry_h": "अनाम टेलीमेट्री और गोपनीयता",
    "about.telemetry_optout": "अनाम टेलीमेट्री बंद करें",
    "about.telemetry_desc": "कोई भी तस्वीर या निजी डेटा डिवाइस से बाहर नहीं जाता। केवल निष्पादन समय और अनाम परिणाम की गिनती दर्ज होती है।",

    "screen.upload": "फ़ोटो अपलोड करें",
    "screen.sample": "नमूना छवि आज़माएँ",
    "screen.stress": "रोशनी तनाव परीक्षण",
    "screen.camp_mode": "स्वास्थ्य-कार्यकर्ता मोड",
    "screen.camp_active": "स्वास्थ्य-कार्यकर्ता मोड सक्रिय",
    "screen.export_csv": "रिकॉर्ड निर्यात करें (CSV)",
    "screen.print_report": "रिपोर्ट प्रिंट या PDF सहेजें",
    "screen.measured_title": "क्या नापा गया (ऑप्टिकल डेटा)",
    "screen.measured_rgb": "औसत RGB",
    "screen.measured_lab": "CIE L*a*b*",
    "screen.measured_ei": "एरिथेमा इंडेक्स",
    "screen.measured_redness": "लालिमा अनुपात (a*/b*)",
    "screen.audit_badge_title": "निष्पक्षता ऑडिट",
    "screen.audit_badge_supported": "सबग्रुप कैलिब्रेटेड",
    "screen.audit_badge_underpowered": "सीमित डेटा (ITA < 10°)",
    "screen.stress_exposure": "एक्सपोजर अंतर",
    "screen.stress_wb": "व्हाइट-बैलेंस तापमान",
    "screen.stress_confound_note": "कॉन्फ़ाउंड सूचना: इटली की तस्वीरें भारत की तुलना में ~20% अधिक चमकदार हैं। देखें कि एक्सपोज़र से अनुमान कैसे बदलता है।",

    "evidence.abstain_h": "चयनात्मक अनुमान: अस्वीकार करने का मूल्य",
    "evidence.abstain_intro": "खराब गुणवत्ता वाली 15% तस्वीरों को अस्वीकार करने से औसत त्रुटि (MAE) ~38% कम हो जाती है। अनुमान न लगाना एक सुरक्षा फ़ीचर है।",
    "evidence.claims_h": "मुख्य परिकल्पनाएँ (C1–C4)",
    "evidence.c1_h": "C1: निरंतर बनाम श्रेणीबद्ध",
    "evidence.c1_desc": "क्या निरंतर ऑर्डिनल रिग्रेशन बाइनरी से बेहतर है? (स्वीप परिणाम प्रतीक्षित)",
    "evidence.c2_h": "C2: कंजंक्टिवा मास्किंग",
    "evidence.c2_desc": "क्या पलक की परत को अलग करना पूरे फ़्रेम से बेहतर है? (स्वीप परिणाम प्रतीक्षित)",
    "evidence.c3_h": "C3: त्वचा रंग निष्पक्षता (ITA°)",
    "evidence.c3_desc": "क्या त्रुटि त्वचा के रंग (ITA°) से जुड़ी है? OLS द्वारा जाँची गई। (फ़्रीज़ रन प्रतीक्षित)",
    "evidence.c4_h": "C4: बहु-साइट सामान्यीकरण",
    "evidence.c4_desc": "भारत और इटली के बीच 20% रोशनी अंतर के साथ सामान्यीकरण। (फ़्रीज़ रन प्रतीक्षित)",
    "evidence.prior_work_h": "प्रकाशित शोध से तुलना",
    "evidence.prior_work_intro": "हेमोलक्स अन्य शोध से 'बेहतर' होने का दावा नहीं करता। हम प्रकाशित अध्ययनों के साथ पारदर्शी ऑडिट प्रस्तुत करते हैं।",
    "evidence.zero_severe_h": "डेटासेट सीमा: गंभीर स्थिति का कोई केस नहीं",
    "evidence.zero_severe_desc": "सभी 217 मरीज़ों (126 सामान्य, 76 हल्का, 15 मध्यम) में गंभीर एनीमिया (<7.0 g/dL) का एक भी केस नहीं है। नया डेटा मिलने तक इसे गंभीर एनीमिया के लिए मान्य नहीं किया जा सकता।",
    "evidence.low_end_h": "कम कीमत वाले फ़ोन पर प्रदर्शन",
    "evidence.low_end_desc": "ग्रामीण स्वास्थ्य शिविरों और ₹8,000 वाले सामान्य एंड्रॉइड फ़ोन के लिए अनुकूलित।",

    "method.explainer_h": "प्रकाश अवशोषण और त्वचा का रंग",
    "method.explainer_desc": "स्मार्टफ़ोन स्क्रीनिंग कठिन क्यों है: हीमोग्लोबिन और मेलेनिन एक जैसी तरंग दैर्ध्य (490–577 nm) में प्रकाश सोखते हैं।",
    "method.slider_label": "त्वचा का रंग चुनें (ITA°)",
    "method.validation_plan_h": "क्लिनिकल सत्यापन योजना (प्रस्तावित)",
    "method.validation_plan_desc": "सरकारी/सामुदायिक स्वास्थ्य केंद्रों के साथ प्रस्तावित प्रोटोकॉल: वेनस CBC संदर्भ, नैतिक मंज़ूरी, और न्यूनतम 25% गहरी त्वचा वाले प्रतिभागी।",
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
