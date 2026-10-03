/* Evidence tab: the frozen validation run, rendered from the real report.
 *
 * There is exactly one source of truth. `scripts/build_site.mjs` stages
 * `artifacts/reports/*.json` into `/data/`, and this module reads whatever the
 * last real run wrote. No figure is typed into the page, and when no report is
 * staged the panel keeps its "pending" notice rather than inventing a table.
 *
 * `summarise` is pure so it can be tested without a DOM; `renderEvidence` only
 * builds elements from that summary.
 */

/** Presentation order. The screening estimate is the ordinal/regression head,
 * so those lead; anything unrecognised falls to the end rather than to the top. */
const HEAD_ORDER = ["ordinal", "regression", "severity", "multitask", "binary"];

function rank(head) {
  const index = HEAD_ORDER.indexOf(head);
  return index === -1 ? HEAD_ORDER.length : index;
}

function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

/**
 * Reduce a `results.json` payload to the fields the panel shows.
 * @returns {object | null} null when there is nothing real to show.
 */
export function summarise(report) {
  if (!report || typeof report !== "object") return null;
  const rows = Array.isArray(report.single_split) ? report.single_split : [];
  if (rows.length === 0) return null;

  const corpus = report.corpus || {};
  const provenance = report.provenance || {};
  const perSite = corpus.per_site && typeof corpus.per_site === "object" ? corpus.per_site : {};

  const heads = rows
    .map((row) => ({
      head: String(row.head),
      n: number(row.n_test),
      mae: number(row.mae),
      rmse: number(row.rmse),
      r2: number(row.r2),
      bias: number(row.bias),
      within1: number(row.within_1),
      within2: number(row.within_2),
      siteMaeGap: number(row.site_mae_gap),
      worstSite: row.worst_site == null ? null : String(row.worst_site),
    }))
    .sort((a, b) => rank(a.head) - rank(b.head));

  // The site audit carries per-site thresholds and, where the head emits a
  // probability, per-site calibration. The pooled ECE is the number that hides
  // the finding, so both the pooled figure and the best-to-worst gap are kept,
  // and the confound sentence travels with them in the artefact rather than
  // being paraphrased here.
  const auditRaw = report.site_audit && typeof report.site_audit === "object" ? report.site_audit : {};
  const siteAudit = {};
  let confound = null;
  for (const head of heads) {
    const block = auditRaw[head.head];
    if (!block || typeof block !== "object" || block.available !== true) continue;
    if (confound === null && typeof block.confound === "string") confound = block.confound;
    const cal = block.calibration && block.calibration.available === true ? block.calibration : null;
    const rawPerSite = cal && cal.per_site && typeof cal.per_site === "object" ? cal.per_site : {};
    const perSiteEce = Object.keys(rawPerSite)
      .sort()
      .map((site) => {
        const curve = rawPerSite[site] || {};
        return { site, ece: number(curve.ece), n: number(curve.n) };
      });
    siteAudit[head.head] = {
      calibratable: cal !== null,
      pooledEce: cal ? number(cal.pooled && cal.pooled.ece) : null,
      gap: cal ? number(cal.max_ece_gap) : null,
      perSite: perSiteEce,
    };
  }

  return {
    nPatients: number(corpus.n_patients),
    sites: Object.keys(perSite).sort(),
    featureDim: number(corpus.feature_dim),
    roi: corpus.roi == null ? null : String(corpus.roi),
    backbone: corpus.backbone == null ? null : String(corpus.backbone),
    excluded: Array.isArray(corpus.excluded_features) ? corpus.excluded_features.map(String) : [],
    heads,
    siteAudit,
    confound,
    commit: provenance.git_commit == null ? null : String(provenance.git_commit),
    fingerprint: provenance.feature_fingerprint == null ? null : String(provenance.feature_fingerprint),
    generated: provenance.generated_at == null ? null : String(provenance.generated_at),
    dirty: provenance.git_dirty === true,
  };
}

/**
 * Reduce a `sweep.json` payload to the fields the panel shows.
 *
 * The sweep exists so the test fold is read exactly once, after the winner is
 * chosen. The panel has to make that visible rather than present the winner as
 * if it had been scored directly, so the candidate table is validation-only and
 * the single test block is labelled as the one number read afterwards.
 *
 * @returns {object | null} null when there is no sweep to show.
 */
export function summariseSweep(report) {
  if (!report || typeof report !== "object") return null;
  const candidates = Array.isArray(report.candidates) ? report.candidates : [];
  if (candidates.length === 0) return null;

  const fold = report.fold && typeof report.fold === "object" ? report.fold : {};
  const provenance = report.provenance || {};
  const winner = report.winner == null ? null : String(report.winner);
  const gap = number(report.selection_gap);

  const rows = candidates
    .map((row) => ({
      label: String(row.label),
      valMae: number(row.val_mae),
      valR2: number(row.val_r2),
      selected: row.selected === true || (winner !== null && String(row.label) === winner),
      nUnmasked: number(row.n_unmasked),
    }))
    .sort((a, b) => {
      // The winner leads, then ascending validation MAE, then label. The same
      // ordering rule the sweep itself uses, so the panel cannot disagree with
      // `sweep.json` about which candidate came first.
      if (a.selected !== b.selected) return a.selected ? -1 : 1;
      const am = Number.isFinite(a.valMae) ? a.valMae : Infinity;
      const bm = Number.isFinite(b.valMae) ? b.valMae : Infinity;
      if (am !== bm) return am - bm;
      return a.label < b.label ? -1 : a.label > b.label ? 1 : 0;
    });

  const unmaskedByLabel = report.n_unmasked_by_label;
  const unmasked =
    unmaskedByLabel && typeof unmaskedByLabel === "object"
      ? Object.entries(unmaskedByLabel)
          .map(([label, count]) => ({ label: String(label), count: Math.trunc(number(count)) }))
          .filter((entry) => entry.count > 0)
          .sort((a, b) => (a.label < b.label ? -1 : 1))
      : [];

  const test = report.test && typeof report.test === "object" ? report.test : null;

  return {
    metric: report.metric == null ? "mae" : String(report.metric),
    winner,
    gap,
    // A gap under a tenth of a gram per decilitre is inside the clinical target
    // and means the grid did not separate the top candidates. The panel says so
    // instead of presenting one draw as a settled choice.
    nearTie: Number.isFinite(gap) && gap < 0.1,
    nCandidates: candidates.length,
    fold: { train: number(fold.n_train), val: number(fold.n_val), test: number(fold.n_test) },
    rows,
    unmasked,
    test: test
      ? {
          n: number(test.n),
          mae: number(test.mae),
          rmse: number(test.rmse),
          r2: number(test.r2),
          within1: number(test.within_1),
          within2: number(test.within_2),
          bias: number(test.bias),
        }
      : null,
    commit: provenance.git_commit == null ? null : String(provenance.git_commit),
    generated: provenance.generated_at == null ? null : String(provenance.generated_at),
  };
}

function line(name, value) {
  const wrap = document.createElement("div");
  wrap.className = "metric-row";
  const key = document.createElement("span");
  key.className = "metric-row__name";
  key.textContent = name;
  const val = document.createElement("span");
  val.className = "metric-row__value";
  val.textContent = value;
  wrap.append(key, val);
  return wrap;
}

function heading(text) {
  const node = document.createElement("h2");
  node.textContent = text;
  return node;
}

function paragraph(text) {
  const node = document.createElement("p");
  node.className = "faint";
  node.textContent = text;
  return node;
}

function fixed(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function percent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

/**
 * @param {HTMLElement} container
 * @param {(key: string, vars?: object) => string} t
 * @param {object} report a parsed results.json
 * @returns {boolean} true when the panel was filled
 */
export function renderEvidence(container, t, report) {
  const data = summarise(report);
  if (!data) return false;

  container.replaceChildren();

  container.append(heading(t("evidence.corpus_h")));
  container.append(line(t("evidence.roi"), data.roi ?? "—"));
  container.append(line(t("evidence.backbone"), data.backbone ?? "—"));
  container.append(line(t("evidence.patients"), String(data.nPatients)));
  container.append(line(t("evidence.sites"), data.sites.length ? data.sites.join(", ") : "—"));
  container.append(line(t("evidence.features"), String(data.featureDim)));
  container.append(
    line(t("evidence.excluded"), data.excluded.length ? data.excluded.join(", ") : t("evidence.none")),
  );

  container.append(heading(t("evidence.heads_h")));
  for (const head of data.heads) {
    container.append(
      line(
        head.head,
        t("evidence.head_row", {
          mae: fixed(head.mae),
          rmse: fixed(head.rmse),
          r2: fixed(head.r2),
          within: percent(head.within1),
          gap: fixed(head.siteMaeGap),
          site: head.worstSite ?? "—",
        }),
      ),
    );
  }

  const calibratable = data.heads.filter((head) => {
    const audit = data.siteAudit[head.head];
    return Boolean(audit && audit.calibratable);
  });
  if (calibratable.length) {
    container.append(heading(t("evidence.calibration_h")));
    // The confound sentence is the artefact's own, rendered verbatim: it has to
    // appear wherever a site contrast does, and paraphrasing it in the UI would
    // let the two drift apart.
    if (data.confound) container.append(paragraph(data.confound));
    for (const head of calibratable) {
      const audit = data.siteAudit[head.head];
      container.append(
        line(
          head.head,
          t("evidence.calibration_row", {
            pooled: fixed(audit.pooledEce, 3),
            gap: fixed(audit.gap, 3),
          }),
        ),
      );
      const list = audit.perSite.map((entry) => `${entry.site} ${fixed(entry.ece, 3)}`).join(" · ");
      container.append(paragraph(t("evidence.calibration_sites", { list })));
    }
  }

  container.append(heading(t("evidence.provenance_h")));
  container.append(line(t("evidence.commit"), data.commit ?? "—"));
  container.append(line(t("evidence.fingerprint"), data.fingerprint ?? "—"));
  container.append(line(t("evidence.generated"), data.generated ?? "—"));
  container.append(line(t("evidence.dirty"), data.dirty ? t("evidence.yes") : t("evidence.no")));

  return true;
}

/**
 * Append the configuration-sweep block to the same container.
 *
 * Deliberately append-only: it never clears, so the caller decides whether the
 * panel starts from the corpus block, from the sweep alone, or from nothing. It
 * returns false and touches nothing when there is no sweep, which is what keeps
 * the served "pending" notice in place on a checkout where only `results.json`
 * was staged.
 *
 * @param {HTMLElement} container
 * @param {(key: string, vars?: object) => string} t
 * @param {object} report a parsed sweep.json
 * @returns {boolean} true when the block was appended
 */
export function renderSweep(container, t, report) {
  const data = summariseSweep(report);
  if (!data) return false;

  container.append(heading(t("evidence.sweep_h")));
  container.append(paragraph(t("evidence.sweep_intro")));
  container.append(line(t("evidence.sweep_metric"), data.metric));
  container.append(line(t("evidence.sweep_winner"), data.winner ?? "—"));
  container.append(line(t("evidence.sweep_gap"), fixed(data.gap, 4)));
  container.append(
    line(
      t("evidence.sweep_fold"),
      t("evidence.sweep_fold_value", {
        train: String(data.fold.train),
        val: String(data.fold.val),
        test: String(data.fold.test),
      }),
    ),
  );
  if (data.nearTie) container.append(paragraph(t("evidence.sweep_near_tie")));

  for (const row of data.rows) {
    container.append(
      line(
        row.selected ? `${row.label}  ←` : row.label,
        t("evidence.sweep_row", { mae: fixed(row.valMae, 4), r2: fixed(row.valR2, 4) }),
      ),
    );
  }

  container.append(heading(t("evidence.sweep_test_h")));
  if (data.test) {
    container.append(
      line(
        data.winner ?? t("evidence.sweep_winner"),
        t("evidence.sweep_test_row", {
          mae: fixed(data.test.mae),
          rmse: fixed(data.test.rmse),
          r2: fixed(data.test.r2),
          within: percent(data.test.within1),
        }),
      ),
    );
  } else {
    container.append(paragraph(t("evidence.sweep_no_test")));
  }

  if (data.unmasked.length) {
    container.append(heading(t("evidence.sweep_unmasked_h")));
    for (const entry of data.unmasked) {
      container.append(
        paragraph(t("evidence.sweep_unmasked_row", { label: entry.label, count: String(entry.count) })),
      );
    }
  }

  container.append(line(t("evidence.commit"), data.commit ?? "—"));
  container.append(line(t("evidence.generated"), data.generated ?? "—"));

  return true;
}
