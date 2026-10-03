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

  return {
    nPatients: number(corpus.n_patients),
    sites: Object.keys(perSite).sort(),
    featureDim: number(corpus.feature_dim),
    roi: corpus.roi == null ? null : String(corpus.roi),
    backbone: corpus.backbone == null ? null : String(corpus.backbone),
    excluded: Array.isArray(corpus.excluded_features) ? corpus.excluded_features.map(String) : [],
    heads,
    commit: provenance.git_commit == null ? null : String(provenance.git_commit),
    fingerprint: provenance.feature_fingerprint == null ? null : String(provenance.feature_fingerprint),
    generated: provenance.generated_at == null ? null : String(provenance.generated_at),
    dirty: provenance.git_dirty === true,
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
        }),
      ),
    );
  }

  container.append(heading(t("evidence.provenance_h")));
  container.append(line(t("evidence.commit"), data.commit ?? "—"));
  container.append(line(t("evidence.fingerprint"), data.fingerprint ?? "—"));
  container.append(line(t("evidence.generated"), data.generated ?? "—"));
  container.append(line(t("evidence.dirty"), data.dirty ? t("evidence.yes") : t("evidence.no")));

  return true;
}
