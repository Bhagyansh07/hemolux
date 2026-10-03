/* Exercises the Evidence reducer without a DOM.
 *
 * The report shape is copied from `training.build_report`: provenance, corpus
 * and a `single_split` list of per-head test rows. tests/test_evidence_js.py
 * checks the fields the panel depends on, including the presentation order.
 */
import { summarise } from "../app/web/src/evidence.js";

const report = {
  provenance: {
    git_commit: "abc123",
    feature_fingerprint: "859399ade17b39e8",
    generated_at: "2026-10-03T12:00:00Z",
    git_dirty: false,
  },
  corpus: {
    n_patients: 217,
    roi: "forniceal_palpebral",
    backbone: "mobilenetv3_small_100",
    feature_dim: 13,
    excluded_features: ["erythema_index"],
    per_site: { Italy: 122, India: 95 },
  },
  single_split: [
    {
      head: "binary",
      n_test: 43,
      mae: 0.44,
      rmse: 0.5,
      r2: 0.1,
      bias: 0.01,
      within_1: 0.9,
      within_2: 1.0,
      site_mae_gap: 0.2,
      worst_site: "India",
    },
    {
      head: "ordinal",
      n_test: 43,
      mae: 1.93,
      rmse: 2.6,
      r2: -0.04,
      bias: 0.41,
      within_1: 0.349,
      within_2: 0.651,
      site_mae_gap: 0.5,
      worst_site: "India",
    },
    {
      head: "regression",
      n_test: 43,
      mae: 1.9,
      rmse: 2.5,
      r2: -0.03,
      bias: 0.4,
      within_1: 0.35,
      within_2: 0.66,
      site_mae_gap: 0.4,
      worst_site: "India",
    },
  ],
};

const out = summarise(report);
process.stdout.write(
  JSON.stringify({
    out,
    headOrder: out.heads.map((head) => head.head),
    empty: summarise(null),
    noRows: summarise({ single_split: [] }),
  }),
);
