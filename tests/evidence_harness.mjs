/* Exercises the Evidence reducers without a DOM.
 *
 * The report shapes are copied from `training.build_report` (results.json) and
 * `sweep.build_sweep_report` (sweep.json). tests/test_evidence_js.py checks the
 * fields the panel depends on, including presentation order for both.
 */
import { summarise, summariseSweep } from "../app/web/src/evidence.js";

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

const sweep = {
  provenance: {
    git_commit: "def456",
    feature_fingerprint: "859399ade17b39e8",
    generated_at: "2026-10-03T12:30:00Z",
    git_dirty: false,
  },
  metric: "mae",
  fold: { name: "sweep-seed42", n_train: 131, n_val: 43, n_test: 43 },
  candidates: [
    {
      label: "colour / forniceal / balanced",
      kind: "colour",
      roi: "forniceal",
      balance: "balanced",
      head: "regression",
      feature_dim: 12,
      val_mae: 5.704,
      val_r2: -8.511,
      best_epoch: 40,
      seconds: 0.3,
      n_unmasked: 6,
      selected: false,
    },
    {
      label: "deep / palpebral",
      kind: "deep",
      roi: "palpebral",
      balance: "balanced",
      head: "regression",
      feature_dim: 1024,
      val_mae: 1.71,
      val_r2: 0.21,
      best_epoch: 38,
      seconds: 3.1,
      n_unmasked: 0,
      selected: true,
    },
    {
      label: "colour / forniceal / raw",
      kind: "colour",
      roi: "forniceal",
      balance: "raw",
      head: "regression",
      feature_dim: 12,
      val_mae: 1.99,
      val_r2: -0.2,
      best_epoch: 40,
      seconds: 0.3,
      n_unmasked: 6,
      selected: false,
    },
  ],
  n_unmasked_by_label: {
    "colour / forniceal / balanced": 6,
    "colour / forniceal / raw": 6,
  },
  winner: "deep / palpebral",
  selection_gap: 0.28,
  test_read: "once, after selection",
  excluded_features: ["erythema_index"],
  test: {
    n: 43,
    mae: 1.62,
    rmse: 2.11,
    r2: 0.24,
    evs: 0.25,
    pearson_r: 0.52,
    bias: 0.1,
    loa_lower: -3.9,
    loa_upper: 4.1,
    within_1: 0.6,
    within_2: 0.88,
  },
  notes: ["test fold scored once, after selection"],
};

const sweepOut = summariseSweep(sweep);

process.stdout.write(
  JSON.stringify({
    out: summarise(report),
    headOrder: summarise(report).heads.map((head) => head.head),
    empty: summarise(null),
    noRows: summarise({ single_split: [] }),
    sweepOut,
    sweepOrder: sweepOut.rows.map((row) => row.label),
    sweepNearTie: summariseSweep({ ...sweep, selection_gap: 0.05 }).nearTie,
    sweepEmpty: summariseSweep(null),
    sweepNoRows: summariseSweep({ candidates: [] }),
  }),
);
