"""Tests for ``scripts/make_figures.py``.

The script exists because the project declares ``matplotlib`` as a runtime
dependency and then nothing imported it, while the production checklist required
every number in ``EVALS.md`` to regenerate from a script. A figure generator that
has never been run on the schema it reads is the failure mode being avoided here:
the run's JSON is the input contract, and if a key moves the script should fail in
CI rather than in front of a reader.

These tests use a hand-built payload matching that contract. They are not a second
copy of a real run -- there are no real numbers below and none are asserted as if
they were measured -- only the shape of the report and the formatting rules.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


figures = _load("make_figures")


def _band(
    n: int, positives: int, sensitivity: float, specificity: float, verdict: str
) -> dict[str, Any]:
    return {
        "band": "anaemia",
        "threshold": None,
        "n": n,
        "positives": positives,
        "tp": round(positives * sensitivity),
        "tn": round((n - positives) * specificity),
        "fp": n - positives - round((n - positives) * specificity),
        "fn": positives - round(positives * sensitivity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": 0.64,
        "npv": 0.78,
        "verdict": verdict,
    }


def _curve(n: int, ece: float, points: list[tuple[float, float, int]]) -> dict[str, Any]:
    return {
        "n": n,
        "ece": ece,
        "bins": [{"confidence": c, "accuracy": a, "count": k} for c, a, k in points],
    }


def _payload() -> dict[str, Any]:
    """The smallest report with every block the script reads, all populated."""
    return {
        "provenance": {
            "git_commit": "deadbeef",
            "git_dirty": False,
            "feature_fingerprint": "0000000000000000",
            "generated_at": "2026-01-01T00:00:00Z",
        },
        "corpus": {
            "n_patients": 217,
            "roi": "palpebral",
            "backbone": "mobilenetv3_small_100",
            "feature_dim": 1024,
            "excluded_features": ["erythema_index"],
            "per_site": {
                "India": {"n": 95, "hb_mean": 11.47, "age_mean": 40.1},
                "Italy": {"n": 122, "hb_mean": 13.83, "age_mean": 45.2},
            },
        },
        "fold": {"name": "single-seed42", "n_train": 131, "n_val": 43, "n_test": 43},
        "single_split": [
            {
                "head": "ordinal",
                "n_test": 43,
                "mae": 1.93,
                "rmse": 2.6,
                "r2": -0.04,
                "bias": 0.41,
                "loa_lower": -4.0,
                "loa_upper": 4.8,
                "within_1": 0.349,
                "within_2": 0.651,
                "site_mae_gap": 0.5,
                "worst_site": "India",
            }
        ],
        "site_holdout": [
            {
                "head": "regression",
                "fold": "India->Italy",
                "n_test": 122,
                "mae": 4.2,
                "r2": -4.835,
                "bias": -4.285,
                "mean_true_hb_test_site": 13.83,
            }
        ],
        "site_audit": {
            "ordinal": {
                "available": True,
                "confound": "site is not a biological category here",
                "screening": {
                    "bands": ["anaemia", "moderate_or_worse", "severe"],
                    "pooled": {"anaemia": _band(43, 20, 0.8, 0.61, "ok")},
                    "per_site": {
                        "India": {"anaemia": _band(20, 12, 0.75, 0.5, "ok")},
                        "Italy": {"anaemia": _band(23, 8, 0.88, 0.73, "ok")},
                    },
                },
                "calibration": {
                    "available": True,
                    "n_bins": 10,
                    "pooled": _curve(43, 0.141, [(0.2, 0.15, 5), (0.5, 0.6, 12), (0.8, 0.72, 20)]),
                    "per_site": {
                        "India": _curve(20, 0.083, [(0.2, 0.1, 3), (0.6, 0.66, 8)]),
                        "Italy": _curve(23, 0.194, [(0.3, 0.2, 4), (0.7, 0.55, 10)]),
                    },
                    "max_ece_gap": 0.111,
                },
            },
            "regression": {
                "available": True,
                "confound": "site is not a biological category here",
                "screening": {
                    "bands": ["anaemia"],
                    "pooled": {"anaemia": _band(43, 20, 0.78, 0.6, "ok")},
                    "per_site": {"India": {"anaemia": _band(20, 12, 0.7, 0.5, "ok")}},
                },
                "calibration": {
                    "available": False,
                    "reason": "a regression head emits no probability",
                },
            },
        },
    }


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def test_fmt_turns_nan_into_a_dash_not_a_number() -> None:
    """A missing metric must not print as ``nan``, which reads as a value."""
    assert figures._fmt(float("nan")) == "-"
    assert figures._fmt(None) == "-"
    assert figures._fmt("not a number") == "-"
    assert figures._fmt(1.23456) == "1.235"


def test_signed_keeps_the_sign_of_a_negative_bias() -> None:
    """The sign of the bias is the finding; a dropped minus sign inverts it."""
    assert figures._signed(-4.285) == "-4.285"
    assert figures._signed(0.0) == "+0.000"


# --------------------------------------------------------------------------- #
# The markdown block
# --------------------------------------------------------------------------- #


def test_block_reports_the_frozen_provenance() -> None:
    block = figures.markdown_block(_payload())
    assert "deadbeef" in block
    assert "0000000000000000" in block
    assert "single-seed42" in block


def test_block_lists_every_head_and_the_excluded_feature() -> None:
    """A number without its head, or a feature dropped silently, is not auditable."""
    block = figures.markdown_block(_payload())
    assert "ordinal" in block
    assert "regression" in block
    assert "erythema_index" in block


def test_block_has_one_row_per_site_in_the_screening_table() -> None:
    block = figures.markdown_block(_payload())
    assert "| ordinal | pooled |" in block
    assert "| ordinal | India |" in block
    assert "| ordinal | Italy |" in block


def test_block_reports_a_calibratable_head_and_skips_a_regression_one() -> None:
    """The calibration table must not invent an ECE for a head that has no probability."""
    block = figures.markdown_block(_payload())
    calibration = block.split("Cross-site calibration")[1]
    assert "0.141" in calibration
    assert "India 0.083" in calibration
    assert "Italy 0.194" in calibration
    assert "regression" not in calibration


def test_block_survives_a_report_with_no_optional_blocks() -> None:
    """``hemolux train`` can legitimately omit the sweep-shaped blocks."""
    block = figures.markdown_block({})
    assert "### Frozen run" in block
    assert "-" in block


# --------------------------------------------------------------------------- #
# The figures
# --------------------------------------------------------------------------- #


def test_reliability_figure_is_written_when_a_probability_exists(tmp_path: Path) -> None:
    path = figures.figure_reliability(_payload(), tmp_path)
    assert path is not None and path.exists()
    assert path.stat().st_size > 0


def test_reliability_figure_is_skipped_without_calibration(tmp_path: Path) -> None:
    payload = _payload()
    payload["site_audit"] = {"regression": {"available": True, "calibration": {"available": False}}}
    assert figures.figure_reliability(payload, tmp_path) is None


def test_screening_figure_is_written(tmp_path: Path) -> None:
    path = figures.figure_screening(_payload(), tmp_path)
    assert path is not None and path.exists()
    assert path.stat().st_size > 0


def test_prediction_figures_are_written_from_the_gitignored_csv(tmp_path: Path) -> None:
    import numpy as np

    predictions = {
        "ordinal": {
            "true": np.array([8.0, 10.0, 12.0, 14.0, 16.0]),
            "pred": np.array([8.5, 9.4, 13.0, 13.2, 17.0]),
        }
    }
    written = figures.figure_predictions(predictions, tmp_path)
    assert {p.name for p in written} == {"predicted_vs_true.png", "bland_altman.png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


# --------------------------------------------------------------------------- #
# The sweep block
# --------------------------------------------------------------------------- #


def _sweep() -> dict[str, Any]:
    """A two-candidate sweep: one winner and one that collapsed on validation."""
    return {
        "provenance": {
            "git_commit": "deadbeef",
            "feature_fingerprint": "0000000000000000",
            "generated_at": "2026-01-01T00:00:00Z",
        },
        "metric": "mae",
        "fold": {"name": "sweep-seed42", "n_train": 131, "n_val": 43, "n_test": 43},
        "candidates": [
            {
                "label": "colour / forniceal / raw",
                "kind": "colour",
                "roi": "forniceal",
                "balance": "raw",
                "head": "regression",
                "feature_dim": 12.0,
                "val_mae": 1.99,
                "val_r2": -0.1999,
                "n_unmasked": 6.0,
                "selected": True,
            },
            {
                "label": "colour / forniceal / balanced",
                "kind": "colour",
                "roi": "forniceal",
                "balance": "balanced",
                "head": "regression",
                "feature_dim": 12.0,
                "val_mae": 5.704,
                "val_r2": -8.511,
                "n_unmasked": 6.0,
                "selected": False,
            },
        ],
        "winner": "colour / forniceal / raw",
        "selection_gap": 3.714,
        "test_read": "once, after selection",
        "excluded_features": ["erythema_index"],
        "test": {
            "n": 43.0,
            "mae": 1.9261,
            "rmse": 2.6039,
            "r2": -0.039,
            "evs": -0.0134,
            "pearson_r": 0.5089,
            "bias": 0.4088,
            "loa_lower": -4.6912,
            "loa_upper": 5.5087,
            "within_1": 0.3488,
            "within_2": 0.6512,
        },
    }


def test_sweep_block_keeps_the_losing_candidate_on_the_record() -> None:
    """A sweep that names only its winner is a claim; the table is the selection."""
    block = figures.sweep_block(_sweep())
    assert "colour / forniceal / raw" in block
    assert "colour / forniceal / balanced" in block
    assert "5.704" in block
    assert "| yes |" in block
    assert "| no |" in block


def test_sweep_block_prints_the_single_test_read() -> None:
    block = figures.sweep_block(_sweep())
    test_section = block.split("scored once after selection")[1]
    assert "1.926" in test_section
    assert "34.9%" in test_section
    assert "once, after selection" in block


def test_sweep_block_survives_an_empty_payload() -> None:
    block = figures.sweep_block({})
    assert "### Configuration sweep" in block
    assert "-" in block
