"""Unit tests for the open fairness-first benchmark harness (`hemolux eval`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hemolux.cli import main
from hemolux.metrics.eval_harness import run_evaluation


@pytest.fixture
def dummy_predictions_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    y_true = rng.uniform(9.0, 16.0, size=n)
    y_pred = y_true + rng.normal(0.0, 0.8, size=n)
    ita = np.linspace(-25.0, 55.0, n)
    site = np.array(["India", "Italy"] * (n // 2), dtype=object)
    sex = ["M", "F"] * (n // 2)
    confidence = np.clip(1.0 - np.abs(y_true - y_pred) / 3.0, 0.1, 0.95)

    return pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "ita": ita,
            "site": site,
            "sex": sex,
            "confidence": confidence,
        }
    )


def test_eval_harness_computes_complete_metrics(dummy_predictions_df: pd.DataFrame) -> None:
    report = run_evaluation(dummy_predictions_df)

    assert report.n_samples == 60
    assert 0.0 < report.regression["mae"] < 2.0
    assert 0.0 < report.regression["rmse"] < 2.5
    assert -5.0 < report.regression["bias"] < 5.0
    assert 0.0 <= report.regression["within_1"] <= 1.0

    # Screening metrics
    assert "pooled" in report.screening
    assert "anaemia" in report.screening["pooled"]
    assert "sensitivity" in report.screening["pooled"]["anaemia"]
    assert "specificity" in report.screening["pooled"]["anaemia"]

    # Fairness gate
    assert report.fairness["verdict"] in {"supported", "not_supported", "underpowered"}
    assert np.isfinite(report.fairness["slope"])

    # Cross-site confound
    assert "confounded with" in report.site_audit["confound"]
    assert set(report.site_audit["screening"]["per_site"]) == {"India", "Italy"}

    # Risk-coverage curve
    assert len(report.risk_coverage) >= 4
    # Rejection of poor predictions must monotonically reduce or keep error stable
    assert report.risk_coverage[0]["coverage"] >= report.risk_coverage[-1]["coverage"]


def test_eval_harness_markdown_output(dummy_predictions_df: pd.DataFrame) -> None:
    report = run_evaluation(dummy_predictions_df)
    md = report.to_markdown()

    assert "# Hemolux Fairness & Performance Benchmark Audit" in md
    assert "Bland-Altman" in md
    assert "Confound Disclosure" in md
    assert "Selective Prediction" in md


def test_eval_cli_runs_and_writes_output(
    tmp_path: Path, dummy_predictions_df: pd.DataFrame
) -> None:
    csv_path = tmp_path / "preds.csv"
    dummy_predictions_df.to_csv(csv_path, index=False)

    out_md = tmp_path / "audit_report.md"
    exit_code = main(
        ["eval", "--predictions", str(csv_path), "--format", "markdown", "--out", str(out_md)]
    )

    assert exit_code == 0
    assert out_md.is_file()
    content = out_md.read_text(encoding="utf-8")
    assert "MAE" in content
    assert "Confound" in content


def test_eval_cli_json_format(
    tmp_path: Path, dummy_predictions_df: pd.DataFrame, capsys: pytest.CaptureFixture
) -> None:
    csv_path = tmp_path / "preds.csv"
    dummy_predictions_df.to_csv(csv_path, index=False)

    exit_code = main(["eval", "--predictions", str(csv_path), "--format", "json"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["n_samples"] == 60
    assert "regression" in payload
    assert "fairness" in payload
