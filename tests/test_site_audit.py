"""Cross-site audit: thresholds, calibration, and the confound that leads it.

The design decision under test is that the exposure confound travels *inside* the
report data rather than in a paragraph around it. A consumer of the JSON cannot
read a site contrast without the sentence explaining that site is not a
biological category here, which is the only way the caveat survives being copied
out of context.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemolux.metrics.site_audit import SITE_CONFOUND, audit_sites, calibration_gate


def _cohort(n: int = 40) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(7)
    truth = rng.uniform(9.0, 16.0, size=n)
    pred = truth + rng.normal(0.0, 0.5, size=n)
    sex = ["M", "F"] * (n // 2)
    site = np.array(["India", "Italy"] * (n // 2), dtype=object)
    return truth, pred, sex, site


def test_the_confound_is_inside_the_data_not_left_to_prose() -> None:
    payload = audit_sites(*_cohort()).to_dict()

    assert payload["confound"] == SITE_CONFOUND
    assert "confounded with" in payload["confound"]
    assert "direction" in payload["confound"], "the direction of the bias is the point"


def test_screening_is_pooled_and_per_site() -> None:
    payload = audit_sites(*_cohort()).to_dict()

    assert set(payload["screening"]["per_site"]) == {"India", "Italy"}
    assert "anaemia" in payload["screening"]["pooled"]


def test_without_probabilities_calibration_says_so() -> None:
    payload = audit_sites(*_cohort()).to_dict()

    assert payload["calibration"]["available"] is False
    assert "no probability" in payload["calibration"]["reason"]


def test_with_probabilities_each_site_gets_a_curve_and_a_gap() -> None:
    truth, pred, sex, site = _cohort()
    confidence = np.clip(np.abs(truth - pred), 0.05, 0.95)
    correct = np.abs(truth - pred) < 1.0

    payload = audit_sites(truth, pred, sex, site, confidence=confidence, correct=correct).to_dict()

    calibration = payload["calibration"]
    assert calibration["available"] is True
    assert set(calibration["per_site"]) == {"India", "Italy"}
    assert calibration["pooled"]["n"] == 40
    assert np.isfinite(calibration["max_ece_gap"])


def test_a_single_site_has_no_gap_to_report() -> None:
    """An unmeasured gap is NaN, not a zero gap."""
    truth, pred, sex, _ = _cohort()
    one_site = np.array(["India"] * truth.size, dtype=object)
    confidence = np.full(truth.size, 0.5)
    correct = np.ones(truth.size, dtype=bool)

    payload = audit_sites(
        truth, pred, sex, one_site, confidence=confidence, correct=correct
    ).to_dict()

    assert np.isnan(payload["calibration"]["max_ece_gap"])


def test_calibration_gate_reuses_underpowered_verbatim() -> None:
    conf = np.linspace(0.0, 1.0, 10)
    correct = np.linspace(0.0, 1.0, 10)
    ita = np.linspace(-30.0, 60.0, 10)

    got = calibration_gate(conf, correct, ita)

    assert got["verdict"] == "underpowered"
    assert got["n"] == 10


def test_calibration_gate_finds_a_monotone_pigment_trend() -> None:
    """Overconfidence that rises with ITA is the hypothesis, and it is found."""
    n = 60
    ita = np.linspace(-30.0, 60.0, n)
    correct = np.where(ita < 20.0, 1.0, 0.0)
    confidence = np.clip(0.5 + 0.01 * ita, 0.0, 1.0)

    got = calibration_gate(confidence, correct, ita)

    assert got["verdict"] == "supported"
    assert got["ci_low"] * got["ci_high"] > 0.0, "the interval must clear zero"


def test_calibration_gate_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        calibration_gate(np.ones(5), np.ones(4), np.ones(5))
