"""Evidence-panel contract.

The page is allowed to show figures only from the staged ``results.json``, so
the reducer that reads it has to be exact about which fields it trusts and in
which order it presents the heads. This runs the reducer in Node and asserts
the shape, not a rendered string.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parent / "evidence_harness.mjs"


@pytest.fixture(scope="module")
def payload() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot exercise the evidence reducer")
    completed = subprocess.run([node, str(_HARNESS)], capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def test_corpus_fields_survive_the_reduce(payload: dict) -> None:
    out = payload["out"]
    assert out["nPatients"] == 217
    assert out["sites"] == ["India", "Italy"]
    assert out["featureDim"] == 13
    assert out["roi"] == "forniceal_palpebral"
    assert out["backbone"] == "mobilenetv3_small_100"
    assert out["excluded"] == ["erythema_index"]


def test_provenance_is_read_verbatim(payload: dict) -> None:
    out = payload["out"]
    assert out["commit"] == "abc123"
    assert out["fingerprint"] == "859399ade17b39e8"
    assert out["generated"] == "2026-10-03T12:00:00Z"
    assert out["dirty"] is False


def test_heads_lead_with_the_screening_estimate(payload: dict) -> None:
    # ordinal and regression are the haemoglobin heads; binary is last.
    assert payload["headOrder"] == ["ordinal", "regression", "binary"]


def test_nothing_to_show_stays_nothing(payload: dict) -> None:
    assert payload["empty"] is None
    assert payload["noRows"] is None


def test_sweep_winner_leads_regardless_of_input_order(payload: dict) -> None:
    # The fixture lists the worst candidate first on purpose: the reducer, not
    # the report's file order, decides what the panel shows first.
    out = payload["sweepOut"]
    assert out["winner"] == "deep / palpebral"
    assert out["metric"] == "mae"
    assert out["nCandidates"] == 3
    assert payload["sweepOrder"] == [
        "deep / palpebral",
        "colour / forniceal / raw",
        "colour / forniceal / balanced",
    ]
    assert out["rows"][0]["selected"] is True


def test_sweep_exposes_the_single_test_read(payload: dict) -> None:
    out = payload["sweepOut"]
    assert out["fold"] == {"train": 131, "val": 43, "test": 43}
    assert out["test"]["n"] == 43
    assert out["test"]["mae"] == pytest.approx(1.62)
    assert out["test"]["r2"] == pytest.approx(0.24)
    assert out["nearTie"] is False


def test_sweep_flags_a_near_tie(payload: dict) -> None:
    assert payload["sweepNearTie"] is True


def test_sweep_keeps_the_whole_frame_patients_named(payload: dict) -> None:
    out = payload["sweepOut"]
    assert [entry["label"] for entry in out["unmasked"]] == [
        "colour / forniceal / balanced",
        "colour / forniceal / raw",
    ]
    assert all(entry["count"] == 6 for entry in out["unmasked"])


def test_sweep_nothing_to_show_stays_nothing(payload: dict) -> None:
    assert payload["sweepEmpty"] is None
    assert payload["sweepNoRows"] is None


def test_site_audit_keeps_pooled_and_per_site_calibration(payload: dict) -> None:
    ordinal = payload["siteAudit"]["ordinal"]
    assert ordinal["calibratable"] is True
    assert ordinal["pooledEce"] == pytest.approx(0.141)
    assert ordinal["gap"] == pytest.approx(0.111)
    per_site = {entry["site"]: entry for entry in ordinal["perSite"]}
    assert sorted(per_site) == ["India", "Italy"]
    assert per_site["India"]["ece"] == pytest.approx(0.083)
    assert per_site["India"]["n"] == 20
    assert per_site["Italy"]["ece"] == pytest.approx(0.194)
    assert per_site["Italy"]["n"] == 23


def test_site_audit_marks_a_head_with_no_probability(payload: dict) -> None:
    binary = payload["siteAudit"]["binary"]
    assert binary["calibratable"] is False
    assert binary["pooledEce"] is None
    assert binary["gap"] is None
    assert binary["perSite"] == []


def test_the_confound_travels_with_the_audit(payload: dict) -> None:
    assert payload["confound"].startswith("Site is not a biological category")
