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
