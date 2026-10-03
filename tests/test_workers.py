"""Contract tests for the two edge endpoints.

The workers run on Cloudflare, not in the test process, so this drives the
shipped handler modules through a mock D1 (see ``workers_harness.mjs``) and
asserts the status, the standard error shape, and exactly what reached the
database. The point is the insert column order and the validation branches: a
mistake there would be invisible in production until a row was wrong.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parent / "workers_harness.mjs"


@pytest.fixture(scope="module")
def results() -> dict[str, dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot exercise the edge functions")
    completed = subprocess.run([node, str(_HARNESS)], capture_output=True, text=True, check=True)
    rows = json.loads(completed.stdout)
    return {row["name"]: row for row in rows}


def _code(row: dict) -> str:
    assert row["body"] is not None, f"expected a JSON error for {row['status']}"
    return row["body"]["error"]["code"]


def test_valid_telemetry_is_recorded_and_unknown_keys_are_dropped(results: dict) -> None:
    row = results["valid_telemetry"]
    assert row["status"] == 204
    assert row["body"] is None
    assert row["inserts"] == 1
    # created_at, model_id, band, hb_hat, sigma, quality, latency_ms — and no room
    # for the extra `evil` / `nested` keys the client sent.
    assert len(row["args"]) == 7
    assert row["args"][1] == "efficnet-b0-v4-a1b2c3d4"
    assert row["args"][2] == "within_range"
    assert row["args"][3] == 11.4
    assert row["args"][6] == 142


@pytest.mark.parametrize(
    ("name", "status", "code"),
    [
        ("wrong_content_type", 415, "UNSUPPORTED_MEDIA_TYPE"),
        ("too_large", 413, "PAYLOAD_TOO_LARGE"),
    ],
)
def test_transport_rules(results: dict, name: str, status: int, code: str) -> None:
    row = results[name]
    assert row["status"] == status
    assert _code(row) == code


def test_wrong_method_is_refused_with_allow(results: dict) -> None:
    row = results["method_not_allowed"]
    assert row["status"] == 405
    assert row["allow"] == "POST"
    assert _code(row) == "METHOD_NOT_ALLOWED"


@pytest.mark.parametrize(
    ("name", "field"),
    [("invalid_band", "band"), ("invalid_model_id", "model_id"), ("bad_hb", "hb_hat")],
)
def test_invalid_body_names_the_field(results: dict, name: str, field: str) -> None:
    row = results[name]
    assert row["status"] == 400
    assert _code(row) == "INVALID_BODY"
    assert row["body"]["error"]["details"]["field"] == field


def test_daily_cap_is_a_429_with_a_retry_hint(results: dict) -> None:
    row = results["daily_cap"]
    assert row["status"] == 429
    assert row["retryAfter"] == "3600"
    assert _code(row) == "RATE_LIMITED"


def test_feedback_is_accepted_and_validated(results: dict) -> None:
    ok = results["feedback_valid"]
    assert ok["status"] == 202
    assert ok["body"]["data"]["recorded"] is True
    assert ok["inserts"] == 1

    bad = results["feedback_invalid"]
    assert bad["status"] == 400
    assert bad["body"]["error"]["details"]["field"] == "hb"


def test_feedback_cap_and_method(results: dict) -> None:
    assert results["feedback_cap"]["status"] == 429
    assert results["feedback_method"]["status"] == 405


def test_per_ip_rate_limit_admits_then_refuses(results: dict) -> None:
    row = results["rate_limit"]
    assert row["allowed"] == 30
    assert row["refused"] == 1
    assert row["status"] == 429
