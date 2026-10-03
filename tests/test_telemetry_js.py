"""The client telemetry module can only send non-identifying fields.

The privacy claim is that no photograph and no identifier leaves the device.
The server tests pin the schema it will accept; these pin that the client cannot
produce anything else, and that a failed send never throws into the screening
flow.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parent / "telemetry_harness.mjs"

_ALLOWED = {"model_id", "band", "hb_hat", "sigma", "quality", "latency_ms"}


@pytest.fixture(scope="module")
def results() -> dict[str, dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot exercise the client telemetry module")
    completed = subprocess.run([node, str(_HARNESS)], capture_output=True, text=True, check=True)
    return {row["name"]: row for row in json.loads(completed.stdout)}


def test_valid_event_keeps_exactly_the_allowed_fields(results: dict) -> None:
    payload = results["valid"]["payload"]
    assert set(payload) == _ALLOWED
    assert payload["hb_hat"] == 11.4
    assert payload["latency_ms"] == 142


def test_unknown_keys_are_dropped(results: dict) -> None:
    payload = results["drops_unknown"]["payload"]
    assert set(payload) <= _ALLOWED
    assert "evil" not in payload and "image" not in payload and "nested" not in payload


@pytest.mark.parametrize("name", ["bad_band", "bad_model"])
def test_unreportable_events_are_refused_locally(results: dict, name: str) -> None:
    assert results[name]["payload"] is None


def test_an_out_of_range_optional_field_drops_only_that_field(results: dict) -> None:
    # The band is the primary datum and is valid, so the screening is still worth
    # one row; the impossible hb is left out rather than poisoning the insert.
    assert results["out_of_range"]["payload"] == {"model_id": "m-v1", "band": "mild"}


def test_a_dead_endpoint_never_throws(results: dict) -> None:
    outcome = results["swallows_failure"]["result"]
    assert outcome["ok"] is False
    assert outcome["reason"] == "network"


def test_the_request_is_a_post_to_the_documented_path(results: dict) -> None:
    seen = results["request_seen"]["seen"]
    assert seen["url"] == "/api/v1/telemetry"
    assert seen["method"] == "POST"
    assert seen["contentType"] == "application/json"
    assert set(seen["body"]) <= _ALLOWED
