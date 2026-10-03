"""Bilingual dictionary contract.

Every user-visible string lives in both dictionaries and nowhere else, so the
failure this guards against is a key added to one language and not the other, or
a ``t("...")`` / ``data-i18n="..."`` reference to a key that was never defined.

Both failures are silent at runtime -- the first falls back to English, the
second renders the raw key -- which is exactly why they need a test rather than
a careful read. The shell files are scanned for literal references rather than
parsed, so a dynamically built key would simply not be seen; that is the
conservative direction, because the test can miss a reference but cannot invent
one.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_WEB = Path(__file__).resolve().parent.parent / "app" / "web"
_HARNESS = Path(__file__).resolve().parent / "i18n_harness.mjs"

# ``\b`` keeps the match off ``format(`` / ``concat(`` and the like; only a
# free-standing ``t("key")`` is a translation call.
_T_CALL = re.compile(r'\bt\(\s*"([^"]+)"')
_ATTR = re.compile(r'data-i18n(?:-aria)?="([^"]+)"')

# Vendored runtime and dependencies are not part of the shell's vocabulary.
_SKIP = ("vendor", "node_modules", "dist")


@pytest.fixture(scope="module")
def keys() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot inspect the dictionaries")
    completed = subprocess.run([node, str(_HARNESS)], capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def test_both_dictionaries_define_the_same_keys(keys: dict) -> None:
    assert keys["en"] == keys["hi"]


def _referenced_keys() -> set[str]:
    referenced: set[str] = set()
    for path in sorted(_WEB.rglob("*.js")):
        if any(part in _SKIP for part in path.parts):
            continue
        referenced.update(_T_CALL.findall(path.read_text(encoding="utf-8")))
    for path in sorted(_WEB.rglob("*.html")):
        if any(part in _SKIP for part in path.parts):
            continue
        referenced.update(_ATTR.findall(path.read_text(encoding="utf-8")))
    return referenced


def test_every_referenced_key_exists(keys: dict) -> None:
    defined = set(keys["en"])
    missing = sorted(_referenced_keys() - defined)
    assert not missing, f"referenced but not defined in either dictionary: {missing}"
