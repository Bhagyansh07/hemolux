"""Hemolux — non-invasive haemoglobin screening from smartphone tissue images.

A screening and triage-support research prototype. Not a diagnostic device.
See ``ETHICS.md`` for the intended-use statement and the limits of what
this tool is allowed to claim.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Bumped whenever the model is retrained and the exported weights change.
# Surfaced in the UI footer so a deployed build can be identified exactly.
MODEL_SCHEMA_VERSION = 1

__all__ = ["MODEL_SCHEMA_VERSION", "__version__"]
