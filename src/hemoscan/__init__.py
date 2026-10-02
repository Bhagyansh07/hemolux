"""HemoScan — non-invasive haemoglobin screening from smartphone tissue images.

A screening and triage-support research prototype. Not a diagnostic device.
See ``brain/01_PRD.md`` §2 for the intended-use statement.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Bumped whenever the model is retrained and the exported weights change.
# Surfaced in the UI footer so a deployed build can be identified exactly.
MODEL_SCHEMA_VERSION = 1

__all__ = ["__version__", "MODEL_SCHEMA_VERSION"]
