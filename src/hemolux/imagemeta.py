"""Keep known-bad image metadata quiet without hiding a real decode failure.

Why this exists
---------------
Every PNG mask in the public corpus carries a malformed ``iCCP`` chunk, so libpng
prints ``libpng warning: iCCP: CRC error`` once per file. Across 639 masks that
is several hundred lines that bury the progress output, and any tool that treats
stderr output as failure then reports a clean run as a broken one.

A mask is a region label: it has no colour profile to manage, and libpng already
falls back to sRGB when the chunk fails to verify, which is exactly what the
preprocessing assumes. So the warning is noise.

Why not mute it broadly
-----------------------
``opencv.setLogLevel(0)`` is the obvious move and it is wrong. It also hides the
message that matters -- that a file could not be decoded at all -- and a project
that measures colour cannot afford a decoder warning that means nothing sitting
next to one that means the pixels are wrong.

Two things here, because the warning arrives by two different routes, and an
earlier version of this module handled only the second while the first produced
every warning the project actually sees:

``silence_corrupt_iccp``
    A logging filter on the ``PIL`` loggers. Correct, and sufficient if and only
    if the bytes are decoded by Pillow.

``quiet_image_decoder``
    A file-descriptor capture for the route this project actually uses. OpenCV
    hands PNGs to libpng, and libpng writes its diagnostics from C to file
    descriptor 2 without consulting Python's logging machinery at all. No
    logging filter can reach it, so the descriptor is captured for the duration
    of the block, and everything that is *not* the one known-bad line is
    written back out afterwards. Suppression is scoped to the message, not to the
    channel, which is the only thing that makes it safe.

Both are disabled by ``HEMOLUX_STRICT_IMAGE_LOG=1``, which is what CI sets: on a
fresh runner with no cached output, a decode problem should be visible.

Thread safety
-------------
``quiet_image_decoder`` redirects a process-wide descriptor, so another thread
writing to stderr inside the block is captured too and relayed afterwards rather
than lost. Output is delayed, not dropped. It is not meant to wrap a long-running
server; it is meant to wrap a decode loop in a script.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from collections.abc import Iterator

#: The only message suppressed, and the reason it is safe to suppress. An iCCP CRC
#: error means the embedded colour profile is corrupt, not that the pixel data
#: is; libpng then falls back to sRGB.
_ICC_PATTERN = "libpng warning: iCCP: CRC error"

#: The escape hatch, checked once per call so a test can flip it without touching
#: the environment of the whole process.
_STRICT_ENV = "HEMOLUX_STRICT_IMAGE_LOG"


class _IccpFilter(logging.Filter):
    """Drops the one known-bad message, and nothing else."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _ICC_PATTERN not in record.getMessage()


_PIL_LOGGERS = ("PIL", "PIL.PngImagePlugin")


def strict_image_logging() -> bool:
    """Whether decoder diagnostics must be left alone.

    Set ``HEMOLUX_STRICT_IMAGE_LOG=1`` to switch the filters off entirely.
    """
    return os.environ.get(_STRICT_ENV, "") == "1"


def silence_corrupt_iccp(force: bool | None = None) -> bool:
    """Filter the iCCP CRC warning on the Pillow loggers.

    Returns whether the filter is in place afterwards. Idempotent, and idempotent
    by inspecting the loggers rather than by remembering: a module-level flag
    claiming the filter is installed goes stale the moment anything else clears or
    reconfigures those loggers, and then this function reports success while
    filtering nothing.

    Parameters
    ----------
    force
        ``True`` installs the filter even when strict logging is on, ``False``
        never installs it, and ``None`` consults :func:`strict_image_logging`.
    """
    if strict_image_logging() if force is None else not force:
        return False

    active = False
    for name in _PIL_LOGGERS:
        logger = logging.getLogger(name)
        if any(isinstance(f, _IccpFilter) for f in logger.filters):
            active = True
            continue
        logger.addFilter(_IccpFilter())
        active = True
    return active


@contextlib.contextmanager
def quiet_image_decoder(force: bool | None = None) -> Iterator[None]:
    """Suppress the known-bad iCCP line for the duration of the block.

    Everything else written to file descriptor 2 inside the block is written back
    out afterwards, in order. So a genuine decoder failure still reaches the
    terminal, which is the whole reason this is a filter on the message rather
    than a mute on the descriptor.

    Parameters
    ----------
    force
        ``False`` forces suppression on even under ``HEMOLUX_STRICT_IMAGE_LOG=1``;
        ``True`` forces it off; ``None`` consults :func:`strict_image_logging`.
    """
    strict = strict_image_logging() if force is None else bool(force)
    if strict:
        yield
        return

    # Flush first: anything already buffered belongs before the captured text.
    sys.stderr.flush()
    saved = os.dup(2)
    text = ""
    try:
        with tempfile.TemporaryFile() as capture:
            os.dup2(capture.fileno(), 2)
            try:
                yield
            finally:
                sys.stderr.flush()
                os.dup2(saved, 2)
                capture.seek(0)
                text = capture.read().decode("utf-8", "replace")
    finally:
        os.close(saved)
        # Relay unconditionally, including when the body raised: the diagnostics
        # in here explain the failure as often as they are incidental to it.
        for line in text.splitlines(keepends=True):
            if _ICC_PATTERN not in line:
                sys.stderr.write(line)
        sys.stderr.flush()
