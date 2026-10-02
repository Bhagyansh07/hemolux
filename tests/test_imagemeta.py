"""Tests for the image-metadata warning filter.

This module redirects a process-wide file descriptor, so its failure modes are
silent and asymmetric. The one that matters is suppressing more than it should: a
filter that hides a decoder failure would let a run report success over corrupt
pixels, and a project that measures colour cannot afford that. So most of what
follows asserts that noise is *relayed*, not that noise is dropped.

The design constraint that produced the module is also the reason for two of these
cases. The warnings arrive by two routes. Pillow routes them through Python's
``logging``; OpenCV hands PNGs to libpng, which writes from C to file descriptor 2
and never consults the logging machinery. An earlier version of this module
installed a logging filter and nothing else, which meant it handled the route
this project does not use and left every warning the project actually emits in
place -- while reporting that it had installed a filter. Two cases below pin that
history down so it cannot recur.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys

import pytest

from hemolux.imagemeta import (
    _ICC_PATTERN,
    quiet_image_decoder,
    silence_corrupt_iccp,
    strict_image_logging,
)


def _emit_fd2(text: str) -> None:
    """Write straight to the descriptor, as libpng does.

    ``sys.stderr.write`` would be intercepted by pytest at the stream level and
    would not exercise the descriptor swap this module performs.
    """
    sys.stderr.flush()
    os.write(2, text.encode("utf-8"))


@pytest.fixture(autouse=True)
def _not_strict_by_default(monkeypatch: pytest.MonkeyPatch):
    """Clear the strict escape hatch for this file.

    CI sets ``HEMOLUX_STRICT_IMAGE_LOG=1`` on purpose, and that setting reaches the
    tests. Left alone it makes every suppression case below a no-op, so the file
    fails on CI and passes locally depending on the environment rather than the
    code. The cases that assert suppression also pass ``force=False`` explicitly, so
    the intent is visible at the call site as well as guaranteed here.
    """
    monkeypatch.delenv("HEMOLUX_STRICT_IMAGE_LOG", raising=False)


@pytest.fixture(autouse=True)
def _strip_pillow_filters():
    """Leave the Pillow loggers as found.

    The filters are process-wide, and ``logging`` keeps them until something clears
    them. Without this, whichever test installs first decides what the rest of the
    file observes.
    """
    yield
    for name in ("PIL", "PIL.PngImagePlugin"):
        logger = logging.getLogger(name)
        for filt in list(logger.filters):
            logger.removeFilter(filt)


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord("PIL", logging.WARNING, __file__, 1, message, None, None)


def _pillow_filter() -> logging.Filter:
    """The most recently installed filter on the Pillow PNG logger."""
    return logging.getLogger("PIL.PngImagePlugin").filters[-1]


# --------------------------------------------------------------------------- #
# The known-bad line, and only that line
# --------------------------------------------------------------------------- #


def test_the_known_bad_line_is_dropped(capfd: pytest.CaptureFixture[str]) -> None:
    with quiet_image_decoder(force=False):
        _emit_fd2("libpng warning: iCCP: CRC error\n")
    assert _ICC_PATTERN not in capfd.readouterr().err


def test_a_real_decoder_failure_is_relayed(capfd: pytest.CaptureFixture[str]) -> None:
    """The property the module exists for: suppression is scoped to the message.

    Asserted alongside ``test_the_known_bad_line_is_dropped`` on purpose. That test
    also passes if nothing was captured at all, so it needs this one to say which of
    the two explanations is the true one.
    """
    with quiet_image_decoder(force=False):
        _emit_fd2("libpng warning: iCCP: CRC error\n")
        _emit_fd2("could not open image: permission denied\n")
    err = capfd.readouterr().err
    assert _ICC_PATTERN not in err
    assert "could not open image: permission denied" in err


def test_interleaved_noise_and_signal_both_survive(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The realistic case: 96 bad lines and one that matters, in a single pass."""
    with quiet_image_decoder(force=False):
        for _ in range(96):
            _emit_fd2("libpng warning: iCCP: CRC error\n")
        _emit_fd2("libpng error: not a PNG file\n")
    err = capfd.readouterr().err
    assert _ICC_PATTERN not in err
    assert "libpng error: not a PNG file" in err


def test_relayed_lines_keep_their_order(capfd: pytest.CaptureFixture[str]) -> None:
    """A run whose diagnostics come out reordered is harder to read than one with none."""
    with quiet_image_decoder(force=False):
        for n in range(3):
            _emit_fd2(f"decoder note {n}\n")
    err = capfd.readouterr().err
    positions = [err.index(f"decoder note {n}") for n in range(3)]
    assert positions == sorted(positions)


def test_output_before_the_block_is_not_swallowed(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Without the flush on entry, earlier output is written into the capture and
    then relayed *after* the block's, reordering the run."""
    _emit_fd2("before the block\n")
    with quiet_image_decoder(force=False):
        _emit_fd2("libpng warning: iCCP: CRC error\n")
    err = capfd.readouterr().err
    assert "before the block" in err
    assert err.index("before the block") < len(err)
    assert _ICC_PATTERN not in err


def test_output_after_the_block_is_untouched(capfd: pytest.CaptureFixture[str]) -> None:
    """The descriptor has to be restored, not merely released."""
    with quiet_image_decoder(force=False):
        _emit_fd2("libpng warning: iCCP: CRC error\n")
    _emit_fd2("after the block\n")
    err = capfd.readouterr().err
    assert "after the block" in err
    assert _ICC_PATTERN not in err


def test_an_exception_inside_the_block_still_restores_the_descriptor(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A decode loop that raises must not leave stderr redirected for the rest of the
    process, which would swallow every later diagnostic in the run."""
    with pytest.raises(ValueError, match="decode failed"), quiet_image_decoder():
        _emit_fd2("libpng warning: iCCP: CRC error\n")
        raise ValueError("decode failed")
    _emit_fd2("still here\n")
    assert "still here" in capfd.readouterr().err


def test_undecodable_bytes_do_not_break_the_relay(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """libpng is not guaranteed to emit UTF-8, and cleanup must not raise."""
    with quiet_image_decoder(force=False):
        os.write(2, b"libpng warning: iCCP: CRC error\n\xff\xfe binary \x00 junk\n")
    assert _ICC_PATTERN not in capfd.readouterr().err


def test_a_stream_capture_would_not_have_worked(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Why the obvious implementation is the wrong one, asserted so it stays wrong.

    ``contextlib.redirect_stderr`` rebinds ``sys.stderr`` in Python. libpng holds its
    own handle on the descriptor from C and writes to it directly, so the iCCP lines
    sail straight past the redirect and land on the terminal regardless.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        os.write(2, b"libpng warning: iCCP: CRC error\n")
    assert buffer.getvalue() == "", "the stream redirect caught a descriptor write"
    assert _ICC_PATTERN in capfd.readouterr().err


# --------------------------------------------------------------------------- #
# The strict escape hatch
# --------------------------------------------------------------------------- #


def test_strict_logging_lets_everything_through(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI sets this: on a fresh runner a decode problem should be visible."""
    monkeypatch.setenv("HEMOLUX_STRICT_IMAGE_LOG", "1")
    assert strict_image_logging() is True
    # force=None on purpose: this case is about the environment being honoured, so
    # pinning the flag would make it assert the opposite of what it is for.
    with quiet_image_decoder():
        _emit_fd2("libpng warning: iCCP: CRC error\n")
    assert _ICC_PATTERN in capfd.readouterr().err


def test_force_overrides_the_environment(capfd: pytest.CaptureFixture[str]) -> None:
    """A test that wants the filter must be able to have it whatever the env says."""
    with quiet_image_decoder(force=False):
        _emit_fd2("libpng warning: iCCP: CRC error\n")
    assert _ICC_PATTERN not in capfd.readouterr().err


def test_force_strict_overrides_a_clear_environment(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with quiet_image_decoder(force=True):
        _emit_fd2("libpng warning: iCCP: CRC error\n")
    assert _ICC_PATTERN in capfd.readouterr().err


# --------------------------------------------------------------------------- #
# The Pillow route
# --------------------------------------------------------------------------- #


def test_the_logging_filter_drops_the_line_from_a_pillow_record() -> None:
    """The route this module can actually reach through ``logging``."""
    assert silence_corrupt_iccp(force=True) is True
    filt = _pillow_filter()
    assert filt.filter(_record(_ICC_PATTERN)) is False
    assert filt.filter(_record("image file is truncated")) is True


def test_the_logging_filter_cannot_see_a_libpng_write(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The bug this module was rewritten for, pinned as a test.

    A logging filter installs cleanly, reports success, and does nothing whatsoever
    to a message libpng writes to file descriptor 2 from C. An implementation that
    relies on ``logging`` alone is therefore suppressing nothing while appearing to
    work, which is why ``quiet_image_decoder`` exists and why the loaders use it.
    """
    assert silence_corrupt_iccp(force=True) is True
    _emit_fd2("libpng warning: iCCP: CRC error\n")
    assert _ICC_PATTERN in capfd.readouterr().err


def test_the_pillow_filter_is_idempotent() -> None:
    """Calling it per decode would otherwise stack a filter per image."""
    logger = logging.getLogger("PIL.PngImagePlugin")
    assert silence_corrupt_iccp(force=True) is True
    assert silence_corrupt_iccp(force=True) is True
    assert len(logger.filters) == 1


def test_the_filter_is_reinstalled_after_the_logger_is_cleared() -> None:
    """Why idempotency is decided by inspecting the loggers, not by a flag.

    An earlier version kept a module-level ``_installed`` boolean. Anything that
    cleared the loggers -- a test fixture, a library reconfiguration, a second
    interpreter state -- left that boolean claiming success while nothing was
    filtered, and the function's return value said otherwise.
    """
    logger = logging.getLogger("PIL.PngImagePlugin")
    assert silence_corrupt_iccp(force=True) is True
    logger.filters.clear()
    assert silence_corrupt_iccp(force=True) is True
    assert len(logger.filters) == 1


def test_strict_logging_disables_the_pillow_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEMOLUX_STRICT_IMAGE_LOG", "1")
    logger = logging.getLogger("PIL")
    before = len(logger.filters)
    assert silence_corrupt_iccp() is False
    assert len(logger.filters) == before
