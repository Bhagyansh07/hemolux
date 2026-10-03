"""What produced a results file, recorded inside it.

Why this exists
---------------
``artifacts/reports/`` is tracked: those files are the evidence ``EVALS.md`` points at,
so they are committed next to the number they support. A file that records a score but
not the conditions of the run is not evidence, it is a number with an author. Three
questions a reader needs answered and cannot get from a bare table:

* **Which corpus was this?** The generated fixture in ``scripts/`` reproduces the real
  layout deliberately -- same site directories, same workbook names, same mask naming --
  because a fixture that standardised any of that would stop covering the branches it
  exists to cover. An eight-patient run and a 217-patient run therefore produce results
  files with the same shape and wildly different numbers, and nothing inside either one
  says which is which. A figure taken from the wrong one is not a rounding error, it is a
  claim about people who do not exist.
* **Which code was this?** The feature cache got an answer to this first, because a stale
  cache returning an old number was the failure that kept happening. A report has the same
  problem one level up: ``results.json`` from before a metric fix is still a valid file,
  still loads, and still looks like a result.
* **Was the tree clean?** A report from a dirty checkout is reproducible from the recorded
  commit only by accident. Recording the commit without recording the dirt would make the
  file *look* reproducible, which is worse than admitting it is not.

What it deliberately does not record
------------------------------------
Machine, hostname, user, or elapsed wall time. None of those change a number, and putting
them in a committed file adds a field that varies per run and invites a diff on every
re-run with nothing to read in it. The corpus, the code, and the clock are the three that
matter; the third is there so a figure can be tied to when it was taken.

The dependency versions are not repeated here: they are already hashed into
:func:`hemolux.fingerprint.digest`, alongside the source. One field, one meaning.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from hemolux import fingerprint
from hemolux.validation import corpus_kind

#: How long a ``git`` call may take before it is treated as unavailable. Every call here
#: is a metadata read against a local repository and completes in milliseconds; anything
#: near this bound means something is wrong (a network filesystem, a locked index) and a
#: report is not worth blocking on.
_GIT_TIMEOUT_S = 10.0


def snapshot(data_root: str | Path | None = None) -> dict[str, object]:
    """The provenance block written into ``results.json`` and ``sweep.json``.

    ``data_root`` is optional so a report can be built from a feature set with no corpus
    on disk -- the tests do exactly that -- in which case the corpus fields are ``None``
    rather than a guess.
    """
    root = None if data_root is None else Path(data_root)
    return {
        "data_root": None if root is None else str(root),
        # "documented", "synthetic" or "unknown". The three-way answer matters because
        # "not the documented corpus" conflates the fixture with a directory that was
        # pointed at by mistake, and only one of those is a small number rather than a
        # bug in the invocation.
        "corpus": None if root is None else corpus_kind(root),
        # The same digest the feature cache is keyed on. Recorded here as well so a report
        # and the caches that fed it can be checked against each other after the fact.
        "feature_fingerprint": fingerprint.digest(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def git_commit() -> str | None:
    """The commit the run executed against, or ``None`` if that cannot be determined.

    ``None`` is not an error. A wheel installed from a release is not a checkout and has
    no commit to report; saying so is more useful than omitting the field, because a
    reader can tell the difference between "not a checkout" and "somebody forgot".
    """
    out = _run_git("rev-parse", "HEAD")
    return None if out is None or not out else out


def git_dirty() -> bool | None:
    """Whether tracked files differed from the commit when the run started.

    ``--untracked-files=no`` on purpose. The artefacts directory accumulates caches and
    reports that git may or may not be tracking depending on which of them are committed,
    and a report that called itself unreproducible every time it wrote its own output
    would be noise. The question is whether the *code and committed content* matched the
    commit, and only tracked modifications answer that.
    """
    out = _run_git("status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out)


def _run_git(*args: str) -> str | None:
    """Run one ``git`` read, returning stripped stdout or ``None`` on any failure.

    Every failure mode returns ``None``: git absent, not a repository, a locked index, a
    timeout. There is nothing a report can do about any of them, and raising would make
    writing a results file depend on having git installed -- which would break the very
    environment a released wheel runs in.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
