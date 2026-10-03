"""The provenance block: a results file has to say what produced it.

``artifacts/reports/`` is tracked, because those files are the evidence ``EVALS.md``
points at. That makes them quotable, and a quotable number with no conditions attached is
a liability. Two of them are dangerous enough to be worth naming:

* the generated fixture in ``scripts/`` reproduces the real corpus layout on purpose --
  same site directories, same workbook names, same mask naming -- so an eight-patient run
  and a 217-patient run produce results files of the same shape. A figure lifted from the
  wrong one is a claim about people who do not exist.
* a ``results.json`` written before a metric fix is still a valid file, still loads, and
  still looks like a result.

The tests below pin the answers. The corpus classification is tested through its three
outcomes rather than through a boolean, because the third one -- a directory that is
neither the download nor the fixture -- is the difference between a small real number and
a command pointed at the wrong path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hemolux import fingerprint, provenance
from hemolux.config import SYNTHETIC_MARKER
from hemolux.validation import corpus_kind, looks_like_the_documented_corpus

#: The fields the block promises. Asserted as a set rather than checked one by one, so a
#: field added without a reader, or removed with one still expecting it, fails here.
FIELDS = {
    "data_root",
    "corpus",
    "feature_fingerprint",
    "git_commit",
    "git_dirty",
    "generated_at",
}


# --------------------------------------------------------------------------- #
# Which corpus
# --------------------------------------------------------------------------- #


def test_an_empty_directory_is_not_a_corpus(tmp_path: Path) -> None:
    """No site directories, no definition of what this could be."""
    assert corpus_kind(tmp_path) == "unknown"


def test_both_site_directories_are_the_documented_shape(tmp_path: Path) -> None:
    """A download unpacked somewhere custom still counts.

    The shape check is deliberately permissive: a user who moved the corpus out of
    ``data/raw/`` should still get the checks that apply to them rather than a wall of
    "not the documented corpus".
    """
    (tmp_path / "India").mkdir()
    (tmp_path / "Italy").mkdir()

    assert corpus_kind(tmp_path) == "documented"


def test_only_one_site_directory_is_unknown(tmp_path: Path) -> None:
    """Half a corpus is a wrong path, not a small corpus."""
    (tmp_path / "India").mkdir()

    assert corpus_kind(tmp_path) == "unknown"


def test_the_synthetic_marker_beats_both_site_directories(tmp_path: Path) -> None:
    """The regression this whole three-way answer exists for.

    The fixture reproduces the layout exactly, so ``India/`` and ``Italy/`` are present
    and the shape check matches it. If the marker did not win, the validator would run
    the 217-patient contract against an eight-patient fixture -- which it used to, and
    which is how a reader is trained to ignore red lines.
    """
    (tmp_path / "India").mkdir()
    (tmp_path / "Italy").mkdir()
    (tmp_path / f"{SYNTHETIC_MARKER}workbook.xlsx").write_text("", encoding="utf-8")

    assert corpus_kind(tmp_path) == "synthetic"


def test_a_directory_that_cannot_be_listed_is_unknown_not_an_exception(
    tmp_path: Path,
) -> None:
    """Classification runs on report-writing paths, which must not fail late.

    A typo in ``--data-root`` should produce a report that says it could not tell, not a
    traceback from inside a writer.
    """
    assert corpus_kind(tmp_path / "does-not-exist") == "unknown"


def test_the_boolean_helper_asks_a_narrower_question(tmp_path: Path) -> None:
    """``looks_like_the_documented_corpus`` is the validator's question, kept separate.

    A synthetic corpus is ``not`` documented, so the validator skips the contract; a
    directory that is neither is also not documented, and ``--contract`` can force it.
    Both are ``False`` here, which is the point: the boolean collapses a distinction the
    reports need and the validator does not.
    """
    (tmp_path / "India").mkdir()
    (tmp_path / "Italy").mkdir()
    assert looks_like_the_documented_corpus(tmp_path) is True

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "India").mkdir()
    (fixture / f"{SYNTHETIC_MARKER}eyes.xlsx").write_text("", encoding="utf-8")
    assert looks_like_the_documented_corpus(fixture) is False
    assert looks_like_the_documented_corpus(tmp_path / "neither") is False


# --------------------------------------------------------------------------- #
# The snapshot
# --------------------------------------------------------------------------- #


def test_the_snapshot_has_exactly_the_promised_fields(tmp_path: Path) -> None:
    assert set(provenance.snapshot(tmp_path)) == FIELDS


def test_no_root_means_no_corpus_claim() -> None:
    """A caller without a data root gets ``None``, not a guess.

    This is reachable: a report built from a feature set on disk has no corpus to
    classify. ``None`` is visibly incomplete, which is what should happen.
    """
    snapshot = provenance.snapshot()

    assert snapshot["data_root"] is None
    assert snapshot["corpus"] is None
    assert snapshot["feature_fingerprint"] == fingerprint.digest()


def test_the_snapshot_classifies_the_root_it_is_given(tmp_path: Path) -> None:
    (tmp_path / "India").mkdir()
    (tmp_path / "Italy").mkdir()

    snapshot = provenance.snapshot(tmp_path)

    assert snapshot["data_root"] == str(tmp_path)
    assert snapshot["corpus"] == "documented"
    assert snapshot["feature_fingerprint"] == fingerprint.digest()


def test_the_snapshot_carries_the_same_fingerprint_as_the_cache(tmp_path: Path) -> None:
    """One digest, two readers.

    If the report and the feature cache recorded different fingerprints, checking a
    results file against the caches that fed it would be checking two numbers that were
    never meant to agree.
    """
    assert provenance.snapshot(tmp_path)["feature_fingerprint"] == fingerprint.digest()


def test_the_timestamp_is_iso_utc() -> None:
    stamp = str(provenance.snapshot()["generated_at"])

    assert stamp.endswith("+00:00"), stamp
    assert "T" in stamp, stamp


# --------------------------------------------------------------------------- #
# Git state, and the fact that it is allowed not to be available
# --------------------------------------------------------------------------- #


def test_a_missing_git_is_reported_as_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A released wheel is not a checkout, and must still be able to write a report.

    Raising here would make writing a results file depend on having git installed --
    which would break precisely the environment a published artefact runs in.
    """

    def absent(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", absent)

    assert provenance.git_commit() is None
    assert provenance.git_dirty() is None


def test_a_failing_git_command_is_unknown_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(*args: str) -> None:
        return None

    monkeypatch.setattr(provenance, "_run_git", failed)

    assert provenance.git_commit() is None
    assert provenance.git_dirty() is None


def test_the_commit_is_read_from_head(monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "0123456789abcdef0123456789abcdef01234567"

    def head(*args: str) -> str:
        assert args == ("rev-parse", "HEAD")
        return sha

    monkeypatch.setattr(provenance, "_run_git", head)

    assert provenance.git_commit() == sha


def test_an_empty_status_is_clean_and_a_nonempty_one_is_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distinction a report has to make to be reproducible from its own commit."""

    def clean(*args: str) -> str:
        return ""

    def modified(*args: str) -> str:
        return " M src/hemolux/cli.py"

    monkeypatch.setattr(provenance, "_run_git", clean)
    assert provenance.git_dirty() is False

    monkeypatch.setattr(provenance, "_run_git", modified)
    assert provenance.git_dirty() is True


def test_this_checkout_reports_a_real_commit() -> None:
    """The end-to-end version, skipped where there is no repository.

    Worth having alongside the monkeypatched ones: those prove the parsing, this proves
    the call actually reaches git and comes back with something commit-shaped.
    """
    commit = provenance.git_commit()
    if commit is None:
        pytest.skip("not a git checkout")

    assert len(commit) == 40, commit
    assert set(commit) <= set("0123456789abcdef"), commit
