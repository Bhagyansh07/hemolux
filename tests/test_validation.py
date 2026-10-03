"""Tests for the corpus contract and the corpus report.

These two live in one module because the second was written after the first and
overlapped it. ``validate_corpus`` already answered "is this the corpus the
documentation names?" with twenty literal assertions. A separate
``scripts/validate_dataset.py`` was then added to answer "what is actually in it,
and what does that forbid concluding?", and it re-checked identity uniqueness,
the haemoglobin range and the count of patients without a forniceal mask. Two
entry points that both print "validate" and do not quite agree on what a pass
means is a defect that shows up later as a question with no answer, so the second
became part of this module and the script is gone.

The distinction the merged report has to keep straight, because it is the whole
value of running it:

FAIL
    The corpus is not what it must be, so a number computed from it would be
    wrong rather than imprecise. Exit code 1.

WARN
    A true fact that limits what may be concluded. The severe band holds nobody,
    so sensitivity in it cannot be computed here. Nothing is broken, and no exit
    code. But it has to be *read*, which is why it is a separate line rather than
    a sentence in a passing check's detail.

OK
    Clean.

The most load-bearing test in the file is the round trip at the end: generate a
fixture with the generator, then require the validator to accept it. That is the
only thing that catches a layout which only one of the two believes in, and it
could not be written before the validator existed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hemolux.config import SYNTHETIC_MARKER
from hemolux.validation import (
    EXPECTED_TOTAL,
    ROI_AREA_MAX,
    ROI_AREA_MIN,
    Check,
    CorpusReport,
    Severity,
    validate_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    """Import a script by path, since ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture_mod = _load("make_synthetic_fixture")


# --------------------------------------------------------------------------- #
# The severity model
# --------------------------------------------------------------------------- #


def test_a_passing_check_is_ok_by_default() -> None:
    assert Check("clean", True).level is Severity.OK


def test_a_failing_check_is_a_failure_whatever_its_declared_severity() -> None:
    """The one rule that keeps WARN from becoming a place to hide defects.

    ``severity`` is the level to report at *when the check passes*. A check that
    fails is a FAIL, so setting ``severity=WARN`` on a red line cannot downgrade
    it into a footnote -- which is exactly what the four WARN-intended checks in
    ``_geometry_checks`` would otherwise have done.
    """
    assert Check("broken", False, severity=Severity.WARN).level is Severity.FAIL
    assert Check("broken", False, severity=Severity.OK).level is Severity.FAIL


def test_a_passing_check_can_be_flagged_as_a_warning() -> None:
    check = Check("empty severe band", True, "0 patients", severity=Severity.WARN)
    assert check.passed is True
    assert check.level is Severity.WARN


def test_severity_str_enum_prints_as_its_name() -> None:
    """The report prints the level bare, so ``Severity.FAIL`` must not render as
    ``Severity.FAIL``."""
    assert str(Severity.FAIL) == "FAIL"
    assert Severity("WARN") is Severity.WARN


def test_a_warning_does_not_fail_the_report() -> None:
    """The exit code is a verdict on the corpus, not on how much there is to say
    about it. A corpus with an empty severity band is a usable corpus."""
    report = CorpusReport(root=Path())
    report.checks += [
        Check("counts", True),
        Check("empty severe band", True, "0 patients", severity=Severity.WARN),
    ]
    assert report.ok is True
    assert report.failures == []
    assert len(report.warnings) == 1


def test_a_failure_fails_the_report_and_is_listed_separately() -> None:
    report = CorpusReport(root=Path())
    report.checks += [
        Check("fine", True),
        Check("severe band", True, "0", severity=Severity.WARN),
        Check("impossible hb", False, "hb=90"),
    ]
    assert report.ok is False
    assert [c.label for c in report.failures] == ["impossible hb"]
    assert [c.label for c in report.warnings] == ["severe band"]


def test_the_report_serialises_with_no_unserialisable_values() -> None:
    """``--json`` used to raise ``TypeError: Object of type method is not JSON
    serializable``.

    ``validate_corpus`` returned a dict containing a bound method and a live
    ``CorpusReport``, so every ``--json`` run crashed after doing all the work.
    Asserted by round-tripping through ``json.dumps``, which is the only way to
    catch a non-serialisable value added later.
    """
    report = CorpusReport(root=Path())
    report.checks += [Check("clean", True, "detail"), Check("warned", True, severity=Severity.WARN)]
    report.stats = {"n_patients": EXPECTED_TOTAL}

    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["ok"] is True
    assert payload["n_failed"] == 0
    assert payload["n_warned"] == 1
    assert {c["level"] for c in payload["checks"]} == {"OK", "WARN"}


def test_the_rendered_report_names_both_severities(capsys) -> None:
    """A warning a reader cannot find is a warning that did nothing.

    The FAIL list and the WARN list are printed under separate headings. Merging
    them into one "not everything passed" section is what turns a constraint on a
    conclusion into a shrug.
    """
    report = CorpusReport(root=Path())
    report.checks += [
        Check("broken thing", False, "because"),
        Check("empty band", True, "no patients", severity=Severity.WARN),
    ]
    report.print()
    out = capsys.readouterr().out
    assert "A FAIL means a number computed from this corpus would be wrong" in out
    assert "A WARN is a true fact that limits what may be concluded" in out
    assert "broken thing" in out
    assert "empty band" in out


# --------------------------------------------------------------------------- #
# A corpus to run against
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthetic_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A generated fixture, produced once for the whole module.

    Generating it is slow enough that running it per-test would dominate the
    file, and every test here reads the same corpus.
    """
    root = tmp_path_factory.mktemp("corpus") / "synthetic"
    root.mkdir()
    saved = fixture_mod.PATHS
    # A namespace, not the real dataclass: PATHS is frozen and carries every
    # artifact path, and a fixture only needs to redirect one of them.
    fixture_mod.PATHS = SimpleNamespace(synthetic=root)
    argv = sys.argv
    sys.argv = ["make_synthetic_fixture.py", "--patients", "8", "--force"]
    try:
        assert fixture_mod.main() == 0
    finally:
        sys.argv = argv
        fixture_mod.PATHS = saved
    return root


@pytest.fixture(scope="module")
def synthetic_report(synthetic_corpus: Path) -> CorpusReport:
    return validate_corpus(synthetic_corpus)


def test_a_generated_corpus_validates(synthetic_report: CorpusReport) -> None:
    """The round trip.

    If the generator and the loader ever disagree about the layout again, this
    fails first and names the disagreement, instead of the first training run
    discovering it.
    """
    failures = [c.label for c in synthetic_report.failures]
    assert not failures, f"the validator rejected the generator's own output: {failures}"


def test_the_report_covers_both_kinds_of_question(synthetic_report: CorpusReport) -> None:
    """Both halves must be in one place. A reader who has only seen the structural
    checks does not know what the corpus contract would have told them."""
    labels = [c.label for c in synthetic_report.checks]
    assert any("severity band" in label for label in labels), "band histogram missing"
    assert any("area fraction" in label for label in labels), "mask geometry missing"
    assert any("holdout" in label for label in labels), "holdout population missing"
    assert any("haemoglobin value lies in" in label for label in labels), "label domain missing"


def test_the_corpus_contract_is_skipped_for_a_marked_synthetic_corpus(
    synthetic_corpus: Path,
) -> None:
    """The bug this module parameter exists for.

    The generator reproduces the real layout, so the fixture is indistinguishable
    from the download by shape. Without the marker check, eight patients produced
    six FAILs about a 217-patient corpus -- and a validator whose red lines are
    routinely wrong is a validator whose red lines get ignored.
    """
    assert any(p.name.startswith(SYNTHETIC_MARKER) for p in synthetic_corpus.iterdir()), (
        "the fixture no longer marks itself, so nothing tells the validator it is synthetic"
    )

    default = validate_corpus(synthetic_corpus)
    assert not any("217 usable patients" in c.label for c in default.checks)

    forced = validate_corpus(synthetic_corpus, contract=True)
    assert any("217 usable patients" in c.label for c in forced.checks), (
        "--contract did not bring the literal assertions back"
    )


def test_forcing_the_contract_onto_a_small_corpus_fails_it(synthetic_corpus: Path) -> None:
    """The other direction. If the contract passes on an eight-patient corpus, then
    it is not asserting anything."""
    forced = validate_corpus(synthetic_corpus, contract=True)
    assert forced.ok is False
    assert forced.failures


def test_every_band_is_listed_even_when_empty(synthetic_report: CorpusReport) -> None:
    """A band holding nobody has to appear as a row, because the reader looking
    for sensitivity in that band is exactly the reader who needs to see the zero."""
    labels = [c.label for c in synthetic_report.checks]
    for band in ("normal", "mild", "moderate", "severe"):
        assert f"severity band {band}" in labels


def test_the_geometric_bounds_are_the_ones_the_quality_gate_uses() -> None:
    """A validator that keeps its own copy of a threshold reports on a gate that no
    longer exists.

    These were literals here before, ``0.015`` and ``0.85``, while
    ``data/quality.py`` held the same two numbers for the gate that actually
    rejects frames. Changing the gate would have left the report describing the old
    one. They are imported now, and this asserts the binding is still an import and
    not a re-statement.
    """
    from hemolux.data import quality

    assert ROI_AREA_MIN is quality.ROI_MIN_AREA
    assert ROI_AREA_MAX is quality.ROI_MAX_AREA


def test_area_fractions_are_reported_as_fractions_not_pixels(
    synthetic_report: CorpusReport,
) -> None:
    """The masks are stored smaller than their frames, so a pixel count would
    describe the storage format rather than the anatomy."""
    area_rows = [c for c in synthetic_report.checks if c.label.endswith("area fraction")]
    assert area_rows, "no area-fraction row"
    for row in area_rows:
        assert row.detail.startswith("min "), row.detail
        assert "median" in row.detail


def test_skipping_pixels_is_reported_rather_than_silent(synthetic_corpus: Path) -> None:
    """``--no-pixels`` makes the run fast. It must not make it look complete."""
    report = validate_corpus(synthetic_corpus, read_pixels=False)
    warned = [c.label for c in report.warnings]
    assert "pixel geometry not verified" in warned
    assert not any(c.label.endswith("area fraction") for c in report.checks)


def test_no_pixels_really_does_skip_the_decoding(synthetic_corpus: Path) -> None:
    """Otherwise the flag would be a claim rather than a switch."""
    fast = validate_corpus(synthetic_corpus, read_pixels=False)
    assert len(fast.checks) < len(validate_corpus(synthetic_corpus).checks)


def test_an_absent_root_raises_with_instructions(tmp_path: Path) -> None:
    """A validator that reports success without having looked at anything is worse
    than no validator, so the missing-directory case must not be a silent pass."""
    with pytest.raises(FileNotFoundError, match="kaggle datasets download"):
        validate_corpus(tmp_path / "absent")


def test_a_corpus_with_the_wrong_layout_fails_rather_than_passing(
    tmp_path: Path,
) -> None:
    """The exact state the old generator left behind: one combined directory and
    no per-site workbooks. Asserted so it cannot return."""
    (tmp_path / "patients" / "P0000").mkdir(parents=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_corpus(tmp_path)


# --------------------------------------------------------------------------- #
# The CLI surface
# --------------------------------------------------------------------------- #


def _run_cli(argv: list[str]) -> int:
    from hemolux.cli import main

    return main(argv)


def test_validate_exits_zero_on_a_good_corpus(synthetic_corpus: Path) -> None:
    assert _run_cli(["validate", "--data-root", str(synthetic_corpus)]) == 0


def test_validate_exits_one_on_a_root_that_does_not_exist(tmp_path: Path) -> None:
    """A command line is not a caller, so the library's exception has to become a
    verdict rather than a traceback."""
    assert _run_cli(["validate", "--data-root", str(tmp_path / "absent")]) == 1


def test_validate_json_is_machine_readable(synthetic_corpus: Path, capsys) -> None:
    """The regression test for the ``TypeError`` above, at the level a user sees."""
    assert _run_cli(["validate", "--data-root", str(synthetic_corpus), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["n_checks"] == len(payload["checks"])
    assert all("level" in c for c in payload["checks"])


@pytest.mark.parametrize("argv", [["validate"], ["features"], ["train"], ["export"], ["report"]])
def test_every_subcommand_has_help(argv: list[str], capsys) -> None:
    """A subcommand whose ``--help`` crashes is a subcommand nobody can discover."""
    with pytest.raises(SystemExit) as exc:
        _run_cli([*argv, "--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_no_subcommand_is_a_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        _run_cli([])
    assert exc.value.code != 0


def test_an_unknown_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        _run_cli(["diagnose"])
    assert exc.value.code != 0
