"""End-to-end tests for the ``hemolux`` command line.

Both of the crashes this file exists to prevent were found by running the commands
by hand, not by a test, and both were the same shape: a subcommand's parser and its
implementation had drifted apart, so the command died on a plain attribute lookup
before it did any work.

* ``hemolux export`` raised ``AttributeError: 'Namespace' object has no attribute
  'backbone'`` because it reached ``_resolve_features`` and declared none of the
  flags that function reads.
* It then raised ``TypeError: export_onnx() got an unexpected keyword argument
  'head'``, because it passed a feature cache to a function that exports a model.

A unit test of ``cmd_validate`` would have missed both, since neither command is
validate. So every subcommand is run end to end here against a generated corpus and
its exit code asserted. That is slower than mocking, and the slowness is the point:
it exercises the parser, the feature cache, the training loop and the ONNX export in
the order a user runs them.

Every artefact is redirected into ``tmp_path``. A test suite that writes checkpoints
and ONNX graphs into the repository is a suite whose runs interfere with each other
and with a developer's own ``hemolux train``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hemolux import fingerprint
from hemolux.cli import build_parser, main
from hemolux.config import IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]

#: One list, used by both the help test and the count test. Written out once here
#: rather than inline in each so that adding a subcommand makes both fail until it has
#: been added to the module docstring too -- which is the review that should happen.
SUBCOMMANDS = ("validate", "features", "train", "sweep", "export", "report")


def _load(name: str):
    """Import a script by path, since ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture_mod = _load("make_synthetic_fixture")

#: Twelve, not six. At six the validation and test folds hold one patient each, and
#: ``regression_report`` refuses a correlation over fewer than two pairs -- correctly,
#: but it made the smallest useful corpus unclear. Twelve gives a fold of two, which
#: is enough to run the whole pipeline and far too few for any number to mean
#: anything. Nothing here asserts on the values.
N_PATIENTS = 12


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A generated corpus, once for the module.

    Six patients: enough for the loader, the split and the heads to all run, and
    few enough that featurising it costs a second rather than a minute.
    """
    root = tmp_path_factory.mktemp("cli") / "synthetic"
    root.mkdir()
    saved = fixture_mod.PATHS
    fixture_mod.PATHS = SimpleNamespace(synthetic=root)
    argv = sys.argv
    sys.argv = ["make_synthetic_fixture.py", "--patients", str(N_PATIENTS), "--force"]
    try:
        assert fixture_mod.main() == 0
    finally:
        sys.argv = argv
        fixture_mod.PATHS = saved
    return root


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Redirect every artefact path into a temporary directory.

    Patched on each module that binds the name at import time, because
    ``ARTIFACT_MODELS`` is a module-level constant in three of them and rebinding it
    in ``config`` alone would leave the other two writing to the repository.
    """
    import hemolux.cli
    import hemolux.export
    import hemolux.training

    root = tmp_path_factory.mktemp("artifacts")
    models, reports = root / "models", root / "reports"
    models.mkdir(parents=True)
    reports.mkdir(parents=True)

    saved = {
        (hemolux.cli, "ARTIFACT_MODELS"): hemolux.cli.ARTIFACT_MODELS,
        (hemolux.cli, "ARTIFACT_REPORTS"): hemolux.cli.ARTIFACT_REPORTS,
        (hemolux.training, "ARTIFACT_MODELS"): hemolux.training.ARTIFACT_MODELS,
        (hemolux.training, "ARTIFACT_REPORTS"): hemolux.training.ARTIFACT_REPORTS,
        (hemolux.export, "ARTIFACT_MODELS"): hemolux.export.ARTIFACT_MODELS,
    }
    for module in (hemolux.cli, hemolux.training):
        module.ARTIFACT_MODELS = models
        module.ARTIFACT_REPORTS = reports
    hemolux.export.ARTIFACT_MODELS = models

    yield SimpleNamespace(models=models, reports=reports, root=root)

    for (module, name), value in saved.items():
        setattr(module, name, value)


# --------------------------------------------------------------------------- #
# The parser
# --------------------------------------------------------------------------- #


def test_every_subcommand_has_help(capsys: pytest.CaptureFixture) -> None:
    """A subcommand whose ``--help`` crashes is a subcommand nobody can find."""
    for command in SUBCOMMANDS:
        with pytest.raises(SystemExit) as exit_code:
            main([command, "--help"])
        assert exit_code.value.code == 0, command
        assert "usage:" in capsys.readouterr().out


def test_the_subcommands_are_the_documented_six() -> None:
    """The module docstring lists them; a seventh has to be added there too."""
    actions = [a for a in build_parser()._actions if a.dest == "command"]
    assert set(actions[0].choices) == set(SUBCOMMANDS)


def test_no_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_code:
        main([])
    assert exit_code.value.code != 0


def test_an_unknown_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit) as exit_code:
        main(["diagnose"])
    assert exit_code.value.code != 0


def test_the_commands_that_featurise_declare_the_flags_they_read() -> None:
    """The regression test for ``--backbone``.

    ``_resolve_features`` reads ``backbone``, ``roi``, ``batch_size``, ``device``,
    ``seed`` and ``data_root``. Whichever subcommands reach it must declare all six,
    or one of them dies on an attribute lookup before doing any work -- which is how
    ``hemolux export`` used to fail.
    """
    required = {"backbone", "roi", "batch_size", "device", "seed", "data_root"}
    parser = build_parser()

    for command in ("features", "train"):
        args = parser.parse_args([command])
        missing = required - set(vars(args))
        assert not missing, f"`hemolux {command}` would raise on {sorted(missing)}"


def test_export_offers_no_backbone_or_roi() -> None:
    """It rebuilds the graph the checkpoint names, so offering them would be a lie.

    If a caller could pass ``--backbone mobilenetv3_large`` to an export, the graph
    would be assembled from one backbone and the weights from another, and it would
    load and predict.
    """
    parser = build_parser()
    subparsers = next(a for a in parser._actions if a.dest == "command")
    flags = {
        option
        for action in subparsers.choices["export"]._actions
        for option in action.option_strings
    }
    assert "--backbone" not in flags
    assert "--roi" not in flags
    assert {"--checkpoint", "--head", "--size", "--out", "--opset"} <= flags


def test_validate_offers_a_way_to_skip_the_contract() -> None:
    """``--contract`` and ``--no-contract`` are mutually exclusive.

    Both default to ``None``, meaning "decide from the corpus", so neither may set
    the destination itself or one of them would silently override the other.
    """
    parser = build_parser()
    assert parser.parse_args(["validate"]).contract is None
    assert parser.parse_args(["validate", "--contract"]).contract is True
    assert parser.parse_args(["validate", "--no-contract"]).contract is False
    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "--contract", "--no-contract"])


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def test_validate_accepts_the_generated_corpus(corpus: Path) -> None:
    assert main(["validate", "--data-root", str(corpus)]) == 0


def test_validate_turns_a_missing_root_into_a_verdict_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A command line is not a caller, so the library's exception becomes a code.

    Exiting 1 with a message is what a shell script or CI step needs. The
    traceback version exits 1 too, but tells the reader nothing about which of a
    hundred possible problems they hit.
    """
    assert main(["validate", "--data-root", str(tmp_path / "absent")]) == 1
    assert "kaggle" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- #
# features -> train -> export -> report
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def trained(corpus: Path, artefacts: SimpleNamespace) -> SimpleNamespace:
    """Run features, train and export once; the tests below read what they wrote.

    stdout is redirected rather than captured because ``capsys`` is function-scoped
    and this fixture is module-scoped. A run prints an epoch per head, which would
    bury the rest of the file's output.
    """
    import contextlib
    import io

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        assert main(["features", "--data-root", str(corpus), "--rebuild"]) == 0
        assert (
            main(
                [
                    "train",
                    "--data-root",
                    str(corpus),
                    "--epochs",
                    "2",
                    "--heads",
                    "ordinal",
                    "--no-site-holdout",
                ]
            )
            == 0
        )
        assert main(["export", "--head", "ordinal"]) == 0
    return artefacts


def test_features_caches_an_npz(trained: SimpleNamespace) -> None:
    caches = list(trained.models.glob("features_*.npz"))
    assert caches, "no feature cache written"


def test_train_writes_a_checkpoint_and_the_three_reports(trained: SimpleNamespace) -> None:
    """A run whose numbers live only on screen is not a run."""
    checkpoints = list(trained.models.glob("ordinal_*.pt"))
    assert len(checkpoints) == 1, f"expected one ordinal checkpoint, got {checkpoints}"
    assert (trained.reports / "results.json").is_file()
    assert (trained.reports / "results.csv").is_file()
    assert (trained.reports / "predictions.csv").is_file()


def test_the_results_file_says_which_corpus_produced_it(trained: SimpleNamespace) -> None:
    """The tracked results file has to be quotable without checking the shell history.

    ``artifacts/reports/`` is committed, so ``results.json`` is the version of the run a
    reader will see. The generated fixture reproduces the real corpus layout deliberately,
    which means this file and one from the 217-patient download have the same shape and
    very different numbers -- and until this block existed, nothing inside either said
    which was which. Asserted on the fixture's own value: ``"synthetic"``, not
    ``"documented"``.
    """
    payload = json.loads((trained.reports / "results.json").read_text(encoding="utf-8"))
    block = payload["provenance"]

    assert block["corpus"] == "synthetic", "the fixture did not identify itself as one"
    assert block["data_root"], "the root the run used is not recorded"
    assert block["feature_fingerprint"] == fingerprint.digest(), (
        "the report and the feature cache disagree about which code produced them"
    )
    assert block["generated_at"], "the run has no timestamp"


def test_the_results_file_carries_the_cross_site_audit(trained: SimpleNamespace) -> None:
    """The decision-axis half of the audit has to survive in the artefact.

    The thresholded numbers are what a screening programme acts on, so they cannot
    live only on screen. The confound string is asserted directly because a
    consumer reading ``per_site`` out of the JSON has no other way to learn that
    site is confounded with exposure in this corpus, and the ordinal head is the
    one that emits a probability, so its calibration block must be available.
    """
    payload = json.loads((trained.reports / "results.json").read_text(encoding="utf-8"))
    audit = payload["site_audit"]["ordinal"]

    assert audit["available"] is True
    assert "confounded with" in audit["confound"], "the exposure confound did not travel"
    assert audit["screening"]["per_site"], "no per-site screening rows were written"
    assert audit["calibration"]["available"] is True, "an ordinal head emits a probability"


# --------------------------------------------------------------------------- #
# Cache routing: each configuration reaches its own file, and nobody else's
# --------------------------------------------------------------------------- #

#: The four configurations the CLI offers, as flag lists. ``balanced`` and ``raw`` are
#: both listed deliberately: they are the pair that a width check cannot separate, since
#: both are thirteen columns wide and differ only in which pixels were divided.
CONFIGS: tuple[tuple[str, ...], ...] = (
    ("--features", "colour", "--balance"),
    ("--features", "colour", "--no-balance"),
    ("--features", "deep"),
    ("--features", "hybrid", "--no-balance"),
)


@pytest.fixture(scope="module")
def routed(corpus: Path, artefacts: SimpleNamespace) -> SimpleNamespace:
    """Extract all four configurations, then ask for each of them again.

    Two runs of each, because the property under test is only visible on the second: the
    first has nothing to read, so it would extract and pass whatever the matching logic
    did. The set of cache filenames is recorded *before* each run as well as the console
    text, because the module-scoped ``artefacts`` directory is shared with the ``trained``
    and ``swept`` fixtures and already holds their files -- so the question is always
    "what did *this* run add or read", never "what exists".
    """
    import contextlib
    import io

    seen: dict[str, SimpleNamespace] = {}
    for flags in CONFIGS:
        argv = ["features", "--data-root", str(corpus), *flags]
        before = {p.name for p in artefacts.models.glob("features_*.npz")}

        first = io.StringIO()
        with contextlib.redirect_stdout(first):
            assert main(argv) == 0, f"{flags} failed on its first run"

        second = io.StringIO()
        with contextlib.redirect_stdout(second):
            assert main(argv) == 0, f"{flags} failed on its second run"

        seen[" ".join(flags)] = SimpleNamespace(
            before=before,
            after={p.name for p in artefacts.models.glob("features_*.npz")},
            first=first.getvalue(),
            second=second.getvalue(),
        )

    artefacts.routed = seen
    return artefacts


def test_each_configuration_writes_a_cache_of_its_own(routed: SimpleNamespace) -> None:
    """Four runs resolve to four distinct files, and none overwrites another.

    Not asserted as "each run added a file", because the shared ``artefacts`` fixture is
    also written by the ``trained`` and ``swept`` fixtures and may already hold one of
    them -- a run that finds its own cache is correct, not broken. So the property is
    stated on the thing that holds either way: at most one file is created per run, and
    the four configurations end up on four different names.

    That is the assertion that catches a collision. If two configurations resolved to the
    same filename, the second would add nothing *and* replace the first's file -- a
    balanced cache lost to a raw one, with no error and no gap in the output, because
    every row of the result would still be finite.
    """
    for label, run in routed.routed.items():
        added = run.after - run.before
        assert len(added) <= 1, f"{label} added {sorted(added)}, expected at most one file"
        assert "wrote" in run.first or "using cached features" in run.first, (
            f"{label} neither wrote a cache nor said it read one: {run.first!r}"
        )

    names = [_cached_name(run.second) for run in routed.routed.values()]
    assert all(names), f"a run reported no cache read: {names}"
    assert len(set(names)) == len(CONFIGS), f"two configurations share a cache: {names}"


def test_balanced_and_raw_never_read_each_others_cache(routed: SimpleNamespace) -> None:
    """Same width, same ROI, same backbone, different pixels.

    This is the substitution the metadata-based match in ``_features_for`` exists to
    prevent, and the one a dimension check would miss: a ``balanced`` cache has 13
    columns and so does a ``raw`` one. Both runs must report reading their own file, and
    neither may report the other's -- a raw cache read as balanced is a wrong number with
    no crash and no gap in the output, because every row is finite and the table still
    prints.
    """
    balanced = _cached_name(routed.routed["--features colour --balance"].second)
    raw = _cached_name(routed.routed["--features colour --no-balance"].second)

    assert balanced and raw, "one of the two did not report a cache read"
    assert balanced != raw, f"both runs read {balanced}"
    assert "balanced" in balanced, f"the balanced run read {balanced}"
    assert "raw" in raw, f"the raw run read {raw}"


def test_a_colour_cache_is_never_read_for_hybrid(routed: SimpleNamespace) -> None:
    """A thirteen-column colour cache is not a 1037-column hybrid cache.

    Included alongside the balanced/raw pair because the two together are the whole
    argument for matching on recorded metadata rather than on filename or width: one
    substitution a width check catches, one it cannot, and only the second is dangerous.
    """
    hybrid = _cached_name(routed.routed["--features hybrid --no-balance"].second)
    colour_raw = _cached_name(routed.routed["--features colour --no-balance"].second)

    assert hybrid and colour_raw
    assert hybrid != colour_raw, f"the hybrid run read the colour cache {colour_raw}"
    assert "hybrid" in hybrid, f"the hybrid run read {hybrid}"
    assert "colour" in colour_raw


def test_the_second_run_reads_a_cache_rather_than_re_extracting(
    routed: SimpleNamespace,
) -> None:
    """Every configuration is cached on its second run.

    The cheap half of the contract. The expensive half -- a cache written by different
    code is refused -- is in ``tests/test_fingerprint.py``, and the two are complementary:
    without this one, a fingerprint that refused everything would pass unnoticed behind a
    feature that always re-extracts.
    """
    for label, run in routed.routed.items():
        assert "using cached features" in run.second, f"{label} re-extracted its own cache"
        assert "extracting features" not in run.second, f"{label} re-extracted its own cache"


def test_rebuild_ignores_a_cache_that_is_there(corpus: Path, routed: SimpleNamespace) -> None:
    """``--rebuild`` has to reach the same resolver the automatic path uses.

    If it took a different route it would be untested by everything above, and the flag
    would work while the cache-matching logic it is meant to override quietly did not.
    """
    import contextlib
    import io

    argv = [
        "features",
        "--data-root",
        str(corpus),
        "--features",
        "colour",
        "--balance",
        "--rebuild",
    ]
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        assert main(argv) == 0

    assert "using cached features" not in sink.getvalue(), "--rebuild still read the cache"
    assert "extracting features" in sink.getvalue(), "--rebuild did not re-extract"


def _cached_name(console: str) -> str:
    """The cache filename a run reported reading, or ``""`` if it did not report one."""
    for line in console.splitlines():
        if "using cached features:" in line:
            return line.rsplit(":", 1)[-1].strip()
    return ""


def test_the_checkpoint_records_what_the_export_needs(trained: SimpleNamespace) -> None:
    """The three things a graph cannot be rebuilt without.

    A checkpoint that omitted the class centres would have to assume them at export
    time, which is right for the ordinal head and wrong for the severity head, and
    the resulting graph would load and predict.
    """
    import torch

    checkpoint = torch.load(
        next(trained.models.glob("ordinal_*.pt")), map_location="cpu", weights_only=True
    )
    assert {"head", "backbone", "feature_dim", "centres", "state_dict"} <= set(checkpoint)
    assert checkpoint["head"] == "ordinal"
    assert len(checkpoint["centres"]) == checkpoint["state_dict"]["fc.weight"].shape[0]


def test_export_writes_a_verified_onnx_graph(trained: SimpleNamespace) -> None:
    """And says it verified, because an export that was never checked is a guess.

    ``verify_onnx`` raises rather than returning a gap, so a graph that disagreed
    with PyTorch would fail the command instead of being written and reported.
    """
    graphs = list(trained.models.glob("*.onnx"))
    assert len(graphs) == 1, f"expected one graph, got {graphs}"
    assert graphs[0].stat().st_size > 0


def test_the_exported_graph_still_matches_pytorch(trained: SimpleNamespace) -> None:
    """Re-checked here, from the file on disk, against a freshly rebuilt model.

    ``cmd_export`` verifies before returning; this makes the same claim
    independently, which is what turns that verification into evidence rather than
    self-attestation. The reference model is rebuilt from the checkpoint rather than
    reused, so a bug in the assembly would show up as a disagreement.
    """
    import onnxruntime as ort

    from hemolux.export import build_screen_model, load_checkpoint

    graph = next(trained.models.glob("*.onnx"))
    checkpoint = load_checkpoint(next(trained.models.glob("ordinal_*.pt")))
    reference = build_screen_model(checkpoint)

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(7).random((2, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    hb, sigma, probs = session.run(None, {"image": x})
    assert hb.shape == (2,)
    assert sigma.shape == (2,)
    assert probs.shape == (2, len(checkpoint["centres"]))

    with torch.no_grad():
        torch_hb, torch_sigma, _ = reference(torch.from_numpy(x))
    assert np.allclose(hb, torch_hb.numpy(), atol=1e-3, rtol=1e-3)
    assert np.allclose(sigma, torch_sigma.numpy(), atol=1e-3, rtol=1e-3)


def test_the_exported_probabilities_sum_to_one(trained: SimpleNamespace) -> None:
    """Checked separately from the haemoglobin value, on purpose.

    A graph whose softmax was folded incorrectly can still produce
    plausible-looking g/dL values, because the posterior mean is a weighted average
    and would merely be slightly off. The distribution summing to one is the
    property that would catch it.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(next(trained.models.glob("*.onnx"))), providers=["CPUExecutionProvider"]
    )
    x = np.random.default_rng(11).random((3, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    _, _, probs = session.run(None, {"image": x})

    assert np.all(probs >= 0.0), "negative probability"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_export_refuses_when_no_checkpoint_exists(
    corpus: Path, artefacts: SimpleNamespace, capsys: pytest.CaptureFixture
) -> None:
    """The other direction, and the one that would ship a random graph.

    ``build_model(pretrained=False)`` builds a working model from random weights, so
    an export with nothing to load would succeed, produce a file, and predict
    nonsense. Refusing is the only safe behaviour.

    The graph check is a set difference rather than an emptiness check: an earlier
    version of this test asserted ``not list(artefacts.models.glob("*.onnx"))``, which
    failed for the wrong reason -- the shared fixture had already exported one for
    ``ordinal``. Asserting emptiness tests the order tests run in.
    """
    before = {p.name for p in artefacts.models.glob("*.onnx")}

    assert main(["export", "--head", "binary"]) == 1

    err = capsys.readouterr().err
    assert "no binary checkpoint" in err
    assert {p.name for p in artefacts.models.glob("*.onnx")} == before


def test_train_reports_a_corpus_too_small_to_evaluate(
    corpus: Path, artefacts: SimpleNamespace, capsys: pytest.CaptureFixture
) -> None:
    """A verdict about the corpus, not a traceback from inside a metrics helper.

    A validation fraction that consumes the corpus leaves one fold empty. Every
    evaluation here needs two pairs to compute a correlation, so the run cannot
    report anything -- but "need at least one array to concatenate" names a numpy
    internal and not a cause. The message says which fold came out empty, how many
    patients were available, and which knob sized it.
    """
    import contextlib
    import io

    before = {p.name for p in artefacts.models.glob("*.pt")}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        code = main(
            [
                "train",
                "--data-root",
                str(corpus),
                "--epochs",
                "1",
                "--heads",
                "ordinal",
                "--val-fraction",
                "0.95",
                "--no-site-holdout",
            ]
        )
    assert code == 1

    err = capsys.readouterr().err
    assert "too few to evaluate" in err
    assert "at least 2" in err, "the requirement is not stated"
    assert "--val-fraction" in err, "the message does not name the knob"
    assert "out of 12" in err, "the message does not size the corpus"
    assert {p.name for p in artefacts.models.glob("*.pt")} == before, (
        "a run that refused to train still wrote a checkpoint"
    )


def test_report_reprints_without_retraining(
    trained: SimpleNamespace, capsys: pytest.CaptureFixture
) -> None:
    """The point of `report`: read the last run, do not recompute it.

    Only the single-split table is asserted, because this run used
    ``--no-site-holdout`` to keep the fixture cheap. Asserting a cross-site section
    that the run was told not to produce would be asserting a lie.
    """
    assert main(["report"]) == 0
    out = capsys.readouterr().out
    assert "ordinal" in out
    assert "results.json" in out, "the file it read is not named"
    assert "holdout" not in out, "a cross-site section appeared for a run without one"


def test_report_fails_loudly_when_there_is_nothing_to_reprint(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Printing an empty table would look like a result.

    A missing file is the single most likely thing to go wrong here -- a fresh
    checkout, a cleaned artifacts directory -- so it is a verdict, not a blank
    screen.
    """
    assert main(["report", "--results", str(tmp_path / "absent.json")]) == 1
    assert "does not exist" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# sweep: configuration selection that does not read the test fold
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def swept(corpus: Path, artefacts: SimpleNamespace) -> SimpleNamespace:
    """Run the sweep once over a three-candidate subset; the tests below read the file.

    The colour candidates only, so the fixture stays cheap: featurising with the frozen
    backbone costs orders of magnitude more than the colour path and none of these tests
    are about the backbone. Redirected rather than captured for the reason ``trained``
    documents -- ``capsys`` is function-scoped and this fixture is module-scoped.
    """
    import contextlib
    import io

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        code = main(
            [
                "sweep",
                "--data-root",
                str(corpus),
                "--epochs",
                "2",
                "--only",
                "colour/palpebral/raw,colour/palpebral/balanced,colour/forniceal/raw",
            ]
        )
    assert code == 0
    artefacts.sweep_stdout = sink.getvalue()
    return artefacts


def test_the_sweep_records_that_it_never_scored_a_candidate_on_test(
    swept: SimpleNamespace,
) -> None:
    """One row per candidate, and exactly one of them carries a test score.

    This is the artefact the whole command exists to produce, so it is asserted from
    the file on disk rather than from the objects in memory: what a reader of the JSON
    can check is the only version that matters.
    """
    payload = json.loads((swept.reports / "sweep.json").read_text(encoding="utf-8"))

    assert len(payload["candidates"]) == 3
    assert payload["test_read"] == "once, after selection"
    assert payload["test"] is not None
    assert sum(1 for c in payload["candidates"] if c["selected"]) == 1
    assert payload["winner"] == next(c["label"] for c in payload["candidates"] if c["selected"])


def test_the_sweep_file_says_which_corpus_produced_it(swept: SimpleNamespace) -> None:
    """The same claim as the single-split report, on the file that carries the winner.

    This is the artefact ``EVALS.md`` will quote, so it is also the one most likely to be
    lifted out of context. It names its corpus and its code for the same reason.
    """
    payload = json.loads((swept.reports / "sweep.json").read_text(encoding="utf-8"))
    block = payload["provenance"]

    assert block["corpus"] == "synthetic"
    assert block["feature_fingerprint"] == fingerprint.digest()
    assert block["generated_at"]
    assert set(block) == {
        "data_root",
        "corpus",
        "feature_fingerprint",
        "git_commit",
        "git_dirty",
        "generated_at",
    }


def test_every_losing_candidate_records_validation_only(
    swept: SimpleNamespace,
) -> None:
    """The losing rows carry no key that could hold a test number.

    Not merely "the numbers differ" -- there is nowhere for one to be. A sweep whose
    discarded candidates carried test metrics would be a test-set search wearing a
    validation split's results table.
    """
    payload = json.loads((swept.reports / "sweep.json").read_text(encoding="utf-8"))

    for row in payload["candidates"]:
        keys = set(row)
        assert {"val_mae", "val_r2"} <= keys
        assert not ({"test_mae", "test_r2", "mae", "r2"} & keys), (
            f"{row['label']} carries a key that could hold a test score"
        )


def test_the_sweep_table_is_ordered_by_the_metric_it_selected_on(
    swept: SimpleNamespace,
) -> None:
    """Best-first, so the first row is the choice and the reader need not re-sort.

    Asserted on the ordering rather than on any particular winner: the fixture has
    twelve patients and its numbers are meaningless. What has to hold is that the
    table is sorted by the metric the run declared, because a table whose order
    disagrees with its own ``metric`` field sends the reader to the wrong row.
    """
    payload = json.loads((swept.reports / "sweep.json").read_text(encoding="utf-8"))

    maes = [c["val_mae"] for c in payload["candidates"]]
    assert maes == sorted(maes), f"the table is not ordered: {maes}"
    assert payload["metric"] == "mae"
    assert payload["candidates"][0]["selected"] is True


def test_a_candidate_reports_the_patients_it_measured_without_its_roi(
    swept: SimpleNamespace,
) -> None:
    """A candidate that is not homogeneous says which patients it is not homogeneous for.

    The fixture has two patients with no forniceal mask, so the forniceal candidate
    measures them on the whole frame while every other row describes the conjunctiva.
    That configuration is still swept -- dropping six patients would shift the fold for
    one candidate only, which is the worse confound -- but the count belongs in the
    artefact, because a table row and its ``roi`` field both read as though the ROI were
    used for everyone.

    This is also the regression test for the bug that made it unrunnable: ``roi_mean``
    averaged the wrong axes on an unmasked frame and handed numpy a (H, W, 3) array
    where a length-13 row belonged, aborting the extraction with "inhomogeneous shape
    after 1 dimensions" and naming neither the patient nor the ROI.
    """
    payload = json.loads((swept.reports / "sweep.json").read_text(encoding="utf-8"))

    counts = {c["label"]: c["n_unmasked"] for c in payload["candidates"]}
    assert counts["colour / palpebral / raw"] == 0
    assert counts["colour / palpebral / balanced"] == 0
    assert counts["colour / forniceal / raw"] == 2
    assert payload["n_unmasked_by_label"] == {"colour / forniceal / raw": 2}


def test_the_sweep_writes_one_row_per_candidate_and_no_patient_rows(
    swept: SimpleNamespace,
) -> None:
    """The CSV is aggregate, so it is safe to commit and adds no patient data.

    ``predictions.csv`` next to it is kept out of the repository for exactly this
    reason. A sweep writing its own patient table would need the same treatment, and
    the validation split is the same patients on every run.
    """
    with (swept.reports / "sweep.csv").open(encoding="utf-8") as handle:
        lines = handle.read().strip().splitlines()

    assert len(lines) == 4, "header plus one row per candidate"
    assert "val_mae" in lines[0]
    # Patient ids are site-qualified -- ``India/5``, ``Italy/109`` -- so a capitalised
    # segment followed by a number is the shape to look for. A bare "/" is not enough:
    # the candidate labels contain them too (``colour / palpebral / raw``).
    assert not any(
        re.match(r"^[A-Z][a-z]+/\d+$", line.split(",")[0].strip()) for line in lines[1:]
    ), "the first column holds a patient identifier"


def test_a_filter_that_matches_part_of_the_grid_is_refused(
    corpus: Path, artefacts: SimpleNamespace, capsys: pytest.CaptureFixture
) -> None:
    """A typo must not quietly shrink the experiment.

    Matching three of four terms would otherwise produce a shorter table that reads as
    complete: every row in it looks like a legitimate result, and the missing
    configuration is indistinguishable from one that lost. So the run refuses, names
    what did not match, and prints the spellings it would have accepted.
    """
    import contextlib
    import io

    before = {p.name for p in artefacts.models.glob("features_*.npz")}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        code = main(
            [
                "sweep",
                "--data-root",
                str(corpus),
                "--epochs",
                "1",
                "--only",
                "colour/palpebral/raw,colour/caruncle/raw",
            ]
        )

    assert code == 2
    err = capsys.readouterr().err
    assert "caruncle" in err
    assert "accepted spellings" in err, "the command does not say what it would have accepted"
    assert {p.name for p in artefacts.models.glob("features_*.npz")} == before, (
        "a refused sweep still extracted features"
    )


def test_a_filter_may_name_a_candidate_without_its_balance(
    corpus: Path,
    artefacts: SimpleNamespace,
) -> None:
    """``--only deep/palpebral`` selects the deep candidate.

    Deep has no illuminant to vary, so demanding the balance suffix there would make
    the obvious spelling select nothing. The sweep would still run -- on one fewer
    candidate than the reader asked for -- which is the failure this avoids.
    """
    from hemolux.sweep import Candidate, default_grid, sweep_label

    grid = default_grid()
    deep = next(c for c in grid if c.kind == "deep" and c.roi == "palpebral")

    spellings = {
        f"{deep.kind}/{deep.roi}",
        f"{deep.kind}/{deep.roi}/{deep.balance}",
        sweep_label(deep),
    }
    assert "deep/palpebral" in spellings
    assert Candidate("deep", "palpebral").balance == "balanced"


def test_the_sweep_refuses_a_validation_fold_of_one(
    corpus: Path, artefacts: SimpleNamespace, capsys: pytest.CaptureFixture
) -> None:
    """Selection reads validation for every candidate, so one patient cannot serve.

    Different from ``train``'s refusal, and deliberately so: a training run needs two
    patients only to compute a correlation afterwards, whereas a sweep needs a
    comparable number *per candidate*. A fold of one would still let the command run
    and rank configurations on a single patient each.
    """
    import contextlib
    import io

    before = {p.name for p in artefacts.reports.glob("sweep.*")}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        code = main(
            [
                "sweep",
                "--data-root",
                str(corpus),
                "--epochs",
                "1",
                "--val-fraction",
                "0.95",
                "--only",
                "colour/palpebral/raw",
            ]
        )

    assert code == 1
    err = capsys.readouterr().err
    assert "candidate" in err, "the message does not say why a fold of one is not enough"
    assert "--val-fraction" in err, "the message does not name the knob"
    assert {p.name for p in artefacts.reports.glob("sweep.*")} == before, (
        "a refused sweep still wrote a report"
    )
