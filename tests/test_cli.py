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
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hemolux.cli import build_parser, main
from hemolux.config import IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]


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
    for command in ("validate", "features", "train", "export", "report"):
        with pytest.raises(SystemExit) as exit_code:
            main([command, "--help"])
        assert exit_code.value.code == 0, command
        assert "usage:" in capsys.readouterr().out


def test_the_subcommands_are_the_documented_five() -> None:
    """The module docstring lists them; a sixth has to be added there too."""
    actions = [a for a in build_parser()._actions if a.dest == "command"]
    assert set(actions[0].choices) == {"validate", "features", "train", "export", "report"}


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
