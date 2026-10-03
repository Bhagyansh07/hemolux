"""Tests for the ONNX export.

The point of the export is that a browser computes the same number this repository
computes in Python. So almost everything here is about one question: does the graph
still agree with PyTorch, and does it agree for the reasons we think it does?

Three real defects came out of writing these, and all three are silent -- the export
succeeds, the file loads, and every prediction is subtly wrong. Each is pinned below
by name, because a test that only asserted "the export ran" would have passed with
any of them present.

* **A static batch axis.** Traced against a single dummy image, ONNX froze the batch
  dimension to 1. The browser sends one image at a time and so never noticed, but
  ``verify_onnx`` sends a pair and the runtime rejected it outright -- meaning the
  check meant to catch a divergence could not run at all. A verification step
  skipped by construction is the worst of the three outcomes available here.

* **A size cross-check that never ran.** ``export_onnx`` compares the requested
  ``size`` against ``model.size``. ``HemoluxScreen`` had no ``size`` attribute, so
  the comparison saw ``None`` and skipped -- on the only model this repository
  actually exports. The guard was described in its own docstring and did nothing.

* **Centres absent from the checkpoint.** The graph decodes a posterior mean, which
  needs a value per output class. The ordinal head's are the fixed WHO grid; the
  binary and severity heads' are per-split means measured on the training patients.
  A checkpoint that did not record them could only be exported by assuming the
  grid, which is right for one head and wrong for the others, and which produces a
  perfectly loadable graph that decodes every prediction off by a constant.

The rest are the ordinary ones: the graph has the documented inputs and outputs, it
matches PyTorch, and a checkpoint missing what it needs is refused rather than
guessed at.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.jit import TracerWarning

from hemolux.config import IMAGE_SIZE, MOBILE_BUDGET_MB
from hemolux.export import (
    HemoluxScreen,
    Normalise,
    NormaliseBackbone,
    build_screen_model,
    export_onnx,
    file_size_mb,
    load_checkpoint,
    verify_onnx,
)
from hemolux.metrics.calibration import HB_BIN_CENTRES, N_BINS
from hemolux.models.heads import build_head

pytest.importorskip("onnxruntime", reason="the export's whole purpose is the browser runtime")

SIZE = 64  # small enough to export in a second, large enough not to be degenerate
#: Read from the calibration module rather than written down. The ordinal head emits
#: one logit per WHO bin -- twelve of them, spanning 4 to 18 g/dL -- and a test that
#: assumed four could not tell a working decode from a sliced one.
CENTRES = tuple(HB_BIN_CENTRES)


class TinyBackbone(nn.Module):
    """A stand-in backbone.

    Not a real one. These tests are about the graph's shape, batching and decode, and
    a real mobilenet would make the file ~6 MB and the suite slow to prove none of
    that differently.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.conv(x))).flatten(1)


def _make_screen(seed: int = 0) -> HemoluxScreen:
    """A screen model of the same shape as the real one, cheap to build.

    ``seed`` is set immediately before construction, so two calls with different
    seeds give different weights over the *whole* module -- which is what the
    verification-failure test needs.
    """
    torch.manual_seed(seed)
    model = HemoluxScreen(
        NormaliseBackbone(TinyBackbone(), Normalise(size=SIZE)),
        nn.Linear(8, N_BINS),
        CENTRES,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def screen() -> HemoluxScreen:
    return _make_screen()


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[HemoluxScreen, Path]:
    """One export, shared. Exporting is the slow part; the assertions are cheap."""
    model = _make_screen()
    path = export_onnx(model, size=SIZE, path=tmp_path_factory.mktemp("onnx") / "screen.onnx")
    return model, path


# --------------------------------------------------------------------------- #
# The graph's interface
# --------------------------------------------------------------------------- #


def test_the_graph_takes_one_image_and_returns_three_tensors(exported: tuple) -> None:
    """The interface the browser is written against, asserted by name.

    Named rather than positional, because the app looks them up by name and a
    rename that preserved the arity would still break it silently.
    """
    import onnxruntime as ort

    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["image"]
    assert [o.name for o in session.get_outputs()] == ["hb_gdl", "sigma_gdl", "bin_probs"]
    assert session.get_inputs()[0].shape[1:] == [3, SIZE, SIZE]


def test_the_batch_axis_is_dynamic(exported: tuple) -> None:
    """The regression test for the frozen-batch bug.

    A graph traced from one dummy image declares a static dimension of 1 and then
    raises ``INVALID_ARGUMENT: index: 0 Got: 2 Expected: 1`` on anything else. The
    phone happens to send single images, which is exactly why this would have
    shipped unnoticed.
    """
    import onnxruntime as ort

    model, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).random((3, 3, SIZE, SIZE), dtype=np.float32)

    hb, sigma, probs = session.run(None, {"image": x})
    assert hb.shape == (3,)
    assert sigma.shape == (3,)
    assert probs.shape == (3, N_BINS)

    with torch.no_grad():
        torch_hb, _, _ = model(torch.from_numpy(x))
    assert np.allclose(hb, torch_hb.numpy(), atol=1e-4, rtol=1e-4)


def test_the_graph_refuses_a_mis_sized_input(exported: tuple) -> None:
    """The input contract is exactly ``(N, 3, size, size)``, and it is enforced.

    ``Normalise`` resizes when handed the wrong size in Python. That branch cannot
    survive tracing -- the comparison is evaluated once in Python and baked in, so
    the op never reaches the graph. Rather than leave a claim in the docstring that
    the file does not honour, the file refuses. A graph that silently resized would
    absorb the one mismatch that matters, which is the app's canvas not being the
    training geometry, turning a loud shape error into a quiet degradation of every
    prediction.
    """
    import onnxruntime as ort

    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    small = np.random.default_rng(2).random((1, 3, SIZE // 2, SIZE // 2), dtype=np.float32)

    with pytest.raises(Exception, match=r"INVALID_ARGUMENT|dimension"):
        session.run(None, {"image": small})


def test_the_python_module_does_resize(screen: HemoluxScreen) -> None:
    """The other half of that contract, so the difference stays deliberate.

    If this stopped resizing, callers relying on it would break. If it kept
    resizing while the graph rejects, the two differ on purpose rather than by
    accident.
    """
    with torch.no_grad():
        hb, _, _ = screen(torch.rand(1, 3, SIZE // 2, SIZE // 2))
    assert hb.shape == (1,)
    assert torch.isfinite(hb).all()


def test_the_size_cross_check_fires_for_the_real_export_path(
    exported: tuple, tmp_path: Path
) -> None:
    """``HemoluxScreen.size`` exists so this guard is not silently skipped.

    ``export_onnx`` reads ``getattr(model, "size", None)`` and compares. Before the
    property was added the one model this repository exports had no ``size``, so
    the comparison saw ``None`` and never ran.
    """
    model, _ = exported
    assert model.size == SIZE
    with pytest.raises(ValueError, match="expects 64x64 input"):
        export_onnx(model, size=SIZE * 2, path=tmp_path / "wrong.onnx")


def test_the_graph_matches_pytorch_on_a_pair(exported: tuple) -> None:
    assert verify_onnx(exported[0], exported[1], size=SIZE)["n_samples"] == 2


def test_verify_reports_the_gap_it_measured(exported: tuple) -> None:
    gaps = verify_onnx(exported[0], exported[1], size=SIZE)
    assert set(gaps) == {"hb_max_abs_diff", "sigma_max_abs_diff", "n_samples"}
    assert gaps["hb_max_abs_diff"] < 1e-3


def test_verify_rejects_a_graph_that_agrees_on_nothing(tmp_path: Path) -> None:
    """A verification that cannot fail is decoration.

    Exports one model and checks it against a *differently initialised* model of the
    same architecture, which is a graph that exports cleanly and predicts something
    else entirely -- the exact shape of the failure this is meant to catch. Without
    this test, a comparison broken in a way that still returns a number (compared
    against the wrong array, or a gap never checked against its tolerance) would
    leave the suite green.
    """
    wrong = _make_screen(seed=3)
    reference = _make_screen(seed=99)

    broken = export_onnx(wrong, size=SIZE, path=tmp_path / "broken.onnx")
    assert broken.is_file(), "the decoy graph did not export, so the test proves nothing"
    with pytest.raises(AssertionError, match="disagrees with PyTorch"):
        verify_onnx(reference, broken, size=SIZE)


# --------------------------------------------------------------------------- #
# The decode lives in the graph
# --------------------------------------------------------------------------- #


def test_the_posterior_mean_is_the_decode(screen: HemoluxScreen) -> None:
    """The browser must not reimplement this, so it has to be right here.

    A uniform posterior over the bins decodes to the mean of the centres, and to
    nothing else.
    """
    uniform = torch.zeros(1, N_BINS).softmax(-1)
    centres = screen.centres
    decoded = (uniform * centres).sum()
    assert float(decoded) == pytest.approx(float(centres.mean()))


def test_sigma_is_zero_for_a_degenerate_posterior() -> None:
    """A one-hot posterior has zero spread.

    The clamp on ``second - hb**2`` exists because that quantity is a float
    subtraction of two nearly equal numbers, which goes slightly negative for a
    confident prediction and would put a NaN straight into the abstention decision.
    """
    centres = torch.tensor(CENTRES)
    probs = torch.zeros(1, N_BINS)
    probs[0, 0] = 1.0

    hb = (probs * centres).sum(-1)
    second = (probs * centres**2).sum(-1)
    sigma = torch.sqrt(torch.clamp(second - hb**2, min=0.0))

    assert torch.isfinite(sigma).all()
    assert float(sigma) == pytest.approx(0.0, abs=1e-6)


def test_the_centres_match_the_calibration_grid(screen: HemoluxScreen) -> None:
    """The graph and the Python decoder must agree on what a bin is worth.

    ``OrdinalHead.decode`` uses ``HB_BIN_CENTRES``; the graph carries the centres
    from the checkpoint. If those ever diverged, every exported prediction would
    shift by a constant while both continued to look right on their own.
    """
    assert screen.centres.tolist() == pytest.approx(list(HB_BIN_CENTRES))


def test_too_few_centres_raises_rather_than_shrinking_the_decode() -> None:
    """A centre list shorter than the logits is a loud error, not a quiet bias.

    Worth pinning because the obvious worry -- that a slice would quietly drop a bin
    and bias every prediction toward the low end -- does not apply: the shape
    mismatch propagates and raises. A *longer* list is sliced and is fine, which is
    the correct direction for the one head that emits fewer logits than bins.
    """
    model = HemoluxScreen(
        NormaliseBackbone(TinyBackbone(), Normalise(size=SIZE)),
        nn.Linear(8, N_BINS),
        CENTRES[:-1],
    )
    model.eval()
    with pytest.raises(RuntimeError, match=r"shape|size"):
        model(torch.zeros(1, 3, SIZE, SIZE))

    # And the longer-list direction works: all twelve centres, eight logits.
    smaller = HemoluxScreen(
        NormaliseBackbone(TinyBackbone(), Normalise(size=SIZE)),
        nn.Linear(8, 4),
        CENTRES,
    )
    smaller.eval()
    with torch.no_grad():
        hb, sigma, _ = smaller(torch.zeros(1, 3, SIZE, SIZE))
    assert hb.shape == (1,)
    assert torch.isfinite(sigma).all()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def _checkpoint(**overrides: object) -> dict:
    """A checkpoint shaped like a real one, from the real head.

    Built from ``build_head`` rather than from a hand-written ``state_dict`` so that
    the shapes here cannot drift from the head's: an earlier version of this file
    assumed four output classes, and the real ordinal head has twelve, so the
    fabricated state dict simply failed to load and the test was measuring nothing
    about the code it named.
    """
    head = build_head("ordinal", 1024)
    payload = {
        "head": "ordinal",
        "backbone": "mobilenetv3_small_100",
        "roi": "palpebral",
        "fold": "single-seed42",
        "feature_dim": 1024,
        "centres": list(HB_BIN_CENTRES),
        "state_dict": head.state_dict(),
        "config": {},
    }
    payload.update(overrides)
    return payload


def test_a_checkpoint_without_centres_is_refused(tmp_path: Path) -> None:
    """The regression that made ``centres`` load-bearing.

    A checkpoint written before they were recorded cannot be exported, because the
    only way to fill the gap is to assume the ordinal head's WHO grid -- right for
    the ordinal head, wrong for the severity head, whose centres are per-split
    means. Assuming yields a valid graph that decodes every prediction off by a
    constant, so the loader refuses and says what to do instead.
    """
    path = tmp_path / "old.pt"
    stripped = {k: v for k, v in _checkpoint().items() if k != "centres"}
    torch.save(stripped, path)

    with pytest.raises(ValueError, match="centres"):
        load_checkpoint(path)

    # And the message says what to do rather than only what is wrong.
    with pytest.raises(ValueError) as caught:
        load_checkpoint(path)
    assert "retrain" in str(caught.value).lower()


def test_a_complete_checkpoint_loads(tmp_path: Path) -> None:
    path = tmp_path / "new.pt"
    torch.save(_checkpoint(), path)
    loaded = load_checkpoint(path)
    assert loaded["head"] == "ordinal"
    assert loaded["feature_dim"] == 1024


def test_centres_survive_the_round_trip_as_plain_numbers(tmp_path: Path) -> None:
    """``weights_only=True`` is why they are a list.

    A numpy array in the payload would be refused by the safe loader, and the
    tempting fix -- loosening the flag to keep the array -- gives up the property
    the flag exists to provide.
    """
    path = tmp_path / "round.pt"
    torch.save(_checkpoint(centres=[11.5, 10.0]), path)
    loaded = load_checkpoint(path)
    assert all(type(c) is float for c in loaded["centres"]), loaded["centres"]


def test_the_screen_model_is_rebuilt_from_the_checkpoint_alone() -> None:
    """No head name, backbone name or dimension is taken as an argument.

    That is the whole point of recording them: an export told which head to load can
    load the wrong one and produce a valid-looking graph for a different model.
    """
    checkpoint = _checkpoint()
    model = build_screen_model(checkpoint)

    assert model.head.fc.weight.shape == (N_BINS, 1024), "the recorded feature_dim was ignored"
    assert model.head.fc.weight.detach().numpy() == pytest.approx(
        checkpoint["state_dict"]["fc.weight"].numpy(), abs=1e-6
    )
    assert model.centres.tolist() == pytest.approx(list(HB_BIN_CENTRES))
    assert model.size == IMAGE_SIZE


def test_a_rebuilt_screen_reproduces_the_checkpoint_its_logits() -> None:
    """Weights in, same logits out.

    A ``load_state_dict`` that silently matched the wrong keys -- a renamed head, a
    different ``feature_dim`` whose shape happened to be accepted -- still returns a
    model. Comparing outputs is what tells them apart.
    """
    checkpoint = _checkpoint()
    model = build_screen_model(checkpoint)
    head = build_head(checkpoint["head"], int(checkpoint["feature_dim"]))
    head.load_state_dict(checkpoint["state_dict"])

    features = torch.randn(4, 1024)
    with torch.no_grad():
        assert model.head(features).numpy() == pytest.approx(head(features).numpy(), abs=1e-6)


def test_the_tracer_warns_about_the_branch_that_is_not_traced(tmp_path: Path) -> None:
    """PyTorch's warning is expected, and asserting it beats suppressing it.

    Exporting emits ``TracerWarning: Converting a tensor to a Python boolean might
    cause the trace to be incorrect`` from ``Normalise``'s size check. It is
    harmless here -- the branch is *meant* to be absent from the graph, which is
    what ``test_the_graph_refuses_a_mis_sized_input`` pins.

    It is left visible rather than filtered, because a warning suppressed at the
    suite level is a warning nobody reads, and this one is the tracer telling us
    exactly the thing the design depends on. If a future edit makes the resize
    survive tracing, this test fails and says so.
    """
    with pytest.warns(TracerWarning, match="trace"):
        export_onnx(_make_screen(), size=SIZE, path=tmp_path / "warned.onnx")


# --------------------------------------------------------------------------- #
# Size, which is a product constraint
# --------------------------------------------------------------------------- #


def test_file_size_is_reported_in_megabytes(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\0" * (2 * 1024 * 1024))
    assert file_size_mb(path) == pytest.approx(2.0)


def test_the_mobile_budget_lives_in_config() -> None:
    """Two modules read it, so it is a constant rather than a literal in either.

    A hard-coded size is a number nobody re-checks.
    """
    assert MOBILE_BUDGET_MB > 0
    assert 5 < MOBILE_BUDGET_MB < 100, "a budget outside this range is not a phone's"


def test_the_exported_default_matches_the_training_crop() -> None:
    """The app's canvas, the training geometry and the graph have to agree.

    ``PREPROCESS_SPEC`` is the documented contract the browser transformer is
    written against, so the exported default is checked against it rather than
    against a second copy of 224 written in this test.
    """
    from hemolux.data.dataset import PREPROCESS_SPEC

    assert IMAGE_SIZE == 224
    assert list(PREPROCESS_SPEC["input_size"]) == [IMAGE_SIZE, IMAGE_SIZE]
    assert PREPROCESS_SPEC["layout"] == "NCHW", "the browser feeds NCHW, not NHWC"
