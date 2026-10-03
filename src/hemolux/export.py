"""ONNX export, so inference runs in the browser rather than on a server.

Why the graph is the whole image, not the feature vector
-------------------------------------------------------
There is no backend. Inference happens on the user's phone through
``onnxruntime-web``, which means the exported graph has to accept pixels and do
the mask-apply, crop, resize and normalise itself -- otherwise the browser would
have to replicate the preprocessing in JavaScript, and any drift between the two
would silently change every prediction.

So the exported module wraps the backbone *and* the normalisation constants as
constants baked into the graph, and the JS side only has to resize and write a
tensor.

What is deliberately *not* in the graph
--------------------------------------
* **The mask.** Masks are hand-drawn annotations in the training corpus. At
  inference the app segments with a separate step, so the mask is an input to the
  model, not something baked in. Two tensors in, two tensors out.
* **The ROI crop.** The bounding-box crop depends on the mask's extent, which is
  data-dependent. Baking it in would freeze the crop geometry at export time.
  Instead the app crops before calling the graph, and the graph sees a
  224x224x3 tensor, which is the documented ``PREPROCESS_SPEC`` contract.

That split means the JavaScript side owns geometry and the graph owns colour, and
both halves are tested: ``tests/test_export.py`` round-trips a real image through
the ONNX runtime and asserts the output matches PyTorch to within float32 noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hemolux.config import ARTIFACT_MODELS, IMAGE_SIZE, ensure_dirs
from hemolux.data.dataset import PREPROCESS_SPEC

#: ImageNet statistics, read from the one definition the training pipeline uses.
#: Duplicated into the graph as constants because the exported artefact has to
#: keep computing the same thing even if this module is refactored.
_MEAN: tuple[float, ...] = tuple(PREPROCESS_SPEC["mean"])
_STD: tuple[float, ...] = tuple(PREPROCESS_SPEC["std"])


class Normalise(nn.Module):
    """``(x - mean) / std`` with the statistics baked in as buffers.

    Folding normalisation into the graph is what lets the browser hand over raw
    pixels. Exported as a standalone module too, so the pipeline can be tested in
    pieces.
    """

    def __init__(self, size: int = IMAGE_SIZE) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1))
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            # Resizes in Python; does *not* reach the exported graph, and that is
            # deliberate on both counts.
            #
            # In eager use a caller may hand anything and this is the convenient
            # place to cope. In the graph it would be the opposite of convenient:
            # tracing evaluates the comparison once in Python and bakes the answer
            # in, so the op is either absent or unconditional -- and the tracer
            # warns about exactly that. Silently resizing in a browser would absorb
            # the one mismatch that matters, which is the app's canvas not being the
            # training geometry, and turn a loud shape error into a quiet
            # degradation of every prediction.
            #
            # So the exported graph declares exactly `size` and refuses anything
            # else. `export_onnx` checks `size` against the model's own, and
            # `HemoluxScreen.size` is what makes that check fire.
            x = torch.nn.functional.interpolate(
                x, size=(self.size, self.size), mode="bilinear", align_corners=False
            )
        return (x - self.mean) / self.std


def export_onnx(
    model: nn.Module,
    *,
    size: int = IMAGE_SIZE,
    opset: int = 17,
    path: Path | None = None,
    input_name: str = "image",
    output_names: tuple[str, ...] = ("hb_gdl", "sigma_gdl", "bin_probs"),
) -> Path:
    """Export a screening model to ONNX.

    ``model`` must accept a float32 ``(N, 3, size, size)`` tensor in ``[0, 1]``
    and return the haemoglobin estimate, its predictive standard deviation and
    the ordinal bin probabilities. ``HemoluxScreen`` below is that model.

    The graph's input contract is exactly ``(N, 3, size, size)``: only the batch
    axis is dynamic. ``Normalise`` resizes in Python when handed something else,
    but that branch cannot survive tracing, so rather than leave a claim in the
    docstring that the file does not honour, the file refuses a wrong size. A
    browser that resizes its canvas to ``size`` -- as ``docs/`` specifies -- is
    unaffected; one that does not gets an error instead of a quiet degradation.

    ``size`` is checked against the model's own, which is why ``HemoluxScreen``
    exposes ``size``: without it this comparison silently passed.
    """
    inner = getattr(model, "size", None)
    if inner is not None and inner != size:
        raise ValueError(
            f"model expects {inner}x{inner} input but export was asked for {size}x{size}"
        )
    ensure_dirs()
    destination = Path(path) if path else ARTIFACT_MODELS / f"hemolux_screen_{size}.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 3, size, size, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(destination),
            input_names=[input_name],
            output_names=list(output_names),
            opset_version=opset,
            do_constant_folding=True,
            # The batch axis is dynamic. Traced against a single dummy image, ONNX
            # freezes the dimension to 1, and the resulting graph then refuses a
            # batch of anything else with `INVALID_ARGUMENT: index: 0 Got: 2
            # Expected: 1`. That is not a cosmetic limit -- the browser happens to
            # send one image at a time so it would never notice, but verify_onnx
            # does send a pair, so the check that exists to catch a PyTorch/ONNX
            # divergence could not run at all. A verification step that is skipped
            # by construction is the worst of the three outcomes available here.
            dynamic_axes={input_name: {0: "batch"}},
            dynamo=False,
        )
    return destination


def verify_onnx(
    reference: nn.Module,
    path: Path,
    *,
    size: int = IMAGE_SIZE,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, float]:
    """Run the exported graph and PyTorch on the same input; return the max gap.

    A graph that exports without error is not a graph that computes the same
    thing. This is the check that would catch an operator the browser runtime
    implements differently, which is the failure mode an export-only test misses.

    ``reference`` is the PyTorch model. It is evaluated in float32 on the CPU, and
    the tolerance is a relative one because ``onnxruntime`` may fuse or reorder
    float operations. Both directions are checked, hb and sigma, because a decode
    that matches while its uncertainty does not would still mis-set the abstention
    threshold.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.random((2, 3, size, size), dtype=np.float32))

    with torch.no_grad():
        torch_hb, torch_sigma, _ = reference(x)
    onnx_hb, onnx_sigma, _ = session.run(None, {session.get_inputs()[0].name: x.numpy()})

    torch_hb = torch_hb.numpy()
    torch_sigma = torch_sigma.numpy()

    hb_gap = float(np.max(np.abs(onnx_hb - torch_hb)))
    sigma_gap = float(np.max(np.abs(onnx_sigma - torch_sigma)))

    if not np.allclose(onnx_hb, torch_hb, atol=atol, rtol=rtol):
        raise AssertionError(
            f"ONNX haemoglobin disagrees with PyTorch: max |diff| = {hb_gap:.3e} g/dL "
            f"exceeds atol={atol:.1e}, rtol={rtol:.1e}. Torch {torch_hb.ravel()} vs ONNX {onnx_hb.ravel()}"
        )
    if not np.allclose(onnx_sigma, torch_sigma, atol=atol, rtol=rtol):
        raise AssertionError(
            f"ONNX predictive sigma disagrees with PyTorch: max |diff| = {sigma_gap:.3e} g/dL "
            f"exceeds atol={atol:.1e}, rtol={rtol:.1e}"
        )
    return {
        "hb_max_abs_diff": hb_gap,
        "sigma_max_abs_diff": sigma_gap,
        "n_samples": int(x.shape[0]),
    }


def file_size_mb(path: Path) -> float:
    """Model size, which is the number that decides whether it loads on a phone."""
    return path.stat().st_size / (1024 * 1024)


class HemoluxScreen(nn.Module):
    """Backbone + ordinal head + decode, as one exportable module.

    Returns ``(hb, sigma, probs)``. The decode lives in the graph so the browser
    does not reimplement the posterior mean -- the single most likely place for a
    Python/JavaScript mismatch to creep in, and one that would shift every
    prediction by a constant.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, centres: tuple[float, ...]) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.register_buffer("centres", torch.tensor(centres, dtype=torch.float32))

    @property
    def size(self) -> int:
        """The input edge this graph requires.

        Exposed so :func:`export_onnx` can check it against the size it was asked
        for. Without it the check read ``getattr(model, "size", None)``, got
        ``None`` for the one model this repository actually exports, and skipped --
        so the guard in ``export_onnx``'s docstring described a check that never
        ran on the real path.
        """
        return self.backbone.size

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        probs = torch.softmax(logits, dim=-1)
        centres = self.centres[: logits.shape[-1]]
        hb = (probs * centres).sum(dim=-1)
        second = (probs * centres**2).sum(dim=-1)
        sigma = torch.sqrt(torch.clamp(second - hb**2, min=0.0))
        return hb, sigma, probs


class NormaliseBackbone(nn.Module):
    """Normalise, then backbone, so the graph accepts raw pixels in ``[0, 1]``.

    A module rather than a line inside :class:`HemoluxScreen` so that the
    normalisation reaches the graph as a traceable op rather than as weights the
    exporter is free to fold into whatever it likes.
    """

    def __init__(self, backbone: nn.Module, normalise: Normalise) -> None:
        super().__init__()
        self.normalise = normalise
        self.backbone = backbone

    @property
    def size(self) -> int:
        return self.normalise.size

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(self.normalise(x))


def load_checkpoint(path: Path) -> dict:
    """Read a training checkpoint, refusing one too old to rebuild faithfully.

    ``weights_only=True`` because the file is local. That flag is also why the
    centres go into the payload as a plain list rather than as a numpy array: a
    pickled array would be rejected, and loosening the flag to keep it would give
    up the property the flag exists to provide.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    missing = {"head", "backbone", "feature_dim", "state_dict", "centres"} - set(checkpoint)
    if missing:
        raise ValueError(
            f"{path.name} is missing {sorted(missing)}. It predates the checkpoint format "
            "that records the class centres. Retrain rather than export -- "
            "reconstructing them here would mean assuming them, and assuming the "
            "ordinal head's WHO grid for a severity head shifts every prediction."
        )
    return checkpoint


def build_screen_model(checkpoint: dict, *, size: int = IMAGE_SIZE) -> HemoluxScreen:
    """Rebuild backbone + head + decode from a checkpoint's recorded shapes.

    The checkpoint carries the backbone name, the head name, the feature dimension
    and the class centres, so the export takes none of them as arguments. An export
    that had to be told which head to load is an export that can silently load the
    wrong one and produce a valid-looking graph for a different model.

    ``models.backbone`` is imported here rather than at module scope because it
    pulls in ``timm``, and the normalisation and size helpers above are useful to
    callers that only want the geometry and should not pay for a vision import to
    reach it.
    """
    from hemolux.models.backbone import ModelSpec, build_model
    from hemolux.models.heads import build_head

    spec = ModelSpec(backbone=checkpoint["backbone"], head=checkpoint["head"], pretrained=False)
    model = build_model(spec, input_size=size)
    head = build_head(checkpoint["head"], int(checkpoint["feature_dim"]))
    head.load_state_dict(checkpoint["state_dict"])

    screen = HemoluxScreen(
        backbone=NormaliseBackbone(model.backbone, Normalise(size=size)),
        head=head,
        centres=tuple(float(c) for c in checkpoint["centres"]),
    )
    screen.eval()
    return screen
