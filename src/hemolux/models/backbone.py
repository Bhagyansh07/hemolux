"""Backbone construction: a timm feature extractor plus one of our heads.

Why a frozen-features-first strategy
-----------------------------------
n = 218 patients, split patient-disjoint into roughly 130 train / 44 val / 44
test. A randomly initialised EfficientNet has about 4 million parameters and
will memorise 130 images inside two epochs while reporting a train R² near 1.0
and a test R² near zero. The honest sequence at this sample size is:

1. **Linear probe** on frozen ImageNet features (``unfreeze_param_fraction=0``).
   About 1,300 free parameters, so the number it reports is a real number. This
   is the primary baseline.
2. **Partial fine-tuning** at a small parameter budget, only if the probe shows
   the signal is present but under-expressed.

Both are reachable through :func:`apply_unfreezing`. The head is always
trainable; only the backbone is ever frozen.

Why the budget is counted in parameters, not in stages
------------------------------------------------------
An earlier version unfroze the last ``f`` *fractions of the stage list*. Measured
on the builder's CPU that produced:

    mobilenetv3_small_100   f=0.25 -> 48.6% of parameters trainable
    efficientnet_b0         f=0.25 -> 68.5% of parameters trainable
    resnet18                f=0.25 ->  0.0% trainable, silently

The third line is the reason this was changed. ResNet has no ``stages`` or
``blocks`` attribute, only ``layer1..layer4``, so the stage list fell back to
``children()`` and "the last 25%" resolved to ``global_pool`` and ``fc``, which
hold zero parameters. The run would have printed a "linear probe" and trained
nothing.

The parameter counts are also wildly uneven across stages -- ``efficientnet_b0``
keeps 3.59M of its 4.0M parameters in ``blocks`` while ``conv_head`` holds
409k, so a stage fraction says very little about how much capacity is being
fitted. ``unfreeze_param_fraction`` therefore counts parameters, takes whole
stages from the end, and returns the fraction it actually achieved
(:class:`FreezeReport`) rather than the one that was requested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hemolux.models.heads import build_head

__all__ = [
    "BROWSER_CANDIDATES",
    "ConjunctivaNet",
    "FreezeReport",
    "ModelSpec",
    "apply_unfreezing",
    "build_model",
    "count_parameters",
    "describe_trainable",
    "discover_stages",
    "probe_feature_dim",
    "resolve_device",
]

#: Backbones small enough to ship to a browser. The ONNX graph is
#: downloaded by the user on a phone connection, so its size is a product
#: constraint, not only an engineering one. Actual exported sizes are printed by
#: ``hemolux export`` and recorded in ``EVALS.md``; they are not
#: hard-coded here, because a hard-coded size is a number nobody re-checks.
BROWSER_CANDIDATES: tuple[str, ...] = (
    "mobilenetv3_small_100",
    "efficientnet_b0",
)


# --------------------------------------------------------------------------- #
# Freeze control
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FreezeReport:
    """What :func:`apply_unfreezing` actually did, as opposed to what was asked."""

    requested_fraction: float
    achieved_fraction: float
    backbone_params: int
    unfrozen_params: int
    stages_total: int
    stages_unfrozen: int

    def __str__(self) -> str:
        return (
            f"unfroze {self.stages_unfrozen}/{self.stages_total} stages = "
            f"{self.unfrozen_params:,}/{self.backbone_params:,} backbone params "
            f"({100 * self.achieved_fraction:.1f}%, requested {100 * self.requested_fraction:.1f}%)"
        )


#: ResNet-style naming: ``layer1`` ... ``layer4``, possibly with gaps.
_LAYER_NAME = re.compile(r"^layer(\d+)$")


def discover_stages(backbone: nn.Module) -> list[nn.Module]:
    """Return the backbone's ordered feature stages.

    Resolution order, first match wins:

    1. a ``blocks`` or ``stages`` child container (timm's MobileNetV3,
       EfficientNet, ConvNeXt and friends),
    2. a contiguous run of ``layer1..layerN`` children (torchvision ResNet and
       VGG), in index order,
    3. all children, as a last resort.

    Order matters for correctness rather than tidiness: option 3 on a ResNet
    yields ``global_pool`` and ``fc`` at the end, and unfreezing "the last
    stages" would then unfreeze zero parameters while appearing to succeed.
    :func:`apply_unfreezing` raises rather than allowing that, so a wrong stage
    list surfaces immediately instead of producing a silently untrained model.
    """
    for attr in ("blocks", "stages"):
        container = getattr(backbone, attr, None)
        if isinstance(container, nn.Module) and len(container) > 0:  # type: ignore[arg-type]
            stages = [s for s in container if isinstance(s, nn.Module)]  # type: ignore[call-overload]
            if stages:
                return stages

    layered = sorted(
        (
            (int(m.group(1)), module)
            for name, module in backbone.named_children()
            if (m := _LAYER_NAME.match(name))
        )
    )
    if layered:
        return [module for _, module in layered]

    return [m for m in backbone.children() if isinstance(m, nn.Module)]


def apply_unfreezing(backbone: nn.Module, param_fraction: float) -> FreezeReport:
    """Freeze the backbone, then thaw whole stages from the end up to a budget.

    Parameters
    ----------
    param_fraction
        Target fraction of the *backbone's* parameters to make trainable, in
        ``[0, 1]``. The realised fraction overshoots, because stages are only
        unfrozen or frozen as whole units; the realised value is returned rather
        than the requested one.

    Raises
    ------
    ValueError
        ``param_fraction`` is outside ``[0, 1]``, or a non-zero request thawed
        nothing at all. The second case means the stage list was wrong, and a
        run that quietly trains nothing is worse than a failed run.
    """
    if not 0.0 <= param_fraction <= 1.0:
        raise ValueError(f"param_fraction must be in [0, 1], got {param_fraction}")

    # Freeze first, unconditionally, so the result never depends on whatever
    # timm happened to leave trainable.
    for param in backbone.parameters():
        param.requires_grad_(False)

    backbone_params = sum(p.numel() for p in backbone.parameters())
    stages = discover_stages(backbone)

    if param_fraction == 0.0:
        return FreezeReport(
            requested_fraction=0.0,
            achieved_fraction=0.0,
            backbone_params=backbone_params,
            unfrozen_params=0,
            stages_total=len(stages),
            stages_unfrozen=0,
        )

    target = param_fraction * backbone_params
    stage_param_ids = {id(p) for s in stages for p in s.parameters()}
    #: Parameters outside every stage: the stem, the head convolution, pooling.
    #: They are the last unit in the walk, so a budget of 1.0 really does thaw
    #: everything rather than everything-except-the-part-timm-hides-outside-
    #: ``blocks``.
    remainder = [p for p in backbone.parameters() if id(p) not in stage_param_ids]

    units: list[nn.Module | list[nn.Parameter]] = [*reversed(stages)]
    if remainder:
        units.append(remainder)

    unfrozen = 0
    n_unfrozen = 0
    for unit in units:
        params = list(unit) if isinstance(unit, list) else list(unit.parameters())
        for param in params:
            param.requires_grad_(True)
        unfrozen += sum(p.numel() for p in params)
        n_unfrozen += 1
        if unfrozen >= target:
            break

    if unfrozen == 0:
        raise ValueError(
            f"asked to unfreeze {param_fraction:.0%} of the backbone but thawing "
            f"{len(stages)} discovered stage(s) released no parameters. The stage "
            f"list for this architecture is wrong; check discover_stages() before "
            f"trusting this run."
        )

    return FreezeReport(
        requested_fraction=param_fraction,
        achieved_fraction=unfrozen / backbone_params if backbone_params else 0.0,
        backbone_params=backbone_params,
        unfrozen_params=unfrozen,
        stages_total=len(stages),
        stages_unfrozen=n_unfrozen,
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """A backbone, a head, and how much of the backbone may be fitted."""

    backbone: str = "efficientnet_b0"
    head: str = "regression"
    pretrained: bool = True
    #: Target fraction of backbone parameters to unfreeze. ``0`` is a linear
    #: probe on frozen features; ``1`` is full fine-tuning. See
    #: :func:`apply_unfreezing` for why this is a parameter budget.
    unfreeze_param_fraction: float = 0.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.unfreeze_param_fraction <= 1.0:
            raise ValueError(
                f"unfreeze_param_fraction must be in [0, 1], got {self.unfreeze_param_fraction}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


def probe_feature_dim(backbone: nn.Module, size: int = 224) -> int:
    """Measure the backbone's output width with a real forward pass.

    ``backbone.num_features`` is metadata, and it is wrong for at least one timm
    family: ``mobilenetv3_small_100`` advertises 576 while its forward pass emits
    1024, so a head built from the attribute dies on the first batch with a
    shape error. Probing costs one no-grad forward on a zero tensor and also
    absorbs a timm upgrade that changes the width.
    """
    with torch.no_grad():
        out = backbone(torch.zeros(1, 3, size, size))
    if out.ndim != 2:
        raise RuntimeError(f"expected a pooled (B, C) feature tensor, got shape {tuple(out.shape)}")
    return int(out.shape[1])


class ConjunctivaNet(nn.Module):
    """The trainable unit: a frozen-or-thawed backbone feeding one head.

    Written as a plain class rather than a frozen dataclass on purpose. A
    dataclass ``__init__`` assigns through ``object.__setattr__``, which bypasses
    ``nn.Module.__setattr__`` and so never registers ``backbone`` or ``head`` in
    ``_modules``. The result is an object that is not a usable ``nn.Module``:
    ``model.parameters()`` raises ``AttributeError: '_backward_hooks'`` on the
    first call, and ``model.to(device)`` silently touches nothing. Both of those
    are the sort of failure that shows up two commits later as a confusing bug.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        #: Measured, not advertised. Recorded in every run's metrics file so a
        #: later reader can tell what the head was actually built for.
        self.feature_dim = feature_dim

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        return self.head(self.backbone(x))


def build_model(
    spec: ModelSpec | None = None, *, input_size: int = 224, **overrides: object
) -> ConjunctivaNet:
    """Build backbone + head.

    The head is created by :func:`hemolux.models.heads.build_head` so the
    training loop and the browser export cannot disagree about the output shape.
    """
    import timm

    spec = spec or ModelSpec(**overrides)  # type: ignore[arg-type]

    backbone = timm.create_model(spec.backbone, pretrained=spec.pretrained, num_classes=0)

    in_features = probe_feature_dim(backbone, input_size)
    advertised = getattr(backbone, "num_features", None)
    if advertised is not None and int(advertised) != in_features:
        print(
            f"[backbone] {spec.backbone} reports num_features={advertised} but emits "
            f"{in_features}; using the measured value."
        )

    report = apply_unfreezing(backbone, spec.unfreeze_param_fraction)
    print(f"[backbone] {spec.backbone} + {spec.head}: {report}")

    return ConjunctivaNet(backbone, build_head(spec.head, in_features), in_features)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Split the parameter count into total, trainable and frozen.

    The trainable fraction goes into every run's metrics file, because at
    n = 218 the number of free parameters is the single best predictor of
    whether a validation R² means anything.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a device, preferring CUDA only when it is actually usable.

    The builder's machine reports an integrated AMD GPU with 0.5 GB of VRAM.
    Some ROCm builds expose that as ``torch.cuda.is_available() == True`` and
    then fail on the first real allocation, so the device is probed by allocating
    a tensor rather than by reading the flag. ``RuntimeError`` covers both
    ``OutOfMemoryError`` (a subclass) and ROCm's initialisation failures.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        try:
            torch.zeros(1024, 1024, device="cuda")
        except RuntimeError:
            return torch.device("cpu")
    return torch.device("cpu")


def describe_trainable(model: nn.Module) -> str:
    """One line naming each stage and whether it is frozen. For run logs."""
    parts: list[str] = []
    for i, stage in enumerate(discover_stages(model)):
        trainable = any(p.requires_grad for p in stage.parameters())
        parts.append(f"{i}:{'open' if trainable else 'shut'}")
    return " ".join(parts)
