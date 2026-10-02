"""Tests for backbone construction and freeze control.

The cases that matter here are the ones written after a real defect, not the
ones written to hit a coverage number.

Real defects these lock in
--------------------------
1. ``mobilenetv3_small_100`` advertises ``num_features == 576`` through timm's
   metadata but emits a 1024-wide feature tensor, so a head built from the
   attribute raised ``RuntimeError: mat1 and mat2 shapes cannot be multiplied``
   on the first batch. :func:`probe_feature_dim` measures instead.
2. Stage *fractions* are not parameter fractions. Measured on the builder's CPU,
   ``unfreeze_fraction=0.25`` thawed 48.6% of mobilenetv3's parameters and
   68.5% of efficientnet_b0's, and on resnet18 -- which has neither ``stages``
   nor ``blocks``, only ``layer1..layer4`` -- it thawed 0.0%, silently, because
   the fallback stage list ended in ``global_pool`` and ``fc``.
   :func:`apply_unfreezing` now counts parameters, and raises instead of
   reporting success when it thaws nothing.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from hemolux.models.backbone import (
    ConjunctivaNet,
    FreezeReport,
    ModelSpec,
    apply_unfreezing,
    build_model,
    count_parameters,
    describe_trainable,
    discover_stages,
    probe_feature_dim,
    resolve_device,
)

HEADS = ("regression", "severity", "ordinal", "binary")
BROWSER_BACKBONES = ("mobilenetv3_small_100", "efficientnet_b0")


# --------------------------------------------------------------------------- #
# Stage discovery
# --------------------------------------------------------------------------- #


class _FakeResNet(nn.Module):
    """ResNet-shaped: ``layer1..layer4`` and no ``stages``/``blocks``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.layer1 = nn.Sequential(nn.Conv2d(8, 16, 3), nn.Conv2d(16, 16, 3))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, 3))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.bn1(self.conv1(x))))))
        return self.fc(self.global_pool(x).flatten(1))


def test_layer_n_children_are_found_as_stages() -> None:
    """The regression that produced a silent no-op: ResNet has no blocks attr."""
    stages = discover_stages(_FakeResNet())
    assert len(stages) == 4, "expected layer1..layer4, fell through to children()"
    # Not the head: fc and global_pool are not feature stages.
    assert all(not isinstance(s, nn.Linear) for s in stages)


def test_blocks_container_wins_over_children() -> None:
    """timm families expose ``blocks``; children() would include conv_head."""

    class _FakeMobilenet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_stem = nn.Conv2d(3, 8, 3)
            self.blocks = nn.Sequential(
                nn.Sequential(nn.Conv2d(8, 16, 3)),
                nn.Sequential(nn.Conv2d(16, 32, 3)),
            )
            self.conv_head = nn.Conv2d(32, 64, 3)

    stages = discover_stages(_FakeMobilenet())
    assert len(stages) == 2


def test_stage_order_is_ascending() -> None:
    """Unfreezing walks stages from the end, so order has to be anatomical."""
    net = _FakeResNet()
    stages = discover_stages(net)
    widths = [s[0].out_channels for s in stages]
    assert widths == sorted(widths), f"stages out of order: {widths}"


# --------------------------------------------------------------------------- #
# Freeze control
# --------------------------------------------------------------------------- #


def test_full_freeze_leaves_nothing_trainable() -> None:
    net = _FakeResNet()
    report = apply_unfreezing(net, 0.0)
    assert report.achieved_fraction == 0.0
    assert all(not p.requires_grad for p in net.parameters())
    assert report.stages_unfrozen == 0
    assert report.backbone_params == sum(p.numel() for p in net.parameters())


def test_achieved_fraction_overshoots_but_never_falls_short() -> None:
    """Stages are whole units, so the realised budget rounds up. It must not round down."""
    net = _FakeResNet()
    report = apply_unfreezing(net, 0.10)
    assert report.achieved_fraction >= 0.10 - 1e-9


def test_larger_budget_never_unfreezes_fewer_parameters() -> None:
    net = _FakeResNet()
    seen = [apply_unfreezing(net, f).unfrozen_params for f in (0.0, 0.05, 0.2, 0.5, 1.0)]
    assert seen == sorted(seen), f"non-monotone unfreezing: {seen}"


def test_full_fraction_unfreezes_everything() -> None:
    net = _FakeResNet()
    report = apply_unfreezing(net, 1.0)
    assert all(p.requires_grad for p in net.parameters())
    assert report.unfrozen_params == sum(p.numel() for p in net.parameters())


def test_raises_when_a_budget_thaws_nothing() -> None:
    """A run that trains nothing must fail loudly, not report a linear probe."""

    class _NoParamStages(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = nn.Sequential(nn.ReLU())
            self.layer2 = nn.Sequential(nn.ReLU())
            self.trivial = nn.Parameter(torch.zeros(0))

    with pytest.raises(ValueError, match="released no parameters"):
        apply_unfreezing(_NoParamStages(), 0.25)


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_out_of_range_budget_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        apply_unfreezing(_FakeResNet(), bad)


def test_freeze_report_is_truthful_about_a_no_op_request() -> None:
    """requested_fraction is echoed; achieved_fraction is measured."""
    net = _FakeResNet()
    report = apply_unfreezing(net, 0.0)
    assert report.requested_fraction == 0.0
    assert report.achieved_fraction == 0.0
    assert isinstance(report, FreezeReport)
    assert "0.0%" in str(report)


# --------------------------------------------------------------------------- #
# Feature-dimension probing
# --------------------------------------------------------------------------- #


def test_probe_matches_the_real_forward_width() -> None:
    """The defect itself: metadata said 576, the tensor was 1024."""
    import timm

    for name in BROWSER_BACKBONES:
        backbone = timm.create_model(name, pretrained=False, num_classes=0)
        with torch.no_grad():
            actual = backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        assert probe_feature_dim(backbone) == actual, name


def test_probe_rejects_a_backbone_that_does_not_pool() -> None:
    class _NotPooled(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    with pytest.raises(RuntimeError, match="pooled"):
        probe_feature_dim(_NotPooled())


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("head", HEADS)
def test_every_head_builds_and_keeps_the_head_trainable(head: str) -> None:
    net = build_model(ModelSpec(backbone="mobilenetv3_small_100", head=head, pretrained=False))
    assert isinstance(net, ConjunctivaNet)
    assert all(p.requires_grad for p in net.head.parameters())


def test_frozen_backbone_still_trains_the_head() -> None:
    """A linear probe must actually optimise something."""
    net = build_model(
        ModelSpec(backbone="mobilenetv3_small_100", pretrained=False, unfreeze_param_fraction=0.0)
    )
    counts = count_parameters(net)
    assert counts["trainable"] < 2_000, counts
    assert counts["frozen"] > 1_000_000, counts


def test_build_model_reports_the_width_mismatch_it_worked_around(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The user must see that the advertised width was ignored."""
    build_model(ModelSpec(backbone="mobilenetv3_small_100", pretrained=False))
    assert "reports num_features=576 but emits 1024" in capsys.readouterr().out


@pytest.mark.parametrize("backbone", BROWSER_BACKBONES)
def test_forward_shape_matches_the_head(backbone: str) -> None:
    net = build_model(ModelSpec(backbone=backbone, pretrained=False))
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 224, 224))
    assert net.feature_dim in (1024, 1280)
    assert out.shape == (2,)  # regression head squeezes


def test_model_spec_rejects_a_nonsense_budget() -> None:
    with pytest.raises(ValueError, match="unfreeze_param_fraction"):
        ModelSpec(unfreeze_param_fraction=1.5)
    with pytest.raises(ValueError, match="dropout"):
        ModelSpec(dropout=1.0)


def test_describe_trainable_marks_each_stage() -> None:
    net = _FakeResNet()
    apply_unfreezing(net, 0.5)
    text = describe_trainable(net)
    assert text.count("open") >= 1
    assert text.count("shut") >= 1


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #


def test_explicit_device_is_honoured() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_resolves_to_something_real() -> None:
    device = resolve_device("auto")
    torch.zeros(8, device=device)  # must not raise
