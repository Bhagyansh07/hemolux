"""Loss functions.

Three issues drive the choices here.

**Regression.** Haemoglobin error is heteroscedastic and heavy-tailed. MSE is
dominated by the rare severe case and the model chases outliers; MAE is
insensitive but has a non-smooth gradient. Huber gives MSE's stable gradients
near zero and MAE's robustness far away, which is exactly the distribution we
have.

**Class imbalance.** The dataset is roughly 90% non-anaemic. Unweighted
cross-entropy makes the majority class the easy majority and the anaemic cases
noise. Inverse-frequency sample weights are the standard correction, and the
focal variant is available for when that is still not enough.

**Ordinal targets.** The ordinal head trains against a *soft* distribution, so
its loss is a soft-target cross-entropy. One-hot cross-entropy would throw away
the ordering, which is the entire reason that head exists.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HuberRegressionLoss(nn.Module):
    """Smooth L1 / Huber loss, reported in g/dL.

    ``delta=1.0`` means errors below 1 g/dL — roughly the clinical acceptability
    band for point-of-care devices — are penalised quadratically, and larger
    errors linearly. The loss therefore concentrates gradient on the clinically
    meaningful range.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return nn.functional.smooth_l1_loss(y_pred, y_true, beta=self.delta)


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy against a soft distribution over ordered bins.

    ``y_true`` is an ``(N, K)`` row-stochastic matrix from
    :func:`hemolux.metrics.calibration.soft_label`. Because it is not one-hot,
    the optimal prediction is the soft target itself, which is exactly the
    property that keeps the output calibrated.
    """

    def forward(self, logits: Tensor, y_soft: Tensor) -> Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(y_soft * log_probs).sum(dim=-1).mean()


class InverseFrequencyWeights:
    """Sample weights inversely proportional to class frequency.

    Computed from the **training split only**. Computing weights from the full
    dataset leaks test-set class frequencies into training, which is a subtle and
    very common leak.

    Normalised to unit mean
    -----------------------
    The raw inverse frequency ``1 / f_c`` does not have mean 1. Weighted by the
    class frequencies it averages to exactly the number of classes:

        sum_c  f_c * (1 / f_c)  =  K

    Measured on this project's class distributions, the unnormalised weights had
    mean 2.0 for a 2-class head, 4.0 for the 4-way severity head and 12.0 for
    the 12-bin ordinal head. That is a silent rescaling of the loss, so the same
    learning rate meant something different in every head, and the C1 comparison
    between heads would have been confounded by the loss scale rather than by the
    head.

    Dividing by that mean makes the weighted mean 1.0, so weighting changes *which*
    samples are emphasised without changing the effective step size.
    """

    def __init__(self, targets: Tensor, *, n_classes: int | None = None, eps: float = 1e-6) -> None:
        """``targets`` are class indices; ``n_classes`` is the full class count.

        ``n_classes`` matters whenever the largest class index is absent from
        ``targets``, which is not hypothetical here. The severity head has four
        bands and the whole corpus holds only a handful of severe cases, so a
        given 131-patient training split can contain none of them. Inferring the
        class count as ``max + 1`` then yields three weights, and
        ``cross_entropy(weight=...)`` rejects it::

            RuntimeError: weight tensor should be defined either for all 4
            classes or no classes but got weight tensor of shape: [3]

        An absent class therefore gets weight 1.0 -- the unweighted contribution
        it would have had anyway -- rather than the ``1/f`` value it would take
        if a single member happened to be present. That asymmetry is deliberate:
        the rare class is rare, and inventing a large weight for a class with no
        members would not correct the imbalance, it would be a large number
        multiplied by zero.
        """
        if targets.numel() == 0:
            raise ValueError("cannot compute class weights from an empty tensor")
        index = targets.long()
        if n_classes is None:
            n_classes = int(index.max().item()) + 1
        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")
        if int(index.max().item()) >= n_classes:
            raise ValueError(
                f"target index {int(index.max().item())} is out of range for {n_classes} classes"
            )

        counts = torch.bincount(index, minlength=n_classes)[:n_classes].float()
        freq = counts / counts.sum().clamp_min(1.0)
        raw = torch.where(counts > 0, 1.0 / (freq + eps), torch.ones_like(freq))
        # Only present classes contribute to the weighted mean, since an absent
        # class has no samples to average over. Dividing by it lands the mean at
        # exactly 1.0 for any class distribution.
        mean_weight = (freq * raw).sum()
        self.weights = raw / mean_weight.clamp_min(eps)

    def to(self, device: torch.device | str) -> Tensor:
        """The per-class weight vector, moved to ``device``.

        Returns the tensor rather than ``self``, because
        ``torch.nn.functional`` wants the tensor and the call sites should not
        have to know that this class is not an ``nn.Module``. A ``.to()``
        returning ``self`` would read like the torch idiom and quietly do
        nothing useful.
        """
        return self.weights.to(device)

    def __call__(self, targets: Tensor) -> Tensor:
        """Per-sample weight for each target class.

        ``targets`` may be float or integer: a float binary label of 0.0 or 1.0
        is an index in all but type, and ``tensor[float_tensor]`` raises rather
        than indexing, so the cast belongs here. Doing it in the callers instead
        means every caller has to remember, and one that does not gets an
        IndexError from deep inside tensor indexing rather than a clear message.
        """
        index = targets.long()
        return self.weights.to(index.device)[index]


class WeightedCrossEntropy(nn.Module):
    """Cross-entropy with per-sample weights."""

    def __init__(self, weights: Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        losses = nn.functional.cross_entropy(logits, targets, reduction="none")
        if self.weights is None:
            return losses.mean()
        return (losses * self.weights[targets]).mean()


class FocalLoss(nn.Module):
    """Focal loss, for class imbalance beyond what weighting fixes.

    ``gamma=2`` is the standard setting. Included because the alternative to
    training it is to not report sensitivity at all, which is not acceptable in a
    screening tool.
    """

    def __init__(self, gamma: float = 2.0, weights: Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weights", weights)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        log_p = log_probs.gather(1, targets[:, None]).squeeze(1)
        p = log_p.exp()
        loss = -((1.0 - p) ** self.gamma) * log_p
        if self.weights is not None:
            loss = loss * self.weights[targets]
        return loss.mean()


def combined_multitask_loss(
    ordinal_logits: Tensor,
    y_soft: Tensor,
    scalar_pred: Tensor,
    hb: Tensor,
    *,
    scalar_weight: float = 0.5,
    huber_delta: float = 1.0,
) -> Tensor:
    """Ordinal cross-entropy plus a Huber term on the scalar head."""
    ordinal = SoftTargetCrossEntropy()(ordinal_logits, y_soft)
    scalar = nn.functional.smooth_l1_loss(scalar_pred, hb, beta=huber_delta)
    return ordinal + scalar_weight * scalar
