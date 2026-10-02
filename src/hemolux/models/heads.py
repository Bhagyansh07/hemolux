"""Model heads.

The heads implement claim C1: does a continuous
haemoglobin estimate with an ordinal-aware target beat a binarised anaemia flag?

Four heads share one backbone so the comparison is controlled:

``BinaryHead``
    The baseline everyone builds. One logit, thresholded at the WHO cut-off.
``SeverityHead``
    Four-way severity. Uses more supervision than binary, still throws away the
    ordering.
``RegressionHead``
    Direct scalar regression under Huber loss.
``OrdinalHead``
    A distribution over ordered Hb bins trained against a **soft** target.
    This is the project's proposal, so it gets the most explanation.

Why soft targets
----------------
Haemoglobin is ordered. A patient at 10.2 g/dL is far more informative about
10.0 than about 14.0, but one-hot cross-entropy treats a bin error of 10->14 as
exactly as wrong as 10->11. A Gaussian soft target centred on the true value
rewards getting the neighbourhood right, and the loss is still a proper
cross-entropy so calibration is preserved.

Decoding then takes the expectation over bin centres, giving a continuous output
that cannot fall outside the bin range — a real deployment advantage over
classification followed by thresholding.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hemolux.metrics.calibration import HB_BIN_CENTRES, N_BINS, expected_hb, soft_label


class BinaryHead(nn.Module):
    """Single logit. The binarised baseline."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc(features).squeeze(-1)


class SeverityHead(nn.Module):
    """Four-way WHO severity classification."""

    N_CLASSES = 4

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, self.N_CLASSES)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc(features)


class RegressionHead(nn.Module):
    """Direct haemoglobin estimate in g/dL."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc(features).squeeze(-1)


class OrdinalHead(nn.Module):
    """Distribution over ordered Hb bins.

    Outputs unnormalised logits; the loss applies log-softmax. Keeping softmax
    out of the module means temperature scaling stays possible downstream, which
    is only defined on logits.
    """

    def __init__(self, in_features: int, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.fc = nn.Linear(in_features, n_bins)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc(features)

    @staticmethod
    def decode(logits: Tensor) -> Tensor:
        """Posterior mean haemoglobin in g/dL."""
        probs = torch.softmax(logits, dim=-1)
        centres = torch.tensor(HB_BIN_CENTRES[: logits.shape[-1]], dtype=probs.dtype)
        return (probs * centres).sum(dim=-1)

    @staticmethod
    def uncertainty(logits: Tensor) -> Tensor:
        """Predictive standard deviation in g/dL, from the bin distribution."""
        probs = torch.softmax(logits, dim=-1)
        centres = torch.tensor(HB_BIN_CENTRES[: logits.shape[-1]], dtype=probs.dtype)
        mean = (probs * centres).sum(dim=-1)
        second = (probs * centres**2).sum(dim=-1)
        return torch.sqrt(torch.clamp(second - mean**2, min=0.0))


class MultiTaskHead(nn.Module):
    """Shared trunk with an ordinal classifier and a scalar regressor.

    Included for C1's completeness: the ordinal distribution is a discretisation
    of a continuous target, so the scalar head can regularise it. Whether that
    helps is an empirical question, and on n = 218 the answer may well be no —
    which is a reportable result.
    """

    def __init__(self, in_features: int, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.ordinal = nn.Linear(128, n_bins)
        self.scalar = nn.Linear(128, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(features)
        return self.ordinal(h), self.scalar(h).squeeze(-1)


HEADS: dict[str, type[nn.Module]] = {
    "binary": BinaryHead,
    "severity": SeverityHead,
    "regression": RegressionHead,
    "ordinal": OrdinalHead,
    "multitask": MultiTaskHead,
}


def build_head(name: str, in_features: int) -> nn.Module:
    """Factory by name. Raises on an unknown head rather than defaulting."""
    if name not in HEADS:
        raise KeyError(f"unknown head {name!r}; available: {sorted(HEADS)}")
    return HEADS[name](in_features)


def soft_targets_from_hb(hb: Tensor, sigma: float = 0.6) -> Tensor:
    """Torch-side wrapper over :func:`hemolux.metrics.calibration.soft_label`.

    Kept as a thin shim so training code never has to round-trip through numpy,
    and so the bin edges have exactly one definition.
    """

    arr = soft_label(hb.detach().cpu().numpy(), sigma=sigma)
    return torch.tensor(arr, dtype=torch.float32, device=hb.device)


def decode_numpy(logits: object) -> object:
    """NumPy decode helper for evaluation code."""
    import numpy as np

    shifted = np.asarray(logits, dtype=np.float64) - np.asarray(logits, dtype=np.float64).max(
        axis=1, keepdims=True
    )
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    return expected_hb(probs), probs
