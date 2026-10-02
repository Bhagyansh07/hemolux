"""Single source of truth for paths, seeds and hyper-parameters.

Every other module imports from here rather than hard-coding a path. That
matters for two practical reasons: the pipeline runs on both Windows (the
builder's machine) and Linux (CI), and a literal ``data/raw`` in five modules is
five chances to break on one of them.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem layout.

    ``data_root`` is overridable via ``HEMOLUX_DATA_ROOT`` so that CI and the
    builder can use the same code against different trees.
    """

    data_root: Path = field(
        default_factory=lambda: Path(os.environ.get("HEMOLUX_DATA_ROOT", _ROOT / "data"))
    )

    @property
    def raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed(self) -> Path:
        return self.data_root / "processed"

    @property
    def synthetic(self) -> Path:
        """Fake data for unit tests. Never read by an experiment."""
        return self.data_root / "synthetic"

    @property
    def labels_csv(self) -> Path:
        return self.processed / "labels.csv"

    @property
    def features_csv(self) -> Path:
        return self.processed / "colour_features.csv"

    @property
    def site_dir(self) -> Path:
        return self.processed / "site"


PATHS = Paths()


def _artifacts() -> Path:
    return _ROOT / "artifacts"


#: Committed evidence. Contains no patient identifiers.
ARTIFACT_REPORTS = _artifacts() / "reports"
ARTIFACT_FIGURES = _artifacts() / "figures"
ARTIFACT_MODELS = _artifacts() / "models"


def ensure_dirs() -> None:
    """Create every output directory. Idempotent."""
    for path in (
        PATHS.processed,
        PATHS.interim,
        PATHS.synthetic,
        ARTIFACT_REPORTS,
        ARTIFACT_FIGURES,
        ARTIFACT_MODELS,
    ):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

#: Master seed. Every experiment records it so a run can be repeated exactly.
SEED = int(os.environ.get("HEMOLUX_SEED", "42"))


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG that can affect a result.

    Called at the top of every experiment. Torch's generator state is included
    because dropping it produces runs that differ only in DataLoader shuffling,
    which is exactly the kind of irreproducibility that makes an n=218 result
    unfalsifiable.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms are left off: some CPU kernels have no
        # deterministic implementation, and raising instead of degrading is
        # worse than a documented tiny non-determinism.
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def configure_torch() -> None:
    """Thread count for CPU training.

    Defaults to 8 threads: on this 16-logical-core machine, using all 16 was
    measurably slower per step due to thread contention, and more than 8 gave no
    further gain.
    """
    threads = int(os.environ.get("HEMOLUX_THREADS", "8"))
    import torch

    torch.set_num_threads(max(1, threads))


# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #

#: WHO haemoglobin cut-offs, g/dL.
WHO_CUTOFF_MALE = 13.0
WHO_CUTOFF_FEMALE = 12.0
WHO_MODERATE = 9.0
WHO_SEVERE = 7.0

#: Physiologically plausible Hb range for an adult. Outside this, a label is
#: treated as a data error rather than a real measurement.
HB_VALID_MIN = 3.0
HB_VALID_MAX = 25.0

#: Target input size. 224 is the native resolution of the EfficientNet and
#: MobileNetV3 families; going larger costs quadratic compute for no measured gain
#: on conjunctival ROIs, which are typically small in the frame.
IMAGE_SIZE = 224

#: Reference points from the published literature, for context in EVALS.md.
#: These are NOT thresholds this project asserts; they are what we compare to.
PUBLISHED_REFERENCE = {
    "erythema_index_2016_r2": 0.27,
    "forniceal_only_2016_r2": 0.09,
    "label_distribution_2022_r2": 0.512,
    "mask_rcnn_2023_r2": 0.503,
    "mask_rcnn_2023_unsegmented_r2": 0.306,
    "hemoconvit_2025_mae": 0.870,
    "hemoconvit_2025_r2": 0.761,
    "smartphone_2021_loa": 4.73,
    "clinical_acceptable_error_gdl": 1.0,
}
