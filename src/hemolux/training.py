"""Training and evaluation: feature extraction, the head comparison, and reporting.

What this module is for
-----------------------
One experiment defines what this project exists to run:

    **C1** Does a continuous haemoglobin estimate beat a binarised anaemia flag?

Everything here exists to make that question answerable rather than arguable, so
the design decisions below are all in service of the answer being *a number*.

Why features are extracted once and cached
------------------------------------------
At n = 217 patients the backbone forward pass is cheap and the head training is
the part that varies. The dataset was captured with a single rig -- a Samsung S6
behind a magnifier under a white LED -- so illumination is already controlled, and
augmentation belongs in the head-comparison arms rather than in the feature cache.
Extracting one 1280-dimensional vector per patient per ROI, then training five
heads on it, turns an experiment that would take hours into one that takes a
minute, and makes the five heads differ *only* in their head.

This is the linear probe described in ``models/backbone.py``. It is the primary
baseline, not a warm-up: with 217 patients a 4-million-parameter backbone reports a
train R^2 near 1.0 and a test R^2 near zero.

Why every head is decoded to g/dL
--------------------------------
MAE and R^2 are only comparable if every head emits haemoglobin in the same unit.
So each head gets a stated decode rule, and the rules are written down here rather
than left implicit:

``binary`` (1 bin per class, then a posterior mean)
    ``p = sigmoid(logit)`` is ``P(anaemic)``. The estimate is
    ``p * mu_anaemic + (1 - p) * mu_normal`` with ``mu`` the **training-set** class
    means. Using training means matters: using test means would leak. This is a
    two-bin posterior mean, so it is the same kind of object as the ordinal
    decode, just cruder.
``severity`` (4 bins)
    Same construction over the four WHO bands.
``regression``
    Direct scalar output. No decoding.
``ordinal`` (12 bins, soft targets)
    Posterior mean over the bin centres, which is the project's proposal.
``multitask``
    Reports both arms, because its ordinal arm is discretised Hb and its scalar
    arm is not, and quoting only one would hide which one did the work.

A head that cannot produce a continuous estimate would be excluded from the MAE
comparison, which would defeat the point of the experiment.

Why the decode is a posterior mean and not an argmax
----------------------------------------------------
An argmax decode can only ever emit one of 12 values. On a continuous clinical
target that is a 1.5 g/dL quantisation, which is larger than the 1.0 g/dL
acceptability target this project is judged against. Taking the expectation
removes the quantisation entirely and makes the output continuous and bounded by
the bin range.

One thing this module will not do
---------------------------------
It will not report a number that was fitted on the data it is scored against.
Temperature scaling is fitted on validation patients only, enforced by the
:class:`~hemolux.metrics.calibration.ValidationOnly` token rather than by
convention, because the test set is touched exactly once per fold.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from hemolux.config import (
    ARTIFACT_MODELS,
    ARTIFACT_REPORTS,
    IMAGE_SIZE,
    SEED,
    WHO_CUTOFF_MALE,
    configure_torch,
    ensure_dirs,
)
from hemolux.data.dataset import ConjunctivaDataset, PatientRecord
from hemolux.data.splits import Fold, severity_bin
from hemolux.imagemeta import silence_corrupt_iccp
from hemolux.losses import (
    HuberRegressionLoss,
    InverseFrequencyWeights,
    SoftTargetCrossEntropy,
    combined_multitask_loss,
)
from hemolux.metrics.calibration import (
    HB_BIN_CENTRES,
    ValidationOnly,
    expected_hb,
    ordinal_target_index,
    risk_coverage_curve,
    soft_label,
    temperature_scale,
)
from hemolux.metrics.fairness import evaluate_subgroups, subgroup_spread
from hemolux.metrics.regression import RegressionReport, regression_report
from hemolux.models.backbone import ModelSpec, build_model, resolve_device
from hemolux.models.heads import build_head

_FloatArr = NDArray[np.float64]

#: Every head in experiment C1, in the order the report lists them.
HEAD_NAMES: tuple[str, ...] = ("binary", "severity", "regression", "ordinal", "multitask")

#: Which head's output each C1 arm is scored through.
#:
#: Four arms are their own decode. ``multitask`` is not: it returns two tensors,
#: and :func:`_forward_all` already takes its ordinal arm, so its haemoglobin
#: estimate and its uncertainty are the ordinal ones. Saying that once here beats
#: repeating ``if head == "multitask": head = "ordinal"`` at four call sites,
#: where one missed site would report a coverage curve for an arm that does not
#: exist.
DECODE_HEAD: dict[str, str] = {
    "binary": "binary",
    "severity": "severity",
    "regression": "regression",
    "ordinal": "ordinal",
    "multitask": "ordinal",
}

#: Published numbers this project is measured against. From
#: :data:`hemolux.config.PUBLISHED_REFERENCE`; repeated in the report so a reader
#: never has to open the source to see what "as good as published work" means.
REFERENCE_NOTE = (
    "Erythema index 2016 R2=0.27 (forniceal only 0.09); smartphone HHR 2021 "
    "R2=0.372; MobileNetV3+SE 2022 R2=0.512 / MAE 1.521; Mask R-CNN 2023 "
    "R2=0.503 vs 0.306 unsegmented; Hemo-ConViT 2025 MAE 0.870, R2 0.761. "
    "Clinical acceptability target: +/-1.0 g/dL."
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainConfig:
    """Every knob the experiments turn.

    Defaults are the ones the report is generated with, so ``hemolux train`` with
    no arguments reproduces the published numbers exactly. Anything else must be
    passed explicitly and lands in the results file.
    """

    backbone: str = "mobilenetv3_small_100"
    roi: str = "palpebral"
    size: int = IMAGE_SIZE
    epochs: int = 40
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    #: 0.0 is a pure linear probe. See ``models/backbone.py`` for why the budget
    #: is counted in parameters rather than in stages.
    unfreeze_param_fraction: float = 0.0
    dropout: float = 0.0
    huber_delta: float = 1.0
    #: Width of the Gaussian soft target on the ordinal head, in g/dL.
    soft_sigma: float = 0.6
    seed: int = SEED
    device: str = "auto"
    num_workers: int = 0
    #: Held-out fraction for the single-split run. 0.2 of 217 is 43 patients,
    #: which is enough to report a MAE with a bootstrap CI and not enough to
    #: support a claim at the severe end of the range.
    val_fraction: float = 0.2

    def with_(self, **overrides: object) -> TrainConfig:
        """Return a copy with overrides applied. Raises on an unknown field."""
        import dataclasses

        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise TypeError(f"unknown TrainConfig field(s): {sorted(unknown)}")
        return dataclasses.replace(self, **overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureSet:
    """Frozen backbone activations for every patient, one row each."""

    patient_ids: tuple[str, ...]
    features: _FloatArr
    hb: _FloatArr
    site: tuple[str, ...]
    sex: tuple[str, ...]
    age: _FloatArr
    roi: str
    backbone: str
    feature_dim: int
    #: WHO severity band per patient, as the string from
    #: :func:`hemolux.data.splits.severity_bin`. Cached here because every reporting
    #: path wants it and it depends on sex, which is not in the haemoglobin array.
    severity: tuple[str, ...] = ()

    def band_index(self) -> list[int]:
        """Severity as integer class indices, in :data:`SEVERITY_CLASSES` order."""
        if not self.severity:
            raise ValueError("this FeatureSet carries no severity labels")
        return [SEVERITY_CLASSES.index(s) for s in self.severity]

    def select(self, patient_ids: Sequence[str]) -> FeatureSet:
        """Row-subset by patient id. Raises rather than silently mismatching."""
        index = {pid: i for i, pid in enumerate(self.patient_ids)}
        missing = [p for p in patient_ids if p not in index]
        if missing:
            raise KeyError(
                f"{len(missing)} patient(s) absent from the feature cache: {missing[:5]}"
            )
        rows = np.array([index[p] for p in patient_ids], dtype=np.int64)
        return FeatureSet(
            patient_ids=tuple(patient_ids),
            features=self.features[rows],
            hb=self.hb[rows],
            site=tuple(self.site[i] for i in rows),
            sex=tuple(self.sex[i] for i in rows),
            age=self.age[rows],
            roi=self.roi,
            backbone=self.backbone,
            feature_dim=self.feature_dim,
            severity=tuple(self.severity[i] for i in rows) if self.severity else (),
        )


def _records_to_arrays(records: Sequence[PatientRecord]) -> tuple[tuple[str, ...], _FloatArr, ...]:
    ids = tuple(r.patient_id for r in records)
    hb = np.array([r.hb for r in records], dtype=np.float64)
    ages = np.array([r.age if r.age is not None else np.nan for r in records], dtype=np.float64)
    return ids, hb, ages


@torch.no_grad()
def extract_features(
    records: Sequence[PatientRecord],
    *,
    backbone: str = "mobilenetv3_small_100",
    roi: str = "palpebral",
    size: int = IMAGE_SIZE,
    batch_size: int = 16,
    device: str = "auto",
    num_workers: int = 0,
    seed: int = SEED,
    verbose: bool = True,
) -> FeatureSet:
    """One forward pass per patient through a frozen ImageNet backbone.

    Augmentation is deliberately off (``train=False``): these are the features a
    screening run would see, and the head comparison is where augmentation
    belongs. Turning it on would make the cache un-reusable across configs.

    The mask is applied and the frame cropped to the mask bounding box inside
    :class:`~hemolux.data.dataset.ConjunctivaDataset`, not here, so the same crop
    logic is shared with training and deployment.
    """
    if not records:
        raise ValueError("cannot extract features from an empty record list")

    torch_device = resolve_device(device)
    silence_corrupt_iccp()

    spec = ModelSpec(
        backbone=backbone, head="regression", pretrained=True, unfreeze_param_fraction=0.0
    )
    model = build_model(spec, input_size=size).to(torch_device)
    model.eval()

    dataset = ConjunctivaDataset(records, roi=roi, train=False, size=size, use_mask=True, crop=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
    )

    chunks: list[_FloatArr] = []
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        # The dataset yields (pixels, hb, patient_id). Only the pixels matter here;
        # the labels come from the FeatureSet, keyed by patient id, so that the
        # cache cannot drift from the labels it was built with.
        images = batch[0]
        logits = model.backbone(images.to(torch_device))
        chunks.append(logits.detach().cpu().numpy().astype(np.float64))
        if verbose and (step % 10 == 0 or step == len(loader)):
            done = min(step * batch_size, len(dataset))
            rate = done / max(time.perf_counter() - started, 1e-6)
            print(f"    features {done}/{len(dataset)}  ({rate:.1f} img/s)")

    ids, hb, ages = _records_to_arrays(records)
    sexes = tuple(r.sex for r in records)
    return FeatureSet(
        patient_ids=ids,
        features=np.concatenate(chunks, axis=0),
        hb=hb,
        site=tuple(r.site for r in records),
        sex=sexes,
        age=ages,
        roi=roi,
        backbone=backbone,
        feature_dim=int(chunks[0].shape[1]),
        severity=tuple(severity_bin(h, s) for h, s in zip(hb, sexes, strict=True)),
    )


def feature_cache_path(features: FeatureSet) -> Path:
    """Cache location, keyed by everything that changes the numbers."""
    return (
        ARTIFACT_MODELS / f"features_{features.backbone}_{features.roi}_{features.feature_dim}d.npz"
    )


def save_feature_cache(features: FeatureSet) -> Path:
    ensure_dirs()
    path = feature_cache_path(features)
    np.savez_compressed(
        path,
        patient_ids=np.array(features.patient_ids, dtype=object),
        features=features.features,
        hb=features.hb,
        site=np.array(features.site, dtype=object),
        sex=np.array(features.sex, dtype=object),
        age=features.age,
        severity=np.array(features.severity, dtype=object),
        meta=np.array([features.roi, features.backbone, str(features.feature_dim)], dtype=object),
    )
    return path


def load_feature_cache(path: Path) -> FeatureSet:
    """Read a cache written by :func:`save_feature_cache`."""
    data = np.load(path, allow_pickle=True)
    roi, backbone, dim = (str(x) for x in data["meta"])
    return FeatureSet(
        patient_ids=tuple(str(x) for x in data["patient_ids"]),
        features=data["features"].astype(np.float64),
        hb=data["hb"].astype(np.float64),
        site=tuple(str(x) for x in data["site"]),
        sex=tuple(str(x) for x in data["sex"]),
        age=data["age"].astype(np.float64),
        roi=roi,
        backbone=backbone,
        feature_dim=int(dim),
        severity=tuple(str(x) for x in data["severity"]) if "severity" in data else (),
    )


# --------------------------------------------------------------------------- #
# Decoding every head to g/dL
# --------------------------------------------------------------------------- #


def _posterior_mean(probs: _FloatArr, centres: _FloatArr) -> _FloatArr:
    """``sum_k p_k * mu_k`` with rows renormalised defensively."""
    p = np.asarray(probs, dtype=np.float64)
    totals = p.sum(axis=1, keepdims=True)
    safe = np.where(totals > 1e-12, totals, 1.0)
    return (p / safe) @ centres


def _sigmoid(x: _FloatArr) -> _FloatArr:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _softmax(x: _FloatArr) -> _FloatArr:
    shifted = x - x.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def training_centres(head: str, train: FeatureSet) -> _FloatArr:
    """The haemoglobin value each class stands for, measured on the training split.

    A posterior-mean decode needs a value per class, and those values have to come
    from somewhere. For the ordinal head they are fixed a priori by the WHO bin
    edges, which is the whole point of that formulation. For the binary and
    severity heads there is no such fixed grid, so the centres are the **training
    patients'** mean Hb within each class.

    Reading them off the training split and no other is the whole point. Deriving
    them from the split being scored would be leakage, and it is a mistake this
    function previously made: the severity decode passed the *test* predictions'
    own argmax as if it were a set of class labels, which both leaked and
    disagreed in length with the array it indexed. Taking the centres as an
    argument, computed once by this function, makes that failure unrepresentable.

    A class with no training member cannot be assigned a mean, so it takes the
    training-split mean -- never the corpus mean, which would leak the test
    patients through the back door.
    """
    head = DECODE_HEAD.get(head, head)

    if head == "ordinal":
        return np.asarray(HB_BIN_CENTRES, dtype=np.float64)

    if head == "regression":
        # Already in g/dL, so there is nothing to map onto.
        return np.empty(0, dtype=np.float64)

    hb = np.asarray(train.hb, dtype=np.float64)
    fallback = float(hb.mean()) if hb.size else 0.0

    if head == "binary":
        # Index 0 is at or above the male cutoff, index 1 below it, which is the
        # order the sigmoid in `decode_predictions` assumes.
        labels = (hb < WHO_CUTOFF_MALE).astype(np.int64)
        k = 2
    elif head == "severity":
        labels = np.asarray(train.band_index(), dtype=np.int64)
        k = len(SEVERITY_CLASSES)
    else:
        raise KeyError(f"unknown head {head!r}")

    out = np.full(k, fallback, dtype=np.float64)
    for index in range(k):
        members = hb[labels == index]
        if members.size:
            out[index] = float(members.mean())
    return out


def decode_predictions(
    head: str,
    outputs: _FloatArr,
    centres: _FloatArr,
) -> _FloatArr:
    """Turn a head's raw output into one haemoglobin estimate per patient.

    ``centres`` comes from :func:`training_centres` on the **training** split, and
    is ignored by ``regression``, whose output is already in g/dL. The per-head
    rules are written out in this module's docstring rather than left to the
    reader to infer from the code.
    """
    if head == "regression":
        return np.asarray(outputs, dtype=np.float64).reshape(-1)

    head = DECODE_HEAD.get(head, head)

    raw = np.asarray(outputs, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)

    if head == "binary":
        # A sigmoid is a two-bin distribution: P(anaemic) = p, P(normal) = 1 - p,
        # in the order `training_centres` built.
        p_anaemic = _sigmoid(raw.reshape(-1))
        probs = np.column_stack([1.0 - p_anaemic, p_anaemic])
    elif head in ("severity", "ordinal"):
        if raw.ndim != 2:
            raise ValueError(f"{head!r} head output must be 2-D, got shape {raw.shape}")
        probs = _softmax(raw)
    else:
        raise KeyError(f"unknown head {head!r}")

    if probs.shape[1] != centres.size:
        raise ValueError(
            f"{head!r} head produced {probs.shape[1]} class probabilities but "
            f"{centres.size} centres were supplied"
        )
    return _posterior_mean(probs, centres)


# --------------------------------------------------------------------------- #
# Head training
# --------------------------------------------------------------------------- #


@dataclass
class HeadFit:
    """Outcome of training one head on one fold."""

    head: str
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_seconds: float
    test_hb_true: _FloatArr
    test_hb_pred: _FloatArr
    test_outputs: _FloatArr
    val_hb_true: _FloatArr
    val_hb_pred: _FloatArr
    val_outputs: _FloatArr
    report: RegressionReport
    temperature: float | None = None
    notes: list[str] = field(default_factory=list)
    #: Serialised weights of the selected epoch, so a result can be reproduced and
    #: an artefact exported without retraining. ``None`` when checkpointing is off.
    weights: dict[str, Tensor] | None = None
    checkpoint_path: Path | None = None


#: WHO severity bands in class order, so band *k* is ``SEVERITY_CLASSES[k]``.
#: Matches the branch order of :func:`hemolux.data.splits.severity_bin`.
SEVERITY_CLASSES: tuple[str, ...] = ("normal", "mild", "moderate", "severe")


def severity_labels(hb: Tensor | NDArray[np.generic], sex: Sequence[str]) -> Tensor:
    """WHO severity band index per patient, as a tensor.

    Reuses :func:`hemolux.data.splits.severity_bin` so the stratification key, the
    reported subgroup label and the training target cannot disagree. Defining the
    ordering a second time here would be exactly the kind of drift that makes a
    four-class head quietly mislabelled.
    """
    values = hb.detach().cpu().numpy() if isinstance(hb, Tensor) else np.asarray(hb)
    if len(values) != len(sex):
        raise ValueError(f"{len(values)} haemoglobin values but {len(sex)} sexes")
    index = [
        SEVERITY_CLASSES.index(severity_bin(float(h), str(s)))
        for h, s in zip(values, sex, strict=True)
    ]
    return torch.tensor(index, dtype=torch.long)


def _loss_for(
    head: str,
    outputs: Tensor,
    y_hb: Tensor,
    cfg: TrainConfig,
    severity: Tensor | None = None,
    severity_weights: Tensor | None = None,
) -> Tensor:
    """The loss each head is trained under.

    Kept in one place so the five arms differ only here and in the decode, which
    is what makes C1 a controlled comparison.

    ``severity`` is required by the severity head and ignored by the others. It is
    passed in rather than recomputed because the band depends on the patient's sex,
    which lives on the :class:`FeatureSet` and not on the haemoglobin tensor.
    ``severity_weights`` is likewise computed once from the training split, because
    class frequencies over 131 patients are very different from class frequencies
    over one batch of 16.
    """
    if head == "regression":
        return HuberRegressionLoss(delta=cfg.huber_delta)(outputs, y_hb)

    if head == "ordinal":
        target = torch.tensor(
            soft_label(y_hb.detach().cpu().numpy(), sigma=cfg.soft_sigma), device=outputs.device
        )
        return SoftTargetCrossEntropy()(outputs, target)

    if head == "binary":
        # Class 1 is below the male WHO threshold. The sex-specific cutoff is
        # applied at decode time, not here, so the head learns one threshold.
        labels = (y_hb < 13.0).long()
        # BCE wants one weight per sample, indexed by that sample's own class.
        # Passing the per-class vector instead is a shape error.
        return nn.functional.binary_cross_entropy_with_logits(
            outputs.reshape(-1),
            labels.float(),
            weight=InverseFrequencyWeights(labels, n_classes=2)(labels),
        )

    if head == "severity":
        if severity is None:
            raise ValueError(
                "the severity head needs severity band indices, which depend on patient sex"
            )
        if severity_weights is None:
            raise ValueError(
                "the severity head needs class weights computed from the training split"
            )
        # One weight per CLASS (shape [K]), which is what cross_entropy wants.
        # These come from the whole training split, never from the batch: a batch
        # of 16 drawn from a 131-patient split frequently lacks the severe band
        # entirely, so per-batch weights would make the loss scale and the class
        # emphasis jitter with every shuffle.
        return nn.functional.cross_entropy(outputs, severity, weight=severity_weights)

    if head == "multitask":
        # Unpack first: this head returns a two-tuple, so `outputs.device` does not
        # exist until the ordinal arm is named.
        ordinal_logits, scalar = outputs
        target = torch.tensor(
            soft_label(y_hb.detach().cpu().numpy(), sigma=cfg.soft_sigma),
            device=ordinal_logits.device,
        )
        return combined_multitask_loss(
            ordinal_logits, target, scalar, y_hb, huber_delta=cfg.huber_delta
        )

    raise KeyError(f"unknown head {head!r}")


def _forward_all(model: nn.Module, features: _FloatArr, batch_size: int = 256) -> _FloatArr:
    """Run a trained head over a feature matrix. Deterministic, no grad."""
    was_training = model.training
    model.eval()
    chunks: list[_FloatArr] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.tensor(features[start : start + batch_size], dtype=torch.float32)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]  # the multitask ordinal arm is the comparable one
            chunks.append(np.asarray(out.detach().cpu().numpy(), dtype=np.float64))
    model.train(was_training)
    return np.concatenate(chunks, axis=0)


def evaluate_fold(
    head: str,
    fold: Fold,
    features: FeatureSet,
    *,
    cfg: TrainConfig,
    verbose: bool = True,
    save_checkpoint: bool = True,
) -> HeadFit:
    """Fit on train, select on val, then score the test fold exactly once.

    The returned :class:`HeadFit` carries the test predictions so the caller can
    build the subgroup and calibration analyses from the same numbers that were
    scored, rather than re-running inference and risking a mismatch.

    The weights of the selected epoch are written to
    :data:`~hemolux.config.ARTIFACT_MODELS` so a published number and a deployable
    artefact come from one training run. The filename carries the head, backbone,
    ROI and fold, so the site-holdout models never overwrite the main split's.
    """
    configure_torch()
    torch_device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    train = features.select(fold.train)
    val = features.select(fold.val)
    test = features.select(fold.test)

    model = build_head(head, train.feature_dim).to(torch_device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_train = torch.tensor(train.features, dtype=torch.float32, device=torch_device)
    y_train = torch.tensor(train.hb, dtype=torch.float32, device=torch_device)
    x_val = torch.tensor(val.features, dtype=torch.float32, device=torch_device)
    y_val = torch.tensor(val.hb, dtype=torch.float32, device=torch_device)
    sev_train = torch.tensor(train.band_index(), dtype=torch.long, device=torch_device)
    sev_val = torch.tensor(val.band_index(), dtype=torch.long, device=torch_device)

    # One class-weight vector per run, from the training split only. Computed here
    # rather than per batch: a batch of 16 out of 131 patients often contains no
    # severe case at all, so batch-local frequencies would make the loss scale and
    # the class emphasis depend on the shuffle.
    #
    # `n_classes=4` is required, not inferred. This corpus holds so few severe
    # cases that a 131-patient split can contain none, and `max + 1` would then
    # produce a 3-long weight vector that `cross_entropy` rejects outright.
    sev_w = InverseFrequencyWeights(sev_train, n_classes=len(SEVERITY_CLASSES)).to(torch_device)

    n_train = x_train.shape[0]
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    started = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator)
        running = 0.0
        for start in range(0, n_train, cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = _loss_for(head, model(x_train[rows]), y_train[rows], cfg, sev_train[rows], sev_w)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(rows)

        model.eval()
        with torch.no_grad():
            val_loss = float(_loss_for(head, model(x_val), y_val, cfg, sev_val, sev_w))
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == cfg.epochs):
            print(
                f"    {head:<10} epoch {epoch:>3}/{cfg.epochs}  train {running / max(n_train, 1):.4f}  val {val_loss:.4f}"
                + ("   *" if best_epoch == epoch else "")
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started

    # Decoded against centres measured on the TRAINING patients only.
    centres = training_centres(head, train)
    test_out = _forward_all(model, test.features)
    val_out = _forward_all(model, val.features)
    test_pred = decode_predictions(head, test_out, centres)

    checkpoint: Path | None = None
    if save_checkpoint:
        ensure_dirs()
        checkpoint = ARTIFACT_MODELS / f"{head}_{features.backbone}_{features.roi}_{fold.name}.pt"
        torch.save(
            {
                "head": head,
                "backbone": features.backbone,
                "roi": features.roi,
                "fold": fold.name,
                "feature_dim": features.feature_dim,
                # The values each output class stands for, measured on this fold's
                # training patients. Recorded rather than left implicit because the
                # ordinal head's centres are the fixed WHO grid while the binary and
                # severity heads' are per-split means: an exporter that assumed the
                # constant grid for all three would ship a valid-looking graph that
                # decodes every prediction off by a constant. For the ordinal head
                # the two agree by construction, which is what makes that assumption
                # invisible until someone exports a severity head.
                "centres": centres.tolist(),
                "state_dict": model.state_dict(),
                "config": asdict(cfg),
            },
            checkpoint,
        )

    notes: list[str] = []
    temperature: float | None = None
    if checkpoint is not None:
        notes.append(f"checkpoint written to {checkpoint.name}")
    if head == "ordinal":
        # Fitted on validation patients only. The token is a type, not a
        # convention, so passing test logits here would be a type error.
        try:
            scaled = temperature_scale(
                val_out,
                ordinal_target_index(val.hb),
                ValidationOnly(len(val.hb)),
            )
            temperature = scaled.temperature
            notes.append(
                f"temperature {scaled.temperature:.3f} fitted on {scaled.ece_before:.4f} -> "
                f"{scaled.ece_after:.4f} ECE (validation only)"
            )
        except ValueError as exc:
            notes.append(f"temperature scaling skipped: {exc}")

    return HeadFit(
        head=head,
        epochs_run=cfg.epochs,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        train_seconds=elapsed,
        test_hb_true=test.hb,
        test_hb_pred=test_pred,
        test_outputs=test_out,
        val_hb_true=val.hb,
        val_hb_pred=decode_predictions(head, val_out, centres),
        val_outputs=val_out,
        report=regression_report(test.hb, test_pred),
        temperature=temperature,
        notes=notes,
        weights=None if best_state is None else {k: v.cpu() for k, v in best_state.items()},
        checkpoint_path=checkpoint,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def results_row(fit: HeadFit, features: FeatureSet, fold: Fold) -> dict[str, object]:
    """One row of the C1 results table, unrounded.

    Kept here rather than in the CLI so the printed table, the CSV and the JSON
    are all the same numbers, derived once. Rounding happens on the way out.
    """
    test = features.select(fold.test)
    spread = subgroup_spread(
        evaluate_subgroups(fit.test_hb_true, fit.test_hb_pred, np.array(test.site, dtype=object))
    )
    r = fit.report
    return {
        "head": fit.head,
        "n_test": r.n,
        "mae": r.mae,
        "rmse": r.rmse,
        "r2": r.r2,
        "pearson_r": r.pearson_r,
        "bias": r.bias,
        "loa_lower": r.loa_lower,
        "loa_upper": r.loa_upper,
        "within_1": r.within_1,
        "within_2": r.within_2,
        "site_mae_gap": spread.get("mae_gap", float("nan")),
        "site_bias_gap": spread.get("bias_gap", float("nan")),
        "worst_site": spread.get("worst_group"),
        "worst_site_n": spread.get("worst_group_n"),
        "best_epoch": fit.best_epoch,
        "val_loss": fit.best_val_loss,
        "seconds": round(fit.train_seconds, 2),
        **{f"note:{i}": n for i, n in enumerate(fit.notes)},
    }


def _mean_age(ages: _FloatArr, sites: tuple[str, ...], site: str) -> float:
    """Mean age for one site, ignoring patients with no age recorded."""
    subset = [a for a, s in zip(ages, sites, strict=True) if s == site and np.isfinite(a)]
    if not subset:
        return float("nan")
    return float(np.mean(subset))


def _rounded(row: dict[str, object]) -> dict[str, object]:
    """Round the numeric fields for the on-disk artefacts; leave text alone.

    Every value is coerced to a built-in type on the way through, not just rounded,
    because a numpy scalar is not JSON-serialisable and two of them arrive here from
    ordinary code: ``np.float64`` from a numpy comparison, and ``np.bool_`` from a
    boolean mask. The second is the trap -- in numpy 2.x ``np.bool_`` does **not**
    subclass ``bool``, so the obvious ``isinstance(value, (int, float))`` guard
    passes it straight through, and the command dies with ``TypeError: Object of type
    bool is not JSON serializable``, naming neither a file nor a column.

    ``bool`` is checked first because it subclasses ``int`` and would otherwise be
    rounded into ``1``/``0``, turning a reported flag into a number.
    """
    out: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, (bool, np.bool_)):
            out[key] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = round(float(value), 4)
        else:
            out[key] = value
    return out


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write a results table, with the union of all keys so notes do not shift columns."""
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(
    path: Path, fits: Sequence[HeadFit], features: FeatureSet, fold: Fold
) -> None:
    """Per-patient predictions, so every aggregate in the report is auditable."""
    test = features.select(fold.test)
    rows: list[dict[str, object]] = []
    for fit in fits:
        if fit.test_hb_true.size == 0:
            continue
        for i, pid in enumerate(test.patient_ids):
            rows.append(
                {
                    "head": fit.head,
                    "fold": fold.name,
                    "patient_id": pid,
                    "site": test.site[i],
                    "sex": test.sex[i],
                    "age": test.age[i],
                    "hb_true": round(float(fit.test_hb_true[i]), 4),
                    "hb_pred": round(float(fit.test_hb_pred[i]), 4),
                    "error": round(float(fit.test_hb_pred[i] - fit.test_hb_true[i]), 4),
                    "severity": severity_bin(float(fit.test_hb_true[i]), test.sex[i]),
                }
            )
    _write_csv(path, rows)


#: Fixed table header. Deliberately ASCII: this is printed to a Windows console
#: that defaults to cp1252, which cannot encode Greek delta, and a table that
#: raises UnicodeEncodeError after training has finished is a poor way to lose a
#: run. ``dMAE`` reads better than the delta anyway.
_TABLE_HEADER = (
    f"{'head':<11}{'n':>4}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'r':>7}{'bias':>8}"
    f"{'LoA-':>8}{'LoA+':>8}{'±1':>7}{'±2':>7}{'dMAE':>8}"
)


def print_results_table(rows: Sequence[dict[str, object]], title: str) -> None:
    """A fixed-width table. Column order is fixed so diffs between runs are readable."""
    print()
    print(title)
    print("=" * len(title))
    print(_TABLE_HEADER)
    print("-" * len(_TABLE_HEADER))
    for r in rows:
        print(
            f"{r['head']!s:<11}{int(r['n_test']):>4}"
            f"{float(r['mae']):>8.3f}{float(r['rmse']):>8.3f}{float(r['r2']):>8.3f}{float(r['pearson_r']):>7.3f}"
            f"{float(r['bias']):>8.3f}{float(r['loa_lower']):>8.3f}{float(r['loa_upper']):>8.3f}"
            f"{float(r['within_1']):>7.2f}{float(r['within_2']):>7.2f}{float(r['site_mae_gap']):>8.3f}"
        )


def run_single_split(
    features: FeatureSet,
    *,
    cfg: TrainConfig,
    fold: Fold,
    heads: Sequence[str] = HEAD_NAMES,
    verbose: bool = True,
) -> list[HeadFit]:
    """Train every head on one patient-disjoint split and score the test fold."""
    fits: list[HeadFit] = []
    for head in heads:
        if verbose:
            print(f"\n  head: {head}")
        fits.append(evaluate_fold(head, fold, features, cfg=cfg, verbose=verbose))
    return fits


def run_site_holdout(
    features: FeatureSet,
    *,
    cfg: TrainConfig,
    heads: Sequence[str] = ("regression", "ordinal"),
    verbose: bool = True,
) -> list[dict[str, object]]:
    """Cross-site validation in both directions.

    There is no validation set to select on when a whole site is held out, so the
    number of epochs is fixed at the config value rather than chosen. That is a
    real limitation and it is recorded in the row rather than hidden: the reported
    figure is what this fixed schedule achieves, not the best this architecture
    could do on Italy.
    """
    from hemolux.data.splits import site_holdout_folds

    site_of = dict(zip(features.patient_ids, features.site, strict=True))
    rows: list[dict[str, object]] = []
    for fold in site_holdout_folds(features.patient_ids, site_of):
        for head in heads:
            if verbose:
                print(f"\n  head: {head}  fold: {fold.name}")
            rows.append(_site_holdout_row(head, fold, features, cfg=cfg, verbose=verbose))
    return rows


def _site_holdout_row(
    head: str,
    fold: Fold,
    features: FeatureSet,
    *,
    cfg: TrainConfig,
    verbose: bool,
) -> dict[str, object]:
    """Fit for a fixed epoch budget on one site, score the other, no selection."""
    configure_torch()
    torch_device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    train = features.select(fold.train)
    test = features.select(fold.test)
    model = build_head(head, train.feature_dim).to(torch_device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_train = torch.tensor(train.features, dtype=torch.float32, device=torch_device)
    y_train = torch.tensor(train.hb, dtype=torch.float32, device=torch_device)
    sev_train = torch.tensor(train.band_index(), dtype=torch.long, device=torch_device)
    sev_w = InverseFrequencyWeights(sev_train, n_classes=len(SEVERITY_CLASSES)).to(torch_device)
    n_train = x_train.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    started = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator)
        running = 0.0
        for start in range(0, n_train, cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = _loss_for(head, model(x_train[rows]), y_train[rows], cfg, sev_train[rows], sev_w)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(rows)
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs):
            print(
                f"    {head:<10} {fold.name:<16} epoch {epoch:>3}/{cfg.epochs}  train {running / max(n_train, 1):.4f}"
            )

    test_out = _forward_all(model, test.features)
    # Class centres come from the TRAINING site. This is the only place in the
    # project where the class means cross a site boundary, so it is the place the
    # cross-site result is most sensitive to: a held-out site whose Hb distribution
    # differs from the training site's cannot be represented by these centres, and
    # the bias column is what shows it.
    centres = training_centres(head, train)
    test_pred = decode_predictions(head, test_out, centres)
    report = regression_report(test.hb, test_pred)
    groups = evaluate_subgroups(test.hb, test_pred, np.array(test.sex, dtype=object))
    spread = subgroup_spread(groups)
    return {
        "head": head,
        "fold": fold.name,
        "train_site": fold.name.split("->")[0],
        "test_site": fold.name.split("->")[-1],
        "n_train": len(fold.train),
        "n_test": report.n,
        "mae": round(report.mae, 4),
        "rmse": round(report.rmse, 4),
        "r2": round(report.r2, 4),
        "pearson_r": round(report.pearson_r, 4),
        "bias": round(report.bias, 4),
        "loa_lower": round(report.loa_lower, 4),
        "loa_upper": round(report.loa_upper, 4),
        "within_1": round(report.within_1, 4),
        "within_2": round(report.within_2, 4),
        "mean_true_hb_test_site": round(float(test.hb.mean()), 3),
        "mean_true_hb_train_site": round(float(train.hb.mean()), 3),
        "sex_mae_gap": round(spread.get("mae_gap", float("nan")), 4),
        "selection": "fixed epoch budget, no validation patients available",
        "seconds": round(time.perf_counter() - started, 2),
    }


def coverage_report(fit: HeadFit) -> dict[str, object]:
    """Risk-coverage curve on the test fold, plus the abstention threshold.

    Uncertainty for the ordinal head is the predictive standard deviation of the
    bin distribution, which needs no ground truth -- that is the whole reason the
    ordinal formulation is proposed over a bare regression. A bare regression
    head has no uncertainty estimate at all, so it has no coverage curve; that
    asymmetry is the practical argument for the ordinal formulation and is
    reported rather than asserted.

    The multitask head's ordinal arm qualifies, because :data:`DECODE_HEAD` routes
    it through the same decode.
    """
    if DECODE_HEAD.get(fit.head) != "ordinal" or fit.test_outputs.size == 0:
        return {
            "available": False,
            "reason": (
                f"a distribution over bins exists only for the ordinal arm, "
                f"not {fit.head!r}; a bare regression has no uncertainty to rank by"
            ),
        }

    probs = _softmax(fit.test_outputs)
    mu = np.asarray(HB_BIN_CENTRES, dtype=np.float64)
    mean = expected_hb(probs)
    sigma = np.sqrt(np.clip((probs * mu**2).sum(axis=1) - mean**2, 0.0, None))

    curve = risk_coverage_curve(sigma, fit.test_hb_true, fit.test_hb_pred)
    points = [(round(p.coverage, 3), round(p.mae, 4)) for p in curve]
    full = curve[0].mae if curve else float("nan")
    return {
        "available": True,
        "uncertainty": "predictive SD of the ordinal bin distribution",
        "mae_at_full_coverage": round(float(full), 4),
        "mae_at_25pct_coverage": round(float(curve[-1].mae), 4) if curve else None,
        "curve": points,
        "n": int(fit.report.n),
    }


def build_report(
    fits: Sequence[HeadFit],
    features: FeatureSet,
    fold: Fold,
    *,
    cfg: TrainConfig,
    site_rows: Sequence[dict[str, object]] = (),
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assemble the machine-readable results file the reports are written from."""
    ensure_dirs()
    rows = [_rounded(results_row(f, features, fold)) for f in fits]
    payload: dict[str, object] = {
        "config": asdict(cfg),
        "corpus": {
            "n_patients": len(features.patient_ids),
            "roi": features.roi,
            "backbone": features.backbone,
            "feature_dim": features.feature_dim,
            "per_site": {
                site: {
                    "n": int(sum(1 for s in features.site if s == site)),
                    "hb_mean": round(
                        float(
                            np.mean(
                                [
                                    h
                                    for h, s in zip(features.hb, features.site, strict=True)
                                    if s == site
                                ]
                            )
                        ),
                        3,
                    ),
                    "age_mean": round(_mean_age(features.age, features.site, site), 2),
                }
                for site in sorted(set(features.site))
            },
        },
        "fold": {
            "name": fold.name,
            "n_train": len(fold.train),
            "n_val": len(fold.val),
            "n_test": len(fold.test),
        },
        "single_split": rows,
        "site_holdout": list(site_rows),
        "coverage": {f.head: coverage_report(f) for f in fits},
        "reference": REFERENCE_NOTE,
    }
    if extra:
        payload.update(extra)

    ARTIFACT_REPORTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_REPORTS / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    _write_csv(ARTIFACT_REPORTS / "results.csv", rows)
    _write_predictions(ARTIFACT_REPORTS / "predictions.csv", fits, features, fold)
    return payload
