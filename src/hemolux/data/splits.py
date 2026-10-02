"""Patient-disjoint split construction.

**This module exists to prevent one specific failure**, and it is the most
important file in the project.

The dataset gives roughly four images per patient: the original photograph plus
three hand-drawn masks of the same eye. If those four land in different splits,
the network is evaluated on images it has effectively already seen, and every
reported metric inflates. With n = 218 and a task this noisy, that single
mistake is enough to turn a mediocre model into an apparently excellent one.

So the grouping key is the **patient**, never the file. Invariants:

1. A patient appears in exactly one split.
2. A file path appears in exactly one split.
3. For cross-site validation, the site is the grouping key too.

All three are asserted in ``tests/test_splits.py`` against the real objects, not
described in a comment.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

#: The two sites present in the public dataset.
SITES: tuple[str, str] = ("India", "Italy")


class Split(StrEnum):
    """Split membership. StrEnum so it round-trips through CSV without ceremony."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    EXTERNAL = "external"


@dataclass(frozen=True)
class Fold:
    """One train/test assignment over patients.

    ``name`` is used verbatim in the ``Prediction`` table, so a fold is always
    traceable from a result back to how it was built.
    """

    name: str
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def assignments(self) -> dict[str, Split]:
        """Map every patient to its split."""
        out: dict[str, Split] = {}
        for pid in self.train:
            out[pid] = Split.TRAIN
        for pid in self.val:
            out[pid] = Split.VAL
        for pid in self.test:
            out[pid] = Split.TEST
        return out

    def __post_init__(self) -> None:
        validate_no_patient_leakage(self)


def validate_no_patient_leakage(fold: Fold) -> None:
    """Raise if any patient or any split overlaps.

    Called from ``Fold.__post_init__``, so an invalid fold cannot be constructed
    and then accidentally used.
    """
    sets = {
        Split.TRAIN: set(fold.train),
        Split.VAL: set(fold.val),
        Split.TEST: set(fold.test),
    }
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise AssertionError(
                    f"patient leakage: {len(overlap)} patient(s) in both {a} and {b}: "
                    f"{sorted(overlap)[:5]}"
                )
    # Counted from the tuples, not from `sets`. Those hold ``set(...)`` values, so
    # ``sum(len(s) for s in sets.values())`` counts each patient once however many
    # times it is repeated, ``total`` and ``unique`` come out equal, and this
    # branch is unreachable. A subject contributing two frames is the realistic
    # route to a repeat, so it has to be reachable.
    total = len(fold.train) + len(fold.val) + len(fold.test)
    unique = len(set().union(*sets.values()))
    if total != unique:
        raise AssertionError("a patient appears more than once within a single split")


def validate_no_file_leakage(
    assignments: dict[str, Split],
    files_by_patient: dict[str, Sequence[str]],
) -> None:
    """Raise if any file path is reachable from two different splits."""
    seen: dict[str, Split] = {}
    for pid, split in assignments.items():
        for path in files_by_patient.get(pid, ()):
            if path in seen and seen[path] != split:
                raise AssertionError(
                    f"file leakage: {path} reachable from both {seen[path]} and {split}"
                )
            seen[path] = split


# --------------------------------------------------------------------------- #
# Random patient-disjoint K-fold
# --------------------------------------------------------------------------- #


def patient_disjoint_kfold(
    patient_ids: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> list[Fold]:
    """Stratified, patient-disjoint K-fold.

    Stratification is on **anemia severity**, not on the raw Hb value, because
    the dataset is severely imbalanced (the large majority are non-anemic). A
    purely random split can easily leave a fold with two or three anaemic cases,
    which makes sensitivity uncomputable in that fold.

    ``val_fraction`` carves a validation set out of each fold's training portion,
    so model selection never touches the fold's test patients.
    """
    ids = sorted({str(p) for p in patient_ids})
    if len(ids) < n_splits:
        raise ValueError(f"cannot build {n_splits} folds from {len(ids)} patients")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ids))

    folds: list[Fold] = []
    for k in range(n_splits):
        test_idx = list(range(k * len(shuffled) // n_splits, (k + 1) * len(shuffled) // n_splits))
        test = tuple(shuffled[i] for i in test_idx)
        remaining = [pid for i, pid in enumerate(shuffled) if i not in set(test_idx)]

        n_val = max(1, round(len(remaining) * val_fraction))
        val = tuple(remaining[:n_val])
        train = tuple(remaining[n_val:])
        folds.append(Fold(name=f"fold{k}", train=train, val=val, test=test))

    return folds


def stratified_assignments(
    patient_ids: Sequence[str],
    severity: Sequence[str] | NDArray[np.generic],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Split]:
    """A single patient-disjoint train/val/test split, stratified on severity.

    Used for the final held-out test evaluation, where K-fold would mean touching
    the test split repeatedly.
    """
    ids = sorted({str(p) for p in patient_ids})
    if len(ids) != len(severity):
        raise ValueError(f"{len(ids)} ids but {len(severity)} severity labels")

    rng = np.random.default_rng(seed)
    by_severity: dict[str, list[str]] = {}
    for pid, sev in zip(ids, (str(s) for s in severity), strict=True):
        by_severity.setdefault(sev, []).append(pid)

    n_val = max(1, round(len(ids) * val_fraction))
    n_test = max(1, round(len(ids) * val_fraction))

    val: list[str] = []
    test: list[str] = []
    # Round-robin over severity strata so both val and test get a proportional
    # share of anaemic cases rather than whatever a single shuffle happened to give.
    order = sorted(by_severity)
    buckets = {s: list(rng.permutation(by_severity[s])) for s in order}

    for i in range(len(ids)):
        sev = order[i % len(order)]
        if not buckets[sev]:
            continue
        pid = buckets[sev].pop()
        if len(val) < n_val:
            val.append(pid)
        elif len(test) < n_test:
            test.append(pid)

    train = [p for p in ids if p not in set(val) | set(test)]
    assignments = dict.fromkeys(train, Split.TRAIN)
    assignments.update(dict.fromkeys(val, Split.VAL))
    assignments.update(dict.fromkeys(test, Split.TEST))
    return assignments


def severity_bin(hb: float, sex: str = "unknown") -> str:
    """WHO severity band, used as the stratification key.

    Thresholds follow the WHO haemoglobin concentration cut-offs for anaemia in
    non-pregnant adults: <13 g/dL for men, <12 g/dL for women. The 7.0 and
    9.0 boundaries mark moderate and severe anaemia.

    Sex is used only to pick the anaemia threshold, not to create separate
    strata — with n = 218, splitting further would leave strata too small to
    report.
    """
    anaemia_cut = 13.0 if str(sex).upper().startswith("M") else 12.0
    if hb < 7.0:
        return "severe"
    if hb < 9.0:
        return "moderate"
    if hb < anaemia_cut:
        return "mild"
    return "normal"


def site_holdout_folds(
    patient_ids: Sequence[str],
    site_of: dict[str, str],
    *,
    sites: tuple[str, str] = SITES,
) -> Iterator[Fold]:
    """Yield cross-site folds: train on one site, test on the other.

    This is the honest version of external validation, and the medical-imaging
    literature notes it is performed in under 6% of published models. Running it
    in both directions matters: a model that only survives India-to-Italy may
    simply have learned one site's camera, lighting protocol or pigmentation
    distribution.
    """
    found = {s for s in site_of.values() if s}
    # Two different failures, both of which used to pass silently.
    #
    # A patient absent from the mapping entirely: caught by the `not in` test.
    #
    # A patient present but mapped to a blank site: the mapping is keyed by
    # patient id, so `p not in site_of` is False and the old check let it
    # through. It then matched neither `site_of.get(p) == train_site` nor
    # `== test_site`, so it was dropped from *both* directions of the holdout
    # and the reported cross-site MAE quietly covered a smaller population than
    # the one the corpus validator counts.
    unknown = sorted({p for p in patient_ids if p not in site_of})
    if unknown:
        raise ValueError(
            f"{len(unknown)} patients have no site label, first: {unknown[0]!r}"
        )
    blank = sorted(p for p in patient_ids if not str(site_of.get(p, "")).strip())
    if blank:
        raise ValueError(
            f"{len(blank)} patients have a blank site label, first: {blank[0]!r}"
        )
    if len(found) < 2:
        raise ValueError(
            f"site holdout needs >= 2 sites, found {sorted(found)}"
        )
    for train_site, test_site in ((sites[0], sites[1]), (sites[1], sites[0])):
        train = tuple(sorted(p for p in patient_ids if site_of.get(p) == train_site))
        test = tuple(sorted(p for p in patient_ids if site_of.get(p) == test_site))
        if not train or not test:
            raise ValueError(
                f"cannot build {train_site}->{test_site}: "
                f"{len(train)} train, {len(test)} test patients"
            )
        yield Fold(name=f"{train_site}->{test_site}", train=train, val=(), test=test)
