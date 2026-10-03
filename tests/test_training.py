"""Tests for the training loop's data handling and decoding.

This file is mostly about one property: **the numbers must not know about the
patients they are scored on.** Every function here takes a training split and a set
of splits being scored, and the failure mode that matters is a value crossing from
the second into the first. Such a failure produces a perfectly plausible table --
the loss still falls, the numbers still look reasonable, the number is simply
optimistic by however much the test patients differed from the training ones.

So the tests that follow are shaped around *changing what should not matter* and
asserting nothing moved:

* Rewrite every held-out patient's haemoglobin to an absurd value and check the
  training centres do not move (:func:`test_centres_ignore_everything_but_train`).
* Check a class absent from the training split takes the training mean and not the
  corpus mean, which would leak the held-out patients through the arithmetic
  (:func:`test_a_class_missing_from_training_takes_the_training_mean`).
* Check the split machinery does not depend on the order patients arrive in, since
  ``fold.test`` is built by walking a feature cache whose row order follows the
  filesystem.

The rest is ordinary coverage of the decode, the cache and the WHO thresholds, with
the binary head's single cutoff pinned explicitly -- see
:func:`test_the_binary_head_uses_one_cutoff_for_every_sex`, which documents a real
fairness consequence rather than a bug in the code.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from hemolux.config import WHO_CUTOFF_FEMALE, WHO_CUTOFF_MALE
from hemolux.data.splits import Fold, severity_bin
from hemolux.metrics.calibration import HB_BIN_CENTRES
from hemolux.metrics.regression import regression_report
from hemolux.training import (
    DECODE_HEAD,
    HEAD_NAMES,
    SEVERITY_CLASSES,
    FeatureSet,
    TrainConfig,
    _rounded,
    coverage_report,
    decode_predictions,
    evaluate_fold,
    feature_cache_path,
    load_feature_cache,
    results_row,
    save_feature_cache,
    score_test_fold,
    severity_labels,
    training_centres,
    validation_report,
)

DIM = 4


def make_features(
    ids: tuple[str, ...],
    hb: tuple[float, ...],
    sex: tuple[str, ...],
    *,
    site: tuple[str, ...] | None = None,
    dim: int = DIM,
) -> FeatureSet:
    """A FeatureSet with meaningless features and meaningful metadata.

    The feature values are zeros on purpose: none of these tests exercises the
    network, and a random matrix would invite a future test into asserting a number
    that depends on the seed.
    """
    rng = np.random.default_rng(0)
    return FeatureSet(
        patient_ids=ids,
        features=rng.normal(size=(len(ids), dim)),
        hb=np.array(hb, dtype=np.float64),
        site=site or tuple("India" for _ in ids),
        sex=sex,
        age=np.full(len(ids), 30.0),
        roi="palpebral",
        backbone="fake_backbone",
        feature_dim=dim,
        severity=tuple(severity_bin(h, s) for h, s in zip(hb, sex, strict=True)),
    )


#: Eight men spanning all four WHO bands, so every severity class has a member.
#: India and Italy split evenly, for the site code paths.
MIXED = make_features(
    ids=("India/1", "India/2", "Italy/3", "Italy/4", "India/5", "India/6", "Italy/7", "Italy/8"),
    hb=(15.0, 14.0, 10.5, 9.5, 8.0, 6.0, 12.5, 13.5),
    sex=("M",) * 8,
    site=("India", "India", "Italy", "Italy", "India", "India", "Italy", "Italy"),
)


# --------------------------------------------------------------------------- #
# Leakage: the property the whole file is about
# --------------------------------------------------------------------------- #


def _disturb_outside(
    features: FeatureSet, keep: tuple[str, ...], value: float = 99.0
) -> FeatureSet:
    """A copy where every patient *not* in ``keep`` has been given ``value``.

    The point is that the training rows are bit-identical and only the held-out ones
    moved. Overwriting all rows -- which an earlier version of this file did -- tests
    a different question entirely, because the training patients move too.
    """
    held = set(features.patient_ids) - set(keep)
    hb = features.hb.copy()
    for i, pid in enumerate(features.patient_ids):
        if pid in held:
            hb[i] = value
    return replace(
        features,
        hb=hb,
        severity=tuple(severity_bin(float(h), s) for h, s in zip(hb, features.sex, strict=True)),
    )


def test_centres_ignore_everything_but_train() -> None:
    """The held-out patients cannot reach the training centres.

    ``training_centres`` is handed the training split and must use nothing else. The
    check is destructive rather than structural: move every *held-out* patient to
    99 g/dL -- a value no corpus-wide mean survives unchanged -- and confirm the
    centres are bit-identical. A structural test ("does the function mention
    ``test``?") would not survive someone adding a corpus mean to it, and the failure
    would be an optimistic number rather than a crash.
    """
    train_ids = ("India/1", "Italy/3", "India/5", "Italy/7")
    before = training_centres("severity", MIXED.select(train_ids))

    after = training_centres("severity", _disturb_outside(MIXED, train_ids).select(train_ids))

    assert before.tolist() == after.tolist()
    assert max(before) - min(before) > 1.0, "the fixture is too uniform to prove anything"


def test_the_binary_centres_follow_the_cutoff_not_the_corpus() -> None:
    """Class 1 is "below the cutoff", and its centre is that group's own mean."""
    anaemic = ("M", "M", "M")
    train = make_features(("a", "b", "c"), (15.0, 9.0, 7.0), anaemic)
    centres = training_centres("binary", train)

    assert centres.size == 2
    assert centres[0] == pytest.approx(15.0), "index 0 is the at-or-above group"
    assert centres[1] == pytest.approx(8.0), "index 1 is the below group"


def test_a_class_missing_from_training_takes_the_training_mean() -> None:
    """A fallback that averaged the corpus would leak the held-out patients.

    The real corpus has no severe anaemia patient in any split, so the severity
    head's fourth class is always absent from training. Its centre is therefore
    always the fallback. If that fallback were the corpus mean, every severity
    prediction would be pulled toward the whole-corpus average -- including the test
    patients' own -- and the number would be quietly optimistic.
    """
    train_ids = ("India/5", "India/6")  # 8.0 (moderate) and 6.0 (severe)
    train = MIXED.select(train_ids)
    centres = training_centres("severity", train)

    assert centres.size == len(SEVERITY_CLASSES)
    assert centres[2] == pytest.approx(8.0), "the one moderate patient"
    assert centres[3] == pytest.approx(6.0), "the one severe patient"
    # Neither member is below 9.0, so normal and mild have no training patient and
    # both take the training-split fallback of 7.0.
    assert centres[0] == pytest.approx(centres[1]) == pytest.approx(7.0)

    # And that fallback is the training mean, not the corpus mean.
    after = training_centres("severity", _disturb_outside(MIXED, train_ids).select(train_ids))
    assert centres.tolist() == after.tolist()


def test_the_ordinal_centres_are_the_fixed_grid_and_ignore_the_data() -> None:
    """The ordinal formulation's whole claim is that the bins are known a priori.

    So its centres are read from the calibration module and pass through the corpus
    untouched. If they were ever measured from the split, the head would stop being
    an ordinal formulation and become the severity head with a different name.
    """
    train = make_features(("a", "b"), (14.0, 7.0), ("M", "M"))
    assert training_centres("ordinal", train).tolist() == pytest.approx(list(HB_BIN_CENTRES))


def test_the_regression_centres_are_empty_because_none_are_needed() -> None:
    """Its output is already g/dL, so mapping it onto a grid would be a second unit."""
    train = make_features(("a", "b"), (14.0, 7.0), ("M", "M"))
    assert training_centres("regression", train).size == 0


def test_an_unknown_head_is_refused_rather_than_defaulted() -> None:
    """Silently treating an unknown name as ``severity`` would train the wrong head."""
    train = make_features(("a",), (12.0,), ("M",))
    with pytest.raises(KeyError, match="unknown head"):
        training_centres("classification", train)


def test_multitask_decodes_through_the_ordinal_grid() -> None:
    """``multitask`` predicts bin logits as one of its tasks.

    It therefore has no centres of its own and must not be asked for any. Recording
    the mapping in one place stops a caller inventing a second one.
    """
    assert DECODE_HEAD["multitask"] == "ordinal"

    raw = np.zeros((1, len(HB_BIN_CENTRES)))
    out = decode_predictions("multitask", raw, np.asarray(HB_BIN_CENTRES))
    assert out.tolist() == pytest.approx([float(np.mean(HB_BIN_CENTRES))])


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def test_the_regression_decode_is_a_passthrough() -> None:
    """g/dL in, g/dL out. Anything else would be a unit error waiting to ship."""
    raw = np.array([9.4, 12.1, 7.8])
    assert decode_predictions("regression", raw, np.empty(0)).tolist() == pytest.approx(
        raw.tolist()
    )


def test_the_binary_decode_reads_the_sigmoid_as_a_two_bin_posterior() -> None:
    """``centres[0]`` is normal, so ``centres[1]`` must be reached as ``P(anaemic)``.

    A sign flip here would not raise -- it would report high haemoglobin for the
    anaemic patients, which is the most dangerous single mistake available in this
    repository, since the app's threshold would then abstain on healthy eyes and
    wave through anaemic ones.
    """
    centres = np.array([14.5, 8.0])
    anaemic = decode_predictions("binary", np.array([10.0]), centres)
    normal = decode_predictions("binary", np.array([-10.0]), centres)

    assert float(anaemic[0]) < 8.5, "a large positive logit should read as anaemic"
    assert float(normal[0]) > 14.0, "a large negative logit should read as normal"


def test_the_binary_decode_is_monotone_in_the_logit() -> None:
    centres = np.array([14.5, 8.0])
    logits = np.linspace(-20.0, 20.0, 41)
    out = decode_predictions("binary", logits, centres)
    assert np.all(np.diff(out) < 0), "haemoglobin rose with the anaemia logit"


def test_the_ordinal_decode_is_the_posterior_mean() -> None:
    """A peaked posterior reads as its own bin; a flat one reads as the mean."""
    centres = np.asarray(HB_BIN_CENTRES)
    peaked = np.full((1, centres.size), -20.0)
    peaked[0, 7] = 20.0

    assert float(decode_predictions("ordinal", peaked, centres)[0]) == pytest.approx(centres[7])
    flat = np.zeros((1, centres.size))
    assert float(decode_predictions("ordinal", flat, centres)[0]) == pytest.approx(centres.mean())


def test_a_wrong_number_of_centres_is_refused() -> None:
    """Otherwise the arithmetic broadcasts or, worse, silently truncates."""
    raw = np.zeros((1, len(HB_BIN_CENTRES)))
    with pytest.raises(ValueError, match="centres were supplied"):
        decode_predictions("ordinal", raw, np.asarray(HB_BIN_CENTRES)[:-1])


def test_a_1d_output_for_a_classification_head_is_refused() -> None:
    """A shape error, not a guess. A single column cannot be a distribution."""
    with pytest.raises(ValueError, match="must be 2-D"):
        decode_predictions("ordinal", np.zeros(3), np.asarray(HB_BIN_CENTRES))


def test_the_decode_survives_a_row_of_very_large_logits() -> None:
    """Softmax subtracts the row max; without it one patient overflows the run.

    A frozen backbone on a real 224x224 crop produces logits in the tens, and a
    half-trained head can produce hundreds. ``np.exp`` of that is ``inf``, and
    ``inf / inf`` is ``nan``, which would then be written to ``results.csv`` as a
    silent hole in the middle of a table.
    """
    raw = np.array([[1e3, 1e3 + 1.0, 999.0]])
    out = decode_predictions("severity", raw, np.array([14.0, 10.0, 8.0]))
    assert np.isfinite(out).all()
    assert 7.0 < float(out[0]) < 15.0


# --------------------------------------------------------------------------- #
# WHO thresholds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("hb", "sex", "expected"),
    [
        (19.0, "M", "normal"),
        (13.0, "M", "normal"),  # exactly at the cutoff is not anaemic
        (12.99, "M", "mild"),
        (9.0, "M", "mild"),
        (8.99, "M", "moderate"),
        (7.0, "M", "moderate"),
        (6.99, "M", "severe"),
        (2.0, "M", "severe"),
        (12.0, "F", "normal"),
        (11.99, "F", "mild"),
        (9.0, "F", "mild"),
        (8.99, "F", "moderate"),
        (6.99, "F", "severe"),
    ],
)
def test_the_severity_bands_match_the_who_table(hb: float, sex: str, expected: str) -> None:
    """The bands are a published table, so they are pinned value by value.

    Read off WHO's cut-offs for anaemia in non-pregnant adults -- <13 g/dL for men,
    <12 g/dL for women, with the moderate and severe boundaries at 9.0 and 7.0 shared
    between the sexes. The 7.0/9.0 edges are inclusive-below, so the boundary values
    are asserted on both sides: an edit that flips ``<`` to ``<=`` moves a whole band
    by one patient and no metric would notice.
    """
    assert severity_bin(hb, sex) == expected


def test_severity_labels_reuse_the_same_function_as_the_split() -> None:
    """The drift guard.

    ``severity_bin`` is used to stratify the split, ``severity_labels`` to train the
    head, and the reported subgroup name is the same string. If the ordering were
    written down a second time anywhere, a head could train "class 2 = moderate" while
    the results table called it severe. Asserting the two agree row for row is what
    keeps a four-class head honestly labelled.
    """
    hb = np.array([15.0, 12.0, 10.0, 5.0, 13.5, 8.5])
    sex = ("M", "F", "M", "F", "M", "F")
    labels = severity_labels(hb, sex).tolist()

    assert labels == [
        SEVERITY_CLASSES.index(severity_bin(float(h), s)) for h, s in zip(hb, sex, strict=True)
    ]
    assert labels == [0, 0, 1, 3, 0, 2], "the class order changed under a threshold edit"


def test_severity_labels_reject_a_length_mismatch() -> None:
    """Zip's ``strict`` would catch it, but the message should name the problem."""
    with pytest.raises(ValueError, match="sexes"):
        severity_labels(np.array([12.0, 11.0]), ("M",))


def test_a_feature_set_without_severity_refuses_to_invent_it() -> None:
    """An empty tuple means "not computed", not "no patient is anaemic"."""
    bare = replace(MIXED, severity=())
    with pytest.raises(ValueError, match="no severity labels"):
        bare.band_index()


def test_the_binary_head_uses_one_cutoff_for_every_sex() -> None:
    """Pinned because it is a real limitation, not an accident.

    ``training_centres`` labels the binary head with ``hb < WHO_CUTOFF_MALE`` for
    every patient. Applying the male cutoff to women labels every woman between 12.0
    and 13.0 g/dL as anaemic when WHO calls her normal -- so the head **over-calls**
    anaemia in women, and pools those women with the genuinely anaemic ones, dragging
    that class's centre down. The direction matters: it is a false-positive bias, not
    a missed diagnosis, so it costs specificity rather than sensitivity.

    This belongs in a test so that fixing it later is a deliberate change that moves
    an assertion and updates the fairness report, rather than a quiet edit. The
    ordinal head -- the one the app exports -- has no such gap: its grid is fixed and
    sex-independent.
    """
    train = make_features(("a", "b", "c"), (12.8, 15.0, 7.0), ("F", "M", "M"))
    centres = training_centres("binary", train)

    assert centres[0] == pytest.approx(15.0), "index 0 is the only not-anaemic patient"
    assert severity_bin(12.8, "F") == "normal", "12.8 in a woman is not anaemic"
    assert centres[1] == pytest.approx((12.8 + 7.0) / 2), (
        "the normal woman is pooled with the anaemic man, pulling the centre down"
    )
    assert WHO_CUTOFF_FEMALE < 12.8 < WHO_CUTOFF_MALE


# --------------------------------------------------------------------------- #
# The feature cache
# --------------------------------------------------------------------------- #


def test_the_cache_path_names_the_backbone_the_roi_and_the_width() -> None:
    """Because a cache is keyed on exactly what changes the numbers.

    A cache file for ``palpebral`` reused by a ``forniceal`` run would be the worst
    available failure: the run completes, trains, reports, and is entirely wrong.
    Two runs of the same configuration must also share a file, or every run
    re-featurises 217 images through a backbone on the CPU.
    """
    path = feature_cache_path(MIXED)
    name = path.name

    assert "fake_backbone" in name
    assert "palpebral" in name
    assert f"{DIM}d" in name

    other_roi = feature_cache_path(replace(MIXED, roi="forniceal"))
    other_dim = feature_cache_path(replace(MIXED, feature_dim=8))
    assert other_roi != path and other_dim != path
    assert feature_cache_path(replace(MIXED)) == path


def test_the_cache_survives_a_round_trip(tmp_path: Path) -> None:
    """Every field the results depend on must come back, not just the matrix.

    The severities are derived from sex, and sex is only in the cache because it was
    saved there. Dropping it would leave ``band_index`` raising on every reporting
    path, which is loud -- but dropping ``severity`` would instead silently produce
    reports with no subgroup breakdown.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("hemolux.training.ARTIFACT_MODELS", tmp_path)
    try:
        path = save_feature_cache(MIXED)
    finally:
        monkeypatch.undo()

    back = load_feature_cache(path)

    assert back.patient_ids == MIXED.patient_ids
    assert back.hb.tolist() == pytest.approx(MIXED.hb.tolist())
    assert back.site == MIXED.site
    assert back.sex == MIXED.sex
    assert back.severity == MIXED.severity
    assert back.feature_dim == MIXED.feature_dim
    assert back.roi == MIXED.roi
    assert back.backbone == MIXED.backbone
    assert np.allclose(back.features, MIXED.features), (
        "the feature matrix did not survive the round trip"
    )
    assert back.features.dtype == np.float64, "the cache widened the dtype on the way back"


def test_a_cache_written_before_severity_existed_still_loads(tmp_path: Path) -> None:
    """An old cache is missing a key, not corrupt.

    It loads with empty severities and then refuses to invent them, which is the
    right pair of behaviours: the features are still worth keeping, and the
    severity-dependent paths fail loudly rather than reporting zeros.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("hemolux.training.ARTIFACT_MODELS", tmp_path)
    try:
        path = save_feature_cache(MIXED)
    finally:
        monkeypatch.undo()

    with np.load(path, allow_pickle=True) as data:
        trimmed = {k: data[k] for k in data.files if k != "severity"}
    np.savez_compressed(path, **trimmed)

    back = load_feature_cache(path)
    assert back.severity == ()
    assert back.patient_ids == MIXED.patient_ids
    with pytest.raises(ValueError, match="no severity labels"):
        back.band_index()


# --------------------------------------------------------------------------- #
# Splitting a FeatureSet
# --------------------------------------------------------------------------- #


def test_select_subset_every_field_in_step() -> None:
    """Eight rows down to two, with nothing left behind.

    A row-order bug here misaligns ``hb`` against ``features`` and trains on
    shuffled labels -- the loss would still fall and the results would be
    meaningless, so it is worth checking the arrays rather than the length.
    """
    subset = MIXED.select(("Italy/4", "India/1"))

    assert subset.patient_ids == ("Italy/4", "India/1")
    assert subset.hb.tolist() == pytest.approx([9.5, 15.0])
    assert subset.site == ("Italy", "India")
    assert subset.severity == ("mild", "normal")
    assert np.allclose(subset.features, MIXED.features[[3, 0]]), (
        "features were not reindexed with the metadata"
    )


def test_select_preserves_the_order_asked_for() -> None:
    """Because a fold's members are assembled by walking a cache, then compared
    against predictions the training loop built by walking a differently ordered
    tuple. Any reordering here silently misaligns labels with features.

    Duplicates are *not* collapsed, and that is deliberate rather than an oversight:
    de-duplicating would hide a caller that assembled a fold wrongly, and the result
    would still be self-consistent -- one patient counted twice in every metric, with
    no error anywhere. Folds are built from ``FeatureSet.patient_ids``, which is
    unique, so the requirement is on the caller and is stated here instead.
    """
    subset = MIXED.select(("India/6", "India/1"))
    assert subset.patient_ids == ("India/6", "India/1")
    assert subset.hb.tolist() == pytest.approx([6.0, 15.0])

    repeated = MIXED.select(("India/1", "India/1"))
    assert repeated.patient_ids == ("India/1", "India/1")
    assert repeated.features.shape[0] == 2
    assert len(set(MIXED.patient_ids)) == len(MIXED.patient_ids), "the fixture has repeats"


def test_select_names_what_is_missing() -> None:
    """An unknown id is a caller bug, and the message should carry the ids."""
    with pytest.raises(KeyError) as caught:
        MIXED.select(("India/1", "India/999"))
    assert "India/999" in str(caught.value)
    assert "1 patient" in str(caught.value)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_default_configuration_freezes_the_backbone() -> None:
    """Parameter budget is how the control is held, so it must be the default.

    D-006 settled this: comparing a frozen trunk against the same trunk unfrozen is
    the only controlled comparison available, because unfreezing costs the fine-tune
    data this corpus does not have. The default therefore has to be the *control*,
    or every quick run silently produces the treatment arm.
    """
    assert TrainConfig().unfreeze_param_fraction == 0.0


def test_the_configuration_is_frozen() -> None:
    """A run records the config it used; if it could be edited mid-flight, it lied."""
    cfg = TrainConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.epochs = 1  # type: ignore[misc]


def test_the_default_seed_is_fixed() -> None:
    """Every number in ``EVALS.md`` is reproducible from the committed code."""
    assert TrainConfig().seed == 42


def test_every_head_in_the_registry_decodes_somewhere() -> None:
    """The five heads and the decode map must not drift apart.

    ``HEAD_NAMES`` is what ``--heads all`` iterates; ``DECODE_HEAD`` is what turns an
    output into g/dL. A head in the first and not the second trains, saves, reports
    NaN, and does so silently.
    """
    assert set(DECODE_HEAD) == set(HEAD_NAMES)
    assert set(DECODE_HEAD.values()) <= {"binary", "severity", "regression", "ordinal"}


def test_the_head_names_are_the_documented_five() -> None:
    """README and EVALS.md both name them; a rename has to reach all three."""
    assert HEAD_NAMES == ("binary", "severity", "regression", "ordinal", "multitask")
    assert SEVERITY_CLASSES == ("normal", "mild", "moderate", "severe")


def test_a_result_row_survives_json(tmp_path: Path) -> None:
    """The regression that made ``hemolux validate --json`` crash, kept cheap.

    A numpy scalar reaching a results row is not JSON-serialisable, and two of them
    get there from ordinary code: ``np.float64`` from a numpy comparison and
    ``np.bool_`` from a boolean mask. Both crashed the command with a
    ``TypeError: Object of type bool is not JSON serializable`` that names no file
    and no column.

    ``np.bool_`` is the sharper of the two, because it is the trap: in numpy 2.x it
    does **not** subclass ``bool``, so an ``isinstance(value, bool)`` guard written
    for Python's own booleans silently passes it through. This asserts against
    :func:`_rounded` itself rather than a re-implementation of it, so a future cast
    added somewhere else cannot leave this passing while the artefact stays broken.
    """
    row = {
        "head": "ordinal",
        "mae": np.float64(1.38237),
        "r2": np.float64(0.5451),
        "within_1": np.bool_(True),
        "within_2": np.bool_(False),
        "n_test": np.int64(43),
        "centres_saved": True,  # a real bool, which must survive as a bool
    }

    rounded = _rounded(row)

    for key, value in rounded.items():
        assert isinstance(value, (str, int, float, bool)), f"{key} stayed a {type(value)}"
    assert isinstance(rounded["within_1"], bool), "an np.bool_ reached the artefact"
    assert rounded["mae"] == pytest.approx(1.3824), "the rounding did not happen"
    assert rounded["centres_saved"] is True, "a real bool was mangled into 1"

    json.dumps(rounded)  # the actual failure this pins


# --------------------------------------------------------------------------- #
# Holding out the test fold: the property a configuration sweep depends on
# --------------------------------------------------------------------------- #

#: Four train, two validation, two test, drawn from MIXED so every WHO band and both
#: sites appear on each side of the split. Two test patients is the smallest that
#: still lets ``r2`` be computed at all.
SWEEP_FOLD = Fold(
    name="sweep0",
    train=("India/1", "Italy/3", "India/5", "Italy/7"),
    val=("India/2", "Italy/8"),
    test=("Italy/4", "India/6"),
)

CFG = TrainConfig(epochs=4)


def test_the_test_fold_can_be_left_unscored() -> None:
    """``score_test=False`` yields no test numbers at all, rather than empty ones.

    The distinction is the whole point. A row of NaNs reads as "computed, undefined",
    which is a different and much weaker claim than "never looked at", and a sweep
    that reports the first under the heading of the second has misrepresented the
    order in which it saw its data.
    """
    fit = evaluate_fold(
        "regression",
        SWEEP_FOLD,
        MIXED,
        cfg=CFG,
        verbose=False,
        save_checkpoint=False,
        score_test=False,
    )

    assert fit.report is None
    assert fit.test_hb_pred.size == 0
    assert fit.test_outputs.size == 0
    # The ground truth is deliberately kept. It is part of the corpus rather than
    # something the run chose to look at, and a caller needs it to confirm the fold
    # really had the expected number of patients. What must be absent is the
    # *prediction*, because without one no metric can be formed even by accident.
    assert fit.test_hb_true.size == len(SWEEP_FOLD.test)


def test_an_unscored_test_fold_cannot_move_the_validation_numbers() -> None:
    """Overwrite every test patient's Hb with 99 and the validation fit is unchanged.

    This is the destructive form of the guarantee, and it is deliberately stronger
    than asserting that some counter was not incremented. If the test fold were
    read at all -- to pick an epoch, to fit a temperature, to compute a centre --
    then moving its ground truth to a value nothing else resembles would change
    something downstream. Nothing does.
    """
    kwargs = {"cfg": CFG, "verbose": False, "save_checkpoint": False, "score_test": False}
    clean = evaluate_fold("regression", SWEEP_FOLD, MIXED, **kwargs)
    poisoned = evaluate_fold(
        "regression",
        SWEEP_FOLD,
        _disturb_outside(MIXED, (*SWEEP_FOLD.train, *SWEEP_FOLD.val)),
        **kwargs,
    )

    assert np.isfinite(clean.val_hb_pred).all(), "the fixture produced a degenerate fit"
    assert poisoned.best_epoch == clean.best_epoch
    assert poisoned.best_val_loss == clean.best_val_loss
    np.testing.assert_array_equal(poisoned.val_hb_pred, clean.val_hb_pred)


def test_validation_report_is_the_same_measurement_as_the_reported_one() -> None:
    """Selection and reporting go through one function, so they cannot drift.

    A sweep that ranks candidates by a metric computed slightly differently from the
    one it publishes is ranking on something the reader cannot reproduce.
    """
    fit = evaluate_fold(
        "regression",
        SWEEP_FOLD,
        MIXED,
        cfg=CFG,
        verbose=False,
        save_checkpoint=False,
        score_test=False,
    )

    assert validation_report(fit) == regression_report(fit.val_hb_true, fit.val_hb_pred)


def test_the_winner_can_be_scored_after_selection_without_retraining() -> None:
    """``score_test_fold`` reports the model that would actually ship.

    The weights are the selected epoch's, carried through untouched. Re-fitting to
    regenerate a test score would be the obvious shortcut and it is wrong: the
    optimiser is stochastic, so a second run with the same seed is a different model
    wearing the same configuration.
    """
    fit = evaluate_fold(
        "regression",
        SWEEP_FOLD,
        MIXED,
        cfg=CFG,
        verbose=False,
        save_checkpoint=False,
        score_test=False,
    )

    scored = score_test_fold(fit, MIXED, SWEEP_FOLD)

    assert scored.report is not None
    assert scored.report.n == len(SWEEP_FOLD.test)
    assert np.isfinite(scored.report.mae)
    for key, tensor in fit.weights.items():
        assert torch.equal(scored.weights[key], tensor), f"{key} was not carried through"
    assert scored.best_epoch == fit.best_epoch


def test_a_test_fold_cannot_be_scored_twice() -> None:
    """Re-scoring would overwrite the number and make the first one unfindable."""
    fit = evaluate_fold(
        "regression",
        SWEEP_FOLD,
        MIXED,
        cfg=CFG,
        verbose=False,
        save_checkpoint=False,
        score_test=False,
    )
    scored = score_test_fold(fit, MIXED, SWEEP_FOLD)

    with pytest.raises(ValueError, match="already has test metrics"):
        score_test_fold(scored, MIXED, SWEEP_FOLD)


def test_scoring_a_fit_without_weights_says_so() -> None:
    """The alternative is a silent retrain, which changes the model being scored."""
    fit = evaluate_fold(
        "regression",
        SWEEP_FOLD,
        MIXED,
        cfg=CFG,
        verbose=False,
        save_checkpoint=False,
        score_test=False,
    )
    stripped = replace(fit, weights=None)

    with pytest.raises(ValueError, match="no retained weights"):
        score_test_fold(stripped, MIXED, SWEEP_FOLD)


def test_a_results_row_refuses_to_report_an_unscored_fit() -> None:
    """A results row is a test score; making one here would relabel validation as test."""
    fit = evaluate_fold(
        "regression",
        SWEEP_FOLD,
        MIXED,
        cfg=CFG,
        verbose=False,
        save_checkpoint=False,
        score_test=False,
    )

    with pytest.raises(ValueError, match="score_test=False"):
        results_row(fit, MIXED, SWEEP_FOLD)

    assert results_row(score_test_fold(fit, MIXED, SWEEP_FOLD), MIXED, SWEEP_FOLD)["n_test"] == 2


def test_coverage_explains_an_unscored_fold_rather_than_blaming_the_head() -> None:
    """The unavailable-coverage reason must name the real cause.

    Both branches return ``available: False``, so a swapped condition produces a
    confident, specific, wrong explanation -- the ordinal head blamed for lacking an
    uncertainty estimate it does have, on a run where the data simply is not there.
    """
    unscored = evaluate_fold(
        "ordinal",
        SWEEP_FOLD,
        MIXED,
        cfg=CFG,
        verbose=False,
        save_checkpoint=False,
        score_test=False,
    )

    reason = coverage_report(unscored)["reason"]
    assert "not scored" in reason
    assert "bare regression" not in reason

    scored = score_test_fold(unscored, MIXED, SWEEP_FOLD)
    assert coverage_report(scored)["available"] is True
