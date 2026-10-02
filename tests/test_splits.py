"""Split-construction tests.

Two things are being defended here.

**Leakage.** A patient who appears in both train and test makes every number in
the report meaningless, and it is the single easiest mistake to make in this
project because the same subject contributes an image *and* a mask *and* a
workbook row. ``Fold.__post_init__`` refuses to construct a leaking fold, and
``TestFoldCannotBeConstructedWithLeakage`` holds that line from both directions:
a patient in two splits, and a patient repeated inside one split.

**The severity band.** ``severity_bin`` is the stratification key *and* the
reported subgroup label *and* the training target for the severity head. If it
moves, three things move together and none of them announce it. The expected
bands below are the WHO haemoglobin cut-offs for anaemia in non-pregnant adults
(men below 13 g/dL, women below 12 g/dL; 7.0 and 9.0 marking the moderate and
severe boundaries), not values captured from this implementation.

Stratification is checked proportionally rather than exactly. Round-robin
distribution cannot guarantee an exact ratio from a small stratum, and a test
demanding exactness would be testing the RNG rather than the intent.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemolux.data.splits import (
    SITES,
    Fold,
    Split,
    patient_disjoint_kfold,
    severity_bin,
    site_holdout_folds,
    stratified_assignments,
    validate_no_file_leakage,
    validate_no_patient_leakage,
)

# --------------------------------------------------------------------------- #
# WHO severity bands. (hb, sex) -> band.
# --------------------------------------------------------------------------- #

SEVERITY_CASES = [
    # Severe anaemia: below 7.0 g/dL for either sex.
    (3.4, "M", "severe"),
    (6.9, "F", "severe"),
    (6.99, "M", "severe"),
    # Moderate: 7.0 to below 9.0.
    (7.0, "M", "moderate"),
    (7.0, "F", "moderate"),
    (8.99, "F", "moderate"),
    # Mild: 9.0 up to the sex-specific anaemia threshold.
    (9.0, "M", "mild"),
    (11.9, "F", "mild"),
    (11.99, "F", "mild"),
    (11.99, "M", "mild"),
    (12.99, "F", "normal"),
    (12.99, "M", "mild"),
    # Normal: at or above the threshold.
    (12.0, "F", "normal"),
    (13.0, "M", "normal"),
    (16.2, "M", "normal"),
    (18.0, "F", "normal"),
    # The sex-dependent boundary, which is the whole reason `sex` is a parameter.
    (11.99, "M", "mild"),
    (11.99, "F", "mild"),
    (12.01, "F", "normal"),
    (12.01, "M", "mild"),
]


class TestSeverityBinMatchesWHO:
    @pytest.mark.parametrize(("hb", "sex", "expected"), SEVERITY_CASES)
    def test_band(self, hb: float, sex: str, expected: str) -> None:
        assert severity_bin(hb, sex) == expected

    def test_men_and_women_differ_only_in_the_anemia_threshold(self) -> None:
        """Between 12 and 13 g/dL the two thresholds put the same value in
        different bands. This is the one place the function is sex-sensitive,
        and it is exactly where a silent default would corrupt the stratum."""
        assert severity_bin(12.5, "M") == "mild"
        assert severity_bin(12.5, "F") == "normal"

    def test_severe_and_moderate_boundaries_ignore_sex(self) -> None:
        """7.0 and 9.0 are absolute; no value of sex moves them."""
        for hb in (6.99, 7.0, 8.99, 9.0):
            assert severity_bin(hb, "M") == severity_bin(hb, "F")

    @pytest.mark.parametrize("sex", ["M", "m", "Male", "male", "MALE"])
    def test_male_detection_is_tolerant(self, sex: str) -> None:
        """``str.upper().startswith("M")`` must accept the spellings the workbook
        actually contains, otherwise a patient is banded as female by accident.

        The real corpus was checked before writing this: both workbooks yield
        exactly ``"M"`` and ``"F"`` with no surrounding whitespace, so no
        stripping is claimed here and none is implemented.
        """
        assert severity_bin(12.5, sex) == "mild"

    def test_unknown_sex_falls_back_to_the_female_threshold(self) -> None:
        assert severity_bin(12.5, "unknown") == "normal"
        assert severity_bin(12.5) == "normal"


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #


class TestPatientLeakage:
    def test_clean_fold_passes(self) -> None:
        fold = Fold(name="ok", train=("a", "b"), val=("c",), test=("d", "e"))
        assert fold.name == "ok"

    def test_train_test_overlap_is_rejected_at_construction(self) -> None:
        with pytest.raises(AssertionError, match="leakage"):
            Fold(name="bad", train=("a", "b"), val=("c",), test=("b",))

    def test_val_test_overlap_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="leakage"):
            Fold(name="bad", train=("a",), val=("c",), test=("c",))

    def test_repeat_inside_one_split_is_rejected(self) -> None:
        """A patient listed twice in train leaks into itself.

        The pairwise overlap check cannot see this, because ``{"a", "a"}`` and
        ``set()`` both collapse to ``{"a"}``. What catches it is comparing the
        sum of the three split sizes against the size of their union. A subject
        contributing two frames is the realistic route to this bug, and the
        dataset does have subjects with more than one image.
        """
        with pytest.raises(AssertionError, match="more than once"):
            _invalid_fold(train=("a", "a"), val=(), test=("b",))

    def test_repeat_is_rejected_even_when_both_copies_stay_in_one_split(self) -> None:
        with pytest.raises(AssertionError, match="more than once"):
            _invalid_fold(train=(), val=("x", "x", "y"), test=("z",))


def _invalid_fold(train: tuple[str, ...], val: tuple[str, ...], test: tuple[str, ...]) -> None:
    """Call the validator on a fold that cannot be built normally.

    ``object.__new__`` bypasses ``__post_init__``, which is the thing under test
    and would otherwise raise before the validator is ever reached.
    """
    fold = object.__new__(Fold)
    object.__setattr__(fold, "name", "invalid")
    object.__setattr__(fold, "train", train)
    object.__setattr__(fold, "val", val)
    object.__setattr__(fold, "test", test)
    validate_no_patient_leakage(fold)
    raise AssertionError("validate_no_patient_leakage accepted a leaking fold")


class TestFileLeakage:
    def test_disjoint_files_pass(self) -> None:
        validate_no_file_leakage(
            {"a": Split.TRAIN, "b": Split.TEST},
            {"a": ["a_1.jpg"], "b": ["b_1.jpg"]},
        )

    def test_shared_file_across_splits_is_rejected(self) -> None:
        """The same frame reachable from two patients in different splits is
        leakage even though the patient ids differ."""
        with pytest.raises(AssertionError, match="file leakage"):
            validate_no_file_leakage(
                {"a": Split.TRAIN, "b": Split.TEST},
                {"a": ["shared.jpg"], "b": ["shared.jpg"]},
            )

    def test_same_file_within_one_split_is_allowed(self) -> None:
        """A subject may contribute several frames, as long as they all land in
        the same split. Rejecting this would be over-strict, not safer."""
        validate_no_file_leakage(
            {"a": Split.TRAIN},
            {"a": ["a_1.jpg", "a_2.jpg"]},
        )

    def test_patient_with_no_files_is_skipped(self) -> None:
        validate_no_file_leakage({"a": Split.TRAIN}, {})


# --------------------------------------------------------------------------- #
# Split construction
# --------------------------------------------------------------------------- #


def _ids(n: int = 60) -> list[str]:
    return [f"p{i:03d}" for i in range(n)]


class TestPatientDisjointKFold:
    def test_five_folds_partition_the_patients(self) -> None:
        ids = _ids(50)
        folds = patient_disjoint_kfold(ids, n_splits=5, seed=42)
        assert len(folds) == 5
        seen: list[str] = []
        for f in folds:
            seen.extend(f.test)
        assert sorted(seen) == sorted(ids)
        assert len(seen) == len(set(seen)), "a patient is in two test folds"

    def test_train_val_test_are_disjoint_within_every_fold(self) -> None:
        for f in patient_disjoint_kfold(_ids(50), n_splits=5, seed=42):
            validate_no_patient_leakage(f)

    def test_same_seed_reproduces_the_same_folds(self) -> None:
        a = patient_disjoint_kfold(_ids(50), seed=7)
        b = patient_disjoint_kfold(_ids(50), seed=7)
        assert [f.test for f in a] == [f.test for f in b]

    def test_different_seed_changes_the_partition(self) -> None:
        a = patient_disjoint_kfold(_ids(50), seed=7)
        b = patient_disjoint_kfold(_ids(50), seed=8)
        assert [f.test for f in a] != [f.test for f in b]

    def test_duplicate_input_ids_are_deduplicated(self) -> None:
        ids = _ids(20) + _ids(20)
        folds = patient_disjoint_kfold(ids, n_splits=5, seed=42)
        total = sum(len(f.test) for f in folds)
        assert total == 20

    def test_val_fraction_is_carved_out_of_train_not_whole_corpus(self) -> None:
        """``val_fraction`` is a fraction of the fold's *remaining* patients.
        Reading it as a fraction of the corpus would silently shrink the
        training set by a factor of ``n_splits``."""
        folds = patient_disjoint_kfold(_ids(50), n_splits=5, seed=42, val_fraction=0.2)
        for f in folds:
            remaining = len(f.train) + len(f.val)
            assert f.val, "validation set must never be empty"
            assert abs(len(f.val) / remaining - 0.2) < 0.06

    def test_too_few_patients_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot build 5 folds"):
            patient_disjoint_kfold(_ids(3), n_splits=5)

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_impossible_val_fraction_is_rejected(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="val_fraction"):
            patient_disjoint_kfold(_ids(50), val_fraction=fraction)


class TestStratifiedAssignments:
    def _assign(self, severity: list[str], val_fraction: float = 0.2) -> dict[str, Split]:
        ids = _ids(len(severity))
        return stratified_assignments(ids, severity, val_fraction=val_fraction, seed=42)

    def test_every_patient_lands_in_exactly_one_split(self) -> None:
        sev = ["normal"] * 40 + ["mild"] * 12 + ["moderate"] * 4 + ["severe"] * 2
        assign = self._assign(sev)
        assert len(assign) == len(sev)
        assert set(assign.values()) <= {Split.TRAIN, Split.VAL, Split.TEST}

    def test_every_split_is_populated(self) -> None:
        sev = ["normal"] * 40 + ["mild"] * 12
        assign = self._assign(sev)
        for split in (Split.TRAIN, Split.VAL, Split.TEST):
            assert sum(1 for v in assign.values() if v == split) > 0

    def test_every_holdout_split_receives_an_anaemic_patient(self) -> None:
        """The reason this function exists rather than a plain shuffle.

        With 82 of 100 patients non-anaemic, a random split can hand the test
        fold two anaemic patients -- sometimes none -- which makes a sensitivity
        claim uncomputable. Round-robin over severity strata is what guarantees
        both holdout splits see the rare bands.
        """
        sev = ["normal"] * 82 + ["mild"] * 10 + ["moderate"] * 5 + ["severe"] * 3
        ids = _ids(len(sev))
        assign = stratified_assignments(ids, sev, val_fraction=0.2, seed=42)
        label = dict(zip(ids, sev, strict=True))

        for split in (Split.VAL, Split.TEST):
            members = [p for p, s in assign.items() if s == split]
            assert members, f"{split} came out empty"
            anaemic = [p for p in members if label[p] != "normal"]
            assert anaemic, (
                f"{split} contains no anaemic patient, so sensitivity cannot be computed on it"
            )

    def test_same_seed_reproduces_the_split(self) -> None:
        sev = ["normal"] * 30 + ["mild"] * 10
        assert self._assign(sev) == self._assign(sev)

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="severity labels"):
            stratified_assignments(_ids(10), ["normal"] * 9)

    def test_accepts_a_numpy_array_of_labels(self) -> None:
        sev = np.array(["normal"] * 30 + ["mild"] * 10)
        assign = stratified_assignments(_ids(40), sev, seed=42)
        assert len(assign) == 40


# --------------------------------------------------------------------------- #
# Cross-site holdout
# --------------------------------------------------------------------------- #


class TestSiteHoldout:
    def _sites(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(10):
            out[f"India/{i}"] = "India"
        for i in range(10):
            out[f"Italy/{i}"] = "Italy"
        return out

    def test_yields_both_directions(self) -> None:
        site_of = self._sites()
        ids = list(site_of)
        names = [f.name for f in site_holdout_folds(ids, site_of)]
        assert names == ["India->Italy", "Italy->India"]

    def test_train_and_test_sites_are_never_mixed(self) -> None:
        site_of = self._sites()
        for fold in site_holdout_folds(list(site_of), site_of):
            train_site, test_site = fold.name.split("->")
            assert all(site_of[p] == train_site for p in fold.train)
            assert all(site_of[p] == test_site for p in fold.test)

    def test_val_is_empty_because_there_is_no_third_site(self) -> None:
        """Honest external validation has nowhere to early-stop. Pretending
        otherwise would mean selecting on the test site."""
        for fold in site_holdout_folds(list(self._sites()), self._sites()):
            assert fold.val == ()

    def test_unlabelled_patient_is_rejected(self) -> None:
        site_of = self._sites()
        del site_of["India/3"]
        with pytest.raises(ValueError, match="no site label"):
            list(site_holdout_folds([*site_of, "India/3"], site_of))

    def test_blank_site_label_is_rejected(self) -> None:
        """A patient mapped to an empty string is in the mapping, so a bare
        ``p not in site_of`` test waves it through. It then matches neither
        direction's ``site_of.get(p) == ...`` and vanishes from the holdout
        entirely, shrinking the reported cross-site population without any
        error. Found by this test; fixed in ``site_holdout_folds``."""
        site_of = self._sites()
        site_of["India/3"] = "   "
        with pytest.raises(ValueError, match="blank site label"):
            list(site_holdout_folds(list(site_of), site_of))

    def test_every_patient_reaches_exactly_one_fold_in_each_direction(self) -> None:
        """The guarantee the blank-label fix exists to protect: no patient may
        be silently absent from the cross-site evaluation."""
        site_of = self._sites()
        ids = list(site_of)
        for fold in site_holdout_folds(ids, site_of):
            _train_site, test_site = fold.name.split("->")
            expected = {p for p in ids if site_of[p] == test_site}
            assert set(fold.test) == expected

    def test_single_site_is_rejected(self) -> None:
        site_of = {f"India/{i}": "India" for i in range(5)}
        with pytest.raises(ValueError, match=">= 2 sites"):
            list(site_holdout_folds(list(site_of), site_of))

    def test_sites_constant_matches_the_real_corpus(self) -> None:
        assert SITES == ("India", "Italy")


# --------------------------------------------------------------------------- #
# Split enum
# --------------------------------------------------------------------------- #


class TestSplitEnum:
    def test_round_trips_through_its_string_value(self) -> None:
        """Used for the CSV columns, so ``Split("train") is Split.TRAIN`` has to hold."""
        for split in Split:
            assert Split(split.value) is split

    def test_assignments_covers_train_val_test(self) -> None:
        fold = Fold(name="f", train=("a", "b"), val=("c",), test=("d",))
        assign = fold.assignments()
        assert assign == {
            "a": Split.TRAIN,
            "b": Split.TRAIN,
            "c": Split.VAL,
            "d": Split.TEST,
        }
