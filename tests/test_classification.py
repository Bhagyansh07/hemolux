"""Thresholded screening metrics at the WHO cutoffs.

Every confusion cell is asserted, not just the rate derived from it. A rate with
the wrong denominator is the failure this module exists to prevent, and a test
that only checked sensitivity would not catch it: ``tp / (tp + fn)`` computed
against ``tp + fp`` gives a plausible number that is not sensitivity.

The two verdicts are pinned separately because they mean different things and a
merged "not enough data" would lose the distinction that matters: ``underpowered``
is about the sample, ``not_computable`` is about a class that is absent. The real
corpus has 217 patients and zero of them below 7 g/dL, so the severe band is the
second, not the first, and a pooled average would hide it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemolux.config import WHO_CUTOFF_FEMALE, WHO_CUTOFF_MALE, WHO_MODERATE, WHO_SEVERE
from hemolux.metrics.classification import (
    MIN_N,
    SCREENING_BANDS,
    by_group,
    screening_metrics,
    screening_report,
    threshold_metrics,
    who_cutoff,
)


def test_lower_haemoglobin_is_the_positive_class_on_both_sides() -> None:
    """Truth positive = 10, 11; model positive = 10.5, 12.5.

    Hand-worked: one true positive, one false negative, one false positive, one
    true negative.
    """
    metrics = threshold_metrics(
        np.array([10.0, 11.0, 13.0, 14.0]),
        np.array([10.5, 12.5, 11.9, 14.0]),
        12.0,
    )

    assert (metrics.tp, metrics.fn, metrics.fp, metrics.tn) == (1, 1, 1, 1)
    assert metrics.positives == 2
    assert metrics.sensitivity == pytest.approx(0.5)
    assert metrics.specificity == pytest.approx(0.5)
    assert metrics.ppv == pytest.approx(0.5)
    assert metrics.npv == pytest.approx(0.5)


def test_the_cutoff_is_strict_so_a_reading_exactly_at_it_is_negative() -> None:
    """<, not <=, matching every other WHO comparison in the project."""
    metrics = threshold_metrics(np.array([12.0]), np.array([12.0]), 12.0)

    assert metrics.positives == 0
    assert metrics.tn == 1


def test_sex_specific_cutoffs_differ_by_one_gram() -> None:
    cut = who_cutoff(["M", "F", "male", "female", "unknown"])

    assert cut.tolist() == [
        WHO_CUTOFF_MALE,
        WHO_CUTOFF_FEMALE,
        WHO_CUTOFF_MALE,
        WHO_CUTOFF_FEMALE,
        WHO_CUTOFF_FEMALE,
    ]


def test_a_man_at_twelve_and_a_half_is_anaemic_and_a_woman_is_not() -> None:
    """The exact case a single pooled cutoff over-calls.

    12.5 g/dL is below the male cutoff of 13 and above the female cutoff of 12,
    so the same reading is a positive for one patient and a negative for another.
    """
    truth = np.array([12.5])
    pred = np.array([12.5])

    assert threshold_metrics(truth, pred, who_cutoff(["M"])).positives == 1
    assert threshold_metrics(truth, pred, who_cutoff(["F"])).positives == 0


def test_all_three_bands_are_reported() -> None:
    truth = np.array([6.0, 8.0, 11.0, 14.0] * 6)
    pred = np.array([6.5, 8.5, 11.0, 15.0] * 6)
    sex = ["M", "F", "M", "F"] * 6

    out = screening_metrics(truth, pred, sex)

    assert set(out) == {band for band, _ in SCREENING_BANDS}
    assert out["anaemia"].threshold is None, "a sex-specific cutoff has no single value"
    assert out["moderate_or_worse"].threshold == WHO_MODERATE
    assert out["severe"].threshold == WHO_SEVERE


def test_small_n_is_underpowered_and_carries_the_count() -> None:
    metrics = threshold_metrics(np.arange(10, dtype=float), np.arange(10, dtype=float), 12.0)

    assert metrics.n == 10
    assert metrics.verdict == "underpowered"


def test_the_power_floor_matches_the_fairness_gate() -> None:
    """One floor, one meaning. A different number here would contradict C3."""
    assert MIN_N == 20


def test_zero_positives_is_not_computable_not_a_missed_detection() -> None:
    """The severe band on this corpus: 40 patients, none below 7 g/dL.

    Sensitivity is undefined. Returning 0.0 would read as total failure; the
    truth is that it was never measurable, and the corpus cannot supply it.
    """
    truth = np.full(40, 14.0)
    pred = np.full(40, 14.0)

    metrics = threshold_metrics(truth, pred, WHO_SEVERE)

    assert metrics.n == 40
    assert metrics.positives == 0
    assert metrics.verdict == "not_computable"
    assert np.isnan(metrics.sensitivity)
    assert metrics.specificity == pytest.approx(1.0)


def test_an_empty_cell_is_not_computable() -> None:
    metrics = threshold_metrics(np.array([]), np.array([]), 12.0)

    assert metrics.n == 0
    assert metrics.verdict == "not_computable"


def test_nan_pairs_are_dropped_before_scoring() -> None:
    """An abstained or failed row must not be counted as a negative."""
    metrics = threshold_metrics(np.array([10.0, np.nan, 14.0]), np.array([10.0, 12.0, 14.0]), 12.0)

    assert metrics.n == 2


def test_shapes_must_agree() -> None:
    with pytest.raises(ValueError):
        threshold_metrics(np.ones(3), np.ones(4), 12.0)


def test_a_per_patient_threshold_must_match_the_cohort() -> None:
    with pytest.raises(ValueError):
        threshold_metrics(np.ones(3), np.ones(3), np.ones(4))


def test_sex_must_match_the_cohort() -> None:
    """A short sex column would apply the wrong cutoff to the wrong patient."""
    with pytest.raises(ValueError):
        screening_metrics(np.ones(3), np.ones(3), ["M", "F"])


def test_grouping_is_stable_and_per_site() -> None:
    truth = np.array([10.0] * 25 + [14.0] * 25)
    pred = np.array([10.0] * 25 + [14.0] * 25)
    sex = ["M", "F"] * 25
    site = np.array(["Italy"] * 25 + ["India"] * 25, dtype=object)

    groups = by_group(truth, pred, sex, site)

    assert list(groups) == ["India", "Italy"], "groups are sorted for a stable diff"
    assert groups["India"]["anaemia"].n == 25


def test_group_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        by_group(np.ones(4), np.ones(4), ["M"] * 4, np.array(["a", "b"]))


def test_report_carries_bands_pooled_and_per_site() -> None:
    truth = np.array([10.0, 14.0] * 11)
    pred = np.array([10.0, 14.0] * 11)
    sex = ["M", "F"] * 11
    site = np.array((["India"] * 11) + (["Italy"] * 11), dtype=object)

    report = screening_report(truth, pred, sex, site)

    assert report["bands"] == ["anaemia", "moderate_or_worse", "severe"]
    assert set(report["pooled"]) == set(report["bands"])
    assert set(report["per_site"]) == {"India", "Italy"}
    assert report["pooled"]["severe"]["verdict"] == "not_computable"
