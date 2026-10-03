from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hemolux.data.dataset import MASK_SUFFIX_ORDER
from hemolux.data.splits import Fold, severity_bin
from hemolux.metrics.regression import regression_report
from hemolux.sweep import (
    Candidate,
    SweepRow,
    default_grid,
    finish_sweep,
    select_best,
    sweep_grid,
    sweep_label,
)
from hemolux.training import (
    FeatureSet,
    TrainConfig,
    evaluate_fold,
    score_test_fold,
    validation_report,
)

DIM = 4

FOLD = Fold(
    name="sweep0",
    train=("India/1", "Italy/3", "India/5", "Italy/7"),
    val=("India/2", "Italy/8"),
    test=("Italy/4", "India/6"),
)

CFG = TrainConfig(epochs=4)


def make_features(
    ids: tuple[str, ...],
    hb: tuple[float, ...],
    *,
    site: tuple[str, ...] | None = None,
    seed: int = 0,
) -> FeatureSet:
    """A FeatureSet whose features carry a real signal, so selection has something to find.

    The training tests use an all-zero matrix because they never exercise the network.
    A sweep has to: if every candidate scored identically the selection logic would be
    untested and any winner would look as good as any other. The signal is planted so
    that the patient whose feature is higher has the higher Hb, which means a head fit
    on these features can actually do better than one fit on noise.
    """
    rng = np.random.default_rng(seed)
    signal = np.linspace(0.0, 1.0, len(ids))
    feats = rng.normal(scale=0.05, size=(len(ids), DIM)) + signal[:, None] * 4.0
    sex = ("M",) * len(ids)
    return FeatureSet(
        patient_ids=ids,
        features=feats,
        hb=np.array(hb, dtype=np.float64),
        site=site or tuple("India" for _ in ids),
        sex=sex,
        age=np.full(len(ids), 30.0),
        roi="palpebral",
        backbone="fake_backbone",
        feature_dim=DIM,
        severity=tuple(severity_bin(h, s) for h, s in zip(hb, sex, strict=True)),
    )


#: Two patients per split, each group with a different feature offset so the candidates
#: are genuinely distinguishable rather than differing only in label.
CLEAN = make_features(
    ("India/1", "India/2", "Italy/3", "Italy/4", "India/5", "India/6", "Italy/7", "Italy/8"),
    (15.0, 14.0, 10.5, 9.5, 8.0, 6.0, 12.5, 13.5),
    site=("India", "India", "Italy", "Italy", "India", "India", "Italy", "Italy"),
    seed=1,
)


def row_for(candidate: Candidate, features: FeatureSet = CLEAN, **kw) -> SweepRow:
    """One fitted row, using ``kw`` to override the validation metric under test."""
    fit = evaluate_fold(
        "regression",
        FOLD,
        features,
        cfg=CFG,
        verbose=False,
        save_checkpoint=False,
        score_test=False,
    )
    return SweepRow(
        candidate=candidate,
        val_report=kw.get("val_report", validation_report(fit)),
        fit=fit,
        features=features,
    )


#: Four patients spanning a realistic Hb range, so variance is not degenerate.
TRUTH = np.array([10.0, 11.0, 12.0, 13.0])


def report_from(errors: np.ndarray):
    """A RegressionReport over a chosen error vector, for testing the ranking alone.

    Takes the errors rather than a target MAE because MAE and R^2 do not order
    candidates the same way, and a sweep supports selecting on either. A constant
    offset beats a spiky one on R^2 at equal MAE -- ``[2.5, 2.5, 2.5, 2.5]`` has no
    error variance while ``[10, 0, 0, 0]`` has the same mean absolute error and far
    worse -- so a helper that took an MAE could not produce a case where the two
    metrics disagree, which is the case that matters.

    Built from two real series rather than a stand-in object, so the tests assert
    against the same class the sweep holds.
    """
    return regression_report(TRUTH, TRUTH + np.asarray(errors, dtype=np.float64))


def flat(mae: float):
    """Equal error on every patient: MAE == mae, and R^2 as good as that MAE allows."""
    return report_from(np.full(TRUTH.size, mae))


def undefined(mae: float):
    """A report whose metric is NaN, as a configuration that failed to train produces.

    Patched onto a real report with ``replace`` rather than computed, because
    :func:`regression_report` refuses fewer than two finite pairs -- it raises rather
    than returning NaNs. Which means a genuinely broken configuration surfaces as an
    *exception* during extraction or training, not as a NaN row, and the sweep's
    handling of a non-finite metric is a second line of defence rather than the
    expected path. It still has to behave, because a metric can be finite at
    extraction and go non-finite later.
    """
    return replace(flat(mae), mae=float("nan"), r2=float("nan"))


# --------------------------------------------------------------------------- #
# Candidate: refusing configurations that describe nothing
# --------------------------------------------------------------------------- #


def test_deep_features_cannot_ask_for_a_removed_illuminant() -> None:
    """A frozen trunk reads the frame as captured; `balance="raw"` says nothing.

    Refused rather than coerced. Silently turning it into `balanced` would let an
    artefact record a normalisation decision that was never a choice, and a sweep
    comparing "deep raw" against "colour raw" would be reporting on a variable that
    only one of the two arms has.
    """
    with pytest.raises(ValueError, match="no balance setting to vary"):
        Candidate(kind="deep", roi="palpebral", balance="raw")

    # The default is accepted, and means "no correction applied".
    assert Candidate(kind="deep", roi="palpebral").balance == "balanced"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("kind", "spectral", "unknown feature kind"),
        ("roi", "caruncle", "unknown roi"),
        ("balance", "auto", "unknown balance"),
    ],
)
def test_an_unknown_setting_is_named_rather_than_silently_accepted(
    field: str, value: str, match: str
) -> None:
    kwargs = {"kind": "colour", "roi": "palpebral", "balance": "raw", field: value}
    with pytest.raises(ValueError, match=match):
        Candidate(**kwargs)


def test_the_cache_key_ignores_the_head() -> None:
    """The head is trained on features and does not change them.

    Folding it into the key would make the same pixels get extracted once per head,
    which on this corpus means another pass over 217 twelve-megapixel frames.
    """
    a = Candidate(kind="colour", roi="palpebral", balance="raw", head="regression")
    b = Candidate(kind="colour", roi="palpebral", balance="raw", head="ordinal")

    assert a.cache_key() == b.cache_key()
    assert a != b, "the candidates must still be distinguishable"


def test_the_label_does_not_advertise_a_varied_setting_that_was_fixed() -> None:
    """Deep rows drop the balance word, because no deep row varies it."""
    deep = Candidate(kind="deep", roi="palpebral")
    colour = Candidate(kind="colour", roi="palpebral", balance="raw")

    assert sweep_label(deep) == "deep / palpebral"
    assert sweep_label(colour) == "colour / palpebral / raw"
    assert deep.label == sweep_label(deep)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_selection_minimises_validation_mae() -> None:
    rows = [
        row_for(Candidate("colour", "palpebral", "raw"), val_report=flat(1.0)),
        row_for(Candidate("colour", "forniceal", "raw"), val_report=flat(0.5)),
        row_for(Candidate("deep", "palpebral"), val_report=flat(1.4)),
    ]

    best = select_best(rows)

    assert best.label == "colour / forniceal / raw"
    assert best.score("mae") == pytest.approx(0.5, abs=1e-9)


def test_selection_can_maximise_r2_instead() -> None:
    """The direction is explicit, not inferred from a sign convention.

    The fixture is built so the two metrics genuinely disagree, because a pair that
    agrees cannot tell a correct direction from a leaked one. Concentrated errors hold
    the mean absolute error down -- one large miss among four averages to a quarter of
    its size -- while carrying all of their variance, so they lose on R^2. Ranking on
    the wrong direction here returns the other candidate, which a same-direction fixture
    would have hidden.
    """
    concentrated = report_from(np.array([3.9, 0.0, 0.0, 0.0]))
    spread = report_from(np.ones(4))
    rows = [
        row_for(Candidate("colour", "palpebral", "raw"), val_report=concentrated),
        row_for(Candidate("colour", "forniceal", "raw"), val_report=spread),
    ]

    assert concentrated.mae < spread.mae, "the fixture must not agree on MAE"
    assert concentrated.r2 < spread.r2, "the fixture must agree on R^2"
    assert select_best(rows, "mae").label == "colour / palpebral / raw"
    assert select_best(rows, "r2").label == "colour / forniceal / raw"


def test_a_tie_is_broken_by_label_and_not_by_whichever_came_first() -> None:
    """Otherwise the published configuration is a function of the grid's ordering.

    Both candidates here validate identically. Which one is published should depend on
    the evidence and on a stated rule -- not on the order the loop happened to visit
    them, which is an implementation detail that a reordered grid would silently change.
    """
    shared = flat(0.5)
    ordered = [
        row_for(Candidate("deep", "forniceal"), val_report=shared),
        row_for(Candidate("colour", "palpebral", "raw"), val_report=shared),
    ]
    reversed_grid = list(reversed(ordered))

    assert select_best(ordered).label == select_best(reversed_grid).label
    assert select_best(ordered).label == "colour / palpebral / raw"


def test_a_candidate_that_did_not_train_loses_to_one_that_did() -> None:
    """A NaN metric is not a good score; it is the absence of one.

    The row is kept rather than dropped, so "this configuration did not train" stays
    visible in the table instead of making the grid look smaller than it is.
    """
    broken = row_for(Candidate("hybrid", "forniceal", "raw"), val_report=undefined(1.0))
    working = row_for(Candidate("colour", "palpebral", "raw"), val_report=flat(2.0))

    assert select_best([broken, working]).label == "colour / palpebral / raw"
    assert select_best([working, broken]).label == "colour / palpebral / raw"


def test_a_grid_where_nothing_trained_says_so_instead_of_selecting_anyway() -> None:
    """Every configuration failing is a different problem from one failing to beat the rest."""
    rows = [
        row_for(Candidate("colour", roi, bal), val_report=undefined(1.0))
        for roi in ("palpebral", "forniceal")
        for bal in ("raw", "balanced")
    ]

    with pytest.raises(ValueError, match="no candidate produced a finite"):
        select_best(rows)


def test_an_empty_grid_is_an_error_rather_than_a_winner() -> None:
    with pytest.raises(ValueError, match="no candidates"):
        select_best([])


def test_the_selection_gap_reports_how_far_ahead_the_winner_was() -> None:
    """A validation winner among near-ties is not a finding, and the table must show it.

    Both numbers are in the same units as the metric, so a small gap is directly
    comparable to the clinical acceptability target of +/-1.0 g/dL rather than needing
    to be interpreted.
    """
    rows = [
        row_for(Candidate("colour", "palpebral", "raw"), val_report=flat(1.00)),
        row_for(Candidate("colour", "forniceal", "raw"), val_report=flat(1.02)),
        row_for(Candidate("deep", "palpebral"), val_report=flat(1.90)),
    ]
    best = select_best(rows)
    result = finish_sweep(rows, FOLD)

    assert result.winner.label == best.label
    assert result.selection_gap == pytest.approx(0.02, abs=1e-9)


# --------------------------------------------------------------------------- #
# The guarantee: one test forward pass, at the end
# --------------------------------------------------------------------------- #


def test_the_grid_produces_no_test_predictions_for_any_candidate() -> None:
    """Every candidate is fitted with the test fold untouched.

    Not "the winner's report looks fine" -- every fit's, because a single scored
    candidate is enough to have chosen on.
    """
    candidates = [
        Candidate("colour", "palpebral", "raw"),
        Candidate("colour", "palpebral", "balanced"),
        Candidate("deep", "palpebral"),
    ]

    rows = sweep_grid(
        candidates,
        FOLD,
        {
            "colour/palpebral/raw": CLEAN,
            "colour/palpebral/balanced": CLEAN,
            "deep/palpebral/balanced": CLEAN,
        },
        cfg=CFG,
        verbose=False,
    )

    assert len(rows) == 3
    for row in rows:
        assert row.fit.report is None, f"{row.label} carries test metrics"
        assert row.fit.test_hb_pred.size == 0, f"{row.label} produced test predictions"
        assert np.isfinite(row.val_report.mae)


def test_finishing_the_sweep_scores_exactly_one_test_fold() -> None:
    """The 43 test patients influence one number, and it is the one that is published.

    ``score_test_fold`` returns a new fit rather than mutating the winner's, so the
    stored fits stay unscored even after the winner has been measured. That is worth
    pinning: it means the losing candidates' rows are not one call away from holding
    test numbers, and a later reader of the table cannot tell which entries were
    inspected during the search.
    """
    candidates = [Candidate("colour", "palpebral", "raw"), Candidate("deep", "palpebral")]
    keys = {c.cache_key(): CLEAN for c in candidates}

    rows = sweep_grid(candidates, FOLD, keys, cfg=CFG, verbose=False)
    result = finish_sweep(rows, FOLD)

    assert result.test_report is not None
    assert result.test_report.n == len(FOLD.test)
    assert not [r for r in rows if r.fit.report is not None], "a stored fit was scored in place"

    # And the published number belongs to the winner, not to whichever row was last.
    rescored = score_test_fold(result.winner.fit, result.winner.features, FOLD)
    assert rescored.report is not None
    assert rescored.report.mae == pytest.approx(result.test_report.mae)


def test_the_sweep_refuses_to_run_on_features_it_was_not_given() -> None:
    """Selection between configurations must not quietly become building them.

    Extraction is left to the caller so it stays cacheable and interruptible -- a sweep
    that rebuilt its own features would be un-interruptible and would re-extract on
    every run.
    """
    candidates = [Candidate("colour", "palpebral", "raw")]

    with pytest.raises(KeyError, match="no features extracted"):
        sweep_grid(candidates, FOLD, {}, cfg=CFG, verbose=False)


def test_the_grid_covers_every_feature_kind_at_every_roi() -> None:
    """A grid missing a cell is not a sweep; it is an experiment.

    Checked structurally because the failure is invisible in the results: a table with
    one missing row reads as a table where that configuration simply lost.
    """
    grid = default_grid()

    for roi in MASK_SUFFIX_ORDER:
        kinds = {c.kind for c in grid if c.roi == roi}
        assert kinds == {"deep", "colour", "hybrid"}, f"{roi} is missing {kinds}"
        for kind in ("colour", "hybrid"):
            balances = {c.balance for c in grid if c.roi == roi and c.kind == kind}
            assert balances == {"raw", "balanced"}, f"{roi}/{kind} varies {balances}"
    assert len(grid) == 15, "3 kinds x 3 rois, with deep unvaried and the rest twice"
