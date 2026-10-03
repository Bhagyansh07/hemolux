"""Configuration selection on validation, with the test fold read exactly once.

The finding this module exists to make defensible is that on this corpus the
*illumination* matters more than the network: dividing out an estimated illuminant
before measuring colour moved R^2 from +0.109 to +0.545, better than a MobileNetV3+SE
trained end to end in 2022. That number came out of a sweep that compared candidates
on **test**, which makes it a description of the search rather than a measurement of
anything -- the winner is whichever configuration happened to suit 43 particular
patients.

So the sweep is run the other way round:

1. Every candidate is fitted with ``score_test=False``. No test prediction is ever
   produced, so there is nothing to have selected on even accidentally.
2. The winner is chosen on :func:`~hemolux.training.validation_report`.
3. Only then is the winner's stored weights scored on test, once.

Step 3 does not retrain. With a stochastic optimiser a second run at the same seed is
a different model wearing the same configuration, so the published number would not
describe the artefact that ships.

Two things are deliberately absent. There is no early stopping on the test fold and no
discarding of a candidate for looking bad on it -- a sweep that does either has used the
test set, whatever the result is called. And selection is on a single fold rather than a
repeated cross-validation, because this corpus has 217 patients at two sites: repeated
CV would place the same patients in a test fold five times, which measures the split as
much as the model. That is a limitation of the design rather than a shortcut, and
:attr:`SweepResult.selection_gap` exists so a near-tie cannot be reported as a finding.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from hemolux.data.dataset import MASK_SUFFIX_ORDER
from hemolux.training import (
    BALANCE_KINDS,
    FEATURE_KINDS,
    FeatureSet,
    Fold,
    HeadFit,
    TrainConfig,
    _rounded,
    _write_csv,
    ensure_dirs,
    evaluate_fold,
    score_test_fold,
    validation_report,
)

if TYPE_CHECKING:  # pragma: no cover - only a type checker sees this import
    from hemolux.metrics.regression import RegressionReport

__all__ = [
    "Candidate",
    "SweepResult",
    "SweepRow",
    "build_sweep_report",
    "default_grid",
    "finish_sweep",
    "select_best",
    "sweep_grid",
    "sweep_label",
]


#: Ranked on validation MAE rather than R^2. MAE is what a screening tool is judged on:
#: a miss of 0.5 g/dL is a different kind of miss from one of 3.0, and R^2 scores them
#: as equally bad. R^2 is recorded on every row regardless, because it is the number
#: the literature reports and a sweep that omitted it would invite the question of what
#: was being hidden.
SelectionMetric = Literal["mae", "r2"]

#: ``mae`` is minimised, ``r2`` is maximised. Written out rather than inferred from a
#: sign convention, so that adding a third metric has to be a deliberate act instead of
#: an accident of ordering.
_LOWER_IS_BETTER: dict[str, bool] = {"mae": True, "r2": False}


@dataclass(frozen=True)
class Candidate:
    """One point in the configuration space."""

    kind: str
    roi: str
    #: Defaults to the correction being applied, which is also the default on the
    #: command line. For ``deep`` it means "no correction applies" -- see
    #: :meth:`__post_init__`.
    balance: str = "balanced"
    head: str = "regression"

    def __post_init__(self) -> None:
        if self.kind not in FEATURE_KINDS:
            raise ValueError(
                f"unknown feature kind {self.kind!r}; expected one of {list(FEATURE_KINDS)}"
            )
        if self.roi not in MASK_SUFFIX_ORDER:
            raise ValueError(f"unknown roi {self.roi!r}; expected one of {list(MASK_SUFFIX_ORDER)}")
        if self.balance not in BALANCE_KINDS:
            raise ValueError(
                f"unknown balance {self.balance!r}; expected one of {list(BALANCE_KINDS)}"
            )
        # `deep` has no illuminant to remove: a frozen trunk reads the frame as
        # captured. Carrying a raw/balanced choice on it would imply the correction
        # applies when it cannot, and the artefact would claim a normalisation that
        # never happened. Refused rather than silently coerced, because a caller who
        # wrote `balance="raw"` for a deep candidate has a model in mind that does not
        # exist.
        if self.kind == "deep" and self.balance != "balanced":
            raise ValueError(
                f"deep features have no balance setting to vary, so balance="
                f"{self.balance!r} with kind='deep' describes nothing. Use the default "
                f"'balanced', which simply means 'no correction applied'."
            )

    @property
    def label(self) -> str:
        return sweep_label(self)

    def cache_key(self) -> str:
        """Identity for matching a cached FeatureSet, deliberately excluding the head.

        The head is trained *on* features and has no effect on them, so two candidates
        differing only in head share one extraction. Folding the head into the key would
        extract the same pixels twice.
        """
        return f"{self.kind}/{self.roi}/{self.balance}"


def sweep_label(candidate: Candidate) -> str:
    """A short stable name, used as the results-table key and as the tie-break.

    The balance word is dropped where it carries no information, so deep rows read
    ``deep / palpebral`` rather than ``deep / palpebral / balanced``. Printing a second
    variable that was held constant would suggest the grid varied it when it did not.
    """
    if candidate.kind == "deep":
        return f"deep / {candidate.roi}"
    return f"{candidate.kind} / {candidate.roi} / {candidate.balance}"


@dataclass
class SweepRow:
    """One candidate's outcome: validation metrics, and the fit that produced them.

    The fit is carried rather than discarded so the winner can be scored afterwards
    without retraining. It holds no test predictions -- ``score_test=False`` guarantees
    that -- so holding it cannot leak anything.
    """

    candidate: Candidate
    val_report: RegressionReport
    fit: HeadFit
    features: FeatureSet

    @property
    def label(self) -> str:
        return self.candidate.label

    def score(self, metric: SelectionMetric) -> float:
        return float(getattr(self.val_report, metric))


@dataclass
class SweepResult:
    """Every candidate, the one chosen, and the winner's single test report."""

    rows: list[SweepRow]
    winner: SweepRow
    metric: SelectionMetric
    test_report: RegressionReport | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def selection_gap(self) -> float:
        """How far ahead the winner was on validation, in the metric's own units.

        Reported because a validation winner among near-ties is not a finding. When
        this is small next to the spread of the column, the sweep has not established a
        preference and the test number is one draw from several configurations that are
        about equally good -- which is a different claim and should not be written up as
        if it were the same one.
        """
        others = [r.score(self.metric) for r in self.rows if r is not self.winner]
        if not others:
            return float("nan")
        rest = min(others) if _LOWER_IS_BETTER[self.metric] else max(others)
        return abs(self.winner.score(self.metric) - rest)


def select_best(rows: Sequence[SweepRow], metric: SelectionMetric = "mae") -> SweepRow:
    """Pick the best candidate on validation, breaking ties deterministically.

    Ties break on the label, ascending. Two configurations that validate equally well
    are not equally trustworthy, and resolving that by dictionary order or by whichever
    the loop visited last would make the published configuration a function of the
    grid's ordering rather than of the evidence. Sorting by label means the same corpus
    and grid always yield the same winner.

    A non-finite validation metric loses to any finite one. The candidate is kept in the
    table rather than dropped, because "this configuration did not train" is itself a
    result and silently omitting it would make the grid look smaller than it is.
    """
    if not rows:
        raise ValueError("no candidates to select from")
    finite = [r for r in rows if math.isfinite(r.score(metric))]
    if not finite:
        raise ValueError(
            f"no candidate produced a finite validation {metric}: every configuration "
            f"failed to train. That is a different problem from one that trained badly, "
            f"and it is worth reading off the table rather than selecting around."
        )
    pick = min if _LOWER_IS_BETTER[metric] else max
    best = pick(finite, key=lambda r: (r.score(metric), r.label))
    return best


def default_grid(head: str = "regression") -> list[Candidate]:
    """The grid issue #2 asks for.

    Three feature kinds across three ROIs, with the colour and hybrid kinds each
    measured both balanced and raw. Deep appears once per ROI because it has no
    illuminant setting to vary.

    Including ``deep`` is the point of the exercise rather than a control: the question
    is whether a frozen ImageNet trunk earns its 1024 columns on this corpus, and the
    only honest way to answer that is to run it in the same sweep, on the same split,
    against the same selection rule as the alternatives. Comparing across two separate
    experiments would confound the architecture with the split.
    """
    out: list[Candidate] = []
    for roi in MASK_SUFFIX_ORDER:
        out.append(Candidate(kind="deep", roi=roi, balance="balanced", head=head))
        for kind in ("colour", "hybrid"):
            for balance in BALANCE_KINDS:
                out.append(Candidate(kind=kind, roi=roi, balance=balance, head=head))
    return out


def sweep_grid(
    candidates: Sequence[Candidate],
    fold: Fold,
    features_by_key: dict[str, FeatureSet],
    *,
    cfg: TrainConfig,
    verbose: bool = True,
) -> tuple[SweepRow, ...]:
    """Fit every candidate with the test fold untouched.

    ``features_by_key`` maps :meth:`Candidate.cache_key` to an extracted FeatureSet, so
    the caller owns extraction -- and therefore the cache -- while this function owns
    only the selection. Nothing here can read the test fold: every fit is called with
    ``score_test=False``, and each row carries validation metrics exclusively.
    """
    rows: list[SweepRow] = []
    for i, candidate in enumerate(candidates, start=1):
        features = features_by_key.get(candidate.cache_key())
        if features is None:
            raise KeyError(
                f"no features extracted for {candidate.cache_key()!r}. The sweep selects "
                f"between configurations and does not build them, so extraction stays "
                f"cacheable and interruptible. Extract this one first."
            )
        if verbose:
            print(
                f"  [{i}/{len(candidates)}] {candidate.label:<32} "
                f"dim={features.feature_dim:<5} fitting",
                flush=True,
            )

        fit = evaluate_fold(
            candidate.head,
            fold,
            features,
            cfg=cfg,
            verbose=False,
            # A sweep writes one checkpoint at the end, for the winner. Twelve here
            # would leave eleven files that look equally deployable.
            save_checkpoint=False,
            score_test=False,
        )
        # Asserted rather than assumed. This is the load-bearing property of the whole
        # module, and a future edit that dropped the flag would otherwise show up as a
        # sweep that still produced a plausible winner.
        assert fit.report is None, "score_test=False must leave the test fold unscored"

        row = SweepRow(
            candidate=candidate,
            val_report=validation_report(fit),
            fit=fit,
            features=features,
        )
        rows.append(row)

        if verbose:
            print(
                f"        val mae {row.val_report.mae:.4f}   r2 {row.val_report.r2:+.4f}   "
                f"epoch {fit.best_epoch}   {fit.train_seconds:.1f}s",
                flush=True,
            )
    return tuple(rows)


def finish_sweep(
    rows: Sequence[SweepRow],
    fold: Fold,
    *,
    metric: SelectionMetric = "mae",
) -> SweepResult:
    """Select on validation, then score the winner's test fold exactly once.

    The test forward pass happens here and nowhere else. Every candidate's fit still
    carries no test predictions, so a reader can confirm that the 43 test patients
    influenced one number in this report and not the choice of which number.
    """
    winner = select_best(rows, metric)
    scored = score_test_fold(winner.fit, winner.features, fold)
    return SweepResult(
        rows=list(rows),
        winner=winner,
        metric=metric,
        test_report=scored.report,
        notes=list(scored.notes),
    )


def build_sweep_report(
    result: SweepResult,
    fold: Fold,
    cfg: TrainConfig,
    *,
    provenance: dict | None = None,
) -> dict:
    """Write ``sweep.json`` and ``sweep.csv``, and return the payload.

    Aggregate only -- one row per *candidate*, never per patient. The single-split
    report next to it does carry patient-level rows in ``predictions.csv``, and that
    file is kept out of the repository for exactly this reason; a sweep that wrote its
    own patient table would need the same treatment, and there is nothing in a sweep
    report that needs one. The validation split is 43 patients and they are the same 43
    every time, so a per-row file would add no information a reader does not already
    have from the fold sizes.
    """
    # Read through the training module at call time rather than binding the constant at
    # import time. The tests redirect ``hemolux.training.ARTIFACT_REPORTS`` to a
    # temporary directory by patching the module attribute, and an import-time binding
    # here would not see that -- so the sweep would write into the repository's real
    # ``artifacts/`` during a test run, and the test would then fail on a file the run
    # had put somewhere else entirely. The message would have named a missing path and
    # pointed at nothing.
    from hemolux import provenance as provenance_module
    from hemolux import training

    ensure_dirs()
    gap = result.selection_gap
    rows = [
        _rounded(
            {
                "label": row.label,
                "kind": row.candidate.kind,
                "roi": row.candidate.roi,
                "balance": row.candidate.balance,
                "head": row.candidate.head,
                "feature_dim": row.features.model_ready().feature_dim,
                "val_mae": row.val_report.mae,
                "val_r2": row.val_report.r2,
                "best_epoch": row.fit.best_epoch,
                "seconds": row.fit.train_seconds,
                # Patients whose row describes the whole frame rather than the ROI, so a
                # reader of the table can tell which configurations were not homogeneous.
                "n_unmasked": len(row.features.unmasked),
                "selected": row is result.winner,
            }
        )
        for row in sorted(result.rows, key=lambda r: r.score(result.metric))
    ]

    payload: dict = {
        "provenance": provenance_module.snapshot() if provenance is None else provenance,
        "config": asdict(cfg),
        "metric": result.metric,
        "fold": {
            "name": fold.name,
            "n_train": len(fold.train),
            "n_val": len(fold.val),
            "n_test": len(fold.test),
        },
        "candidates": rows,
        # At the top level as well as per row, because the honest summary of the sweep
        # is "this configuration won" *and* "these configurations were not measuring the
        # same thing for every patient". A reader who checks one and not the other would
        # otherwise be left with the wrong impression.
        "n_unmasked_by_label": {
            row.label: len(row.features.unmasked) for row in result.rows if row.features.unmasked
        },
        "winner": result.winner.label,
        "selection_gap": gap,
        # Spelled out rather than left implicit in the ordering of the two numbers
        # above. "The test fold was scored once, after the winner was chosen" is the
        # claim this artefact exists to support, and a reader should not have to infer
        # it from the schema.
        "test_read": "once, after selection",
        # Which cached columns no candidate was allowed to use. Named at the top level
        # as well as left out of every ``feature_dim``, because a sweep table that adds
        # up to fewer columns than the cache holds should say why in the file itself.
        "excluded_features": sorted(
            {
                name
                for row in result.rows
                for name in training.excluded_feature_columns(row.candidate.kind)
            }
        ),
        "test": None if result.test_report is None else _rounded(result.test_report.to_dict()),
        "notes": list(result.notes),
    }

    reports = training.ARTIFACT_REPORTS
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    _write_csv(reports / "sweep.csv", rows)
    return payload
