"""The ``hemolux`` command line.

``pyproject.toml`` declares ``hemolux = "hemolux.cli:main"``. Until this module
existed the entry point pointed at nothing, so ``pip install hemolux`` produced a
console script that failed on first invocation with an ``ImportError`` and no
explanation. That is the sort of thing a smoke test in CI exists to catch.

Subcommands, in the order a fresh checkout needs them::

    hemolux validate   # the corpus matches what the docs claim
    hemolux features   # cache frozen backbone activations
    hemolux train      # experiment C1 plus the cross-site check
    hemolux sweep      # choose a configuration on validation, read test once
    hemolux export     # ONNX, for the browser
    hemolux report     # print the tables again without retraining

Every subcommand that writes something writes under ``artifacts/`` and prints the
absolute path, so a run's outputs are never a matter of memory.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hemolux.config import (
    ARTIFACT_MODELS,
    ARTIFACT_REPORTS,
    IMAGE_SIZE,
    MOBILE_BUDGET_MB,
    SEED,
    set_seed,
)
from hemolux.data.dataset import DEFAULT_ROOT
from hemolux.training import FEATURE_KINDS

_DEFAULT_DATA = DEFAULT_ROOT


def _parse_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(_DEFAULT_DATA),
        help=f"path to the extracted dataset (default: {_DEFAULT_DATA})",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'")


def _parse_feature_inputs(parser: argparse.ArgumentParser) -> None:
    """Add the flags that choose *what* to featurise.

    Shared by ``features`` and ``train`` because both reach ``_resolve_features``,
    which reads every one of them. They were previously declared per-subcommand,
    which is how ``hemolux export`` came to crash with ``AttributeError: 'Namespace'
    object has no attribute 'backbone'``: it read the same arguments as its siblings
    and declared none of them. A flag list kept in step by hand across parsers is
    one that eventually is not.
    """
    parser.add_argument("--backbone", default="mobilenetv3_small_100")
    parser.add_argument(
        "--roi", default="palpebral", help="palpebral | forniceal | forniceal_palpebral"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--features",
        default="deep",
        choices=list(FEATURE_KINDS),
        help=(
            "deep = frozen backbone activations, colour = hand-crafted CIELAB/HSV "
            "features, hybrid = both concatenated. 'colour' never loads a backbone, "
            "which is how the sweep finds out whether one is needed."
        ),
    )
    balance = parser.add_mutually_exclusive_group()
    balance.add_argument(
        "--balance",
        dest="balance",
        action="store_true",
        default=True,
        help="divide out an estimated illuminant on the colour path (default)",
    )
    balance.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help="measure the raw frame; the baseline the normalisation has to beat",
    )


def build_parser() -> argparse.ArgumentParser:
    """The whole CLI, so ``--help`` and the test suite see the same surface."""
    parser = argparse.ArgumentParser(
        prog="hemolux",
        description="Non-invasive haemoglobin screening from conjunctival images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="check the corpus against the documented counts")
    _parse_common(p_validate)
    p_validate.add_argument("--json", action="store_true", help="emit machine-readable output")
    contract = p_validate.add_mutually_exclusive_group()
    contract.add_argument(
        "--contract",
        dest="contract",
        action="store_true",
        default=None,
        help="check the 217-patient assertions even if this is not that download",
    )
    contract.add_argument(
        "--no-contract",
        dest="contract",
        action="store_false",
        help="check only the structure, never the per-site counts of the download",
    )
    p_validate.add_argument(
        "--no-pixels",
        action="store_true",
        help="skip image decoding; frame and mask geometry stay unverified",
    )
    p_validate.add_argument(
        "--require-masks",
        action="store_true",
        help="treat a missing mask as a failure rather than a warning",
    )
    p_validate.add_argument(
        "--strict-image-log",
        action="store_true",
        help="let libpng's iCCP warnings through instead of filtering that one line",
    )

    p_features = sub.add_parser("features", help="cache frozen backbone activations")
    _parse_common(p_features)
    _parse_feature_inputs(p_features)
    p_features.add_argument("--rebuild", action="store_true", help="ignore an existing cache")

    p_train = sub.add_parser("train", help="run experiment C1 and the cross-site check")
    _parse_common(p_train)
    _parse_feature_inputs(p_train)
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--val-fraction", type=float, default=0.2)
    p_train.add_argument("--heads", default="all", help="'all', or a comma-separated subset")
    p_train.add_argument(
        "--unfreeze",
        type=float,
        default=0.0,
        help="fraction of backbone PARAMETERS to unfreeze; 0 is a linear probe",
    )
    p_train.add_argument("--no-site-holdout", action="store_true", help="skip the cross-site check")
    p_train.add_argument("--rebuild-features", action="store_true", help="ignore the feature cache")

    p_sweep = sub.add_parser(
        "sweep",
        help="choose a configuration on validation, then read the test fold once",
    )
    _parse_common(p_sweep)
    p_sweep.add_argument("--backbone", default="mobilenetv3_small_100")
    p_sweep.add_argument("--epochs", type=int, default=40)
    p_sweep.add_argument("--lr", type=float, default=1e-3)
    p_sweep.add_argument("--val-fraction", type=float, default=0.2)
    p_sweep.add_argument("--batch-size", type=int, default=16)
    p_sweep.add_argument(
        "--head", default="regression", help="the head to sweep over (one head at a time)"
    )
    p_sweep.add_argument(
        "--metric",
        default="mae",
        choices=["mae", "r2"],
        help="validation metric to select on; MAE by default because that is what a "
        "screening tool is judged on, and R^2 scores a 0.5 and a 3.0 g/dL miss alike",
    )
    p_sweep.add_argument(
        "--only",
        help="comma-separated subset of the grid, by label or by kind/roi/balance",
    )
    p_sweep.add_argument("--rebuild-features", action="store_true", help="ignore the feature cache")

    p_export = sub.add_parser("export", help="write the ONNX graph for in-browser inference")
    _parse_common(p_export)
    # Deliberately no --backbone or --roi. This command reads a trained checkpoint
    # and rebuilds the graph the checkpoint names, so offering them would imply the
    # export could be steered away from the weights it is given -- which is exactly
    # the failure mode build_screen_model refuses to allow.
    p_export.add_argument("--head", default="ordinal", help="which trained head to export")
    p_export.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="default: the most recently written checkpoint for --head",
    )
    p_export.add_argument(
        "--size", type=int, default=IMAGE_SIZE, help="input edge; must match the app's canvas"
    )
    p_export.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: artifacts/models/hemolux_screen_<size>.onnx",
    )
    p_export.add_argument("--opset", type=int, default=17)

    p_report = sub.add_parser("report", help="print the last results without retraining")
    p_report.add_argument("--results", type=Path, default=ARTIFACT_REPORTS / "results.json")

    return parser


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_validate(args: argparse.Namespace) -> int:
    """Run the corpus check and exit non-zero on a failure."""
    from hemolux.imagemeta import quiet_image_decoder
    from hemolux.validation import validate_corpus

    try:
        with quiet_image_decoder(force=args.strict_image_log):
            report = validate_corpus(
                args.data_root,
                contract=args.contract,
                require_masks=args.require_masks,
                read_pixels=not args.no_pixels,
            )
    except (FileNotFoundError, ValueError) as exc:
        # The library raises rather than returning a short list when the layout is
        # wrong, which is right for a caller. A command line is not a caller, so
        # the traceback is translated into the same verdict the checks would give.
        print(f"FAIL\n  the corpus could not be read: {exc}", file=sys.stderr)
        return 1

    if args.json:
        import json

        print(json.dumps(report.as_dict(), indent=2))
    else:
        report.print()
    return 0 if report.ok else 1


def _resolve_features(args: argparse.Namespace):
    """Load the cached features, building them if absent or for a different config.

    The cache is found by reading each candidate's own recorded ``kind`` and
    ``balance`` rather than by matching the filename. The filename already encodes
    both, and matching on it would have been shorter -- but the glob then depends on
    the naming staying in step with the content, and the failure when it drifts is a
    cache read as the wrong configuration: same command, plausible numbers, silently
    the wrong features. Reading the metadata makes the two disagree loudly instead.
    """
    from hemolux.data.dataset import build_records

    kind = getattr(args, "features", "deep")
    balance = "balanced" if getattr(args, "balance", True) else "raw"
    records = build_records(args.data_root, verbose=False)
    return _features_for(kind, args.roi, balance, args, records), records


def _features_for(kind: str, roi: str, balance: str, args, records):
    """The cached FeatureSet for one (kind, roi, balance), extracting it if absent.

    Split out from :func:`_resolve_features` so that a sweep resolving fifteen
    configurations goes through exactly the same cache-matching code as a single
    training run. Two copies of this would be two chances for the sweep to build
    features the single-run path would have rejected as a stale read.
    """
    from hemolux.data.dataset import build_records
    from hemolux.fingerprint import StaleFeatureCacheError
    from hemolux.training import (
        extract_features,
        feature_cache_path,
        load_feature_cache,
        save_feature_cache,
    )

    if not getattr(args, "rebuild_features", False) and not getattr(args, "rebuild", False):
        for path in sorted(ARTIFACT_MODELS.glob("features_*.npz")):
            try:
                features = load_feature_cache(path)
            except StaleFeatureCacheError as stale:
                # The reason is printed rather than swallowed. Re-extracting is the
                # right response, but doing it silently hides the one thing worth
                # knowing: that the feature code changed since this cache was written,
                # so no number derived from it is reproducible from a checkout of the
                # commit that produced it.
                print(f"  ignoring stale cache {path.name}: {stale}")
                continue
            except Exception:
                # A cache this build cannot read is not a cache, and a half-written
                # .npz from an interrupted run is the ordinary case here. Skipping it
                # is right: the alternative is refusing to train because of a
                # leftover, and re-extracting is the correct response either way.
                continue
            same = (
                features.roi == roi
                and features.kind == kind
                and features.balance == balance
                # The colour path never loads a backbone, so ``--backbone`` does not
                # identify it -- and comparing it would look for a backbone named
                # "none", missing the cache on every call. Skipped rather than faked,
                # because for colour features the trunk genuinely is not part of what
                # the numbers are. It still matters for hybrid, which concatenates it.
                and (kind == "colour" or features.backbone == args.backbone)
            )
            if same:
                print(f"  using cached features: {path.name}")
                return features

    label = kind if kind == "deep" else f"{kind} ({balance})"
    print(f"\n  extracting features: {label}  backbone={args.backbone} roi={roi}")
    if kind != "colour":
        # The corpus banner belongs to the extraction that reads the pixels. Repeating
        # it for each of fifteen configurations would bury the part that matters.
        build_records(args.data_root, verbose=True)
    features = extract_features(
        records,
        backbone=args.backbone,
        roi=roi,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        kind=kind,
        balance=balance == "balanced",
    )
    path = save_feature_cache(features)
    print(f"  wrote {path}  ({feature_cache_path(features).name})")
    return features


def cmd_features(args: argparse.Namespace) -> int:
    features, _ = _resolve_features(args)
    print(f"\n  patients: {len(features.patient_ids)}")
    print(f"  feature kind: {features.kind} ({features.balance})")
    print(f"  feature dim: {features.feature_dim}")
    for site in sorted(set(features.site)):
        hb = [h for h, s in zip(features.hb, features.site, strict=True) if s == site]
        print(f"    {site:<6} n={len(hb):<4} Hb mean {sum(hb) / len(hb):.2f}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    try:
        return _sweep(args)
    except ValueError as exc:
        # Same shape as cmd_train: the metrics helpers refuse a degenerate split by
        # raising, and "need at least 2 finite pairs" is not an answer to "why did my
        # sweep stop". A traceback for a condition fixed by generating a larger fixture
        # is the wrong shape for a command line.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    try:
        return _train(args)
    except ValueError as exc:
        # The split machinery refuses a corpus too small to evaluate -- correctly,
        # since a MAE over one patient is not a MAE -- but "need at least 2 finite
        # pairs" from inside a metrics helper is not an answer to "why did my run
        # stop". A traceback for a condition the user can fix by generating a larger
        # fixture is the wrong shape for a command line.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _sweep(args: argparse.Namespace) -> int:
    """Select a configuration on validation, then read test exactly once."""
    from hemolux.data.dataset import build_records
    from hemolux.data.splits import Fold, Split, severity_bin, stratified_assignments
    from hemolux.sweep import build_sweep_report, finish_sweep, sweep_grid
    from hemolux.training import TrainConfig

    set_seed(args.seed)
    records = build_records(args.data_root, verbose=True)
    ordered = [r.patient_id for r in records]
    severities = [severity_bin(r.hb, r.sex) for r in records]

    cfg = TrainConfig(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )

    candidates = _sweep_candidates(args)
    if candidates is None:
        return 2
    print(f"\n  {len(candidates)} candidate(s), selecting on validation {args.metric}")

    # Distinct extraction keys only. Two candidates that differ solely in head share
    # one FeatureSet, and asking for it twice would read every frame twice.
    keys = sorted({c.cache_key() for c in candidates})
    print(f"  {len(keys)} distinct feature set(s) to resolve")
    by_key: dict[str, object] = {}
    for key in keys:
        kind, roi, balance = key.split("/")
        by_key[key] = _features_for(kind, roi, balance, args, records)

    assignments = stratified_assignments(
        ordered, severities, val_fraction=cfg.val_fraction, seed=cfg.seed
    )
    fold = Fold(
        name=f"sweep-seed{cfg.seed}",
        train=tuple(p for p in ordered if assignments[p] == Split.TRAIN),
        val=tuple(p for p in ordered if assignments[p] == Split.VAL),
        test=tuple(p for p in ordered if assignments[p] == Split.TEST),
    )
    for name, members in (("test", fold.test), ("val", fold.val)):
        if len(members) < 2:
            print(
                f"error: the {name} fold holds {len(members)} patient(s) out of "
                f"{len(ordered)}. Selection reads the validation fold for every "
                f"candidate, so a fold of one gives nothing to compare. Lower "
                f"--val-fraction, or use a larger corpus.",
                file=sys.stderr,
            )
            return 1
    print(
        f"  {fold.name}: {len(fold.train)} train / {len(fold.val)} val / "
        f"{len(fold.test)} test  (test not scored until the winner is chosen)"
    )

    print("\n  sweeping")
    rows = sweep_grid(candidates, fold, by_key, cfg=cfg)

    header = f"{'configuration':<34}{'dim':>6}{'val MAE':>10}{'val R2':>9}{'epoch':>7}{'sec':>7}{'no ROI':>7}"
    print("\n  validation, every candidate (no test numbers exist for any of these)")
    print("  " + header)
    print("  " + "-" * len(header))
    # Sorted best-first, so the first row *is* the selection -- the marker cannot
    # disagree with the choice made a few lines later, because it is the same ordering.
    ranked = sorted(rows, key=lambda r: (r.score(args.metric), r.label))
    for i, row in enumerate(ranked):
        mark = f"  <- best validation {args.metric}" if i == 0 else ""
        print(
            f"  {row.label:<32}{row.features.feature_dim:>6}"
            f"{row.val_report.mae:>10.4f}{row.val_report.r2:>9.4f}"
            f"{row.fit.best_epoch:>7}{row.fit.train_seconds:>7.1f}"
            f"{len(row.features.unmasked):>7}{mark}"
        )
    partial = {r.label: len(r.features.unmasked) for r in rows if r.features.unmasked}
    if partial:
        print(
            f"\n  no ROI: {len(partial)} configuration(s) measured some patients on the "
            f"whole frame, because their record holds no mask for the ROI requested."
        )
        for label, count in sorted(partial.items()):
            ids = sorted({p for r in rows if r.label == label for p in r.features.unmasked})
            print(f"    {label:<38} {count:>3} patient(s): {', '.join(ids)}")

    print("\n  selected on validation, then scoring the test fold once")
    result = finish_sweep(rows, fold, metric=args.metric)
    gap = result.selection_gap
    print(f"  winner: {result.winner.label}")
    print(
        f"  selection gap: {gap:.4f} g/dL ahead of the next candidate on validation {args.metric}"
    )
    if np.isfinite(gap) and gap < 0.10:
        # Said here rather than left to the reader. A tenth of a gram per decilitre is
        # far inside the +/-1.0 clinical acceptability target, so a gap this small means
        # the sweep has not established a preference -- and a test number chosen from
        # near-ties is one draw rather than a measurement.
        print(
            f"  note: {gap:.4f} g/dL is a near-tie. The grid does not separate these "
            f"configurations, so treat the test number as one draw from several "
            f"candidates that are about equally good."
        )

    r = result.test_report
    print(f"\n  test fold, {r.n} patients, read once:")
    print(f"    MAE   {r.mae:.3f} g/dL")
    print(f"    RMSE  {r.rmse:.3f}")
    print(f"    R2    {r.r2:+.3f}")
    print(
        f"    bias  {r.bias:+.3f}   95% limits of agreement [{r.loa_lower:+.2f}, {r.loa_upper:+.2f}]"
    )
    print(f"    within +/-1 g/dL  {100 * r.within_1:.1f}%")
    print(f"    within +/-2 g/dL  {100 * r.within_2:.1f}%")

    build_sweep_report(result, fold, cfg)
    print(f"\n  wrote {ARTIFACT_REPORTS / 'sweep.json'}")
    print(f"  wrote {ARTIFACT_REPORTS / 'sweep.csv'}")
    return 0


def _sweep_candidates(args: argparse.Namespace) -> list | None:
    """The candidate list: the full grid, or the subset ``--only`` names.

    Three spellings are accepted per candidate -- ``kind/roi/balance``, ``kind/roi``,
    and the label -- because the balance suffix is meaningless for a ``deep`` candidate
    that has no illuminant to vary, and demanding it there would make the obvious
    ``--only deep/palpebral`` quietly select nothing. A filter that matches less than it
    appears to is worse than one that refuses, because the sweep then reports a shorter
    grid than the reader asked for and every row of it looks like a complete result.

    A partially-unmatched filter is an error rather than a partial run. Silently
    dropping one term would leave the reader believing the grid was the one they asked
    for.
    """
    from hemolux.sweep import default_grid, sweep_label

    grid = default_grid(head=args.head)
    if not args.only:
        return grid

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    def spellings(candidate) -> set[str]:
        return {
            f"{candidate.kind}/{candidate.roi}/{candidate.balance}",
            f"{candidate.kind}/{candidate.roi}",
            sweep_label(candidate),
        }

    chosen = [c for c in grid if spellings(c) & wanted]
    matched = {s for c in chosen for s in spellings(c)}
    unmatched = wanted - matched

    if unmatched or not chosen:
        print(f"error: --only {sorted(unmatched or wanted)} matched no candidate.", file=sys.stderr)
        print("  accepted spellings per candidate:", file=sys.stderr)
        for c in grid:
            print(f"    {sweep_label(c):<38} {sorted(spellings(c))}", file=sys.stderr)
        return None
    return chosen


def _train(args: argparse.Namespace) -> int:
    from hemolux.data.dataset import build_records
    from hemolux.data.splits import Fold, Split, severity_bin, stratified_assignments
    from hemolux.training import (
        HEAD_NAMES,
        TrainConfig,
        build_report,
        print_results_table,
        results_row,
        run_single_split,
        run_site_holdout,
    )

    set_seed(args.seed)
    features, _ = _resolve_features(args)

    heads = HEAD_NAMES if args.heads == "all" else tuple(h.strip() for h in args.heads.split(","))

    cfg = TrainConfig(
        backbone=args.backbone,
        roi=args.roi,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_param_fraction=args.unfreeze,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )

    print("\n  building the patient-disjoint split")
    ordered = list(features.patient_ids)
    by_id = {r.patient_id: r for r in build_records(args.data_root, verbose=False)}
    missing = [p for p in ordered if p not in by_id]
    if missing:
        print(
            f"error: the feature cache holds {len(missing)} patient(s) not in the corpus "
            f"(e.g. {missing[:3]}). Re-run with --rebuild-features.",
            file=sys.stderr,
        )
        return 2
    severities = [severity_bin(by_id[p].hb, by_id[p].sex) for p in ordered]

    assignments = stratified_assignments(
        ordered, severities, val_fraction=cfg.val_fraction, seed=cfg.seed
    )
    fold = Fold(
        name=f"single-seed{cfg.seed}",
        train=tuple(p for p in ordered if assignments[p] == Split.TRAIN),
        val=tuple(p for p in ordered if assignments[p] == Split.VAL),
        test=tuple(p for p in ordered if assignments[p] == Split.TEST),
    )
    # Checked here rather than left to the metrics helpers. Every evaluation in this
    # file needs at least two pairs to compute a correlation, and the failure when
    # there are not arrives from ``numpy.concatenate`` with no mention of the corpus.
    # An unusable split is a fact about the corpus the user chose, so it is reported
    # as one, naming the fold that came out empty and the knob that sized it.
    for name, members in (("test", fold.test), ("train", fold.train)):
        if len(members) < 2:
            print(
                f"error: the {name} fold holds {len(members)} patient(s) out of "
                f"{len(ordered)}, which is too few to evaluate -- a correlation needs "
                f"at least 2. With --val-fraction {cfg.val_fraction:g} the corpus is "
                f"too small for a {len(fold.train)}/{len(fold.val)}/{len(fold.test)} "
                f"split. Lower --val-fraction, or use a larger corpus.",
                file=sys.stderr,
            )
            return 1

    print(
        f"  {fold.name}: {len(fold.train)} train / {len(fold.val)} val / {len(fold.test)} test"
        f"  (unfreezing {cfg.unfreeze_param_fraction:.0%} of backbone parameters)"
    )

    print(f"\n  training {len(heads)} head(s): {', '.join(heads)}")
    fits = run_single_split(features, cfg=cfg, fold=fold, heads=heads)
    rows = [results_row(f, features, fold) for f in fits]
    print_results_table(
        rows, f"Experiment C1 -- head comparison, held-out {len(fold.test)} patients"
    )

    site_rows: list[dict] = []
    if not args.no_site_holdout:
        print("\n  cross-site validation (train on one site, test on the other)")
        site_rows = run_site_holdout(features, cfg=cfg, heads=("regression", "ordinal"))
        print()
        header = f"{'head':<11}{'fold':<18}{'n':>4}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'bias':>8}{'±1':>7}{'±2':>7}{'test Hb':>9}"
        print(header)
        print("-" * len(header))
        for r in site_rows:
            print(
                f"{r['head']!s:<11}{r['fold']!s:<18}{int(r['n_test']):>4}"
                f"{float(r['mae']):>8.3f}{float(r['rmse']):>8.3f}{float(r['r2']):>8.3f}"
                f"{float(r['bias']):>8.3f}{float(r['within_1']):>7.2f}{float(r['within_2']):>7.2f}"
                f"{float(r['mean_true_hb_test_site']):>9.2f}"
            )

    print("\n  building the results file")
    payload = build_report(fits, features, fold, cfg=cfg, site_rows=site_rows)
    print(f"  wrote {ARTIFACT_REPORTS / 'results.json'}")
    print(f"  wrote {ARTIFACT_REPORTS / 'results.csv'}")
    print(f"  wrote {ARTIFACT_REPORTS / 'predictions.csv'}")
    best = max(rows, key=lambda r: r["r2"])
    print(
        f"\n  best head by test R2: {best['head']} (R2 {best['r2']:.3f}, MAE {best['mae']:.3f} g/dL)"
    )
    print(f"  {payload['reference']}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Write the ONNX graph, then prove it computes what PyTorch computes.

    Verification is not optional here. ``export_onnx`` is where a silent
    Python/JavaScript divergence would be introduced, and an ONNX file that exports
    without error but decodes differently still looks like a successful export --
    every prediction shifts by a constant and nothing raises. So the graph is run
    against its PyTorch reference before the command reports success.
    """
    from hemolux.export import (
        build_screen_model,
        export_onnx,
        file_size_mb,
        load_checkpoint,
        verify_onnx,
    )

    checkpoint_path = args.checkpoint or _newest_checkpoint(args.head)
    if checkpoint_path is None:
        print(
            f"error: no {args.head} checkpoint under {ARTIFACT_MODELS}. Run `hemolux train "
            f"--heads {args.head}` first -- the export needs trained weights, and a "
            f"randomly initialised graph would still load and still predict.",
            file=sys.stderr,
        )
        return 1
    if not checkpoint_path.is_file():
        print(f"error: {checkpoint_path} does not exist", file=sys.stderr)
        return 1

    checkpoint = load_checkpoint(checkpoint_path)
    print(f"  checkpoint: {checkpoint_path.name}")
    print(
        f"    backbone {checkpoint['backbone']}  head {checkpoint['head']}  "
        f"roi {checkpoint['roi']}  fold {checkpoint['fold']}  "
        f"feature_dim {checkpoint['feature_dim']}"
    )

    screen = build_screen_model(checkpoint, size=args.size)
    path = export_onnx(screen, size=args.size, opset=args.opset, path=args.out)

    size_mb = file_size_mb(path)
    print(f"\n  wrote {path}")
    print(f"    {size_mb:.2f} MB")
    if size_mb > MOBILE_BUDGET_MB:
        print(
            f"    {size_mb:.1f} MB is over the {MOBILE_BUDGET_MB:.0f} MB a phone will "
            f"download without asking twice. Quantise to int8 before shipping."
        )

    gaps = verify_onnx(screen, path, size=args.size)
    print("\n  ONNX vs PyTorch on identical input:")
    print(f"    hb    max |diff|: {gaps['hb_max_abs_diff']:.3e} g/dL")
    print(f"    sigma max |diff|: {gaps['sigma_max_abs_diff']:.3e} g/dL")
    print("    verified")
    return 0


def _newest_checkpoint(head: str) -> Path | None:
    """The most recently written checkpoint for ``head``.

    Newest by modification time rather than by sorted name. The filename carries
    backbone, ROI and fold, so two runs that differ in any of those produce two
    files whose alphabetical order has nothing to do with which is current.
    """
    candidates = list(ARTIFACT_MODELS.glob(f"{head}_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def cmd_report(args: argparse.Namespace) -> int:
    """Reprint the last run. Fails loudly if there is nothing to reprint."""
    import json

    if not args.results.is_file():
        print(f"error: {args.results} does not exist. Run `hemolux train` first.", file=sys.stderr)
        return 1
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    print(f"\n  {args.results}")
    print(
        f"  config: backbone={payload['config']['backbone']} roi={payload['config']['roi']} "
        f"epochs={payload['config']['epochs']} seed={payload['config']['seed']}"
    )
    print()
    header = f"{'head':<11}{'n':>4}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'r':>7}{'bias':>8}{'LoA-':>8}{'LoA+':>8}{'±1':>7}{'±2':>7}{'siteΔ':>8}"
    print(header)
    print("-" * len(header))
    for r in payload["single_split"]:
        print(
            f"{r['head']:<11}{int(r['n_test']):>4}{float(r['mae']):>8.3f}{float(r['rmse']):>8.3f}"
            f"{float(r['r2']):>8.3f}{float(r['pearson_r']):>7.3f}{float(r['bias']):>8.3f}"
            f"{float(r['loa_lower']):>8.3f}{float(r['loa_upper']):>8.3f}"
            f"{float(r['within_1']):>7.2f}{float(r['within_2']):>7.2f}{float(r['site_mae_gap']):>8.3f}"
        )
    if payload.get("site_holdout"):
        print("\n  cross-site")
        for r in payload["site_holdout"]:
            print(
                f"    {r['head']:<11}{r['fold']:<18}MAE {float(r['mae']):.3f}  "
                f"R2 {float(r['r2']):.3f}  bias {float(r['bias']):.3f}  "
                f"test-site mean Hb {float(r['mean_true_hb_test_site']):.2f}"
            )
    return 0


_COMMANDS = {
    "validate": cmd_validate,
    "features": cmd_features,
    "train": cmd_train,
    "export": cmd_export,
    "report": cmd_report,
    "sweep": cmd_sweep,
}


def _make_stdout_unicode_safe() -> None:
    """Stop a console encoding from killing a finished run.

    Windows consoles default to cp1252, which cannot encode a Greek delta, a
    degree sign or Devanagari. Any of those in a report would raise
    ``UnicodeEncodeError`` at the moment of printing -- after the training,
    inference and scoring had all succeeded -- and take the whole run's output
    with it.

    ``backslashreplace`` rather than ``replace``: an unrepresentable character
    becomes ``\\u0394``, which is visible and greppable, instead of a silent
    ``?`` that looks like intentional formatting.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # A redirected or already-closed stream has nothing to fix.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling ``sys.exit``."""
    _make_stdout_unicode_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
