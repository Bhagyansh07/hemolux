#!/usr/bin/env python
"""Regenerate the EVALS numbers and the fairness figures from the frozen run.

Why this is a script and not a table typed by hand
--------------------------------------------------
A results table copied out of a console scrollback is a second copy of the run, and
the two drift -- usually at the moment someone fixes the first one. This reads the
artefacts the run already wrote (``artifacts/reports/results.json``, and
``sweep.json`` when a sweep has run) and emits the exact markdown block that
``EVALS.md`` transcribes, plus the figures. A number in ``EVALS.md`` that this
script does not print is a mistake, and that is the point.

``predictions.csv`` is optional. It carries a patient id, so it is git-ignored; the
per-patient figures (predicted-versus-true, Bland-Altman) are the only outputs that
need it. Every aggregate table and the reliability figure depend only on the
tracked ``results.json``, so those regenerate in a clean checkout::

    python scripts/make_figures.py

The per-site reliability curve is the figure that matters most here. A pooled ECE
can be small while one site is badly miscalibrated, and that gap is the finding a
screening programme is run on, so it is drawn rather than summarised.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hemolux.config import ARTIFACT_FIGURES, ARTIFACT_REPORTS, ensure_dirs

#: Colour-blind-safe, and only four of them: a fifth group on one axes is a chart
#: nobody can read, and this corpus has two sites.
COLOURS = ("#1b6ca8", "#d1495b", "#2a9d8f", "#7d5ba6")

#: The one screening band whose per-site contrast is the headline. The moderate and
#: severe bands are reported in the JSON and, on this corpus, are mostly
#: ``not_computable`` -- drawing an empty severe curve would imply a measurement.
HEADLINE_BAND = "anaemia"


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    x = _num(value)
    return "-" if not np.isfinite(x) else f"{x:.{digits}f}"


def _signed(value: Any, digits: int = 3) -> str:
    x = _num(value)
    return "-" if not np.isfinite(x) else f"{x:+.{digits}f}"


def _pct(value: Any) -> str:
    x = _num(value)
    return "-" if not np.isfinite(x) else f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def markdown_block(payload: dict[str, Any]) -> str:
    """The EVALS block, transcribed from the artefacts rather than from memory."""
    provenance = payload.get("provenance") or {}
    corpus = payload.get("corpus") or {}
    fold = payload.get("fold") or {}
    parts: list[str] = []

    parts.append("### Frozen run\n")
    parts.append(
        _table(
            ["Field", "Value"],
            [
                ["Commit", f"`{provenance.get('git_commit') or '-'}`"],
                ["Working tree dirty", str(provenance.get("git_dirty", "-")).lower()],
                ["Feature fingerprint", f"`{provenance.get('feature_fingerprint') or '-'}`"],
                ["Generated", str(provenance.get("generated_at") or "-")],
                [
                    "Corpus",
                    f"{corpus.get('n_patients', '-')} patients, {corpus.get('roi', '-')} ROI, "
                    f"{corpus.get('backbone', '-')}",
                ],
                [
                    "Split",
                    f"{fold.get('n_train', '-')} train / {fold.get('n_val', '-')} val / "
                    f"{fold.get('n_test', '-')} test ({fold.get('name', '-')})",
                ],
                [
                    "Excluded features",
                    ", ".join(corpus.get("excluded_features") or []) or "none",
                ],
            ],
        )
    )

    per_site = corpus.get("per_site") or {}
    if per_site:
        parts.append("\nPer site, before any model:\n")
        parts.append(
            _table(
                ["Site", "n", "Hb mean (g/dL)", "Age mean"],
                [
                    [
                        site,
                        str(block.get("n", "-")),
                        _fmt(block.get("hb_mean")),
                        _fmt(block.get("age_mean"), 1),
                    ]
                    for site, block in sorted(per_site.items())
                ],
            )
        )

    rows = payload.get("single_split") or []
    if rows:
        parts.append("\nTest fold, patient-disjoint, read once:\n")
        parts.append(
            _table(
                [
                    "Head",
                    "n",
                    "MAE",
                    "RMSE",
                    "R2",
                    "Bias",
                    "LoA",
                    "within +-1",
                    "within +-2",
                    "Site MAE gap",
                ],
                [
                    [
                        str(row.get("head", "-")),
                        str(row.get("n_test", "-")),
                        _fmt(row.get("mae")),
                        _fmt(row.get("rmse")),
                        _signed(row.get("r2")),
                        _signed(row.get("bias")),
                        f"[{_signed(row.get('loa_lower'), 2)}, {_signed(row.get('loa_upper'), 2)}]",
                        _pct(row.get("within_1")),
                        _pct(row.get("within_2")),
                        f"{_fmt(row.get('site_mae_gap'))} ({row.get('worst_site', '-')})",
                    ]
                    for row in rows
                ],
            )
        )

    holdout = payload.get("site_holdout") or []
    if holdout:
        parts.append("\nCross-site transfer (train on one site, test on the other):\n")
        parts.append(
            _table(
                ["Head", "Fold", "n", "MAE", "R2", "Bias", "Test-site Hb mean"],
                [
                    [
                        str(row.get("head", "-")),
                        str(row.get("fold", "-")),
                        str(row.get("n_test", "-")),
                        _fmt(row.get("mae")),
                        _signed(row.get("r2")),
                        _signed(row.get("bias")),
                        _fmt(row.get("mean_true_hb_test_site"), 2),
                    ]
                    for row in holdout
                ],
            )
        )

    audit = payload.get("site_audit") or {}
    screening_rows: list[list[str]] = []
    calibration_rows: list[list[str]] = []
    for head, block in audit.items():
        if not isinstance(block, dict) or not block.get("available"):
            continue
        screening = block.get("screening") or {}
        pooled = (screening.get("pooled") or {}).get(HEADLINE_BAND) or {}
        if pooled:
            screening_rows.append(
                [
                    head,
                    "pooled",
                    str(pooled.get("n", "-")),
                    str(pooled.get("positives", "-")),
                    _pct(pooled.get("sensitivity")),
                    _pct(pooled.get("specificity")),
                    _pct(pooled.get("ppv")),
                    _pct(pooled.get("npv")),
                    str(pooled.get("verdict", "-")),
                ]
            )
        for site, bands in sorted((screening.get("per_site") or {}).items()):
            band = bands.get(HEADLINE_BAND) or {}
            screening_rows.append(
                [
                    head,
                    site,
                    str(band.get("n", "-")),
                    str(band.get("positives", "-")),
                    _pct(band.get("sensitivity")),
                    _pct(band.get("specificity")),
                    _pct(band.get("ppv")),
                    _pct(band.get("npv")),
                    str(band.get("verdict", "-")),
                ]
            )

        calibration = block.get("calibration") or {}
        if calibration.get("available"):
            sites = calibration.get("per_site") or {}
            by_site = ", ".join(
                f"{site} {_fmt((sites[site] or {}).get('ece'))}" for site in sorted(sites)
            )
            calibration_rows.append(
                [
                    head,
                    _fmt((calibration.get("pooled") or {}).get("ece")),
                    by_site or "-",
                    _fmt(calibration.get("max_ece_gap")),
                ]
            )

    if screening_rows:
        parts.append(f"\nScreening at the {HEADLINE_BAND} cutoff, pooled and per site:\n")
        parts.append(
            _table(
                [
                    "Head",
                    "Group",
                    "n",
                    "Positives",
                    "Sensitivity",
                    "Specificity",
                    "PPV",
                    "NPV",
                    "Verdict",
                ],
                screening_rows,
            )
        )
    if calibration_rows:
        parts.append("\nCross-site calibration (anaemia probability, ten bins):\n")
        parts.append(
            _table(
                ["Head", "Pooled ECE", "Per-site ECE", "Best-worst gap"],
                calibration_rows,
            )
        )

    return "\n".join(parts) + "\n"


def sweep_block(payload: dict[str, Any]) -> str:
    """The configuration-selection section, read from ``sweep.json``.

    This is separate from :func:`markdown_block` because the two artefacts answer
    different questions. ``results.json`` is how the trained model did; ``sweep.json``
    is *how the configuration was chosen*, and a winner named without the validation
    numbers it beat is a claim, not a selection. The candidate table is the whole
    point: it is where a loser such as a collapsed balanced-colour feature set stays
    on the record instead of being deleted from the story.
    """
    provenance = payload.get("provenance") or {}
    fold = payload.get("fold") or {}
    candidates = payload.get("candidates") or []
    parts: list[str] = []

    parts.append("### Configuration sweep\n")
    parts.append(
        _table(
            ["Field", "Value"],
            [
                ["Commit", f"`{provenance.get('git_commit') or '-'}`"],
                ["Feature fingerprint", f"`{provenance.get('feature_fingerprint') or '-'}`"],
                ["Generated", str(provenance.get("generated_at") or "-")],
                ["Metric", str(payload.get("metric") or "-")],
                [
                    "Selection fold",
                    f"{fold.get('n_train', '-')} train / {fold.get('n_val', '-')} val "
                    f"({fold.get('name', '-')})",
                ],
                ["Test read", str(payload.get("test_read") or "-")],
                ["Winner", str(payload.get("winner") or "-")],
                ["Selection gap", _fmt(payload.get("selection_gap"))],
                [
                    "Excluded features",
                    ", ".join(payload.get("excluded_features") or []) or "none",
                ],
            ],
        )
    )

    if candidates:
        rows: list[list[str]] = []
        for candidate in candidates:
            unmasked = _num(candidate.get("n_unmasked"))
            rows.append(
                [
                    str(candidate.get("label", "-")),
                    str(candidate.get("kind", "-")),
                    str(candidate.get("roi", "-")),
                    str(candidate.get("balance", "-")),
                    str(candidate.get("head", "-")),
                    _fmt(candidate.get("feature_dim"), 0),
                    str(int(unmasked)) if np.isfinite(unmasked) else "-",
                    _fmt(candidate.get("val_mae")),
                    _signed(candidate.get("val_r2")),
                    "yes" if candidate.get("selected") else "no",
                ]
            )
        parts.append("\nCandidates, ranked on validation only:\n")
        parts.append(
            _table(
                [
                    "Candidate",
                    "Kind",
                    "ROI",
                    "Balance",
                    "Head",
                    "Dim",
                    "Whole-frame",
                    "Val MAE",
                    "Val R2",
                    "Selected",
                ],
                rows,
            )
        )

    test = payload.get("test") or {}
    if test:
        parts.append("\nWinner on the test fold, scored once after selection:\n")
        parts.append(
            _table(
                ["n", "MAE", "RMSE", "R2", "EVS", "Pearson r", "Bias", "LoA", "+-1", "+-2"],
                [
                    [
                        str(test.get("n", "-")),
                        _fmt(test.get("mae")),
                        _fmt(test.get("rmse")),
                        _signed(test.get("r2")),
                        _signed(test.get("evs")),
                        _signed(test.get("pearson_r")),
                        _signed(test.get("bias")),
                        f"[{_signed(test.get('loa_lower'), 2)}, {_signed(test.get('loa_upper'), 2)}]",
                        _pct(test.get("within_1")),
                        _pct(test.get("within_2")),
                    ]
                ],
            )
        )
    return "\n".join(parts) + "\n"


def figure_reliability(payload: dict[str, Any], outdir: Path) -> Path | None:
    """Reliability curves, pooled against each site, for every calibratable head."""
    audit = payload.get("site_audit") or {}
    panels = [
        (head, block)
        for head, block in sorted(audit.items())
        if isinstance(block, dict) and (block.get("calibration") or {}).get("available")
    ]
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.6), squeeze=False)
    for ax, (head, block) in zip(axes[0], panels, strict=True):
        calibration = block["calibration"]
        curves = {"pooled": calibration.get("pooled") or {}, **(calibration.get("per_site") or {})}
        for index, (name, curve) in enumerate(
            sorted(curves.items(), key=lambda kv: kv[0] != "pooled")
        ):
            bins = curve.get("bins") or []
            conf = np.array([_num(b.get("confidence")) for b in bins], dtype=float)
            acc = np.array([_num(b.get("accuracy")) for b in bins], dtype=float)
            count = np.array([_num(b.get("count")) for b in bins], dtype=float)
            keep = np.isfinite(conf) & np.isfinite(acc) & (count > 0)
            if not keep.any():
                continue
            colour = COLOURS[index % len(COLOURS)]
            ax.plot(conf[keep], acc[keep], "o-", color=colour, markersize=5, linewidth=1.5)
            ax.plot([], [], "o-", color=colour, label=f"{name} (ECE {_fmt(curve.get('ece'))})")
        ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1.0, label="perfect")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="estimated P(Hb below cutoff)",
            ylabel="observed frequency",
            title=head,
        )
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Calibration, pooled and per site")
    fig.tight_layout()
    destination = outdir / "reliability_pooled_vs_site.png"
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def figure_screening(payload: dict[str, Any], outdir: Path) -> Path | None:
    """Sensitivity and specificity at the anaemia cutoff, pooled against each site."""
    audit = payload.get("site_audit") or {}
    heads = [
        head
        for head, block in sorted(audit.items())
        if isinstance(block, dict) and (block.get("screening") or {}).get("pooled")
    ]
    if not heads:
        return None

    # Build an explicit head-by-group table rather than appending as we go. A head
    # can be missing a site (the regression heads emit no probability on one arm),
    # and an append-then-plot loop would then hand matplotlib a shorter height
    # vector than the x positions, which numpy broadcasts into a bar drawn from
    # another head's value rather than a gap.
    groups: list[str] = []
    cells_by_head: dict[str, dict[str, dict[str, Any]]] = {}
    for head in heads:
        screening = audit[head]["screening"]
        cells: dict[str, dict[str, Any]] = {
            "pooled": (screening.get("pooled") or {}).get(HEADLINE_BAND) or {}
        }
        for site, bands in (screening.get("per_site") or {}).items():
            cells[site] = (bands or {}).get(HEADLINE_BAND) or {}
        cells_by_head[head] = cells
        groups.extend(name for name in cells if name not in groups)
    groups.sort(key=lambda name: name != "pooled")

    fig, axes = plt.subplots(1, 2, figsize=(7.0 * 2, 4.6))
    for ax, metric in zip(axes, ("sensitivity", "specificity"), strict=True):
        x = np.arange(len(heads), dtype=float)
        width = 0.8 / max(len(groups), 1)
        for index, name in enumerate(groups):
            heights = np.array(
                [_num((cells_by_head.get(head, {}).get(name) or {}).get(metric)) for head in heads]
            )
            ax.bar(
                x + index * width,
                heights,
                width,
                color=COLOURS[index % len(COLOURS)],
                label=name,
            )
        ax.set(
            xticks=x + 0.4 - width / 2,
            ylim=(0, 1.05),
            ylabel=metric,
            title=f"{metric.capitalize()} at the anaemia cutoff",
        )
        ax.set_xticklabels(heads)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Screening accuracy, pooled against each site")
    fig.tight_layout()
    destination = outdir / "screening_by_site.png"
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def _read_predictions(path: Path) -> dict[str, dict[str, np.ndarray]]:
    per_head: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            head = row.get("head", "")
            bucket = per_head.setdefault(head, {"true": [], "pred": []})
            bucket["true"].append(_num(row.get("hb_true")))
            bucket["pred"].append(_num(row.get("hb_pred")))
    return {
        head: {
            "true": np.asarray(bucket["true"], dtype=float),
            "pred": np.asarray(bucket["pred"], dtype=float),
        }
        for head, bucket in per_head.items()
    }


def figure_predictions(predictions: dict[str, dict[str, np.ndarray]], outdir: Path) -> list[Path]:
    """Predicted-versus-true and Bland-Altman, one panel per head."""
    heads = sorted(predictions)
    if not heads:
        return []
    written: list[Path] = []

    fig, axes = plt.subplots(1, len(heads), figsize=(4.6 * len(heads), 4.6), squeeze=False)
    for ax, head in zip(axes[0], heads, strict=True):
        true = predictions[head]["true"]
        pred = predictions[head]["pred"]
        lo, hi = 5.0, 19.0
        ax.plot([lo, hi], [lo, hi], "--", color="#888888", linewidth=1.0)
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="#1b6ca8", alpha=0.08)
        ax.scatter(true, pred, s=16, color=COLOURS[0], alpha=0.75, edgecolors="none")
        ax.set(
            xlim=(lo, hi),
            ylim=(lo, hi),
            xlabel="measured Hb (g/dL)",
            ylabel="estimated Hb (g/dL)",
            title=head,
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.suptitle("Estimated versus measured Hb, test fold; band is +-1 g/dL")
    fig.tight_layout()
    path = outdir / "predicted_vs_true.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, len(heads), figsize=(4.6 * len(heads), 4.2), squeeze=False)
    for ax, head in zip(axes[0], heads, strict=True):
        true = predictions[head]["true"]
        pred = predictions[head]["pred"]
        keep = np.isfinite(true) & np.isfinite(pred)
        mean = (true[keep] + pred[keep]) / 2.0
        diff = pred[keep] - true[keep]
        bias = float(np.mean(diff)) if diff.size else float("nan")
        sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
        ax.scatter(mean, diff, s=16, color=COLOURS[1], alpha=0.75, edgecolors="none")
        ax.axhline(bias, color="#333333", linewidth=1.2, label=f"bias {_signed(bias, 2)}")
        if np.isfinite(sd):
            ax.axhline(bias + 1.96 * sd, color="#888888", linestyle="--", linewidth=1.0)
            ax.axhline(bias - 1.96 * sd, color="#888888", linestyle="--", linewidth=1.0)
        ax.set(xlabel="mean of the two (g/dL)", ylabel="estimate - measured (g/dL)", title=head)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Bland-Altman, test fold; dashed lines are the 95% limits of agreement")
    fig.tight_layout()
    path = outdir / "bland_altman.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ARTIFACT_REPORTS / "results.json")
    parser.add_argument("--sweep", type=Path, default=ARTIFACT_REPORTS / "sweep.json")
    parser.add_argument("--predictions", type=Path, default=ARTIFACT_REPORTS / "predictions.csv")
    parser.add_argument("--out", type=Path, default=ARTIFACT_FIGURES)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"no results at {args.results}; run `hemolux train` first")
        return 1
    ensure_dirs()
    args.out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    block = markdown_block(payload)
    if args.sweep.exists():
        block += "\n" + sweep_block(json.loads(args.sweep.read_text(encoding="utf-8")))
    block_path = ARTIFACT_REPORTS / "evals_block.md"
    block_path.write_text(block, encoding="utf-8")
    print(block)

    written: list[Path] = []
    for figure in (figure_reliability, figure_screening):
        path = figure(payload, args.out)
        if path is not None:
            written.append(path)
    if args.predictions.exists():
        written.extend(figure_predictions(_read_predictions(args.predictions), args.out))

    print(f"\nwrote {block_path}")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
