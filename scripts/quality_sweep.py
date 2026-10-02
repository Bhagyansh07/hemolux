"""Reproduce the measurement behind ``QUALITY_ABSTAIN``.

The abstention gate refuses a prediction when the image is poor as well as when
the model is uncertain. That second condition compares the quality score against
``hemolux.data.quality.QUALITY_ABSTAIN``, and the constant was previously 0.35 --
below every score in the corpus, which made the condition inert.

This script is how the replacement value was arrived at. It scores every
photograph in the dataset, reports the distribution, the hard-gate rejections and
what fraction of survivors each candidate threshold would refuse, so the number is
traceable to a measurement rather than to a preference.

Usage::

    python scripts/quality_sweep.py

Requires the dataset at ``data/raw/eyes-defy-anemia/dataset anemia``. Takes a few
minutes over ~860 images. Exit code 1 means at least one candidate threshold
abstains on nothing, which is the failure this script exists to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running straight from a checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemolux.data.dataset import DEFAULT_ROOT
from hemolux.data.quality import QUALITY_ABSTAIN, check_quality
from hemolux.imagemeta import silence_corrupt_iccp

#: Candidates reported alongside the live constant, to bracket the region where
#: the branch starts doing something.
#:
#: 0.35 is here on purpose: it is the value the constant used to hold, and seeing
#: it abstain on nothing is what shows the old gate was one condition wide. Only
#: the live constant is allowed to be inert without failing the run.
CANDIDATES = (0.35, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def main() -> int:
    silence_corrupt_iccp()
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_ROOT)
    if not root.is_dir():
        print(f"dataset not found at {root}", file=sys.stderr)
        return 2

    rows = []
    for path in sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        rows.append(check_quality(image))

    if not rows:
        print(f"no readable images under {root}", file=sys.stderr)
        return 2

    scores = np.array([r.score for r in rows])
    passed = np.array([r.reason is None for r in rows])
    n_passed = int(passed.sum())

    print(f"root              {root}")
    print(f"images scored     {len(rows)}")
    print(f"hard-gate pass    {n_passed}   fail: {int((~passed).sum())}")
    print()
    print(f"all scores        min={scores.min():.4f}  p05={np.percentile(scores, 5):.4f}  "
          f"median={np.median(scores):.4f}  max={scores.max():.4f}")
    print(f"passing only      min={scores[passed].min():.4f}  "
          f"p05={np.percentile(scores[passed], 5):.4f}  "
          f"median={np.median(scores[passed]):.4f}")
    print()

    reasons: dict[str, int] = {}
    for r in rows:
        if r.reason is not None:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
    print("hard-gate rejections:")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<14} {count}")
    print()

    print(f"QUALITY_ABSTAIN = {QUALITY_ABSTAIN}")
    print(f"candidate thresholds over the {n_passed} passing images:")
    useless = []
    for t in CANDIDATES:
        n = int((scores[passed] < t).sum())
        share = n / max(1, n_passed)
        marker = "  <- live" if t == QUALITY_ABSTAIN else ""
        print(f"  < {t:.2f}  ->  {n:4d} abstain ({share:5.1%}){marker}")
        if share == 0.0:
            useless.append(t)
    print()
    live_refusals = int((scores[passed] < QUALITY_ABSTAIN).sum())
    if live_refusals == 0:
        print(f"RESULT: the live constant {QUALITY_ABSTAIN} abstains on no passing image")
        return 1

    inert = [t for t in CANDIDATES if int((scores[passed] < t).sum()) == 0]
    print(f"RESULT: the live constant refuses {live_refusals} of {n_passed} passing images "
          f"({live_refusals / max(1, n_passed):.1%})")
    if inert:
        print(f"  candidates that would refuse nothing, kept as reference: {inert}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
