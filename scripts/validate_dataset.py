"""Validate a corpus before anything downstream of it is allowed to run.

Why this exists
---------------
Two places in this repository already point at it, and until now it did not
exist. That is the whole reason it does: a fixture generator that ends by telling
the reader which command to run next is only useful if that command exists.

What it checks, and what the exit code means
--------------------------------------------
``FAIL`` is a property that would make a number produced from this corpus wrong
rather than merely imprecise: a duplicated patient identity, a haemoglobin value
outside the physiological range, an identifier whose site prefix contradicts the
directory it was found in, a mask whose shape cannot belong to its image. Any
``FAIL`` exits non-zero, so a pipeline that calls this first cannot proceed on a
corpus that would mislead it.

``WARN`` is a fact about the corpus that constrains what may be concluded from
it. An empty severity band is the important one here: this dataset has no severe
patients, so sensitivity in that band cannot be computed at all. That is not a
defect to be fixed but a limit to be published.

``OK`` lines are printed so the report is evidence rather than a complaint, and so
that a later change in corpus shape shows up as a difference in the output.

Usage
-----
    python scripts/validate_dataset.py                 # the real corpus
    python scripts/validate_dataset.py --synthetic     # the generated fixture
    python scripts/validate_dataset.py --root PATH
    python scripts/validate_dataset.py --no-pixels     # skip image decoding

Mask geometry is reported as an area *fraction*, which is resolution-independent,
because the released masks are not stored at the frame resolution. That is a
property of the release and the loader resizes to match; a check that compared raw
mask shape to raw image shape would report every patient as broken.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics as st
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

# Allow running straight from a checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from numpy.typing import NDArray

from hemolux.config import HB_VALID_MAX, HB_VALID_MIN, PATHS
from hemolux.data.dataset import (
    DEFAULT_ROOT,
    MASK_SUFFIX_ORDER,
    PatientRecord,
    build_records,
    crop_to_mask_bbox,
    load_mask,
    load_rgb,
)
from hemolux.data.splits import severity_bin, site_holdout_folds

SITES = ("India", "Italy")
EXPECTED_SEX = {"M", "F"}
BANDS = ("normal", "mild", "moderate", "severe")

#: An ROI smaller than this fraction of the frame matched nothing; larger than this
#: and the "segmentation" is really the photograph. These are the bounds the image
#: quality gate applies, restated here only because this script must not import the
#: gate to check the gate; a test asserts the two agree.
ROI_AREA_MIN = 0.015
ROI_AREA_MAX = 0.85

#: How far a mask's aspect ratio may differ from its image's before the pairing is
#: declared broken rather than merely inconsistent.
ASPECT_TOLERANCE = 0.02


@contextlib.contextmanager
def _silence_icc_warnings() -> Iterator[None]:
    """Drop libpng's iCCP chatter for the duration of the block.

    The released masks are PNG files whose authors' tooling left a malformed iCCP
    chunk behind. A mask is binary region data with no colour profile to manage, so
    the warning is noise, and 96 lines of it drown out the report this script
    exists to produce. Redirecting the file descriptor is the only way to stop
    libpng, which writes from C and does not consult Python's warning filters.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


class Report:
    """Collects findings and owns the exit code.

    Separating collection from presentation keeps every check a single expression
    that returns a verdict, which is the same shape the fairness gate uses.
    """

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.oks: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def ok(self, message: str) -> None:
        self.oks.append(message)

    def render(self) -> str:
        out = ["", "FAIL"]
        out += [f"  {m}" for m in self.failures] or ["  none"]
        out += ["", "WARN"]
        out += [f"  {m}" for m in self.warnings] or ["  none"]
        out += ["", "OK"]
        out += [f"  {m}" for m in self.oks] or ["  none"]
        return "\n".join([*out, ""])


def _quantiles(values: Iterable[float], fmt: str = "{:,.4f}") -> str:
    ordered = sorted(values)
    if not ordered:
        return "no data"
    if len(ordered) == 1:
        return fmt.format(ordered[0])
    return (
        f"min {fmt.format(ordered[0])}  p10 {fmt.format(ordered[len(ordered) // 10])}  "
        f"median {fmt.format(st.median(ordered))}  max {fmt.format(ordered[-1])}"
    )


def check_identities(records: list[PatientRecord], rep: Report) -> None:
    """Every patient must have exactly one identity, and it must agree with its site.

    Both countries number their folders from 1, so a bare folder number is not an
    identity. A collision here would place one patient in a training split and
    their namesake in the test split, which is the single most damaging defect this
    repository could ship.
    """
    ids = [r.patient_id for r in records]
    duplicates = sorted(pid for pid, n in Counter(ids).items() if n > 1)
    if duplicates:
        rep.fail(f"duplicated patient identifiers: {duplicates}")
    else:
        rep.ok(f"{len(ids)} patient identifiers, all distinct")

    bad_qualifier = [r.patient_id for r in records if "/" not in r.patient_id]
    if bad_qualifier:
        rep.fail(f"identifiers that are not site-qualified: {bad_qualifier[:8]}")

    mismatched = [
        r.patient_id
        for r in records
        if "/" in r.patient_id and r.patient_id.split("/", 1)[0] != r.site
    ]
    if mismatched:
        rep.fail(
            f"{len(mismatched)} identifier(s) whose site prefix contradicts their directory: "
            f"{mismatched[:8]}"
        )
    if not bad_qualifier and not mismatched:
        rep.ok("every identifier is site-qualified and agrees with its directory")


def check_labels(records: list[PatientRecord], rep: Report) -> None:
    """Haemoglobin must be physiologically possible, and the categoricals known."""
    out_of_range = [
        f"{r.patient_id} hb={r.hb:g}" for r in records if not (HB_VALID_MIN <= r.hb <= HB_VALID_MAX)
    ]
    if out_of_range:
        head = ", ".join(out_of_range[:8])
        more = " ..." if len(out_of_range) > 8 else ""
        rep.fail(f"haemoglobin outside [{HB_VALID_MIN:g}, {HB_VALID_MAX:g}] g/dL: {head}{more}")
    else:
        rep.ok(f"all haemoglobin values within [{HB_VALID_MIN:g}, {HB_VALID_MAX:g}] g/dL")

    unknown_sex = sorted({r.sex for r in records if r.sex not in EXPECTED_SEX})
    if unknown_sex:
        rep.warn(f"unexpected sex values {unknown_sex}; severity bands assume M or F")
    else:
        rep.ok(f"sex values are exactly {sorted(EXPECTED_SEX)}")

    unknown_site = sorted({r.site for r in records if r.site not in SITES})
    if unknown_site:
        rep.fail(f"unexpected site values {unknown_site}; expected {list(SITES)}")
    else:
        rep.ok(f"site values are exactly {list(SITES)}")


def check_severity_coverage(records: list[PatientRecord], rep: Report) -> None:
    """Report the band histogram, and warn about any band that is empty.

    This is a limit on the conclusions rather than a defect. An empty severe band
    means sensitivity in that band cannot be computed from this corpus at all, and
    a pooled score that quietly averages over the bands hides exactly that.
    """
    bands = Counter(severity_bin(r.hb, r.sex) for r in records)
    for band in BANDS:
        count = bands.get(band, 0)
        share = 100.0 * count / len(records) if records else 0.0
        rep.ok(f"band {band:<9} {count:>4} patients  ({share:5.1f}%)")
    empty = [b for b in BANDS if bands.get(b, 0) == 0]
    if empty:
        rep.warn(
            f"no patients in {empty}: sensitivity in that band cannot be computed from this "
            "corpus, and no pooled figure should be read as covering it"
        )


def check_frames(records: list[PatientRecord], rep: Report, *, read_pixels: bool) -> None:
    """Frame size drives the sharpness threshold, so more than one size is a warning."""
    if not read_pixels:
        rep.warn("pixel checks skipped (--no-pixels); frame size and mask geometry unverified")
        return
    sizes = Counter()
    for record in records:
        rgb: NDArray[np.uint8] = load_rgb(record.image)
        sizes[rgb.shape[:2]] += 1
    (height, width), top = sizes.most_common(1)[0]
    rep.ok(
        f"{len(sizes)} distinct frame size(s); {width}x{height} in {top} of {len(records)} frames"
    )
    if len(sizes) > 1:
        rep.warn(
            "frames are not a single resolution; a sharpness threshold fixed for one "
            "resolution is not comparable across the others"
        )


def check_masks(records: list[PatientRecord], rep: Report, *, read_pixels: bool) -> None:
    """Check that masks matched, that their geometry is plausible, and that they align.

    An unmatched mask does not raise anywhere downstream; it degrades a measurement
    quietly, which is the worst failure mode a mask has. So the matched count is
    reported outright rather than being inferred from a run that appeared to work.
    """
    for roi in MASK_SUFFIX_ORDER:
        present = sum(1 for r in records if roi in r.masks)
        line = f"roi {roi:<20} matched {present:>4}/{len(records)}"
        if present < len(records):
            rep.warn(f"{line}; {len(records) - present} patient(s) have no {roi} mask")
        else:
            rep.ok(line)

    if not read_pixels:
        return

    resolutions: dict[tuple[int, int], int] = {}
    for record in records:
        for path in record.masks.values():
            mask = load_mask(path)
            if mask is not None:
                resolutions[mask.shape[:2]] = resolutions.get(mask.shape[:2], 0) + 1
    if resolutions:
        summary = ", ".join(f"{w}x{h} x{n}" for (h, w), n in sorted(resolutions.items()))
        line = f"stored mask resolutions: {summary}"
        if len(resolutions) == 1:
            rep.ok(line)
        else:
            rep.warn(
                f"{line}; masks are not all stored at one size, so a raw shape comparison "
                "against the frame would call every patient broken"
            )

    for roi in MASK_SUFFIX_ORDER:
        fractions: list[float] = []
        misaligned: list[str] = []
        uncroppable: list[str] = []
        for record in records:
            path = record.masks.get(roi)
            if path is None:
                continue
            mask = load_mask(path)
            rgb = load_rgb(record.image)
            if mask is None:
                uncroppable.append(record.patient_id)
                continue
            mask_h, mask_w = mask.shape[:2]
            frame_h, frame_w = rgb.shape[:2]
            mask_aspect = mask_w / mask_h
            frame_aspect = frame_w / frame_h
            if abs(mask_aspect - frame_aspect) / frame_aspect > ASPECT_TOLERANCE:
                misaligned.append(record.patient_id)
                continue
            fractions.append(float(mask.sum()) / float(mask.size))
            upscaled = np.zeros((frame_h, frame_w), dtype=bool)
            ys = (np.arange(mask_h) * frame_h / mask_h).astype(np.int64)
            xs = (np.arange(mask_w) * frame_w / mask_w).astype(np.int64)
            upscaled[np.ix_(ys, xs)] = mask
            if crop_to_mask_bbox(rgb, upscaled) is None:
                uncroppable.append(record.patient_id)

        if not fractions:
            rep.warn(f"roi {roi:<20} no mask could be measured")
            continue
        rep.ok(f"roi {roi:<20} area fraction  {_quantiles(fractions)}")
        outside = sum(1 for f in fractions if not (ROI_AREA_MIN <= f <= ROI_AREA_MAX))
        if outside:
            rep.warn(
                f"roi {roi:<20} {outside} mask(s) outside [{ROI_AREA_MIN:g}, {ROI_AREA_MAX:g}] "
                "of the frame, which is what a failed match or a near-empty mask looks like"
            )
        if misaligned:
            rep.fail(
                f"roi {roi:<20} {len(misaligned)} mask(s) whose aspect ratio contradicts the "
                f"frame, so the pairing cannot be right: {misaligned[:6]}"
            )
        if uncroppable:
            rep.warn(f"roi {roi:<20} {len(uncroppable)} mask(s) yield no crop: {uncroppable[:6]}")


def check_holdout(records: list[PatientRecord], rep: Report) -> None:
    """Both directions of the site holdout must actually contain patients.

    A patient whose site label is blank vanishes from both directions at once, so a
    holdout can look populated while quietly shrinking the population a reported MAE
    was computed over.
    """
    site_of = {r.patient_id: r.site for r in records}
    for train_site, test_site in (("India", "Italy"), ("Italy", "India")):
        evaluated = sum(
            len(fold.test)
            for fold in site_holdout_folds(site_of, site_of, sites=(train_site, test_site))
        )
        if evaluated == 0:
            rep.fail(f"holdout {train_site} -> {test_site} evaluates no patients")
        else:
            rep.ok(f"holdout {train_site} -> {test_site} evaluates {evaluated} patients")


def summarise(records: list[PatientRecord]) -> None:
    print(f"patients: {len(records)}")
    for site in SITES:
        subset = [r for r in records if r.site == site]
        if not subset:
            continue
        hb = [r.hb for r in subset]
        print(
            f"  {site:<6} n={len(subset):<4} hb mean {st.mean(hb):5.2f}  sd {st.stdev(hb):4.2f}  "
            f"range {min(hb):.1f}-{max(hb):.1f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="validate the generated fixture")
    source.add_argument("--root", type=Path, help="explicit corpus root")
    parser.add_argument(
        "--require-masks", action="store_true", help="a missing mask is a failure, not a warning"
    )
    parser.add_argument("--no-pixels", action="store_true", help="skip all image decoding")
    parser.add_argument(
        "--allow-icc-warnings",
        action="store_true",
        help="let libpng's iCCP warnings through instead of suppressing them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.synthetic:
        root, label = PATHS.synthetic, "synthetic fixture"
    elif args.root is not None:
        root, label = args.root, str(args.root)
    else:
        root, label = Path(DEFAULT_ROOT), "Eyes Defy Anemia corpus"

    print(f"validating {label}")
    print(f"  root: {root}")
    if label != "synthetic fixture":
        print("  the corpus is not redistributed here and is never committed; see docs/DATASET.md")
    print()

    if not root.is_dir():
        print(f"FAIL\n  {root} does not exist", file=sys.stderr)
        print(
            "\nRun scripts/make_synthetic_fixture.py to generate a stand-in corpus.",
            file=sys.stderr,
        )
        return 1

    # build_records raises rather than returning a short list when the layout is
    # wrong, which is the right behaviour for a library: a caller that ignores it
    # gets a stack trace. A validator that then propagates the traceback is not
    # reporting anything, so the failure is caught and stated in its own terms.
    try:
        records = build_records(root, require_masks=args.require_masks)
    except (FileNotFoundError, ValueError) as exc:
        rep = Report()
        rep.fail(f"the corpus could not be read: {exc}")
        print(rep.render(), file=sys.stderr)
        return 1

    if not records:
        print("FAIL\n  no usable patients: every workbook row was dropped", file=sys.stderr)
        return 1

    rep = Report()
    summarise(records)
    print()
    with contextlib.nullcontext() if args.allow_icc_warnings else _silence_icc_warnings():
        check_frames(records, rep, read_pixels=not args.no_pixels)
        check_masks(records, rep, read_pixels=not args.no_pixels)
    check_identities(records, rep)
    check_labels(records, rep)
    check_severity_coverage(records, rep)
    check_holdout(records, rep)

    print(rep.render())
    if rep.failures:
        print(f"{len(rep.failures)} failure(s). Numbers from this corpus would be wrong.")
        return 1
    if rep.warnings:
        print(f"no failures, {len(rep.warnings)} warning(s). Read them before quoting a number.")
    else:
        print("no failures, no warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
