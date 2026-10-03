"""Corpus validation: assert the data matches what the documentation claims.

Why this is a module and not a script
-------------------------------------
The project's own rule is that a claim in a docstring has to be checkable. The dataset module's docstring states the
corpus is 217 patients, 95 India and 122 Italy, that fifteen Italian haemoglobin
values are stored with a decimal comma, and that two mask filenames are
misspelled. Those are all *assertions about a real download*, and a script under
``scripts/`` that someone has to remember to run does not hold up to that.

So the checks live here, importable and runnable:

    hemolux validate            # prints a table, exit code 0/1
    from hemolux.validation import validate_corpus

The download itself is not in git (``.gitignore`` has ``data/raw/``), so every
check that needs the real corpus raises a clearly-labelled error rather than
silently passing when the data is absent. A validator that reports success
without having looked at anything is worse than no validator.

The synthetic fixture cannot substitute here
---------------------------------------------
``scripts/make_synthetic_fixture.py`` generates a tree from an assumed layout.
When that assumption was wrong -- and it was, in five ways -- the fixture
reproduced the assumption's mistakes exactly, so every test built on it passed
while the real loader could not read a single real folder. Checks here are
therefore written against literals read off the real files: patient 93's
``ELIMINATO``, patients 53-80's comma decimals, ``India/7``'s ``_papebral.png``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from hemolux.config import HB_VALID_MAX, HB_VALID_MIN, SYNTHETIC_MARKER
from hemolux.data.dataset import (
    DEFAULT_ROI,
    DEFAULT_ROOT,
    MASK_SUFFIX_ORDER,
    PatientRecord,
    build_records,
    crop_to_mask_bbox,
    discover_masks,
    load_mask,
    load_rgb,
    mask_matches,
    parse_hb,
)
from hemolux.data.quality import ROI_MAX_AREA, ROI_MIN_AREA
from hemolux.data.splits import severity_bin, site_holdout_folds

#: The six Italian folders that genuinely lack a forniceal mask. Named in the
#: dataset's own documentation, and confirmed by listing the directory. The
#: workbook's ``Note`` column claims 30 such patients, which is stale.
EXPECTED_NO_FORNICEAL: frozenset[str] = {
    "Italy/1",
    "Italy/35",
    "Italy/54",
    "Italy/58",
    "Italy/75",
    "Italy/109",
}

#: The only patient the dataset's authors withdrew.
EXPECTED_EXCLUDED: str = "Italy/93"

#: Counts measured on the real download. India 95 folders, Italy 123.
EXPECTED_TOTAL: int = 217
EXPECTED_BY_SITE: dict[str, int] = {"India": 95, "Italy": 122}

#: Haemoglobin means measured on the real download, comma decimals included.
#: A coercing loader reports Italy at 13.74 because it silently loses 15
#: patients from one contiguous block.
EXPECTED_MEAN_HB: dict[str, float] = {"India": 11.47, "Italy": 13.83}

#: The two sites in the corpus. Anything else means a workbook was misread.
EXPECTED_SITES: tuple[str, ...] = ("India", "Italy")

#: Severity bands, ordered as ``severity_bin`` names them. Kept here so the report
#: lists all four even when one holds nobody, which is the point.
BANDS: tuple[str, ...] = ("normal", "mild", "moderate", "severe")

#: The bounds the quality gate applies to an ROI, as a fraction of the frame.
#: Imported rather than restated: a validator that keeps its own copy of a
#: threshold ends up reporting on a gate that no longer exists.
#:
#: A mask outside this band is what a failed match or a near-empty segmentation
#: looks like, and roughly half the palpebral corpus sits outside it.
ROI_AREA_MIN = ROI_MIN_AREA
ROI_AREA_MAX = ROI_MAX_AREA

#: How far a mask's aspect ratio may disagree with its frame's before the
#: mask-to-frame pairing is treated as wrong. Both are portrait 4:3, so any real
#: disagreement is a mismatch, not a rounding difference.
ASPECT_TOLERANCE = 0.02


class Severity(StrEnum):
    """How a check is reported.

    The distinction between the three is not decoration. A FAIL is a defect: the
    corpus is not what the documentation says it is, and any number computed from
    it is wrong rather than imprecise. A WARN is a true fact about the corpus that
    constrains what may be concluded from it. The severe band being empty is a
    WARN -- nothing is broken, and sensitivity in that band simply cannot be
    computed here. A WARN that a reader skips is a WARN that did its job by being
    looked at, so the exit code ignores them.
    """

    FAIL = "FAIL"
    WARN = "WARN"
    OK = "OK"


@dataclass
class Check:
    """One assertion, with the numbers that made it pass or fail.

    ``severity`` is the level to report at when the check *passes*. A check that
    fails is always a FAIL, so the field never downgrades a failure into a
    footnote; it only decides whether a passing check is worth flagging.
    """

    label: str
    passed: bool
    detail: str = ""
    severity: Severity = Severity.OK

    @property
    def level(self) -> Severity:
        """The level this check is reported at."""
        return self.level_of(passed=self.passed, severity=self.severity)

    @staticmethod
    def level_of(*, passed: bool, severity: Severity) -> Severity:
        if not passed:
            return Severity.FAIL
        return severity

    def print(self) -> None:
        line = f"  [{self.level}] {self.label}"
        if self.detail:
            line += f"  --  {self.detail}"
        print(line)


@dataclass
class CorpusReport:
    """The outcome of a validation run."""

    root: Path
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether nothing failed.

        Warnings do not clear this. They are facts, not defects, so they do not
        belong in an exit code -- but they do need to be countable.
        """
        return not self.failures

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.level is Severity.FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.level is Severity.WARN]

    def print(self) -> None:
        print()
        print("=" * 78)
        print(f"Hemolux corpus validation -- {self.root}")
        print("=" * 78)
        if not self.checks:
            print("  no checks ran")
            return
        for check in self.checks:
            check.print()
        n_failed = len(self.failures)
        n_warned = len(self.warnings)
        n_ok = len(self.checks) - n_failed - n_warned
        print()
        print(f"  {n_ok} OK  {n_warned} WARN  {n_failed} FAIL")
        if self.failures:
            print()
            print("  A FAIL means a number computed from this corpus would be wrong:")
            for check in self.failures:
                print(f"    - {check.label}: {check.detail}")
        if self.warnings:
            print()
            print("  A WARN is a true fact that limits what may be concluded:")
            for check in self.warnings:
                print(f"    - {check.label}: {check.detail}")
        if self.stats:
            print()
            print("  corpus as measured:")
            for key, value in self.stats.items():
                print(f"    {key}: {value}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "ok": self.ok,
            "n_checks": len(self.checks),
            "n_failed": len(self.failures),
            "n_warned": len(self.warnings),
            "checks": [
                {
                    "label": c.label,
                    # bool() because the arithmetic above produces numpy.bool_ on
                    # at least one check -- numpy.float64 < float is a numpy bool --
                    # and json.dumps rejects it with "Object of type bool is not
                    # JSON serializable", which reads as if the label were the
                    # problem. Coerce at the boundary rather than at twenty call
                    # sites, so a future check cannot reintroduce it.
                    "passed": bool(c.passed),
                    "level": str(c.level),
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "stats": self.stats,
        }


def _looks_like_the_documented_corpus(root: Path) -> bool:
    """Whether ``root`` is the download the literal contract assertions describe.

    The discriminator has to be more than shape. The generated fixture reproduces
    the real layout deliberately -- same site directories, same workbook names, same
    mask naming -- because a fixture that standardised any of those would stop
    covering the branches it exists to cover. A check for ``India/`` and ``Italy/``
    therefore matches the fixture too, and an eight-patient corpus produced six
    FAILs about a 217-patient download, which is how a validator teaches its reader
    to ignore red lines.

    So the fixture marks itself. ``SYNTHETIC_MARKER`` is the prefix the generator
    already puts on every file it writes, and its presence is taken as the corpus
    declaring what it is. Beyond that, the documented root is accepted on path, and
    anything else with both site directories is accepted on shape -- so a user who
    unpacked the download somewhere custom still gets the checks that apply to them.
    Deliberately a heuristic rather than a gate: ``--contract`` is one flag away.
    """
    documented = Path(DEFAULT_ROOT)
    try:
        if root.resolve() == documented.resolve():
            return True
    except OSError:
        pass
    if any(entry.name.startswith(SYNTHETIC_MARKER) for entry in root.iterdir()):
        return False
    return (root / "India").is_dir() and (root / "Italy").is_dir()


def validate_corpus(
    root: Path,
    *,
    contract: bool | None = None,
    require_masks: bool = False,
    read_pixels: bool = True,
) -> CorpusReport:
    """Run the corpus checks and return the report.

    A :class:`CorpusReport`, not a dict. An earlier version returned
    ``{**report.as_dict(), "print": report.print, "report": report}`` so the CLI
    could serialise without importing the dataclass -- which put a bound method and
    a live object into the same mapping the caller was invited to ``json.dumps``.
    That raised ``TypeError: Object of type method is not JSON serializable`` on
    every ``--json`` run. A caller now gets one type and calls ``as_dict()`` on it.

    Parameters
    ----------
    root
        Where the corpus lives.
    contract
        Whether to check the *specific* assertions about the 217-patient Kaggle
        download -- the per-site counts, the Italian haemoglobin means, the six
        patients without a forniceal mask, ``Italy/93`` withdrawn. ``None`` means
        "only if this looks like that download", which is the default because a
        fixture of eight patients otherwise fails six assertions about a corpus it
        is not. ``True`` forces them, which is what you want when checking the
        real download by hand. ``False`` skips them.
    require_masks
        Promote a missing mask from a warning to a failure. Off by default because
        six Italian patients genuinely ship without a forniceal mask and the
        palpebral default is present for all 217.
    read_pixels
        Decode frames and masks to measure geometry. Off makes the run fast enough
        for a pre-commit hook, at the cost of leaving the geometry unverified --
        which the report then says out loud, rather than implying otherwise.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} does not exist. The Kaggle download is not in git (see .gitignore); "
            f"fetch it with `kaggle datasets download -d eyasdefy/eyes-defy-anemia` and "
            f"unpack into data/raw/."
        )
    run_contract = _looks_like_the_documented_corpus(root) if contract is None else bool(contract)

    report = CorpusReport(root=root)

    # ---------------------------------------------------------------- units --
    # These need no corpus, so they run first: if the parser is wrong, every
    # downstream number is wrong and the rest of the output would be noise.
    report.checks += _unit_checks()

    records = build_records(root, verbose=False, require_masks=require_masks)
    by_site: dict[str, list] = {}
    for r in records:
        by_site.setdefault(r.site, []).append(r)

    # ------------------------------------------------------- corpus contract --
    # Everything from here to the marker below asserts *this specific corpus*:
    # 217 patients, 95 and 122 by site, six Italians without a forniceal mask,
    # Italy/93 withdrawn. Those are true statements about the Kaggle download and
    # false ones about any other directory -- so a synthetic fixture of eight
    # patients produced five FAILs about a corpus it is not, which is how a
    # validator teaches its reader to ignore red lines.
    #
    # So the contract runs by default only against the documented root, and
    # otherwise only when the caller demands it. Everything after the marker is
    # structural and runs against whatever it is given.
    contract_start = len(report.checks)

    # ---------------------------------------------------------------- counts --
    report.checks.append(
        Check(
            f"all {EXPECTED_TOTAL} usable patients load",
            len(records) == EXPECTED_TOTAL,
            f"got {len(records)}",
        )
    )
    for site, expected in sorted(EXPECTED_BY_SITE.items()):
        got = len(by_site.get(site, []))
        report.checks.append(
            Check(f"{site} holds {expected} patients", got == expected, f"got {got}")
        )

    report.checks.append(
        Check(
            "the withdrawn patient is the only one dropped",
            EXPECTED_EXCLUDED not in {r.patient_id for r in records},
            f"{EXPECTED_EXCLUDED} present",
        )
    )

    # ------------------------------------------------------- identity safety --
    ids = [r.patient_id for r in records]
    report.checks.append(Check("patient ids are unique", len(ids) == len(set(ids))))
    report.checks.append(Check("patient ids are site-qualified", all("/" in i for i in ids)))
    report.checks.append(
        Check(
            "India/1 and Italy/1 are distinct patients",
            {"India/1", "Italy/1"} <= set(ids),
            "both sites number their folders from 1, so a bare number would collide",
        )
    )

    # ------------------------------------------------------------- decoding --
    # The single most consequential check in the file. A loader that coerces
    # the comma decimals drops 15 Italian patients in one contiguous block,
    # which is exactly the per-site distribution the fairness audit compares.
    italy = by_site.get("Italy", [])
    italy_hb = np.array([r.hb for r in italy], dtype=np.float64)
    report.checks.append(
        Check(
            "Italian mean Hb is 13.83, not the 13.74 a coercing loader reports",
            abs(italy_hb.mean() - EXPECTED_MEAN_HB["Italy"]) < 0.02,
            f"got {italy_hb.mean():.3f} over n={len(italy_hb)}",
        )
    )
    india_hb = np.array([r.hb for r in by_site.get("India", [])], dtype=np.float64)
    if india_hb.size:
        report.checks.append(
            Check(
                "Indian mean Hb is 11.47",
                abs(india_hb.mean() - EXPECTED_MEAN_HB["India"]) < 0.02,
                f"got {india_hb.mean():.3f} over n={len(india_hb)}",
            )
        )
    report.checks.append(
        Check(
            "the decimal-comma block parses rather than being dropped",
            parse_hb("15,1") == 15.1 and parse_hb("_") is None,
            "Italy.xlsx patients 53-80 store Hb as '15,1'; patient 93 as '_'",
        )
    )

    # ----------------------------------------------------------------- masks --
    masks_by_id = {r.patient_id: r.masks for r in records}
    no_default = [pid for pid, m in masks_by_id.items() if DEFAULT_ROI not in m]
    report.checks.append(
        Check(
            f"every kept patient has the default ROI ({DEFAULT_ROI})",
            not no_default,
            f"missing on {no_default}" if no_default else "includes the two misspelled filenames",
        )
    )
    no_forniceal = {pid for pid, m in masks_by_id.items() if "forniceal" not in m}
    report.checks.append(
        Check(
            "exactly six patients lack a forniceal mask, all Italian",
            no_forniceal == EXPECTED_NO_FORNICEAL,
            f"got {sorted(no_forniceal)}; the Note column claims 30, which is stale",
        )
    )
    india8 = root / "India" / "8"
    if india8.is_dir():
        masks8 = discover_masks(india8)
        report.checks.append(
            Check(
                "India/8 collapses its browser '(1)' duplicates to three masks",
                len(masks8) == 3 and all("(1)" not in p.name for p in masks8.values()),
                f"got {sorted(masks8)}",
            )
        )
    italy1 = root / "Italy" / "1"
    if italy1.is_dir():
        masks1 = discover_masks(italy1)
        report.checks.append(
            Check(
                "Italy/1 resolves its zero-padded mask prefix",
                "palpebral" in masks1,
                f"files: {sorted(p.name for p in italy1.iterdir())}",
            )
        )
    report.checks.append(
        Check(
            "the two misspelled mask names resolve to the right ROI",
            mask_matches("20200124_202058_papebral") == "palpebral"
            and mask_matches("T_64_20190612_092742_forniceal_palplebral") == "forniceal_palpebral",
            "India/7 ships _papebral.png, Italy/95 ships _forniceal_palplebral.png",
        )
    )

    # --------------------------------------------------- structure, geometry --
    # ---- end of corpus contract -------------------------------------------

    if not run_contract:
        del report.checks[contract_start:]
    else:
        report.checks += [
            Check(
                "the corpus contract was checked",
                True,
                f"{len(report.checks) - contract_start} assertions about the "
                f"{EXPECTED_TOTAL}-patient download",
            )
        ]

    # The contract checks above ask "is this the corpus the documentation names?".
    # These ask "what is actually in it, and what does that forbid concluding?".
    # Both belong in one report: a reader who has only run one of them has no way
    # to know what the other would have told them.
    report.checks += _domain_checks(records)
    report.checks += _band_checks(records)
    report.checks += _geometry_checks(records, read_pixels=read_pixels)
    report.checks += _holdout_checks(records)

    # ------------------------------------------------------ the confounding --
    # Not a pass/fail check on the data, but the fact that makes the fairness
    # analysis hard, so it is printed as a check that can only pass by being
    # present. Site moves age and sex along with pigmentation.
    ages = {
        site: float(np.nanmean([r.age for r in rows]))
        for site, rows in sorted(by_site.items())
        if rows
    }
    sexes = {site: Counter(r.sex for r in rows) for site, rows in sorted(by_site.items())}
    prevalence: dict[str, float] = {}
    for site, rows in sorted(by_site.items()):
        anaemic = sum(1 for r in rows if (r.hb < 13.0 if r.sex == "M" else r.hb < 12.0))
        prevalence[site] = 100.0 * anaemic / len(rows)
    sex_summary = {site: dict(counts) for site, counts in sorted(sexes.items())}
    report.checks.append(
        Check(
            "the site confound is present and will be reported",
            len(ages) == 2 and all(age < 60 for age in ages.values()),
            f"mean age {ages}; sex {sex_summary}",
        )
    )

    report.stats = {
        "n_patients": len(records),
        "per_site": {
            site: {
                "n": len(rows),
                "hb_mean": round(float(np.mean([r.hb for r in rows])), 3),
                "hb_sd": round(float(np.std([r.hb for r in rows], ddof=1)), 3),
                "age_mean": round(ages.get(site, float("nan")), 2),
                "anaemic_pct": round(prevalence.get(site, float("nan")), 1),
                "sex": dict(sorted(Counter(r.sex for r in rows).items())),
            }
            for site, rows in sorted(by_site.items())
        },
        "roi": DEFAULT_ROI,
    }

    return report


def _domain_checks(records: list[PatientRecord]) -> list[Check]:
    """Labels must be physiologically possible and categorically known.

    The corpus contract checks whether the counts match. These check whether the
    *values* could have come from a human, which a count assertion cannot tell: a
    workbook column read one row off produces the right number of patients and an
    impossible haemoglobin.
    """
    checks: list[Check] = []

    out_of_range = [
        f"{r.patient_id} hb={r.hb:g}" for r in records if not (HB_VALID_MIN <= r.hb <= HB_VALID_MAX)
    ]
    checks.append(
        Check(
            f"every haemoglobin value lies in [{HB_VALID_MIN:g}, {HB_VALID_MAX:g}] g/dL",
            not out_of_range,
            (
                f"{len(out_of_range)} outside: {', '.join(out_of_range[:8])}"
                if out_of_range
                else f"n={len(records)}"
            ),
        )
    )

    sexes = sorted({r.sex for r in records})
    unexpected = set(sexes) - {"F", "M"}
    checks.append(
        Check(
            "sex is one of the two values the severity bands are defined for",
            True,
            (
                f"got {sexes}; the band for any other value is undefined, so those "
                "patients drop out of every per-band figure"
                if unexpected
                else f"got {sexes}"
            ),
            severity=Severity.WARN if unexpected else Severity.OK,
        )
    )

    sites = sorted({r.site for r in records})
    checks.append(
        Check(
            f"site is one of {list(EXPECTED_SITES)}",
            set(sites) <= set(EXPECTED_SITES),
            f"got {sites}",
        )
    )

    return checks


def _band_checks(records: list[PatientRecord]) -> list[Check]:
    """Report the severity histogram and flag any band that holds nobody.

    This is a limit on the conclusions rather than a defect. An empty severe band
    means sensitivity in that band cannot be computed from this corpus at all, and
    a pooled score that quietly averages over the bands hides exactly that.
    """
    checks: list[Check] = []
    bands = Counter(severity_bin(r.hb, r.sex) for r in records)
    n = len(records)

    for band in BANDS:
        count = bands.get(band, 0)
        share = 100.0 * count / n if n else 0.0
        checks.append(
            Check(
                f"severity band {band}",
                True,
                f"{count:>4} patients ({share:5.1f}%)",
            )
        )

    # The per-band rows above stay OK whatever the count. A band holding nobody is
    # not wrong, and colouring it WARN as well would report the same fact twice --
    # once in the table a reader skims, and once in the summary a reader reads.
    empty = [b for b in BANDS if bands.get(b, 0) == 0]
    if empty:
        checks.append(
            Check(
                f"no patients in {empty}",
                True,
                "sensitivity in that band cannot be computed from this corpus, and no "
                "pooled figure should be read as covering it",
                severity=Severity.WARN,
            )
        )
    return checks


def _geometry_checks(records: list[PatientRecord], *, read_pixels: bool) -> list[Check]:
    """Frame and mask geometry, reported as facts about the corpus.

    Masks are not stored at the frame resolution: every frame is 3984x2988 and
    almost every mask is 800x1067, about a quarter of the area. So area is reported
    as a *fraction*, which is resolution-independent, and a raw shape comparison
    against the frame would call every patient broken.
    """
    checks: list[Check] = []

    if not read_pixels:
        return [
            Check(
                "pixel geometry not verified",
                True,
                "run without --no-pixels to check frame sizes, mask resolutions and ROI areas",
                severity=Severity.WARN,
            )
        ]

    sizes: Counter[tuple[int, ...]] = Counter()
    for record in records:
        sizes[load_rgb(record.image).shape[:2]] += 1
    (height, width), _top = sizes.most_common(1)[0]
    if len(sizes) == 1:
        detail = f"{width}x{height} in all {len(records)}"
    else:
        detail = (
            f"{len(sizes)} sizes: "
            + ", ".join(f"{w}x{h} x{c}" for (h, w), c in sorted(sizes.items()))
            + "; a sharpness threshold fixed for one is not comparable across the rest"
        )
    checks.append(
        Check(
            "frames are a single resolution",
            True,
            detail,
            severity=Severity.OK if len(sizes) == 1 else Severity.WARN,
        )
    )

    resolutions: dict[tuple[int, ...], int] = {}
    for record in records:
        for path in record.masks.values():
            mask = load_mask(path)
            if mask is not None:
                resolutions[mask.shape[:2]] = resolutions.get(mask.shape[:2], 0) + 1
    if resolutions:
        summary = ", ".join(f"{w}x{h} x{c}" for (h, w), c in sorted(resolutions.items()))
        checks.append(
            Check(
                "stored mask resolutions",
                True,
                summary,
                severity=Severity.WARN if len(resolutions) > 1 else Severity.OK,
            )
        )

    for roi in MASK_SUFFIX_ORDER:
        present = sum(1 for r in records if roi in r.masks)
        # Six Italian patients genuinely ship without a forniceal mask, so this is
        # a fact about the corpus rather than a defect. Only --require-masks turns
        # it into one.
        checks.append(
            Check(
                f"roi {roi} is matched for every patient",
                True,
                f"{present}/{len(records)}"
                + ("" if present == len(records) else f"; {len(records) - present} unmatched"),
                severity=Severity.OK if present == len(records) else Severity.WARN,
            )
        )

        fractions: list[float] = []
        misaligned: list[str] = []
        uncroppable: list[str] = []
        for record in records:
            path = record.masks.get(roi)
            if path is None:
                continue
            mask = load_mask(path)
            if mask is None:
                uncroppable.append(record.patient_id)
                continue
            mask_h, mask_w = mask.shape[:2]
            rgb = load_rgb(record.image)
            frame_h, frame_w = rgb.shape[:2]
            if abs((mask_w / mask_h) / (frame_w / frame_h) - 1.0) > ASPECT_TOLERANCE:
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
            checks.append(
                Check(
                    f"roi {roi} area fractions could not be measured",
                    False,
                    "no mask decoded",
                )
            )
            continue

        checks.append(
            Check(
                f"roi {roi} area fraction",
                True,
                _quantiles(fractions),
                severity=Severity.OK,
            )
        )
        outside = sum(1 for f in fractions if not (ROI_AREA_MIN <= f <= ROI_AREA_MAX))
        if outside:
            checks.append(
                Check(
                    f"roi {roi} area fraction within [{ROI_AREA_MIN:g}, {ROI_AREA_MAX:g}]",
                    True,
                    f"{outside} of {len(fractions)} outside, which is what a failed match "
                    "or a near-empty segmentation looks like",
                    severity=Severity.WARN,
                )
            )
        if misaligned:
            checks.append(
                Check(
                    f"roi {roi} masks agree with their frame's aspect ratio",
                    False,
                    f"{len(misaligned)} contradict it, so the pairing cannot be right: "
                    f"{misaligned[:6]}",
                )
            )
        if uncroppable:
            checks.append(
                Check(
                    f"roi {roi} masks yield a crop",
                    True,
                    f"{len(uncroppable)} yield none: {uncroppable[:6]}",
                    severity=Severity.WARN,
                )
            )

    return checks


def _holdout_checks(records: list[PatientRecord]) -> list[Check]:
    """Both directions of the site holdout must actually contain patients.

    A patient whose site label is blank vanishes from both directions at once, so a
    holdout can look populated while quietly shrinking the population a reported MAE
    was computed over.
    """
    site_of = {r.patient_id: r.site for r in records}
    checks: list[Check] = []
    for train_site, test_site in (("India", "Italy"), ("Italy", "India")):
        evaluated = sum(
            len(fold.test)
            for fold in site_holdout_folds(site_of, site_of, sites=(train_site, test_site))
        )
        checks.append(
            Check(
                f"holdout {train_site} -> {test_site} evaluates patients",
                evaluated > 0,
                f"{evaluated} patients",
            )
        )
    return checks


def _quantiles(values: Sequence[float], fmt: str = "{:,.4g}") -> str:
    """min, p10, median, max -- the shape of a distribution in one line.

    Significant digits rather than fixed decimals, because the smallest area
    fraction in the corpus is 0.000001 and a four-decimal format prints that as
    ``0.0000``. A quantile that reads as exactly zero is worse than no quantile.
    """
    ordered = sorted(values)
    if not ordered:
        return "no data"
    if len(ordered) == 1:
        return fmt.format(ordered[0])
    return (
        f"min {fmt.format(ordered[0])}  p10 {fmt.format(ordered[len(ordered) // 10])}  "
        f"median {fmt.format(float(np.median(ordered)))}  max {fmt.format(ordered[-1])}"
    )


def _unit_checks() -> list[Check]:
    """Parser and mask-matching checks, which need no download.

    Run before anything touches the corpus so that a broken parser is reported as
    a broken parser rather than as a wrong patient count.
    """
    checks: list[Check] = []

    cases: list[tuple[object, object]] = [
        ("13.7", 13.7),
        ("15,1", 15.1),
        ("  13,7 ", 13.7),
        ("15", 15.0),
        (13, 13.0),
        ("_", None),
        (float("nan"), None),
        (None, None),
    ]
    bad = [(raw, got) for raw, got in cases if parse_hb(raw) != got]
    checks.append(
        Check(
            "parse_hb reads dots, commas, markers and NaN",
            not bad,
            f"wrong on {bad}" if bad else f"{len(cases)} cases",
        )
    )

    for text, reason in (("1,234", "thousands"), ("13.7,1", "ambiguous")):
        try:
            parse_hb(text)
        except ValueError:
            checks.append(Check(f"parse_hb refuses {text!r} ({reason})", True))
        else:
            checks.append(
                Check(f"parse_hb refuses {text!r} ({reason})", False, "it returned a value")
            )

    stems: list[tuple[str, object]] = [
        ("20200118_164733_forniceal", "forniceal"),
        ("20200118_164733_palpebral", "palpebral"),
        ("20200118_164733_forniceal_palpebral", "forniceal_palpebral"),
        ("001_palpebral", "palpebral"),
        ("20200124_202947_palpebral(1)", "palpebral"),
        ("20200124_202058_papebral", "palpebral"),
        ("T_64_20190612_092742_palplebral", "palpebral"),
        ("T_64_20190612_092742_forniceal_palplebral", "forniceal_palpebral"),
        ("20200118_164733", None),
        ("1", None),
    ]
    wrong = [(s, mask_matches(s), want) for s, want in stems if mask_matches(s) != want]
    checks.append(
        Check(
            "mask_matches handles suffixes, variants, typos and photographs",
            not wrong,
            f"wrong on {wrong}" if wrong else f"{len(stems)} real stems",
        )
    )

    return checks
