"""Tests for the dataset loader.

Every case in this file corresponds to something the real download actually
contained. The synthetic fixture could not have found any of them: the fixture
was generated from an assumption about the layout, and the assumption was wrong,
so the fixture reproduced the assumption's mistakes faithfully. The list below is
the corrected inventory of what the release really does.

The two most consequential:

* ``Italy.xlsx`` stores 15 of its 123 haemoglobin values as **text with a
  European decimal comma** (``'15,1'``). A ``pd.to_numeric(errors="coerce")``
  turns them into NaN and drops patients 53-80 -- one contiguous block, so the
  surviving Italian subset is biased by construction, and that subset is exactly
  what the fairness audit compares across sites.
* Both sites number their folders from 1, so ``India/1`` and ``Italy/1`` are two
  different patients. Using the bare folder number as identity risks putting one
  in train and the other in test.

Values below are read from the real files, not invented.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemolux.data.dataset import (
    HB_MISSING_MARKERS,
    _dedup_key,
    _edit_distance,
    _mask_rank,
    build_records,
    discover_masks,
    is_excluded,
    mask_matches,
    parse_hb,
    resolve_label_columns,
)

# --------------------------------------------------------------------------- #
# parse_hb -- the comma-decimal trap
# --------------------------------------------------------------------------- #

# Verbatim from data/raw/eyes-defy-anemia/dataset anemia/Italy.xlsx, row 52.
ITALY_COMMA_HB: dict[str, str] = {
    "53": "15,1",
    "59": "15,5",
    "60": "15,6",
    "69": "14,1",
    "70": "15,3",
    "71": "13,7",
    "72": "15,4",
    "73": "12,3",
    "74": "12,9",
    "75": "13,5",
    "76": "13,7",
    "77": "16,8",
    "78": "15,4",
    "79": "14,7",
    "80": "13,5",
}


def test_there_were_exactly_fifteen_comma_decimals() -> None:
    """Guards the fixture against drifting away from the real file."""
    assert len(ITALY_COMMA_HB) == 15


@pytest.mark.parametrize(("number", "cell"), sorted(ITALY_COMMA_HB.items()))
def test_real_comma_decimal_values_parse(number: str, cell: str) -> None:
    """``cell`` is the verbatim text from ``Italy.xlsx``; it must read as a
    decimal. An earlier draft of this test called ``float(cell)``, which raises
    on a comma, so it failed on the parser's own input.
    """
    parsed = parse_hb(cell)
    assert parsed is not None, f"Italy/{number}: {cell!r} was dropped"
    assert parsed == pytest.approx(float(cell.replace(",", ".")))


@pytest.mark.parametrize("raw", sorted(ITALY_COMMA_HB.values()))
def test_comma_and_dot_agree_on_the_same_number(raw: str) -> None:
    """A comma decimal and a dot decimal for the same value must be identical.

    This is the invariant the whole fix rests on: the comma form is not a
    different measurement, it is the same number written differently.
    """
    from_comma = parse_hb(raw)
    from_dot = parse_hb(raw.replace(",", "."))
    assert from_comma == from_dot
    assert from_comma is not None


def test_no_hb_value_is_lost_to_the_comma() -> None:
    """All 15 parse, so the Italian block 53-80 survives intact.

    A loader that coerced these to NaN would still *run*: it would just quietly
    hold 15 fewer patients, all from one site and one contiguous range.
    """
    parsed = [parse_hb(v) for v in ITALY_COMMA_HB.values()]
    assert all(v is not None for v in parsed)
    assert len(parsed) == 15
    # Sanity on the range: every value is a plausible adult Hb.
    assert all(12.0 <= v <= 17.0 for v in parsed)


def test_underscore_means_missing_not_zero() -> None:
    """``Italy.xlsx`` patient 93 holds ``'_'`` with the note "Hgb not available"."""
    assert parse_hb("_") is None


@pytest.mark.parametrize("marker", sorted(HB_MISSING_MARKERS))
def test_every_declared_marker_is_missing(marker: str) -> None:
    assert parse_hb(marker) is None


@pytest.mark.parametrize("marker", ["", "  ", "_", "N/A", "na", "None", "null"])
def test_marker_matching_ignores_case_and_whitespace(marker: str) -> None:
    assert parse_hb(marker) is None


def test_nan_and_none_are_missing() -> None:
    assert parse_hb(float("nan")) is None
    assert parse_hb(None) is None
    assert parse_hb(float("inf")) is None


def test_numbers_pass_through_unchanged() -> None:
    assert parse_hb(13) == 13.0
    assert parse_hb(13.7) == 13.7
    assert parse_hb(np.float64(9.3)) == 9.3


def test_whitespace_around_a_comma_decimal_is_tolerated() -> None:
    assert parse_hb("  15,1  ") == 15.1


def test_ambiguous_separators_raise_rather_than_guessing() -> None:
    """Both a dot and a comma is undecidable, and a wrong guess is a 10x error."""
    with pytest.raises(ValueError, match="ambiguous"):
        parse_hb("13.7,1")


def test_thousands_separator_is_rejected_not_read_as_a_decimal() -> None:
    """'1,234' is grouped thousands. Reading it as 1.234 would be catastrophic."""
    with pytest.raises(ValueError, match="thousands"):
        parse_hb("1,234")


def test_unparseable_text_returns_none_rather_than_raising() -> None:
    """A word that is not a number and not a marker is simply no measurement."""
    assert parse_hb("not available") is None


# --------------------------------------------------------------------------- #
# mask_matches -- suffixes, variants and real typos
# --------------------------------------------------------------------------- #

# Every stem below is a real filename from the release.
REAL_MASK_STEMS: list[tuple[str, str]] = [
    ("20200118_164733_forniceal", "forniceal"),
    ("20200118_164733_palpebral", "palpebral"),
    ("20200118_164733_forniceal_palpebral", "forniceal_palpebral"),
    ("001_palpebral", "palpebral"),
    ("20200124_202947_palpebral(1)", "palpebral"),
    ("20200124_202058_papebral", "palpebral"),
    ("T_64_20190612_092742_palplebral", "palpebral"),
    ("T_64_20190612_092742_forniceal_palplebral", "forniceal_palpebral"),
]


@pytest.mark.parametrize(("stem", "expected"), REAL_MASK_STEMS)
def test_real_mask_stems_resolve(stem: str, expected: str) -> None:
    assert mask_matches(stem) == expected


@pytest.mark.parametrize("stem", ["20200118_164733", "1", "20200124_202058", "T_64_20190612_092742"])
def test_photograph_stems_are_not_masks(stem: str) -> None:
    """These are the real image stems. If any resolved to an ROI, ``_pick_image``
    would exclude the patient's only photograph."""
    assert mask_matches(stem) is None


def test_combined_mask_is_not_truncated_to_forniceal() -> None:
    """The longest-suffix ordering. A plain endswith('forniceal') gets this
    wrong on every combined mask in the corpus."""
    assert mask_matches("20200118_164733_forniceal_palpebral") == "forniceal_palpebral"


def test_typo_papebral_resolves_to_palpebral() -> None:
    """Real filename in India/7: 20200124_202058_papebral.png"""
    assert mask_matches("20200124_202058_papebral") == "palpebral"


def test_typo_palplebral_resolves_to_palpebral() -> None:
    """Real filename in Italy/95: T_64_20190612_092742_palplebral.png"""
    assert mask_matches("T_64_20190612_092742_palplebral") == "palpebral"


def test_variant_suffix_does_not_hide_the_roi() -> None:
    """'(1)' ends the stem, so an unmasked endswith check returns None -- and
    because _pick_image excludes masks with this same function, the duplicate
    would then be selected as the photograph."""
    assert mask_matches("20200124_202947_palpebral(1)") == "palpebral"
    assert mask_matches("20200124_202947_palpebral(12)") == "palpebral"


def test_unknown_word_within_one_edit_still_resolves() -> None:
    """The bounded fallback, so an unseen typo does not silently drop a mask."""
    assert mask_matches("photo_forniceel") == "forniceal"
    assert mask_matches("photo_palpebrall") == "palpebral"


def test_unrelated_stem_does_not_resolve() -> None:
    """The fuzzy fallback must not invent ROIs. Three edits is too far."""
    assert mask_matches("patient_photograph") is None
    assert mask_matches("thumbnail") is None


def test_a_typo_in_the_combined_name_stays_a_combined_mask() -> None:
    """``Italy/95`` ships ``..._forniceal_palplebral.png``.

    The obvious tail-only check resolves it to ``palpebral``, because the tail
    after the last underscore is ``palplebral`` -- one edit from ``palpebral``,
    four from ``forniceal_palpebral``. That is a real ROI, so nothing errors and
    the combined mask is silently mislabelled, which would quietly remove the
    combined-ROI condition from the corpus. The whole-stem typo table is what
    prevents it.
    """
    assert mask_matches("T_64_20190612_092742_forniceal_palplebral") == "forniceal_palpebral"


def test_fuzzy_fallback_prefers_the_longer_roi_on_a_tie() -> None:
    """'forniceal' must not absorb a combined mask that is one edit away."""
    assert mask_matches("x_forniceal_palpebra") == "forniceal_palpebral"


# --------------------------------------------------------------------------- #
# _edit_distance -- used by the fallback, so it needs its own cases
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("a", "", 1),
        ("", "a", 1),
        ("abc", "abc", 0),
        ("abc", "abd", 1),
        ("abc", "ab", 1),
        ("ab", "abc", 1),
        ("abc", "acb", 2),
        ("kitten", "sitting", 3),
        ("forniceal", "forniceel", 1),
        # "palplebral" is "palpebral" with an extra 'l' -- an insertion, so the
        # distance is 1. An earlier draft of this table expected 2, on the theory
        # that the real typo was a transposition. It is not: the letters were
        # reordered *and* one duplicated. Verified character by character.
        ("palpebral", "palplebral", 1),
        ("papebral", "palpebral", 1),
    ],
)
def test_edit_distance(a: str, b: str, expected: int) -> None:
    assert _edit_distance(a, b, cap=3) == expected


def test_edit_distance_beyond_the_cap_is_reported_as_over() -> None:
    """The cap lets the caller reject early instead of scanning to the end."""
    assert _edit_distance("abc", "xyzxyzxyz", cap=2) > 2


def test_edit_distance_length_gap_short_circuits() -> None:
    """abs(len diff) > cap cannot possibly be within the cap."""
    assert _edit_distance("a", "aaaaaaaa", cap=2) > 2


# --------------------------------------------------------------------------- #
# _mask_rank -- the '(1)' duplicate must lose regardless of iteration order
# --------------------------------------------------------------------------- #


def test_unsuffixed_original_outranks_its_variant() -> None:
    """'(' is 0x28 and '.' is 0x2E, so sorted() lists the '(1)' copy FIRST. A
    rule phrased as 'prefer the one I saw second' silently loses; a rank cannot."""
    from pathlib import Path

    original = Path("20200124_202947_palpebral.png")
    variant = Path("20200124_202947_palpebral(1).png")
    assert _mask_rank(original) < _mask_rank(variant)


def test_sorted_puts_the_variant_first_so_rank_is_what_matters() -> None:
    """Documents the ordering that made the old rule fail."""
    from pathlib import Path

    names = ["20200124_202947_palpebral(1).png", "20200124_202947_palpebral.png"]
    assert sorted(names) == [
        "20200124_202947_palpebral(1).png",
        "20200124_202947_palpebral.png",
    ]
    assert ord("(") < ord(".")
    assert Path(names[0]).stem.endswith("(1)")


def test_rank_is_total_so_the_choice_never_depends_on_the_filesystem() -> None:
    from pathlib import Path

    a = Path("x_palpebral.png")
    b = Path("x_palpebral.png")
    assert _mask_rank(a) == _mask_rank(b)


def test_dedup_key_ignores_the_variant_suffix() -> None:
    """The two India/8 files are byte-identical (SHA-256 verified) and differ
    only by the browser's '(1)', so they must collapse to one key."""

    class _FakePath:
        def __init__(self, stem: str, size: int) -> None:
            self.stem = stem
            self._size = size

        def stat(self) -> object:
            class S:
                st_size = self._size

            return S()

    a = _FakePath("20200124_202947_palpebral", 282397)
    b = _FakePath("20200124_202947_palpebral(1)", 282397)
    assert _dedup_key(a) == _dedup_key(b)  # type: ignore[arg-type]


def test_dedup_key_distinguishes_different_sizes() -> None:

    class _FakePath:
        def __init__(self, stem: str, size: int) -> None:
            self.stem = stem
            self._size = size

        def stat(self) -> object:
            class S:
                st_size = self._size

            return S()

    a = _FakePath("20200124_202947_palpebral", 282397)
    b = _FakePath("20200124_202947_palpebral(1)", 999999)
    assert _dedup_key(a) != _dedup_key(b)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# is_excluded -- Italy patient 93
# --------------------------------------------------------------------------- #


def test_eliminato_row_is_excluded() -> None:
    """Italy.xlsx patient 93: Hgb='_', Note='Hgb not available',
    'Unnamed: 6'='ELIMINATO'. The flag sits in an unnamed column, so this must
    scan every column rather than look one up by name."""
    row = {"Number": 93, "Hgb": "_", "Note": "Hgb not available", "Unnamed: 6": "ELIMINATO"}
    assert is_excluded(row) is not None


def test_exclusion_reason_names_the_column() -> None:
    row = {"Unnamed: 6": "ELIMINATO"}
    reason = is_excluded(row)
    assert reason is not None
    assert "Unnamed: 6" in reason


def test_an_ordinary_row_is_not_excluded() -> None:
    row = {"Number": 1, "Hgb": 12.2, "Gender": "M", "Age": 29, "Note": None}
    assert is_excluded(row) is None


def test_nan_cells_do_not_trip_exclusion() -> None:
    """India's Note column is all-NaN. float('nan') contains no letters, but a
    defensive check keeps the behaviour explicit."""
    row = {"Number": 1, "Note": float("nan")}
    assert is_excluded(row) is None


def test_a_free_text_note_mentioning_removal_is_caught() -> None:
    row = {"Note": "patient excluded from the study"}
    assert is_excluded(row) is not None


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #


def test_number_resolves_as_the_patient_id() -> None:
    """The real header. An alias list lacking 'number' rejected both workbooks."""
    assert resolve_label_columns(["Number", "Hgb", "Gender", "Age", "Note"]) == {
        "patient_id": "Number",
        "hb": "Hgb",
        "sex": "Gender",
        "age": "Age",
        "note": "Note",
    }


def test_italys_extra_unnamed_columns_do_not_break_resolution() -> None:
    columns = [
        "Number", "Hgb", "Gender", "Age", "Note",
        "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8",
    ]
    resolved = resolve_label_columns(columns)
    assert resolved["patient_id"] == "Number"
    assert resolved["hb"] == "Hgb"


def test_missing_hb_column_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="cannot identify the haemoglobin column"):
        resolve_label_columns(["Number", "Gender", "Age"])


def test_column_matching_ignores_case_and_punctuation() -> None:
    assert resolve_label_columns(["number", "HGB", "sex"])["patient_id"] == "number"
    assert resolve_label_columns(["number", "HGB", "sex"])["hb"] == "HGB"


# --------------------------------------------------------------------------- #
# End-to-end, against the real download when it is present
# --------------------------------------------------------------------------- #

REAL_ROOT = None
try:
    from pathlib import Path

    _candidate = Path(r"data/raw/eyes-defy-anemia/dataset anemia")
    if _candidate.is_dir():
        REAL_ROOT = _candidate
except Exception:  # pragma: no cover
    REAL_ROOT = None

requires_real = pytest.mark.skipif(
    REAL_ROOT is None,
    reason="the Kaggle download is not present; run `make data` first",
)


@requires_real
def test_loader_finds_every_patient_but_the_excluded_one() -> None:
    """95 India + 123 Italy folders = 218. Italy/93 is flagged ELIMINATO and has
    no Hb, so 217 is the expected count."""
    records = build_records(REAL_ROOT, verbose=False)
    assert len(records) == 217


@requires_real
def test_all_fifteen_comma_decimal_italian_patients_survive() -> None:
    records = build_records(REAL_ROOT, verbose=False)
    by_number = {r.local_id: r for r in records if r.site == "Italy"}
    for number in ITALY_COMMA_HB:
        assert number in by_number, f"Italy/{number} was dropped"


@requires_real
def test_patient_ids_are_site_qualified_and_unique() -> None:
    records = build_records(REAL_ROOT, verbose=False)
    ids = [r.patient_id for r in records]
    assert len(ids) == len(set(ids))
    assert all("/" in i for i in ids)
    assert {"India/1", "Italy/1"} <= set(ids), "both sites number from 1"


@requires_real
def test_every_kept_patient_has_the_default_roi() -> None:
    """This is the check that caught the two typos: India/7 ships _papebral.png
    and Italy/95 ships _palplebral.png, and a strict suffix match left both
    patients without a palpebral mask while the metadata claimed otherwise."""
    records = build_records(REAL_ROOT, verbose=False)
    missing = [r.patient_id for r in records if "palpebral" not in r.masks]
    assert missing == [], f"no default ROI: {missing}"


@requires_real
def test_the_six_forniceal_less_patients_are_italy_1_35_54_58_75_109() -> None:
    """Named in the dataset's own documentation. Palpebral is the default ROI,
    so these are still trainable; forniceal is the ROI that is missing."""
    records = build_records(REAL_ROOT, verbose=False)
    without = {r.local_id for r in records if "forniceal" not in r.masks}
    assert without == {"1", "35", "54", "58", "75", "109"}
    assert all(r.site == "Italy" for r in records if r.local_id in without and r.site == "Italy")


@requires_real
def test_italy_93_is_the_only_dropped_patient() -> None:
    records = build_records(REAL_ROOT, verbose=False)
    assert "Italy/93" not in {r.patient_id for r in records}


@requires_real
def test_hb_distribution_matches_the_workbooks() -> None:
    """Read from the real workbooks. India mean 11.47, Italy 13.83 once the
    comma decimals are included -- Italy's mean is higher than the 13.74 that a
    coercing loader reports, because that one silently loses 15 patients."""
    records = build_records(REAL_ROOT, verbose=False)
    india = np.array([r.hb for r in records if r.site == "India"])
    italy = np.array([r.hb for r in records if r.site == "Italy"])
    assert len(india) == 95
    assert len(italy) == 122
    assert india.mean() == pytest.approx(11.47, abs=0.01)
    assert italy.mean() == pytest.approx(13.83, abs=0.01)
    assert india.min() == pytest.approx(7.6, abs=0.05)
    assert italy.min() == pytest.approx(7.0, abs=0.05)


@requires_real
def test_sex_and_age_landed() -> None:
    records = build_records(REAL_ROOT, verbose=False)
    assert all(r.sex in {"M", "F"} for r in records)
    assert all(r.age is not None for r in records)
    # The site/age confound that has to be stated in the fairness report.
    india_age = np.mean([r.age for r in records if r.site == "India"])  # type: ignore[misc]
    italy_age = np.mean([r.age for r in records if r.site == "Italy"])  # type: ignore[misc]
    assert italy_age - india_age == pytest.approx(15.6, abs=0.5)


@requires_real
def test_india_8_masks_collapse_to_the_unsuffixed_originals() -> None:
    masks = discover_masks(REAL_ROOT / "India" / "8")
    assert len(masks) == 3
    for path in masks.values():
        assert "(1)" not in path.name


@requires_real
def test_italy_1_zero_padded_prefix_resolves() -> None:
    """Italy/1 holds '1.jpg' and '001_palpebral.png' -- a different naming
    convention from India's capture timestamps."""
    masks = discover_masks(REAL_ROOT / "Italy" / "1")
    assert "palpebral" in masks
    assert masks["palpebral"].name == "001_palpebral.png"


@requires_real
def test_records_are_sorted_deterministically() -> None:
    """Same input, same order, so a results table is reproducible."""
    first = [r.patient_id for r in build_records(REAL_ROOT, verbose=False)]
    second = [r.patient_id for r in build_records(REAL_ROOT, verbose=False)]
    assert first == second
    assert first == sorted(first, key=lambda p: (p.split("/")[0], int(p.split("/")[1])))
