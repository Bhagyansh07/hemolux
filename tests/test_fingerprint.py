"""The cache fingerprint: a code change must make a cache unreadable.

Every test here is about one failure mode. A feature cache exists to make an experiment
cheap to repeat, and the specific way that goes wrong is not a crash -- it is a cache
that still loads after the code that produced it has changed, so the fix you just made
does nothing to the number it was meant to change and the table on screen is the old one.

Both of the project's real instances of that behaved identically: a colour statistic fed
CIELAB where sRGB was meant, which made it NaN for every patient in the corpus, and a
percentile taken over the true extremes of ~12 million pixels, which made it NaN for any
frame with a zero in the red channel. Neither raised. So most of what follows is about
refusing to read, and the non-refusal cases exist to prove the check is not simply
refusing everything -- a check that rejects all input is indistinguishable from a
misconfigured one, and would be discovered only after a long run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hemolux import fingerprint
from hemolux.fingerprint import SOURCES, StaleFeatureCacheError, digest
from hemolux.training import FeatureSet, load_feature_cache, save_feature_cache

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Import a script by path, since ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture_mod = _load("make_synthetic_fixture")

SMALL = FeatureSet(
    patient_ids=("India/1", "India/2", "Italy/1"),
    features=np.arange(9, dtype=np.float64).reshape(3, 3),
    hb=np.array([11.0, 12.0, 13.0]),
    site=("India", "India", "Italy"),
    sex=("F", "M", "F"),
    age=np.array([30.0, 40.0, 50.0]),
    roi="palpebral",
    backbone="none",
    feature_dim=3,
    kind="colour",
    balance="raw",
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A twelve-patient generated corpus, built once and shared with the CLI tests.

    ``cap``d at twelve for the reason given in ``tests/test_cli.py``: a fold of two runs
    the whole pipeline, and nothing here asserts on the values it produces.
    """
    root = tmp_path_factory.mktemp("fingerprint") / "synthetic"
    root.mkdir()
    saved = fixture_mod.PATHS
    fixture_mod.PATHS = SimpleNamespace(synthetic=root)
    argv = sys.argv
    sys.argv = ["make_synthetic_fixture.py", "--patients", "12", "--force"]
    try:
        assert fixture_mod.main() == 0
    finally:
        sys.argv = argv
        fixture_mod.PATHS = saved
    return root


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Redirect ``ARTIFACT_MODELS`` for the CLI-level test, and restore it after.

    Patched on both modules that bind the name at import time, for the reason
    ``tests/test_cli.py``'s ``artefacts`` fixture documents: rebinding it in ``config``
    alone would leave the other two writing into the repository.
    """
    import hemolux.cli
    import hemolux.export
    import hemolux.training

    root = tmp_path_factory.mktemp("models")
    saved = {
        (hemolux.cli, hemolux.cli.ARTIFACT_MODELS),
        (hemolux.training, hemolux.training.ARTIFACT_MODELS),
        (hemolux.export, hemolux.export.ARTIFACT_MODELS),
    }
    for module in (hemolux.cli, hemolux.training, hemolux.export):
        module.ARTIFACT_MODELS = root
    yield root
    for module, value in saved:
        module.ARTIFACT_MODELS = value


# --------------------------------------------------------------------------- #
# The digest itself
# --------------------------------------------------------------------------- #


def test_the_digest_is_stable_within_a_process() -> None:
    """Two calls, one answer -- or the check it guards is itself noise.

    Cheap to state and worth stating: a digest that varied per call would make every
    cache unreadable the moment it was written, and the symptom would look like a
    corrupt-file bug rather than a fingerprint bug.
    """
    assert digest() == digest()


def test_the_digest_is_short_and_hex() -> None:
    """Sixteen hex characters: a change detector, not a security control.

    The question it answers is "is this the same code that wrote the file". A full
    SHA-256 would be stored in every cache and printed in every refusal message and
    would not answer that any better.
    """
    value = digest()
    assert len(value) == fingerprint.LENGTH
    assert all(c in "0123456789abcdef" for c in value), f"not hex: {value!r}"


def test_the_digest_changes_when_a_fingerprinted_body_changes(monkeypatch) -> None:
    """The property the whole module exists for, tested directly.

    ``high_hue_ratio`` is fingerprinted, so a new body for it must move the digest. If
    this fails, nothing else in the module is load-bearing: caches would be refused at
    random, or -- worse -- accepted after a change.
    """
    from hemolux.metrics import colorimetry

    before = digest()

    def nan_ratio(rgb_patch):
        return float("nan")

    monkeypatch.setattr(colorimetry, "high_hue_ratio", nan_ratio)

    assert digest() != before


def test_the_digest_ignores_a_change_that_cannot_move_a_number(monkeypatch) -> None:
    """A cache must survive an edit to code that does not compute one.

    This is the reason the inputs are listed by hand rather than by hashing whole
    modules. An edit to a regression metric must not cost twenty minutes of backbone
    passes over 217 twelve-megapixel frames, and "just hash the module" is exactly the
    simplification this test blocks.
    """
    from hemolux.metrics import regression

    def fake_report(*args, **kwargs):
        return None

    before = digest()
    monkeypatch.setattr(regression, "regression_report", fake_report)

    assert digest() == before, "an unrelated edit invalidated the cache"


def test_a_library_version_is_part_of_the_digest(monkeypatch) -> None:
    """``opencv`` does the resizing, so its release is an input even with the code fixed.

    A different wheel under the same version string, or a headless install registering
    under a different distribution name, both change pixels. Recorded rather than assumed
    away.
    """
    before = digest()
    monkeypatch.setattr(fingerprint, "_versions", lambda: [("numpy", "0.0.0-imaginary")])

    assert digest() != before


def test_a_distribution_a_wheel_does_not_register_falls_back_to_the_module(
    monkeypatch,
) -> None:
    """``opencv-python`` is absent from a headless install, and must not stay that way.

    The distribution metadata reports it missing in a perfectly good environment, so a
    naive lookup would record ``"absent"`` forever and track nothing at all for the one
    library that does the resizing. Falls back to ``cv2.__version__`` instead, and the
    fallback is the thing under test -- not that the digest exists.
    """
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    versions = dict(fingerprint._versions())

    assert "absent" not in versions.values(), f"a version went unrecorded: {versions}"
    assert versions["opencv-python"], "the fallback produced an empty string"
    assert digest(), "the digest stopped being computable"


def test_a_library_that_is_genuinely_gone_is_recorded_as_absent(monkeypatch) -> None:
    """``"absent"`` is a real answer, not an error.

    Raising here would make the fingerprint uncomputable in exactly the environments
    where it is most useful -- a server install with no imaging stack at all.
    """

    def absent(name: str) -> str:
        return "absent"

    monkeypatch.setattr(fingerprint, "_module_version", absent)

    versions = dict(fingerprint._versions())
    assert "absent" in versions.values()
    assert digest()


def test_a_moved_fingerprint_input_is_an_error_not_a_silent_drop(monkeypatch) -> None:
    """A refactor must not quietly shrink the set of things being watched.

    If a rename moved ``high_hue_ratio``, skipping it would leave every pre-rename cache
    readable -- the exact failure the module prevents, arriving by way of the fix for
    something else. So the name is refused, and the message says where to update it.
    """
    monkeypatch.setattr(
        fingerprint, "SOURCES", (("hemolux.metrics.colorimetry", "high_hue_raito"),)
    )

    with pytest.raises(AttributeError) as caught:
        digest()

    message = str(caught.value)
    assert "high_hue_raito" in message, "the name that failed is not in the message"
    assert "SOURCES" in message, "the message does not say what to edit"


def test_every_declared_input_actually_resolves() -> None:
    """``SOURCES`` is hand-maintained, so it drifts.

    Asserted as one test over the whole list because a missing entry is a property of
    the list rather than of any entry. Note the limit of what this can catch: an input
    that was *forgotten* is invisible here, which is the cost of listing by hand.
    """
    for module_name, qualname in SOURCES:
        assert fingerprint._resolve(module_name, qualname) is not None, (
            f"{module_name}:{qualname} does not resolve"
        )


def test_a_non_callable_input_is_refused() -> None:
    """A constant in the list would be hashed as a body and read as a function.

    Currently unreachable -- every entry is a function -- but the check costs one line
    and stops the list from growing an accidental ``("module", "SOME_CONSTANT")`` that
    silently tracks nothing.
    """
    monkeypatch_src = (("hemolux.config", "IMAGE_SIZE"),)
    with pytest.raises(TypeError, match="not callable"):
        fingerprint._resolve(*monkeypatch_src[0])


def test_the_name_is_hashed_alongside_the_body() -> None:
    """Two functions with identical bodies are still two separate inputs.

    Without the name in the hash, swapping which one gets called would go unnoticed if
    the bodies happened to match. One line to include; a real class of blind spot.
    """
    import hashlib

    left = hashlib.sha256(b"hemolux.a:fn\nbody\nnumpy=1\n").hexdigest()
    right = hashlib.sha256(b"hemolux.b:fn\nbody\nnumpy=1\n").hexdigest()
    assert left != right


# --------------------------------------------------------------------------- #
# The cache contract
# --------------------------------------------------------------------------- #


def test_a_freshly_written_cache_reads_back(tmp_path: Path, monkeypatch) -> None:
    """The check must not refuse everything, or it is not a check."""
    from hemolux import training

    monkeypatch.setattr(training, "ARTIFACT_MODELS", tmp_path)
    path = save_feature_cache(SMALL)

    back = load_feature_cache(path)
    assert back.patient_ids == SMALL.patient_ids
    assert back.kind == "colour"
    assert back.balance == "raw"
    assert back.unmasked == ()
    assert np.allclose(back.features, SMALL.features)


def test_a_cache_written_by_other_code_is_refused(tmp_path: Path, monkeypatch) -> None:
    """The regression test for the failure the module was added for.

    A cache is written, the feature code changes, and the next run reads the old
    numbers. Everything a reader can see is identical; the file's own contents are the
    only place the difference exists at all.
    """
    from hemolux import training

    monkeypatch.setattr(training, "ARTIFACT_MODELS", tmp_path)
    path = save_feature_cache(SMALL)

    other = digest()[::-1]
    assert other != digest(), "the reversed digest is accidentally still the digest"
    _rewrite_fingerprint(path, other)

    with pytest.raises(StaleFeatureCacheError, match="different code"):
        load_feature_cache(path)


def test_the_refusal_names_the_file_and_both_digests(tmp_path: Path, monkeypatch) -> None:
    """A reader has to be able to act on this without reading the source.

    Which build wrote the file and which build is running are the two facts that decide
    whether to re-extract, and both are in the message. A bare "stale" would leave the
    reader guessing.
    """
    from hemolux import training

    monkeypatch.setattr(training, "ARTIFACT_MODELS", tmp_path)
    path = save_feature_cache(SMALL)
    _rewrite_fingerprint(path, "0123456789abcdef")

    with pytest.raises(StaleFeatureCacheError) as caught:
        load_feature_cache(path)

    message = str(caught.value)
    assert "0123456789abcdef" in message, "the recorded digest is missing"
    assert digest() in message, "this build's digest is missing"
    assert path.name in message, "the file at fault is not named"
    assert "re-extract" in message, "the message does not say what to do"


def test_a_cache_with_no_fingerprint_at_all_is_refused(tmp_path: Path, monkeypatch) -> None:
    """Absent is a mismatch, not permission.

    The tempting reading of a missing field is "this predates the field, so it is
    probably fine". That is a guess, and it is the guess that produced both real
    instances of this bug in the project. A file that cannot say what wrote it is treated
    as a file written by something that has since changed.
    """
    from hemolux import training

    monkeypatch.setattr(training, "ARTIFACT_MODELS", tmp_path)
    path = save_feature_cache(SMALL)
    _drop_fingerprint(path)

    with pytest.raises(StaleFeatureCacheError, match="recorded nothing"):
        load_feature_cache(path)


def test_one_digest_covers_all_four_configurations() -> None:
    """Colour, deep, balanced and raw share it, on purpose.

    They call overlapping helpers, so a per-configuration digest would have to be a
    union of all of them for each -- which is the same single digest with more places to
    get it wrong. The per-configuration key is already carried by ``kind``, ``balance``,
    ``roi`` and ``backbone``.
    """
    assert digest() == fingerprint.digest()


# --------------------------------------------------------------------------- #
# End to end: the refusal has to be actionable, and the caller is what makes it so
# --------------------------------------------------------------------------- #


def test_a_stale_cache_is_re_extracted_rather_than_fatal(
    corpus: Path, models_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    """A stale cache in the artefacts directory does not stop the run -- and says why.

    The refusal only helps if a caller handles it, and the only place that can happen is
    the CLI. Asserted through the command rather than against the resolver, because the
    property worth having is that *no caller anywhere* has to learn about the
    fingerprint: the pre-existing "skip a cache we cannot read" behaviour already covers
    it. The extra work is that the reason gets printed instead of discarded.
    """
    from hemolux.cli import main

    argv = ["features", "--data-root", str(corpus), "--features", "colour", "--balance"]
    assert main(argv) == 0
    capsys.readouterr()

    caches = sorted(models_dir.glob("features_*.npz"))
    assert caches, "the first run wrote no cache to poison"
    for path in caches:
        _rewrite_fingerprint(path, "ffffffffffffffff")

    assert main(argv) == 0, "a stale cache stopped the run"

    out = capsys.readouterr().out
    assert "ignoring stale cache" in out, "the run did not say why it re-extracted"
    for path in caches:
        assert path.name in out, f"{path.name} was rejected without being named"


def test_an_unreadable_cache_is_still_skipped_quietly(
    corpus: Path, models_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    """A truncated ``.npz`` from an interrupted run is not worth a paragraph.

    Both refusals end the same way -- re-extract -- but only one of them means anything
    is wrong with the code. Printing "ignoring stale cache" for a half-written file would
    be a lie, and printing a stack trace for a normal interrupted run would be noise.
    Distinguishing them is the reason the exception is its own type.
    """
    from hemolux.cli import main

    argv = ["features", "--data-root", str(corpus), "--features", "colour", "--balance"]
    assert main(argv) == 0
    capsys.readouterr()

    caches = sorted(models_dir.glob("features_*.npz"))
    assert caches
    for path in caches:
        path.write_bytes(b"PK\x03\x04 truncated")

    assert main(argv) == 0, "an unreadable cache stopped the run"

    out = capsys.readouterr().out
    assert "ignoring stale cache" not in out, (
        "a corrupt file was reported as stale code, which is a different claim"
    )
    assert "extracting features" in out, "the run did not re-extract"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _payload(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _rewrite_fingerprint(path: Path, value: str) -> None:
    """Replace the recorded digest, leaving every other array untouched."""
    payload = _payload(path)
    payload["fingerprint"] = np.array(value, dtype=object)
    np.savez_compressed(path, **payload)


def _drop_fingerprint(path: Path) -> None:
    """Remove the key entirely, as a build predating the field would have."""
    payload = _payload(path)
    payload.pop("fingerprint", None)
    np.savez_compressed(path, **payload)
