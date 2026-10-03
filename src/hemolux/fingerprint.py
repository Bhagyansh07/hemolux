"""A digest of everything except the arguments that decides what a cache holds.

Why this module exists
----------------------
A feature cache is keyed by ``kind``, ``balance``, ``roi`` and ``backbone``: everything
the *caller* chooses. It is not keyed by the code that does the work, so a cache
extracted before a fix kept being read afterwards and the fix did nothing to the number
it was supposed to change.

That happened twice in this project, and neither raised:

* a colour statistic was fed CIELAB where sRGB was meant, which made it NaN for every
  patient in the corpus;
* a "max minus min" hue statistic was computed over the true extremes of ~12 million
  pixels, which returned NaN for any frame with a zero in the red channel -- 1 patient
  in 217.

Both produced finite-looking arrays, plausible tables, and a number that was quietly the
old one. A cache is supposed to make an experiment cheap to repeat; what it must not do
is make it cheap to *believe*.

Why a digest and not a version number
-------------------------------------
The inputs are spread across three modules and four library versions, and a hand-kept
integer has to be bumped by hand -- which is precisely the step that gets forgotten
during the exact kind of rushed edit that caused the problem. Hashing the sources means
forgetting is impossible: change a body, the digest changes, the cache is refused. The
failure mode becomes "wasted a re-extraction" instead of "published the old number".

Listed by hand rather than by walking the modules
-------------------------------------------------
A module-level walk would hash reporting code, docstrings, type aliases and everything
else that cannot move a number. Then fixing a typo in a comment would throw away a
cache that was still correct, and on the deep path that is twenty minutes of backbone
passes over 217 twelve-megapixel frames. The list below is deliberately the smallest set
that can produce a different number from the same inputs.

What it does not cover
----------------------
The pretrained ImageNet weights, which are a file in the torch hub cache rather than
Python source. A ``timm`` release can replace them under the same name and same call
signature. The versions that shipped the trunk are in the digest; the bytes are not. A
deep cache is therefore only as trustworthy as the pin in ``pyproject.toml``, and that
is a dependency pin doing work the cache itself is not. It is stated here rather than
left to be discovered by someone whose R2 moved for a reason this module cannot name.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from importlib import metadata
from typing import Any

#: ``(module, qualname)`` for every function whose body can change a cached value.
#:
#: The colour list is the whole measurement chain, not just the two entry points:
#: ``extract_colour_features`` calls the others, so hashing only the entry points would
#: have missed both of the defects in the module docstring -- one of which was three
#: calls below the entry point.
SOURCES: tuple[tuple[str, str], ...] = (
    # Colourimetry: the entire chain, in the order the entry points call it.
    ("hemolux.metrics.colorimetry", "srgb_to_linear"),
    ("hemolux.metrics.colorimetry", "rgb_to_lab"),
    ("hemolux.metrics.colorimetry", "_spatial_mask"),
    ("hemolux.metrics.colorimetry", "_select"),
    ("hemolux.metrics.colorimetry", "mask_mean_lab"),
    ("hemolux.metrics.colorimetry", "redness_ratio"),
    ("hemolux.metrics.colorimetry", "erythema_index"),
    ("hemolux.metrics.colorimetry", "high_hue_ratio"),
    ("hemolux.metrics.colorimetry", "otsu_vessel_redness"),
    ("hemolux.metrics.colorimetry", "_as_uint8"),
    ("hemolux.metrics.colorimetry", "_input_peak"),
    ("hemolux.metrics.colorimetry", "_reference_pixels"),
    ("hemolux.metrics.colorimetry", "white_balance"),
    ("hemolux.metrics.colorimetry", "balanced_rgb"),
    ("hemolux.metrics.colorimetry", "extract_colour_features"),
    ("hemolux.metrics.colorimetry", "extract_colour_features_balanced"),
    # Dataset: how a frame is read, masked, cropped and resized.
    ("hemolux.data.dataset", "apply_mask"),
    ("hemolux.data.dataset", "build_preprocess"),
    ("hemolux.data.dataset", "ConjunctivaDataset.__getitem__"),
    # Training: how a feature set is assembled from those.
    ("hemolux.training", "_mask_at_frame_size"),
    ("hemolux.training", "extract_colour_rows"),
    ("hemolux.training", "_standardise"),
    ("hemolux.training", "_deep_features"),
)

#: Library versions whose releases can change a pixel value with the code unchanged.
#:
#: ``opencv`` because it does the mask resize and the frame resize, and interpolation
#: choices have changed across releases. ``numpy`` because its promotion rules decide
#: dtypes, and a float32/int64 disagreement becomes a silent cast. ``torch`` and
#: ``timm`` because the trunk is theirs to define.
#:
#: A patch bump therefore discards a cache that was very probably fine. That is the
#: direction to be wrong in: it costs minutes, and the alternative costs a number.
VERSIONS: tuple[str, ...] = ("numpy", "opencv-python", "torch", "timm")

#: Hex characters kept. Sixteen is 64 bits, which answers "is this the same code that
#: wrote the file" for the handful of versions a working tree ever sees. A full SHA-256
#: would be longer to store in every cache and to print in every message that refused
#: one, for a question it does not answer any better.
LENGTH = 16


class StaleFeatureCacheError(RuntimeError):
    """Raised when a cache was written by code that no longer matches.

    Distinct from ``OSError`` and from a malformed ``.npz``, which the callers also
    have to survive: an interrupted run leaves a truncated file that reads as neither a
    valid cache nor a stale one, and all three end the same way -- re-extract. Keeping
    them apart exists so the reason can be printed rather than swallowed.
    """


def digest() -> str:
    """The fingerprint recorded in every feature cache.

    Cheap enough to call twice per candidate: the sources are a few kilobytes of text
    and the versions come from installed metadata, so this costs microseconds next to a
    single frame resize. There is no reason to cache it, and caching it would create a
    second thing that can go stale.
    """
    hasher = hashlib.sha256()
    for module_name, qualname in SOURCES:
        hasher.update(f"{module_name}:{qualname}\n".encode())
        hasher.update(inspect.getsource(_resolve(module_name, qualname)).encode("utf-8"))
    for label, value in _versions():
        hasher.update(f"{label}={value}\n".encode())
    return hasher.hexdigest()[:LENGTH]


def _resolve(module_name: str, qualname: str) -> Any:
    """Look up one input, refusing rather than skipping it.

    A refactor that moves ``high_hue_ratio`` would otherwise drop it from the digest
    without a word, and every cache written before the move would keep being read --
    the exact failure this module exists to prevent, arriving through the fix for a
    different problem. So a missing name is an error that names the name, and the fix
    is one line in :data:`SOURCES`.
    """
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise AttributeError(
                f"fingerprint input {module_name}:{qualname} no longer resolves "
                f"({part!r} is missing); update SOURCES in hemolux/fingerprint.py"
            ) from exc
    if not callable(target):
        raise TypeError(f"fingerprint input {module_name}:{qualname} is not callable")
    return target


def _versions() -> list[tuple[str, str]]:
    """Installed versions of :data:`VERSIONS`, in a fixed order.

    ``opencv-python-headless`` and ``opencv-contrib-python`` both install as ``cv2``
    while registering under their own distribution names, so the distribution metadata
    reports ``opencv-python`` as absent in a perfectly good environment -- and the
    digest would then record ``"absent"`` forever, tracking nothing at all for the one
    library that does the resizing. Falls back to ``cv2.__version__``, which is the
    version that actually ships whichever wheel got installed. A genuinely missing
    distribution with no importable module reports ``"absent"``, which is a real answer:
    it means the environment changed, the event the digest exists to notice.
    """
    found: list[tuple[str, str]] = []
    for label in VERSIONS:
        try:
            found.append((label, metadata.version(label)))
        except metadata.PackageNotFoundError:
            found.append((label, _module_version("cv2")))
    return found


def _module_version(name: str) -> str:
    """The version an imported module reports, or ``"absent"`` if it will not import."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        return "absent"
    return str(getattr(module, "__version__", "unknown"))
