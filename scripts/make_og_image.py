#!/usr/bin/env python
"""Draw the 1200x630 social card the site and the repository point at.

The card is generated rather than exported from a design tool so that it carries
the same colours as the app (`app/web/styles.css`) and can be redrawn when the
domain changes -- the image URL in the Open Graph tags is absolute, so a card
naming one host while the page is served from another is a real, reachable bug.

It uses the DejaVu faces bundled with matplotlib rather than a system font,
because this corpus is built on Windows and CI runs on Linux, and a committed
image that only regenerates on one of them is not reproducible.

Usage::

    python scripts/make_og_image.py
    python scripts/make_og_image.py --out app/web/og.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

#: The palette from app/web/styles.css: dark surface, signal blue, dim text.
BACKGROUND = (11, 12, 14)
TEXT = (232, 234, 237)
DIM = (154, 160, 166)
FAINT = (107, 113, 120)
SIGNAL = (76, 194, 255)

WIDTH, HEIGHT = 1200, 630
MARGIN = 84


def _font(name: str) -> str:
    """An absolute path to a DejaVu face shipped with matplotlib."""
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
    if not path.exists():
        raise FileNotFoundError(f"matplotlib did not ship {name}")
    return str(path)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, limit: int) -> list[str]:
    """Greedy word wrap measured with the real font, not an assumed width."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= limit or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_card() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    wordmark = ImageFont.truetype(_font("DejaVuSans-Bold.ttf"), 92)
    eyebrow = ImageFont.truetype(_font("DejaVuSans.ttf"), 24)
    heading = ImageFont.truetype(_font("DejaVuSans-Bold.ttf"), 42)
    body = ImageFont.truetype(_font("DejaVuSans.ttf"), 28)
    foot = ImageFont.truetype(_font("DejaVuSansMono.ttf"), 22)

    # A spine down the left edge, the same signal blue the app uses for a live
    # reading. It reads as "instrument" rather than "logo".
    draw.rectangle([MARGIN - 28, MARGIN, MARGIN - 20, HEIGHT - MARGIN], fill=SIGNAL)

    y = MARGIN
    draw.text((MARGIN, y), "HEMO", font=wordmark, fill=TEXT)
    hemo_width = draw.textlength("HEMO", font=wordmark)
    draw.text((MARGIN + hemo_width, y), "LUX", font=wordmark, fill=SIGNAL)
    y += 118

    draw.text((MARGIN, y), "NON-INVASIVE HAEMOGLOBIN SCREENING", font=eyebrow, fill=SIGNAL)
    y += 56

    for line in _wrap(
        draw,
        "From a photograph of the palpebral conjunctiva",
        heading,
        WIDTH - 2 * MARGIN - 40,
    ):
        draw.text((MARGIN, y), line, font=heading, fill=TEXT)
        y += 56

    y += 14
    for line in _wrap(
        draw,
        "Colour features, a frozen backbone and a skin-tone fairness audit, "
        "selected on validation and reported openly.",
        body,
        WIDTH - 2 * MARGIN - 40,
    ):
        draw.text((MARGIN, y), line, font=body, fill=DIM)
        y += 40

    draw.text(
        (MARGIN, HEIGHT - MARGIN - 28),
        "screening aid, not a diagnosis  \u00b7  on-device inference  \u00b7  open evaluation",
        font=foot,
        fill=FAINT,
    )
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("app/web/og.png"),
        help="destination PNG (default: app/web/og.png)",
    )
    args = parser.parse_args()

    image = draw_card()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out, "PNG", optimize=True)
    print(f"wrote {args.out} ({image.width}x{image.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
