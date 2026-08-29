#!/usr/bin/env python3
"""Render raw P01 source-folder contact sheets for manual session auditing."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "data/raw/P01"
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    for folder in sorted(source.glob("S*")):
        files = sorted(folder.glob("*.jpg"))
        cols, cw, ch = 4, 320, 250
        rows = (len(files) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(files):
            image = Image.open(path).convert("RGB")
            image.thumbnail((cw - 10, ch - 34))
            x = (index % cols) * cw + 5
            y = (index // cols) * ch
            sheet.paste(image, (x + (cw - 10 - image.width) // 2, y + 28))
            draw.text((x, y + 5), path.stem, fill="black", font=font)
        sheet.save(args.output / f"{folder.name}.jpg", quality=92)


if __name__ == "__main__":
    main()
