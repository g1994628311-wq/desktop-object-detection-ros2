#!/usr/bin/env python3
"""Render pixel-coordinate grids to support manual bounding-box audits."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    for path in sorted(args.source.glob("*.jpg")):
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        for x in range(0, width, args.step):
            draw.line((x, 0, x, height), fill=(255, 255, 0, 110), width=2)
            draw.text((x + 3, 3), str(x), fill=(255, 255, 0, 255), font=font,
                      stroke_width=2, stroke_fill=(0, 0, 0, 255))
        for y in range(0, height, args.step):
            draw.line((0, y, width, y), fill=(0, 255, 255, 110), width=2)
            draw.text((3, y + 3), str(y), fill=(0, 255, 255, 255), font=font,
                      stroke_width=2, stroke_fill=(0, 0, 0, 255))
        image.save(args.output / path.name, quality=94)


if __name__ == "__main__":
    main()
