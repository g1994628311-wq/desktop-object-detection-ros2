#!/usr/bin/env python3
"""Render canonical YOLO labels for visual quality control."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CLASSES = (ROOT / "configs/classes.txt").read_text(encoding="utf-8").splitlines()
COLORS = [(0, 220, 255), (255, 180, 0), (0, 255, 80), (255, 70, 70), (190, 80, 255)]


def render(contributor: str, session: str) -> None:
    image_dir = ROOT / "data/raw" / contributor / session
    label_dir = ROOT / "data/labels" / contributor / session
    output_dir = ROOT / "data/annotation_preview" / contributor / session
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=28)
    for image_path in sorted(image_dir.glob("*.jpg")):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        iw, ih = image.size
        for line in (label_dir / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id_s, cx_s, cy_s, bw_s, bh_s = line.split()
            class_id = int(class_id_s)
            cx, cy, bw, bh = map(float, (cx_s, cy_s, bw_s, bh_s))
            box = ((cx - bw / 2) * iw, (cy - bh / 2) * ih, (cx + bw / 2) * iw, (cy + bh / 2) * ih)
            color = COLORS[class_id]
            draw.rectangle(box, outline=color, width=6)
            draw.text((box[0] + 4, max(0, box[1] + 4)), CLASSES[class_id], fill=color, font=font, stroke_width=2, stroke_fill="black")
        image.save(output_dir / image_path.name, quality=92)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("contributor")
    parser.add_argument("session")
    args = parser.parse_args()
    render(args.contributor, args.session)
