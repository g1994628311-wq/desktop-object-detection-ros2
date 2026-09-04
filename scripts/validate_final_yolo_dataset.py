#!/usr/bin/env python3
"""训练前 Dataset Validation（数据集验证），不执行模型推理。

检查 images/labels 的一一对应、YOLO normalized 标签、类别 ID、bbox 边界及
Train/Val/Test 的 filename/session 泄露，避免错误数据进入 Training。
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

# ============================================================
# 1. 数据集路径：images/{train,val,test} 与 labels/{train,val,test} 必须配对。
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/yolo_dataset_final"
NAMES = ("mouse", "keyboard", "laptop", "cup", "headphones")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    """执行只读 QA；YOLO 每行是 class_id x_center y_center width height（0~1），
    不是像素 xyxy。BBox 由 center ± width/height/2 检查，微小越界也须修复。
    """
    errors: list[str] = []
    rows = []
    split_stems: dict[str, set[str]] = {}
    hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        image_dir, label_dir = DATA / "images" / split, DATA / "labels" / split
        images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS}
        labels = {p.stem: p for p in label_dir.glob("*.txt")}
        split_stems[split] = set(images)
        # Missing label 是图片无标注；Orphan label 是 txt 找不到对应图片。
        for stem in sorted(set(images) - set(labels)): errors.append(f"missing label: {split}/{stem}")
        for stem in sorted(set(labels) - set(images)): errors.append(f"orphan label: {split}/{stem}")
        counts = Counter()
        for stem in sorted(set(images) & set(labels)):
            image, label = images[stem], labels[stem]
            try:
                with Image.open(image) as im: im.verify()
            except Exception as exc: errors.append(f"unreadable image {image}: {exc}")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes: errors.append(f"exact duplicate across materialized paths: {image} == {hashes[digest]}")
            hashes[digest] = str(image)
            for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip(): continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label}:{number}: expected 5 columns"); continue
                try: cls = int(parts[0]); x, y, w, h = map(float, parts[1:])
                except ValueError:
                    errors.append(f"{label}:{number}: non-numeric value"); continue
                # 五类 ID 固定 0..4；非法 ID 或 bbox 超出 [0,1] 都会污染训练。
                if cls not in range(5): errors.append(f"{label}:{number}: invalid class {cls}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1): errors.append(f"{label}:{number}: invalid normalized values")
                if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6: errors.append(f"{label}:{number}: bbox outside image")
                if cls in range(5): counts[cls] += 1
        rows.append({"split":split,"images":len(images),"objects":sum(counts.values()),**{NAMES[i]:counts[i] for i in range(5)}})
    for a,b in (("train","val"),("train","test"),("val","test")):
        overlap=split_stems[a]&split_stems[b]
        if overlap: errors.append(f"filename overlap {a}/{b}: {len(overlap)}")

    # Session Leakage：连续采集图可能共享背景/光照/物体，跨 split 会造成数据泄露。
    session_split = {}
    for split, stems in split_stems.items():
        for stem in stems:
            parts=stem.split("_")
            if len(parts)>=2 and parts[0].startswith("P") and parts[1].startswith("S"):
                key=(parts[0],parts[1])
                if key in session_split and session_split[key]!=split: errors.append(f"session leakage: {key} in {session_split[key]} and {split}")
                session_split[key]=split
    out=ROOT/"results/final_model_comparison/dataset_summary.csv"
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("split","images","objects",*NAMES)); writer.writeheader(); writer.writerows(rows)
    for row in rows: print(row)
    print(f"missing_labels={sum('missing label' in e for e in errors)} orphan_labels={sum('orphan label' in e for e in errors)} errors={len(errors)}")
    for error in errors: print("ERROR:",error)
    return 1 if errors else 0


if __name__ == "__main__": sys.exit(main())
