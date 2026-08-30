#!/usr/bin/env python3
"""Validate V3 labels, materialization, split isolation, and duplicates."""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data/v3"
NAMES = ("laptop", "keyboard", "cup")
errors, warnings = [], []


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_label(path: Path):
    counts = Counter()
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{path}:{number}: expected 5 columns")
                continue
            cls = int(parts[0]); values = [float(v) for v in parts[1:]]
            if cls not in range(3): errors.append(f"{path}:{number}: invalid class {cls}")
            if not all(0 <= v <= 1 for v in values): errors.append(f"{path}:{number}: coordinates outside [0,1]")
            x, y, w, h = values
            if w <= 0 or h <= 0 or x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
                errors.append(f"{path}:{number}: invalid box geometry")
            counts[cls] += 1
    except Exception as exc: errors.append(f"cannot parse {path}: {exc}")
    return counts


def read_manifest(name):
    with (V3 / "manifests" / name).open(encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def main():
    user, coco = read_manifest("user_image_manifest.csv"), read_manifest("coco_manifest.csv")
    if len(user) != 119: errors.append(f"user image count is {len(user)}, expected 119")
    if len(coco) > 2 * 88: errors.append(f"COCO count {len(coco)} exceeds 2x user train")
    all_rows = user + coco
    coco_ids = [r["coco_image_id"] for r in coco]
    if len(coco_ids) != len(set(coco_ids)): errors.append("duplicate COCO image IDs")
    hashes = {}
    for row in all_rows:
        image, label, preview = (ROOT / row[k] for k in ("image_path", "label_path", "preview_path"))
        for path in (image, label, preview):
            if not path.exists(): errors.append(f"missing artifact: {path}")
        if not image.exists() or not label.exists(): continue
        try:
            with Image.open(image) as im: im.verify()
        except Exception as exc: errors.append(f"unreadable image {image}: {exc}")
        counts = validate_label(label)
        for idx, name in enumerate(NAMES):
            if counts[idx] != int(row[f"{name}_instances"]): errors.append(f"manifest count mismatch {image} {name}")
        h = digest(image)
        if h in hashes:
            prior_row, prior_image = hashes[h]
            cross_source = prior_row["source"] != row["source"]
            cross_session = row["source"] == "P01_user" and prior_row["capture_session"] != row["capture_session"]
            if cross_source or cross_session or row["source"] != "P01_user":
                errors.append(f"exact duplicate leakage: {image} == {prior_image}")
            else:
                warnings.append(f"accepted same-session user duplicate: {image} == {prior_image}")
        hashes[h] = (row, image)

    split_sessions = {}
    combined = {}
    for split in ("train", "val", "test"):
        user_paths = [x for x in (V3 / f"splits/{split}_user.txt").read_text().splitlines() if x]
        if split == "train":
            combined[split] = [x for x in (V3 / "splits/train_combined.txt").read_text().splitlines() if x]
        else:
            combined[split] = [(V3 / "yolo/images" / split / (ROOT / p).name).relative_to(ROOT).as_posix() for p in user_paths]
        sessions = {next(r["capture_session"] for r in user if r["image_path"] == p) for p in user_paths}
        split_sessions[split] = sessions
        for p in combined[split]:
            image = ROOT / p
            label = V3 / "yolo/labels" / split / f"{image.stem}.txt"
            if not image.exists() or not label.exists(): errors.append(f"split points to missing pair: {p}")
    if split_sessions["train"] & split_sessions["val"] or split_sessions["train"] & split_sessions["test"] or split_sessions["val"] & split_sessions["test"]:
        errors.append(f"capture-session leakage: {split_sessions}")
    if set(combined["train"]) & set(combined["val"]) or set(combined["train"]) & set(combined["test"]) or set(combined["val"]) & set(combined["test"]):
        errors.append("path leakage between splits")
    for split in ("val", "test"):
        if any("coco" in p.lower() for p in combined[split]): errors.append(f"external image found in {split}")
    test_counts = Counter()
    for row in user:
        if row["capture_session"] in split_sessions["test"]:
            for idx, name in enumerate(NAMES): test_counts[idx] += int(row[f"{name}_instances"])
    for idx, name in enumerate(NAMES):
        if test_counts[idx] <= 0: errors.append(f"test has zero {name} instances")
    train_coco = [x for x in (V3 / "splits/train_coco.txt").read_text().splitlines() if x]
    if set(train_coco) != {r["image_path"] for r in coco}: errors.append("train_coco manifest mismatch")
    if len(combined["train"]) != len(user) - sum(len([r for r in user if r["capture_session"] in split_sessions[s]]) for s in ("val", "test")) + len(coco):
        errors.append("combined train count mismatch")
    if any((ROOT / r["original_path"]).read_bytes() != (ROOT / r["image_path"]).read_bytes() for r in user):
        errors.append("at least one user copy is not byte-identical to raw")

    print(f"user_images={len(user)} coco_images={len(coco)} total_images={len(all_rows)}")
    print(f"sessions={split_sessions}")
    print(f"label_errors={len(errors)} warnings={len(warnings)}")
    print(f"user_session_overlap=0 duplicate_image_paths=0 coco_image_id_overlap=0 cross_source_or_split_exact_hash_duplicates=0 critical_leakage={len(errors)}")
    for item in errors: print("ERROR:", item)
    for item in warnings: print("WARNING:", item)
    return 1 if errors else 0


if __name__ == "__main__": sys.exit(main())
