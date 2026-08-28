#!/usr/bin/env python3
"""Build the reproducible YOLO dataset from canonical labels and split manifests."""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from validate_labels import CLASSES, LABELS, RAW, ROOT, validate

SPLITS = ("train", "val", "test")
SPLIT_DIR = ROOT / "data/splits"
OUTPUT = ROOT / "data/yolo_dataset"


def rows_for_session(contributor: str, session: str):
    image_dir = RAW / contributor / session
    return sorted(p.relative_to(ROOT).as_posix() for p in image_dir.glob("*.jpg"))


def create_manifests():
    # Seed 42 policy for this five-group collection: preserves whole sessions,
    # puts the hard S04 group in test, and keeps negatives in training.
    assignments = {"train": [("P01", "S01"), ("P01", "S02"), ("P01", "S05")],
                   "val": [("P01", "S03")], "test": [("P01", "S04")]}
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split, groups in assignments.items():
        lines = [item for group in groups for item in rows_for_session(*group)]
        (SPLIT_DIR / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifests():
    result = {}
    seen = {}
    for split in SPLITS:
        path = SPLIT_DIR / f"{split}.txt"
        if not path.exists(): raise RuntimeError(f"missing manifest: {path}")
        result[split] = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for rel in result[split]:
            group = "/".join(Path(rel).parts[2:4])
            if group in seen and seen[group] != split: raise RuntimeError(f"session leakage: {group}")
            seen[group] = split
    expected = {p.relative_to(ROOT).as_posix() for p in RAW.rglob("*.jpg")}
    actual = {p for paths in result.values() for p in paths}
    if expected != actual: raise RuntimeError(f"manifest mismatch: missing={expected-actual}, extra={actual-expected}")
    return result


def label_counts(image_rel: str):
    parts = Path(image_rel).parts
    label = LABELS.joinpath(*parts[2:]).with_suffix(".txt")
    counts = Counter()
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        if line.strip(): counts[int(line.split()[0])] += 1
    return counts


def write_summary(manifests):
    fields = ["split", "contributor", "session", "image_count", "object_count", *[f"{n}_count" for n in CLASSES], "negative_count"]
    rows = []
    for split, paths in manifests.items():
        groups = {}
        for rel in paths:
            parts = Path(rel).parts
            key = (parts[2], parts[3])
            groups.setdefault(key, []).append(rel)
        for (contributor, session), items in sorted(groups.items()):
            class_counts = Counter(); negatives = 0
            for rel in items:
                counts = label_counts(rel); class_counts.update(counts); negatives += not counts
            rows.append([split, contributor, session, len(items), sum(class_counts.values()), *[class_counts[i] for i in range(5)], negatives])
    with (SPLIT_DIR / "split_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)


def build(manifests):
    for split in SPLITS:
        for kind in ("images", "labels"):
            target = OUTPUT / kind / split
            if target.exists(): shutil.rmtree(target)
            target.mkdir(parents=True)
        for rel in manifests[split]:
            source = ROOT / rel
            name = source.name
            shutil.copy2(source, OUTPUT / "images" / split / name)
            parts = Path(rel).parts
            label = LABELS.joinpath(*parts[2:]).with_suffix(".txt")
            shutil.copy2(label, OUTPUT / "labels" / split / f"{source.stem}.txt")
    (OUTPUT / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n" +
        "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--create-splits", action="store_true"); args = parser.parse_args()
    errors, _, _ = validate()
    if errors: raise SystemExit("validation failed; dataset was not built")
    if args.create_splits: create_manifests()
    loaded = load_manifests(); write_summary(loaded); build(loaded)
    print("built:", ", ".join(f"{s}={len(v)}" for s, v in loaded.items()))
