#!/usr/bin/env python3
"""Validate canonical YOLO labels and print dataset statistics."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
LABELS = ROOT / "data/labels"
CLASSES = tuple((ROOT / "configs/classes.txt").read_text(encoding="utf-8-sig").splitlines())
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def inventory(raw=RAW, labels_root=LABELS):
    images = {p.relative_to(raw).with_suffix(""): p for p in raw.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    labels = {p.relative_to(labels_root).with_suffix(""): p for p in labels_root.rglob("*.txt")}
    return images, labels


def validate(verbose: bool = True, raw=RAW, labels_root=LABELS):
    images, labels = inventory(raw, labels_root)
    errors: list[str] = []
    warnings: list[str] = []
    objects = Counter()
    class_images = Counter()
    contributor_images = Counter()
    session_images = Counter()
    negative_images = 0

    for key in sorted(images, key=str):
        parts = key.parts
        contributor_images[parts[0]] += 1
        session_images["/".join(parts[:2])] += 1
        label = labels.get(key)
        if label is None:
            errors.append(f"missing label: {key}")
            continue
        seen = set()
        nonempty = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not nonempty:
            negative_images += 1
            if "NEG" not in key.name.upper():
                warnings.append(f"unexpected empty label: {key}")
        elif "NEG" in key.name.upper():
            errors.append(f"NEG image contains annotations (session anomaly): {key}")
        for number, line in enumerate(nonempty, 1):
            fields = line.split()
            if len(fields) != 5:
                errors.append(f"malformed {label.relative_to(ROOT)}:{number}: expected 5 columns")
                continue
            try:
                class_id = int(fields[0])
                x, y, width, height = map(float, fields[1:])
            except ValueError:
                errors.append(f"malformed {label.relative_to(ROOT)}:{number}: non-numeric value")
                continue
            if not 0 <= class_id < len(CLASSES):
                errors.append(f"invalid class {label.relative_to(ROOT)}:{number}: {class_id}")
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                errors.append(f"invalid center {label.relative_to(ROOT)}:{number}")
            if not 0 < width <= 1 or not 0 < height <= 1:
                errors.append(f"invalid size {label.relative_to(ROOT)}:{number}")
            if x - width / 2 < -1e-6 or y - height / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y + height / 2 > 1 + 1e-6:
                errors.append(f"bbox outside image {label.relative_to(ROOT)}:{number}")
            if 0 <= class_id < len(CLASSES):
                objects[class_id] += 1
                seen.add(class_id)
        class_images.update(seen)

    for key in sorted(set(labels) - set(images), key=str):
        errors.append(f"orphan label: {key}")

    stats = {
        "images": len(images), "labels": len(labels), "objects": sum(objects.values()),
        "negative_images": negative_images, "objects_by_class": objects,
        "images_by_class": class_images, "images_by_contributor": contributor_images,
        "images_by_session": session_images,
    }
    if verbose:
        print(f"images={stats['images']} labels={stats['labels']} objects={stats['objects']} negatives={negative_images}")
        for class_id, name in enumerate(CLASSES):
            print(f"class {class_id} {name}: images={class_images[class_id]} instances={objects[class_id]}")
        for name, count in sorted(contributor_images.items()): print(f"contributor {name}: images={count}")
        for name, count in sorted(session_images.items()): print(f"session {name}: images={count}")
        for warning in warnings: print(f"WARNING: {warning}")
        for error in errors: print(f"ERROR: {error}")
        print(f"validation errors={len(errors)} warnings={len(warnings)}")
    return errors, warnings, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--curated", action="store_true", help="deprecated alias for V2")
    parser.add_argument("--v2", action="store_true", help="validate data/v2/images/P01 and data/v2/labels/P01")
    args = parser.parse_args()
    found_errors, _, _ = validate(not args.quiet, ROOT / "data/v2/images/P01", ROOT / "data/v2/labels/P01") if (args.curated or args.v2) else validate(not args.quiet)
    raise SystemExit(1 if found_errors else 0)
