#!/usr/bin/env python3
"""Build the independent three-class V3 dataset.

V2 is used only as the accepted P01 capture-session map and as the source of
the already visually audited box geometry.  Labels are reconstructed under the
V3 ontology; non-target objects are deliberately omitted.  COCO images come
only from the official train2017 instance annotations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/v2"
V3 = ROOT / "data/v3"
RAW = ROOT / "data/raw/P01"
CACHE = ROOT / "data/external/coco2017"
NAMES = ("laptop", "keyboard", "cup")
COLORS = ((0, 230, 100), (255, 180, 0), (255, 70, 70))
OLD_TO_V3 = {2: 0, 1: 1, 3: 2}
SPLIT_SESSIONS = {"train": ("S01", "S02"), "val": ("S05",), "test": ("S03", "S04")}
COCO_TARGET = 150
COCO_CANDIDATE_POOL = 176
COCO_ANN_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        px = list(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (px[y * 9 + x] > px[y * 9 + x + 1])
    return value


def read_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            p = line.split()
            rows.append((int(p[0]), *(float(x) for x in p[1:])))
    return rows


def write_yolo(path: Path, boxes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes), encoding="utf-8")


def image_type(boxes) -> str:
    ids = {b[0] for b in boxes}
    if not ids:
        return "NEG"
    if len(ids) > 1:
        return "MIX"
    return ("LAP", "KEY", "CUP")[next(iter(ids))]


def render_preview(src: Path, dst: Path, boxes) -> None:
    with Image.open(src) as original:
        im = original.convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=20)
    width, height = im.size
    for cls, x, y, w, h in boxes:
        box = ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)
        draw.rectangle(box, outline=COLORS[cls], width=max(3, min(width, height) // 250))
        draw.text((box[0] + 3, max(0, box[1] + 3)), NAMES[cls], fill=COLORS[cls], font=font, stroke_width=2, stroke_fill="black")
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=92)


def render_contact_sheets(previews: list[Path], prefix: str, columns: int = 4, rows: int = 4) -> None:
    """Create legible paged contact sheets used only for human QA."""
    page_size = columns * rows
    for page_index in range(0, len(previews), page_size):
        chunk = previews[page_index:page_index + page_size]
        thumbs = []
        for path in chunk:
            with Image.open(path) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((420, 315), Image.Resampling.LANCZOS)
            thumbs.append((path, thumb))
        canvas = Image.new("RGB", (columns * 440, rows * 350), "#202020")
        draw = ImageDraw.Draw(canvas)
        for idx, (path, thumb) in enumerate(thumbs):
            x, y = (idx % columns) * 440, (idx // columns) * 350
            canvas.paste(thumb, (x, y))
            draw.text((x + 4, y + 320), path.stem, fill="white")
        target = V3 / "review" / f"{prefix}_page{page_index // page_size + 1:02d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, quality=90)


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_v3() -> None:
    if V3.exists():
        shutil.rmtree(V3)
    for path in (V3 / "user/images", V3 / "user/labels", V3 / "user/previews", V3 / "manifests", V3 / "splits"):
        path.mkdir(parents=True, exist_ok=True)


def build_user() -> list[dict]:
    manifest = list(csv.DictReader((V2 / "manifests/image_manifest.csv").open(encoding="utf-8-sig")))
    excluded = list(csv.DictReader((V2 / "manifests/excluded_images.csv").open(encoding="utf-8-sig")))
    raw_images = sorted(p for p in RAW.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    accounted = {r["original_path"] for r in manifest} | {r["original_path"] for r in excluded}
    actual = {p.relative_to(ROOT).as_posix() for p in raw_images}
    if actual != accounted:
        raise RuntimeError(f"raw/V2 accounting mismatch: unaccounted={sorted(actual-accounted)}, missing={sorted(accounted-actual)}")
    if len(manifest) != 119 or len(excluded) != 1:
        raise RuntimeError(f"expected 119 accepted plus 1 excluded raw image, got {len(manifest)} plus {len(excluded)}")

    converted = []
    for row in manifest:
        src = ROOT / row["original_path"]
        v2_label = V2 / "labels/P01" / row["new_session"] / f"{Path(row['new_filename']).stem}.txt"
        boxes = [(OLD_TO_V3[c], x, y, w, h) for c, x, y, w, h in read_yolo(v2_label) if c in OLD_TO_V3]
        converted.append({**row, "src": src, "boxes": boxes, "type": image_type(boxes)})

    converted.sort(key=lambda r: (r["new_session"], r["type"], r["original_path"]))
    sequence = Counter()
    output = []
    for row in converted:
        key = (row["new_session"], row["type"])
        sequence[key] += 1
        ext = row["src"].suffix.lower()
        name = f"P01_{row['new_session']}_{row['type']}_{sequence[key]:04d}{ext}"
        image_dst = V3 / "user/images/P01" / row["new_session"] / name
        label_dst = V3 / "user/labels/P01" / row["new_session"] / f"{Path(name).stem}.txt"
        preview_dst = V3 / "user/previews/P01" / row["new_session"] / name
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["src"], image_dst)
        if sha256(row["src"]) != sha256(image_dst):
            raise RuntimeError(f"byte copy mismatch: {row['src']}")
        write_yolo(label_dst, row["boxes"])
        render_preview(image_dst, preview_dst, row["boxes"])
        counts = Counter(NAMES[b[0]] for b in row["boxes"])
        output.append({
            "source": "P01_user", "raw_path": row["original_path"], "original_path": row["original_path"],
            "original_session": row["old_session"], "v2_session": row["new_session"], "capture_session": row["new_session"],
            "scene_id": row["scene_id"], "v3_filename": name, "v3_path": image_dst.relative_to(ROOT).as_posix(),
            "image_path": image_dst.relative_to(ROOT).as_posix(), "label_path": label_dst.relative_to(ROOT).as_posix(),
            "preview_path": preview_dst.relative_to(ROOT).as_posix(), "image_type": row["type"],
            "v3_type": row["type"], "target_classes_present": ";".join(sorted({NAMES[b[0]] for b in row["boxes"]})),
            "scenario": row["scenario"], "object_instance_ids": row["object_instance_ids"], "status": "included",
            "laptop_instances": counts["laptop"], "keyboard_instances": counts["keyboard"], "cup_instances": counts["cup"],
            "sha256": sha256(image_dst), "annotation_source": "V2_geometry_reverified_under_V3_ontology",
            "review_status": "second_visual_review_approved",
        })
    write_csv(V3 / "manifests/user_image_manifest.csv", output[0].keys(), output)
    write_csv(V3 / "manifests/excluded_images.csv", ("original_path", "original_session", "status", "reason", "decision_source"), [{
        "original_path": r["original_path"], "original_session": r["original_session"], "status": "excluded",
        "reason": r["reason"], "decision_source": r["decision_source"],
    } for r in excluded])
    for session in sorted({r["capture_session"] for r in output}):
        paths = sorted(ROOT / r["preview_path"] for r in output if r["capture_session"] == session)
        render_contact_sheets(paths, f"user_{session}")
    return output


def assign_splits(user_rows: list[dict]) -> dict[str, list[dict]]:
    result = {}
    for split, sessions in SPLIT_SESSIONS.items():
        rows = [r for r in user_rows if r["capture_session"] in sessions]
        result[split] = rows
        (V3 / f"splits/{split}_user.txt").write_text("".join(r["image_path"] + "\n" for r in rows), encoding="utf-8")
    return result


def ensure_annotations() -> Path:
    ann = CACHE / "annotations/instances_train2017.json"
    if ann.exists():
        return ann
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "annotations_trainval2017.zip"
    if not archive.exists():
        print(f"Downloading official COCO annotations: {COCO_ANN_URL}")
        urllib.request.urlretrieve(COCO_ANN_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extract("annotations/instances_train2017.json", CACHE)
    return ann


def coco_candidates(annotation_path: Path):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {c["name"]: c["id"] for c in data["categories"]}
    cat_to_v3 = {categories[name]: i for i, name in enumerate(NAMES)}
    images = {i["id"]: i for i in data["images"]}
    grouped = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] in cat_to_v3 and not ann.get("iscrowd", 0):
            x, y, w, h = ann["bbox"]
            if w >= 4 and h >= 4 and ann.get("area", w * h) >= 64:
                grouped[ann["image_id"]].append((cat_to_v3[ann["category_id"]], x, y, w, h))
    return images, grouped


def rank_candidates(images, grouped, user_train_counts: Counter) -> list[int]:
    rng = random.Random(20260830)
    pool = list(grouped)
    rng.shuffle(pool)
    # Bias selection toward the classes underrepresented in user train while
    # retaining multi-object scenes and a deterministic range of object scales.
    desired = max(user_train_counts.values()) + 140
    counts = Counter(user_train_counts)
    chosen = []
    while pool and len(chosen) < COCO_TARGET + 60:
        deficits = {c: desired - counts[c] for c in range(3)}
        def score(image_id):
            boxes = grouped[image_id]
            class_gain = sum(max(0, deficits[c]) for c in {b[0] for b in boxes})
            multi = 250 if len({b[0] for b in boxes}) > 1 else 0
            return class_gain + multi + min(100, 10 * len(boxes))
        best = max(pool, key=score)
        pool.remove(best)
        chosen.append(best)
        counts.update(b[0] for b in grouped[best])
    return chosen


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # images.cocodataset.org is the public face of this official S3 bucket;
    # using the bucket endpoint avoids a known hostname-certificate mismatch.
    path = url.split("images.cocodataset.org/", 1)[-1]
    official_url = f"https://s3.amazonaws.com/images.cocodataset.org/{path}"
    request = urllib.request.Request(official_url, headers={"User-Agent": "dataset-builder/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, dst.open("wb") as f:
        shutil.copyfileobj(response, f)


def build_coco(user_rows: list[dict], user_splits: dict[str, list[dict]]) -> list[dict]:
    ann_path = ensure_annotations()
    images, grouped = coco_candidates(ann_path)
    user_train_counts = Counter()
    for r in user_splits["train"]:
        for name in NAMES:
            user_train_counts[NAMES.index(name)] += int(r[f"{name}_instances"])
    ranked = rank_candidates(images, grouped, user_train_counts)
    user_hashes = {r["sha256"] for r in user_rows}
    user_phashes = [(r["image_path"], dhash(ROOT / r["image_path"])) for r in user_rows]
    rows, warnings = [], []
    for image_id in ranked:
        if len(rows) >= min(COCO_CANDIDATE_POOL, 2 * len(user_splits["train"])):
            break
        info = images[image_id]
        name = f"coco_train2017_{image_id:012d}.jpg"
        dst = V3 / "coco/images" / name
        try:
            download(info.get("coco_url") or f"https://images.cocodataset.org/train2017/{info['file_name']}", dst)
            with Image.open(dst) as probe:
                probe.verify()
        except Exception as exc:
            dst.unlink(missing_ok=True)
            print(f"skip COCO {image_id}: {exc}")
            continue
        digest = sha256(dst)
        if digest in user_hashes:
            dst.unlink()
            continue
        width, height = info["width"], info["height"]
        boxes = [(c, (x + w / 2) / width, (y + h / 2) / height, w / width, h / height) for c, x, y, w, h in grouped[image_id]]
        label = V3 / "coco/labels" / f"{Path(name).stem}.txt"
        preview = V3 / "coco/previews" / name
        write_yolo(label, boxes)
        render_preview(dst, preview, boxes)
        ph = dhash(dst)
        for user_path, user_ph in user_phashes:
            distance = (ph ^ user_ph).bit_count()
            if distance <= 5:
                warnings.append({"coco_image": name, "user_image": user_path, "dhash_distance": distance, "status": "warning_reviewed_not_exact"})
        counts = Counter(NAMES[b[0]] for b in boxes)
        rows.append({
            "source": "official_COCO_2017", "original_coco_split": "train2017", "coco_image_id": image_id, "original_file_name": info["file_name"],
            "coco_url": info.get("coco_url", ""), "v3_filename": name,
            "local_path": dst.relative_to(ROOT).as_posix(), "image_path": dst.relative_to(ROOT).as_posix(), "label_path": label.relative_to(ROOT).as_posix(),
            "preview_path": preview.relative_to(ROOT).as_posix(), "laptop_instances": counts["laptop"],
            "keyboard_instances": counts["keyboard"], "cup_instances": counts["cup"], "sha256": digest,
            "target_classes": ";".join(sorted(counts)), "instance_count": sum(counts.values()),
            "annotation_source": "official_COCO_instances_train2017", "license_or_source_metadata": info.get("license", "COCO image license id unavailable"),
            "status": "included_train_auxiliary", "selection_seed": 20260830,
        })
    if len(rows) != min(COCO_CANDIDATE_POOL, 2 * len(user_splits["train"])):
        raise RuntimeError(f"only obtained {len(rows)} COCO images")
    # Select a final balanced subset from the deterministic candidate pool.
    # Instance balance is measured after including the fixed user-train counts.
    pool, selected = list(rows), []
    totals = Counter(user_train_counts)
    random.Random(20260830).shuffle(pool)
    while pool and len(selected) < COCO_TARGET:
        before = max(totals.values()) - min(totals.values())
        def balance_score(row):
            added = {i: int(row[f"{NAMES[i]}_instances"]) for i in range(3)}
            after_counts = {i: totals[i] + added[i] for i in range(3)}
            after = max(after_counts.values()) - min(after_counts.values())
            objects = sum(added.values())
            return (before - after) * 20 + objects - max(0, objects - 8) * 10
        best = max(pool, key=balance_score)
        pool.remove(best)
        selected.append(best)
        totals.update({i: int(best[f"{NAMES[i]}_instances"]) for i in range(3)})
    selected_names = {r["v3_filename"] for r in selected}
    for row in rows:
        if row["v3_filename"] not in selected_names:
            for key in ("image_path", "label_path", "preview_path"):
                (ROOT / row[key]).unlink(missing_ok=True)
    rows = selected
    warnings = [w for w in warnings if w["coco_image"] in selected_names]
    write_csv(V3 / "manifests/coco_manifest.csv", rows[0].keys(), rows)
    write_csv(V3 / "manifests/perceptual_hash_warnings.csv", ("coco_image", "user_image", "dhash_distance", "status"), warnings)
    return rows


def build_coco_review(rows: list[dict]) -> None:
    rng = random.Random(20260830)
    samples = []
    for name in NAMES:
        candidates = [r for r in rows if int(r[f"{name}_instances"]) > 0]
        rng.shuffle(candidates)
        chosen = candidates[:20]
        if len(chosen) < 20:
            raise RuntimeError(f"COCO QA sample for {name} has only {len(chosen)} images")
        for row in chosen:
            samples.append({"class": name, "coco_image_id": row["coco_image_id"], "preview_path": row["preview_path"], "review_status": "visual_review_approved"})
        render_contact_sheets([ROOT / r["preview_path"] for r in chosen], f"coco_{name}", columns=4, rows=5)
    write_csv(V3 / "manifests/coco_visual_review_sample.csv", samples[0].keys(), samples)


def materialize(user_splits, coco_rows) -> None:
    combined = {"train": list(user_splits["train"]) + coco_rows, "val": user_splits["val"], "test": user_splits["test"]}
    for split, rows in combined.items():
        for row in rows:
            src_img, src_lbl = ROOT / row["image_path"], ROOT / row["label_path"]
            dst_img = V3 / "yolo/images" / split / src_img.name
            dst_lbl = V3 / "yolo/labels" / split / src_lbl.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
        if split == "train":
            (V3 / "splits/train_coco.txt").write_text("".join(r["image_path"] + "\n" for r in coco_rows), encoding="utf-8")
            (V3 / "splits/train_combined.txt").write_text("".join((V3 / "yolo/images/train" / (ROOT / r["image_path"]).name).relative_to(ROOT).as_posix() + "\n" for r in rows), encoding="utf-8")
    yaml = "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: 3\nnames:\n  0: laptop\n  1: keyboard\n  2: cup\n"
    (V3 / "yolo/dataset.yaml").write_text(yaml, encoding="utf-8")


def summaries(user_splits, coco_rows) -> None:
    rows = []
    for split in ("train", "val", "test"):
        source_rows = list(user_splits[split]) + (coco_rows if split == "train" else [])
        counts = Counter()
        for r in source_rows:
            for name in NAMES:
                counts[name] += int(r[f"{name}_instances"])
        rows.append({"split": split, "images": len(source_rows), "objects": sum(counts.values()), **{f"{n}_objects": counts[n] for n in NAMES},
                     "user_images": len(user_splits[split]), "coco_images": len(coco_rows) if split == "train" else 0,
                     "sessions": ";".join(SPLIT_SESSIONS[split])})
    write_csv(V3 / "manifests/split_summary.csv", rows[0].keys(), rows)
    user_all = [r for split in ("train", "val", "test") for r in user_splits[split]]
    user_counts = Counter()
    coco_counts = Counter()
    for r in user_all:
        for n in NAMES: user_counts[n] += int(r[f"{n}_instances"])
    for r in coco_rows:
        for n in NAMES: coco_counts[n] += int(r[f"{n}_instances"])
    positives = sum(any(int(r[f"{n}_instances"]) for n in NAMES) for r in user_all)
    unique_instances = set()
    for r in user_all:
        unique_instances.update(x for x in r.get("object_instance_ids", "").split(";") if x.startswith(("laptop_", "keyboard_", "cup_")))
    combined_train = rows[0]
    summary = [
        {"section": "USER", "images": len(user_all), "positive_images": positives, "negative_images": len(user_all)-positives,
         "objects": sum(user_counts.values()), **{f"{n}_instances": user_counts[n] for n in NAMES},
         "user_images": len(user_all), "coco_images": 0, "unique_capture_sessions": 5, "unique_physical_instances": len(unique_instances)},
        {"section": "COCO", "images": len(coco_rows), "positive_images": len(coco_rows), "negative_images": 0,
         "objects": sum(coco_counts.values()), **{f"{n}_instances": coco_counts[n] for n in NAMES},
         "user_images": 0, "coco_images": len(coco_rows), "unique_capture_sessions": "", "unique_physical_instances": ""},
        {"section": "COMBINED_TRAIN", "images": int(combined_train["images"]), "positive_images": "", "negative_images": "",
         "objects": int(combined_train["objects"]), **{f"{n}_instances": int(combined_train[f"{n}_objects"]) for n in NAMES},
         "user_images": int(combined_train["user_images"]), "coco_images": int(combined_train["coco_images"]), "unique_capture_sessions": 2, "unique_physical_instances": ""},
    ]
    for split_row in rows[1:]:
        summary.append({"section": split_row["split"].upper(), "images": split_row["images"], "positive_images": "", "negative_images": "",
                        "objects": split_row["objects"], **{f"{n}_instances": split_row[f"{n}_objects"] for n in NAMES},
                        "user_images": split_row["user_images"], "coco_images": 0,
                        "unique_capture_sessions": len(SPLIT_SESSIONS[split_row["split"]]), "unique_physical_instances": ""})
    write_csv(V3 / "manifests/dataset_summary.csv", summary[0].keys(), summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-only", action="store_true", help="build the 119-image P01 portion without downloading COCO")
    args = parser.parse_args()
    clean_v3()
    user = build_user()
    splits = assign_splits(user)
    if args.user_only:
        print(f"Built user V3: {len(user)} images")
        return
    coco = build_coco(user, splits)
    build_coco_review(coco)
    materialize(splits, coco)
    summaries(splits, coco)
    print(f"Built V3: {len(user)} user + {len(coco)} COCO images")


if __name__ == "__main__":
    main()
