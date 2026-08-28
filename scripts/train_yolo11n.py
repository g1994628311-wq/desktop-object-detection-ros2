#!/usr/bin/env python3
"""Train and evaluate the reproducible YOLO11n baseline v1."""
from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

import torch
import ultralytics
from ultralytics import YOLO

CLASSES = ("mouse", "keyboard", "laptop", "cup", "headphones")
DATA = ROOT / "data/yolo_dataset/dataset.yaml"
RUNS = ROOT / "runs/detect"
RESULTS = ROOT / "results/yolo11n_baseline_v1"


def unique_name(base: str) -> str:
    if not (RUNS / base).exists():
        return base
    number = 2
    while (RUNS / f"{base}_{number}").exists():
        number += 1
    return f"{base}_{number}"


def class_instances(split: str) -> Counter:
    counts = Counter()
    for path in (ROOT / f"data/yolo_dataset/labels/{split}").glob("*.txt"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def metric_rows(split: str, metrics):
    box = metrics.box
    instances = class_instances(split)
    rows = []
    for i, name in enumerate(CLASSES):
        rows.append([split, i, name, float(box.p[i]), float(box.r[i]), float(box.ap50[i]), float(box.maps[i]), instances[i]])
    return rows


def aggregate(split: str, metrics):
    box = metrics.box
    return [split, float(box.mp), float(box.mr), float(box.map50), float(box.map)]


def iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def analyze_errors(best: Path) -> None:
    model = YOLO(str(best), task="detect")
    output = []
    for split in ("val", "test"):
        source = ROOT / f"data/yolo_dataset/images/{split}"
        for result in model.predict(source=str(source), imgsz=640, conf=0.25, device=0, verbose=False, stream=True):
            image = Path(result.path)
            height, width = result.orig_shape
            label = ROOT / f"data/yolo_dataset/labels/{split}/{image.stem}.txt"
            ground_truth = []
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                class_id, x, y, w, h = map(float, line.split())
                ground_truth.append((int(class_id), ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)))
            predictions = [
                (int(cls), tuple(map(float, box)), float(conf))
                for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())
            ]
            used_gt, used_pred = set(), set()
            candidates = sorted(
                ((iou(gt[1], pred[1]), gi, pi) for gi, gt in enumerate(ground_truth)
                 for pi, pred in enumerate(predictions) if gt[0] == pred[0]),
                reverse=True,
            )
            for overlap, gi, pi in candidates:
                if overlap >= 0.5 and gi not in used_gt and pi not in used_pred:
                    used_gt.add(gi); used_pred.add(pi)
                    if predictions[pi][2] < 0.5:
                        output.append([image.name, split, CLASSES[ground_truth[gi][0]], CLASSES[predictions[pi][0]], f"{predictions[pi][2]:.6f}", "low_confidence_correct", f"IoU={overlap:.3f}"])
            for gi, gt in enumerate(ground_truth):
                if gi not in used_gt:
                    best_same = max((iou(gt[1], pred[1]) for pred in predictions if pred[0] == gt[0]), default=0.0)
                    kind = "localization_error" if best_same >= 0.1 else "false_negative"
                    output.append([image.name, split, CLASSES[gt[0]], "", "", kind, f"best same-class IoU={best_same:.3f}"])
            for pi, pred in enumerate(predictions):
                if pi in used_pred:
                    continue
                overlaps = [(iou(pred[1], gt[1]), gt[0]) for gt in ground_truth]
                best_overlap, best_class = max(overlaps, default=(0.0, -1))
                kind = "class_confusion" if best_overlap >= 0.5 and best_class != pred[0] else "false_positive"
                note = f"IoU={best_overlap:.3f} with {CLASSES[best_class]}" if best_class >= 0 else "no ground truth"
                output.append([image.name, split, CLASSES[best_class] if best_class >= 0 else "", CLASSES[pred[0]], f"{pred[2]:.6f}", kind, note])
    with (RESULTS / "error_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "split", "ground_truth", "prediction", "confidence", "error_type", "notes"])
        writer.writerows(output)
    print(f"error cases written: {len(output)}")


def driver_version() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, encoding="utf-8", errors="replace"
        ).strip().splitlines()[0]
    except Exception:
        return "unknown"


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing full baseline training")
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_name = unique_name("yolo11n_baseline_v1")
    started = datetime.now().astimezone()
    start_clock = time.perf_counter()

    model = YOLO("yolo11n.pt", task="detect")
    # Ultralytics resolves dataset.yaml's "path: ." against the process CWD.
    # Run from the YAML directory so the committed portable config stays valid.
    os.chdir(DATA.parent)
    train_result = model.train(
        data=str(DATA), project=str(RUNS), name=run_name,
        epochs=100, patience=20, imgsz=640, batch=16, device=0,
        workers=4, seed=42, deterministic=True, optimizer="auto",
        pretrained=True, plots=True, save=True, cache=False,
    )
    run_dir = Path(train_result.save_dir)
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    if not best.exists() or not last.exists():
        raise RuntimeError("training completed without best.pt and last.pt")

    best_model = YOLO(str(best), task="detect")
    val_metrics = best_model.val(
        data=str(DATA), split="val", imgsz=640, batch=16, device=0,
        workers=4, plots=True, project=str(run_dir), name="val_eval",
    )
    # The test set is evaluated exactly once after best.pt is fixed.
    test_metrics = best_model.val(
        data=str(DATA), split="test", imgsz=640, batch=16, device=0,
        workers=4, plots=True, project=str(run_dir), name="test_eval",
    )
    best_model.predict(
        source=str(ROOT / "data/yolo_dataset/images/val"), imgsz=640,
        conf=0.25, device=0, save=True, project=str(RESULTS / "predictions"), name="val",
    )
    best_model.predict(
        source=str(ROOT / "data/yolo_dataset/images/test"), imgsz=640,
        conf=0.25, device=0, save=True, project=str(RESULTS / "predictions"), name="test",
    )

    duration = time.perf_counter() - start_clock
    ended = datetime.now().astimezone()
    results_csv = run_dir / "results.csv"
    epochs_completed = sum(1 for _ in csv.DictReader(results_csv.open(encoding="utf-8-sig")))
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8-sig")))
    best_epoch = max(
        range(len(rows)),
        key=lambda i: 0.1 * float(rows[i]["metrics/mAP50(B)"]) + 0.9 * float(rows[i]["metrics/mAP50-95(B)"])
    ) + 1

    with (RESULTS / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["split", "precision", "recall", "mAP50", "mAP50_95"])
        writer.writerow(aggregate("val", val_metrics)); writer.writerow(aggregate("test", test_metrics))
    with (RESULTS / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["split", "class_id", "class_name", "precision", "recall", "AP50", "AP50_95", "instances"])
        writer.writerows(metric_rows("val", val_metrics)); writer.writerows(metric_rows("test", test_metrics))

    environment = {
        "date_started": started.isoformat(), "date_ended": ended.isoformat(),
        "Python version": platform.python_version(), "PyTorch version": torch.__version__,
        "CUDA version": torch.version.cuda, "Ultralytics version": ultralytics.__version__,
        "GPU": torch.cuda.get_device_name(0), "driver": driver_version(),
        "training image size": 640, "batch size": 16, "epochs requested": 100,
        "epochs completed": epochs_completed, "best epoch": best_epoch,
        "early stopping": epochs_completed < 100, "seed": 42,
        "training duration seconds": f"{duration:.3f}", "run directory": run_dir,
        "best.pt": best, "last.pt": last,
    }
    (RESULTS / "training_environment.txt").write_text(
        "".join(f"{key}: {value}\n" for key, value in environment.items()), encoding="utf-8"
    )
    analyze_errors(best)
    print(f"BASELINE_COMPLETE run={run_dir} best_epoch={best_epoch} duration={duration:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", type=Path, metavar="BEST_PT")
    args = parser.parse_args()
    if args.analyze_only:
        RESULTS.mkdir(parents=True, exist_ok=True)
        analyze_errors(args.analyze_only.resolve())
    else:
        main()
