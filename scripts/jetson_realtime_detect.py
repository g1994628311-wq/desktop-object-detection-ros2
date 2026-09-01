"""Standalone camera inference for a YOLO .pt or Jetson-local TensorRT .engine."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


EXPECTED_NAMES = {0: "mouse", 1: "keyboard", 2: "laptop", 3: "cup", 4: "headphones"}


def source_value(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a .pt or .engine model")
    parser.add_argument("--source", required=True, help="Camera node, video path, or numeric camera index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output", default="results/jetson_yolo11n/demo.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until the source ends")
    parser.add_argument("--warmup-frames", type=int, default=0, help="Frames excluded from benchmark statistics")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--jsonl", help="Optional per-frame JSONL output path")
    parser.add_argument("--csv", dest="csv_path", help="Optional per-detection CSV output path")
    parser.add_argument("--benchmark-csv", help="Optional one-row benchmark CSV output path")
    return parser.parse_args()


def model_names(model: YOLO) -> dict[int, str]:
    names: Any = model.names
    normalized = {int(key): str(value) for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    if normalized != EXPECTED_NAMES:
        raise RuntimeError(f"Class mapping mismatch: expected {EXPECTED_NAMES}, got {normalized}")
    print(f"Class mapping verified: {normalized}")
    return normalized


def open_writer(path: Path, frame: Any, fps: float) -> tuple[cv2.VideoWriter | None, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if writer.isOpened():
        return writer, path
    writer.release()
    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if writer.isOpened():
        print(f"MP4 writer unavailable; using MJPG AVI: {fallback}")
        return writer, fallback
    writer.release()
    print("Video writer unavailable; continuing without saved video.")
    return None, path


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    backend = "TensorRT FP16" if model_path.suffix.lower() == ".engine" else "PyTorch CUDA"
    model = YOLO(str(model_path))
    names = model_names(model)

    capture = cv2.VideoCapture(source_value(args.source), cv2.CAP_V4L2 if args.source.startswith("/dev/") else cv2.CAP_ANY)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    camera_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer: cv2.VideoWriter | None = None
    video_path: Path | None = None
    json_file = None
    csv_file = None
    csv_writer = None
    if args.jsonl:
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        json_file = open(args.jsonl, "w", encoding="utf-8")
    if args.csv_path:
        Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["frame_id", "timestamp", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "fps"])
        csv_writer.writeheader()

    frame_id = 0
    rolling_fps: deque[float] = deque(maxlen=30)
    measured_fps: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    try:
        while args.max_frames <= 0 or frame_id < args.max_frames:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp = time.time()
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=0, verbose=False)[0]
            detections = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                confidence = float(box.conf.item())
                detections.append({"class_id": class_id, "class_name": names[class_id], "confidence": confidence, "bbox_xyxy": [x1, y1, x2, y2]})
                cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{names[class_id]} {confidence:.2f}", (round(x1), max(20, round(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            displayed_fps = sum(rolling_fps) / len(rolling_fps) if rolling_fps else 0.0
            cv2.putText(frame, f"FPS: {displayed_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_id}  Backend: {backend}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            # Pipeline timing intentionally includes capture, inference, postprocess and drawing.
            elapsed = max(time.perf_counter() - started, 1e-9)
            instant_fps = 1.0 / elapsed
            rolling_fps.append(instant_fps)
            if frame_id >= args.warmup_frames:
                measured_fps.append(instant_fps)
                preprocess_ms.append(float(result.speed.get("preprocess", 0.0)))
                inference_ms.append(float(result.speed.get("inference", 0.0)))
                postprocess_ms.append(float(result.speed.get("postprocess", 0.0)))
            if args.save_video and writer is None:
                writer, video_path = open_writer(Path(args.output), frame, camera_fps)
            if writer is not None:
                writer.write(frame)
            if json_file:
                json_file.write(json.dumps({"frame_id": frame_id, "timestamp": timestamp, "fps": instant_fps, "detections": detections}) + "\n")
            if csv_writer:
                for detection in detections:
                    x1, y1, x2, y2 = detection["bbox_xyxy"]
                    csv_writer.writerow({"frame_id": frame_id, "timestamp": timestamp, "confidence": detection["confidence"], "fps": instant_fps, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": detection["class_id"], "class_name": detection["class_name"]})
            if args.display:
                cv2.imshow("Jetson YOLO realtime detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frame_id += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if json_file:
            json_file.close()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()
    if measured_fps:
        benchmark = {
            "backend": backend,
            "model": str(model_path),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "frames": len(measured_fps),
            "warmup_frames": args.warmup_frames,
            "avg_pipeline_fps": sum(measured_fps) / len(measured_fps),
            "min_pipeline_fps": min(measured_fps),
            "max_pipeline_fps": max(measured_fps),
            "avg_preprocess_ms": sum(preprocess_ms) / len(preprocess_ms),
            "avg_inference_ms": sum(inference_ms) / len(inference_ms),
            "avg_postprocess_ms": sum(postprocess_ms) / len(postprocess_ms),
            "video_saved": bool(writer),
            "video_path": str(video_path) if writer else "",
            "device": args.source,
        }
        print("Benchmark: " + json.dumps(benchmark))
        if args.benchmark_csv:
            benchmark_path = Path(args.benchmark_csv)
            benchmark_path.parent.mkdir(parents=True, exist_ok=True)
            with open(benchmark_path, "w", newline="", encoding="utf-8") as benchmark_file:
                writer_fields = list(benchmark)
                benchmark_writer = csv.DictWriter(benchmark_file, fieldnames=writer_fields)
                benchmark_writer.writeheader()
                benchmark_writer.writerow(benchmark)
    else:
        print("No measured frames: increase --max-frames beyond --warmup-frames.")
    print(f"Processed frames: {frame_id}; backend: {backend}; video: {video_path if writer else 'not saved'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
