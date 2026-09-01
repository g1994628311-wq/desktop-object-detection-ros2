"""Windows USB-camera qualitative and pipeline-performance validation for final YOLO11n."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2


EXPECTED_NAMES = {0: "mouse", 1: "keyboard", 2: "laptop", 3: "cup", 4: "headphones"}
BACKENDS = (("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF), ("default", cv2.CAP_ANY))
BACKEND_KEYS = {"dshow": ("DirectShow", cv2.CAP_DSHOW), "msmf": ("Media Foundation", cv2.CAP_MSMF), "any": ("default", cv2.CAP_ANY)}
RESOLUTIONS = ((640, 480), (1280, 720), (1920, 1080))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/detect/yolo11n_final/weights/best.pt")
    parser.add_argument("--camera", type=int, default=1, help="Fixed USB camera index (default: 1)")
    parser.add_argument("--backend", choices=("dshow", "msmf", "any"), default="dshow")
    parser.add_argument("--scan-max", type=int, default=10, help="Highest camera index tested when --camera is omitted")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0", help="Ultralytics device, default NVIDIA GPU 0")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", default="results/usb_camera_test")
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until Q is pressed")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def print_test_plan() -> None:
    print("Suggested qualitative test order:")
    for item in ("mouse", "keyboard", "laptop", "cup", "headphones", "laptop + keyboard", "keyboard + mouse", "cup + headphones", "three or more classes"):
        print(f"  - {item}")
    print("For each object, vary distance, angle and light occlusion.")
    print("Keyboard focus: stability, confidence, near/far misses, laptop built-in keyboard false positives, and external keyboard+laptop separation.")


def backend_options(requested: str) -> tuple[tuple[str, str, int], ...]:
    if requested != "auto":
        name, backend = BACKEND_KEYS[requested]
        return ((requested, name, backend),)
    return (("dshow", "DirectShow", cv2.CAP_DSHOW), ("msmf", "Media Foundation", cv2.CAP_MSMF), ("any", "default", cv2.CAP_ANY))


def open_camera(index: int, requested_backend: str, width: int | None, height: int | None, verbose: bool = False) -> tuple[cv2.VideoCapture | None, str, tuple[int, int]]:
    resolutions = ((width, height),) if width and height else RESOLUTIONS
    for backend_key, name, backend in backend_options(requested_backend):
        for requested_width, requested_height in resolutions:
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                if verbose:
                    print(f"Camera {index} / {backend_key} / {requested_width}x{requested_height}: opened=False")
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
            capture.set(cv2.CAP_PROP_FPS, 30)
            time.sleep(0.7)
            success_count, frame_size = 0, (0, 0)
            for _ in range(10):
                ok, frame = capture.read()
                if ok and frame is not None:
                    success_count += 1
                    frame_size = (frame.shape[1], frame.shape[0])
            if verbose:
                fps = capture.get(cv2.CAP_PROP_FPS)
                print(f"Camera {index} / {backend_key} / requested {requested_width}x{requested_height}: opened=True, read frames={success_count}/10, resolution={frame_size[0]}x{frame_size[1]}, fps={fps:.2f}")
            if success_count:
                return capture, backend_key, frame_size
            capture.release()
    return None, "unavailable", (0, 0)


def probe_cameras(scan_max: int, requested_backend: str, width: int | None, height: int | None) -> dict[int, tuple[str, tuple[int, int]]]:
    available: dict[int, tuple[str, tuple[int, int]]] = {}
    for index in range(scan_max + 1):
        capture, backend, frame_size = open_camera(index, requested_backend, width, height, verbose=True)
        if capture is None:
            print(f"Camera {index}: unavailable")
            continue
        print(f"Camera {index}: available ({backend}, {frame_size[0]}x{frame_size[1]})")
        available[index] = (backend, frame_size)
        capture.release()
    return available


def select_camera(requested: int | None, available: dict[int, tuple[str, tuple[int, int]]]) -> int:
    if requested is not None:
        if requested not in available:
            raise RuntimeError(f"Camera {requested} is unavailable or cannot read three frames.")
        return requested
    if not available:
        raise RuntimeError("No usable camera was found among indexes 0 through 4.")
    # Prefer the highest reasonable capture resolution, then the lower index.
    return max(available, key=lambda index: (available[index][1][0] * available[index][1][1], -index))


def verify_cuda(device: str) -> None:
    if device.lower() == "cpu":
        print("Device explicitly set to CPU.")
        return
    try:
        import torch
    except Exception as error:
        raise RuntimeError(f"PyTorch cannot be imported: {error}") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu only for a deliberate CPU check.")
    print(f"CUDA available: True ({torch.cuda.get_device_name(int(device))})")


def normalize_names(names: Any) -> dict[int, str]:
    return {int(key): str(value) for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(names))


def make_video_writer(output_dir: Path, frame: Any, fps: float) -> tuple[cv2.VideoWriter | None, Path | None]:
    height, width = frame.shape[:2]
    mp4_path = output_dir / "demo.mp4"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if writer.isOpened():
        return writer, mp4_path
    writer.release()
    avi_path = output_dir / "demo.avi"
    writer = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if writer.isOpened():
        print(f"MP4 codec unavailable; saving MJPG AVI instead: {avi_path}")
        return writer, avi_path
    writer.release()
    print("Video encoder unavailable; continuing without saved video.")
    return None, None


def main() -> int:
    args = parse_args()
    print_test_plan()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise RuntimeError(f"Model checkpoint not found: {model_path}")
    # Single-open mode: this handle is retained through warm-up, inference and shutdown.
    camera_index = args.camera
    capture, backend, resolution = open_camera(camera_index, args.backend, args.width, args.height, verbose=True)
    if capture is None:
        raise RuntimeError(f"Camera {camera_index} is unavailable or cannot read validation frames.")
    print(f"Selected camera: {camera_index} ({backend}, {resolution[0]}x{resolution[1]}); reusing this single capture handle.")
    verify_cuda(args.device)
    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError(f"Ultralytics cannot be imported: {error}") from error
    model = YOLO(str(model_path))
    names = normalize_names(model.names)
    print(f"Model names: {names}")
    if names != EXPECTED_NAMES:
        raise RuntimeError(f"Model class mapping mismatch; expected {EXPECTED_NAMES}, got {names}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, jsonl_path = output_dir / "detections.csv", output_dir / "detections.jsonl"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    jsonl_file = open(jsonl_path, "w", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["frame_id", "timestamp", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "fps"])
    csv_writer.writeheader()
    writer: cv2.VideoWriter | None = None
    video_path: Path | None = None
    video_writer_attempted = False
    rolling_fps: deque[float] = deque(maxlen=30)
    benchmark_fps: list[float] = []
    inference_ms: list[float] = []
    class_counts: defaultdict[str, int] = defaultdict(int)
    confidence_sum: defaultdict[str, float] = defaultdict(float)
    frame_id, warmup = 0, 30
    paused = False
    consecutive_read_failures = 0
    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    if reported_fps < 1.0:
        print(f"Camera reported invalid FPS ({reported_fps}); using 30.0 FPS for video encoding.")
        reported_fps = 30.0
    try:
        while args.max_frames <= 0 or frame_id < args.max_frames:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = False
                continue
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_read_failures += 1
                print(f"Camera frame read failed ({consecutive_read_failures}/5); retrying.")
                if consecutive_read_failures >= 5:
                    print("Camera read failed five consecutive times; ending test.")
                    break
                continue
            consecutive_read_failures = 0
            timestamp = time.time()
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
            detections = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = names[class_id]
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append({"class_id": class_id, "class_name": class_name, "confidence": confidence, "bbox_xyxy": [x1, y1, x2, y2]})
                class_counts[class_name] += 1
                confidence_sum[class_name] += confidence
                cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (round(x1), max(22, round(y1) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            display_fps = sum(rolling_fps) / len(rolling_fps) if rolling_fps else 0.0
            cv2.putText(frame, "Model: YOLO11n", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Inference: {result.speed.get('inference', 0.0):.1f} ms", (10, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_id}  Detections: {len(detections)}", (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            pipeline_fps = 1.0 / max(time.perf_counter() - started, 1e-9)
            rolling_fps.append(pipeline_fps)
            if frame_id >= warmup:
                benchmark_fps.append(pipeline_fps)
                inference_ms.append(float(result.speed.get("inference", 0.0)))
            if args.save_video and not video_writer_attempted:
                video_writer_attempted = True
                writer, video_path = make_video_writer(output_dir, frame, reported_fps)
            if writer is not None:
                writer.write(frame)
            jsonl_file.write(json.dumps({"frame_id": frame_id, "timestamp": timestamp, "fps": pipeline_fps, "detections": detections}) + "\n")
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                csv_writer.writerow({"frame_id": frame_id, "timestamp": timestamp, "class_id": detection["class_id"], "class_name": detection["class_name"], "confidence": detection["confidence"], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "fps": pipeline_fps})
            if not args.no_display:
                cv2.imshow("USB Camera YOLO11n Model Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    screenshot_dir = output_dir / "screenshots"
                    screenshot_dir.mkdir(exist_ok=True)
                    screenshot = screenshot_dir / f"frame_{frame_id:06d}.jpg"
                    cv2.imwrite(str(screenshot), frame)
                    print(f"Saved screenshot: {screenshot}")
                if key == ord(" "):
                    paused = True
                    print("Paused; press SPACE to continue or Q to quit.")
            frame_id += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        jsonl_file.close()
        cv2.destroyAllWindows()

    print("\n================================")
    print("USB CAMERA MODEL TEST SUMMARY")
    print("================================")
    print(f"Camera index: {camera_index} ({backend})")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print(f"Frames processed: {frame_id}")
    print(f"Benchmark frames (after {warmup} warm-up): {len(benchmark_fps)}")
    print(f"Average pipeline FPS: {sum(benchmark_fps) / len(benchmark_fps):.3f}" if benchmark_fps else "Average pipeline FPS: NOT TESTED")
    print(f"Average inference ms: {sum(inference_ms) / len(inference_ms):.3f}" if inference_ms else "Average inference ms: NOT TESTED")
    print("Detection counts / average confidence:")
    for class_name in EXPECTED_NAMES.values():
        count = class_counts[class_name]
        average_confidence = confidence_sum[class_name] / count if count else 0.0
        print(f"  {class_name}: {count} / {average_confidence:.3f}" if count else f"  {class_name}: 0 / N/A")
    print(f"Output video: {video_path if video_path else 'not saved'}")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
