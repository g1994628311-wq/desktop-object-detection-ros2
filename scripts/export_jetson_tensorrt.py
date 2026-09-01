"""Export the fixed YOLO11n checkpoint to a TensorRT FP16 engine on Jetson."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo11n_final_best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    started = time.perf_counter()
    engine_path = YOLO(args.model).export(format="engine", imgsz=args.imgsz, half=True, device=0)
    engine = Path(engine_path)
    print(f"Engine: {engine}")
    print(f"Size bytes: {engine.stat().st_size}")
    print(f"Export seconds: {time.perf_counter() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
