"""Minimal Windows camera test with no YOLO or model imports."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


BACKENDS = (("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-max", type=int, default=10)
    parser.add_argument("--output-dir", default="results/camera_debug")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    for index in range(args.scan_max + 1):
        for backend_name, backend in BACKENDS:
            capture = cv2.VideoCapture(index, backend)
            opened = capture.isOpened()
            print(f"Camera {index} / {backend_name}: isOpened={opened}")
            if not opened:
                capture.release()
                continue
            time.sleep(1.0)
            reads, frame = 0, None
            for _ in range(20):
                ok, candidate = capture.read()
                if ok and candidate is not None:
                    reads += 1
                    frame = candidate
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            print(f"  successful reads={reads}/20, width={width}, height={height}, fps={fps:.2f}")
            if frame is not None:
                screenshot = output_dir / f"camera_{index}_{backend_name}.jpg"
                saved = cv2.imwrite(str(screenshot), frame)
                print(f"  screenshot_saved={saved}, path={screenshot}")
                successes += 1
            capture.release()
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
