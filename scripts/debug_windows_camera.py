"""Detailed Windows OpenCV USB-camera diagnostic; does not load YOLO."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2


BACKENDS = (("dshow", "DSHOW", cv2.CAP_DSHOW), ("msmf", "MSMF", cv2.CAP_MSMF), ("any", "ANY", cv2.CAP_ANY))
RESOLUTIONS = ((640, 480), (1280, 720), (1920, 1080))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-max", type=int, default=10)
    parser.add_argument("--output-dir", default="results/camera_debug")
    return parser.parse_args()


def test_candidate(index: int, backend_name: str, backend: int, width: int, height: int) -> tuple[bool, object | None, tuple[int, int], float, str]:
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        print(f"Camera {index} / {backend_name} / {width}x{height}: opened=False")
        capture.release()
        return False, None, (0, 0), 0.0, ""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(0.7)
    read_ok, frame = 0, None
    for _ in range(10):
        ok, candidate = capture.read()
        if ok and candidate is not None:
            read_ok += 1
            frame = candidate
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> 8 * position) & 0xFF) for position in range(4)).strip() or "unavailable"
    print(f"Camera {index} / {backend_name} / requested {width}x{height}: opened=True, read frames={read_ok}/10, resolution={actual_width}x{actual_height}, fps={fps:.2f}, FOURCC={fourcc}")
    if read_ok:
        return True, (capture, frame), (actual_width, actual_height), fps, fourcc
    capture.release()
    return False, None, (actual_width, actual_height), fps, fourcc


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Platform: {platform.platform()}")
    print(f"OpenCV: {cv2.__version__}")
    found = []
    for index in range(args.scan_max + 1):
        for backend_key, backend_name, backend in BACKENDS:
            for width, height in RESOLUTIONS:
                usable, held, actual_resolution, fps, fourcc = test_candidate(index, backend_name, backend, width, height)
                if usable and held is not None:
                    capture, frame = held
                    screenshot = output_dir / f"camera_{index}_{backend_key}_{actual_resolution[0]}x{actual_resolution[1]}.jpg"
                    saved = cv2.imwrite(str(screenshot), frame)
                    print(f"  USABLE: Camera {index} / {backend_name}; screenshot_saved={saved}; path={screenshot}")
                    found.append((index, backend_key, backend_name, actual_resolution, fps, fourcc, screenshot if saved else None))
                    capture.release()
                    break
            # A usable configuration for this backend is sufficient; continue next backend.
    print("\nCamera summary:")
    if not found:
        print("Camera access still failed: no candidate from the requested scan range produced a frame.")
        return 1
    for index, key, name, resolution, fps, fourcc, screenshot in found:
        print(f"Camera {index}: backend={key} ({name}), resolution={resolution[0]}x{resolution[1]}, fps={fps:.2f}, FOURCC={fourcc}, screenshot={screenshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
