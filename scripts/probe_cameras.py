"""Probe available V4L2 camera nodes with OpenCV; make no device changes."""

from __future__ import annotations

from pathlib import Path

import cv2


def main() -> int:
    for index in range(4):
        device = f"/dev/video{index}"
        if not Path(device).exists():
            print(f"{device}: missing")
            continue

        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        opened = capture.isOpened()
        read_ok = False
        width = height = 0
        if opened:
            for _ in range(5):
                read_ok, frame = capture.read()
                if read_ok and frame is not None:
                    height, width = frame.shape[:2]
                    break
        reported_fps = capture.get(cv2.CAP_PROP_FPS) if opened else 0.0
        capture.release()
        print(
            f"{device}: open_success={opened}, frame_read_success={read_ok}, "
            f"width={width}, height={height}, reported_fps={reported_fps:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
