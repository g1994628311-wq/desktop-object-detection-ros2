"""Combine measured PyTorch and TensorRT benchmark rows into the final report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_one(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one benchmark row in {path}, got {len(rows)}")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch", required=True)
    parser.add_argument("--tensorrt", required=True)
    parser.add_argument("--output-csv", default="results/jetson_yolo11n/benchmark.csv")
    parser.add_argument("--output-summary", default="results/jetson_yolo11n/benchmark_summary.md")
    args = parser.parse_args()
    rows = [read_one(Path(args.pytorch)), read_one(Path(args.tensorrt))]
    if rows[0]["backend"] != "PyTorch CUDA" or rows[1]["backend"] != "TensorRT FP16":
        raise ValueError("Inputs must be PyTorch CUDA followed by TensorRT FP16 benchmark rows")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pytorch_fps = float(rows[0]["avg_pipeline_fps"])
    tensorrt_fps = float(rows[1]["avg_pipeline_fps"])
    speedup = tensorrt_fps / pytorch_fps if pytorch_fps else float("inf")
    summary = Path(args.output_summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        "# Jetson YOLO11n benchmark\n\n"
        "Measured with identical camera settings; warm-up frames are excluded.\n\n"
        "| Backend | Frames | Avg pipeline FPS | Avg inference ms | >=5 FPS |\n"
        "| --- | ---: | ---: | ---: | --- |\n"
        f"| PyTorch CUDA | {rows[0]['frames']} | {pytorch_fps:.3f} | {float(rows[0]['avg_inference_ms']):.3f} | {'yes' if pytorch_fps >= 5 else 'no'} |\n"
        f"| TensorRT FP16 | {rows[1]['frames']} | {tensorrt_fps:.3f} | {float(rows[1]['avg_inference_ms']):.3f} | {'yes' if tensorrt_fps >= 5 else 'no'} |\n\n"
        f"TensorRT vs PyTorch pipeline speedup: **{speedup:.3f}x**.\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_csv} and {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
