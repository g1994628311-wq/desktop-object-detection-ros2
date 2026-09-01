# Jetson YOLO11n deployment

This package deploys the fixed five-class `YOLO11n` checkpoint only. It does not retrain, edit data, alter splits, or alter the checkpoint. TensorRT engines must be exported on the target Jetson, not on the Windows development PC.

## Transfer the checkpoint

From Windows PowerShell, create the target folder and transfer the model with SCP (replace both placeholders):

```powershell
ssh <jetson-user>@<jetson-ip> "mkdir -p ~/desktop-object-detection-ros2/models"
scp runs/detect/yolo11n_final/weights/best.pt <jetson-user>@<jetson-ip>:~/desktop-object-detection-ros2/models/yolo11n_final_best.pt
```

Alternatively, copy the same file by a removable drive or another managed file-transfer method to `~/desktop-object-detection-ros2/models/yolo11n_final_best.pt`. The checkpoint is ignored by Git and must be transferred separately after cloning the repository.

## On the Jetson

Run the following from the repository root. Save the first two reports so the deployment environment and available camera nodes are documented.

```bash
python3 scripts/check_jetson_env.py | tee results/jetson_yolo11n/environment.txt
python3 scripts/probe_cameras.py | tee results/jetson_yolo11n/camera_probe.txt
```

Select a device reported as both openable and frame-readable, then verify the PyTorch CUDA path before attempting TensorRT. This 330-frame run warms up for 30 frames; use the final 300 frames only for the benchmark calculation.

```bash
python3 scripts/jetson_realtime_detect.py \
  --model models/yolo11n_final_best.pt --source /dev/video0 \
  --imgsz 640 --conf 0.25 --iou 0.7 --max-frames 330 --warmup-frames 30 --display \
  --save-video --output results/jetson_yolo11n/pytorch_demo.mp4 \
  --jsonl results/jetson_yolo11n/pytorch_detections.jsonl \
  --csv results/jetson_yolo11n/pytorch_detections.csv \
  --benchmark-csv results/jetson_yolo11n/pytorch_benchmark.csv
```

The script prints and verifies the embedded class mapping exactly as `mouse`, `keyboard`, `laptop`, `cup`, `headphones`. Stop if it reports a mismatch. Do not export TensorRT until this PyTorch CUDA inference succeeds.

After confirming the PyTorch path and recording the JetPack, CUDA, TensorRT, PyTorch and Ultralytics versions from the environment report, export on the Jetson:

```bash
python3 scripts/export_jetson_tensorrt.py --model models/yolo11n_final_best.pt --imgsz 640
```

Run the identical camera conditions with the emitted `.engine` path:

```bash
python3 scripts/jetson_realtime_detect.py \
  --model models/yolo11n_final_best.engine --source /dev/video0 \
  --imgsz 640 --conf 0.25 --iou 0.7 --max-frames 330 --warmup-frames 30 --display \
  --save-video --output results/jetson_yolo11n/tensorrt_demo.mp4 \
  --jsonl results/jetson_yolo11n/tensorrt_detections.jsonl \
  --csv results/jetson_yolo11n/tensorrt_detections.csv \
  --benchmark-csv results/jetson_yolo11n/tensorrt_benchmark.csv
```

For each backend, benchmark at least 30 warm-up frames plus 300 measured frames under the same camera, scene, resolution, `imgsz=640`, confidence and IoU. Report total pipeline FPS (capture through drawing) and Ultralytics `preprocess`, `inference`, and `postprocess` timings. Keep the raw video and JSONL local; `.pt`, `.engine`, videos and `runs/` remain Git-ignored.

After both runs, combine the two measured rows into the required two-row CSV and summary (this does not rerun inference):

```bash
python3 scripts/summarize_jetson_benchmark.py \
  --pytorch results/jetson_yolo11n/pytorch_benchmark.csv \
  --tensorrt results/jetson_yolo11n/tensorrt_benchmark.csv
```

Perform qualitative checks for each individual class and multi-object scenes. Record keyboard behaviour separately for near, far, standalone, keyboard+laptop and keyboard+mouse scenes. No ROS2 integration is part of this stage.
