# Windows USB-camera YOLO11n capability validation

Run this standalone validation on the Windows PC with the final fixed checkpoint. It is not a Jetson, TensorRT or ROS2 test, and it does not alter training data or the model.

```powershell
.\.venv\Scripts\python.exe scripts\usb_camera_model_test.py
```

The script probes camera indexes 0–10 using DirectShow, Media Foundation and the OpenCV default backend. If more than one camera is available, specify the preferred index and backend:

```powershell
.\.venv\Scripts\python.exe scripts\usb_camera_model_test.py --camera 1 --backend dshow
```

For a full recorded test:

```powershell
.\.venv\Scripts\python.exe scripts\usb_camera_model_test.py `
  --camera 1 `
  --imgsz 640 `
  --conf 0.25 `
  --save-video
```

Use `Q` to quit, `S` to save an annotated screenshot and `SPACE` to pause or resume. The first 30 frames are warm-up; run at least 330 total frames for a 300-frame benchmark. The script reports rolling pipeline FPS (capture through drawing) and average inference time, and writes local outputs under `results/usb_camera_test/`. Keyboard observations must be recorded as observed; do not change the model to improve them.
