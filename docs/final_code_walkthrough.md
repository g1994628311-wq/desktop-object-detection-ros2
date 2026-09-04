# 最终代码讲解总览

## 1. Training Code

### 模块 1：迁移学习与训练

关键代码：

```python
model = YOLO(weight, task="detect")
model.train(data=str(DATA), epochs=100, imgsz=640, batch=16,
            device=0, patience=20, pretrained=True)
```

【作用】加载 COCO Pretrained Weights（COCO 预训练权重），对五类桌面物体 Fine-tuning（微调）。`train()` 已封装 image→augmentation→forward→与 Ground Truth（人工真实标注）比较→box/cls/dfl loss→backward→optimizer update。box_loss 是框位置误差，cls_loss 是类别误差，dfl_loss（Distribution Focal Loss）帮助精细定位。

【为什么需要】小型自建数据集从零训练通常不足；预训练网络已有边缘、纹理和形状知识。`epochs=100` 是上限，`patience=20` 是 Early Stopping（早停），不代表一定训练满 100 轮。

【输入/输出】输入为 `dataset.yaml` 和 `yolo11n.pt`/`yolo11s.pt`；输出 `runs/`、训练曲线、`last.pt` 和 Validation 指标最佳的 `best.pt`。

【老师可能问】为什么不用 last.pt？【建议回答】last.pt 是最后一轮，best.pt 是 Validation 最好一轮，因此部署使用 best.pt。

### 模块 2：Validation、Test 与逐类指标

关键代码：

```python
val = fixed.val(data=str(DATA), split="val", imgsz=640, batch=16)
test = fixed.val(data=str(DATA), split="test", imgsz=640, batch=16)
```

【作用】Validation 用于模型选择和早停；固定 best.pt 后的 Test 才报告最终泛化。Precision（精确率）表示预测为目标中有多少正确，Recall（召回率）表示真实目标找回多少；mAP50 使用 IoU（Intersection over Union，交并比）≥0.5，mAP50-95 在 0.50–0.95 平均、更严格。

【为什么需要】Test 不参与训练或阈值/超参数选择，避免乐观偏差。逐类 mouse、keyboard、laptop、cup、headphones 指标能发现整体 mAP 隐藏的弱类。

【老师可能问】YOLO11s 参数更多为何不一定更好？【建议回答】更大容量需要足够数据；本项目更关注精度、速度和部署代价的平衡，n 是 nano、s 是 small。

## 2. Dataset Validation Code

### 模块 1：训练前数据质量门

关键代码：

```python
images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS}
labels = {p.stem: p for p in label_dir.glob("*.txt")}
```

【作用】逐 split 检查 missing labels（图片无 txt）和 orphan labels（txt 无图片），并统计每类 object instances。

【为什么需要】YOLO 标签每行是 `class_id x_center y_center width height`，坐标 normalized 到 0–1，不是 `x1 y1 x2 y2`。ID 必须是 0–4；`center ± size/2` 越界（即使 1.0005）也属于规范错误。

### 模块 2：泄露检查

关键代码：

```python
overlap = split_stems[a] & split_stems[b]
if overlap: errors.append(...)
```

【作用】检查文件名、精确图片 hash，以及 Pxx_Sxx session 是否跨 Train/Val/Test。

【为什么需要】同 session 常共享背景、光照、相机位置与物体。随机拆开会形成 Data Leakage（数据泄露），使 Test 虚高。流程必须是 Dataset Validation → 无错误 → Training。

## 3. Inference Code

### 模块 1：USB 摄像头与 Single Open

关键代码：

```python
capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ok, frame = capture.read()
result = model.predict(frame, imgsz=640, conf=0.25, iou=0.45, device=0)[0]
```

【作用】USB 2.0 Camera 使用 index 1、DirectShow（Windows DSHOW）。Single Open 从打开到退出只保留一个 handle，避免 probe/release/reopen 的二次打开失败。

【为什么需要】`device=0` 使用 RTX 4060。Inference 不会更新权重；Confidence 0.25 以下不显示，过低增加 False Positive、过高增加 False Negative。IoU 参数服务 NMS（Non-Maximum Suppression，非极大值抑制），从重叠候选框中保留合理框。

### 模块 2：结果、FPS 与资源释放

关键代码：

```python
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
capture.release(); writer.release(); cv2.destroyAllWindows()
```

【作用】`boxes.xyxy/conf/cls` 是推理输出（像素 xyxy，区别于训练标签）；OpenCV 绘制 bbox、class、confidence。CSV 每个 detection 一行，JSONL 每帧一个 JSON，视频保存已绘制画面。

【为什么需要】前 30 帧 Warm-up 排除 CUDA 初始开销；300 帧平均 Pipeline FPS 包含 capture、preprocess、inference、postprocess、drawing，比单帧和纯 inference FPS 更贴近应用。最后释放资源防止摄像头被占用。

## 整体流程

```text
Raw Images → Manual Annotation → YOLO Labels → Dataset Validation → Train / Val / Test
COCO Pretrained YOLO11 → Fine-tuning → Train → Validation → Early Stopping → best.pt
best.pt → Test Dataset → Precision / Recall → mAP50 / mAP50-95 → Per-class Metrics
USB Camera → OpenCV → Frame → YOLO best.pt → Prediction → NMS → BBox + Class + Confidence → Realtime Display → Video / CSV / JSONL
```
