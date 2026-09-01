# Final model comparison

正式训练前进行数据完整性校验，发现 1 个贴近图像边缘的 bbox 因归一化小数舍入产生 0.0005 越界，经原图视觉确认后裁剪至合法图像边界并重新归一化。该修复发生在任何模型训练和测试之前。

## Aggregate test metrics

| Model | P | R | mAP50 | mAP50-95 | Best epoch | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| yolo11n | 0.842738 | 0.725178 | 0.768520 | 0.419046 | 23 | 2025.345 |
| yolo11s | 0.877823 | 0.710737 | 0.794705 | 0.420278 | 18 | 3175.534 |

## Per-class test comparison

| Class | YOLO11n AP50 | YOLO11s AP50 | Difference | YOLO11n Recall | YOLO11s Recall | Recall difference |
|---|---:|---:|---:|---:|---:|---:|
| mouse | 0.848984 | 0.836233 | -0.012751 | 0.786494 | 0.801866 | +0.015372 |
| keyboard | 0.310810 | 0.393780 | +0.082970 | 0.206924 | 0.201220 | -0.005705 |
| laptop | 0.875237 | 0.912654 | +0.037416 | 0.952381 | 0.880952 | -0.071429 |
| cup | 0.889439 | 0.883636 | -0.005803 | 0.759036 | 0.774908 | +0.015872 |
| headphones | 0.918127 | 0.947225 | +0.029097 | 0.921053 | 0.894737 | -0.026316 |

## Interpretation

Accuracy winner: **yolo11s**, but only marginally. Relative to YOLO11n, YOLO11s gains +0.026186 mAP50 and only +0.001232 mAP50-95, while aggregate Recall falls by 0.014441. This is not a clear across-the-board improvement.

Keyboard is the dominant weakness for both models: Test Recall is 0.206924 for YOLO11n and 0.201220 for YOLO11s despite 164 instances. YOLO11s improves keyboard AP50 by 0.082970, but does not reduce its missed-keyboard rate. Headphones has the best AP50 for both models; mouse has the best AP50-95 for both.

At confidence 0.25, YOLO11n error analysis recorded 162 false negatives, 101 false positives, 21 class confusions, and 12 localization errors. YOLO11s recorded 147 false negatives, 177 false positives, 6 class confusions, and 10 localization errors. The main dataset-level issue is missed detections, especially keyboards; YOLO11s trades fewer false negatives/confusions for substantially more false positives.

Training losses continued to decrease while validation losses and metrics remained highly variable. Early stopping selected epoch 23 for YOLO11n and epoch 18 for YOLO11s, then stopped at epochs 43 and 38. This indicates validation instability from the small 33-image validation set and a clear risk of overfitting after the selected epochs. Test aggregate metrics being higher than validation metrics suggests a distribution/difficulty difference rather than evidence against overfitting.

Deployment candidate: **yolo11n**. It is 5.207 MB with 2.58M parameters versus 18.278 MB and 9.41M parameters for YOLO11s, while retaining essentially the same mAP50-95 and slightly higher Recall. Both should still be benchmarked on Jetson/TensorRT to measure the actual FPS/accuracy tradeoff.

Test results were not used for model selection, hyperparameter adjustment, confidence tuning, or dataset changes.
