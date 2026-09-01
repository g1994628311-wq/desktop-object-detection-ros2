# yolo11n final fine-tuning

正式训练前进行数据完整性校验，发现 1 个贴近图像边缘的 bbox 因归一化小数舍入产生 0.0005 越界，经原图视觉确认后裁剪至合法图像边界并重新归一化。该修复发生在任何模型训练和测试之前。

- Pretrained weights: `yolo11n.pt`
- Run: `D:\codes\desktop-object-detection-ros2\runs\detect\yolo11n_final`
- Best epoch: 23; completed: 43/100; training seconds: 2025.345
- Val: P=0.576337, R=0.409073, mAP50=0.485836, mAP50-95=0.371489
- Test: P=0.842738, R=0.725178, mAP50=0.768520, mAP50-95=0.419046

| Class | P | R | AP50 | AP50-95 | Instances |
|---|---:|---:|---:|---:|---:|
| mouse | 0.9311629783927285 | 0.7864944521678636 | 0.8489843506237629 | 0.5850957411281973 | 86 |
| keyboard | 0.8945534949852628 | 0.20692410662999922 | 0.3108100222646353 | 0.16398546285785204 | 164 |
| laptop | 0.48754516106983764 | 0.9523809523809523 | 0.8752374302152838 | 0.46073661272391 | 42 |
| cup | 0.9681519880271319 | 0.7590361445783133 | 0.889438721489539 | 0.47328131598420164 | 83 |
| headphones | 0.9322751993443864 | 0.9210526315789473 | 0.918127272727273 | 0.41212882396816014 | 38 |

Error counts at conf=0.25: {'false_negative': 162, 'class_confusion': 21, 'low_confidence_correct': 25, 'false_positive': 101, 'localization_error': 12}. Test was used once after fixing best.pt and did not participate in selection or tuning.
