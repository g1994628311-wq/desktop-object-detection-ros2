# YOLO11n Baseline V2 总结

## 训练结论

YOLO11n 在 V2 上训练成功：请求 100 epoch，在 69 epoch 提前停止，最佳 epoch 为 49。正式选择仅使用 val；`best.pt` 固定后再评估 test。

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Val | 0.5298 | 0.5625 | 0.5826 | 0.3869 |
| Test | 0.2796 | 0.3596 | 0.3752 | 0.3081 |

首次正式 test 未完整保存逐类别指标；在模型、测试集和全部评估参数均固定后，对同一 `best.pt` 与 test split 重新执行一次指标导出。该运行未参与模型选择或超参数调整。

## 实验限制与适用范围

- 当前 V2 test 是严格 capture-group / scene-disjoint diagnostic test，仅有 13 images / 22 objects。
- keyboard 在 test 中 instances = 0，因此无法评估 keyboard test AP；mouse、cup、headphones 的 test instances 分别只有 3、4、3，逐类指标方差很大。
- 因此，V2 test 不应作为最终五类别综合性能报告。本实验主要说明：仅使用 P01 数据时，跨场景泛化明显不足。
- 最终性能测试将在 P02-P05 扩充不同物理实例后，重新建立具有充分五类覆盖且保持 capture-group 隔离的正式测试集。
- V1/V2 的 test split 不同，数值只能作为工程趋势，不能用于计算严格受控的提升百分比。

## 类别表现

Test 最强类别是 laptop（AP50-95 0.6531）；headphones 次之（0.5389）。mouse 与 cup 在 test 的 Precision、Recall 为 0，说明模型没有在该独立场景中稳定检出它们。test 没有 keyboard 实例，因此该类为 N/A，不能据此判断泛化能力。

## 主要问题

- test 中 mouse/cup 的漏检和误检集中，反映 train 与独立 floor/doll 场景存在明显背景、尺度和构图差异。
- val 中 cup 被误识别为 mouse/laptop，且出现 class confusion 与 false positive。
- V2 的 val 与 test 均显著低于 V1；在更严格 capture-group split 和重新构建数据集下，这提示模型对跨场景泛化仍然不足，而非可直接归因于单一数据变更。

## 过拟合与下一步

训练损失下降但 val 指标早期达到平台，存在泛化不足/轻度过拟合迹象。当前首要瓶颈是独立采集场景和类别覆盖不足，其次才是 YOLO11n 容量。建议优先继续采集 P02-P05，重点覆盖 mouse、cup、laptop、headphones 的尺度、遮挡、边缘和不同背景；补齐更多 session 后再比较模型容量或推进 Jetson 部署。
