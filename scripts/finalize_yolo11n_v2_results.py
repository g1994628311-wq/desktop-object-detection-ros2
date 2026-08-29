"""Write tracked YOLO11n V2 experiment reports from fixed exported metrics."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/yolo11n_baseline_v2'
CLASSES = ('mouse', 'keyboard', 'laptop', 'cup', 'headphones')
METRICS = {
 'val': (0.5297613233, 0.5624984052, 0.5826123397, 0.3868758146),
 'test': (0.2796056247, 0.3595916292, 0.3751554516, 0.3081176027),
}
# p, r, AP50, AP50-95, instances.  None means no GT instance in that split.
PER_CLASS = {
 'val': [(0,0,0,0,2),(.1190452930,1,.5616666667,.4363958333,2),(None,None,None,None,0),(1,.6376639423,.8556398349,.5256383556,14),(1,.6123296786,.9131428571,.5854690696,11)],
 'test': [(0,0,.0176315789,.0105789474,3),(None,None,None,None,0),(.4825725739,.8333333333,.8668642534,.6530762688,12),(0,0,.0493402597,.0299223377,4),(.6358499251,.6050331835,.5667857143,.5388928571,3)],
}
V1 = {
 'val': (.8942314363,.8857099610,.95575,.8578528404), 'test': (.8345265985,.7812286238,.8252335773,.5977041399),
}
def n(v): return 'N/A' if v is None else f'{v:.6f}'
def main():
 OUT.mkdir(parents=True, exist_ok=True)
 with (OUT/'metrics_summary.csv').open('w', newline='', encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['split','precision','recall','mAP50','mAP50_95']); [w.writerow([s,*METRICS[s]]) for s in ('val','test')]
 with (OUT/'per_class_metrics.csv').open('w', newline='', encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['split','class_id','class_name','precision','recall','AP50','AP50_95','instances'])
  for s in ('val','test'):
   for i,name in enumerate(CLASSES): w.writerow([s,i,name,*PER_CLASS[s][i]])
 with (OUT/'v1_v2_comparison.csv').open('w', newline='', encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['version','split','precision','recall','mAP50','mAP50_95'])
  for version,data in [('v1',V1),('v2',METRICS)]:
   for s in ('val','test'): w.writerow([version,s,*data[s]])
 (OUT/'v1_v2_comparison.md').write_text('''# YOLO11n Baseline V1 / V2 比较\n\n| Version | Split | Precision | Recall | mAP50 | mAP50-95 |\n|---|---|---:|---:|---:|---:|\n| V1 | val | 0.8942 | 0.8857 | 0.9558 | 0.8579 |\n| V1 | test | 0.8345 | 0.7812 | 0.8252 | 0.5977 |\n| V2 | val | 0.5298 | 0.5625 | 0.5826 | 0.3869 |\n| V2 | test | 0.2796 | 0.3596 | 0.3752 | 0.3081 |\n\nV1 与 V2 的 test split、样本数量和 capture-group 定义均不同；以上只能作为工程趋势参考，不能作为同一测试集上的严格对照消融。\n''', encoding='utf-8')
 (OUT/'training_environment.txt').write_text('''Python: 3.12.13\nPyTorch: 2.13.0+cu130\nCUDA: 13.0\nUltralytics: 8.4.131\nGPU: NVIDIA GeForce RTX 4060 Laptop GPU\nmodel: yolo11n.pt\nimgsz: 640\nbatch: 16\nepochs_requested: 100\nepochs_completed: 69\nbest_epoch: 49\nearly_stopping: true\ntraining_time_seconds: 137.764\nrun_dir: runs/detect/yolo11n_baseline_v2\nbest.pt: runs/detect/yolo11n_baseline_v2/weights/best.pt\nlast.pt: runs/detect/yolo11n_baseline_v2/weights/last.pt\ntest_per_class_export_note: 首次正式 test 未完整保存逐类别指标；固定 best.pt、测试集与全部评估参数后，对同一 test split 重运行一次指标导出，未参与模型选择或超参数调整。\n''', encoding='utf-8')
 (OUT/'summary.md').write_text('''# YOLO11n Baseline V2 总结\n\n## 训练结论\n\nYOLO11n 在 V2 上训练成功：请求 100 epoch，在 69 epoch 提前停止，最佳 epoch 为 49。正式选择仅使用 val；`best.pt` 固定后再评估 test。\n\n| Split | Precision | Recall | mAP50 | mAP50-95 |\n|---|---:|---:|---:|---:|\n| Val | 0.5298 | 0.5625 | 0.5826 | 0.3869 |\n| Test | 0.2796 | 0.3596 | 0.3752 | 0.3081 |\n\n首次正式 test 未完整保存逐类别指标；在模型、测试集和全部评估参数均固定后，对同一 `best.pt` 与 test split 重新执行一次指标导出。该运行未参与模型选择或超参数调整。\n\n## 类别表现\n\nTest 最强类别是 laptop（AP50-95 0.6531）；headphones 次之（0.5389）。mouse 与 cup 在 test 的 Precision、Recall 为 0，说明模型没有在该独立场景中稳定检出它们。test 没有 keyboard 实例，因此该类为 N/A，不能据此判断泛化能力。\n\n## 主要问题\n\n- test 中 mouse/cup 的漏检和误检集中，反映 train 与独立 floor/doll 场景存在明显背景、尺度和构图差异。\n- val 中 cup 被误识别为 mouse/laptop，且出现 class confusion 与 false positive。\n- V2 的 val 与 test 均显著低于 V1；在更严格 capture-group split 和重新构建数据集下，这提示模型对跨场景泛化仍然不足，而非可直接归因于单一数据变更。\n\n## 过拟合与下一步\n\n训练损失下降但 val 指标早期达到平台，存在泛化不足/轻度过拟合迹象。当前首要瓶颈是独立采集场景和类别覆盖不足，其次才是 YOLO11n 容量。建议优先继续采集 P02-P05，重点覆盖 mouse、cup、laptop、headphones 的尺度、遮挡、边缘和不同背景；补齐更多 session 后再比较模型容量或推进 Jetson 部署。\n''', encoding='utf-8')
if __name__ == '__main__': main()
