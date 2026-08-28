# Dataset Summary

Generated from validated canonical annotations on 2026-08-28.

| Metric | Count |
|---|---:|
| Images / label files | 87 / 87 |
| Object instances | 168 |
| Negative images | 6 |

| Class | Images | Instances |
|---|---:|---:|
| mouse | 41 | 42 |
| keyboard | 18 | 18 |
| laptop | 32 | 32 |
| cup | 42 | 45 |
| headphones | 31 | 31 |

Only P01 contains images: S01 21, S02 20, S03 23, S04 17, S05 6. P02-P05 are empty and were not fabricated.

The minimum split group is a complete `PXX/SXX` session. With only five non-empty groups, exact 70/15/15 ratios and negative coverage in all splits are impossible. The deterministic seed-42 policy prioritizes zero leakage, five-class coverage in every positive split, a challenging S04 test, and S05 negatives in training.

| Split | Sessions | Images | Objects | Negatives | mouse | keyboard | laptop | cup | headphones |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | S01, S02, S05 | 47 | 63 | 6 | 15 | 3 | 14 | 21 | 10 |
| val | S03 | 23 | 47 | 0 | 12 | 7 | 6 | 11 | 11 |
| test | S04 | 17 | 58 | 0 | 15 | 8 | 12 | 13 | 10 |

Validation: **0 errors, 0 warnings**. Manual review remaining: **0**.
