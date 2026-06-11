# src/ 文件说明

## 核心模块
| 文件 | 功能 |
|------|------|
| `config.py` | 全局配置参数 |
| `data_loader.py` | CFD 数据加载、VOF 过滤、归一化 |
| `model.py` | MLP 网络定义 |
| `physics.py` | RANS 残差计算 |
| `inference.py` | R²/RMSE 评估 |

## 实验脚本（按顺序）
| 文件 | 内容 |
|------|------|
| `01_arch_sweep.py` | 架构扫描（宽度/深度） |
| `02_full_train.py` | 最佳模型全量训练 |
| `03_ablation.py` | νt + 物理项消融 |
| `04_baseline_train.py` | 基准 MLP 训练 |
| `05_loss_compare.py` | 损失函数 + baseline 对比 |
| `06_sweep_lambda.py` | λ_phys × λ_bc 权重扫描 |
| `07_optimize_current.py` | 当前优化（VOF软加权/混合长度） |
| `08_plot_figures.py` | 论文配图生成 |

## 归档
`_archive/` — 20 个废弃/中间版本脚本
