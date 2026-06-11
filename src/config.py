"""
config.py — 全局配置（唯一总定义）
所有训练/消融/扫描/inference 脚本均从此 import，改参数只改此处。
"""
import os
import torch

# ============================================================
# 0. 路径
# ============================================================
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_BOTTOM = os.path.join(ROOT_DIR, "data", "cfd", "bottom")   # 底吹 CFD CSV (q*-1.csv)
DATA_SIDE   = os.path.join(ROOT_DIR, "data", "cfd", "side")     # 侧吹 CFD CSV
DATA_BC     = os.path.join(ROOT_DIR, "data", "bc")              # BC CSV (wall only, DPM 无连续相 inlet)
MODEL_DIR   = os.path.join(ROOT_DIR, "models")
RESULT_DIR  = os.path.join(ROOT_DIR, "results")
FIG_DIR     = os.path.join(RESULT_DIR, "figures")
TBL_DIR     = os.path.join(RESULT_DIR, "tables")
LOG_DIR     = os.path.join(RESULT_DIR, "logs")

for d in [MODEL_DIR, FIG_DIR, TBL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 1. 设备 & 精度
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

# ============================================================
# 2. 网络结构
# ============================================================
# 输入: (x, y, z, Q, m)  — m: injection position (0=center bottom, 1=eccentric bottom)
# 输出: (u, v, w, p)
N_INPUT  = 5                # (x, y, z, Q, m), m=0 中吹
N_OUTPUT = 4                # (u, v, w, p)
HIDDEN   = [128] * 6         # 6 层 × 128
LAYERS   = [N_INPUT] + HIDDEN + [N_OUTPUT]
ACTIVATION = "swish"
FOURIER_L     = 0             # Fourier 特征频率数 (0=关闭)
FOURIER_SIGMA = 1.0           # 频率尺度

# ============================================================
# 3. 训练超参数
# ============================================================
EPOCHS       = 2000
BATCH_DATA   = 8192           # 数据点 mini-batch
BATCH_PHYS   = 8192           # 物理残差点 mini-batch
LR           = 1e-3
LR_DECAY     = 0.96           # 每 500 epoch 衰减系数 (cosine 下备用)
LR_STEP      = 500
SEED         = 42
MAX_EPOCHS   = 20000          # 训练硬上限（early stopping 通常先触发）
PATIENCE     = 2000            # early stopping patience (epoch 数)
PHASE1       = 500             # Phase 1 epoch 数（纯数据）

# ============================================================
# 4. 物理参数 — 混合长度模型
# ============================================================
NU        = 1e-6              # 分子运动黏度 (归一化后)
L_M       = 1e-2              # 混合长度 (m)，钢包特征尺度 ~0.01 m
# ν_eff = ν + ν_t = ν + l_m² · |S|
# |S| = sqrt(2 S_ij S_ij)，S_ij 从 autograd 实时计算

# ---- 消融用：旧版常数 νt ----
NU_T_CONST = 5e-4             # 原稿使用的恒定涡黏度
# ν_eff_const = NU + NU_T_CONST

# ============================================================
# 4.5 计算域 & VOF 过滤
# ============================================================
Z_STEEL_TOP      = 1.85       # 钢-渣界面 z 坐标 (m)
ALPHA_STEEL_MIN  = 0.01       # data loss 权重下限（α_s 低于此值的 cell 不参与训练）
# 软加权: w_data = α_steel, clamped to [ALPHA_STEEL_MIN, 1.0]
# PDE 域: w_pde = w_data * w_spatial(z)

# ============================================================
# 4.6 空间衰减（顶面过渡带，消除 hard δ 任意性）
# ============================================================
SPATIAL_DECAY_TYPE = "cosine"  # "cosine" | "sigmoid" | "hard"
# cosine:  w(z) = 0.5 * (1 + cos(π * min(Δz / T, 1)))   Δz = z_top - z
# sigmoid: w(z) = σ(Δz / T)
# hard:    w(z) = 1 if z < z_top - δ else 0
SPATIAL_DECAY_T    = 0.20     # T (m)，衰减尺度。cosine: z_top - T 处 w≈0.5; sigmoid: 过渡区宽度

# ============================================================
# 5. 损失权重（默认值，sweep_weights.py 会覆盖）
# ============================================================
LAMBDA_DATA  = 1.0
LAMBDA_PHYS  = 0.1
LAMBDA_BC    = 1.0

# ============================================================
# 6. 训练/测试工况
# ============================================================
TRAIN_Q        = [40, 60, 80]      # 底吹训练工况 (NL/min)
TEST_EXTRAP_Q  = [100, 120]        # 外推测试工况（全部 cell，Q 值未在训练中出现）
HOLD_OUT       = 0.2               # 训练工况空间 holdout 比例
# 内插测试: 从 TRAIN_Q 每个 Q 独立随机 hold out HOLD_OUT 比例 cell
# 外推测试: TEST_EXTRAP_Q 全部 cell
# 严禁 TRAIN_Q ∩ TEST_EXTRAP_Q = ∅  (config 级断言)
_overlap = set(TRAIN_Q) & set(TEST_EXTRAP_Q)
if _overlap:
    raise ValueError(f"Q values in both TRAIN_Q and TEST_EXTRAP_Q: {_overlap}")
SIDE_Q  = []                        # 偏心底吹工况 (放入后填充)

# ============================================================
# 7. 归一化参数（运行时由 data_loader 填入）
# ============================================================
NORM = {
    "x_min": 0.0, "x_max": 1.0,
    "y_min": 0.0, "y_max": 1.0,
    "z_min": 0.0, "z_max": 1.0,
    "Q_max": 120.0,
    "U_ref":  1.0,
}

# ============================================================
# 8. 不确定性量化
# ============================================================
MC_DROPOUT_P   = 0.1           # MC Dropout 概率
MC_SAMPLES     = 50            # 推理采样次数

# ============================================================
# 9. 权重 & 超参数扫描范围
# ============================================================
SWEEP_PHYS = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
SWEEP_BC   = [0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

# ---- νt 敏感性 ----
L_M_SWEEP = [0.005, 0.01, 0.02, 0.05]           # 混合长度 (m)

# ---- VOF 阈值敏感性 ----
ALPHA_THRESH_SWEEP = [0.3, 0.5, 0.7, 0.9]        # PDE 域 α_s 阈值

# ---- 空间衰减消融 ----
SPATIAL_DECAY_SWEEP = ["cosine", "sigmoid", "hard"]

# ---- 多 seed ----
SEEDS = [42, 123, 456]                             # 稳定性评估

# ============================================================
# 10. 渣眼检测
# ============================================================
SLAG_EYE_P_THRESH = 0.0          # (已弃用) 旧 p 推断法
SLAG_EYE_ALPHA_SLAG = 0.5        # 渣眼判据：顶面 α_slag < 0.5 → 暴露 (VOF 直接测量)

# ============================================================
# 11. 日志 & 保存
# ============================================================
LOG_INTERVAL  = 200            # 每 N epoch 打印一次 loss
SAVE_BEST     = True           # 保存在验证集上最优的模型
CKPT_NAME     = "pinn_best.pt"
