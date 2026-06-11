"""
data_loader.py — CFD 数据加载 & 归一化 & VOF 软加权
读取 q*-1.csv (12 列: 4 VOF → 合并 argon+air 为 gas)，构建 (x,y,z,Q,m) → (u,v,w,p) 数据集。

关键设计：
- gas_vof = argon-vof + air-vof (Fluent 残量合并)
- w_data = clamp(α_steel, 0.01, 1)                      VOF 软加权
- w_pde  = w_data × w_spatial(z)                         PDE 空间衰减
- Wall BC: α_steel>0.5 精准过滤 (CSV 自带 VOF)
- DPM 气泡注入 → 连续相无 inlet，底部即 wall no-slip
- 内插测试: TRAIN_Q 每 Q 独立 80/20 随机空间 holdout   (防测试集污染)
- 外推测试: TEST_EXTRAP_Q 全部 cell
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import torch
from config import (
    ROOT_DIR, DATA_BOTTOM, DATA_SIDE, DATA_BC,
    DEVICE, NORM, TRAIN_Q, TEST_EXTRAP_Q, HOLD_OUT, SEED,
    Z_STEEL_TOP, ALPHA_STEEL_MIN,
    SPATIAL_DECAY_TYPE, SPATIAL_DECAY_T,
)


# ============================================================
# Fluent 列名映射 — 处理不同命名约定
# ============================================================
_FLUENT_COLUMN_ALIASES = {
    "x-coordinate":    ["x-coordinate", "x coordinate", "x_coordinate", "x"],
    "y-coordinate":    ["y-coordinate", "y coordinate", "y_coordinate", "y"],
    "z-coordinate":    ["z-coordinate", "z coordinate", "z_coordinate", "z"],
    "x-velocity":      ["x-velocity", "x velocity", "x_velocity", "u"],
    "y-velocity":      ["y-velocity", "y velocity", "y_velocity", "v"],
    "z-velocity":      ["z-velocity", "z velocity", "z_velocity", "w"],
    "pressure":        ["pressure", "static pressure", "p"],
    "cellnumber":      ["cellnumber", "cell number", "cell-number", "cell_id", "node id", "id"],
    "alpha_steel":     ["alpha_steel", "alpha steel", "volume-fraction-steel",
                        "volume fraction of steel", "vf-steel", "phase-1-volume-fraction",
                        "steel-vof", "steel vof"],
    "alpha_slag":      ["alpha_slag", "alpha slag", "volume-fraction-slag",
                        "volume fraction of slag", "vf-slag", "phase-2-volume-fraction",
                        "slag-vof", "slag vof"],
    "alpha_argon":     ["alpha_argon", "argon-vof", "argon vof", "argon"],
    "alpha_air":       ["alpha_air", "alpha air", "volume-fraction-air",
                        "volume fraction of air", "vf-air", "phase-3-volume-fraction",
                        "air-vof", "air vof"],
}


def _resolve_column(df: pd.DataFrame, canonical: str):
    """按别名表查找 DataFrame 中对应的列名。找不到返回 None。"""
    for alias in _FLUENT_COLUMN_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


def _get_col(df: pd.DataFrame, canonical: str, default=None):
    """安全取列，找不到时返回 default 或报错。"""
    col = _resolve_column(df, canonical)
    if col is not None:
        return df[col].values.astype(np.float64)
    if default is not None:
        return default
    raise KeyError(f"Column '{canonical}' not found. Available: {list(df.columns)}")


# ============================================================
# 空间衰减函数
# ============================================================

def _spatial_decay(z: np.ndarray, z_top: float, T: float,
                   decay_type: str = "cosine") -> np.ndarray:
    """PDE loss 空间权重 w_spatial(z) ∈ [0, 1]。z 越大越接近顶面 → w → 0。"""
    dz = z_top - z
    dz = np.maximum(dz, 0.0)

    if decay_type == "cosine":
        ratio = np.minimum(dz / T, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * (1.0 - ratio)))

    elif decay_type == "sigmoid":
        k = 4.0 / T
        z0 = T / 2.0
        return 1.0 / (1.0 + np.exp(-k * (dz - z0)))

    elif decay_type == "hard":
        return (dz > T).astype(np.float64)

    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


# ============================================================
# CSV 解析
# ============================================================

def _parse_csv(csv_path: str, mode: int):
    """读取单个 q*-1.csv (12 列: 4 VOF)，合并 argon+air → gas_vof。

    返回 X_raw, Y_raw, w_data, w_pde, Q_val
    """
    # 提取 Q
    m = re.search(r"[qQ](\d+)", os.path.basename(csv_path))
    if m is None:
        raise ValueError(f"无法从文件名提取 Q: {csv_path}")
    Q_val = float(m.group(1))

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 坐标 & 流场
    x = _get_col(df, "x-coordinate")
    y = _get_col(df, "y-coordinate")
    z = _get_col(df, "z-coordinate")
    u = _get_col(df, "x-velocity")
    v = _get_col(df, "y-velocity")
    w = _get_col(df, "z-velocity")
    p = _get_col(df, "pressure")
    c = _get_col(df, "cellnumber", default=np.arange(len(x), dtype=np.float64))

    # VOF: 4 列 → 合并 argon+air 为 gas，保留 steel/slag
    alpha_steel = _get_col(df, "alpha_steel", default=np.ones(len(x)))
    alpha_slag  = _get_col(df, "alpha_slag",  default=np.zeros(len(x)))
    alpha_argon = _get_col(df, "alpha_argon", default=np.zeros(len(x)))
    alpha_air   = _get_col(df, "alpha_air",   default=np.zeros(len(x)))
    # gas = argon + air (Fluent 残量合并，少数 cell 有 air 残量)
    alpha_gas = alpha_argon + alpha_air

    # VOF 一致性检查：steel + slag + gas 应 ≡ 1（容忍 numerical noise）
    vof_sum = alpha_steel + alpha_slag + alpha_gas
    if np.any(np.abs(vof_sum - 1.0) > 0.01):
        n_bad = int(np.sum(np.abs(vof_sum - 1.0) > 0.01))
        print(f"  [WARN] {csv_path}: {n_bad}/{len(x)} cells VOF sum != 1")

    # 逐点权重
    # w_data = α_steel，clamp 到 [ALPHA_STEEL_MIN, 1]，低于阈值 → 0
    w_data_raw = np.where(alpha_steel >= ALPHA_STEEL_MIN, alpha_steel, 0.0)
    w_data_raw = np.clip(w_data_raw, 0.0, 1.0)

    # w_pde = w_data × w_spatial(z)
    w_spatial = _spatial_decay(z, Z_STEEL_TOP, SPATIAL_DECAY_T, SPATIAL_DECAY_TYPE)
    w_pde_raw = w_data_raw * w_spatial

    # 组装
    n = len(x)
    X = np.column_stack([
        x, y, z,
        np.full(n, Q_val, dtype=np.float64),
        np.full(n, mode, dtype=np.float64),
        c.astype(np.float64),
    ])
    Y = np.column_stack([u, v, w, p])

    return X, Y, w_data_raw.astype(np.float32), w_pde_raw.astype(np.float32), Q_val


# ============================================================
# 全量加载
# ============================================================

def load_all_data(bottom_globs=None, side_globs=None):
    """加载全部 CFD 场数据。

    Returns: X_raw(5), Y_raw(4), w_data, w_pde, norm, bc, Q_list, stats
    """
    if bottom_globs is None:
        bottom_globs = [os.path.join(DATA_BOTTOM, "q*-1.csv")]
    if side_globs is None:
        side_globs = [os.path.join(DATA_SIDE, "*ecc*.csv"), os.path.join(DATA_SIDE, "case_side_*.csv")]

    X_list, Y_list, Wd_list, Wp_list = [], [], [], []
    Q_set = set()

    for pattern in bottom_globs:
        for f in sorted(glob.glob(pattern)):
            X, Y, wd, wp, Q_val = _parse_csv(f, mode=0)
            X_list.append(X); Y_list.append(Y)
            Wd_list.append(wd); Wp_list.append(wp)
            Q_set.add(Q_val)

    for pattern in side_globs:
        for f in sorted(glob.glob(pattern)):
            X, Y, wd, wp, Q_val = _parse_csv(f, mode=1)
            X_list.append(X); Y_list.append(Y)
            Wd_list.append(wd); Wp_list.append(wp)
            Q_set.add(Q_val)

    if not X_list:
        raise FileNotFoundError(f"未找到 CFD CSV。底吹: {bottom_globs}, 侧吹: {side_globs}")

    X_all  = np.concatenate(X_list, axis=0)
    Y_all  = np.concatenate(Y_list, axis=0)
    WD_all = np.concatenate(Wd_list, axis=0)
    WP_all = np.concatenate(Wp_list, axis=0)

    # VOF 统计（仅用 w_data > 0 的 cell）
    mask_steel = WD_all > 0.0
    n_total = len(WD_all)
    n_steel = int(mask_steel.sum())
    stats = {
        "n_total":     n_total,
        "n_steel":     n_steel,
        "n_slag_air":  n_total - n_steel,
        "steel_frac":  n_steel / n_total if n_total > 0 else 0.0,
    }

    # 归一化参数 — 仅钢水区
    if mask_steel.sum() == 0:
        mask_steel = np.ones(n_total, dtype=bool)

    xs, ys, zs = X_all[mask_steel, 0], X_all[mask_steel, 1], X_all[mask_steel, 2]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()
    Lx = max(x_max - x_min, 1e-8)
    Ly = max(y_max - y_min, 1e-8)
    Lz = max(z_max - z_min, 1e-8)

    Q_all = X_all[:, 3]
    Q_max = Q_all.max() if Q_all.max() > 0 else 120.0

    Y_steel = Y_all[mask_steel]
    # z-score normalization per channel (matches old working code)
    u_mean, u_std = Y_steel[:, 0].mean(), Y_steel[:, 0].std()
    v_mean, v_std = Y_steel[:, 1].mean(), Y_steel[:, 1].std()
    w_mean, w_std = Y_steel[:, 2].mean(), Y_steel[:, 2].std()
    p_mean, p_std = Y_steel[:, 3].mean(), Y_steel[:, 3].std()
    # Ensure no zero std
    u_std = max(float(u_std), 1e-8); v_std = max(float(v_std), 1e-8)
    w_std = max(float(w_std), 1e-8); p_std = max(float(p_std), 1e-8)

    norm = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "Lx": Lx, "Ly": Ly, "Lz": Lz,
        "Q_max": Q_max,
        "u_mean": u_mean, "u_std": u_std,
        "v_mean": v_mean, "v_std": v_std,
        "w_mean": w_mean, "w_std": w_std,
        "p_mean": p_mean, "p_std": p_std,
    }

    # BC — 仅 inlet + wall
    bc = _load_boundary_conditions(norm, sorted(Q_set))

    # 剔除 cellnumber
    X_out = X_all[:, :5]
    return X_out, Y_all, WD_all, WP_all, norm, bc, sorted(Q_set), stats


# ============================================================
# BC 加载
# ============================================================

def _load_boundary_conditions(norm: dict, Q_list: list):
    """加载 wall BC（DPM 无连续相 inlet），按 α_steel > 0.5 过滤。

    wall CSV 为 12 列含 VOF，直接使用其 α_steel 列过滤，
    无需与场数据坐标匹配 —— Fluent 导出的 face value 已自带 VOF。
    """
    bc = {}

    # DPM 气泡注入 → 连续相底部无 inlet，只有 wall (no-slip)
    bc["inlet"] = None

    # wall: 按 Q 值独立加载，按 α_steel > 0.5 过滤
    wall_by_q = {}
    for q in Q_list:
        Q_str = f"Q{int(q)}" if int(q) >= 100 else f"q{int(q)}"
        # 尝试多种命名
        candidates = [
            os.path.join(DATA_BC, f"q{int(q)}_wall.csv"),
            os.path.join(DATA_BC, f"Q{int(q)}_wall.csv"),
        ]
        f_w = None
        for cand in candidates:
            if os.path.isfile(cand):
                f_w = cand
                break
        if f_w is None:
            continue

        df = pd.read_csv(f_w)
        df.columns = [str(c).strip().lower() for c in df.columns]

        x = _get_col(df, "x-coordinate")
        y = _get_col(df, "y-coordinate")
        z = _get_col(df, "z-coordinate")

        # 使用 wall CSV 自带的 α_steel 过滤（face value，非 cell center）
        alpha_steel = _get_col(df, "alpha_steel", default=np.ones(len(x)))
        mask = alpha_steel >= ALPHA_STEEL_MIN
        x, y, z = x[mask], y[mask], z[mask]
        n_pts = len(x)
        if n_pts == 0:
            continue

        Q_arr = np.full((n_pts, 1), float(q))
        m_arr = np.zeros((n_pts, 1))
        x_raw = np.hstack([np.column_stack([x, y, z]), Q_arr, m_arr])
        wall_by_q[q] = torch.tensor(
            normalize_x(x_raw, norm), dtype=torch.float32, device=DEVICE
        )

    if wall_by_q:
        # 所有 Q 的 wall 坐标相同（同一 mesh），只取第一个 Q 的即可
        first_q = sorted(wall_by_q.keys())[0]
        bc["wall"] = wall_by_q[first_q]
    else:
        bc["wall"] = None

    return bc


# ============================================================
# 归一化
# ============================================================

def normalize_x(X_raw: np.ndarray, norm: dict) -> np.ndarray:
    """X: (N,5) [x,y,z,Q,m] → [-1,1]⁴ × m_orig."""
    Xn = np.empty_like(X_raw, dtype=np.float32)
    Lx, Ly, Lz = max(norm["Lx"], 1e-8), max(norm["Ly"], 1e-8), max(norm["Lz"], 1e-8)
    Xn[:, 0] = (X_raw[:, 0] - norm["x_min"]) / Lx * 2 - 1
    Xn[:, 1] = (X_raw[:, 1] - norm["y_min"]) / Ly * 2 - 1
    Xn[:, 2] = (X_raw[:, 2] - norm["z_min"]) / Lz * 2 - 1
    Xn[:, 3] = (X_raw[:, 3] / max(norm["Q_max"], 1e-8)) * 2 - 1
    Xn[:, 4] = X_raw[:, 4]
    return Xn


def normalize_y(Y_raw: np.ndarray, norm: dict) -> np.ndarray:
    """Y: (N,4) [u,v,w,p] → z-score normalization。"""
    Yn = np.empty_like(Y_raw, dtype=np.float32)
    Yn[:, 0] = (Y_raw[:, 0] - norm["u_mean"]) / norm["u_std"]
    Yn[:, 1] = (Y_raw[:, 1] - norm["v_mean"]) / norm["v_std"]
    Yn[:, 2] = (Y_raw[:, 2] - norm["w_mean"]) / norm["w_std"]
    Yn[:, 3] = (Y_raw[:, 3] - norm["p_mean"]) / norm["p_std"]
    return Yn


def denormalize_y(Y_norm: np.ndarray, norm: dict) -> np.ndarray:
    """反归一化到物理单位。"""
    Y_raw = np.empty_like(Y_norm, dtype=np.float64)
    Y_raw[:, 0] = Y_norm[:, 0] * norm["u_std"] + norm["u_mean"]
    Y_raw[:, 1] = Y_norm[:, 1] * norm["v_std"] + norm["v_mean"]
    Y_raw[:, 2] = Y_norm[:, 2] * norm["w_std"] + norm["w_mean"]
    Y_raw[:, 3] = Y_norm[:, 3] * norm["p_std"] + norm["p_mean"]
    return Y_raw


# ============================================================
# 数据集切分 ★ 空间 holdout 防测试集污染
# ============================================================

def split_train_test_interp_extrap(X_raw, Y_raw, WD_raw, WP_raw):
    """按 Q 值切分，训练工况内部独立随机空间 holdout。

    返回:
      train           — 训练集 (TRAIN_Q 的 80% cell)
      test_interp     — 内插测试 (TRAIN_Q 的 20% cell)
      test_extrap     — 外推测试 (TEST_EXTRAP_Q 全部 cell)
      test_all        — test_interp ∪ test_extrap (统一评估用)
    """
    Q_all = X_raw[:, 3]
    rng = np.random.default_rng(SEED)

    train_idx       = []
    test_interp_idx = []
    test_extrap_idx = []

    all_Q = sorted(set(Q_all))
    for q in all_Q:
        q_mask = np.where(np.isclose(Q_all, q))[0]

        if q in TRAIN_Q:
            n = len(q_mask)
            n_holdout = max(int(n * HOLD_OUT), 1)
            perm = rng.permutation(n)
            train_idx.extend(q_mask[perm[n_holdout:]])
            test_interp_idx.extend(q_mask[perm[:n_holdout]])
        elif q in TEST_EXTRAP_Q:
            test_extrap_idx.extend(q_mask)
        else:
            # 既不在训练也不在外推的 Q（如未来侧吹混合工况）→ 暂时跳过
            pass

    # 组合测试集
    test_all_idx = np.concatenate([
        np.array(test_interp_idx, dtype=int),
        np.array(test_extrap_idx, dtype=int),
    ])

    def _take(idx_list):
        if len(idx_list) == 0:
            return np.empty((0, X_raw.shape[1])), np.empty((0, Y_raw.shape[1])), \
                   np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        idx = np.array(idx_list, dtype=int)
        return X_raw[idx], Y_raw[idx], WD_raw[idx], WP_raw[idx]

    X_tr,  Y_tr,  Wd_tr,  Wp_tr  = _take(train_idx)
    X_ti,  Y_ti,  Wd_ti,  Wp_ti  = _take(test_interp_idx)
    X_te,  Y_te,  Wd_te,  Wp_te  = _take(test_extrap_idx)
    X_all_t, Y_all_t, Wd_all_t, Wp_all_t = _take(test_all_idx)

    return (X_tr, Y_tr, Wd_tr, Wp_tr,
            X_ti, Y_ti, Wd_ti, Wp_ti,
            X_te, Y_te, Wd_te, Wp_te,
            X_all_t, Y_all_t, Wd_all_t, Wp_all_t)


# ============================================================
# 一键加载
# ============================================================

def build_datasets():
    """一键加载 → 归一化 → tensor → 返回完整 dict。"""
    X_raw, Y_raw, WD_raw, WP_raw, norm, bc, Q_list, stats = load_all_data()

    (X_tr_raw, Y_tr_raw, Wd_tr_raw, Wp_tr_raw,
     X_ti_raw, Y_ti_raw, Wd_ti_raw, Wp_ti_raw,
     X_te_raw, Y_te_raw, Wd_te_raw, Wp_te_raw,
     X_all_t_raw, Y_all_t_raw, Wd_all_t_raw, Wp_all_t_raw,
    ) = split_train_test_interp_extrap(X_raw, Y_raw, WD_raw, WP_raw)

    def _to_t(X, Y, Wd, Wp):
        if len(X) == 0:
            return tuple(torch.empty(0, device=DEVICE) for _ in range(4))
        return (
            torch.tensor(normalize_x(X, norm), dtype=torch.float32, device=DEVICE),
            torch.tensor(normalize_y(Y, norm), dtype=torch.float32, device=DEVICE),
            torch.tensor(Wd, dtype=torch.float32, device=DEVICE),
            torch.tensor(Wp, dtype=torch.float32, device=DEVICE),
        )

    X_tr,  Y_tr,  Wd_tr,  Wp_tr  = _to_t(X_tr_raw,  Y_tr_raw,  Wd_tr_raw,  Wp_tr_raw)
    X_ti,  Y_ti,  Wd_ti,  Wp_ti  = _to_t(X_ti_raw,  Y_ti_raw,  Wd_ti_raw,  Wp_ti_raw)
    X_te,  Y_te,  Wd_te,  Wp_te  = _to_t(X_te_raw,  Y_te_raw,  Wd_te_raw,  Wp_te_raw)
    X_all, Y_all, Wd_all, Wp_all = _to_t(X_all_t_raw, Y_all_t_raw, Wd_all_t_raw, Wp_all_t_raw)

    return {
        # 训练
        "X_train":    X_tr,  "Y_train":    Y_tr,
        "wd_train":   Wd_tr, "wp_train":   Wp_tr,
        # 内插测试
        "X_interp":   X_ti,  "Y_interp":   Y_ti,
        "wd_interp":  Wd_ti, "wp_interp":  Wp_ti,
        # 外推测试
        "X_extrap":   X_te,  "Y_extrap":   Y_te,
        "wd_extrap":  Wd_te, "wp_extrap":  Wp_te,
        # 合并测试（方便 inference）
        "X_test":     X_all, "Y_test":     Y_all,
        "wd_test":    Wd_all,"wp_test":    Wp_all,
        # 元数据
        "norm":       norm,
        "bc":         bc,
        "Q_list":     Q_list,
        "n_train":    X_tr.shape[0],
        "n_interp":   X_ti.shape[0],
        "n_extrap":   X_te.shape[0],
        "stats":      stats,
    }


# ============================================================
# self-test
# ============================================================

if __name__ == "__main__":
    data = build_datasets()

    print("=" * 60)
    print("DATA LOADER SELF-TEST")
    print("=" * 60)

    # 样本统计
    print(f"\nSamples:")
    print(f"  Train:           {data['n_train']:>8d}")
    print(f"  Test (interp):   {data['n_interp']:>8d}  (spatial holdout from TRAIN_Q)")
    print(f"  Test (extrap):   {data['n_extrap']:>8d}  (all cells from TEST_EXTRAP_Q)")
    print(f"  Q values:        {data['Q_list']}")

    # 归一化参数
    norm = data["norm"]
    print(f"\nNormalization:")
    print(f"  U_ref={norm['U_ref']:.4f}, P_ref={norm['P_ref']:.4f}, Q_max={norm['Q_max']}")
    print(f"  Domain: x=[{norm['x_min']:.2f},{norm['x_max']:.2f}], "
          f"y=[{norm['y_min']:.2f},{norm['y_max']:.2f}], "
          f"z=[{norm['z_min']:.2f},{norm['z_max']:.2f}]")

    # VOF 统计
    s = data["stats"]
    print(f"\nVOF stats:")
    print(f"  Steel cells:     {s['n_steel']:>8d} / {s['n_total']}  ({s['steel_frac']:.1%})")
    print(f"  Slag/air cells:  {s['n_slag_air']:>8d}")

    # 权重统计
    for name, wd, wp in [
        ("Train", data["wd_train"], data["wp_train"]),
        ("Interp", data["wd_interp"], data["wp_interp"]),
        ("Extrap", data["wd_extrap"], data["wp_extrap"]),
    ]:
        if wd.numel() > 0:
            wdn, wpn = wd.cpu().numpy(), wp.cpu().numpy()
            print(f"\n  {name}: w_data mean={wdn.mean():.4f}, >0={(wdn>0).sum():,}/{len(wdn):,}")
            print(f"  {'':7s} w_pde  mean={wpn.mean():.4f}, >0={(wpn>0).sum():,}/{len(wpn):,}")

    # BC 状态
    print(f"\nBoundary conditions:")
    for k in ["inlet", "wall"]:
        v = data["bc"].get(k)
        if v is None:
            print(f"  {k}: [MISSING]")
        elif k == "inlet":
            print(f"  {k}: OK {len(v)} Q-values ({list(v.keys())})")
        else:
            print(f"  {k}: OK {v.shape[0]} points")

    # 验证无泄漏
    print(f"\nSanity checks:")
    # 训练集和外推测试集不应有相同的 Q 值
    train_Q_vals = set()
    if data["X_train"].numel() > 0:
        # Q 在归一化空间的第 3 列 (index 3)
        train_Q_norm = data["X_train"][:, 3].cpu().numpy()
        train_Q_phys = (train_Q_norm + 1) / 2 * norm["Q_max"]
        train_Q_vals = set(np.unique(np.round(train_Q_phys)))

    extrap_Q_vals = set()
    if data["X_extrap"].numel() > 0:
        extrap_Q_norm = data["X_extrap"][:, 3].cpu().numpy()
        extrap_Q_phys = (extrap_Q_norm + 1) / 2 * norm["Q_max"]
        extrap_Q_vals = set(np.unique(np.round(extrap_Q_phys)))

    overlap = train_Q_vals & extrap_Q_vals
    if overlap:
        print(f"  [LEAK] Q values in both train and extrap: {overlap}")
    else:
        print(f"  [OK] No Q-value leakage between train and extrap")
    print(f"    Train Q:  {sorted(train_Q_vals)}")
    print(f"    Extrap Q: {sorted(extrap_Q_vals)}")

    print("\n" + "=" * 60)
    print("SELF-TEST COMPLETE")
    print("=" * 60)
