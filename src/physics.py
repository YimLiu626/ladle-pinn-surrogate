"""
physics.py — RANS 残差（混合长度模型 ν_t = l_m²·|S|）

ν_eff = ν + ν_t = ν + l_m² · √(2 S_ij S_ij)
S_ij = 0.5 (∂u_i/∂x_j + ∂u_j/∂x_i) — 应变率张量
|S| = √(2 S_ij S_ij)

用于消融：νt_mode ∈ {"mixing", "const", "none"}
"""
import torch
from torch.autograd import grad
from config import NU, L_M


def _grad_scalar(y, x):
    """∂y/∂x, y: (B,1), x: (B,D) → (B,D)"""
    g = grad(y, x, grad_outputs=torch.ones_like(y),
             create_graph=True, retain_graph=True)[0]
    return g


def compute_rans_residuals(model, X, nu_t_mode="mixing"):
    """计算 RANS 残差的平方均值。

    Parameters
    ----------
    model : PINN
    X : (B, 5) normalized [x,y,z,Q,m]
    nu_t_mode : "mixing" | "const" | "none"
        mixing: ν_t = l_m²·|S|
        const:  ν_t = NU_T_CONST (from config, for ablation)
        none:   ν_t = 0 (laminar, for ablation)

    Returns
    -------
    L : scalar loss (mean of cont² + mom_u² + mom_v² + mom_w²)
    cont, mom_u, mom_v, mom_w : (B,1) individual residuals
    """
    X.requires_grad_(True)
    out = model(X)
    u, v, w, p_ = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

    # ---- 一阶导数 ----
    gu = _grad_scalar(u, X)  # (B, 5)
    gv = _grad_scalar(v, X)
    gw = _grad_scalar(w, X)
    gp = _grad_scalar(p_, X)  # (B, 5)

    # 只取空间导数 (index 0,1,2)
    du_dx, du_dy, du_dz = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3]
    dv_dx, dv_dy, dv_dz = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3]
    dw_dx, dw_dy, dw_dz = gw[:, 0:1], gw[:, 1:2], gw[:, 2:3]
    dp_dx, dp_dy, dp_dz = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]

    # ---- 连续性 ----
    cont = du_dx + dv_dy + dw_dz

    # ---- 对流项 ----
    conv_u = u * du_dx + v * du_dy + w * du_dz
    conv_v = u * dv_dx + v * dv_dy + w * dv_dz
    conv_w = u * dw_dx + v * dw_dy + w * dw_dz

    # ---- 二阶导数（Laplacian） ----
    lap_u = _grad_scalar(du_dx, X)[:, 0:1] + _grad_scalar(du_dy, X)[:, 1:2] + _grad_scalar(du_dz, X)[:, 2:3]
    lap_v = _grad_scalar(dv_dx, X)[:, 0:1] + _grad_scalar(dv_dy, X)[:, 1:2] + _grad_scalar(dv_dz, X)[:, 2:3]
    lap_w = _grad_scalar(dw_dx, X)[:, 0:1] + _grad_scalar(dw_dy, X)[:, 1:2] + _grad_scalar(dw_dz, X)[:, 2:3]

    # ---- 涡黏度 ----
    if nu_t_mode == "mixing":
        # |S| = √(2 S_ij S_ij)
        S11 = du_dx; S22 = dv_dy; S33 = dw_dz
        S12 = 0.5 * (du_dy + dv_dx); S13 = 0.5 * (du_dz + dw_dx)
        S23 = 0.5 * (dv_dz + dw_dy)
        # 2 S_ij S_ij = 2(S11²+S22²+S33²+2S12²+2S13²+2S23²)
        S_mag_sq = 2.0 * (S11**2 + S22**2 + S33**2 +
                          2.0 * S12**2 + 2.0 * S13**2 + 2.0 * S23**2)
        S_mag = torch.sqrt(S_mag_sq.clamp(min=1e-16))
        nu_t = L_M * L_M * S_mag
    elif nu_t_mode == "const":
        from config import NU_T_CONST
        nu_t = torch.full_like(du_dx, NU_T_CONST)
    else:  # "none"
        nu_t = torch.zeros_like(du_dx)

    nu_eff = NU + nu_t

    # ---- 动量残差: u·∇u + ∇p - ν_eff ∇²u = 0 ----
    mom_u = conv_u + dp_dx - nu_eff * lap_u
    mom_v = conv_v + dp_dy - nu_eff * lap_v
    mom_w = conv_w + dp_dz - nu_eff * lap_w

    # 各分量 MSE
    L = (cont.pow(2).mean() + mom_u.pow(2).mean() +
         mom_v.pow(2).mean() + mom_w.pow(2).mean())

    return L, cont.detach(), mom_u.detach(), mom_v.detach(), mom_w.detach()


# ============================================================
# self-test
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '')
    from model import PINN

    torch.manual_seed(42)
    model = PINN()
    X = torch.randn(16, 5)  # 5D: x,y,z,Q,m

    for mode in ["mixing", "const", "none"]:
        L, cont, mu, mv, mw = compute_rans_residuals(model, X, nu_t_mode=mode)
        print(f"  {mode:8s}: L={L.item():.6f}")

    print("  ✓ Physics self-test passed")
