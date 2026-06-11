"""
Physics loss with physical-space scaling.
Continuity: ∇·u = 0 in physical units.
Momentum: (u·∇)u = -∇p/ρ + ν∇²u (simplified for steel).
Uses norm dict to convert normalized→physical gradients.
"""
import torch
from config import NU, L_M, NU_T_CONST

def _grad1(y, x):
    """∂y/∂x for scalar y (B,) or (B,1), x (B,5) → (B,5)"""
    if y.dim()>1: y=y.squeeze(-1)
    return torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True)[0]

def scaled_physics(model, X, norm, nu_mode="mixing"):
    """
    Compute physics residuals in PHYSICAL space.

    Input X is NORMALIZED. Convert gradients to physical:
    du/dx_phys = du_norm/dx_norm * (2*u_std/Lx)

    Returns: L_cont, L_mom (scalar losses)
    """
    X.requires_grad_(True)
    out = model(X)  # (B,4) normalized

    # Get normalized gradients for each output component
    gu = _grad1(out[:,0], X)  # (B,5)
    gv = _grad1(out[:,1], X)
    gw = _grad1(out[:,2], X)
    gp = _grad1(out[:,3], X)

    # Scale factors: du_phys/dx_phys = du_norm/dx_norm * scale_x
    # scale_x = 2 * u_std / Lx, etc.
    Lx, Ly, Lz = norm['Lx'], norm['Ly'], norm['Lz']
    s_ux = 2*norm['u_std']/Lx; s_uy = 2*norm['u_std']/Ly; s_uz = 2*norm['u_std']/Lz
    s_vx = 2*norm['v_std']/Lx; s_vy = 2*norm['v_std']/Ly; s_vz = 2*norm['v_std']/Lz
    s_wx = 2*norm['w_std']/Lx; s_wy = 2*norm['w_std']/Ly; s_wz = 2*norm['w_std']/Lz
    # Pressure gradient scaling
    s_px = 2*norm['p_std']/Lx; s_py = 2*norm['p_std']/Ly; s_pz = 2*norm['p_std']/Lz

    # Physical velocity gradients
    du_dx = gu[:,0]*s_ux; du_dy = gu[:,1]*s_uy; du_dz = gu[:,2]*s_uz
    dv_dx = gv[:,0]*s_vx; dv_dy = gv[:,1]*s_vy; dv_dz = gv[:,2]*s_vz
    dw_dx = gw[:,0]*s_wx; dw_dy = gw[:,1]*s_wy; dw_dz = gw[:,2]*s_wz

    # Pressure gradients (physical)
    dp_dx = gp[:,0]*s_px; dp_dy = gp[:,1]*s_py; dp_dz = gp[:,2]*s_pz

    # === Continuity (physical) ===
    cont = du_dx + dv_dy + dw_dz

    # === Physical velocities (denormalize) ===
    u_p = out[:,0]*norm['u_std'] + norm['u_mean']
    v_p = out[:,1]*norm['v_std'] + norm['v_mean']
    w_p = out[:,2]*norm['w_std'] + norm['w_mean']

    # === Convection (physical) ===
    conv_u = u_p*du_dx + v_p*du_dy + w_p*du_dz
    conv_v = u_p*dv_dx + v_p*dv_dy + w_p*dv_dz
    conv_w = u_p*dw_dx + v_p*dw_dy + w_p*dw_dz

    # === Viscous term ===
    # Need second derivatives
    lap_u = _grad1(du_dx, X)[:,0]*s_ux + _grad1(du_dy, X)[:,1]*s_uy + _grad1(du_dz, X)[:,2]*s_uz
    lap_v = _grad1(dv_dx, X)[:,0]*s_vx + _grad1(dv_dy, X)[:,1]*s_vy + _grad1(dv_dz, X)[:,2]*s_vz
    lap_w = _grad1(dw_dx, X)[:,0]*s_wx + _grad1(dw_dy, X)[:,1]*s_wy + _grad1(dw_dz, X)[:,2]*s_wz

    # Eddy viscosity
    if nu_mode == "mixing":
        S11=du_dx; S22=dv_dy; S33=dw_dz
        S12=0.5*(du_dy+dv_dx); S13=0.5*(du_dz+dw_dx); S23=0.5*(dv_dz+dw_dy)
        S_mag_sq = 2*(S11**2+S22**2+S33**2 + 2*S12**2+2*S13**2+2*S23**2)
        S_mag = torch.sqrt(S_mag_sq.clamp(min=1e-16))
        nu_t = L_M*L_M*S_mag
    elif nu_mode == "const":
        nu_t = torch.full_like(du_dx, NU_T_CONST)
    else:
        nu_t = torch.zeros_like(du_dx)

    nu_eff = NU + nu_t

    # === Momentum residual: u·∇u + ∇p/ρ - ν_eff ∇²u = 0 ===
    # ρ_steel ≈ 7000 kg/m³ (cancels if pressure is in Pa)
    rho = 7000.0
    mom_u = conv_u + dp_dx/rho - nu_eff*lap_u
    mom_v = conv_v + dp_dy/rho - nu_eff*lap_v
    mom_w = conv_w + dp_dz/rho - nu_eff*lap_w

    L_cont = cont.pow(2).mean()
    L_mom = (mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean()) / 3.0

    return L_cont, L_mom
