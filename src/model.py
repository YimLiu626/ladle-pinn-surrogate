"""
model.py — 网络定义
MLP with Fourier feature encoding + MC Dropout。所有超参数从 config.py 读取。
"""
import math
import torch
import torch.nn as nn
from config import LAYERS, ACTIVATION, MC_DROPOUT_P, N_INPUT, N_OUTPUT, FOURIER_L, FOURIER_SIGMA


class FourierFeatures(nn.Module):
    """对 (x,y,z) 逐坐标做 Fourier feature mapping。

    γ(p) = [sin(π p), cos(π p), sin(2π p), cos(2π p), ..., sin(2^{L-1}π p), cos(2^{L-1}π p)]
    3 个坐标 → 6L 维。
    """

    def __init__(self, L=6, sigma=1.0):
        super().__init__()
        self.L = L
        # 频率: [1, 2, 4, ..., 2^{L-1}] * π
        freqs = (2.0 ** torch.arange(L)) * math.pi * sigma  # (L,)
        self.register_buffer("freqs", freqs)

    def forward(self, xyz):
        """xyz: (B, 3) → (B, 6L)"""
        # xyz: (B, 3), freqs: (L,) → outer → (B, 3, L)
        proj = xyz.unsqueeze(-1) * self.freqs  # (B, 3, L)
        proj = proj.flatten(1)                  # (B, 3*L)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, 6L)


class PINN(nn.Module):
    """MLP + Fourier features for (x,y,z) + raw (Q,m)。

    Parameters
    ----------
    layers : list[int]
        每层神经元数。第一层自动调整为 Fourier 特征维度 + 2。
    activation : str
        "tanh" | "relu" | "swish" | "gelu"
    dropout : float
    fourier_L : int
        Fourier 特征频率数。
    fourier_sigma : float
        频率尺度。
    """

    def __init__(self, layers=None, activation=None, dropout=None,
                 fourier_L=None, fourier_sigma=None):
        super().__init__()
        self.act_name   = activation if activation is not None else ACTIVATION
        self.dropout_p  = dropout if dropout is not None else MC_DROPOUT_P
        self.fourier_L  = fourier_L if fourier_L is not None else FOURIER_L

        base_layers = layers if layers is not None else LAYERS

        if self.fourier_L > 0:
            self.fourier_sigma = fourier_sigma if fourier_sigma is not None else FOURIER_SIGMA
            self.fourier = FourierFeatures(L=self.fourier_L, sigma=self.fourier_sigma)
            ff_dim = self.fourier_L * 6
            self.layers_spec = [ff_dim + (base_layers[0] - 3)] + base_layers[1:]
        else:
            self.fourier = None
            self.layers_spec = list(base_layers)

        blocks = []
        for i in range(len(self.layers_spec) - 1):
            blocks.append(nn.Linear(self.layers_spec[i], self.layers_spec[i + 1]))
            if i < len(self.layers_spec) - 2:
                blocks.append(self._get_activation())
                if self.dropout_p > 0:
                    blocks.append(nn.Dropout(self.dropout_p))
        self.net = nn.Sequential(*blocks)
        self._init_weights()

    def _get_activation(self):
        act = self.act_name.lower()
        if act == "tanh":    return nn.Tanh()
        elif act == "relu":  return nn.ReLU()
        elif act in ("swish", "silu"): return nn.SiLU()
        elif act == "gelu":  return nn.GELU()
        raise ValueError(f"Unknown activation: {self.act_name}")

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if self.act_name == "relu":
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B, 4) or (B, 5) → (B, 4) [u, v, w, p]"""
        if self.fourier is not None:
            xyz = x[:, :3]
            rest = x[:, 3:]
            return self.net(torch.cat([self.fourier(xyz), rest], dim=-1))
        return self.net(x)

    def enable_dropout(self):
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_dropout(self):
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.eval()


def make_model(layers=None, activation=None, dropout=None,
               fourier_L=None, fourier_sigma=None):
    """工厂函数。"""
    return PINN(layers=layers, activation=activation, dropout=dropout,
                fourier_L=fourier_L, fourier_sigma=fourier_sigma)


# ============================================================
# self-test
# ============================================================

if __name__ == "__main__":
    model = PINN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PINN: {model.layers_spec}")
    print(f"  Activation: {model.act_name}")
    print(f"  Dropout:    {model.dropout_p}")
    print(f"  Params:     {n_params:,}")

    # 前向测试
    B = 4
    x = torch.randn(B, N_INPUT)
    out = model(x)
    print(f"  Input:  {x.shape} → Output: {out.shape}")
    assert out.shape == (B, N_OUTPUT), f"Expected ({B},{N_OUTPUT}), got {out.shape}"

    # MC Dropout 测试
    model.enable_dropout()
    out1 = model(x)
    out2 = model(x)
    diff = (out1 - out2).abs().max().item()
    print(f"  MC Dropout: max|out1-out2| = {diff:.6f} (should be >0)")
    model.disable_dropout()

    print("  ✓ All tests passed")
