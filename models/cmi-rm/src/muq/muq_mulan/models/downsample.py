import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask if mask.dtype == torch.bool else (mask != 0)


def _trim_to_multiple(x: torch.Tensor, mask: torch.Tensor, m: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trim time dimension to multiple of m so that output length is exact floor.
    x:    (B,T,D)
    mask: (B,T)
    """
    if m <= 1:
        return x, mask
    B, T, D = x.shape
    T_trim = (T // m) * m
    if T_trim == T:
        return x, mask
    return x[:, :T_trim], mask[:, :T_trim]


def _mask_downsample_any(mask: torch.Tensor, stride: int) -> torch.Tensor:
    """
    mask: (B,T) bool
    stride: s
    returns: (B, T//s) bool, using window-any over non-overlapping chunks.
    """
    if stride <= 1:
        return mask
    B, T = mask.shape
    T_trim = (T // stride) * stride
    mask = mask[:, :T_trim].view(B, T_trim // stride, stride)
    return mask.any(dim=-1)


def _choose_two_stage_strides(factor: int, prefer: Tuple[int, ...] = (3, 2, 4)) -> Tuple[int, int]:
    """
    Find (s1, s2) such that s1*s2 = factor and both are "small",
    with preference ordering favoring (3,3), then involving 3, then 2, etc.

    Raises ValueError if cannot represent factor by 2 integer strides.
    """
    if factor <= 1:
        return (1, 1)

    best = None
    best_score = None

    # search divisors
    for s1 in range(2, factor + 1):
        if factor % s1 != 0:
            continue
        s2 = factor // s1

        # score: smaller max stride is better; also prefer s1,s2 in prefer list and near 3
        def pref_rank(s: int) -> int:
            return prefer.index(s) if s in prefer else 999

        score = (
            max(s1, s2),                 # primary: keep strides small
            pref_rank(s1) + pref_rank(s2),
            abs(s1 - 3) + abs(s2 - 3),   # prefer close to 3
            abs(s1 - s2),                # prefer balanced
        )

        if best_score is None or score < best_score:
            best_score = score
            best = (s1, s2)

    if best is None:
        raise ValueError(f"factor={factor} cannot be decomposed into 2 integer strides.")
    # reorder so that first stage stride is "more preferred" if possible
    s1, s2 = best
    # put 3 first if it exists, else smaller first
    if (s2 == 3 and s1 != 3) or (s2 < s1 and s1 != 3):
        s1, s2 = s2, s1
    return s1, s2


# ---------------------------
# Baseline 0: meanpool only
# ---------------------------

class MeanPoolDownsampler(nn.Module):
    """
    One-shot masked mean downsample to floor(T/N), then MLP keeps dim=D.
    """
    def __init__(self, dim: int, factor: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        assert factor >= 1
        self.factor = factor
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,T,D), mask: (B,T) True=valid
        returns y: (B,floor(T/N),D), mask_out: (B,floor(T/N))
        """
        mask = _to_bool_mask(mask)
        if self.factor == 1:
            x = x * mask.unsqueeze(-1).to(x.dtype)
            return x, mask

        # trim to multiple of factor => exact floor(T/N)
        x, mask = _trim_to_multiple(x, mask, self.factor)
        B, T, D = x.shape
        T_out = T // self.factor
        if T_out == 0:
            # no valid output tokens
            y = x.new_zeros(B, 0, D)
            m = mask.new_zeros(B, 0)
            return y, m

        x = x.view(B, T_out, self.factor, D)
        m = mask.view(B, T_out, self.factor, 1).to(x.dtype)
        denom = m.sum(dim=2).clamp_min(1.0)
        y = (x * m).sum(dim=2) / denom  # masked mean
        mask_out = (m.sum(dim=2).squeeze(-1) > 0)
        # avoid leaking padding in downstream
        y = y * mask_out.unsqueeze(-1).to(y.dtype)
        return y, mask_out


# ---------------------------
# Baseline 1: meanpool + MLP
# ---------------------------
class MeanPoolMLPDownsampler(nn.Module):
    """
    One-shot masked mean downsample to floor(T/N), then MLP keeps dim=D.
    """
    def __init__(self, dim: int, factor: int, mlp_ratio: float = 2.0, dropout: float = 0.0, pre_mlp: bool = False ):
        super().__init__()
        assert factor >= 1
        self.factor = factor
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.pre_mlp = pre_mlp

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,T,D), mask: (B,T) True=valid
        returns y: (B,floor(T/N),D), mask_out: (B,floor(T/N))
        """
        mask = _to_bool_mask(mask)
        if self.factor == 1:
            x = x * mask.unsqueeze(-1).to(x.dtype)
            return self.mlp(x), mask

        # trim to multiple of factor => exact floor(T/N)
        if self.pre_mlp:
            x = self.mlp(x)
        x, mask = _trim_to_multiple(x, mask, self.factor)
        B, T, D = x.shape
        T_out = T // self.factor
        if T_out == 0:
            # no valid output tokens
            y = x.new_zeros(B, 0, D)
            m = mask.new_zeros(B, 0)
            return y, m

        x = x.view(B, T_out, self.factor, D)
        m = mask.view(B, T_out, self.factor, 1).to(x.dtype)
        denom = m.sum(dim=2).clamp_min(1.0)
        y = (x * m).sum(dim=2) / denom  # masked mean
        mask_out = (m.sum(dim=2).squeeze(-1) > 0)
        # avoid leaking padding in downstream
        y = y * mask_out.unsqueeze(-1).to(y.dtype)    
        if not self.pre_mlp:
            y = self.mlp(y)
        return y, mask_out


# --------------------------------
# Baseline 2: (conv + SiLU) * 2
# --------------------------------
class _DepthwiseConvDown(nn.Module):
    def __init__(self, dim: int, stride: int, kernel_size: int = 5, use_layernorm: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.stride = stride
        self.dw = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=dim,
            bias=False,
        )
        self.use_layernorm = use_layernorm
        self.ln = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # trim so SAME padding + stride gives exact T//stride
        x, mask = _trim_to_multiple(x, mask, self.stride)
        mask = _to_bool_mask(mask)
        B, T, D = x.shape
        if T == 0:
            return x, mask[:, :0]

        # zero padded tokens before conv
        x = x * mask.unsqueeze(-1).to(x.dtype)

        # conv in (B,D,T)
        y = self.dw(x.transpose(1, 2)).transpose(1, 2)  # (B, T//s, D)
        mask_out = _mask_downsample_any(mask, self.stride)

        # token-wise LN is safer for extreme variable lengths
        y = self.ln(y)
        y = self.act(y)

        # re-apply mask
        y = y * mask_out.unsqueeze(-1).to(y.dtype)
        return y, mask_out


class ConvSiLUTwiceDownsampler(nn.Module):
    """
    2-stage downsample: (conv+SiLU) * 2, strides picked to match factor, prefer (3,3).
    Keeps dim=D.
    """
    def __init__(self, dim: int, factor: int, kernel_size: int = 5, use_layernorm: bool = True):
        super().__init__()
        assert factor >= 1
        self.factor = factor
        if factor == 1:
            self.s1 = self.s2 = 1
        else:
            self.s1, self.s2 = _choose_two_stage_strides(factor, prefer=(3, 2, 4))
        self.block1 = _DepthwiseConvDown(dim, self.s1, kernel_size=kernel_size, use_layernorm=use_layernorm)
        self.block2 = _DepthwiseConvDown(dim, self.s2, kernel_size=kernel_size, use_layernorm=use_layernorm)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = _to_bool_mask(mask)
        if self.factor == 1:
            x = x * mask.unsqueeze(-1).to(x.dtype)
            return x, mask

        x, mask = self.block1(x, mask)
        x, mask = self.block2(x, mask)
        return x, mask

class ConvSiLuDownsampler(nn.Module):
    '''
    can assign the stage number
    '''
    def __init__(self, dim: int, factor: int, stage: int =1, kernel_size: int = 5, use_layernorm: bool = True):
        super().__init__()
        assert factor >= 1
        self.factor = factor
        self.stride = factor
        self.blocks = nn.ModuleList([_DepthwiseConvDown(dim, self.stride, kernel_size=kernel_size, use_layernorm=use_layernorm)]*stage)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = _to_bool_mask(mask)
        if self.factor == 1:
            x = x * mask.unsqueeze(-1).to(x.dtype)
            return x, mask

        for block in self.blocks:
            x, mask = block(x, mask)
        return x, mask
# -------------------------------------------------------
# Baseline 3: (pointwise+GLU+conv+SiLU) * 2 + pointwise
# -------------------------------------------------------
class _GLUConvDown(nn.Module):
    def __init__(self, dim: int, stride: int, kernel_size: int = 5, use_layernorm: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.stride = stride

        # pointwise + GLU in conv space (B,D,T)
        self.pw1 = nn.Conv1d(dim, 2 * dim, kernel_size=1, bias=False)
        self.glu = nn.GLU(dim=1)

        # depthwise conv with stride
        self.dw = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=dim,
            bias=False,
        )

        self.use_layernorm = use_layernorm
        self.ln = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # trim => exact floor
        x, mask = _trim_to_multiple(x, mask, self.stride)
        mask = _to_bool_mask(mask)
        B, T, D = x.shape
        if T == 0:
            return x, mask[:, :0]

        x = x * mask.unsqueeze(-1).to(x.dtype)

        h = x.transpose(1, 2)          # (B,D,T)
        h = self.pw1(h)                # (B,2D,T)
        h = self.glu(h)                # (B,D,T)
        h = self.dw(h)                 # (B,D,T//s)
        y = h.transpose(1, 2)          # (B,T//s,D)

        mask_out = _mask_downsample_any(mask, self.stride)
        y = self.ln(y)
        y = self.act(y)
        y = y * mask_out.unsqueeze(-1).to(y.dtype)
        return y, mask_out


class GLUConvSiLUTwicePlusPointwiseDownsampler(nn.Module):
    """
    (pointwise+GLU+conv+SiLU)*2 + final pointwise (keeps dim=D)
    """
    def __init__(self, dim: int, factor: int, kernel_size: int = 5, use_layernorm: bool = True):
        super().__init__()
        assert factor >= 1
        self.factor = factor
        if factor == 1:
            self.s1 = self.s2 = 1
        else:
            self.s1, self.s2 = _choose_two_stage_strides(factor, prefer=(3, 2, 4))

        self.block1 = _GLUConvDown(dim, self.s1, kernel_size=kernel_size, use_layernorm=use_layernorm)
        self.block2 = _GLUConvDown(dim, self.s2, kernel_size=kernel_size, use_layernorm=use_layernorm)

        # final pointwise projection
        self.final_pw = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = _to_bool_mask(mask)
        if self.factor == 1:
            x = x * mask.unsqueeze(-1).to(x.dtype)
            return x, mask

        x, mask = self.block1(x, mask)
        x, mask = self.block2(x, mask)

        if x.size(1) == 0:
            return x, mask

        # final pointwise in conv space
        y = self.final_pw(x.transpose(1, 2)).transpose(1, 2)
        y = y * mask.unsqueeze(-1).to(y.dtype)
        return y, mask

class Indentity(nn.Module):
    def forward(self, *args):
        return args
# ---------------------------
# Quick factory (optional)
# ---------------------------
def build_downsampler(kind: str, dim: int = 512, factor: int = 1, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind in ["mean", "meanpool"]:
        return MeanPoolDownsampler(dim, factor, **kwargs)
    if kind in ["meanpool+mlp", "mean+mlp", "baseline1", "b1"]:
        return MeanPoolMLPDownsampler(dim, factor, **kwargs)
    if kind in ["conv*2", "baseline2", "b2"]:
        return ConvSiLUTwiceDownsampler(dim, factor, **kwargs)
    if kind in ["conv", "baseline2simple", "b2s"]:
        return ConvSiLuDownsampler(dim, factor, **kwargs)
    if kind in ["gluconv*2+pw", "baseline3", "b3"]:
        return GLUConvSiLUTwicePlusPointwiseDownsampler(dim, factor, **kwargs)
    if kind in ['none']:
        return Indentity()
    raise ValueError(f"Unknown kind: {kind}")
if __name__ == "__main__":
    B, T, D = 4, 173, 768
    x = torch.randn(B, T, D)
    mask = torch.arange(T).unsqueeze(0).repeat(B, 1) < torch.tensor([[160],[120],[80],[173]])

    N = 9  # want floor(T/N)

    m1 = MeanPoolMLPDownsampler(D, N)
    y1, mk1 = m1(x, mask)   # (B, floor(T/9), D)
    print(f"{y1.shape}, {mk1.shape}")
    m2 = ConvSiLUTwiceDownsampler(D, N, kernel_size=5, use_layernorm=True)
    y2, mk2 = m2(x, mask)   # 2-stage conv, prefer (3,3)
    print(f"{y2.shape}, {mk2.shape}")
    m3 = GLUConvSiLUTwicePlusPointwiseDownsampler(D, N, kernel_size=5, use_layernorm=True)
    y3, mk3 = m3(x, mask)
    print(f"{y3.shape}, {mk3.shape}")