from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    convnext_small,
    convnext_tiny,
    resnet50,
    swin_t,
)
from torchvision.models.feature_extraction import create_feature_extractor


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        d: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScaledDotAttention(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(dim)

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = q_feat.shape
        n = h * w

        q = self.q_proj(q_feat).flatten(2).transpose(1, 2)  # [B, N, C]
        k = self.k_proj(kv_feat).flatten(2).transpose(1, 2)  # [B, N, C]
        v = self.v_proj(kv_feat).flatten(2).transpose(1, 2)  # [B, N, C]

        attn = torch.matmul(q, k.transpose(1, 2)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return self.out_proj(out)


class PseudoEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNAct(1, 32, s=2),   # /2
            ConvBNAct(32, 64, s=2),  # /4
            ConvBNAct(64, 96, s=2),  # /8
            ConvBNAct(96, hidden_dim, s=2),  # /16
            ConvBNAct(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ASPPContext(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        branch_dim = max(16, dim // 2)
        self.b1 = ConvBNAct(dim, branch_dim, k=1, p=0)
        self.b2 = ConvBNAct(dim, branch_dim, k=3, p=2, d=2)
        self.b3 = ConvBNAct(dim, branch_dim, k=3, p=4, d=4)
        self.b4 = ConvBNAct(dim, branch_dim, k=3, p=6, d=6)
        self.project = nn.Sequential(
            ConvBNAct(branch_dim * 4, dim, k=1, p=0),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.project(x_cat)


class MFEContext(nn.Module):
    # from useful_module/dream_code_v3/6_4_MFEblock.py (adapted)
    def __init__(self, dim: int, atrous_rates: tuple[int, int, int] = (2, 4, 8)) -> None:
        super().__init__()
        r1, r2, r3 = atrous_rates
        self.layer1 = ConvBNAct(dim, dim, k=3, p=1, d=1)
        self.layer2 = ConvBNAct(dim, dim, k=3, p=max(1, int(r1)), d=max(1, int(r1)))
        self.layer3 = ConvBNAct(dim, dim, k=3, p=max(1, int(r2)), d=max(1, int(r2)))
        self.layer4 = ConvBNAct(dim, dim, k=3, p=max(1, int(r3)), d=max(1, int(r3)))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.se2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.se3 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.se4 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.project = ConvBNAct(dim, dim, k=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.layer1(x)
        y1 = self.layer2(y0 + x)
        y2 = self.layer3(y1 + x)
        y3 = self.layer4(y2 + x)

        w0 = self.se1(self.gap(y0))
        w1 = self.se2(self.gap(y1))
        w2 = self.se3(self.gap(y2))
        w3 = self.se4(self.gap(y3))

        # Keep the branch-weight normalization in fp32 for bf16/channels-last stability.
        w = torch.cat([w0, w1, w2, w3], dim=2)
        w = torch.sigmoid(w).to(dtype=torch.float32)
        w = self.softmax(w).to(dtype=x.dtype).contiguous()

        out = (
            y0 * w[:, :, 0:1, :]
            + y1 * w[:, :, 1:2, :]
            + y2 * w[:, :, 2:3, :]
            + y3 * w[:, :, 3:4, :]
        )
        return self.project(out + x)


class SEBlock(nn.Module):
    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, dim // reduction)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, output_size=1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ECAAttention(nn.Module):
    # from useful_module/dream_code_v3/1_6_ECA-Net.py (adapted)
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x).squeeze(-1).transpose(1, 2)  # [B,1,C]
        y = self.sigmoid(self.conv(y))
        y = y.transpose(1, 2).unsqueeze(-1)  # [B,C,1,1]
        return x * y


class EMAContext(nn.Module):
    # from useful_module/dream_code_v3/2_3_EMAttention.py (adapted)
    def __init__(self, channels: int, factor: int = 32) -> None:
        super().__init__()
        g = max(1, min(int(factor), int(channels)))
        while channels % g != 0 and g > 1:
            g -= 1
        self.groups = g
        c_per_group = channels // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # align with reference implementation behavior
        self.gn = nn.GroupNorm(c_per_group, c_per_group)
        self.conv1x1 = nn.Conv2d(c_per_group, c_per_group, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(c_per_group, c_per_group, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        y1 = torch.matmul(x11, x12)

        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        y2 = torch.matmul(x21, x22)

        weights = (y1 + y2).reshape(b * self.groups, 1, h, w).sigmoid()
        return (group_x * weights).reshape(b, c, h, w)


class SAFMDecoder(nn.Module):
    # from useful_module/dream_code_v3/6_5_SAFM.py (adapted)
    def __init__(self, dim: int, n_levels: int = 4) -> None:
        super().__init__()
        lv = max(1, min(int(n_levels), int(dim)))
        while dim % lv != 0 and lv > 1:
            lv -= 1
        self.n_levels = lv
        chunk_dim = dim // self.n_levels
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, kernel_size=3, stride=1, padding=1, groups=chunk_dim, bias=False) for _ in range(self.n_levels)]
        )
        self.aggr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_h = max(1, h // (2 ** i))
                p_w = max(1, w // (2 ** i))
                s = F.adaptive_max_pool2d(xc[i], (p_h, p_w))
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode="nearest")
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        fused = self.aggr(torch.cat(out, dim=1))
        return self.act(fused) * x


class MSAAChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // max(1, reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        return self.sigmoid(w)


class MSAASpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class MSAAFusion(nn.Module):
    # from useful_module/dream_code_v3/6_7_MSAA.py (adapted)
    def __init__(self, dim: int, factor: int = 4) -> None:
        super().__init__()
        mid = max(16, dim // max(1, int(factor)))
        self.down = nn.Sequential(
            nn.Conv2d(dim * 3, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(mid, mid, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(mid, mid, kernel_size=7, padding=3, bias=False)
        self.channel_attn = MSAAChannelAttention(mid, reduction=4)
        self.spatial_attn = MSAASpatialAttention(kernel_size=7)
        self.up = nn.Sequential(
            nn.Conv2d(mid, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        x = self.down(torch.cat([x1, x2, x3], dim=1))
        x_c = x * self.channel_attn(x)
        x_s = self.conv3(x) + self.conv5(x) + self.conv7(x)
        x_s = x_s * self.spatial_attn(x_s)
        return self.up(x_c + x_s) + x2


class CBAMChannelAttention(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channel // reduction)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channel, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.mlp(self.maxpool(x)) + self.mlp(self.avgpool(x)))


class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_result = torch.max(x, dim=1, keepdim=True)[0]
        avg_result = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([max_result, avg_result], dim=1)))


class CBAMBlock(nn.Module):
    # from useful_module/dream_code_v3/1_3_CBAM.py (adapted)
    def __init__(self, channel: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attn = CBAMChannelAttention(channel=channel, reduction=reduction)
        self.spatial_attn = CBAMSpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = x * self.channel_attn(x)
        out = out * self.spatial_attn(out)
        return out + residual


class SimAMAttention(nn.Module):
    # from useful_module/dream_code_v3/2_24_SimamAttention.py (adapted)
    def __init__(self, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        n = max(1, h * w - 1)
        x_centered_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        attn = x_centered_sq / (4.0 * (x_centered_sq.sum(dim=[2, 3], keepdim=True) / n + self.epsilon)) + 0.5
        return x * self.activation(attn)


class TripletAttentionGate(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        avg_map = torch.mean(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([max_map, avg_map], dim=1)))
        return x * scale


class TripletAttention(nn.Module):
    # from useful_module/dream_code_v3/1_4_TripletAttention.py (adapted)
    def __init__(self, no_spatial: bool = False, kernel_size: int = 7) -> None:
        super().__init__()
        self.cw = TripletAttentionGate(kernel_size=kernel_size)
        self.hc = TripletAttentionGate(kernel_size=kernel_size)
        self.no_spatial = bool(no_spatial)
        self.hw = None if self.no_spatial else TripletAttentionGate(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()

        if self.hw is None:
            return 0.5 * (x_out1 + x_out2)
        x_out3 = self.hw(x)
        return (x_out1 + x_out2 + x_out3) / 3.0


class SKContext(nn.Module):
    # from useful_module/dream_code_v3/1_2_SKNets.py (adapted)
    def __init__(
        self,
        dim: int,
        kernels: tuple[int, ...] = (1, 3, 5),
        reduction: int = 8,
        groups: int = 8,
        min_hidden: int = 32,
    ) -> None:
        super().__init__()
        branch_ops = []
        g = max(1, min(int(groups), int(dim)))
        while dim % g != 0 and g > 1:
            g -= 1
        for k in kernels:
            kk = max(1, int(k))
            if kk % 2 == 0:
                kk += 1
            branch_ops.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kk, padding=kk // 2, groups=g, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branch_ops)

        hidden = max(int(min_hidden), int(dim // max(1, reduction)))
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.attn_heads = nn.ModuleList([nn.Conv2d(hidden, dim, kernel_size=1, bias=False) for _ in kernels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [op(x) for op in self.branches]
        fused = feats[0]
        for f in feats[1:]:
            fused = fused + f
        z = self.fc(F.adaptive_avg_pool2d(fused, output_size=1))
        logits = torch.stack([head(z) for head in self.attn_heads], dim=0)  # [K,B,C,1,1]
        weights = torch.softmax(logits, dim=0)
        out = torch.zeros_like(feats[0])
        for i, f in enumerate(feats):
            out = out + weights[i] * f
        return out


class CoTContext(nn.Module):
    # from useful_module/dream_code_v3/2_2_CoTAttention.py (adapted)
    def __init__(self, dim: int, kernel_size: int = 3, factor: int = 4, groups: int = 4) -> None:
        super().__init__()
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1
        g = max(1, min(int(groups), int(dim)))
        while dim % g != 0 and g > 1:
            g -= 1
        hidden = max(16, (2 * dim) // max(1, int(factor)))

        self.dim = int(dim)
        self.kernel_size = k
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=g, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, (k * k) * dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).reshape(b, c, -1)
        att = self.attention_embed(torch.cat([k1, x], dim=1))
        att = att.reshape(b, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(dim=2).reshape(b, c, -1)
        att = torch.softmax(att, dim=-1)
        k2 = (att * v).reshape(b, c, h, w)
        return k1 + k2


class AnchorContext(nn.Module):
    # from useful_module/dream_code_v3/2_23_AnchorAttention.py (adapted)
    def __init__(
        self,
        dim: int,
        anchors: int = 8,
        heads: int = 4,
        dim_head: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        h = max(1, int(heads))
        d = max(8, int(dim_head))
        inner = h * d
        a = max(2, int(anchors))

        self.heads = h
        self.dim_head = d
        self.scale = d ** -0.5

        self.qk_proj = nn.Conv2d(dim, inner, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, inner, kernel_size=1, bias=False)
        self.anchor_proj = nn.Linear(d, a, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(inner, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        n = h * w

        qk = self.qk_proj(x).reshape(b, self.heads, self.dim_head, n).permute(0, 1, 3, 2).contiguous()
        v = self.v_proj(x).reshape(b, self.heads, self.dim_head, n).permute(0, 1, 3, 2).contiguous()

        anchor_logits = self.anchor_proj(qk) * self.scale  # [B,H,N,A]
        anchor_attn = torch.softmax(anchor_logits, dim=-1)

        # aggregate token features into anchors then redistribute back to tokens
        anchor_feat = torch.matmul(anchor_attn.transpose(-1, -2), v)  # [B,H,A,D]
        anchor_degree = anchor_attn.sum(dim=-2, keepdim=False).clamp_min(1e-6).unsqueeze(-1)  # [B,H,A,1]
        anchor_feat = anchor_feat / anchor_degree
        out = torch.matmul(anchor_attn, anchor_feat)  # [B,H,N,D]

        out = out.permute(0, 1, 3, 2).contiguous().reshape(b, self.heads * self.dim_head, h, w)
        return x + self.out_proj(out)


class SMFADecoder(nn.Module):
    # from useful_module/dream_code_v3/2_18_SMFA.py (adapted)
    def __init__(self, dim: int, growth_rate: float = 2.0, down_scale: int = 8) -> None:
        super().__init__()
        hidden = max(16, int(dim * growth_rate))
        self.linear_0 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.linear_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.linear_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.local_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
        )
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.gelu = nn.GELU()
        self.down_scale = max(2, int(down_scale))

        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.out_norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        _, _, h, w = x.shape
        y, z = self.linear_0(x).chunk(2, dim=1)

        p_h = max(1, h // self.down_scale)
        p_w = max(1, w // self.down_scale)
        z_s = self.dw_conv(F.adaptive_max_pool2d(z, output_size=(p_h, p_w)))
        z_centered = z - z.mean(dim=(-2, -1), keepdim=True)
        z_v = (z_centered * z_centered).mean(dim=(-2, -1), keepdim=True)
        z_gate = self.gelu(self.linear_1(z_s * self.alpha + z_v * self.beta))
        z_local = z * F.interpolate(z_gate, size=(h, w), mode="nearest")

        y_local = self.local_mlp(y)
        out = self.linear_2(z_local + y_local)
        return F.relu(self.out_norm(x + out), inplace=True)


class TFFRefine(nn.Module):
    # from useful_module/dream_code_v3/6_8_TFF.py (adapted)
    def __init__(self, dim: int) -> None:
        super().__init__()

        def dsconv(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.catconv_a = dsconv(dim * 2, dim)
        self.catconv_b = dsconv(dim * 2, dim)
        self.catconv = dsconv(dim * 2, dim)
        self.conv_a = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.conv_b = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.out_norm = nn.BatchNorm2d(dim)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        x_diff = xa - xb
        x_diff_a = self.catconv_a(torch.cat([x_diff, xa], dim=1))
        x_diff_b = self.catconv_b(torch.cat([x_diff, xb], dim=1))

        w_a = torch.sigmoid(self.conv_a(x_diff_a))
        w_b = torch.sigmoid(self.conv_b(x_diff_b))

        xa_w = w_a * xa
        xb_w = w_b * xb
        fused = self.catconv(torch.cat([xa_w, xb_w], dim=1))
        return F.relu(self.out_norm(fused), inplace=True)


class LSKDecoder(nn.Module):
    # from useful_module/dream_code_v3/2_5_LSKNet.py (adapted)
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(8, dim // 2)
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim, bias=False)
        self.conv0_s = nn.Conv2d(dim, mid, kernel_size=1, bias=False)
        self.conv1_s = nn.Conv2d(dim, mid, kernel_size=1, bias=False)
        self.conv_squeeze = nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=True)
        self.conv_m = nn.Conv2d(mid, dim, kernel_size=1, bias=False)
        self.out_norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn1 = self.conv0(x)
        attn2 = self.conv1(attn1)
        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn = torch.max(attn, dim=1, keepdim=True)[0]
        weights = torch.sigmoid(self.conv_squeeze(torch.cat([avg_attn, max_attn], dim=1)))
        fused = attn1 * weights[:, 0:1] + attn2 * weights[:, 1:2]
        fused = self.conv_m(fused)
        return F.relu(self.out_norm(x + x * fused), inplace=True)


class CoordAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, dim // reduction)
        self.conv1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.shape[2]
        w = x.shape[3]

        x_h = x.mean(dim=3, keepdim=True)  # [B, C, H, 1]
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)  # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


class SobelEdgeMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W], normalized input image
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.kx.to(dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.ky.to(dtype=gray.dtype), padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-7)
        return torch.clamp(mag, min=0.0, max=1.0)


class EGARefine(nn.Module):
    # inspired by useful_module/dream_code_v3/2_27_EGA.py (adapted)
    def __init__(self, dim: int) -> None:
        super().__init__()
        kx = torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

        self.edge_proj = ConvBNAct(1, dim, k=3, p=1)
        self.mix = ConvBNAct(dim, dim)
        self.refine = nn.Sequential(
            ConvBNAct(dim, dim),
            ConvBNAct(dim, dim, k=1, p=0),
        )
        self.out_norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.kx.to(dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.ky.to(dtype=gray.dtype), padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-7)
        edge = torch.clamp(edge, min=0.0, max=1.0)

        edge_feat = self.edge_proj(edge)
        gate = torch.sigmoid(torch.sum(self.mix(x) * edge_feat, dim=1, keepdim=True) / max(1, x.shape[1]))
        x_att = x + gate * edge_feat
        out = x + self.refine(x_att)
        return F.relu(self.out_norm(out), inplace=True)


class UncertaintyCrossGateRefine(nn.Module):
    """
    Cross-gate decoder refinement guided by boundary/uncertainty maps.
    """
    def __init__(self, dim: int, large_kernel: int = 7) -> None:
        super().__init__()
        k = max(3, int(large_kernel))
        if k % 2 == 0:
            k += 1
        gate_hidden = max(8, dim // 4)

        self.large_h = nn.Conv2d(dim, dim, kernel_size=(1, k), padding=(0, k // 2), groups=dim, bias=False)
        self.large_v = nn.Conv2d(dim, dim, kernel_size=(k, 1), padding=(k // 2, 0), groups=dim, bias=False)
        self.local_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        self.large_mix = ConvBNAct(dim, dim, k=1, p=0)
        self.local_mix = ConvBNAct(dim, dim, k=1, p=0)

        self.prior_gate = nn.Sequential(
            ConvBNAct(2, gate_hidden, k=1, p=0),
            nn.Conv2d(gate_hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(dim * 2, dim, k=1, p=0),
            ConvBNAct(dim, dim),
        )
        self.out_norm = nn.BatchNorm2d(dim)

    def forward(
        self,
        x: torch.Tensor,
        boundary_prob: torch.Tensor,
        uncertainty_prob: torch.Tensor,
    ) -> torch.Tensor:
        coarse = self.large_mix(self.large_v(self.large_h(x)))
        local = self.local_mix(self.local_dw(x))

        # Cross-gate the two branches by their residual discrepancy.
        coarse_gate = torch.sigmoid(x - local)
        local_gate = torch.sigmoid(x - coarse)

        prior = self.prior_gate(torch.cat([boundary_prob, uncertainty_prob], dim=1))
        coarse = coarse * coarse_gate * (1.0 + 0.5 * prior)
        local = local * local_gate * (1.0 + 0.5 * (1.0 - prior))

        refined = self.fuse(torch.cat([coarse, local], dim=1))
        return F.relu(self.out_norm(x + refined), inplace=True)


class UncertaintyGatedRefine(nn.Module):
    """
    Refine decoder feature using boundary/uncertainty-guided detail injection.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.detail_proj = ConvBNAct(dim, dim)
        self.gate = nn.Sequential(
            ConvBNAct(dim + 2, dim, k=1, p=0),
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = ConvBNAct(dim, dim)

    def forward(
        self,
        seg_feat: torch.Tensor,
        shallow_feat: torch.Tensor,
        boundary_prob: torch.Tensor,
        uncertainty_prob: torch.Tensor,
    ) -> torch.Tensor:
        detail = self.detail_proj(shallow_feat)
        detail = F.interpolate(detail, size=seg_feat.shape[-2:], mode="bilinear", align_corners=False)
        focus = torch.cat([boundary_prob, uncertainty_prob], dim=1)
        g = self.gate(torch.cat([detail, focus], dim=1))
        return seg_feat + self.refine(detail * g)


class BackgroundDecoupleHead(nn.Module):
    """
    Learn an explicit background suppression logit and subtract it from
    the foreground logit to reduce low-threshold false positives.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bg_branch = nn.Sequential(
            ConvBNAct(dim, dim),
            nn.Conv2d(dim, 1, kernel_size=1, bias=True),
        )
        self.scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, feat: torch.Tensor, fg_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bg_logit = self.bg_branch(feat)
        alpha = F.softplus(self.scale).to(dtype=fg_logit.dtype)
        return fg_logit - alpha * bg_logit, bg_logit


class UncertaintyLogitGate(nn.Module):
    """
    Suppress uncertain positives in logit space to improve low-threshold robustness.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(16, dim // 2)
        self.gate = nn.Sequential(
            ConvBNAct(dim + 1, mid, k=1, p=0),
            ConvBNAct(mid, mid),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        self.scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        uncertainty_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if uncertainty_logits is None:
            return logits, None
        u = torch.sigmoid(uncertainty_logits)
        gate = torch.sigmoid(self.gate(torch.cat([feat, u], dim=1)))
        beta = F.softplus(self.scale).to(dtype=logits.dtype)
        suppress = beta * gate * u
        return logits - suppress, suppress


class GlobalLogitCalibrator(nn.Module):
    """
    Per-image logit calibration to reduce over-segmentation under fixed low threshold.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(16, dim // 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, kernel_size=1, bias=True),
        )

    def forward(self, feat: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.mlp(self.pool(feat)).flatten(1)
        # keep calibration bounded for stable resume fine-tuning
        scale = 0.75 + 0.5 * torch.sigmoid(raw[:, 0:1])     # [0.75, 1.25]
        bias = 1.5 * torch.tanh(raw[:, 1:2])                # [-1.5, 1.5]
        scale_map = scale.view(-1, 1, 1, 1).to(dtype=logits.dtype)
        bias_map = bias.view(-1, 1, 1, 1).to(dtype=logits.dtype)
        return logits * scale_map + bias_map, scale_map, bias_map


class BoundaryUncertaintyLogitRefine(nn.Module):
    """
    Refine segmentation logits using boundary confidence and uncertainty maps.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(16, dim // 2)
        self.detail = nn.Sequential(
            nn.Conv2d(dim, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        self.prior_gate = nn.Sequential(
            ConvBNAct(2, mid, k=1, p=0),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.pos_scale = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.neg_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

    def forward(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        boundary_logits: torch.Tensor | None,
        uncertainty_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boundary_prob = (
            torch.sigmoid(boundary_logits)
            if boundary_logits is not None
            else torch.zeros_like(logits)
        )
        uncertainty_prob = (
            torch.sigmoid(uncertainty_logits)
            if uncertainty_logits is not None
            else torch.zeros_like(logits)
        )
        detail_strength = torch.sigmoid(self.detail(feat))
        refine_gate = self.prior_gate(torch.cat([boundary_prob, uncertainty_prob], dim=1))

        pos_focus = refine_gate * (1.0 - uncertainty_prob)
        neg_focus = (1.0 - refine_gate) * uncertainty_prob

        alpha = F.softplus(self.pos_scale).to(dtype=logits.dtype)
        beta = F.softplus(self.neg_scale).to(dtype=logits.dtype)
        delta = (alpha * pos_focus - beta * neg_focus) * detail_strength
        return logits + delta, delta, refine_gate


class ResidualLogitRefine(nn.Module):
    """
    Residual logit refinement head.
    Starts from identity behavior by zero-initializing the final projection.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(16, dim // 2)
        self.refine = nn.Sequential(
            ConvBNAct(dim, mid, k=3, p=1),
            ConvBNAct(mid, mid, k=3, p=1),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, feat: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.refine(feat).to(dtype=logits.dtype)
        return logits + delta, delta


class MambaLite2DBlock(nn.Module):
    """
    Lightweight 2D token mixer inspired by state-space style gated updates.
    """
    def __init__(self, dim: int, k_small: int = 3, k_large: int = 11) -> None:
        super().__init__()
        p_small = k_small // 2
        p_large = k_large // 2
        self.pre = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.dw_small = nn.Conv2d(dim, dim, kernel_size=k_small, padding=p_small, groups=dim, bias=False)
        self.dw_large = nn.Conv2d(dim, dim, kernel_size=k_large, padding=p_large, groups=dim, bias=False)
        self.mix_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        nn.init.zeros_(self.post[1].weight)
        nn.init.zeros_(self.post[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.pre(x).chunk(2, dim=1)
        u = self.dw_small(u)
        v = self.dw_large(v)
        g = self.mix_gate(torch.cat([u, v], dim=1))
        y = u * g + v * (1.0 - g)
        y = self.post(y)
        return x + y


class MambaLogitRefine(nn.Module):
    """
    Logit refinement head using stacked MambaLite2DBlock.
    Initialized as identity by zero-initialized output projection.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(24, dim // 2)
        self.stem = ConvBNAct(dim, mid, k=1, p=0)
        self.block1 = MambaLite2DBlock(mid, k_small=3, k_large=11)
        self.block2 = MambaLite2DBlock(mid, k_small=5, k_large=15)
        self.out = nn.Conv2d(mid, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(feat)
        x = self.block1(x)
        x = self.block2(x)
        delta = self.out(x).to(dtype=logits.dtype)
        return logits + delta, delta


class _ChannelAvgGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.pool(x))


class _PLKB2D(nn.Module):
    """
    Partial large-kernel block adapted from useful_module CGFN.
    """
    def __init__(self, dim: int, large_kernel: int = 11, split_group: int = 4) -> None:
        super().__init__()
        k = max(3, int(large_kernel))
        if k % 2 == 0:
            k += 1
        group = max(1, math.gcd(dim, max(1, int(split_group))))
        split = max(1, dim // group)
        split = min(split, dim - 1) if dim > 1 else 1
        self.group = group
        self.split = split
        self.ca = _ChannelAvgGate()
        self.dw_h = nn.Conv2d(split, split, kernel_size=(k, 1), padding=(k // 2, 0), groups=split, bias=False)
        self.dw_w = nn.Conv2d(split, split, kernel_size=(1, k), padding=(0, k // 2), groups=split, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def _channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.group
        if c % g != 0:
            return x
        x = x.reshape(b, g, c // g, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._channel_shuffle(x)
        x1 = y[:, : self.split]
        x2 = y[:, self.split :]
        x1 = self.ca(x1)
        x1 = self.dw_h(self.dw_w(x1))
        y = torch.cat([x1, x2], dim=1)
        return self.mix(y)


class _CGFN2DBlock(nn.Module):
    """
    Cross-gate feed-forward block adapted from useful_module CGFN.
    """
    def __init__(self, dim: int, large_kernel: int = 11, split_group: int = 4) -> None:
        super().__init__()
        self.a = _PLKB2D(dim, large_kernel=large_kernel, split_group=split_group)
        self.b = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.scale_a = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1))
        self.scale_b = nn.Parameter(1e-5 * torch.ones(1, dim, 1, 1))
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xa = self.a(x)
        xb = self.b(x)
        da = self.scale_a * (x - xa)
        db = self.scale_b * (x - xb)
        xa = xa * db
        xb = xb * da
        y = self.proj(torch.cat([xa, xb], dim=1))
        return x + y


class CGFNLogitRefine(nn.Module):
    """
    Logit refinement head using two lightweight CGFN-style blocks.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(24, dim // 2)
        self.stem = ConvBNAct(dim, mid, k=1, p=0)
        self.block1 = _CGFN2DBlock(mid, large_kernel=11, split_group=4)
        self.block2 = _CGFN2DBlock(mid, large_kernel=15, split_group=4)
        self.out = nn.Conv2d(mid, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(feat)
        x = self.block1(x)
        x = self.block2(x)
        delta = self.out(x).to(dtype=logits.dtype)
        return logits + delta, delta


class RecallRecoveryRefine(nn.Module):
    """
    Recall-oriented logit refinement.
    Uses boundary/uncertainty priors and conservative residual correction to
    recover hard positive regions while keeping identity-safe startup.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(16, dim // 2)
        self.gate = nn.Sequential(
            ConvBNAct(dim + 3, mid, k=1, p=0),
            ConvBNAct(mid, mid),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        self.delta_head = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        # Positive boost is stronger than negative suppression by design.
        self.boost_scale = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))
        self.suppress_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        boundary_logits: torch.Tensor | None,
        uncertainty_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        boundary_prob = (
            torch.sigmoid(boundary_logits)
            if boundary_logits is not None
            else torch.zeros_like(logits)
        )
        uncertainty_prob = (
            torch.sigmoid(uncertainty_logits)
            if uncertainty_logits is not None
            else torch.zeros_like(logits)
        )
        p = torch.sigmoid(logits)

        # Prior: uncertain edge-like low-confidence positives.
        recover_prior = boundary_prob * (0.5 + 0.5 * uncertainty_prob) * (1.0 - p)
        suppress_prior = (1.0 - boundary_prob) * (1.0 - uncertainty_prob) * p

        g = torch.sigmoid(self.gate(torch.cat([feat, boundary_prob, uncertainty_prob, p], dim=1)))
        residual_delta = self.delta_head(feat).to(dtype=logits.dtype)

        alpha = F.softplus(self.boost_scale).to(dtype=logits.dtype)
        beta = F.softplus(self.suppress_scale).to(dtype=logits.dtype)
        prior_delta = alpha * g * recover_prior - 0.25 * beta * (1.0 - g) * suppress_prior
        delta = residual_delta + prior_delta
        return logits + delta, delta, recover_prior, g


class RecallCandidateHead(nn.Module):
    """
    Candidate-positive branch for recall improvement.
    Produces a conservative positive-only delta in logit space.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(16, dim // 2)
        self.head = nn.Sequential(
            ConvBNAct(dim + 2, mid, k=1, p=0),
            ConvBNAct(mid, mid),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, -6.0)
        self.scale = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))

    def forward(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        boundary_logits: torch.Tensor | None,
        uncertainty_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        boundary_prob = (
            torch.sigmoid(boundary_logits)
            if boundary_logits is not None
            else torch.zeros_like(logits)
        )
        uncertainty_prob = (
            torch.sigmoid(uncertainty_logits)
            if uncertainty_logits is not None
            else torch.zeros_like(logits)
        )
        candidate_logits = self.head(torch.cat([feat, boundary_prob, uncertainty_prob], dim=1))
        candidate_prob = torch.sigmoid(candidate_logits)
        miss_prob = 1.0 - torch.sigmoid(logits)
        alpha = F.softplus(self.scale).to(dtype=logits.dtype)
        delta = alpha * candidate_prob * miss_prob
        return logits + delta, candidate_logits, delta, candidate_prob


class MultiScaleRecallHead(nn.Module):
    """
    Multi-scale recall branch with c2/fused16/seg interaction.
    Targets missed positives near uncertain boundaries.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(16, dim // 2)
        self.seg_proj = ConvBNAct(dim, mid, k=1, p=0)
        self.c2_proj = ConvBNAct(dim, mid, k=1, p=0)
        self.f16_proj = ConvBNAct(dim, mid, k=1, p=0)
        self.fuse = nn.Sequential(
            ConvBNAct(mid * 3 + 2, mid, k=1, p=0),
            ConvBNAct(mid, mid),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.constant_(self.fuse[-1].bias, -6.0)
        self.scale = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))

    def forward(
        self,
        seg_full: torch.Tensor,
        c2: torch.Tensor,
        fused_16: torch.Tensor,
        logits: torch.Tensor,
        boundary_logits: torch.Tensor | None,
        uncertainty_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_hw = logits.shape[-2:]
        c2_u = F.interpolate(c2, size=target_hw, mode="bilinear", align_corners=False)
        f16_u = F.interpolate(fused_16, size=target_hw, mode="bilinear", align_corners=False)
        b = (
            torch.sigmoid(boundary_logits)
            if boundary_logits is not None
            else torch.zeros_like(logits)
        )
        u = (
            torch.sigmoid(uncertainty_logits)
            if uncertainty_logits is not None
            else torch.zeros_like(logits)
        )
        p = torch.sigmoid(logits)
        x = torch.cat(
            [self.seg_proj(seg_full), self.c2_proj(c2_u), self.f16_proj(f16_u), b, u],
            dim=1,
        )
        candidate_logits = self.fuse(x)
        candidate_prob = torch.sigmoid(candidate_logits)
        miss_prob = 1.0 - p
        edge_focus = (0.5 + 0.5 * b) * (0.5 + 0.5 * u)
        alpha = F.softplus(self.scale).to(dtype=logits.dtype)
        delta = alpha * candidate_prob * miss_prob * edge_focus
        return logits + delta, candidate_logits, delta, candidate_prob


class ReliabilityAdaptiveBoundaryCalibration(nn.Module):
    """
    Reliability-adaptive boundary calibration in logit space.

    The module predicts a per-pixel diffusion coefficient and threshold-shift map
    conditioned on decoder features, boundary confidence, uncertainty, and the
    current segmentation probability. This keeps calibration local and
    image-adaptive rather than relying on a fixed global Gaussian kernel.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(16, dim // 2)
        self.context = nn.Sequential(
            ConvBNAct(dim + 3, mid, k=1, p=0),
            ConvBNAct(mid, mid),
            nn.Conv2d(mid, 3, kernel_size=1, bias=True),
        )
        with torch.no_grad():
            # Near-identity initialization: do not perturb pretrained logits
            # before the calibration branch has learned a useful correction.
            nn.init.zeros_(self.context[-1].weight)
            nn.init.zeros_(self.context[-1].bias)
            self.context[-1].bias[0] = -6.0
            self.context[-1].bias[2] = -5.0
        self.local_mixer = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.local_mixer.weight.fill_(1.0 / 9.0)
        self.residual_delta = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.residual_delta.weight)
        nn.init.zeros_(self.residual_delta.bias)
        self.tau_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.bg_scale = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))

    def forward(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        boundary_logits: torch.Tensor | None,
        uncertainty_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        boundary_prob = (
            torch.sigmoid(boundary_logits)
            if boundary_logits is not None
            else torch.zeros_like(logits)
        )
        uncertainty_prob = (
            torch.sigmoid(uncertainty_logits)
            if uncertainty_logits is not None
            else torch.zeros_like(logits)
        )
        p = torch.sigmoid(logits)
        candidate = boundary_prob * (0.35 + 0.65 * uncertainty_prob) * (1.0 - p)
        ctx = self.context(torch.cat([feat, boundary_prob, uncertainty_prob, p], dim=1))
        alpha_map = torch.sigmoid(ctx[:, 0:1]) * candidate
        tau_shift = torch.tanh(ctx[:, 1:2]) * F.softplus(self.tau_scale).to(dtype=logits.dtype) * candidate
        suppress_gate = torch.sigmoid(ctx[:, 2:3])

        local_logits = self.local_mixer(logits)
        pos_delta = alpha_map * (local_logits - logits) + tau_shift
        far_bg = (1.0 - boundary_prob) * (1.0 - uncertainty_prob) * p
        neg_delta = suppress_gate * F.softplus(self.bg_scale).to(dtype=logits.dtype) * far_bg
        residual_delta = self.residual_delta(feat).to(dtype=logits.dtype)
        delta = pos_delta - neg_delta + candidate * residual_delta
        return logits + delta, delta, alpha_map, tau_shift


class BoundaryRecallRecalibration(nn.Module):
    """
    Conservative boundary recall recalibration.

    A dedicated recall branch predicts candidate-positive regions around
    uncertain boundaries and applies a gated positive-only correction while
    preserving far-background stability.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        mid = max(16, dim // 2)
        self.candidate_head = nn.Sequential(
            ConvBNAct(dim + 3, mid, k=1, p=0),
            ConvBNAct(mid, mid),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        self.gate_head = nn.Sequential(
            ConvBNAct(dim + 3, mid, k=1, p=0),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        self.residual_head = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.candidate_head[-1].weight)
        nn.init.constant_(self.candidate_head[-1].bias, -6.0)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -4.0)
        self.boost_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))
        self.suppress_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
        boundary_logits: torch.Tensor | None,
        uncertainty_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        boundary_prob = (
            torch.sigmoid(boundary_logits)
            if boundary_logits is not None
            else torch.zeros_like(logits)
        )
        uncertainty_prob = (
            torch.sigmoid(uncertainty_logits)
            if uncertainty_logits is not None
            else torch.zeros_like(logits)
        )
        p = torch.sigmoid(logits)
        x = torch.cat([feat, boundary_prob, uncertainty_prob, p], dim=1)
        candidate_logits = self.candidate_head(x)
        candidate_prob = torch.sigmoid(candidate_logits)
        conservative_gate = torch.sigmoid(self.gate_head(x))
        # Restrict recall corrections to uncertain frontier neighborhoods
        # instead of broadly expanding all positives.
        frontier_band = torch.clamp(1.0 - torch.abs(p - 0.5) / 0.35, min=0.0, max=1.0)
        support_map = (
            candidate_prob
            * conservative_gate
            * boundary_prob
            * (0.25 + 0.75 * uncertainty_prob)
            * frontier_band
            * (1.0 - p)
        )
        residual_delta = torch.tanh(self.residual_head(feat)).to(dtype=logits.dtype)

        alpha = F.softplus(self.boost_scale).to(dtype=logits.dtype)
        beta = F.softplus(self.suppress_scale).to(dtype=logits.dtype)
        pos_delta = alpha * support_map + 0.25 * support_map * residual_delta
        neg_delta = beta * (1.0 - support_map) * (1.0 - boundary_prob) * (1.0 - uncertainty_prob) * p
        delta = pos_delta - neg_delta
        return logits + delta, candidate_logits, delta, support_map, conservative_gate


class FocalModulationContext(nn.Module):
    """
    Focal Modulation context module adapted from useful_module/dream_code_v3/2_9_FocalNet.py.

    Replaces stacked SE+ECA+CoordAttn with a single O(N) focal modulation block.
    Uses hierarchical DWConv layers with increasing kernel sizes for multi-scale
    context aggregation, gated per-level weighting, and elementwise modulation.

    Reference: Yang et al., "Focal Modulation Networks", NeurIPS 2022.
    """
    def __init__(self, dim: int, focal_window: int = 3, focal_level: int = 2, focal_factor: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.focal_level = focal_level

        # Projects input to query (dim), context (dim), and per-level gates (focal_level+1)
        self.f_proj = nn.Conv2d(dim, 2 * dim + focal_level + 1, kernel_size=1, bias=True)
        self.h_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.act = nn.GELU()

        self.focal_layers = nn.ModuleList()
        for k in range(focal_level):
            kernel_size = focal_factor * k + focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        residual = x
        y = self.f_proj(x)  # (B, 2C + L+1, H, W)
        q, ctx, gates = torch.split(y, (self.dim, self.dim, self.focal_level + 1), dim=1)

        # Hierarchical context aggregation with per-level gating
        ctx_all = torch.zeros_like(ctx)
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * gates[:, level : level + 1]

        # Global context (last level)
        ctx_global = self.act(ctx.mean(dim=(2, 3), keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level : self.focal_level + 1]

        # Focal modulation: query * modulator
        modulator = self.h_proj(ctx_all)
        x_out = q * modulator
        x_out = self.out_proj(x_out)
        return residual + x_out


class CGASkipFusion(nn.Module):
    """
    Channel-Guided Attention skip-connection fusion adapted from
    useful_module/dream_code_v3/2_12_DEANet.py (CGAFusion).

    Replaces simple element-wise addition of decoder features with shallow
    features (c2) by a learned pixel-level attention gate that adaptively
    blends deep semantic features with shallow boundary-detail features.

    Key benefit for SEN: preserves high-resolution boundary details from c2,
    directly reducing boundary under-segmentation (the main source of FN).

    Reference: DEANet (Detail-Enhanced Attention for image restoration).
    """
    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, dim // max(1, reduction))
        # Spatial attention branch
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, padding_mode='zeros', bias=True)
        # Channel attention branch
        self.ca_gap = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
        )
        # Pixel attention branch (combines spatial + channel cues)
        self.pa_conv = nn.Conv2d(dim * 2, dim, 7, padding=3, groups=dim, bias=True)
        self.conv_out = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: deep features (decoder seg features), (B, C, H, W)
        y: shallow features (c2 or detail), (B, C, H, W) — same spatial size as x
        """
        initial = x + y

        # Channel attention
        cattn = self.ca_fc(self.ca_gap(initial))  # (B, C, 1, 1)

        # Spatial attention
        x_avg = torch.mean(initial, dim=1, keepdim=True)
        x_max = torch.max(initial, dim=1, keepdim=True)[0]
        sattn = self.sa_conv(torch.cat([x_avg, x_max], dim=1))  # (B, 1, H, W)

        # Combined prior
        pattn1 = sattn + cattn  # broadcast: (B, C, H, W)

        # Pixel attention: concatenate initial and pattn1 along a pseudo-dim then group conv
        b, c, h, w = initial.shape
        pa_input = torch.cat([initial.unsqueeze(2), pattn1.unsqueeze(2)], dim=2)  # (B, C, 2, H, W)
        pa_input = pa_input.reshape(b, c * 2, h, w)  # (B, 2C, H, W)
        pattn2 = self.sigmoid(self.pa_conv(pa_input))  # (B, C, H, W)

        # Adaptive blending with residual
        result = initial + pattn2 * x + (1.0 - pattn2) * y
        return self.conv_out(result)


class RPINet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        backbone: str = "resnet50",
        backbone_pretrained: bool = True,
        use_boundary_head: bool = False,
        use_uncertainty_head: bool = False,
        refine_with_heads: bool = False,
        use_aspp: bool = False,
        use_aux_head: bool = False,
        use_se_context: bool = False,
        use_detail_fusion: bool = False,
        use_coord_attn: bool = False,
        use_edge_prompt: bool = False,
        use_dual_supervised_head: bool = False,
        use_eca_context: bool = False,
        use_ema_context: bool = False,
        use_cbam_decoder: bool = False,
        use_simam_refine: bool = False,
        use_safm_decoder: bool = False,
        use_uncertainty_gated_refine: bool = False,
        use_sk_context: bool = False,
        use_triplet_decoder: bool = False,
        use_msaa_fusion: bool = False,
        use_ega_refine: bool = False,
        use_cot_context: bool = False,
        use_lsk_decoder: bool = False,
        use_ucgf_refine: bool = False,
        use_anchor_context: bool = False,
        use_smfa_decoder: bool = False,
        use_tff_refine: bool = False,
        use_mfe_context: bool = False,
        use_bg_decouple_head: bool = False,
        use_uncertainty_logit_gate: bool = False,
        use_global_logit_calibrator: bool = False,
        use_boundary_uncertainty_logit_refine: bool = False,
        use_residual_logit_refine: bool = False,
        use_mamba_logit_refine: bool = False,
        use_cgfn_logit_refine: bool = False,
        use_recall_recover_refine: bool = False,
        use_recall_candidate_head: bool = False,
        use_multiscale_recall_head: bool = False,
        use_rabc: bool = False,
        use_brr: bool = False,
        use_focal_context: bool = False,
        use_cga_fusion: bool = False,
    ) -> None:
        super().__init__()

        self.use_boundary_head = bool(use_boundary_head)
        self.use_uncertainty_head = bool(use_uncertainty_head)
        self.refine_with_heads = bool(refine_with_heads) and (self.use_boundary_head or self.use_uncertainty_head)
        self.use_aspp = bool(use_aspp)
        self.use_aux_head = bool(use_aux_head)
        self.use_se_context = bool(use_se_context)
        self.use_detail_fusion = bool(use_detail_fusion)
        self.use_coord_attn = bool(use_coord_attn)
        self.use_edge_prompt = bool(use_edge_prompt)
        self.use_dual_supervised_head = bool(use_dual_supervised_head)
        self.use_eca_context = bool(use_eca_context)
        self.use_ema_context = bool(use_ema_context)
        self.use_cbam_decoder = bool(use_cbam_decoder)
        self.use_simam_refine = bool(use_simam_refine)
        self.use_safm_decoder = bool(use_safm_decoder)
        self.use_uncertainty_gated_refine = bool(use_uncertainty_gated_refine)
        self.use_sk_context = bool(use_sk_context)
        self.use_triplet_decoder = bool(use_triplet_decoder)
        self.use_msaa_fusion = bool(use_msaa_fusion)
        self.use_ega_refine = bool(use_ega_refine)
        self.use_cot_context = bool(use_cot_context)
        self.use_lsk_decoder = bool(use_lsk_decoder)
        self.use_ucgf_refine = bool(use_ucgf_refine)
        self.use_anchor_context = bool(use_anchor_context)
        self.use_smfa_decoder = bool(use_smfa_decoder)
        self.use_tff_refine = bool(use_tff_refine)
        self.use_mfe_context = bool(use_mfe_context)
        self.use_bg_decouple_head = bool(use_bg_decouple_head)
        self.use_uncertainty_logit_gate = bool(use_uncertainty_logit_gate)
        self.use_global_logit_calibrator = bool(use_global_logit_calibrator)
        self.use_boundary_uncertainty_logit_refine = bool(use_boundary_uncertainty_logit_refine)
        self.use_residual_logit_refine = bool(use_residual_logit_refine)
        self.use_mamba_logit_refine = bool(use_mamba_logit_refine)
        self.use_cgfn_logit_refine = bool(use_cgfn_logit_refine)
        self.use_recall_recover_refine = bool(use_recall_recover_refine)
        self.use_recall_candidate_head = bool(use_recall_candidate_head)
        self.use_multiscale_recall_head = bool(use_multiscale_recall_head)
        self.use_rabc = bool(use_rabc)
        self.use_brr = bool(use_brr)
        self.use_focal_context = bool(use_focal_context)
        self.use_cga_fusion = bool(use_cga_fusion)

        self.backbone_name = backbone
        self.encoder, c2_in, c3_in, c4_in, self.backbone_channel_last = self._build_backbone(
            backbone=backbone,
            pretrained=backbone_pretrained,
        )

        self.c2_proj = ConvBNAct(c2_in, hidden_dim, k=1, p=0)
        self.c2_down = ConvBNAct(hidden_dim, hidden_dim, s=2)  # /8 -> /16
        self.c3_proj = ConvBNAct(c3_in, hidden_dim, k=1, p=0)
        self.c4_proj = ConvBNAct(c4_in, hidden_dim, k=1, p=0)

        self.fuse = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim),
            nn.Dropout2d(dropout),
            ConvBNAct(hidden_dim, hidden_dim),
        )
        self.msaa_fusion = MSAAFusion(hidden_dim, factor=4) if self.use_msaa_fusion else None
        self.context = ASPPContext(hidden_dim, dropout=dropout) if self.use_aspp else nn.Identity()
        self.mfe_context = MFEContext(hidden_dim, atrous_rates=(2, 4, 8)) if self.use_mfe_context else nn.Identity()
        self.sk_context = SKContext(hidden_dim) if self.use_sk_context else nn.Identity()
        self.se_context = SEBlock(hidden_dim) if self.use_se_context else nn.Identity()
        self.coord_attn = CoordAttention(hidden_dim) if self.use_coord_attn else nn.Identity()
        self.eca_context = ECAAttention(kernel_size=3) if self.use_eca_context else nn.Identity()
        self.ema_context = EMAContext(hidden_dim, factor=32) if self.use_ema_context else nn.Identity()
        self.cot_context = CoTContext(hidden_dim, kernel_size=3, factor=4, groups=4) if self.use_cot_context else nn.Identity()
        self.anchor_context = (
            AnchorContext(hidden_dim, anchors=8, heads=4, dim_head=max(8, hidden_dim // 4), dropout=dropout)
            if self.use_anchor_context
            else nn.Identity()
        )
        self.focal_context = (
            FocalModulationContext(hidden_dim, focal_window=3, focal_level=2, focal_factor=2)
            if self.use_focal_context
            else nn.Identity()
        )

        if self.use_edge_prompt:
            self.edge_extractor = SobelEdgeMap()
            self.edge_proj = ConvBNAct(1, hidden_dim)
            self.edge_gate = nn.Sequential(
                ConvBNAct(hidden_dim * 2, hidden_dim, k=1, p=0),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.edge_extractor = None
            self.edge_proj = None
            self.edge_gate = None

        if self.use_detail_fusion:
            self.detail_proj = ConvBNAct(hidden_dim, hidden_dim)
            self.detail_gate = nn.Sequential(
                ConvBNAct(hidden_dim * 2, hidden_dim, k=1, p=0),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            self.detail_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        else:
            self.detail_proj = None
            self.detail_gate = None
            self.detail_head = None
        self.tff_refine = TFFRefine(hidden_dim) if self.use_tff_refine else None
        self.cga_skip_fusion = CGASkipFusion(hidden_dim, reduction=8) if self.use_cga_fusion else None

        self.seg_refine = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim),
            ConvBNAct(hidden_dim, hidden_dim),
        )
        self.cbam_decoder = CBAMBlock(hidden_dim, reduction=16, kernel_size=7) if self.use_cbam_decoder else nn.Identity()
        self.simam_refine = SimAMAttention(epsilon=1e-4) if self.use_simam_refine else nn.Identity()
        self.safm_decoder = SAFMDecoder(hidden_dim, n_levels=4) if self.use_safm_decoder else nn.Identity()
        self.ega_refine = EGARefine(hidden_dim) if self.use_ega_refine else nn.Identity()
        self.triplet_decoder = TripletAttention(no_spatial=False, kernel_size=7) if self.use_triplet_decoder else nn.Identity()
        self.lsk_decoder = LSKDecoder(hidden_dim) if self.use_lsk_decoder else nn.Identity()
        self.smfa_decoder = SMFADecoder(hidden_dim, growth_rate=2.0, down_scale=8) if self.use_smfa_decoder else nn.Identity()
        self.seg_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.bg_decouple_head = BackgroundDecoupleHead(hidden_dim) if self.use_bg_decouple_head else None
        self.uncertainty_logit_gate = UncertaintyLogitGate(hidden_dim) if self.use_uncertainty_logit_gate else None
        self.global_logit_calibrator = GlobalLogitCalibrator(hidden_dim) if self.use_global_logit_calibrator else None
        self.boundary_uncertainty_logit_refine = (
            BoundaryUncertaintyLogitRefine(hidden_dim) if self.use_boundary_uncertainty_logit_refine else None
        )
        self.residual_logit_refine = ResidualLogitRefine(hidden_dim) if self.use_residual_logit_refine else None
        self.mamba_logit_refine = MambaLogitRefine(hidden_dim) if self.use_mamba_logit_refine else None
        self.cgfn_logit_refine = CGFNLogitRefine(hidden_dim) if self.use_cgfn_logit_refine else None
        self.recall_recover_refine = RecallRecoveryRefine(hidden_dim) if self.use_recall_recover_refine else None
        self.recall_candidate_head = RecallCandidateHead(hidden_dim) if self.use_recall_candidate_head else None
        self.multiscale_recall_head = MultiScaleRecallHead(hidden_dim) if self.use_multiscale_recall_head else None
        self.rabc = ReliabilityAdaptiveBoundaryCalibration(hidden_dim) if self.use_rabc else None
        self.brr = BoundaryRecallRecalibration(hidden_dim) if self.use_brr else None
        if self.use_dual_supervised_head:
            self.supervised_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
            self.dual_head_gate = nn.Conv2d(2, 1, kernel_size=1)
            # Conservative startup: keep blending close to the train head initially.
            nn.init.zeros_(self.dual_head_gate.weight)
            nn.init.constant_(self.dual_head_gate.bias, -4.0)
        else:
            self.supervised_head = None
            self.dual_head_gate = None
        self.aux_head = nn.Conv2d(hidden_dim, 1, kernel_size=1) if self.use_aux_head else None

        if self.use_boundary_head:
            self.boundary_fuse = nn.Sequential(
                ConvBNAct(hidden_dim * 2, hidden_dim),
                ConvBNAct(hidden_dim, hidden_dim),
            )
            self.boundary_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        else:
            self.boundary_fuse = None
            self.boundary_head = None

        if self.use_uncertainty_head:
            self.uncertainty_head = nn.Sequential(
                ConvBNAct(hidden_dim, hidden_dim),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
        else:
            self.uncertainty_head = None

        if self.refine_with_heads:
            self.head_guided_refine = nn.Sequential(
                ConvBNAct(hidden_dim + 2, hidden_dim),
                ConvBNAct(hidden_dim, hidden_dim),
            )
        else:
            self.head_guided_refine = None
        self.uncertainty_gated_refine = (
            UncertaintyGatedRefine(hidden_dim) if self.use_uncertainty_gated_refine else None
        )
        self.ucgf_refine = UncertaintyCrossGateRefine(hidden_dim, large_kernel=7) if self.use_ucgf_refine else None

        self.pseudo_encoder = PseudoEncoder(hidden_dim)
        self.ipc_attn = ScaledDotAttention(hidden_dim, dropout=dropout)
        self.pia_attn = ScaledDotAttention(hidden_dim, dropout=dropout)

        self.sigma_heads = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(hidden_dim, hidden_dim),
                    nn.Conv2d(hidden_dim, 1, kernel_size=1),
                )
                for _ in range(4)
            ]
        )

    @staticmethod
    def _build_backbone(
        backbone: str,
        pretrained: bool,
    ) -> tuple[nn.Module, int, int, int, bool]:
        name = backbone.lower()
        if name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            bb = resnet50(weights=weights)
            encoder = create_feature_extractor(
                bb,
                return_nodes={
                    "layer2": "c2",  # /8
                    "layer3": "c3",  # /16
                    "layer4": "c4",  # /32
                },
            )
            return encoder, 512, 1024, 2048, False

        if name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            bb = convnext_tiny(weights=weights)
            encoder = create_feature_extractor(
                bb,
                return_nodes={
                    "features.3": "c2",  # /8
                    "features.5": "c3",  # /16
                    "features.7": "c4",  # /32
                },
            )
            return encoder, 192, 384, 768, False

        if name == "convnext_small":
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            bb = convnext_small(weights=weights)
            encoder = create_feature_extractor(
                bb,
                return_nodes={
                    "features.3": "c2",  # /8
                    "features.5": "c3",  # /16
                    "features.7": "c4",  # /32
                },
            )
            return encoder, 192, 384, 768, False

        if name == "swin_t":
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            bb = swin_t(weights=weights)
            encoder = create_feature_extractor(
                bb,
                return_nodes={
                    "features.3": "c2",  # /8, BHWC
                    "features.5": "c3",  # /16, BHWC
                    "features.7": "c4",  # /32, BHWC
                },
            )
            return encoder, 192, 384, 768, True

        raise ValueError(
            f"Unsupported model.backbone={backbone}. "
            "Choose resnet50/convnext_tiny/convnext_small/swin_t"
        )

    def _image_features(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        feats = self.encoder(image)
        c2_in = feats["c2"]
        c3_in = feats["c3"]
        c4_in = feats["c4"]
        if self.backbone_channel_last:
            c2_in = c2_in.permute(0, 3, 1, 2).contiguous()
            c3_in = c3_in.permute(0, 3, 1, 2).contiguous()
            c4_in = c4_in.permute(0, 3, 1, 2).contiguous()

        c2 = self.c2_proj(c2_in)
        c3 = self.c3_proj(c3_in)
        c4 = self.c4_proj(c4_in)

        c2_d = self.c2_down(c2)
        c4_u = F.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        fused_input = self.msaa_fusion(c2_d, c3, c4_u) if self.msaa_fusion is not None else (c2_d + c3 + c4_u)
        fused_16 = self.fuse(fused_input)
        fused_16 = self.context(fused_16)
        fused_16 = self.mfe_context(fused_16)
        fused_16 = self.sk_context(fused_16)
        fused_16 = self.se_context(fused_16)
        fused_16 = self.coord_attn(fused_16)
        fused_16 = self.eca_context(fused_16)
        fused_16 = self.ema_context(fused_16)
        fused_16 = self.cot_context(fused_16)
        fused_16 = self.anchor_context(fused_16)
        fused_16 = self.focal_context(fused_16)

        seg_8 = F.interpolate(fused_16, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        if (
            self.use_edge_prompt
            and self.edge_extractor is not None
            and self.edge_proj is not None
            and self.edge_gate is not None
        ):
            edge = self.edge_extractor(image)
            edge = F.interpolate(edge, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            edge_feat = self.edge_proj(edge)
            edge_gate = self.edge_gate(torch.cat([seg_8, edge_feat], dim=1))
            seg_8 = seg_8 + edge_gate * edge_feat

        detail_logits: torch.Tensor | None = None
        if (
            self.use_detail_fusion
            and self.detail_proj is not None
            and self.detail_gate is not None
            and self.detail_head is not None
        ):
            detail_feat = self.detail_proj(c2)
            gate = self.detail_gate(torch.cat([seg_8, detail_feat], dim=1))
            seg_8 = seg_8 + gate * detail_feat
            detail_logits = self.detail_head(F.interpolate(detail_feat, scale_factor=8, mode="bilinear", align_corners=False))
        else:
            detail_feat = c2

        if self.tff_refine is not None:
            seg_8 = seg_8 + self.tff_refine(seg_8, detail_feat)

        # CGA-guided skip fusion (preserves boundary details from c2)
        if self.cga_skip_fusion is not None:
            seg = self.seg_refine(self.cga_skip_fusion(seg_8, c2))
        else:
            seg = self.seg_refine(seg_8 + c2)
        seg = self.cbam_decoder(seg)
        seg = self.simam_refine(seg)
        seg = self.safm_decoder(seg)
        seg = self.ega_refine(seg)
        seg = self.triplet_decoder(seg)
        seg = self.lsk_decoder(seg)
        seg = self.smfa_decoder(seg)
        seg = F.interpolate(seg, scale_factor=8, mode="bilinear", align_corners=False)

        aux_logits: torch.Tensor | None = None
        if self.use_aux_head and self.aux_head is not None:
            aux_logits = self.aux_head(
                F.interpolate(
                    fused_16,
                    size=seg.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )

        boundary_logits: torch.Tensor | None = None
        if self.use_boundary_head and self.boundary_fuse is not None and self.boundary_head is not None:
            boundary_in = torch.cat(
                [c2, F.interpolate(fused_16, size=c2.shape[-2:], mode="bilinear", align_corners=False)],
                dim=1,
            )
            boundary_feat = self.boundary_fuse(boundary_in)
            boundary_logits = self.boundary_head(
                F.interpolate(boundary_feat, scale_factor=8, mode="bilinear", align_corners=False)
            )

        uncertainty_logits: torch.Tensor | None = None
        if self.use_uncertainty_head and self.uncertainty_head is not None:
            uncertainty_logits = self.uncertainty_head(fused_16)
            uncertainty_logits = F.interpolate(
                uncertainty_logits,
                size=seg.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.uncertainty_gated_refine is not None:
            boundary_prob_ugr = (
                torch.sigmoid(boundary_logits)
                if boundary_logits is not None
                else torch.zeros_like(seg[:, :1])
            )
            uncertainty_prob_ugr = (
                torch.sigmoid(uncertainty_logits)
                if uncertainty_logits is not None
                else torch.zeros_like(seg[:, :1])
            )
            seg = self.uncertainty_gated_refine(
                seg_feat=seg,
                shallow_feat=c2,
                boundary_prob=boundary_prob_ugr,
                uncertainty_prob=uncertainty_prob_ugr,
            )

        if self.refine_with_heads and self.head_guided_refine is not None:
            boundary_prob = (
                torch.sigmoid(boundary_logits)
                if boundary_logits is not None
                else torch.zeros_like(seg[:, :1])
            )
            uncertainty_prob = (
                torch.sigmoid(uncertainty_logits)
                if uncertainty_logits is not None
                else torch.zeros_like(seg[:, :1])
            )
            gate = torch.cat([boundary_prob, uncertainty_prob], dim=1)
            seg = seg + self.head_guided_refine(torch.cat([seg, gate], dim=1))

        if self.ucgf_refine is not None:
            boundary_prob = (
                torch.sigmoid(boundary_logits)
                if boundary_logits is not None
                else torch.zeros_like(seg[:, :1])
            )
            uncertainty_prob = (
                torch.sigmoid(uncertainty_logits)
                if uncertainty_logits is not None
                else torch.zeros_like(seg[:, :1])
            )
            seg = self.ucgf_refine(seg, boundary_prob=boundary_prob, uncertainty_prob=uncertainty_prob)

        fg_logits = self.seg_head(seg)
        bg_logits: torch.Tensor | None = None
        if self.bg_decouple_head is not None:
            train_logits, bg_logits = self.bg_decouple_head(seg, fg_logits)
        else:
            train_logits = fg_logits
        uncertainty_suppress_map: torch.Tensor | None = None
        if self.uncertainty_logit_gate is not None:
            train_logits, uncertainty_suppress_map = self.uncertainty_logit_gate(
                seg,
                train_logits,
                uncertainty_logits,
            )
        boundary_uncertainty_delta: torch.Tensor | None = None
        boundary_uncertainty_refine_gate: torch.Tensor | None = None
        if self.boundary_uncertainty_logit_refine is not None:
            train_logits, boundary_uncertainty_delta, boundary_uncertainty_refine_gate = (
                self.boundary_uncertainty_logit_refine(
                    seg,
                    train_logits,
                    boundary_logits,
                    uncertainty_logits,
                )
            )
        calibration_scale: torch.Tensor | None = None
        calibration_bias: torch.Tensor | None = None
        if self.global_logit_calibrator is not None:
            train_logits, calibration_scale, calibration_bias = self.global_logit_calibrator(seg, train_logits)
        residual_logit_delta: torch.Tensor | None = None
        if self.residual_logit_refine is not None:
            train_logits, residual_logit_delta = self.residual_logit_refine(seg, train_logits)
        mamba_logit_delta: torch.Tensor | None = None
        if self.mamba_logit_refine is not None:
            train_logits, mamba_logit_delta = self.mamba_logit_refine(seg, train_logits)
        cgfn_logit_delta: torch.Tensor | None = None
        if self.cgfn_logit_refine is not None:
            train_logits, cgfn_logit_delta = self.cgfn_logit_refine(seg, train_logits)
        recall_recover_delta: torch.Tensor | None = None
        recall_recover_prior: torch.Tensor | None = None
        recall_recover_gate: torch.Tensor | None = None
        if self.recall_recover_refine is not None:
            train_logits, recall_recover_delta, recall_recover_prior, recall_recover_gate = (
                self.recall_recover_refine(
                    seg,
                    train_logits,
                    boundary_logits,
                    uncertainty_logits,
                )
            )
        recall_candidate_logits: torch.Tensor | None = None
        recall_candidate_delta: torch.Tensor | None = None
        recall_candidate_prob: torch.Tensor | None = None
        if self.recall_candidate_head is not None:
            train_logits, recall_candidate_logits, recall_candidate_delta, recall_candidate_prob = (
                self.recall_candidate_head(
                    seg,
                    train_logits,
                    boundary_logits,
                    uncertainty_logits,
                )
            )
        multiscale_recall_logits: torch.Tensor | None = None
        multiscale_recall_delta: torch.Tensor | None = None
        multiscale_recall_prob: torch.Tensor | None = None
        if self.multiscale_recall_head is not None:
            train_logits, multiscale_recall_logits, multiscale_recall_delta, multiscale_recall_prob = (
                self.multiscale_recall_head(
                    seg,
                    c2,
                    fused_16,
                    train_logits,
                    boundary_logits,
                    uncertainty_logits,
                )
            )
        base_train_logits = train_logits
        rabc_delta: torch.Tensor | None = None
        rabc_alpha_map: torch.Tensor | None = None
        rabc_tau_shift: torch.Tensor | None = None
        if self.rabc is not None:
            train_logits, rabc_delta, rabc_alpha_map, rabc_tau_shift = self.rabc(
                seg,
                train_logits,
                boundary_logits,
                uncertainty_logits,
            )
        brr_candidate_logits: torch.Tensor | None = None
        brr_delta: torch.Tensor | None = None
        brr_support_map: torch.Tensor | None = None
        brr_gate: torch.Tensor | None = None
        if self.brr is not None:
            train_logits, brr_candidate_logits, brr_delta, brr_support_map, brr_gate = self.brr(
                seg,
                train_logits,
                boundary_logits,
                uncertainty_logits,
            )
        supervised_logits: torch.Tensor | None = None
        logits = train_logits
        if (
            self.use_dual_supervised_head
            and self.supervised_head is not None
            and self.dual_head_gate is not None
        ):
            supervised_logits = self.supervised_head(seg)
            blend_gate = torch.sigmoid(self.dual_head_gate(torch.cat([train_logits, supervised_logits], dim=1)))
            logits = (1.0 - blend_gate) * train_logits + blend_gate * supervised_logits

        return fused_16, {
            "logits": logits,
            "train_logits": train_logits,
            "fg_logits": fg_logits,
            "bg_logits": bg_logits,
            "uncertainty_suppress_map": uncertainty_suppress_map,
            "boundary_uncertainty_delta": boundary_uncertainty_delta,
            "boundary_uncertainty_refine_gate": boundary_uncertainty_refine_gate,
            "calibration_scale": calibration_scale,
            "calibration_bias": calibration_bias,
            "residual_logit_delta": residual_logit_delta,
            "mamba_logit_delta": mamba_logit_delta,
            "cgfn_logit_delta": cgfn_logit_delta,
            "recall_recover_delta": recall_recover_delta,
            "recall_recover_prior": recall_recover_prior,
            "recall_recover_gate": recall_recover_gate,
            "recall_candidate_logits": recall_candidate_logits,
            "recall_candidate_delta": recall_candidate_delta,
            "recall_candidate_prob": recall_candidate_prob,
            "multiscale_recall_logits": multiscale_recall_logits,
            "multiscale_recall_delta": multiscale_recall_delta,
            "multiscale_recall_prob": multiscale_recall_prob,
            "base_train_logits": base_train_logits,
            "rabc_delta": rabc_delta,
            "rabc_alpha_map": rabc_alpha_map,
            "rabc_tau_shift": rabc_tau_shift,
            "brr_candidate_logits": brr_candidate_logits,
            "brr_delta": brr_delta,
            "brr_support_map": brr_support_map,
            "brr_gate": brr_gate,
            "supervised_logits": supervised_logits,
            "aux_logits": aux_logits,
            "detail_logits": detail_logits,
            "boundary_logits": boundary_logits,
            "uncertainty_logits": uncertainty_logits,
        }

    def _interaction(self, image_feat: torch.Tensor, pseudo: torch.Tensor) -> List[torch.Tensor]:
        # pseudo: [B, 4, H, W]
        pseudo_feats = []
        for i in range(pseudo.shape[1]):
            p = pseudo[:, i : i + 1]
            pseudo_feats.append(self.pseudo_encoder(p))

        ipc_feats = [self.ipc_attn(image_feat, pfeat) for pfeat in pseudo_feats]

        pia_feats = []
        for i in range(len(pseudo_feats)):
            agg = ipc_feats[i]
            for j in range(len(pseudo_feats)):
                if i == j:
                    continue
                agg = agg + self.pia_attn(pseudo_feats[i], pseudo_feats[j])
            pia_feats.append(agg)

        return pia_feats

    def forward(
        self,
        image: torch.Tensor,
        pseudo: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | List[torch.Tensor] | None]:
        image_feat, image_outputs = self._image_features(image)
        outputs: dict[str, torch.Tensor | List[torch.Tensor] | None] = {
            "logits": image_outputs["logits"],
            "train_logits": image_outputs["train_logits"],
            "fg_logits": image_outputs["fg_logits"],
            "bg_logits": image_outputs["bg_logits"],
            "uncertainty_suppress_map": image_outputs["uncertainty_suppress_map"],
            "boundary_uncertainty_delta": image_outputs["boundary_uncertainty_delta"],
            "boundary_uncertainty_refine_gate": image_outputs["boundary_uncertainty_refine_gate"],
            "calibration_scale": image_outputs["calibration_scale"],
            "calibration_bias": image_outputs["calibration_bias"],
            "residual_logit_delta": image_outputs["residual_logit_delta"],
            "mamba_logit_delta": image_outputs["mamba_logit_delta"],
            "cgfn_logit_delta": image_outputs["cgfn_logit_delta"],
            "recall_recover_delta": image_outputs["recall_recover_delta"],
            "recall_recover_prior": image_outputs["recall_recover_prior"],
            "recall_recover_gate": image_outputs["recall_recover_gate"],
            "recall_candidate_logits": image_outputs["recall_candidate_logits"],
            "recall_candidate_delta": image_outputs["recall_candidate_delta"],
            "recall_candidate_prob": image_outputs["recall_candidate_prob"],
            "multiscale_recall_logits": image_outputs["multiscale_recall_logits"],
            "multiscale_recall_delta": image_outputs["multiscale_recall_delta"],
            "multiscale_recall_prob": image_outputs["multiscale_recall_prob"],
            "base_train_logits": image_outputs["base_train_logits"],
            "rabc_delta": image_outputs["rabc_delta"],
            "rabc_alpha_map": image_outputs["rabc_alpha_map"],
            "rabc_tau_shift": image_outputs["rabc_tau_shift"],
            "brr_candidate_logits": image_outputs["brr_candidate_logits"],
            "brr_delta": image_outputs["brr_delta"],
            "brr_support_map": image_outputs["brr_support_map"],
            "brr_gate": image_outputs["brr_gate"],
            "supervised_logits": image_outputs["supervised_logits"],
            "aux_logits": image_outputs["aux_logits"],
            "detail_logits": image_outputs["detail_logits"],
            "boundary_logits": image_outputs["boundary_logits"],
            "uncertainty_logits": image_outputs["uncertainty_logits"],
            "sigma_maps": [],
        }
        logits = image_outputs["logits"]
        if logits is None:
            raise RuntimeError("RPINet forward produced empty logits")

        if pseudo is None or pseudo.numel() == 0:
            return outputs

        pia_feats = self._interaction(image_feat, pseudo)
        sigma_maps = []
        for i, feat in enumerate(pia_feats):
            sigma = self.sigma_heads[i](feat)
            sigma = F.interpolate(sigma, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            sigma_maps.append(sigma)

        outputs["sigma_maps"] = sigma_maps
        return outputs
