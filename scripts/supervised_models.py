#!/usr/bin/env python3
"""
supervised_models.py
六个监督分割基线模型的网络结构定义：
  1. UNet
  2. UNetPlusPlus
  3. AttDeepLabv3Plus
  4. TransFuse
  5. FATNet
  6. MALUNet
全部使用 torchvision / timm 提供的预训练骨干，
输出接口统一为 forward(x) -> {"logits": Tensor[B,1,H,W]}
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet34, ResNet34_Weights,
)

# ═══════════════════════════════════════════════════════════════════════
# 公共模块
# ═══════════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dilation=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )


def _up(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


# ═══════════════════════════════════════════════════════════════════════
# 1. U-Net  (Ronneberger 2015, MICCAI)
#    骨干: ResNet-34 encoder（替代原始 vanilla encoder）
# ═══════════════════════════════════════════════════════════════════════

class UNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.e0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # /2, 64
        self.pool = enc.maxpool                                   # /4
        self.e1 = enc.layer1   # /4,  64
        self.e2 = enc.layer2   # /8,  128
        self.e3 = enc.layer3   # /16, 256
        self.e4 = enc.layer4   # /32, 512

        self.d4 = DoubleConv(512 + 256, 256)
        self.d3 = DoubleConv(256 + 128, 128)
        self.d2 = DoubleConv(128 +  64, 64)
        self.d1 = DoubleConv( 64 +  64, 64)
        self.d0 = DoubleConv( 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        s0 = self.e0(x)          # /2
        s1 = self.e1(self.pool(s0))  # /4
        s2 = self.e2(s1)         # /8
        s3 = self.e3(s2)         # /16
        s4 = self.e4(s3)         # /32

        d = self.d4(torch.cat([_up(s4, s3), s3], 1))
        d = self.d3(torch.cat([_up(d,  s2), s2], 1))
        d = self.d2(torch.cat([_up(d,  s1), s1], 1))
        d = self.d1(torch.cat([_up(d,  s0), s0], 1))
        d = self.d0(F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False))
        return {"logits": self.head(d)}


# ═══════════════════════════════════════════════════════════════════════
# 2. U-Net++  (Zhou 2018, DLMIA)
#    Dense nested skip connections on the same ResNet-34 encoder
# ═══════════════════════════════════════════════════════════════════════

class _DenseConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, *xs):
        return self.conv(torch.cat(xs, dim=1))


class UNetPlusPlus(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.e0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.e1 = enc.layer1   # 64
        self.e2 = enc.layer2   # 128
        self.e3 = enc.layer3   # 256
        self.e4 = enc.layer4   # 512

        ch = [64, 64, 128, 256, 512]
        nb = 32  # node base channel

        # nodes x^{i,j}:  row i (depth), col j (node index from encoder)
        self.n01 = _DenseConv(ch[0]+ch[1], nb*2)
        self.n11 = _DenseConv(ch[1]+ch[2], nb*4)
        self.n21 = _DenseConv(ch[2]+ch[3], nb*8)
        self.n31 = _DenseConv(ch[3]+ch[4], nb*8)

        self.n02 = _DenseConv(ch[0]+nb*2+nb*4, nb*2)
        self.n12 = _DenseConv(ch[1]+nb*4+nb*8, nb*4)
        self.n22 = _DenseConv(ch[2]+nb*8+nb*8, nb*8)

        self.n03 = _DenseConv(ch[0]+nb*2+nb*2+nb*4, nb*2)
        self.n13 = _DenseConv(ch[1]+nb*4+nb*4+nb*8, nb*4)

        self.n04 = _DenseConv(ch[0]+nb*2+nb*2+nb*2+nb*4, nb*2)

        self.pre = ConvBnRelu(nb*2, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        x0 = self.e0(x)
        x1 = self.e1(self.pool(x0))
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)

        n01 = self.n01(x0, _up(x1, x0))
        n11 = self.n11(x1, _up(x2, x1))
        n21 = self.n21(x2, _up(x3, x2))
        n31 = self.n31(x3, _up(x4, x3))

        n02 = self.n02(x0, n01, _up(n11, x0))
        n12 = self.n12(x1, n11, _up(n21, x1))
        n22 = self.n22(x2, n21, _up(n31, x2))

        n03 = self.n03(x0, n01, n02, _up(n12, x0))
        n13 = self.n13(x1, n11, n12, _up(n22, x1))

        n04 = self.n04(x0, n01, n02, n03, _up(n13, x0))

        out = self.pre(F.interpolate(n04, (H, W), mode="bilinear", align_corners=False))
        return {"logits": self.head(out)}


# ═══════════════════════════════════════════════════════════════════════
# 3. Att-DeepLabv3+  (皮肤分割常用变体)
#    ResNet-50 ASPP + attention gate skip
# ═══════════════════════════════════════════════════════════════════════

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        self.b0 = ConvBnRelu(in_ch, out_ch, 1, p=0)
        self.bs = nn.ModuleList([
            ConvBnRelu(in_ch, out_ch, 3, p=r, dilation=r) for r in rates
        ])
        self.gp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.proj = ConvBnRelu((len(rates)+2)*out_ch, out_ch, 1, p=0)

    def forward(self, x):
        gp = F.interpolate(self.gp(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([self.b0(x)] + [b(x) for b in self.bs] + [gp], 1))


class AttGate(nn.Module):
    def __init__(self, g_ch, x_ch, mid_ch):
        super().__init__()
        self.Wg = nn.Conv2d(g_ch, mid_ch, 1, bias=False)
        self.Wx = nn.Conv2d(x_ch, mid_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(mid_ch, 1, 1, bias=False),
                                  nn.Sigmoid())

    def forward(self, g, x):
        a = self.psi(F.relu(self.Wg(_up(g, x)) + self.Wx(x), inplace=True))
        return x * a


class AttDeepLabv3Plus(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        # 将 layer3/4 改为空洞卷积（output stride 保持 /8）
        # layer3: dilation=2，layer4: dilation=4
        for block in enc.layer3:
            # 主 3×3 conv: stride→1, dilation→2
            block.conv2.stride  = (1, 1)
            block.conv2.dilation = (2, 2)
            block.conv2.padding  = (2, 2)
            # downsample shortcut conv: stride→1
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        for block in enc.layer4:
            block.conv2.stride  = (1, 1)
            block.conv2.dilation = (4, 4)
            block.conv2.padding  = (4, 4)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)

        self.layer0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu, enc.maxpool)
        self.layer1 = enc.layer1   # /4, 256
        self.layer2 = enc.layer2   # /8, 512
        self.layer3 = enc.layer3   # /8, 1024 (dilated)
        self.layer4 = enc.layer4   # /8, 2048 (dilated)

        self.aspp   = ASPP(2048, 256)
        self.low_proj = ConvBnRelu(256, 48, 1, p=0)
        self.att    = AttGate(256, 48, 64)
        self.fuse   = nn.Sequential(ConvBnRelu(256+48, 256), ConvBnRelu(256, 256))
        self.head   = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        l0 = self.layer0(x)
        l1 = self.layer1(l0)   # low-level
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        hi = self.aspp(l4)                             # /8
        lo = self.low_proj(l1)                         # /4
        lo = self.att(hi, lo)
        hi = _up(hi, lo)
        out = self.fuse(torch.cat([hi, lo], 1))
        out = F.interpolate(out, (H, W), mode="bilinear", align_corners=False)
        return {"logits": self.head(out)}


# ═══════════════════════════════════════════════════════════════════════
# 4. TransFuse  (Zhang 2021, MICCAI)
#    并行 ResNet + DeiT-Small，BiFusion 融合
# ═══════════════════════════════════════════════════════════════════════

class _DeiTTiny(nn.Module):
    """轻量 ViT 替代 DeiT-Small（保持接口一致，降低显存）"""
    def __init__(self, img_size=224, patch=16, dim=192, depth=6, heads=3):
        super().__init__()
        n = (img_size // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.pos   = nn.Parameter(torch.zeros(1, n, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, dropout=0.1,
                                            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.dim  = dim
        self.n    = n
        self.ps   = img_size // patch

    def forward(self, x):
        B = x.shape[0]
        t = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        t = self.norm(self.enc(t))                        # [B, N, dim]
        return t.transpose(1, 2).reshape(B, self.dim, self.ps, self.ps)


class BiFusion(nn.Module):
    def __init__(self, cnn_ch, vit_ch, out_ch):
        super().__init__()
        self.cnn_proj = ConvBnRelu(cnn_ch, out_ch, 1, p=0)
        self.vit_proj = ConvBnRelu(vit_ch, out_ch, 1, p=0)
        self.fuse     = ConvBnRelu(out_ch*2, out_ch)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch, out_ch//4), nn.ReLU(inplace=True),
            nn.Linear(out_ch//4, out_ch), nn.Sigmoid(),
        )

    def forward(self, c, v):
        v = _up(v, c)
        c = self.cnn_proj(c);  v = self.vit_proj(v)
        f = self.fuse(torch.cat([c, v], 1))
        w = self.se(f).view(f.shape[0], -1, 1, 1)
        return f * w


class TransFuse(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.cnn0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu, enc.maxpool)
        self.cnn1 = enc.layer1   # 64
        self.cnn2 = enc.layer2   # 128
        self.cnn3 = enc.layer3   # 256
        self.cnn4 = enc.layer4   # 512

        self.vit  = _DeiTTiny()  # 192-dim, /16 spatial
        vd = self.vit.dim

        self.bf4 = BiFusion(512, vd, 256)
        self.bf3 = BiFusion(256, vd, 128)
        self.bf2 = BiFusion(128, vd, 64)

        self.d4 = DoubleConv(256, 128)
        self.d3 = DoubleConv(128+128, 64)
        self.d2 = DoubleConv(64+64, 32)
        self.d1 = DoubleConv(32+64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        v = self.vit(x)

        c0 = self.cnn0(x)
        c1 = self.cnn1(c0)
        c2 = self.cnn2(c1)
        c3 = self.cnn3(c2)
        c4 = self.cnn4(c3)

        f4 = self.bf4(c4, v)
        f3 = self.bf3(c3, v)
        f2 = self.bf2(c2, v)

        d = self.d4(f4)
        d = self.d3(torch.cat([_up(d, f3), f3], 1))
        d = self.d2(torch.cat([_up(d, f2), f2], 1))
        d = self.d1(torch.cat([_up(d, c1), c1], 1))
        d = F.interpolate(d, (H, W), mode="bilinear", align_corners=False)
        return {"logits": self.head(d)}


# ═══════════════════════════════════════════════════════════════════════
# 5. FAT-Net  (Wu 2022, Medical Image Analysis)
#    Feature Adaptive Transformer: axial attention + CNN 融合
# ═══════════════════════════════════════════════════════════════════════

class _AxialAttn(nn.Module):
    """Height-wise axial attention (简化版)"""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv1d(ch, ch*3, 1, bias=False)
        self.proj = nn.Conv1d(ch, ch, 1, bias=False)
        self.scale = (ch // heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        # height-wise: treat W as sequence, average over H
        xh = x.mean(2)                          # [B, C, W]
        qkv = self.qkv(xh).chunk(3, dim=1)      # each [B, C, W]
        q, k, v = [t.view(B, self.heads, C//self.heads, W) for t in qkv]
        attn = torch.softmax(q.transpose(-1,-2) @ k * self.scale, dim=-1)
        out  = (attn @ v.transpose(-1,-2)).transpose(-1,-2)
        out  = out.reshape(B, C, W)
        out  = self.proj(out).unsqueeze(2).expand_as(x)
        return x + out


class FATBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.attn  = _AxialAttn(ch)
        self.conv  = DoubleConv(ch, ch)
        self.fuse  = ConvBnRelu(ch*2, ch, 1, p=0)

    def forward(self, x):
        return self.fuse(torch.cat([self.conv(x), self.attn(x)], 1))


class FATNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.e0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu, enc.maxpool)
        self.e1 = enc.layer1   # 256
        self.e2 = enc.layer2   # 512
        self.e3 = enc.layer3   # 1024
        self.e4 = enc.layer4   # 2048

        self.fat4 = FATBlock(2048)
        self.fat3 = FATBlock(1024)
        self.fat2 = FATBlock(512)

        self.d4 = DoubleConv(2048+1024, 512)
        self.d3 = DoubleConv(512+512,   256)
        self.d2 = DoubleConv(256+256,   128)
        self.d1 = DoubleConv(128+64,    64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        f4 = self.fat4(e4)
        f3 = self.fat3(e3)
        f2 = self.fat2(e2)

        d = self.d4(torch.cat([_up(f4, f3), f3], 1))
        d = self.d3(torch.cat([_up(d,  f2), f2], 1))
        d = self.d2(torch.cat([_up(d,  e1), e1], 1))
        d = self.d1(torch.cat([_up(d,  e0), e0], 1))
        d = F.interpolate(d, (H, W), mode="bilinear", align_corners=False)
        return {"logits": self.head(d)}


# ═══════════════════════════════════════════════════════════════════════
# 6. MALUNet  (Azad 2022, ICASSP)
#    Multi-scale Axial attention Light U-Net
# ═══════════════════════════════════════════════════════════════════════

class _MSA(nn.Module):
    """Multi-scale axial attention (轻量版)"""
    def __init__(self, ch, scales=(1, 2, 4)):
        super().__init__()
        # 每个分支通道数整除，proj 接实际拼接通道数
        branch_ch = ch // len(scales)
        concat_ch = branch_ch * len(scales)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(s, s) if s > 1 else nn.Identity(),
                nn.Conv2d(ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )
            for s in scales
        ])
        self.proj = ConvBnRelu(concat_ch, ch, 1, p=0)
        self.scales = scales

    def forward(self, x):
        outs = []
        for b, s in zip(self.branches, self.scales):
            o = b(x)
            if s > 1:
                o = F.interpolate(o, x.shape[-2:], mode="bilinear", align_corners=False)
            outs.append(o)
        return self.proj(torch.cat(outs, 1)) + x


class MALUNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # 轻量骨干: 自定义简洁编码器（不依赖大模型）
        def blk(ic, oc, s=1):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, s, 1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            )
        self.enc1 = blk(3,   32)          # /1
        self.enc2 = blk(32,  64,  s=2)   # /2
        self.enc3 = blk(64,  128, s=2)   # /4
        self.enc4 = blk(128, 256, s=2)   # /8
        self.bot  = blk(256, 512, s=2)   # /16

        self.msa4 = _MSA(512)
        self.msa3 = _MSA(256)
        self.msa2 = _MSA(128)

        self.dec4 = DoubleConv(512+256, 256)
        self.dec3 = DoubleConv(256+128, 128)
        self.dec2 = DoubleConv(128+64,  64)
        self.dec1 = DoubleConv(64+32,   32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.msa4(self.bot(e4))

        d = self.dec4(torch.cat([_up(b,  self.msa3(e4)), self.msa3(e4)], 1))
        d = self.dec3(torch.cat([_up(d,  self.msa2(e3)), self.msa2(e3)], 1))
        d = self.dec2(torch.cat([_up(d,  e2), e2], 1))
        d = self.dec1(torch.cat([_up(d,  e1), e1], 1))
        d = F.interpolate(d, (H, W), mode="bilinear", align_corners=False)
        return {"logits": self.head(d)}


# ═══════════════════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════════════════

MODELS = {
    "unet":            UNet,
    "unetplusplus":    UNetPlusPlus,
    "attdeeplabv3plus":AttDeepLabv3Plus,
    "transfuse":       TransFuse,
    "fatnet":          FATNet,
    "malunet":         MALUNet,
}


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower().replace("-", "").replace("_", "").replace("+", "plus")
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS)}")
    return MODELS[name](pretrained=pretrained)
