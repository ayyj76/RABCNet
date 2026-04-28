# 模型代码思路与训练参数说明

> 本文档整理最终模型的架构思路、训练流程、关键参数和实验结果，用于论文写作参考。

## 目录结构

```
draw/
├── README.md                          # 本文档（思路 + 参数 + 结果汇总）
├── configs/                           # 最终训练配置
│   ├── rpi_net_isic2017_unsup_v17_msaa.yaml          # 阶段1：源域训练
│   ├── rpi_net_isic2018_unsup_unified_v20sigma.yaml  # 阶段2：ISIC-2018适应
│   └── rpi_net_ph2_unsup_unified_v20sigma.yaml       # 阶段2：PH2适应
├── src/                               # 核心模型代码
│   ├── model.py                       # 网络结构（ConvNeXt-Tiny + 多模块解码器）
│   ├── losses.py                      # 损失函数（Tversky / 边界 / 不确定性）
│   ├── metrics.py                     # 指标计算（ACC/DIC/JAC/SEN/SPE/HD95）
│   ├── datasets.py                    # 数据集加载（ISIC-2017/2018/PH2）
│   └── pseudo_labels.py              # 伪标签生成（K-means + Otsu）
├── scripts/                           # 训练与评估脚本
│   ├── train_rpi_net.py              # 主训练入口
│   ├── evaluate_rpi_net.py           # 模型评估
│   ├── select_inference_by_val.py    # 验证集操作点搜索（τ/σ/TTA扫描）
│   ├── plot_gps_frontier.py          # GPS操作前沿曲线绘图
│   ├── run_decoder_ablation.py       # 解码器消融实验
│   └── paired_significance_test.py  # 配对bootstrap显著性检验
└── results/                          # 最终实验结果数据
    ├── final_v20sigma_three_dataset_results.json      # 三数据集最终指标（主结果）
    ├── final_operating_points_vs_paper_with_std.json  # 含bootstrap CI的完整结果
    ├── final_operating_points_vs_paper.json           # 操作点 vs 原文对比
    └── final_three_dataset_main_table.md              # 主对比表（Markdown格式）
```

---

## 一、整体框架思路

### 训练流程（两阶段）

```
阶段 1：源域全量训练（ISIC-2017）
  ├── 用 K-means 伪标签初始化分割
  ├── 多轮迭代精化（iterative pseudo-label refinement）
  └── 保存最优 checkpoint → outputs/rpi_net_isic2017_unsup_v17_msaa/checkpoints/best.pt

阶段 2：目标域 Sigma-Only 适应（ISIC-2018 / PH2）
  ├── 冻结主干分割路径（freeze_segmentation_path: true）
  ├── 仅训练不确定性相关模块：pseudo_encoder, ipc_attn, pia_attn, sigma_heads
  ├── 极短训练（2 epochs）防止过拟合
  └── 保存最优 checkpoint → outputs/rpi_net_isic2018_unsup_unified_v20sigma/ 等
```

### 核心创新点

1. **DUL（Deep Uncertainty Learning）**：模型输出 logit 均值 + 方差（σ_i），用不确定性加权伪标签监督
2. **GPS（Gaussian Probability Smoothing）**：推理时对 sigmoid 概率图做高斯平滑（`scipy.ndimage.gaussian_filter`），重塑 SEN-SPE 操作前沿
3. **MSAA（Multi-Scale Attention Aggregation）**：多尺度注意力融合解码模块，替代标准 FPN 颈部
4. **伪标签迭代精化**：先用 Otsu K-means 初始化，再用模型自身高置信度预测逐步替换噪声标签

---

## 二、模型架构参数

所有数据集使用**完全相同的模型结构**：

| 参数                           | 值                           |
| ---------------------------- | --------------------------- |
| Backbone                     | ConvNeXt-Tiny（ImageNet 预训练） |
| hidden_dim                   | 128                         |
| dropout                      | 0.1                         |
| 输入分辨率                        | 224 × 224                   |
| use_boundary_head            | true — 边界辅助监督               |
| use_uncertainty_head         | true — DUL σ 预测头            |
| use_aux_head                 | true — 辅助分割头                |
| use_msaa_fusion              | true — 多尺度注意力聚合             |
| use_cbam_decoder             | true — CBAM 通道空间注意力解码       |
| use_simam_refine             | true — SimAM 无参数注意力精化       |
| use_safm_decoder             | true — 空间自适应特征混合解码          |
| use_triplet_decoder          | true — 三元素解码头               |
| use_se_context               | true — SE 通道上下文             |
| use_eca_context              | true — ECA 注意力上下文           |
| use_ema_context              | true — EMA 上下文              |
| use_sk_context               | true — SK 选择核上下文            |
| use_coord_attn               | true — 坐标注意力                |
| use_edge_prompt              | true — 边缘提示                 |
| use_detail_fusion            | true — 细节融合                 |
| use_uncertainty_gated_refine | true — 不确定性门控精化             |
| refine_with_heads            | true — 用辅助头输出精化主输出          |

---

## 三、训练参数详解

### 阶段 1：ISIC-2017 源域训练

**Config**：`configs/rpi_net_isic2017_unsup_v17_msaa.yaml`
**Checkpoint**：`outputs/rpi_net_isic2017_unsup_v17_msaa/checkpoints/best.pt`

#### 基础训练参数

| 参数              | 值           | 说明        |
| --------------- | ----------- | --------- |
| seed            | 42          | 全局随机种子    |
| epochs          | 24          | 主训练轮数     |
| batch_size      | 16          | 训练批大小     |
| lr              | 6e-6        | AdamW 学习率 |
| weight_decay    | 1e-4        | L2 正则     |
| amp / amp_dtype | true / bf16 | 混合精度训练    |
| grad_clip       | 1.0         | 梯度裁剪      |
| image_size      | 224         | 输入分辨率     |

#### 伪标签迭代精化参数

| 参数                               | 值           | 说明               |
| -------------------------------- | ----------- | ---------------- |
| iterative_rounds                 | 1           | 迭代精化轮数           |
| iterative_epochs                 | 8           | 每轮迭代 epoch 数     |
| iterative_blend                  | 0.35        | 精化标签混合权重         |
| iterative_uncertainty_selective  | true        | 用不确定性选择高置信度样本    |
| iterative_uncertainty_percentile | 0.8         | 选取不确定性最低的 80% 样本 |
| iterative_uncertainty_from_sigma | true        | 从 DUL σ 头提取不确定性  |
| iterative_entropy_weight         | 0.6         | 熵加权系数            |
| iterative_blend_high / low       | 0.35 / 0.05 | 高/低置信度区域混合上下限    |

#### 损失函数权重

| 损失项                            | 权重   | 说明                 |
| ------------------------------ | ---- | ------------------ |
| pseudo_tversky_weight          | 0.20 | 伪标签 Tversky 损失     |
| pseudo_tversky_alpha           | 0.65 | Tversky α（假阴性惩罚更大） |
| pseudo_tversky_beta            | 0.35 | Tversky β          |
| pseudo_tversky_gamma           | 1.10 | Focal-Tversky γ    |
| pseudo_consensus_weight        | 0.15 | 伪标签一致性损失           |
| boundary_weight                | 0.03 | 边界损失（主输出）          |
| boundary_head_weight           | 0.15 | 边界头辅助损失            |
| uncertainty_supervision_weight | 0.08 | 不确定性监督损失           |
| aux_consensus_weight           | 0.05 | 辅助头一致性损失           |
| detail_consensus_weight        | 0.08 | 细节融合一致性损失          |

#### 伪标签生成参数

| 参数               | 值    |
| ---------------- | ---- |
| kmeans_clusters  | 2    |
| threshold_method | otsu |
| batch_size       | 8    |

---

### 阶段 2：目标域 Sigma-Only 适应

**ISIC-2018 Config**：`configs/rpi_net_isic2018_unsup_unified_v20sigma.yaml`
**ISIC-2018 Checkpoint**：`outputs/rpi_net_isic2018_unsup_unified_v20sigma/checkpoints/best.pt`

**PH2 Config**：`configs/rpi_net_ph2_unsup_unified_v20sigma.yaml`
**PH2 Checkpoint**：`outputs/rpi_net_ph2_unsup_unified_v20sigma/checkpoints/best.pt`

#### 关键差异（与阶段 1 相比）

| 参数                           | 值                                                     | 说明              |
| ---------------------------- | ----------------------------------------------------- | --------------- |
| epochs                       | 2                                                     | 极短训练，防止过拟合      |
| iterative_rounds             | 0                                                     | 不做伪标签迭代         |
| batch_size                   | 12                                                    | 略小于源域           |
| lr                           | 5e-6                                                  | 更低学习率           |
| **freeze_segmentation_path** | **true**                                              | **冻结主干分割路径**    |
| **freeze_bn_stats**          | **true**                                              | **冻结 BN 统计量**   |
| **trainable_prefixes**       | **[pseudo_encoder, ipc_attn, pia_attn, sigma_heads]** | **仅训练不确定性相关模块** |

#### 损失权重（适配域适应，整体降低）

| 损失项                            | 权重   |
| ------------------------------ | ---- |
| pseudo_tversky_weight          | 0.15 |
| pseudo_consensus_weight        | 0.10 |
| boundary_weight                | 0.02 |
| boundary_head_weight           | 0.10 |
| uncertainty_supervision_weight | 0.05 |
| aux_consensus_weight           | 0.03 |
| detail_consensus_weight        | 0.05 |

#### PH2 特有参数（随机划分）

| 参数               | 值                 |
| ---------------- | ----------------- |
| ph2_split_seed   | 42                |
| ph2_split_ratios | [0.7, 0.15, 0.15] |

---

## 四、推理操作点（Operating Point）

推理时每个数据集有独立的后处理配置：

| 数据集       | 阈值 τ | GPS σ | TTA   | fill_holes | keep_largest |
| --------- | ---- | ----- | ----- | ---------- | ------------ |
| ISIC-2017 | 0.02 | 1.0   | flip4 | true       | true         |
| ISIC-2018 | 0.13 | 0.0   | none  | false      | false        |
| PH2       | 0.10 | 1.0   | flip4 | true       | true         |

**注**：低阈值（τ=0.02）是 DUL logit 压缩的直接结果——σ_i 使 logit 方差收缩，导致 sigmoid 输出集中在 0.5 附近，因此最优阈值远低于通常的 0.5。

### GPS 推理流程

```python
probs = torch.sigmoid(model(image, None)["logits"])[:, 0]  # [B, H, W]
# 对每张图单独做高斯平滑
for i in range(B):
    probs[i] = scipy.ndimage.gaussian_filter(probs[i].cpu().numpy(), sigma=sigma)
mask = (probs > threshold).float()
```

---

## 五、最终实验结果

### 主对比表（vs RPI-Net 原文）

| 数据集       | 指标   | 原文（RPI-Net） | 我们的方法 | Δ     |
| --------- | ---- |:-----------:|:-----:|:-----:|
| ISIC-2017 | ACC  | 87.62       | 91.83 | +4.21 |
| ISIC-2017 | DICE | 75.84       | 81.03 | +5.19 |
| ISIC-2017 | JAC  | 66.21       | 72.05 | +5.84 |
| ISIC-2017 | SEN  | 84.15       | 86.45 | +2.30 |
| ISIC-2017 | SPE  | 91.73       | 94.08 | +2.35 |
| ISIC-2018 | ACC  | 90.21       | 91.13 | +0.92 |
| ISIC-2018 | DICE | 82.88       | 84.22 | +1.34 |
| ISIC-2018 | JAC  | 74.14       | 76.50 | +2.36 |
| ISIC-2018 | SEN  | 87.17       | 87.28 | +0.11 |
| ISIC-2018 | SPE  | 94.05       | 94.96 | +0.91 |
| PH2       | ACC  | 93.25       | 94.77 | +1.52 |
| PH2       | DICE | 88.91       | 92.14 | +3.23 |
| PH2       | JAC  | 81.98       | 86.35 | +4.37 |
| PH2       | SEN  | 93.66       | 93.78 | +0.12 |
| PH2       | SPE  | 94.60       | 95.09 | +0.49 |

### Bootstrap 置信区间（ISIC-2018，n=1000 bootstrap）

| 指标   | 均值     | bootstrap 95% CI |
| ---- | ------ | ---------------- |
| ACC  | 91.13% | [90.17, 91.98]   |
| DICE | 84.22% | [83.02, 85.38]   |
| JAC  | 76.50% | [75.13, 77.91]   |
| SEN  | 87.28% | [86.01, 88.48]   |
| SPE  | 94.96% | [94.33, 95.60]   |

### 边界距离指标（最终模型）

| 数据集       | HD95     | ASSD     |
| --------- | -------- | -------- |
| ISIC-2017 | 22.91 px | 10.26 px |
| ISIC-2018 | 40.51 px | 12.71 px |
| PH2       | 12.14 px | 4.66 px  |

---

## 六、Checkpoint 文件位置汇总

| 数据集           | Checkpoint 路径                                                         |
| ------------- | --------------------------------------------------------------------- |
| ISIC-2017（源域） | `outputs/rpi_net_isic2017_unsup_v17_msaa/checkpoints/best.pt`         |
| ISIC-2018（适应） | `outputs/rpi_net_isic2018_unsup_unified_v20sigma/checkpoints/best.pt` |
| PH2（适应）       | `outputs/rpi_net_ph2_unsup_unified_v20sigma/checkpoints/best.pt`      |

**Checkpoint 内部结构**：

```python
ckpt = torch.load("best.pt", map_location="cpu")
# 权重键为 "model"（不是 "model_state_dict"）
state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
model.load_state_dict(state_dict)
```

---

## 七、运行时配置

| 参数              | 值           |
| --------------- | ----------- |
| device          | cuda        |
| amp / amp_dtype | true / bf16 |
| tf32            | true        |
| channels_last   | true        |
| compile         | false       |
| cpu_threads     | 8–12        |

---

## 八、数据集划分说明

| 数据集       | 划分方式          | 比例              |
| --------- | ------------- | --------------- |
| ISIC-2017 | 官方固定划分        | train/val/test  |
| ISIC-2018 | 官方固定划分        | train/val/test  |
| PH2       | 随机划分（seed=42） | 70% / 15% / 15% |

所有实验统一使用 `seed: 42`，保证可复现性。
