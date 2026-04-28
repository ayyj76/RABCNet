# RPI-Net (ISIC2017) Reproduction Baseline

这是基于论文 **Reliable Multi-Source Contrastive Pseudo-Labels Interaction Network (RPI-Net)** 的复现代码基线（从零实现）。

## 目录结构

- `src/rpi_net/`：核心代码（数据、伪标签、模型、DUL 损失、指标）
- `scripts/generate_pseudo_labels.py`：离线生成 `P1~P4`
- `scripts/train_rpi_net.py`：训练与迭代自训练
- `scripts/evaluate_rpi_net.py`：加载 checkpoint 评估
- `configs/rpi_net_isic2017.yaml`：完整训练配置
- `configs/rpi_net_isic2017_smoke.yaml`：快速冒烟配置

## 安装依赖

```bash
pip install -r requirements.txt
```

## 1) 生成伪标签

```bash
PYTHONPATH=src python3 scripts/generate_pseudo_labels.py \
  --config configs/rpi_net_isic2017.yaml \
  --split train
```

## 2) 训练（基线复现）

```bash
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2017.yaml
```

## 3) 快速冒烟

```bash
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2017_smoke.yaml
```

## 4) 评估

```bash
PYTHONPATH=src python3 scripts/evaluate_rpi_net.py \
  --config configs/rpi_net_isic2017.yaml \
  --checkpoint outputs/rpi_net_isic2017/checkpoints/best.pt \
  --split test
```

## 说明

- 当前代码严格保留了 RPI-Net 的主干思想：多源伪标签、IPC/PIA 交互、DUL 动态不确定性加权、迭代更新。
- 由于原文未公开代码且部分模块细节未完全披露，部分实现采用等价工程实现（例如 CAM 与特征提取细节），用于可运行复现与后续改进实验。

---

## 训练交接（2026-02-22）

更新时间：2026-02-22 14:49（本机时间）

### 1. 当前状态

1. 已按要求停止自动训练。
2. 停止方式：向 `auto_train_evolve_keepbest.py` 主进程发送 `SIGTERM`，脚本捕获后触发 `KeyboardInterrupt` 正常退出。
3. 当前无训练相关进程在跑（`auto_train_evolve_keepbest.py` / `train_rpi_net.py` / `select_inference_by_val.py` 均已退出）。
4. 停止邮件已发送（日志中有 `mail sent: [skin] training stopped (KeyboardInterrupt)`）。

### 2. 关键目标与判定规则（本轮）

1. 目标：严格纯无监督框架下，三数据集（PH2、ISIC-2017、ISIC-2018）5项指标都超过论文值。
2. 论文阈值（脚本内置）：
   ACC/DICE/JAC/SEN/SPE  
   ISIC-2018: 0.9021/0.8288/0.7414/0.8717/0.9405  
   ISIC-2017: 0.8762/0.7584/0.6621/0.8415/0.9173  
   PH2: 0.9325/0.8891/0.8198/0.9366/0.9460
3. 历史最优更新规则：`all5_gt`，即 5 项都必须同时优于当前 best 才算改进。
4. 运行流：PH2 先筛（不过论文阈值就直接结束该 run，不进入 ISIC）。

### 3. 自动训练框架与主要配置

1. 入口脚本：`scripts/auto_train_evolve_keepbest.py`
2. 训练脚本：`scripts/train_rpi_net.py`
3. 当前候选（每轮 4 个）：
   `evolve_csmall_fullft_hybrid`  
   `evolve_csmall_iterative`  
   `evolve_ctiny_headmix_iterative`  
   `evolve_ctiny_mamba_fullft`
4. 基础特征：
   纯无监督（`supervised_weight=0`, `labeled_ratio=0`）  
   AMP BF16（`amp=True`, `amp_dtype=bf16`）  
   默认 backbone 为 `convnext_small`，部分候选用 `convnext_tiny`  
   启用较多增强模块（EGA、CBAM、SAFM、CGFN、recall refine 等）
5. 近期做过的工程改造（已落地）：
   磁盘保护与自动清理旧 run  
   轻量 checkpoint（减少 optimizer/scaler 保存）  
   按 PH2 快速筛选后再决定是否跑 ISIC

### 4. 已尝试的方法（可追踪）

1. 严格同构自动候选（当前主线）
   `evolve_csmall_fullft_hybrid`：ConvNeXt-Small，全量混合头训练  
   `evolve_csmall_iterative`：加入迭代伪标签更新（uncertainty-selective）  
   `evolve_ctiny_headmix_iterative`：ConvNeXt-Tiny + recall candidate/multiscale recall 头，头部微调  
   `evolve_ctiny_mamba_fullft`：ConvNeXt-Tiny + mamba logit refine
2. 历史无监督版本（目录可见）
   存在大量 `rpi_net_isic2017_unsup_vXX_*` / `rpi_net_isic2018_unsup_*` / `rpi_net_ph2_unsup_*` 结果目录。  
   覆盖模块族：EGA、bg decouple、bgsup、ulogit gate、SAM hybrid、pseudo refine、MFE、CoT、residual logit refine、CGFN、recall refine/candidate/multiscale recall 等。  
   说明：这些目录可用于让下一位AI做对照和迁移，但当前严格自动循环仅使用上面 4 个候选。

### 5. 最近保留 runs 的可用结果（PH2）

1. `20260222_143725_evolve_csmall_fullft_hybrid_s42`  
   PH2 train-best test: ACC 0.6821, DICE 0.5480, JAC 0.4032, SEN 0.9229, SPE 0.5490  
   run 结果：`select_failed`
2. `20260222_143949_evolve_csmall_iterative_s42`  
   PH2 train-best test: ACC 0.6653, DICE 0.5438, JAC 0.4002, SEN 0.9409, SPE 0.5196  
   run 结果：`select_failed`
3. `20260222_144226_evolve_ctiny_headmix_iterative_s42`  
   PH2 train-best test: ACC 0.2920, DICE 0.4073, JAC 0.2920, SEN 1.0000, SPE 0.0000  
   run 结果：`select_failed`
4. `20260222_144430_evolve_ctiny_mamba_fullft_s42`  
   PH2 train-best test: ACC 0.6068, DICE 0.5169, JAC 0.3760, SEN 0.9771, SPE 0.4174  
   run 结果：`select_failed`
5. `20260222_144818_evolve_csmall_fullft_hybrid_s42`  
   在 epoch 3 附近被用户中止，report 已标记 `stopped_by_user`

### 6. 全局进展结论

1. 自动搜索累计到 round 83（`progress_board.json` 最近 300 条均为 `continue`）。
2. `best_registry.json` 仍为空 `{}`。
3. 主要原因不是训练完全不收敛，而是大量 run 在 `select_inference_by_val.py` 阶段触发 `No valid operating point found`，导致无法进入正式 test-eval 写入 best。
4. PH2 上常见现象：SEN 高、SPE 低，说明偏向过分割（前景召回高但背景误报大）。

### 7. 对下一位AI的直接建议（可执行）

1. 优先处理 `select_failed`：
   放宽 guard 或分层 guard（先保证能选出 operating point，再做 stricter 二筛）。
2. 调整后处理搜索空间：
   先收缩阈值/最小面积的组合范围，避免“所有组合都不满足约束”。
3. 针对“高 SEN 低 SPE”专门优化：
   提升背景抑制损失权重或引入更强背景校正头，减少假阳性。
4. 保留当前磁盘策略：
   继续使用自动清理与轻量 checkpoint，避免再次因磁盘满导致 `torch.save` 崩溃。

### 8. 关键文件与路径

1. 训练主控脚本：`scripts/auto_train_evolve_keepbest.py`
2. 训练实现脚本：`scripts/train_rpi_net.py`
3. 主日志：`outputs/strict_autotrain_loop/launcher.log`
4. 进度板：`outputs/strict_autotrain_loop/progress_board.json`
5. 当前状态：`outputs/strict_autotrain_loop/loop_status.json`
6. 历史最优注册：`outputs/strict_autotrain_loop/best_registry.json`
7. 最近 run 报告目录：`outputs/strict_autotrain_loop/runs/`

### 9. 本次停止动作的证据

1. `launcher.log` 尾部显示：`[skin] training stopped (KeyboardInterrupt)` 邮件已发送。
2. `pgrep` 结果为空（仅剩查询命令本身），确认训练已停。
