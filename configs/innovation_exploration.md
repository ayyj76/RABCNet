# Innovation Exploration

本轮探索不覆盖当前论文主线配置，只新增两条实验线。

1. `RABC`: Reliability-Adaptive Boundary Calibration。目标是把固定 GPS 升级为可学习、边界局部、可靠性感知的 logit-space 校准模块。
2. `BRR`: Boundary Recall Recalibration。目标是在保持远背景稳定的前提下，用保守的候选边界召回分支提升 SEN。

建议顺序：

1. 先跑 `ISIC-2017` 子集 smoke，确认损失项数值稳定。
2. 再跑 `ISIC-2018` / `PH2`，比较是否出现更高 SEN 且 SPE 下降可控。
3. 如果 `RABC` 只带来轻微增益但论文叙事更强，可保留为主创新线；如果 `BRR` 显著更稳，可让 `BRR` 做结果主线，`RABC` 做方法升级线。

新配置文件：

- `rpi_net_isic2017_innovation_rabc.yaml`
- `rpi_net_isic2018_innovation_rabc.yaml`
- `rpi_net_ph2_innovation_rabc.yaml`
- `rpi_net_isic2017_innovation_brr.yaml`
- `rpi_net_isic2018_innovation_brr.yaml`
- `rpi_net_ph2_innovation_brr.yaml`

直接运行命令：

```bash
PYTHONPATH=src python3 scripts/train_rpi_net.py --config configs/rpi_net_isic2017_innovation_rabc.yaml
PYTHONPATH=src python3 scripts/train_rpi_net.py --config configs/rpi_net_isic2018_innovation_rabc.yaml
PYTHONPATH=src python3 scripts/train_rpi_net.py --config configs/rpi_net_ph2_innovation_rabc.yaml

PYTHONPATH=src python3 scripts/train_rpi_net.py --config configs/rpi_net_isic2017_innovation_brr.yaml
PYTHONPATH=src python3 scripts/train_rpi_net.py --config configs/rpi_net_isic2018_innovation_brr.yaml
PYTHONPATH=src python3 scripts/train_rpi_net.py --config configs/rpi_net_ph2_innovation_brr.yaml
```

推荐先跑：

1. `rpi_net_isic2017_innovation_rabc.yaml`
2. `rpi_net_isic2017_innovation_brr.yaml`
3. `rpi_net_isic2018_innovation_brr.yaml`

这三条最适合先判断新创新是否值得继续全量扩展。
