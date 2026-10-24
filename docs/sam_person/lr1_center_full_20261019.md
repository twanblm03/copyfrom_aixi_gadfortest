# 中心采帧、历史 backbone 学习率复训（2026-10-19）

用户选择停止未完成的低学习率 B0/H1-Sup 对照并保留断点，再从头训练 B0、H1-Sup、SAM 三组。先同时启动 B0 与 H1-Sup，随后在显存允许的情况下加入 SAM。旧 B0 的完整 `epoch18.pth`、旧 H1-Sup 的完整 `epoch10.pth` 和各自当时的 `best_val.pth` 都保留，不用于新训练初始化。

新实验保留 SAM cache 所需的固定中心帧 T=5，并保持四卡 DDP、有效 batch 16、seed 1、split-by-place TRAIN 6,444 → VAL 1,386 clips、GT actor tracklets、ResNet18、Adam、30 epochs、5 epoch 线性 warmup、无 MAE、每轮修正后 VAL mAP@1.0 选模。与上一轮低学习率对照相比，head LR 仍为 1e-5→1e-4；backbone LR scale 从 0.1 改为历史值 1，因此 backbone LR 从 1e-6→1e-5 改为 1e-5→1e-4；梯度裁剪从第 8 轮起 max norm 1.0 改为全程关闭。H1-Sup 使用历史 target 的 sigma ratio 0.15；SAM 后续仍使用同一 mask cache 与 0.5 上下文混合。保留当前修正后的评价器，因为旧版按单元素 set 排序置信度会改变 AP 和最佳权重选择。

本轮只恢复历史优化超参数；与 2026-10-12 历史研究运行相比，TRAIN 仍为固定中心帧，执行仍为四卡 DDP 而非双卡 host parallel，评价器也保持修正版。因此新结果既可在新三组之间比较，也可与中止的低学习率运行做阶段诊断，但不等于历史运行逐步复现。

执行入口：

```bash
cd /data2/mzy/GAD/CAFE_codebase/.worktrees/cafe-sam-person-mask-20261018
bash scripts/run_lr1_center_full.sh b0
bash scripts/run_lr1_center_full.sh h1_sup
bash scripts/run_lr1_center_full.sh sam
```

B0 于 2026-10-19 16:31:51 +08:00 启动，H1-Sup 于 16:32:00 +08:00 启动。状态、日志和 checkpoint 分别保存在 `.local_data/historical_lr_center/{b0,h1_sup}/full_training.status`、`full_training.log`、`full_checkpoints/`。各实验结果另写入 `result/` 与 `outputs/`，标签分别为 `b0_center_lr1_noclip_seed1`、`h1_sup_center_lr1_noclip_seed1`。最终结果待训练结束后记录。

2026-10-20 三组均已有完整 30 轮产物。最终比较、逐类 AP、群组覆盖诊断和曲线见 [完整分析](lr1_center_full_analysis_20261020.md)。启动 wrapper 未留下退出码，已保留原启动状态并将状态文件标记为产物核验完成；详情见分析文档。

16:35 两项均已进入第 1 轮，运行参数文件确认固定中心帧、backbone LR scale 1、关闭梯度裁剪，H1-Sup sigma ratio 0.15。两项共同占用每张 GPU 约 30.2 GiB，各卡剩余约 18.3 GiB。

用户于 17:08 授权三项并跑，SAM 于 17:08:07 +08:00 启动，标签 `sam_center_lr1_noclip_seed1`，使用原 SAM projected cache、`sam_roi_blend` 与 0.5 上下文混合。17:14 时 SAM 已到第 1 轮 30/402 步，B0/H1-Sup 已到第 3 轮 30/402 步；三项并跑每卡约占 45.2 GiB，剩余约 3.4 GiB。三项均未见启动错误。

监控状态与最近一次训练进度：

```bash
cd /data2/mzy/GAD/CAFE_codebase/.worktrees/cafe-sam-person-mask-20261018
python - <<'PY'
from pathlib import Path
for variant in ('b0', 'h1_sup', 'sam'):
    root = Path('.local_data/historical_lr_center') / variant
    status = (root / 'full_training.status').read_text().strip()
    lines = (root / 'full_training.log').read_text(errors='replace').splitlines()
    progress = [line for line in lines if 'Train progress epoch' in line]
    print(variant, status)
    print(progress[-1] if progress else 'No step progress yet')
PY
nvidia-smi
```
