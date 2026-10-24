# 随机分段采帧复现 H1-Sup（2026-10-20）

**状态更新（21:43）：本页记录的四卡 DDP 两组已按用户要求停止并保留断点；未完成 30 轮。用户改为四卡 DataParallel 训练，后续以新的独立目录和标签重跑 B0/H1-Sup。本页原自动汇总任务已停止，不会产出下述 DDP 四组全量比较。**

目的：检查历史 H1-Sup 相对 B0 的优势为何未在固定中心帧训练中复现。用四卡 DDP 再跑一组配对 B0/H1-Sup，**只将 TRAIN 的每段固定中心帧改为每段随机一帧**，保持 VAL 中心帧及其它 2026-10-19 高学习率协议。新训练从 ImageNet ResNet18 初始化，不续训旧 checkpoint。SAM 不参与，因为其离线 mask cache 仅覆盖中心帧。

历史 VAL 采帧已复核：`/data2/mzy/GAD/CAFE_heatmap_research_20261012/dataloader/dataloader.py` 创建 `val_set` 时传入 `is_training=False`；同一历史源码的 `dataloader/cafe.py` 在该状态下调用 `_sample_segment_centers`。原始 B0、H1-Sup 的 `prediction_export_epoch1.txt` 与 `prediction_export_epoch2.txt` 各有 6,930 个 VAL 帧键（1,386 clips × 5 帧），各自两轮的帧键集合完全相同。因此历史高分运行是 **TRAIN 随机分段、VAL 固定中心**，本轮采帧设置与之对应。

新旧四卡协议共同项：place TRAIN 6,444 → VAL 1,386 clips，ResNet18、T=5、无 MAE、GT actor tracks、有效 batch16、每 rank 4 clips、seed1、Adam、head/backbone LR 1e-5→1e-4、5 epoch warmup、线性衰减、无梯度裁剪、30 epochs、每轮按修正后的 VAL group mAP@1.0 选模。H1-Sup 使用 `heatmap_variant=h1_sup`、sigma ratio 0.15；B0 为 `off`。`random_sampling` 仍为 false，TRAIN dataset 处于 `is_training=true` 时实际调用 `CafeDataset._sample_random_segment_frames`；VAL dataset 为 `is_training=false`，仍调用 `_sample_segment_centers`。关闭 `--fixed_center_train` 的路径已用 50 帧样本验证：重复 TRAIN 读取产生不同时间索引，VAL 始终相同。

与 2026-10-13 历史双卡 DataParallel 运行相比，本轮仍不同于历史的并行方式、每卡普通 BatchNorm 批量和 H1-Sup mask loss 的跨 rank 归一化。此轮主要检验采帧因素，**不是历史运行的逐步完全复现**。比较时看随机分段 pair 内 `H1-Sup − B0` 的同协议差值，并与固定中心 pair 的 −10.4650 mAP@1.0 点对照；不要把不同协议的绝对值直接解释成采帧的单独因果效应。

入口：`bash scripts/run_lr1_randomsegments_full.sh {b0|h1_sup}`。脚本通过 `FIXED_CENTER_TRAIN=0` 调用共同 launcher；SAM 模式要求固定中心帧，launcher 会拒绝与随机采帧组合。两项分别使用 DDP 端口 29631/29632、独立 checkpoint 目录及标签 `b0_randomseg_lr1_noclip_seed1`、`h1_sup_randomseg_lr1_noclip_seed1`。

B0 于 2026-10-20 20:11:59 +08:00、H1-Sup 于 20:12:04 +08:00 以独立会话启动，wrapper PID 分别为 2778114、2778169。状态、日志、PID 和 checkpoint 位于 `.local_data/historical_lr_randomsegments/{b0,h1_sup}/`。启动时两项均进入数据标注加载阶段，进程仍在运行；首轮与参数文件待复核。

20:15 两项均进入第 1 轮：B0 10/402 步、H1-Sup 9/402 步。实际 `args.json` 与各自的固定中心帧高学习率训练相比，除输出目录/标签外，**唯一配置字段差异为 `fixed_center_train: true → false`**；`random_sampling=false`，因此走随机分段而非全局随机取帧。四卡占用约 30.7–33.9 GiB/卡，仍有约 14.6–17.9 GiB/卡空闲。

20:22 两项都推进到第 1 轮 180/402 步，未见异常，显存占用稳定。自动汇总进程已在后台等待两项跑满 30 轮，届时执行 `python tools/compare_h1_sampling_full.py`；脚本会检查四个 run 的配置、VAL clip 覆盖、单一最佳 checkpoint，再输出 `results/sam_person/h1_sampling_4run.json` 和同名 `.md` 结论，其中包括随机采帧和固定中心帧的两组 `H1-Sup − B0` 差值及其变化。汇总状态和错误日志分别在 `.local_data/historical_lr_randomsegments/comparison.status`、`.local_data/historical_lr_randomsegments/comparison.log`。

监控命令：

```bash
cd /data2/mzy/GAD/CAFE_codebase/.worktrees/cafe-sam-person-mask-20261018
python - <<'PY'
from pathlib import Path
for variant in ('b0', 'h1_sup'):
    root = Path('.local_data/historical_lr_randomsegments') / variant
    print(variant, (root / 'full_training.status').read_text().strip())
    lines = (root / 'full_training.log').read_text(errors='replace').splitlines()
    steps = [line for line in lines if 'Train progress epoch' in line]
    print(steps[-1] if steps else lines[-1] if lines else 'no output')
comparison = Path('.local_data/historical_lr_randomsegments/comparison.status')
print('comparison', comparison.read_text().strip() if comparison.exists() else 'not started')
PY
nvidia-smi
```
