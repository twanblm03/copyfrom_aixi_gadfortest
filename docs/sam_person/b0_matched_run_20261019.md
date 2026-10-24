# 同协议 B0 全量对照（2026-10-19）

目标是比较 [SAM 全量运行](full_run_20261018.md) 中 RoI mask 混合的贡献。运行同一 checkout 的 B0 路径，`train_mask_mode=disabled`；其它设置沿用 SAM 运行：split-by-place TRAIN 6,444 clips → VAL 1,386 clips、GT actor tracklets、固定中心帧 T=5、ResNet18、无 MAE、heatmap off、seed 1、四卡 DDP、有效 batch 16、Adam、线性 5 epoch warmup + 30 epoch schedule、backbone LR scale 0.1、每轮验证并按 group mAP@1.0 选择 checkpoint。为匹配 SAM 首轮在第 7 轮之后续训的实际训练协议，B0 第 1–7 轮不裁剪梯度，从第 8 轮起使用 max norm 1.0。`train.py` 的 `gradient_clipping_start_epoch` 仅改变裁剪生效时间，默认 1 保留此前行为。

本轮从头训练、不加载历史 B0 权重，也不读取 TEST。SAM 在第 7 轮后重启而 B0 持续运行，随机数轨迹不会逐步相同；这个对照匹配训练配置与 split，并保留这项运行路径差异。

启动入口：

```bash
cd /data2/mzy/GAD/CAFE_codebase/.worktrees/cafe-sam-person-mask-20261018
bash scripts/run_b0_matched_full.sh
```

四卡训练于 2026-10-19 10:20:46 +08:00 启动。状态和日志分别是 `.local_data/b0_matched/full_training.status`、`.local_data/b0_matched/full_training.log`；checkpoint 在 `.local_data/b0_matched/full_checkpoints/`。实验标签为 `b0_fixedcenter_matched_sam_seed1`，结果写在 `result/` 和 `outputs/` 的对应目录。CPU 语法检查与 SAM 协议关键字段核对通过，宿主 GPU 启动时四卡空闲。

按用户后续决定，本轮于 2026-10-19 16:29:52 +08:00 主动停止，保留完整的 `epoch18.pth` 与当时的 `best_val.pth`；`epoch18.pth` 已确认可读取、模型参数有限、含 18 轮 TRAIN/VAL 历史。状态文件标为 `training_stopped_by_user`，原 wrapper 退出状态另存于 `.local_data/b0_matched/full_training_wrapper_exit.status`。它不是 30 轮完整对照，不生成最终比较结论。后续改用 backbone LR scale 1、全程无裁剪，并保留固定中心帧与四卡 DDP 从头重训。
