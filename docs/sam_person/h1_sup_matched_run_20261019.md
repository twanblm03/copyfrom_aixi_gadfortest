# 同协议 H1-Sup only 全量训练（2026-10-19）

本次 H1-Sup 与 [当前 B0 对照](b0_matched_run_20261019.md) 共用四卡运行。仅启用 `heatmap_variant=h1_sup` 和历史 H1-Sup 的 `heatmap_sigma_ratio=0.15`，SAM mask 关闭；其它训练与评价设置按当前 B0：split-by-place TRAIN 6,444 → VAL 1,386 clips，固定中心帧 T=5，ResNet18，无 MAE，seed 1，有效 batch 16，Adam，backbone LR scale 0.1，30 epochs，第 8 轮起 max norm 1.0 梯度裁剪，每轮按修正后 group mAP@1.0 选模，不读 TEST。

## 历史 H1-Sup 与当前协议的差异

| 项目 | 2026-10-12 历史 H1-Sup | 当前 H1-Sup / B0 / SAM |
|---|---|---|
| TRAIN 采帧 | 每个时间段随机抽 1 帧；`random_sampling=false` 仍走 `_sample_random_segment_frames` | `fixed_center_train=true`，每段取确定中心帧 |
| Backbone LR | scale 1；初始 1e-5，峰值 1e-4 | scale 0.1；初始 1e-6，峰值 1e-5；head 仍为 1e-5→1e-4 |
| 梯度裁剪 | 全部 30 轮关闭 | 1–7 轮关闭，8–30 轮 max norm 1.0；对应 SAM 第 7 轮后续训的实际协议 |
| VAL 选模评价器 | 旧版 AP 用单元素 set 排序置信度；按该 mAP@1.0 选权重 | 按数值置信度排序、以明确的 1,386 VAL clip 清单限定 GT；按修正后 mAP@1.0 选权重 |
| 并行方式 | 双卡 host parallel，有效 batch 16 | 四卡 DDP，每卡 batch 4，有效 batch 16 |

两边均为 30 epochs、5 帧、ResNet18、Adam、VAL 选模，heatmap target 的 sigma ratio 均为 0.15。历史 H1-Sup 和历史 B0 是当时的配对运行；用当前修正评价器对两者**已选权重的保存预测**在相同 1,386 VAL clips 上重评：H1-Sup 为 group mAP@1.0 22.95、mAP@0.5 43.94、outlier mIoU 73.38；B0 为 18.27、39.26、67.73。这只描述历史协议下所选权重的表现；没有重新按修正评价器搜索历史 30 轮的最佳权重。

本轮入口为 `bash scripts/run_h1_sup_matched_full.sh`。它会先要求四卡各剩余至少 20 GiB 显存，再共享当前 B0 使用的设备。运行状态、日志和 checkpoint 分别位于 `.local_data/h1_sup_matched/full_training.status`、`full_training.log`、`full_checkpoints/`；实验标签为 `h1_sup_fixedcenter_matched_seed1`。训练于 2026-10-19 11:44:20 +08:00 启动，已进入第 1 轮；与 B0 同时训练时四卡各占用约 30.1 GiB，尚余约 18.4 GiB。

按用户后续决定，本轮于 2026-10-19 16:29:52 +08:00 主动停止，保留完整的 `epoch10.pth` 与当时的 `best_val.pth`；`epoch10.pth` 已确认可读取、模型参数有限、含 10 轮 TRAIN/VAL 历史。状态文件标为 `training_stopped_by_user`，原 wrapper 退出状态另存于 `.local_data/h1_sup_matched/full_training_wrapper_exit.status`。它不是 30 轮完整对照，不生成最终比较结论。后续改用 backbone LR scale 1、全程无裁剪，并保留固定中心帧与四卡 DDP 从头重训。
