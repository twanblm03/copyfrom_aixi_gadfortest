# 三组历史学习率、固定中心帧全量训练结果（2026-10-20）

本报告比较 B0、H1-Sup only、SAM ROI blend 三组。三组均为 place TRAIN 6,444 → VAL 1,386，固定中心帧 T=5、ResNet18、GT actor tracks、四卡 DDP、有效 batch 16、seed 1、Adam、head/backbone LR 1e-5→1e-4、5 epoch warmup、无梯度裁剪，训练 30 epochs、每轮完整 VAL、按修正后的 group mAP@1.0 选唯一 checkpoint。没有评估 TEST。H1-Sup sigma ratio 为 0.15；SAM 使用同一 projected mask cache，actor ROI 采用 0.5 上下文混合。SAM 与 H1-Sup 没有在同一模型中叠加。

## 完整性

三组均有连续 epoch 1–30 的 TRAIN 与 VAL 记录、`best_val.pth` 和 `epoch30.pth`；日志各出现 30 次训练与验证标记，无 traceback。读取六个 checkpoint 后，选中/末轮分别为 B0 15/30、H1-Sup 5/30、SAM 25/30，选模状态与 summary 完全一致，模型张量全部有限。三个训练进程已经退出，但启动 wrapper 的原始状态文件没有写入退出码；原文件已保存为各目录的 `full_training_launcher.status`，`full_training.status` 标记为 `training_artifacts_complete ... exit_code_unavailable`。因此确认训练产物完整，不声称已取得 wrapper 的正常退出码。

重新用同一修正评价器读取各自**由 mAP@1.0 选中的 epoch 预测**，均覆盖相同的 1,386 VAL clips，所得三指标与训练 summary 在 1e-6 内一致。完整逐类结果及可重跑比较脚本见 [JSON](../../results/sam_person/lr1_center_full_20261020.json) 和 [比较脚本](../../tools/compare_lr1_center_full.py)；[30 轮曲线](../../results/sam_person/lr1_center_curves_20261020.png) 中圆点为选中 checkpoint。

## 主结果：同一 checkpoint 的三指标

| 模型 | 最佳 epoch | mAP@1.0 | mAP@0.5 | outlier mIoU | 对 B0 的 mAP@1.0 差值 |
|---|---:|---:|---:|---:|---:|
| B0 | 15 | **20.8631** | 36.6850 | **71.3912** | — |
| H1-Sup | 5 | 10.3981 | 35.9173 | 58.9806 | −10.4650 |
| SAM ROI blend | 25 | 13.0550 | **42.9830** | 64.3844 | −7.8081 |

B0 在主要指标 mAP@1.0 上明显领先。SAM 在同一选模规则下提升 mAP@0.5 6.2980 点，却降低 mAP@1.0 7.8081 点与 outlier mIoU 7.0068 点。H1-Sup 三指标均低于 B0 的所选 checkpoint。**不要把不同 epoch 的各指标峰值拼成一个模型的结果**：B0 独立的 mAP@0.5 峰值为 42.7083（epoch 11），与 SAM 的 42.9830 差仅 0.2747 点；H1-Sup 的 mAP@0.5 峰值为 40.9030（epoch 17），但它的 mAP@1.0 最佳在 epoch 5。

## 群组覆盖诊断

以下是对 2,224 个 VAL 真值群组的额外诊断，忽略置信度排序与重复预测惩罚：每个真值组寻找同 clip 中最高的预测成员集合 IoU。`类别正确` 再要求预测活动类别相同。它不是官方 AP，不可与上表混作同一指标。

| 模型 | 结构 IoU=1 覆盖 | 结构 IoU≥0.5 覆盖 | 类别正确且 IoU=1 | 类别正确且 IoU≥0.5 |
|---|---:|---:|---:|---:|
| B0 | 46.90% | 85.70% | **38.31%** | 55.71% |
| H1-Sup | 34.22% | 89.30% | 24.33% | 55.35% |
| SAM | 42.09% | **91.32%** | 33.23% | **65.42%** |

SAM 更常找到类别正确且至少半数成员交叠的组，但比 B0 更少给出完全一致的成员集合；H1-Sup 也更少给出完全一致的组。这与 mAP@0.5 和 mAP@1.0 的方向相符。逐类 AP 进一步显示：SAM 的宽松 AP 在 Eating/Drinking、Working/Studying、Fighting、TakingSelfie 上均高于 B0，但严格 AP 在 Eating/Drinking、Working/Studying、TakingSelfie 上更低；H1-Sup 的严格 AP 在六类中仅 Queueing 与 B0 同为零，其余均低于 B0。Queueing/Ordering 的严格 AP 对三组都接近零。

## 训练曲线与解释边界

B0 的 mAP@1.0 在 epoch 15 达峰，之后大致在 13–19 间波动；H1-Sup 在 epoch 5 达峰后 25 轮都未恢复到 10.3981；SAM 在 epoch 25 才达到 13.0550。epoch 30 的 TRAIN `loss_group_code` 三组均约 0.183，但 VAL 分别为 B0 0.586、H1-Sup 0.671、SAM 0.552。H1-Sup 的 TRAIN mask Dice 从 epoch 1 的 0.926 降至 epoch 30 的 0.348，VAL 末轮仍为 0.586，显示辅助空间监督存在训练/验证差距。总 loss 不宜直接跨 H1 与其它模型比较，因为 H1 额外包含 mask loss。

历史双卡 DataParallel、随机分段采帧的 H1-Sup 在旧评价器下领先配对 B0；对其已选权重的保存预测用当前修正评价器复评，H1-Sup/B0 为 22.95/18.27 mAP@1.0。因此本轮排名翻转不能仅由评价器修正解释。历史与本轮仍同时改变了 TRAIN 采帧、并行方式、每卡局部 BatchNorm 批量，以及 H1 matched-group loss 的跨 rank 归一化；无法从现有三组数据单独归因。旧的低 backbone LR SAM 完整训练曾在不同裁剪协议下达到 14.81 mAP@1.0，也不能直接当作本轮的学习率单因素对照。

本轮的可靠结论只针对单 seed、VAL、这个训练协议：B0 是严格群组指标最强的独立模型；SAM 改善宽松群组覆盖，但没有改善严格成员集合与 outlier；H1-Sup underperforms B0。未训练 SAM+H1 组合，因此不能由此判断联合训练的效果。下一轮若要解释 H1 历史收益消失，应先在四卡 DDP 下成对比较随机分段采帧与固定中心帧，再检查普通 BatchNorm 和 matched-group 的全局归一化；每次保持其它变量不变。
