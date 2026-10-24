# H1-Sup 当前低分核查（2026-10-19）

截至 22:37 +08:00，三项高 backbone LR 训练仍在运行，以下均为已完成的 VAL epoch 1–14，不是最终 30 轮结论。B0 最佳 mAP@1.0 为 19.3095（epoch 12）；H1-Sup 最佳为 10.3981（epoch 5）。相同第 14 轮为 B0 17.7551、H1-Sup 7.2107；H1-Sup mAP@0.5 最佳为 39.0845，说明问题主要集中在更严格的完整成员匹配。

本次 B0/H1-Sup 的 `args.json` 差异只有实验标签、checkpoint 目录、`heatmap_variant` 和 H1-Sup 所需的 `heatmap_sigma_ratio=0.15`（B0 的 sigma 参数不参与计算）。两项均用固定中心帧、四卡 DDP、有效 batch 16、backbone LR scale 1、无梯度裁剪、相同修正评价器；H1-Sup 未启用 SAM。

与历史成功的 H1-Sup 完整运行保存的源码快照相比，当前 `models/heatmap.py`、`models/heatmap_criterion.py`、`models/__init__.py`、`util/group_export.py` 和 `models/group_matcher.py` 的 SHA256 完全一致。`models/b0_repro.py` 仅增加了 SAM 分支，H1-Sup 模式下该分支不执行。当前 H1-Sup 仍以 B0 embedding membership 做推理，heatmap 只作为训练期辅助监督，匹配器保持原 B0 规则。

使用本轮环境和已缓存 ResNet18 权重执行目标几何、预测一致性及匹配器等 9 项定向 CPU 合同测试，全部通过。第 14 轮 checkpoint 的 316 个模型张量全部有限。训练损失逐轮下降，没有检测到非有限 loss；由于本轮无梯度裁剪，也没有逐步梯度范数日志，不能据此断言梯度从未异常。

有明显的泛化差距：第 14 轮 TRAIN `loss_group_code`，B0 为 0.1981，H1-Sup 为 0.1943；同轮 VAL 为 B0 0.4983、H1-Sup 0.6584。H1-Sup TRAIN mask Dice 从 epoch 1 的 0.9256 降到 epoch 14 的 0.3764，但 epoch 14 VAL mask Dice 仍为 0.6105。H1-Sup 在 epoch 5 mAP@1.0 为 10.3981，高于同轮 B0 的 9.0294；之后 B0 继续提升，H1-Sup 则回落。低 backbone LR 的中止运行在固定中心帧协议下，H1-Sup 截至 epoch 10 最佳 9.4328，高于 B0 截至 epoch 18 最佳 6.9709；这不支持“固定中心帧必然使 H1-Sup 失效”的简单解释。

目前**没有证据确认 H1-Sup 接线、目标或评价存在实现错误，也不能保证算法设计本身没有问题**。较可能的交互因素包括较高 backbone LR 下的辅助监督过拟合、全程无梯度裁剪、固定中心帧削弱时间采样变化，以及四卡 DDP 相比历史双卡 DataParallel 的每卡局部 BatchNorm 统计。还有一个待量化的协议差异：`heatmap_losses` 对每个 rank 内的 matched groups 求平均，再由 DDP 等权平均四个 rank 的梯度；历史 DataParallel 在完整 batch 上求平均，若各 rank 的 matched group 数不同，两者的辅助项权重并不完全相同。当前数据不能区分这些因素，也不能把低分归咎于其中一个。

三项运行保持原样继续完成 30 轮，以免把未完成曲线误当最终结论。若 H1-Sup 最终仍明显落后，优先用独立试验量化 DDP group 归一化与局部 BatchNorm 的影响，再做高/低 backbone LR 和梯度裁剪消融；每次只改一个因素。
