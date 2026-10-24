# SAM 四卡 DataParallel 随机采帧训练（2026-10-21）

## 协议

- TRAIN：与同期 B0、H1-Sup 相同，每个 clip 的 5 个时间段各随机抽一帧；VAL 保持 5 个中心帧。
- SAM v1 ViT-B box prompt，原始二值 mask 及 23×40 projected cache，RoI blend 的上下文比例 0.5。
- 四卡 HostStagedDataParallel，全局 batch 16，seed 1，backbone LR scale 1，初始 LR 1e-5、峰值 LR 1e-4，不裁剪梯度，30 epochs，按 VAL group mAP@1.0 选模。
- 完整 TRAIN 候选共有 216,740 帧，VAL 中心帧 6,930 帧；原缓存中的中心帧可复用。新增 mask 缓存约 184,520 帧。

## 缓存与异常保护

- `tools/sam_person_full.py manifest --train-all-frames` 为每个 TRAIN 候选帧记录图像、人物框与身份指纹。VAL 的清单与现有中心帧协议相同。
- 缓存生成后逐帧验证原 mask 和投影缓存。清单缺帧、任一时间段无可用缓存时，数据集在训练前拒绝启动。
- 随机抽帧只从同时具有 raw/projected 缓存文件的候选帧中选择。如果训练过程中缓存读取失败，该帧的 SAM 权重退回全 1，使用原 B0 框 RoI；VAL 读取仍严格报错。
- 单个人物的 SAM 预测为空或失败时，`raw_mask_valid=False`，只对该人物使用框 RoI 回退，其他人物仍用 mask。此类帧可以参与随机抽帧。

## 后台流水线

`scripts/queue_sam_randomsegments_dp4_full.sh` 于 2026-10-21 11:30:03 +08:00 启动，PID 3368699。它顺序完成全帧清单、等待显卡空闲、四卡生成与投影、完整缓存核验、2 步四卡训练验证，以及 30 轮四卡训练。缓存准备预计数小时，具体以日志为准。

状态与日志：

```bash
cd /data2/mzy/GAD/CAFE_codebase/.worktrees/cafe-sam-person-mask-20261018
cat .local_data/sam/dp4_randomsegments_lr1/queue.status
tail -n 20 .local_data/sam/dp4_randomsegments_lr1/manifest.log
tail -n 20 .local_data/sam/dp4_randomsegments_lr1/cache_build/generate_and_project.log
tail -n 20 .local_data/sam/dp4_randomsegments_lr1/cache_build/verify.log
tail -n 20 .local_data/sam/dp4_randomsegments_lr1/full/training.log
```

`queue.status=full_training_started` 表示完整训练已经启动；`full/training.status=training_complete` 表示训练正常结束。生成或验证失败时，`queue.status` 记录失败阶段，完整训练不会启动。
