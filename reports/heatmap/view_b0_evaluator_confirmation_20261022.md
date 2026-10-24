# 当前 view B0 评价器核查（2026-10-22）

结论：当前 B0 使用已固定的 `new_dp4_corrected_evaluator` 修正版。最初训练与两次断点续训的评价器文件 SHA256 相同，没有在续训时回退。

从宿主机读取真实训练进程 PID 4079470 的脚本、工作目录和 PYTHONPATH，并复现其 Python 导入路径，解析到本次恢复快照中的 `evaluation/cafe_eval.py`。SHA256 为 `f1cd5f05b63ec769c149ad1b8926922eb03bb70ab3665fb7e6bf75cec28b40a0`，与最初冻结的修正版、工作目录版本及启动审计一致。

该快照 `train.py:621` 明确使用 `GAD_Evaluation(args, clip_ids=test_set.frames)`，当前参数为 `split=view`、`val_mode=true`、`eval_split=val`，按 VAL group mAP@1.0 选模。GT 被限制到完整 view VAL clip 集合，缺少 GT 会报错；预测置信度按数值排序，漏预测 clip 仍保留在 GT 统计中。指定快照上的 4 项评价器 CPU 回归全部通过。

独立 CPU 复算了最新完整的第 13 轮预测。预测覆盖 2,441 个 clip，缺失 0、额外 0，仅含视频 3、7、11、15、19、23；评价器过滤后的 GT 也为这 2,441 个 clip。

| 指标 | 修正版 CPU 复算 | 与训练 summary 的绝对差 |
| --- | ---: | ---: |
| group mAP@1.0 | 9.22021969071188 | 0 |
| group mAP@0.5 | 25.963500430507136 | 0 |
| outlier mIoU | 54.459326348920776 | 0 |

实际评价器在 worktree `.local_data/sam/dp4_randomsegments_lr1/recovery_e27_20261022/source_snapshot/evaluation/cafe_eval.py`；工作目录为 `/data2/mzy/GAD/CAFE_codebase/.worktrees/cafe-sam-person-mask-20261018`。

机器可读核查记录：`.local_data/view_dp4_20261021/continuation_e4_nas_20261022/b0/evaluator_confirmation.json`。复算使用保存的 `outputs/[cafe]_GAD_[view_b0_randomseg_dp4_lr1_noclip_seed1_resume_e4_nas]_[2026-10-22_17-23-35]/pred_group_epoch_13.txt`，与同名 `result/` 目录的 `summary.json` 比对。本次未修改训练或评价代码，也未使用 GPU。
