# View split：B0 / H1-sup / SAM 固定配置

最新状态（2026-10-22 14:02）：原 B0 完成 3 轮后因显存不足退出；已启动 `continuation_e3_20261022` 续训队列（PID `3921389`）及 B0 进程（PID `3921517`）。恢复 epoch3 的模型、Adam、调度器与历史，继续第 4–30 轮。按用户最新指示，只等待显存条件，无需等 place SAM 完成。续训控制/日志在 `.local_data/view_dp4_20261021/continuation_e3_20261022/`，原 `queue_state.json` 路径同步当前状态。旧检查点不含随机数状态，续训不保证与未中断运行逐位等价。

2026-10-22 12:03：按用户指示现在启动 view B0，解除最初等待 place SAM 完成的条件。B0 训练进程 PID `3855229`，当前队列 PID `3855093`；原 place SAM 继续运行。训练使用下述原冻结配置，后续顺序保持 H1-sup→SAM；实时状态仍读 `.local_data/view_dp4_20261021/queue_state.json`。调度变更记录为同目录 `scheduling_release_20261022.json`。

每项从对应的真实 place `args.json` 派生，保留完整 115 个参数字段。
只变更 `split=view`、独立实验标签及 checkpoint 路径；SAM 另外使用 view 专属清单和缓存根目录。
`result_path` 保持 `./outputs/`，每项通过独立标签区分输出。

共同设定：ResNet18、5 帧、随机分段 TRAIN、固定中心 VAL、四卡 Host DataParallel、普通 BN、batch16、test_batch4、seed1、30 轮、warmup5、Adam、WD0、LR 1e-5→1e-4、backbone LR scale1、无梯度裁剪、无 VideoMAE、每轮按修正后的 VAL group mAP@1.0 选模。

| 方法 | heatmap_variant | sigma ratio | train_mask_mode |
| --- | --- | ---: | --- |
| B0 | off | 0.25（未启用热图） | disabled |
| H1-sup | h1_sup | 0.15 | disabled |
| SAM | off | 0.25（未启用热图） | sam_roi_blend，context fraction=0.5 |

`*.args.json` 是最终解析参数，`*.argv.json` 是传给 `train.py` 的参数列表，不含 Python 解释器或 `train.py`。
`contract.plan.json` 的 `runs` 项以 `b0`、`h1_sup`、`sam` 为键，分别提供 `reference_args`、`resolved_args`、`argv` 路径；相对路径均以 worktree 为根。
队列状态不进入 trainer args。

SAM 使用 `.local_data/sam/view_randomsegments_manifest.jsonl` 和 `.local_data/sam/view_cache`，保留相同 SAM v1 ViT-B、box prompt、FP32、23×40 投影和 cache version。
缓存 metadata 的指纹包含 protocol 和 TRAIN/VAL 身份，因此不能直接写入正在使用的 place 缓存目录。

重新生成并验证配置（只解析参数，不执行训练或 GPU 计算）：

```bash
/data2/mzy/GAD/.miniconda3/envs/gad/bin/python configs/view_dp4/prepare_configs.py
```

生成器通过 `runpy` 以非 main 名称加载真实 `train.py` parser，将每项 CLI 重新解析，补齐 `protocol`、`eval_split`、`use_mae`、`mae_dim` 四个训练时派生字段，并对全部 115 字段逐项比对。
之后验证 place→view 允许差异、三方法 105 项共同字段、方法开关、输出隔离及固定修正评价器 SHA256。
审计明细写入 `.local_data/view_dp4_20261021/config_audit.json`。

2026-10-21 已启动宿主机后台队列（PID 3648144）：等待当前 place SAM 正常结束，再依次执行 B0→H1-sup→SAM。每个 GPU 阶段前要求四卡连续两次各有至少 25,000 MiB 空闲显存；当前未启动新的 GPU 作业。
实际执行使用 `.local_data/view_dp4_20261021/source_snapshot/` 内冻结的源码、参数与参考 args，计划为 `.local_data/view_dp4_20261021/queue.plan.json`。
实时状态读取 `.local_data/view_dp4_20261021/queue_state.json`；日志在同目录 `queue.log` 和各方法子目录的 `training.log`。
SAM 训练前必须补齐全部 174,532 个候选帧并通过全量缓存验证；当前仅在 CPU 构建 manifest 和准备复用。
异常会停止后续阶段并保留日志；队列拒绝重复启动及覆盖已存在的训练。
