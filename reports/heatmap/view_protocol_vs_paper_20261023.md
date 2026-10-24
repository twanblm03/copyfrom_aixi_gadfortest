# view、place 实验协议与论文及 cherixi fork 比较（2026-10-23）

今后所有 view 实验按用户最新要求固定为：**官方 TRAIN+VAL 三视角、从相同 ImageNet 初始化规则重新训练 30 轮，每轮在第四视角 TEST 评价，保存每轮 checkpoint，并按 TEST group mAP@1.0 选择 best_test.pth**。mAP@0.5 和 outlier mIoU 从同一个选中 checkpoint 报告，同时保留 epoch30.pth。结果明确标为 **best-on-TEST**；这是用户指定的 TEST 选模流程。B0、H1-sup、SAM 共享训练与评价设置，仅保留各方法所需的分支差异。不得用两视角开发 checkpoint 接着训练来代替这次重新训练。

现有 TRAIN→VAL 两视角 B0 保留并完成，定位为开发对照；其分数不能当作论文的 view TEST 分数。place 仍使用已固定的 TRAIN→VAL 开发协议，本次不改变其训练或选模方式。本页规则覆盖旧文档中对未来 view 沿用 TRAIN→VAL 的安排。[根协议](../../BASELINE_PROTOCOL.md)与[研究 worktree 协议](../../.worktrees/cafe-sam-person-mask-20261018/BASELINE_PROTOCOL.md)已记录同一规则。

## 数据划分与 clip 口径

论文的 view 原始划分为两个训练视角、一个验证视角、一个测试视角；正式训练合并 TRAIN+VAL 后使用三个视角。六个场地对应 12/6/6 个原始 TRAIN/VAL/TEST 视频。原始 clip 数分别为 5,146 / 2,576 / 2,575，其中 TRAIN 为两个视角的 2,574+2,572。以上是数据集统计，不是本地加载器过滤后的计数。[论文附录 Table S2、A.2](https://arxiv.org/html/2312.02878v2#A1.SS2)

| 集合 | 官方视频编号 | 本地沿用官方 group 过滤后的 clips |
|---|---|---:|
| TRAIN（两个视角） | 1,2,5,6,9,10,13,14,17,18,21,22 | 4,857 |
| VAL（第三视角） | 3,7,11,15,19,23 | 2,441 |
| TRAIN+VAL（后续重新训练） | 上述两组之并集，18 个视频 | 7,298 |
| TEST（第四视角） | 4,8,12,16,20,24 | 2,414 |

官方解析器只保留至少一个已标注 group 的 clip，本地沿用该逻辑。因此原始 5,146/2,576 与实际 4,857/2,441 的差别不能解释成重新划分、随机丢弃 clip 或视角不一致。[固定版本数据划分](https://github.com/dk-kim/CAFE_codebase/blob/caee21a8ccdd5faadecbc5a5034107084a7cd270/dataloader/dataloader.py)、[固定版本标注过滤](https://github.com/dk-kim/CAFE_codebase/blob/caee21a8ccdd5faadecbc5a5034107084a7cd270/dataloader/cafe.py#L127-L128)

每个 clip 的单位与原始标注保持一致，没有重新切分视频窗口。最终训练保持 batch16、drop_last 和 30 轮时，每轮为 floor(7,298/16)=456 个 batch；开发训练为 floor(4,857/16)=303 个 batch。相同轮数下，数据量与更新次数随合并训练集一起增加，不能把可能的分数变化仅归因于 backbone 或评价器。

## 与论文及官方发布代码的关系

论文主要设定一致：ImageNet ResNet18、GT actor tracklets、5 帧分段采样、256 维特征、5×5 RoIAlign、6 层 4 头、12 个 group tokens；Adam、batch16、30 轮、前 5 轮从 1e-5 升至 1e-4 再线性下降；membership/consistency 系数 5/2，temperature 0.2。[论文 §5.1](https://arxiv.org/html/2312.02878v2#S5.SS1)

下表把论文文字与官方默认实现分开。官方代码固定引用 commit `caee21a8ccdd5faadecbc5a5034107084a7cd270`，不把未写在论文中的默认值说成论文明确要求。

| 项目 | 论文 / 固定版本官方代码 | 当前及后续本地设置 |
|---|---|---|
| TRAIN 随机采帧 | 论文规定分段采帧；官方每段长度 `floor(L/5)`，每段随机抽 1 帧，余数尾部不进入候选集 | 每段随机抽 1 帧；用 `linspace(0,L,6)` 边界覆盖尾部。采样策略相同，候选边界存在实现差异 |
| VAL/TEST 采帧 | 论文未明确固定中心；官方非训练分支仍随机分段抽帧 | 过去 VAL、后续 TEST 均取确定的分段中心帧；这是保留的实现差异 |
| Weight decay | 论文未述；官方默认 1e-4 | 保持 0 |
| 第 30 轮训练学习率 | 论文未明确终值；官方默认 CyclicLR 轨迹该轮为 1e-5 | 保持当前计划，该轮为 0；不能描述成该轮仍作非零 LR 更新 |
| 并行与 BN | 论文未报告训练 GPU 数；官方默认双卡 DataParallel、普通 BN | 保持四卡 Host DataParallel、普通 BN、全局 batch16；每卡局部 batch 不同 |
| 正式训练与评价时机 | 官方默认不传 `--val_mode`，TRAIN+VAL→TEST，每轮评价并保存 checkpoint，没有自动 best 选择；不能据此确定论文实际的选模方式 | 后续 view TRAIN+VAL 重训，每轮 TEST 并保存每轮 checkpoint；本地新增按 TEST group mAP@1.0 显式选择 best_test.pth，标签为 best-on-TEST，并保留 epoch30.pth |
| 指标实现 | 官方 confidence 被单元素 set 包装后直接排序，不能实现预期的数值置信度排序 | 永久保留修正后的数值排序及完整评价 clip 清单，不回退旧评价器 |

采帧依据：[官方 `select_frames`](https://github.com/dk-kim/CAFE_codebase/blob/caee21a8ccdd5faadecbc5a5034107084a7cd270/dataloader/cafe.py#L179-L205)。优化、并行与评价时机依据：[官方 `train.py`](https://github.com/dk-kim/CAFE_codebase/blob/caee21a8ccdd5faadecbc5a5034107084a7cd270/train.py)。评价器问题及修正依据：[原版评价器](https://github.com/dk-kim/CAFE_codebase/blob/caee21a8ccdd5faadecbc5a5034107084a7cd270/evaluation/cafe_eval.py)、[本地评价器核查](view_b0_evaluator_confirmation_20261022.md)、[此前完整协议审计](b0_baseline_vs_cafe_paper_20261021.md)。

上述差异均保留在报告中；这是一套基于论文主要设定的受控实验协议，不宣称严格完全复现原论文。三视角训练更接近正式测试的数据口径，但是否涨分、涨幅多少仍需实际实验；开发 VAL 与最终 TEST 分数也不能直接相减当作改动的因果收益。

## 艾师兄提供的 cherixi fork：place 与 view 单独核对（2026-10-23）

本节只审查用户提供的 [cherixi/CAFE_codebase](https://github.com/cherixi/CAFE_codebase)，不将上文 dk-kim 原版默认值套用到该 fork。读取时 `main` 固定为 commit **`1c5d5733d33eaed72867c5fe9bf4d5620fded3e9`**，提交时间为 2026-01-12 12:11:27 UTC。[固定提交](https://github.com/cherixi/CAFE_codebase/commit/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9)

**该 fork 的 place 和 view 脚本只有 split 参数不同，均为 TRAIN+VAL→TEST，每轮 TEST、保存每轮权重，训练和 TEST 都随机分段采帧。** 它还会统计各指标自己的 best epoch，但没有保存按单一主指标选择的 `best_test.pth`。两个脚本均未覆盖模型、输入、优化器、BN、MAE、GPU 等默认值；下面列出的默认配置因此确实作用于这两个裸脚本。仓库代码不能证明艾师兄某次实际运行没有附加参数。

| 项目 | fork place 脚本 | fork view 脚本 | 当前本地口径 |
|---|---|---|---|
| 脚本参数 | `data_path=Dataset/`、`split=place` | `data_path=Dataset/`、`split=view` | 后续 view 使用用户指定的新流程；place 不随之改变 |
| `--val_mode` | 未传，false | 未传，false | 当前 place 为 true；新 view 为 false |
| 训练视频 | 16 TRAIN+4 VAL=20；5 个场景 | 12 TRAIN+6 VAL=18；每个场景 3 个视角 | 当前 place 仅 16 TRAIN、6,444 clips；新 view 合并 18 视频、7,298 clips |
| 评价视频 | 4 TEST，剩余第 6 个场景 | 6 TEST，每个场景剩余第 4 个视角 | 当前 place 是 4 VAL、1,386 clips；新 view 是 6 TEST |
| 每轮结果处理 | TEST 每轮评价、保留各轮权重、summary 按指标各自求 best | 同 place | place 沿用 VAL 主指标选模；新 view 按 TEST mAP@1.0 选单一 best_test，并用同轮其他指标，标 best-on-TEST |

两脚本来源：[place 启动脚本](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/scripts/train_cafe_place.sh)、[view 启动脚本](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/scripts/train_cafe_view.sh)。两套视频清单与 TRAIN+VAL 合并行为来自[同一数据入口](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/dataloader/dataloader.py#L10-L65)。本地 place 当前是开发对照，不能按 fork 默认脚本误写为已经完成 5 场景→1 场景 TEST。

| 核查项 | cherixi 固定版本的实际代码行为与依据 |
|---|---|
| 启动命令是否覆盖默认值 | place/view 均只指定 `data_path=Dataset/` 与对应 split，未传 `--val_mode`，也未传下列其他配置覆盖项。[place 脚本](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/scripts/train_cafe_place.sh)、[view 脚本](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/scripts/train_cafe_view.sh) |
| 数据流 | `val_mode=false` 分支合并 TRAIN+VAL 并加载 TEST；视频编号与本页上表相同。[数据入口](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/dataloader/dataloader.py#L14-L65) |
| 帧采样 | 默认 5 帧，`random_sampling=false` 进入分段随机分支；TRAIN 与非训练分支均使用 `L//5` 每段抽 1 帧，尾部余数不覆盖，TEST 不是固定中心。[采帧实现](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/dataloader/cafe.py#L185-L212) |
| clip 顺序 | 非 distributed 情况下 TRAIN、TEST 均用 RandomSampler；因此随机的是 clip 遍历顺序与 clip 内的帧，未重新划分 clip 窗口。[sampler 与 loader](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L178-L204) |
| 轮数与评价频率 | 默认 30 轮、`test_freq=1`，每轮调用 TEST validate。[默认参数](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L94-L104)、[训练循环](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L285-L335) |
| checkpoint 保留 | 每次 TEST 后保存 `epochN.pth`，没有删除旧 epoch 的逻辑；正常完成默认流程会保留 epoch1 至 epoch30。预测另按轮写文件。[权重保存](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L328-L335)、[预测路径](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L412-L422) |
| best 统计 | TEST 结果写入名为 `history['val']` 的字段，再分别记录 mAP@0.5、mAP@1.0 最大值和 loss 最小值及其各自轮次；未对 outlier 指标单独求 best。这些名称不改变其实际 TEST 身份。[调用处](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L311-L325)、[update_best](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/util/experiment.py#L47-L62) |
| 统一 best checkpoint | 没有 `best_test.pth` 保存，也没有以 mAP@1.0 为唯一准则同时绑定其余指标的选模实现；summary 中的两个 mAP best 可能来自不同轮。[完整保存分支](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L302-L335)、[逐指标 best](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/util/experiment.py#L47-L54) |
| 优化器、WD、LR | 该 fork 使用 **AdamW**，默认 WD=1e-4；head LR 为 1e-5→1e-4 的 CyclicLR，up/down=4/25、triangular2；backbone 默认 LR scale=0.1。它不是 dk-kim 原版的单组 Adam，也不是当前本地 Adam/WD0/scale1。[参数](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L47-L47)、[优化器和调度器](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L222-L262) |
| backbone 与冻结 | 两脚本均使用默认 `backbone=resnet18`、`frozen_batch_norm=false`、`freeze_backbone=false`，ResNet 分支未冻结其参数。`unfreeze_blocks=0` 是 DINO 分支的控制字段，不能解释成默认 ResNet 冻结。[backbone 参数](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L42-L47)、[ResNet 构造](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/models/backbone.py#L54-L75) |
| GPU、BN 与 batch | 默认设备 0、1，普通 DataParallel，训练 batch16、TEST batch16；默认 ResNet 使用普通 BN，未转换为 SyncBN。[设备与 batch 默认值](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L94-L112)、[DP 包装](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L206-L207)、[ResNet BN 分支](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/models/backbone.py#L54-L65) |
| 输入尺寸与 MAE | 两脚本均沿用默认输入 1120×630；未传 `--no_mae`，VideoMAE-v2 读取分支默认开启，实际特征取决于缓存。不能把这个 fork 的裸启动脚本视为论文式纯 B0 配置。[输入及 MAE 参数](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L32-L42)、[use_mae 派生](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/train.py#L148-L167) |
| 评价器 | 该 fork 仍将 group score 存入 set，并直接按该对象排序；未包含本地已固定的数值排序与显式 clip 清单修正。[group score 组织](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/evaluation/cafe_eval.py#L170-L188)、[排序实现](https://github.com/cherixi/CAFE_codebase/blob/1c5d5733d33eaed72867c5fe9bf4d5620fded3e9/evaluation/cafe_eval.py) |

本次只用该 fork 核实两套训练流。用户对新 view 的最新要求仍为本页顶部的 **best-on-TEST**：每轮保存，按 TEST mAP@1.0 自动保存统一 `best_test.pth`；summary 的 mAP@1.0、mAP@0.5、outlier mIoU、loss 均绑定该同一轮，不拼接各自最优轮。place 实际队列与开发口径不变。现有论文式 B0 的 Adam、WD0、backbone LR scale1、四卡 Host DP、输入 1280×720、无 MAE、完整候选采帧和修正评价器不因 fork 默认差异而自动变更；后续 TEST 固定中心采帧的本地差异也保留记录。

## 执行与记录

新队列已于 **2026-10-23 10:51:32+08:00** 启动，队列 PID **229326**，顺序为 **B0-final → H1-sup-final → SAM-final**。10:52 的实测状态为 `waiting_development_b0_exit`：旧两视角 B0 正在第 30 轮，完成并退出后自动交接新 B0-final；旧 B0 继续保留为开发对照，后续旧两视角任务安排由本新队列替代。实时进展见[队列状态](../../.worktrees/cafe-sam-person-mask-20261018/.local_data/view_final_20261023/queue_state.json)与[队列日志](../../.worktrees/cafe-sam-person-mask-20261018/.local_data/view_final_20261023/queue.log)，启动时间、PID 和实际命令见[启动记录](../../.worktrees/cafe-sam-person-mask-20261018/.local_data/view_final_20261023/launch.json)。

三份可读配置：[B0](../../.worktrees/cafe-sam-person-mask-20261018/configs/view_final_dp4/b0.args.json)、[H1-sup](../../.worktrees/cafe-sam-person-mask-20261018/configs/view_final_dp4/h1_sup.args.json)、[SAM](../../.worktrees/cafe-sam-person-mask-20261018/configs/view_final_dp4/sam.args.json)。实际执行使用冻结快照和[队列计划](../../.worktrees/cafe-sam-person-mask-20261018/.local_data/view_final_20261023/queue.plan.json)；[启动配置审计](../../.worktrees/cafe-sam-person-mask-20261018/.local_data/view_final_20261023/launch_config_audit.json)通过，包含 117 个参数字段及 107 个三方法公共字段检查，保留原有 105 个公共字段的一致性。源码等 90 份文件已按[文件 hash 清单](../../.worktrees/cafe-sam-person-mask-20261018/.local_data/view_final_20261023/source_manifest.json)冻结并复核；22 项 view-final 测试、5 项 best-on-TEST 策略测试及 TEST 曲线绘制 smoke 检查均通过。

正式训练数据为 **TRAIN+VAL 7,298 clips→TEST 2,414 clips**。SAM-final 必须使用覆盖 TRAIN+VAL 全部随机候选帧和 TEST 固定中心帧的独立、完整缓存，不能依靠缺缓存缩窄采帧范围。place 实际队列和开发口径不变。

运行控制字段固定为 `best_on_test=true`、`keep_all_checkpoints=true`、`protocol=best_on_test`、`eval_split=test`，主选模字段为 `selection_metric=group_mAP_1.0`。每轮的 TEST 预测、指标和 checkpoint 均保留；`best_test.pth` 的其他指标来自其对应的同一轮，不分别挑选。第 30 轮权重独立保留为 `epoch30.pth`。

修正评价器固定 SHA256 为 `f1cd5f05b63ec769c149ad1b8926922eb03bb70ab3665fb7e6bf75cec28b40a0`。正式结果同时保留完整参数、数据清单、源码与评价器 hash、全部 epoch checkpoint、best_test.pth、每轮 TEST 预测和指标，以及 best-on-TEST 选模记录。旧开发 B0 的续训 checkpoint 未保存完整 Python/NumPy/CUDA RNG 状态，因此虽能恢复模型、Adam 和 LR 进度，不能声称它与从未中断的 seed1 运行逐位相同；该限制随旧开发结果保留。[旧续训说明](view_dp4_resume_20261022.md)
