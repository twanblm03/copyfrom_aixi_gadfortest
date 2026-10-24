# B0 与 Café 论文的协议比较及后续基线建议（2026-10-21）

建议后续 heatmap / SAM 消融统一采用今天完成的 `b0_randomseg_dp4_lr1_noclip_seed1` 所属协议：ResNet18、TRAIN 随机分段采帧、VAL 固定中心帧、四卡 Host DataParallel、全局 batch16、修正评价器、TRAIN→VAL 选模。固定的是训练与评价协议；19.0741 是这一协议下现有的 seed1 参考结果，尚不是多 seed 均值。

同日补充：[旧30轮完整重选](old_30epoch_reselection_20261021/README.md)已完成。相同修正评价器下，旧B0最佳为epoch28、18.3370，新B0为19.0741（+0.7370pp）；旧H1-sup仍为epoch21、22.9485，新H1-sup为21.9788（−0.9697pp）。因此选择当前开发协议的依据是固定比较条件，不是已证明它使两模型共同涨分。旧B0 epoch28只保留完整预测，未找到该轮权重。

用户已明确要求后续不再使用有问题的旧评价器。未来训练、评价及选模统一使用数值置信度排序、固定评价clip清单的修正版，旧实现仅保留为历史归档，不再执行。根目录和当前研究worktree的默认评价实现及train/test入口均按该规则核对。

报告中的三个名称属于结果阶段：`new_dp4_corrected_evaluator`为新DP4训练并按修正VAL指标选模；`old_reselected_corrected_evaluator`为旧30轮在修正指标下重新选模；`old_original_corrected_evaluator`为保留旧已选epoch、仅重算指标。三者不是三种评价器。后续采用第一项对应的训练协议，第二项作为历史公平对照，第三项用于分解评价器修正的影响。

这是一套基于论文主要设定的内部开发基线。它与官方发布脚本有明确差异，应在报告中保留这些差异，不能标为官方协议的完全复现。以下比较区分论文文字、官方代码默认行为和本地真实运行。

| 项目 | 论文 / 官方发布代码 | 当前建议保留的协议 |
|---|---|---|
| 模型与输入 | 论文：ImageNet ResNet18、GT actor tracklets、5帧、12 group tokens、256维、6层4头、5×5 RoIAlign | 主要设定一致；实际入口为 `research_b0_repro=true`，不启用 DINO、VideoMAE、HOI/额外 temporal encoder、heatmap 或 SAM |
| 优化主体 | 论文：Adam、batch16、30轮；LR 从1e-5在前5轮升至1e-4，再线性衰减；membership/consistency 系数5/2 | 一致；backbone LR scale=1；无梯度裁剪；保留当前实现的损失及解码 |
| Weight decay | 论文未给出；官方代码默认1e-4 | 实际为0；这是明确的实现差异，不改写已有运行的配置 |
| LR 衰减终值 | 论文未给出；官方 CyclicLR 的训练第1/5/6/30轮实际LR为1e-5 / 1e-4 / 9.64e-5 / 1e-5 | 当前分别为1e-5 / 1e-4 / 9.6e-5 / 0；以保存的30轮实际LR为准 |
| TRAIN 采帧 | 论文为分段采帧；官方代码每段随机一帧，用 `L//5` 划分，余数尾部未覆盖 | 每段随机一帧；用 `linspace(0,L,6)` 的分段边界覆盖全片段。属于相同策略下的边界实现差异 |
| VAL 采帧 | 论文未明确；官方代码验证分支也随机采帧 | 每段确定中心帧；固定 checkpoint 的验证输入可重复 |
| 并行与BN | 论文未报告训练GPU数；官方代码默认双卡 DataParallel、普通BN | 四卡 Host DataParallel、普通BN、每卡4 clips；每卡约20帧参与BN，而双卡约40帧。GPU数是协议的一部分 |
| 数据与选模 | 论文有独立TRAIN/VAL/TEST；官方默认脚本不传 `--val_mode`，会TRAIN+VAL→TEST，每轮评估并保存权重；未实现best选择，不能据此推断论文如何选模 | 开发阶段TRAIN→VAL，每轮按完整VAL的group mAP@1.0选择单个checkpoint，其余指标从同一checkpoint报告；不以TEST选模 |
| 评价器 | 官方代码将单元素confidence set直接作为排序键，不能实现预期的数值置信度降序 | 解包为浮点数排序，固定同分排序；明确限定VAL clip清单。保留该修正 |

论文来源：[论文§5.1](https://arxiv.org/html/2312.02878v2#S5.SS1)、[附录数据划分](https://arxiv.org/html/2312.02878v2#A1.SS2)。官方实现来源：[训练参数及优化器](https://github.com/dk-kim/CAFE_codebase/blob/main/train.py)、[采帧与过滤](https://github.com/dk-kim/CAFE_codebase/blob/main/dataloader/cafe.py)、[划分及val_mode](https://github.com/dk-kim/CAFE_codebase/blob/main/dataloader/dataloader.py)、[place启动脚本](https://github.com/dk-kim/CAFE_codebase/blob/main/scripts/train_cafe_place.sh)、[评价器](https://github.com/dk-kim/CAFE_codebase/blob/main/evaluation/cafe_eval.py)。以上官方main源码于本次审计在线读取。

官方LR不能仅凭“CyclicLR”这个名字判断：up=4、down=25、每轮结束后step，在30轮内也是一次先升后降；实际区别是衰减的数值轨迹与终值。[PyTorch1.7.1实现](https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/lr_scheduler.py#L809-L822)。本地summary已确认第30轮训练LR=0，不能将该轮描述为非零LR更新。

数据计数也需保持口径：当前place TRAIN/VAL为6,444/1,386 clips。论文附录给出原始TRAIN/VAL为6,709/1,423；官方解析器排除没有已标注group的clip，本地沿用该过滤。它们不是两个不同的place划分；当前结果不能描述为覆盖全部原始1,423条VAL的评估。

**为何选择当前协议。** 随机分段TRAIN采帧与论文的方法设定相符；固定VAL采帧和修正排序适合持续开发与可靠选模；四卡环境可直接沿用；DataParallel先汇集完整batch再求loss，也保留了原有损失求值范围。选择依据是方法一致性、可比性和后续执行条件，不是选择对某种辅助方法最有利的B0分数。

| 已有版本 | 后续定位 |
|---|---|
| 今天完成的随机分段、DP4版本 | 推荐作为后续开发协议和seed1参考；所有候选方法共享它的共同设置 |
| 10月13日完成的随机分段、DP2版本 | 历史对照；其BN局部批量、环境和评价/选模不同，不直接替代当前B0 |
| 10月19日启动、20日汇总的固定中心TRAIN、DDP版本 | 可保留为采帧/并行消融；其B0严格mAP=20.8631高于当前，但不能仅按单seed最高分选协议 |
| 早期低backbone LR、途中停止的matched版本 | 未完成30轮，不作为完整基线 |
| 根目录原 `BASELINE_PROTOCOL.md` 的DINO/8帧/MAE方案 | 属于另一套增强模型，不是本次论文式ResNet18 B0 |

**后续固定条件。** 保留当前模型、数据过滤、采帧算法、Adam/WD0/LR轨迹、普通BN、DP4、每卡batch4、FP32、完整batch损失、阈值与解码、修正评价器及VAL选模。H1只改变热图辅助项，SAM只改变其研究分支；共同条件不随分支变动。每个seed均从相同规则的ImageNet初始化重新训练，新增模块不扰动共同参数初始化和采样随机流。更换GPU数量、DDP、SyncBN、学习率、WD、采帧或损失归一化时，应建立新的对照并重训B0。

本次选定后续协议方向，尚未完成稳定baseline的全部复现验收。当前入口使用未经排序的目录列表、全局RandomSampler RNG和2个persistent workers，尚未留存本次完整运行的输入顺序/采样帧/初始化hash与完整源码快照。后续启动前应保存有序clip清单、实际配置、初始状态、环境和源码hash，以及输入顺序核验记录。若为了复现性而改排序或独立sampler generator，会改变随机流，需要重新生成B0参考，不能把19.0741补写为变更后的成绩。seed2/3与固定checkpoint重复评价结果也尚待补齐；本次没有新增训练或复评来冒充这些验收。

**分数与论文比较的边界。** 当前B0按VAL严格mAP选择epoch21，同一checkpoint为19.0741 / 37.0989 / 63.8847（mAP@1.0 / mAP@0.5 / outlier mIoU）。论文Table2的place、12-token结果为10.85 / 30.90 / 63.84，属于论文测试基准，不能据此宣称当前B0超过原论文：split、采帧、评价器与最终训练口径不一致。正式论文对比应在开发结束后固定配置与训练轮数，再用TRAIN+VAL训练、在TEST评价；同一评价器下重新评价官方参考模型和本地模型。若同时给出旧官方评价器结果，应单列其口径。

本地证据：[本轮结果](../../.worktrees/cafe-sam-person-mask-20261018/results/sam_person/h1_dp4_randomseg_full.json)、[启动说明](../../.worktrees/cafe-sam-person-mask-20261018/docs/sam_person/h1_dp4_randomseg_repro_20261020.md)、[原始args及审计记录](b0_baseline_vs_cafe_paper_20261021.json)。源码hash标注的是本次审计时状态，不追认它们为训练启动时快照。
