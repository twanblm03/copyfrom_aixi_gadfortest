# CAFE heatmap 首轮执行报告 — 2026-10-12

后续四卡有界续跑的结果和硬件故障交接见 [continuation_20261012.md](continuation_20261012.md)
及 [handoff_20261012.md](handoff_20261012.md)。下文保留首轮短实验的历史结论，
不是最新500步结果；完整训练仍未启动。

本轮已完成代码最小闭环、真实数据审计、oracle、GPU smoke 和严格 paired pilot。
现有证据支持目标/ROI几何与梯度接线正确，**不支持 heatmap 改善群组检测的结论**。
短实验仍处于 no-group/近均匀 membership；过拟合 gate 未通过，完整训练未启动。

## 核验与实现

B0 取自 `exp/b0-repro`，不取 main 的 DINO/VideoMAE 增强基线。
保持 ResNet18、5帧、12 frame-specific slots、720×1280、原 activity mean、K-softmax membership、
原 decoder/threshold/consistency、Adam及完整训练的5-epoch warmup/linear decay协议。
数据为 place TRAIN 6,444 / VAL 1,386，TEST未使用。审计细节见
[baseline_audit.md](baseline_audit.md)，接口与变量控制见
[implementation_contract.md](implementation_contract.md)。

新增可关闭 mask head、Gaussian member-max targets、explicit track/frame validity、
可微 ROI/time pooling、soft focal/Dice/unmatched empty、共享 clip assignment、
H1-Sup/H1-Match/H2-S。H2-S 的 membership/cost 确实来自预测热图，保留原 K-softmax/BCE；
没有 embedding fallback。H3、独立sigmoid membership、SAM、FPN、explicit background尚未开展。

15项宿主机合同测试通过，包括实际 CUDA 上的 membership-only 梯度和 GT 隔离、
CPU/CUDA RNG、关闭功能逐元素一致、有效性、空GT、slot/actor permutation、原导出/evaluator往返。
原有8项 DINO helper tests也通过。原始日志：
`outputs/heatmap/place/contract_tests_cuda_runnerfinal_20261012.log` 和
`outputs/heatmap/place/dino_regression_final_20261012.log`。
另外完成全部改动Python的编译检查、shell语法检查及 `git diff --check`。

## Oracle：只代表几何上限

主诊断使用固定manifest等间隔的64 TRAIN / 32 VAL clips。仅按训练集比较
ratio=.15/.25 与 alpha=5/10/20/40，冻结 ratio=.15、alpha20、sigma clamp[.5,4] grid units。
ROIAlign=3×3、sampling_ratio2、aligned=True，网格23×40。

| H2-S ideal-target oracle | 成员 F1 | 集合 Group IoU | 完全恢复群组比例 |
|---|---:|---:|---:|
| 训练64 clips，101 groups | .9852 | .9754 | .9109 |
| 独立诊断VAL32 clips，52 groups | .9869 | .9731 | .9038 |

理想图使用 GT target slots + zero occupancy unmatched slots，经12-slot softmax和原.5阈值读取，
不含真实模型的分类/定位/slot学习误差，不是 mAP。raw ROI threshold=.13的结果也独立保留。
ratio=.25 的 raw ROI 在train上更好，但保留原softmax/threshold的整体读取并非更好，故没有按VAL挑选它。
完全重合框在合成 oracle 中读数严格相同，构成 heatmap-only ROI 的表示上限；近邻/尺度分桶与
全部响应在 `outputs/heatmap/place/oracle_spread_20261012/oracle.json`。

## 发布 checkpoint 参考复评与 probe

`cafe_place.pth` 在完整1,386 VAL clips复评两次，预测内容hash完全相同：
`0a9bfd92d1ff25c8c40d118308e692898d6fa1f180f74783cea9824b1eda1c35`。
实测 mAP@1.0=92.58、mAP@.5=98.86、outlier mIoU=99.30；per-class AP在相同输出中保存。

该checkpoint没有训练args、split或val-selection元数据，可能已经见过VAL，
**这些分数不能当作本轮开发B0，更不能与短训练结果计算方法增益**。
加载需显式忽略不参与冻结B0 forward的旧 bbox_emb 六个参数；第一次严格加载失败保留。

Probe 只取最后decoder layer的原生head平均spatial softmax，不改变预测。
按原assignment匹配后的ROI响应均值：member=.001958，nonmember=.001140。
这说明该已训练参考有空间相关信号，不说明未训练的独立mask head会学好。
数据、NPZ、响应分布和可视化在 `released_reference_val_compat_20261012`，不是独立性能实验。

## 实际短实验

3-step H2-S smoke：原未拆分batch16、3张已分配3090、FP32，forward/loss/backward/step/
保存加载/原导出/重复评价均通过，主卡峰值约13.3GiB（13,654MiB）。
早期单GPU微批smoke仅作工程诊断，不混入受控结果。

严格 paired pilot 使用 seed1、同一64-clip训练集/32-clip VAL诊断集、8 optimizer steps、
batch16、micro_batch0、Adam 1e-5、相同初始化和采样帧顺序。8 steps均在原完整首epoch
402 steps以内，使用warmup的初始LR；不是8个完整epochs。各组保存last bounded-step checkpoint，
不根据短VAL结果挑选epoch或阈值。

| 配置 | mAP@1.0 | mAP@.5 | outlier mIoU | warm秒/step | 主卡峰值GiB |
|---|---:|---:|---:|---:|---:|
| B0 | 0.00 | 0.00 | 43.87 | 3.086 | 13.321 |
| H1-Sup | 0.00 | 0.00 | 43.87 | 3.176 | 13.337 |
| H1-Match | 0.00 | 0.00 | 43.87 | 3.148 | 13.337 |
| H2-S | 0.00 | 0.00 | 43.87 | 3.208 | 13.334 |

这不是正式性能对照：单seed、128个训练clip exposures、只32个诊断clips。
H1/H2共384个VAL slots均选择no-group，匹配成员F1=0；H2成员/非成员mask ROI均值
.12013/.11979，几乎重叠；mask Dice loss由.96818到.96974，没有改善。
原 B0同样还未学会group，不能由这8步认定 H2 比 B0好/坏。

四配置共同权重hash、CPU/CUDA RNG hash、源码快照、完整sample/frame顺序、dataset/evaluator、
batch/steps全部检查通过，见 `results/heatmap/paired_checks.json`。
第一轮pilot发现CUDA seed混杂后已判为不可作配对结论，原结果保留；只解释
`paired_pilot8_rngfixed_seed1_20261012` 这轮。

固定16 TRAIN clips的首轮32-step overfit（修复CUDA RNG前）未通过：individual CE
4.579→.070，group CE 2.531→.539，但 group class error=100%、member BCE .2048→.2045、
mask Dice .96885→.96900，训练集合mAP@1.0/.5均0。它只是独立负诊断，不用于配对结论。
RNG修复后的同预算overfit也已完成，路径为 `h2_s_overfit32_rngfixed_dp3_20261012`：
individual CE 4.65813→.05791、group CE 2.55378→.53480，但 group class error仍100%，
member BCE .20463→.20410、mask Dice .96875→.96892；训练集mAP@1.0/.5仍0，
outlier mIoU=40.77。192/192个训练slots为no-group，member/nonmember ROI均值
.114490/.114199；32步训练289.83秒，总wall324.68秒。mask梯度有限非零但未形成成员分离。
两轮均未通过overfit gate；只有32步，不能据此宣称方法无法收敛。

失败分析优先级是早期no-group占优、mask/member响应不分离、分项梯度/权重尺度和短预算优化。
mask gradient非零、oracle可恢复大多数成员，当前没有证据指向坐标错误或必须更换backbone。
不得因全零短mAP改阈值/归一化、引入H3或覆盖失败结果来制造提升。

## 下一阶段与实测资源需求

下一轮将有界overfit扩展到最多500steps，并在固定train小集检查 matched/unmatched
mask、member separation、no-group率。可先做少量train-only分项梯度和 BCE+Dice 诊断；配置已提供，
不按VAL搜索。该gate通过后再做最长1000-step paired pilot，而不是直接多seed完整矩阵。
H3-P/H3-A及其他表示/归一化分支届时分别设gate。

当前计量：3张24GB3090、并发1个DataParallel训练；温态约3.1–3.2秒/step（不含startup）；
overfit每一小集pass重启worker，最终32step训练289.83秒，不能把它当大数据集稳定吞吐。
完整B0为 floor(6444/16)=402 steps/epoch，共12,060 optimizer steps；
VAL一次约344秒（由重复参考复评总wall/2粗估，包含probe/启动，非独立profiling）。
据此单配置单seed约13–14小时；完整B0/H1-Match/H2-S三配置×3seeds约120小时串行wall，
预留三卡约360 GPU-hours；若也完整跑H1-Sup则约160小时/480 GPU-hours。
这是本机器当前runner的外推，至少预留±30%，I/O、类别组成、GPU并发及验证频率均会影响它。
现有完整训练预算不明确，且overfit gate未通过，本轮没有启动这一矩阵。

最后补齐runner协议保护：full必须全TRAIN/VAL、batch16且按固定mAP@1.0选择checkpoint；
选中clip清单记录实际全集。以后有界运行的LR按完整TRAIN的402steps/epoch推进，
不是每遍历16/64个clips便完成一个原B0 epoch；500/1000步运行会在step402、804改变LR。
这些保护在本轮实验后加入并通过单测；全部已执行短跑低于402steps，其LR和数值路径不受影响。
每个已执行run保留自己的源码快照，不拿最终源码hash回填旧记录。

下一阶段500-step overfit按当前小集路径约80分钟/三卡；1000-step四配置pilot约3.6小时纯训练/
三卡，另计验证、启动与checkpoint。具体继续命令见下文；这些是待执行配置，不伪装为已完成run。

```bash
# 下一次有界overfit：独立目录，最多500steps，固定16 TRAIN clips。
.miniconda3/envs/gad/bin/python tools/run_heatmap_research.py \
  --stage overfit --config configs/heatmap/h2_s.json --steps 500 \
  --train-clips 16 --eval-clips 32 --workers 2 \
  --device cuda:0 --parallel-gpus 0,1,2 \
  --output outputs/heatmap/place/h2_s_overfit500_next

# Gate通过且已有预算后，再进行有界paired pilot。
bash scripts/run_heatmap_paired_pilot.sh --steps 1000 --seed 1 \
  --root outputs/heatmap/place/paired_pilot1000_next --gpus 0,1,2

# 正式矩阵中单个run（预算和gates满足后）：全TRAIN、全VAL、best_val按mAP@1.0。
.miniconda3/envs/gad/bin/python tools/run_heatmap_research.py \
  --stage full --config configs/heatmap/b0.json --seed 1 --eval-clips 0 \
  --workers 2 --device cuda:0 --parallel-gpus 0,1,2 \
  --output outputs/heatmap/place/full_b0_seed1_next
```

## 产物索引

- 实现与配置：`models/heatmap.py`、`models/heatmap_criterion.py`、`configs/heatmap/`。
- 执行/导出：`tools/run_heatmap_research.py`，原decoder副本 `util/group_export.py`。
- 原始命令、全run表、配对核验：`results/heatmap/commands.md`、`run_registry.csv`、
  `comparison.csv`、`paired_checks.json`、`invalidated_runs.json`。
- 严格pilot：`outputs/heatmap/place/paired_pilot8_rngfixed_seed1_20261012/`。
- 最终overfit：`outputs/heatmap/place/h2_s_overfit32_rngfixed_dp3_20261012/`。
- 每run：配置/runtime、环境、revision/diff、源文件快照/hash、split/data manifest、init hash、
  sample/frame顺序、steps.jsonl、train.log、checkpoint、text预测、per-class/aggregate指标、NPZ。
- 固定前4个VAL诊断clips的12-slot连续热图、GT targets/成员、activity、matching和响应分布：
  对应run的 `visualizations/`。这些是固定样本而非挑选的成功例；全5帧连续图保存在NPZ。

目前只完成工程闭环，未建立可认证的三seed B0、未通过充分overfit、未完成完整训练或H3研究。
所有正式有效性问题仍须后续同协议实验回答。
