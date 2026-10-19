# H1-Sup-place 拥挤约束：seed1 同样本诊断（2026-10-17）

结论：B 在本项目现有冻结评价流程下，最佳 checkpoint 的严格 mAP 提升 **0.7302 pp**，但没有表现为成员分组的全面改善。拥挤子集的 individual 混入减少，同时真实成员召回下降；全 VAL 的 no-group 拒绝、成员 F1 和集合 IoU 均变差。当前证据支持“局部收益与召回损失并存”，不支持“已解决跨组误分”。另查出冻结评价器的置信度排序问题，见末节；本次没有修改评价器或替换历史指标。

## 产物与核验

- A：`/home/zhenyu/CAFE_heatmap/remote_h1_sup_seed1_full_run1/`，未收紧目标的 **H1-Sup**，不是 H1-Match；best epoch21。
- B：`outputs/heatmap/place/full_h1_sup_crowding_seed1_dualgpu23_20261017_1000/h1_sup_crowding_full/`；best epoch28。
- 诊断目录：`outputs/heatmap/place/h1_sup_crowding_comparison_20261017/`。
- 两组完整训练均为 30 epochs、12,060 steps；已逐步核对 LR 相同。配置除 runtime 外只差 `heatmap_target_mode: nearest_center`。
- 两组 data manifest、split、官方 GT 文件和 evaluator 哈希一致；已检查源码快照哈希、checkpoint epoch/metrics、预测哈希，并用 CPU 从 best 导出文件复算全部主指标，精确复现。
- 全部 **1,386** 个 VAL clip 的五个采样帧、人物框坐标和行顺序均与冻结输入核对一致。旧 H2 诊断仅提供 boxes/validity/GT，不使用其预测、概率或 slot。
- 新增分析测试 6 项及原 saved-set 测试 2 项全部通过；**CUDA 未初始化，模型前向 0，optimizer updates 0**。原产物未覆盖。

重要限制：A 是双 A6000、host 传输、persistent workers；B 是双 3090、原生 DataParallel、非 persistent workers。两组记录中的 B0 初始化 hash 和 initial RNG hash 相同，但训练样本/采帧记录从 **step403（epoch2 起）**即不同；A 未记录完整初始模型状态 hash。因此这是同样本评估的历史对照，**不是严格配对训练，也不是单变量因果证明**。源码差异已保存到 `source_diffs/`；H2 transition/center readout 开关没有在本次 H1-Sup 配置中启用。

## 主结果：各自 VAL-best 单个 checkpoint

以下数值单位为 %，差值单位为百分点；没有混用不同 epoch 的指标。

| 配置 | best epoch | group mAP@1.0 | group mAP@0.5 | outlier mIoU |
|---|---:|---:|---:|---:|
| A：H1-Sup | 21 | 18.4665 | 39.8956 | 73.3826 |
| B：H1-Sup-Crowding | 28 | 19.1968 | 38.0967 | 71.8678 |
| B − A | — | **+0.7302** | **−1.7989** | **−1.5148** |

注意：前面根据截图四舍五入后估出的 `+0.7303`，这里用原始 JSON 精度计算应为 `+0.7302 pp`。

严格 AP 的主要增益落在 Working/Studying 和 Fighting，并非所有类别一致改善：

| 类别 | A AP@1.0 | B AP@1.0 | 差值 |
|---|---:|---:|---:|
| Queueing | 0.0286 | 0.1250 | +0.0964 |
| Ordering | 0.3795 | 0.3948 | +0.0153 |
| Eating/Drinking | 27.4281 | 24.5868 | −2.8413 |
| Working/Studying | 39.8220 | 44.7386 | +4.9166 |
| Fighting | 22.2864 | 27.2214 | +4.9350 |
| TakingSelfie | 20.8547 | 18.1141 | −2.7407 |

`epoch_metrics.csv` 保存两组全部 30 轮历史。若固定比较 epoch30，原记录分别为 17.7304 / 38.9269 / 72.3541 与 18.0812 / 37.2113 / 71.7869；严格 mAP 差值为 +0.3508 pp，仍不是另外两项指标的全面改善。本次 CPU 重放的是 best 导出，非全部 60 个 epoch 导出。

## 预注册拥挤子集

定义没有根据成绩调整：五帧中任一帧存在有效异组人物或 group–individual 人物对，框 IoU≥0.3，即归入拥挤子集。同组人物对及 individual–individual 人物对不触发该标记。中心距离使用 23×40 网格，按同帧相关人物对、跨五帧取最小值。

成员指标的诊断口径：先按原导出得到最终有效活动组，要求至少两人；忽略活动和 score，以成员集合 IoU 作一对一 Hungarian 对齐。未匹配预测成员计 FP，未恢复 GT 成员计 FN；集合 IoU 以全部 GT 组为分母，未匹配 GT 记 0。跨组误分率的分子是已匹配预测组内、属于其他 GT 组的人次，分母是这些已匹配预测组中的全部 GT 成员人次；未匹配预测组中的 GT 成员另列，不强行赋予群组身份。这些是**描述性诊断，不是 AP 的匹配规则，也不是训练 matcher 的分配**。

| 范围 | clips / GT组 | A→B 成员 P | A→B 成员 R | A→B 成员 F1 | A→B 集合 IoU | A→B 跨组误分率 |
|---|---:|---:|---:|---:|---:|---:|
| 全 VAL | 1,386 / 2,224 | 87.03→86.01 | 70.84→67.19 | 78.10→75.45 | 64.55→61.23 | 4.73→5.23 |
| 拥挤 | 401 / 654 | 82.74→82.68 | 60.19→56.78 | 69.69→67.32 | 51.88→48.14 | 6.01→7.68 |
| 非拥挤 | 985 / 1,570 | 88.49→87.12 | 75.07→71.33 | 81.23→78.44 | 69.83→66.69 | 4.32→4.42 |

对应的冻结评价器子集结果：

| 范围 | A→B mAP@1.0 | A→B mAP@0.5 | A→B outlier mIoU |
|---|---:|---:|---:|
| 拥挤 | 11.0434→12.1210 | 35.3214→31.0288 | 66.0753→66.2204 |
| 非拥挤 | 21.6316→22.1976 | 41.6992→40.3100 | 76.3575→74.1669 |

子集 AP 非线性、类别组成不同，不能加权相加来分解全 VAL AP，不能据此认定严格 AP 增益均来自拥挤场景；零正例类别也沿用原评价器约定。

距离分桶的成员 F1 / 集合 IoU（%）：

| 最小相关中心距离 | clips / GT组 | A→B F1 | A→B 集合 IoU | A→B 集合与活动均正确组数 |
|---|---:|---:|---:|---:|
| <1 网格单位 | 39 / 67 | 63.59→64.02 | 43.98→42.84 | 14→15 |
| 1–2 网格单位 | 219 / 361 | 66.14→63.54 | 46.99→44.07 | 76→81 |
| ≥2 网格单位 | 1,128 / 1,796 | 80.54→77.73 | 68.85→65.37 | 826→810 |

<1 的样本很少，不能据其单个正确组净增，声称已克服网格分辨率限制。此处距离桶与 IoU 拥挤桶是两种不同划分。

## 失败分解：收益与代价在哪里

| 诊断项 | A | B | B − A |
|---|---:|---:|---:|
| 全 VAL 最终检测组数 | 1,764 | 1,661 | −103 |
| GT组：集合与活动均正确 | 916 | 906 | −10 |
| GT组：集合精确正确，不计活动 | 1,007 | 991 | −16 |
| GT组：与任何检测组都无成员交集 | 447 | 549 | +102 |
| GT成员被 no-group 拒绝 | 1,601 | 1,929 | +328 |
| GT成员因阈值拒绝 | 422 | 378 | −44 |
| 被混入检测组的 individual（全 VAL） | 557 | 573 | +16 |
| 被混入检测组的 individual（拥挤） | 195 | 164 | **−31** |
| 被混入检测组的 individual（非拥挤） | 362 | 409 | **+47** |

这说明拥挤子集确有 individual 排除的局部收益，但这与 no-group 拒绝更多真实组员、最终组检出减少同时发生。不能把“跨组混并拓扑标记数减少”单独解释为分离更好：组检出减少本身也会减少这类标记。按已对齐成员计算，拥挤跨组误分人数为 **88→108**，误分率也升高。

逐 GT 组的同样本转移：全 VAL 有 224 组从错误转为“精确集合+正确活动”，另有 234 组从正确退化为错误；拥挤子集分别是 56 和 58 组。AP 是逐类宏平均且依赖评价器的序列计算，并不等于正确组数；不能把 +0.7302 pp 直接换算成正确组增加。

预先已知的两个 annotation/官方 GT 不一致 clip `(13,5)`、`(15,181)` 保留在主指标中，没有改标签。排除它们的敏感性检查仍为 A 18.3901、B 19.1044，严格 AP 差值 +0.7142 pp；成员 F1 仍为 78.10→75.43，结论未翻转。

## 已确认的评价器排序风险：单列，不修改主协议

只读核验发现，`evaluation/cafe_eval.py::make_groups` 把组 score 保存为 `set`，`group_mAP_eval` 随后直接用这个 set 作 `sorted` 的 key，而非取出浮点置信度。不同单元素 set 的比较并不是数值大小比较。

已对 A/B 的六个类别分别实测：各组 score 均为单元素 set；现用排序均保留导出迭代顺序，且**均不等于数值置信度降序**。证据及首三个预测 score 保存在 `evaluator_ranking_audit/ranking_audit.json`。例如 A 的 Ordering 序列头部置信度为 `0.5846, 0.9932, 0.6689`，并非降序。

这是本项目冻结 evaluator 中既有的行为，不是 B 的修改造成；尚不能据本机文件断言它与论文发布版的关系。本次没有改 evaluator、没有按另一种排序重算并替换历史成绩，也没有将 AP 增益解释为置信度校准改善。下一步应优先独立审计评价协议；如需纠正，必须对 B0/H1/H2 的现存导出统一重放、同时保留 legacy 结果，不能只给 B 换评价器。

## 复现、留存与下一步边界

实际执行：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .miniconda3/envs/gad/bin/python tools/analyze_heatmap_crowding_control.py --baseline /home/zhenyu/CAFE_heatmap/remote_h1_sup_seed1_full_run1 --candidate outputs/heatmap/place/full_h1_sup_crowding_seed1_dualgpu23_20261017_1000/h1_sup_crowding_full --input-diagnostic outputs/heatmap/place/h2_s_failure_margin_fullval_20261014_1354 --output outputs/heatmap/place/h1_sup_crowding_comparison_20261017
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .miniconda3/envs/gad/bin/python -m unittest tests.heatmap.test_crowding_analysis tests.heatmap.test_saved_set_audit -v
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .miniconda3/envs/gad/bin/python tools/audit_heatmap_export_ranking.py --baseline /home/zhenyu/CAFE_heatmap/remote_h1_sup_seed1_full_run1 --candidate outputs/heatmap/place/full_h1_sup_crowding_seed1_dualgpu23_20261017_1000/h1_sup_crowding_full --output outputs/heatmap/place/h1_sup_crowding_comparison_20261017/evaluator_ranking_audit
```

输出包括真实命令、`analysis.log`、`tests.log`、分析源码快照、`audit.json`、`verification.json`、`comparison.json`、逐 actor/GT组/clip CSV、逐类 AP、完整 epoch 历史、每个子集的原分数预测与 GT，以及逐组 A/B 转移表。`cases.json` 固定保存按 clip/组顺序选取的三个拥挤成功与三个退化案例，含 actor ID、框、GT 和两组最终集合；它们不是重新选择的评估子集。

完整 VAL 的原始热图、embedding membership 和所有 slot logits 未保存，不能从最终 TXT 倒推出 slot margin 或因果机制。**本次只完成 CPU 的结果层失败分解**；热图/embedding 的定点机制核查仍需用两组 best checkpoint 做受限前向，不需要重新训练。未启动额外训练、没有调阈值、没有依据 VAL 选择新的高斯半径。建议先厘清上述评价器问题，再决定机制前向及同机严格配对重训预算。
