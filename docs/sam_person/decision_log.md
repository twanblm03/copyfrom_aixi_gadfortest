# 决策记录

- 2026-10-18：只新增独立 SAM person namespace，保留原有 heatmap 和 B0-repro 工作区改动。
- 用户先批准 GPU4，随后明确 GPU4/5/6/7 均可用；总上限不变：生成1 GPU小时、训练2 GPU小时。
- 使用已经安装且有权重的官方原版 SAM ViT-B，FP32、关闭 TF32 生成，box_only 单候选，阈值0；不冒充 SAM2/2.1，不升级依赖。
- 新缓存保留原图二值 mask，按 video/clip/原图文件帧/actor row 显式绑定；checksum、源图/框/model/config fingerprint 和原子完成记录全部校验。
- 不复用旧的5×5 cropped mask缓存；不做面积/score过滤，不强裁框、不取最大连通域、不做 NMS。
- 首轮 QC 固定120 TRAIN+80 VAL帧。采样先冻结，提示策略不在 VAL 调参。缺像素GT，IoU/Dice=null，质量预测只叫 predicted_quality。
- 当前几何增强只有 resize；测试覆盖共享 resize/crop/flip 约定；训练投影后保持连续覆盖。
- MaskRoI 使用原框和原 RoIAlign 参数，mask 特征网格必须等于真实 backbone shape；raw失败/空/局部网格近空才回退普通RoI，计数且不删actor。
- CPU四个测试方法已通过：编码/几何/身份/损坏拒绝，参数与RNG、原B0输出/loss/导出、ones误差0、梯度/因果/GT隔离。
- 数据盘全目录扫描过慢，改为既有审计 split manifest抽样并核对实际所选ann哈希；原解析器仅新增可选 clip_ids，默认路径不改。
- pilot预声明：16TRAIN/8VAL clips，8TRAIN overfit，seed1，batch4，两侧同一 checkpoint独立warm start；smoke3 batches，overfit40步，pilot80步，固定最后一步，不新增验证选模。
- 不运行 H1-SAM-TARGET 或 S2F-SCALE；前者可后续独立验证，后者仅在S2F有收益证据后考虑。
- QC生成共享盘逐人原子记账开销较大：保留完成缓存并中断本任务自己的生成进程，改为逐帧预留全部prompt额度、逐帧原子落账；中断帧预留不退回，active_since恢复保守补计耗时。SAM模型、提示和mask处理不变，失败与重试时间不清零。
- 首次真实GPU全1检查失败：RoIAlign(ones)最小值0.9999999403953552，原B0重复前向误差0，但TF32将该级舍入放大至activity最大6.32e-4。保留失败B0和已完成的旧S2F 3batch smoke，停止后续训练。
- GPU诊断验证：关闭TF32后直接插值误差约1.91e-6；保留原TF32、使用1-RoIAlign(1-S)后全1输出误差0。最终mask采样同样使用原B0的sample_valid零外推；不放宽容差、不更改B0精度flags。
- 修复后6个CPU测试通过，随机非恒定mask在内部网格与直接插值近似等价。200帧重做QC，仍2054/2054非空，无网格近空。
- 最终配对run位于paired_stable；两侧各40步overfit（前3步兼作技术smoke）和80步pilot，无额外smoke更新。旧S2F 3batch只作为被数值门槛否决版本的失败记录。
- 主训练进程上限缩至合计1.8 GPU小时，另留0.2 GPU小时给诊断；复用原累计ledger，所有失败和中断保守计时。完整训练未授权、未运行。
- 初次失败run源码已归档：source_manifest SHA256逐字节等于config_actual中记录的1bb6be87df7ec16a23178ef7538904665303bb42cfe02af5529ea77e3bd22a97。
- 80步pilot主指标下降、宽松指标改善，且VAL仅8clips/13个真实组；在原预算内追加唯一一组成对诊断：24TRAIN/16VAL、固定300步。仅用已冻结200帧缓存，不新增SAM提示或后处理，两侧都重新从相同checkpoint开始；累计pilot更新80+300=380≤500。
- 追加诊断同时扩大训练/验证清单和更新数，不能独立归因于训练时长。保留原80步结果，不做VAL最佳步选模。结束后停止扩大。
- B0全1检查残留last_mask_stats造成旧80步日志计数错误；mode始终disabled，forward/loss/预测/权重未使用该属性。按实际sample_order重建有效actor-frame计数（overfit8220、pilot16380），SAM使用次数均0，原文件不改，修正见diagnostic_corrections.json。后续仅清理该日志状态，追加实验采用新的独立source hash。
- 原evaluator存在预测clip顺序敏感性，CPU反序复核已保存；原实现保持不变，主paired按同一确定顺序导出，不混入评价器修正。

- 300步诊断最终结束：16VAL上S2F−B0为mAP@1.0 −3.5284pp、mAP@0.5 −3.6464pp、outlier −2.4628pp、membership F1 −4.6903pp；不支持当前方法收益，停止扩大，不启动4卡完整训练。
- 累计SAM 0.268351 GPU小时、训练/诊断/评估 0.912697 GPU小时，所有累计ledger无active_since；全部own GPU任务已结束。
- 最终注册表合并80/300两组及失败/中断历史，旧汇总备份保留；最终收集器为outputs/sam_person/20261018_v1/collect_final.py，不更改冻结运行源码。
