# SAM 人物 Mask 验证结果（2026-10-18）

GitHub分支的文件范围、服务器冻结快照与复核入口见 [README.md](README.md)。

**本轮已完成代码、6项CPU验收、SAM smoke/QC、配对overfit及有界pilot。没有启动4卡完整训练，目前不建议启动。** 当前 SAM v1、box-only、直接 MaskRoI 在较大的固定诊断中，三项主指标及 membership F1 都低于同预算 B0。该结论只适用于这里的开发协议，不能推广为所有 SAM 或所有人物分割方法无效。

最新结果为 seed1、24 TRAIN / 16 VAL clips、各300步、固定最后一步；两侧从同一checkpoint独立重新开始：

| 配置 | mAP@1.0 | mAP@0.5 | outlier mIoU | membership P / R / F1 |
|---|---:|---:|---:|---|
| B0 | 15.1687 | 38.4325 | 64.6192 | 0.7957 / 0.7789 / 0.7872 |
| S2F | 11.6402 | 34.7862 | 62.1564 | 0.7791 / 0.7053 / 0.7403 |
| S2F−B0（百分点） | -3.5284 | -3.6464 | -2.4628 | F1 -4.6903 |

主表沿用未修改的原 evaluator。它存在下文说明的置信度集合排序问题，因此这些数字是当前实现下的开发诊断，不能作正式论文结论或统计显著性声明。

## 当前基线、数据与公平性

B0 是项目 `research_b0_repro=true`、heatmap off 的 ResNet18 开发基线。人物位置来自 **GT actor tracklets / boxes**，这是给定框的 GAD 协议，未测自动检测误差。框的 visible/amodal 语义未另行确认。保留 T=5、N=14、K=12、hidden256、原actor encoder、group reasoning、Hungarian matcher、activity/membership、解码和导出。

实际 backbone 网格为 512×23×40；原 RoIAlign 为5×5、aligned=True、sampling_ratio=1、spatial_scale=1，并保留原B0越界采样零外推。S2F只在原 `fc_emb` 前对原人物 RoI 特征乘局部权重。没有新参数、新loss、heatmap target、CLIP、FPN、SAM-CP affinity或新grouping。

当前真实几何变换只有 resize 至720×1280。原图二值mask以nearest同步resize，再area投影到实际特征网格，RoI权重保持连续值；不按mask面积归一化，也不删除actor。mask不需要梯度，backbone及actor路径保留原loss梯度。

原始采帧和框解析调用真实 CafeDataset adapter。actor UID为 video/clip/annotation row，trackID=row+1；帧号是segment-center采样得到的clip内JPEG文件坐标，不能冒充全视频绝对帧号。缓存另外绑定原图帧，避免同UID跨帧串位。

split-by-place TRAIN videos1–4、9–12、17–24，VAL13–16；既有审计清单6,444/1,386 clips。所选ann内容hash重新核对，200张原图内容hash互异且TRAIN/VAL无重复。共享盘全目录重扫耗时过长，未完成全数据跨split图像内容去重，因此P0是有界核验，不是全量重新审计。统一tracks pickle与原GT文本物理上包含全数据；仅索引/筛选本轮TRAIN/VAL。没有读取TEST图像或逐clip ann，没有TEST预测、评价或调参。读取合并文件不应表述成“未接触任何TEST字节”。

共同warm start为既有 `b0_repro_val_place_seed1` epoch22 `best_val.pth`。checkpoint SHA256为 `88b8fd6a99bdd41c8e5777ba1538321404d75da203baeee484a0d37d0f374a48`；统一移除DataParallel `module.`前缀后309个state keys严格匹配，张量初始化hash为 `246102479882f403054c15e41550782b751d24640e0d53d7a1177a575e63e80d`。旧checkpoint没有完整args，配置由本次实际模型/源码审计固定，不能补写成官方复现。

各stage独立重置同一初始化、同seed1、batch4、不拆microbatch、Adam lr=1e-5/weight_decay=0、FP32及原TF32 flags。两侧使用同固定segment centers、同clip顺序和独立固定sampler RNG；固定最后一步，不新增VAL选模。batch与采帧方式有别于历史完整训练，旧checkpoint也曾经过VAL选模，本轮属于开发诊断。

两组配对的code/config/evaluator/input/clip/frame/order/init hashes、batch、steps均核对一致，预测导出前7列和actor行序逐行相同。证据：[paired_checks_final.json](../../results/sam_person/20261018_v1/paired_checks_final.json)、各run的 `input_manifest.json`、`sample_order.json` 与 `config_actual.json`。

## 实际使用的SAM与缓存

复用已经安装并有权重的 **原版SAM ViT-B，非SAM2/2.1**；没有安装或升级训练依赖。来源为已安装包direct_url中记录的官方archive commit，并另外锁定源码内容hash；这不是独立Git checkout验证。API语义参照[官方原版SamPredictor](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/predictor.py)。

权重为 `/mzy/cafe_runs/sam-probe/checkpoints/sam_vit_b_01ec64.pth`，SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`。原RGB图、原图pixel XYXY box、逐人单候选、无点提示；同帧只算一次image embedding。FP32生成、关闭SAM环境TF32、logit threshold0，保存最终原图分辨率二值mask；无score/面积筛选、无NMS、无最大连通域、无强裁框。

缓存版本 `raw_bc294f454bfdfad21798`。原图mask按C-order/MSB-first packbits保存于NPZ，JSON原子完成记录含shape、checksum、UID、原图/框/模型/config指纹；所有正式读取拒绝不匹配、损坏或缺失缓存。未复用此前只能还原5×5框内权重的旧缓存。[cache_actor_records.jsonl](../../results/sam_person/20261018_v1/cache_actor_records.jsonl)补齐2054条逐actor来源、prompt/postprocess hashes与存储索引，不改原缓存。

smoke为8张独立图、83个人物prompt；最终QC包含smoke，总计 **200张不同文件且不同内容的原图、2054个有效actor-frame**：TRAIN120帧/1224人次，VAL80帧/830人次。这里“人次”是actor-frame，并非2054个独立真人。

## 身份、几何和mask质量

6个CPU测试方法通过（3.382s），覆盖非方形编码往返、源图/框/模型失效、真实parser有界/默认一致、同帧embedding复用及UID、resize/crop/flip、padding/空mask/全部fallback、关闭功能与原B0的输出/loss/导出一致、全1、排列恢复、有限梯度、移动mask因果依赖以及group/activity标签隔离。真实训练只使用resize，合成crop/flip通过不能被解释为真实训练启用了这些增强。详见 `tests/sam_person/test_contract.py` 和 [cpu_acceptance_stable.log](../../results/sam_person/20261018_v1/cpu_acceptance_stable.log)。

真实GPU最初的直接 `RoIAlign(ones)` 出现最小值0.9999999404，原TF32放大到activity输出最大约6.32e-4，未通过门槛。保留失败B0及旧S2F已完成的3步smoke（额外3步及耗时均保留并计入总账）后，改用同一线性采样的常量保持形式 `1-RoIAlign(1-S)`，再应用原B0 sample_valid零外推；不放宽容差、不改原B0精度flags。最终两组真实GPU全1检查在actor embeddings、actions、activities、membership上最大绝对误差均为 **0**；随机非恒定内部网格与直接插值在既定容差内一致。详见 [gpu_ones_diagnosis.json](../../results/sam_person/20261018_v1/gpu_ones_diagnosis.json)。

最终结构状态：2054/2054生成且原图非空；0缺失、0失败、0特征网格近空、0fallback。小框面积<图像1%的226人次，中框1517、大框311，均未在特征网格结构性消失。mask/box面积比中位数 0.3381，feature-RoI权重均值的跨actor中位数 0.3078；这些是覆盖proxy，不能解释为人体精度。

**没有独立人物像素真值，pixel IoU/Dice均为null。** SAM predicted_quality不是真实IoU，部分值超过1，不能当作已校准概率。100%结构可读和非空不代表100%分对人物。

固定smoke的83个actor overlay由助手全部筛看，另检查两条各5帧人物时序；未见全局坐标偏移或缓存UID重排，但出现局部语义错误。逐actor QC表仍保留全部2054人次，83条标记为助手目检，其余未逐人审阅；6例失败画廊只是说明性分诊，不是人工标注准确率。全QC画廊还含166个人物panel及4条时序，不能声称全部均做了像素审核。

| 场景 | 本轮证据与边界 |
|---|---|
| 遮挡/身体缺失 | `13/82 row_6`等出现残缺人体、椅子/背景碎片；未提供遮挡等级GT，无法报告遮挡分桶精度 |
| 邻人/拥挤 | `1/112 row_1`、`23/273 row_7`疑似邻人或重复轮廓；`16/259 row_3`有合并/泄漏。局部错人仍可能发生，不能用UID正确排除语义错人 |
| 小人物 | 有226个小框proxy，0网格近空；缺独立小人物mask GT，无法给真实小人物IoU |
| 手持物/交互上下文 | 未独立标注手、杯等物体，未量化保留率；mask可能损失任务上下文只是待验证解释，不能当成本轮已证实原因 |
| 时序 | 固定序列可视化保留；没有独立时序一致性或漂移真值，数值指标null |

可复核图片：[smoke_gallery/index.html](../../results/sam_person/20261018_v1/smoke_gallery/index.html)、[failure_gallery/index.html](../../results/sam_person/20261018_v1/failure_gallery/index.html)、[qc_gallery/index.html](../../results/sam_person/20261018_v1/qc_gallery/index.html)；原JPEG字节副本在 `smoke_originals/`。QC全表与目检说明为 [qc_actor_table.csv](../../results/sam_person/20261018_v1/qc_actor_table.csv)、[smoke_visual_review.json](../../results/sam_person/20261018_v1/smoke_visual_review.json)。

## 下游实验、负结果与误差

技术smoke为正式overfit前3个更新，包含在40步内。8个TRAIN clips、40步overfit：B0训练mAP@1.0从100到100，S2F从47.2222到100，梯度均有限。S2F首/末不同batch loss为1.2581/0.2915；B0为0.2026/0.2708，不能声称B0 loss单调下降。此项只证明接线可学习，不证明泛化收益。

首次固定pilot为16TRAIN/8VAL、80步（13个VAL真实组）：

| 配置 | mAP@1.0 | mAP@0.5 | outlier mIoU | membership P / R / F1 |
|---|---:|---:|---:|---|
| B0 | 15.0463 | 36.5741 | 59.1766 | 0.7660 / 0.7500 / 0.7579 |
| S2F | 7.4074 | 37.9630 | 64.4048 | 0.7727 / 0.7083 / 0.7391 |
| S2F−B0（百分点） | -7.6389 | +1.3889 | +5.2282 | F1 -1.8764 |

原8VAL没有Queueing、Ordering GT组，原evaluator这两项AP返回0；它们不表示观测到这两类识别失败。各类AP原始值保留在run metrics与 [final_summary.json](../../results/sam_person/20261018_v1/final_summary.json)。

80步结果严格成员指标下降、宽松指标有提升，因此仅追加一组成对诊断：24TRAIN/16VAL、300步、重新共同warm start，全部沿用已有200帧缓存，无prompt/阈值/后处理调参。累计pilot为80+300=380步/配置≤500；overfit40步/配置≤200。追加同时改变样本量及更新数，不能归因于某一个因素，更不能把8VAL与16VAL绝对分数直接比较成提升。

最新300步在16VAL、26个GT组上，各类AP如下：

| 类别 | GT组数 | B0 AP@1.0 | S2F AP@1.0 | B0 AP@0.5 | S2F AP@0.5 |
|---|---:|---:|---:|---:|---:|
| Queueing | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Ordering | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Eating/Drinking | 6 | 16.6667 | 33.3333 | 33.3333 | 66.6667 |
| Working/Studying | 7 | 11.4286 | 14.2857 | 71.4286 | 57.1429 |
| Fighting | 5 | 40.0000 | 0.0000 | 80.0000 | 40.0000 |
| TakingSelfie | 6 | 22.9167 | 22.2222 | 45.8333 | 44.9074 |

补充误差诊断：

| 300步诊断 | B0 | S2F |
|---|---:|---:|
| 预测组数 | 19 | 19 |
| 存在同活动且成员完全一致预测的GT组数 | 8 | 7 |
| 匹配后漏成员 | 21 | 28 |
| 匹配后多成员 | 19 | 19 |
| 匹配组活动错分 | 10 | 10 |
| 拆组重叠事件 | 1 | 0 |
| 并组重叠事件 | 2 | 1 |
| 高置信且同活动成员IoU<0.5的预测 | 2 | 3 |

membership P/R/F1使用原Hungarian匹配GT组之后的argmax+原阈值成员集合，忽略活动类别，是补充proxy。漏/多成员、活动错分来自该匹配；完全匹配、拆并组和高置信错误来自独立重叠诊断，后者不代替原AP的一对一匹配，也不是校准误差。置信错误固定定义为score≥0.9且同活动最大成员IoU<0.5，没有据此筛选预测。详见 [error_analysis_extended.json](../../results/sam_person/20261018_v1/error_analysis_extended.json)。

从300步已保存预测切回原8VAL（没有新前向或选模），B0/S2F依次为：mAP@1.0 15.7407/12.9630，mAP@0.5 31.0185/37.9630，outlier61.8006/60.7143。它保留了小样本宽松AP混合趋势，但没有扭转严格指标下降；不能挑该子集替换16VAL主表。见 [extended_on_original8_val.json](../../results/sam_person/20261018_v1/extended_on_original8_val.json)。

S2F在300步实际使用61200/61200个训练actor-frame mask、VAL830/830，fallback均0。旧80步S2F为16380/16380，VAL414/414。B0实际SAM使用为0。原B0全1诊断曾遗留 `last_mask_stats`，使80步日志误写mask计数；该属性未参与forward/loss，权重及预测不受影响。保留原记录，通过真实sample_order重建B0普通RoI计数（overfit8220、pilot16380），修正在 [diagnostic_corrections.json](../../results/sam_person/20261018_v1/diagnostic_corrections.json)；300步仅增加清理该日志状态的一行，没有模型数学变化。

## 评价器风险与来源记录

原 `make_groups` 将score存为单元素set，`group_mAP_eval` 直接按set排序，无法保证按数值置信度排序。CPU保持完全相同预测行、仅反转clip顺序：80步B0 mAP@1.0 15.0463→15.9722，S2F 7.4074→4.6296；outlier不变。证据为 [evaluator_order_probe.json](../../results/sam_person/20261018_v1/evaluator_order_probe.json) 及同名脚本。两侧主结果采用相同确定导出顺序，evaluator未改，SHA256仍为 `8d7f5d5f2bebaa911e30199036b20fed1cc9d1c31bfebb454b68f1901a50f181`。相同顺序能保证本轮操作一致，不能消除该实现的评价风险；正式训练前应独立修复/审计，并对所有B0/S2F保存预测统一重评，不能将评价器修正当作SAM收益。

起始revision `33ca78cffd46d3bc50b7d3515ede28f70de842be`，保留用户原未提交工作。冻结源码内容hash：

- 首次失败版本 `1bb6be87df7ec16a23178ef7538904665303bb42cfe02af5529ea77e3bd22a97`，逐字节重建核验后归档于 `source_archives/`，重建来源单独记录。
- 80步正式版本 `0a3fb1b2cc1d9a7db0cea0de38b1007d807a941c323664e026f17000d86fda11`，源码、diff和冻结配置均保留。
- 300步服务器冻结版本 `e2aa8527641ae97b13215f89d95c5cc8076a32c7e9e552bed3f7dd0152d65f94`，与80步仅日志清理一行之差，见 `source_change_between_pilots.json`、`source_delivery/`、`source_manifest_delivery.json`。

配置为 [qc_resolved.yaml](../../configs/sam_person/qc_resolved.yaml)、[pilot_resolved.yaml](../../configs/sam_person/pilot_resolved.yaml)、[pilot_extended_resolved.yaml](../../configs/sam_person/pilot_extended_resolved.yaml)。真实命令见 [commands_actual.md](commands_actual.md)，配置实际副本、运行日志、初始化与输入hash在各run目录；tracks、合并GT文本、采样清单、当前manifest及evaluator的实际文件SHA见 [input_asset_hashes.json](../../results/sam_person/20261018_v1/input_asset_hashes.json)；交付核验见 [delivery_verification.json](../../results/sam_person/20261018_v1/delivery_verification.json)。总注册表为 [CSV](../../results/sam_person/run_registry.csv) / [JSON](../../results/sam_person/run_registry.json)。失败和负结果均未删除。旧80步汇总备份为 `before_extended_*`；最终汇总使用 [collect_final.py](../../results/sam_person/20261018_v1/collect_final.py)，旧 `tools/report_sam_person.py` 仅收集80步，不能用它覆盖最终表。

## 成本与4卡完整训练决策

本任务实际只在已获批物理GPU4/5上分别进行生成或配对运行，没有为凑4卡而扩大实验。SAM与GAD用各自已有环境：SAM Python3.9/torch1.13/torchvision0.14；GAD Python3.8/torch1.7.1/torchvision0.8.2，使用已有项目numpy/scipy/Pillow overlay，未安装升级依赖。

累计生成 **0.268351 GPU小时≤1**；训练、GPU诊断及评估 **0.912697 GPU小时≤2**，包括初次失败和中断恢复的保守计时。两类合计约 1.181048 GPU小时。这里是设备占用墙钟累加，包含I/O等待及恢复间隔，不是纯GPU kernel时间。ledger保留旧failure_reason供追溯，最终status为finished且无active_since。

SAM实际完成2054个唯一prompt，累计额度计2058次（中断帧重复/预留不退回）；200帧。成功帧encoder合计43.508s、中位0.123662s/帧，首次15.915s；decoder合计17.273s、平均0.008410s/人。SAM峰值allocated约2.70GiB，原图缓存NPZ+JSON共5,660,453字节（约5.40MiB），不包含画廊、checkpoint及源码归档。

原始ledger中还保留一次可选设备属性查询 `Invalid device id` 的失败（0帧）、一次28帧后中断及后续164帧恢复；逐人NFS记账开销大后改成逐帧预留，未改变mask策略或缓存版本。所有费用保守保留。

一个12人物固定帧重复20次的热缓存测试：checksum+解码中位85.70ms/帧，nearest+area投影43.87ms/帧，不包括源图重新hash。只测一个帧，不能当作全数据p95。详见 [cache_read_benchmark.json](../../results/sam_person/20261018_v1/cache_read_benchmark.json)。

| 300步运行 | 训练 clips/s（step累计） | 已驻留输入H2D+前向 clips/s | 峰值 allocated GiB |
|---|---:|---:|---:|
| B0 | 4.800 | 17.396 | 8.660 |
| S2F | 4.832 | 15.426 | 8.661 |

pilot先把有限clip图像/目标/投影mask放入CPU内存，step吞吐包含训练计算，前向吞吐包含已驻留输入H2D与模型前向，均不包含完整数据流的NFS读取、SAM、全量训练数据加载与DDP通信。单次短run也不能用来声称加速效果。部署到新帧仍需SAM encoder、逐人decoder和投影；这里已知人物框，不含检测成本。多帧/多卡完整端到端吞吐未测，记null。

按既有全split清单估算，若覆盖TRAIN随机采帧所有可能文件及固定VAL采帧，上界约223670个clip-frame条目、2340076个actor prompts；相邻clip绝对视频帧去重未知，不能叫223670张已独立确认的全视频帧。按本次压缩率缓存约6.33GB，按热encoder+decoder仅计算约13.15 GPU小时，尚未含I/O、模型加载、失败和日志，已经超出本轮生成预算。详见 [full_scale_estimate.json](../../results/sam_person/20261018_v1/full_scale_estimate.json)。

若仅作30epoch/6444clips的计算下限粗估，本次约4.8 clips/s对应单方法约11.2 GPU小时，理想4卡约2.8小时墙钟；这是忽略DDP/I/O/评估/随机采帧加载的下限，不能作为获批额度。保守计划可暂按每方法每seed 4卡6–10小时墙钟（24–40 GPU小时）估算，正式预算必须先完成全量数据流profile；3个配对seeds共6次训练约144–240 GPU小时，另加缓存生成。当前磁盘约45GB余量，全量产物需另行核算。

**本轮停止于有界P4，不启动P5。** 支持的结论：身份/几何和特征接线通过验收；给定框的SAM mask可离线缓存且在本次网格上不消失；当前直接过滤会改变特征、可训练，但没有提供优于成对B0的下游证据。零fallback意味着回退没有掩盖本轮下降。未证实下降具体来自语义分割错误、上下文丢失、幅度缩放或训练时长，不能根据画廊直接归因。

未运行：完整TRAIN/VAL或TEST训练评价、3 seeds、SAM2/2.1、提示消融、S2F-SCALE、H1-SAM-TARGET、全量DDP/streaming S2F入口与性能profile。均不填零分或伪造待运行全量命令。现有runner明确限制有界TRAIN→VAL，实际可复核命令已交付；完整4卡入口还需实现和验证，不能把pilot命令改个卡数冒充正式训练。

下一项最小工作应为独立CPU评价器审计：对已存预测统一确认数值排序与匹配规则，并保留原协议对照，预计0 GPU小时，不需生成新mask。若仍要研究mask质量，应先预声明一个TRAIN来源、人工标注的遮挡/邻人/手持物小集，明确标注工时和新的GPU预算；当前证据不足以跳到4卡长训练。本轮不再追加实验。
