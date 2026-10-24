# SAM 人物 Mask：有界验证与 GitHub 交付

本分支保存2026-10-18的SAM人物MaskRoI代码、实际配置、报告和轻量证据。
**完整4卡训练未运行，当前pilot没有支持扩大训练的收益证据。**

- [完整结果报告](report.md)：80/300步配对结果、QC、成本、失败及结论边界。
- [最终摘要](../../results/sam_person/20261018_v1/final_summary.json)和[运行注册表](../../results/sam_person/run_registry.csv)。
- [固定smoke画廊](../../results/sam_person/20261018_v1/smoke_gallery/index.html)与[失败画廊](../../results/sam_person/20261018_v1/failure_gallery/index.html)。GitHub不直接渲染HTML；clone后用浏览器打开，或在仓库根目录运行 `python -m http.server 8000 --bind 127.0.0.1` 后访问对应路径。
- [本次打包说明](../../results/sam_person/20261018_v1/github_packaging.json)及[实际服务器命令](commands_actual.md)。

## 代码范围与历史来源

活动代码只包含本次SAM支线及原B0/loader的可关闭接入。原工作区的heatmap resume/view改动、相关未提交测试及文档没有套用到本分支。SAM调用的6个B0 helper与基准提交33ca78c逐函数AST相同，因此本分支可以直接使用已提交的helper。

实际运行时完整源码内容hash、原始diff、日志、失败重试以及原始配置仍保存在 `results/sam_person/20261018_v1/` 中。`source_delivery/` 和 `source_archives/` 是**历史快照**，其中包含运行时原有的B0辅助文件；这些归档不会作为活动代码导入。原 `delivery_verification.json` 记录服务器交付时的检查，当前分支打包验证单独记录。

为使提交后的测试仍有独立参照，活动 `tests/sam_person/test_contract.py` 固定读取实验前B0 revision `33ca78cffd46d3bc50b7d3515ede28f70de842be`，不再随HEAD移动。此项是交付修正，没有重新训练或修改模型数学。训练时使用的原测试源码也保留在冻结快照中。

## 复核

下列汇总只使用已保存JSON/预测/日志，不需要数据集、SAM权重或GPU：

```bash
python tools/collect_sam_person_results.py
```

它合并80步和300步结果，核验共同初始化、采样、更新数、导出顺序及累计GPU预算。旧 `tools/report_sam_person.py` 是实际试验保存的80步收集器；最终结果使用上面的新入口。

在本服务器既有GAD环境进行CPU验收：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 TORCH_HOME=/mzy/copyfrom_aixi_gadfortest/.local_data/torch171 /share/apps/anaconda3/envs/torch171/bin/python -m unittest discover -s tests/sam_person
```

配置和原命令保留**实际服务器绝对路径**以便审计，不是另一台机器上的即用配置。重新运行GPU实验必须解析自己的数据/权重/输出路径并冻结新源码清单，不得覆盖原run或清零累计ledger。报告入口修改及测试reference固定也与原训练源码hash不同；原训练成绩只对应已归档的冻结版本。

## 附件范围

包含逐actor QC、manifest、预算账本、80/300步结果、预测导出、失败记录、源码快照和诊断画廊。原图副本、模型checkpoint、原始二值mask缓存、数据集和环境不上传Git；它们留在服务器，路径和校验和在记录中保留。报告中的 `smoke_originals/` 等代码格式路径因此指向服务器原产物。

这是单seed、有界TRAIN→VAL开发诊断，缺独立人物像素GT；没有完整训练/TEST或稳定收益结论。原evaluator存在集合置信度排序问题，本分支保留原实现，未混入评价器修正。
