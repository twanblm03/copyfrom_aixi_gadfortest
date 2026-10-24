# 当前 H1-sup 完整分支

本分支保存当前 H1-sup 模型、修正评价器、place/view 数据划分与视角协议、完整参数、训练与测试工具。新增入口为 [`scripts/train_h1_sup.sh`](scripts/train_h1_sup.sh)，便携配置位于 [`configs/h1_sup`](configs/h1_sup)。模型、训练器和评价器均来自当前研究工作区；新入口只整理配置与本地路径，不改变它们的算法。

## 数据划分与视角

视频编号来自 [`dataloader/dataloader.py`](dataloader/dataloader.py)。

| 划分 | TRAIN 视频 | VAL 视频 | TEST 视频 |
| --- | --- | --- | --- |
| place | 1,2,3,4,9,10,11,12,17,18,19,20,21,22,23,24 | 13,14,15,16 | 5,6,7,8 |
| view | 1,2,5,6,9,10,13,14,17,18,21,22 | 3,7,11,15,19,23 | 4,8,12,16,20,24 |

place 保持开发协议：16 个 TRAIN 视频训练，4 个 VAL 视频评价及选模，加载器过滤后为 **6,444→1,386 clips**。TEST 在此开发流程中不参与训练或选模。

view 正式协议在每个场景合并两个 TRAIN 视角与一个 VAL 视角，共三个视角、18 个视频重新训练，使用剩余第四视角的 6 个 TEST 视频评价，过滤后为 **7,298→2,414 clips**。每项从相同 ImageNet 初始化规则重新开始 30 轮，不从两视角开发 checkpoint 续训。每轮 TEST，保留所有 `epochN.pth`（包括 `epoch30.pth`），按 **TEST group mAP@1.0** 选择 `best_test.pth`；mAP@0.5、outlier mIoU 与 loss 来自同一个选中 epoch。结果必须标记 **best-on-TEST**，不能解释为独立 VAL 选模后的 TEST 结果。

原始 view TRAIN/VAL/TEST 的视频数为 12/6/6；合并 TRAIN+VAL 后才是三视角。clip 数是本地加载器过滤后的实际计数，不等同于论文原始数据集统计。详细协议比较见 [`reports/heatmap/view_protocol_vs_paper_20261023.md`](reports/heatmap/view_protocol_vs_paper_20261023.md)。

## 固定训练设置

标准 H1-sup 使用 `research_b0_repro=true`、`heatmap_variant=h1_sup`、`heatmap_target_mode=box`、sigma ratio **0.15**、`train_mask_mode=disabled`。它保留 B0 匹配与推理路径，并添加群组热图辅助监督。`crowd` 是另外的显式目标变体，不是标准 H1-sup；仅在指定 `--target-mode crowd` 时开启。

共同设置为 ResNet18、普通 BN、1280×720、5 帧、12 个群组 token、256 维隐藏层、6 层 Group Transformer、4 个 attention heads、FFN 512、dropout 0.1。TRAIN 在完整候选帧内随机分段采样；评价使用确定的分段中心帧。这里 `random_sampling=false` 表示采用分段策略，`fixed_center_train=false` 保持训练随机性。

默认四卡 **Host DataParallel**（device `0,1,2,3`），全局 batch16、test_batch4、seed1、30 轮、warmup5、Adam、WD0、LR `1e-5→1e-4` 后线性衰减、backbone LR scale1，无梯度裁剪、无 VideoMAE、无 SAM mask。全部参数见两个便携 JSON；新入口对路径、device、输出和标签的覆盖不会自动改变其他参数。修改 GPU 数量会改变每卡 batch 与 BN 行为，应作为协议差异记录。

## 运行

在已安装项目依赖的 Python 环境中运行；`DATA_PATH` 指向含 `cafe/` 的 `Dataset` 目录。数据集、轨迹、GT 和预训练权重由使用者在本机准备，不随源码上传。

```bash
export PYTHON=python
export DATA_PATH=/your/dataset/Dataset
export GROUNDTRUTH=/your/evaluation/gt_tracks.txt
# 可选：轨迹放在其他位置时设置，否则读取 $DATA_PATH/cafe/gt_tracks.pkl
export TRACKS_PATH=/your/local/gt_tracks.pkl

# 只打印最终参数和命令，不读数据、不初始化模型、不启动训练
bash scripts/train_h1_sup.sh place --dry-run
bash scripts/train_h1_sup.sh view --dry-run

# place 开发：TRAIN→VAL，选择 best_val.pth
bash scripts/train_h1_sup.sh place

# view 正式：TRAIN+VAL→TEST，每轮评价，选择 best_test.pth
bash scripts/train_h1_sup.sh view
```

也可通过 `--data-path`、`--tracks-path`、`--groundtruth`、`--device`、`--output-root` 与 `--tag` 覆盖本机路径或输出。默认输出分别为 `outputs/h1_sup/place/` 和 `outputs/h1_sup/view/`，checkpoint 位于各自的 `checkpoints/`，训练器继续使用独立的带时间戳实验目录。`--dry-run` 仅依赖 Python 标准库，显示配置后退出；正式运行会先检查本地数据与评价文件存在。

[`configs/view_final_dp4`](configs/view_final_dp4) 与 [`configs/view_dp4`](configs/view_dp4) 保留原始运行审计参数：前者是正式三视角协议，后者是历史两视角 TRAIN→VAL 开发控制。它们含原服务器 NAS 与输出绝对路径；迁移运行应使用上述新入口。历史脚本与队列保留作为审计记录，不能混用其默认采样、LR、并行或选模设置。

## 评价器与验证

修正评价器 [`evaluation/cafe_eval.py`](evaluation/cafe_eval.py) 的固定 SHA256 为：

```text
f1cd5f05b63ec769c149ad1b8926922eb03bb70ab3665fb7e6bf75cec28b40a0
```

新入口检查此 hash。训练与测试显式提供评价 clip 集合，评价限定到目标划分；数值置信度排序、缺预测 clip 的 GT 召回分母及 outlier 缺失统计均有回归测试。历史旧评价分数应保留其评价器版本，不能直接当作修正后的结果。

在上述依赖环境中，可从仓库根目录运行不依赖数据或 GPU 的针对性检查：

```bash
PYTHONPATH=. python tests/sam_person/test_evaluator.py
PYTHONPATH=. python tests/sam_person/test_view_training_contract.py
PYTHONPATH=. python tests/heatmap/test_crowd_targets.py
```

部分历史 `view_final` 策略、恢复和配置测试会读取原实验忽略的 `.local_data/source_snapshot`、参考 `args.json` 或 NAS 数据；裸 clone 无这些实验产物，不能声称全套历史集成测试可直接执行。便携入口的 `--dry-run` 可独立核对两种协议的完整命令。

## 环境与共享源码

当前实际运行环境记录为 Python **3.8.20**、PyTorch **2.4.1+cu121**、torchvision **0.19.1+cu121**、NumPy **1.24.3**、SciPy **1.10.1**、Pillow **9.5.0**。根 `environment.yml` 是历史环境记录，不能作为这次运行环境的精确锁定文件。

完整 `models`、`util`、`dataloader` 与相关工具中保留 SAM 共用实现，因为当前训练器和数据加载器存在共享 import。标准 H1-sup 不启用 SAM，也无需加载 SAM checkpoint 或 mask 缓存。数据集、训练 checkpoint、大型缓存和本地依赖环境保留在本机。
