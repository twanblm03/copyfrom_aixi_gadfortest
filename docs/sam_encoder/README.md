# SAM image encoder 全替换训练入口

本实现对应设计中的 **SAM-only backbone**：冻结原版 SAM ViT-B image encoder，用它同时提供人物和场景特征。ResNet 不再参与视觉特征提取；不使用 SAM mask、prompt encoder、mask decoder、旧 mask cache、H1 或 VideoMAE。默认仅准备可执行命令，不启动训练。

## 特征流与训练范围

```text
同一帧原始 RGB / 既有人物轨迹框
  → loader：SAM 长边1024缩放，uint8 RGB [BT,3,576,1024]（16:9）
  → wrapper：SAM pixel normalization，右/下补至1024×1024
  → frozen SAM ViT-B image encoder，最后 block 经 neck
  → [BT,256,64,64]
  → 取真实图像有效视场36×64，再插值到23×40
  → [BT,256,23,40]
      ├─ input_proj：256→256 → scene
      └─ 人物 RoIAlign：256×5×5 → flatten6400 → fc_emb → actor
  → 原 box position、12 group slots、Group Transformer
  → 原时间聚合、群组成员和活动预测、任务损失
```

wrapper 位于 `models/sam_backbone.py::SAMImageBackbone`。SAM 参数 `requires_grad=False`，外层进入 train 时它仍保持 eval，冻结 forward 不保留梯度。**冻结 SAM 不等于整个模型冻结**：`input_proj`、人物 `fc_emb`、Group Transformer 和预测头仍由 GAD 标签训练。`SAM_ENCODER_CHUNK_SIZE=1` 限制单次处理帧数，不改变 clip 的 T=5。它不增加时序编码，仍沿用原聚合。

与 ResNet18 的512通道相比，scene 的 `input_proj` 输入变成256，actor 的 `fc_emb` 输入从12800变成6400。这两层必须按新维度初始化。默认从新 run 开始；不能把 B0、MaskRoI 或 H1 checkpoint 当成这个模型的 resume。相同 SAM backbone、输出尺寸和训练协议下才可恢复 optimizer/scheduler。`util/sam_encoder.py::validate_sam_resume` 校验保存的架构契约，随后严格 state_dict 加载核对参数键和维度；不能改成宽松加载来掩盖架构差异。

## 准备路径与命令

在本 worktree 根目录使用入口：

```bash
bash scripts/train_cafe_sam_encoder.sh --help
bash scripts/train_cafe_sam_encoder.sh --dry-run
```

默认路径可由环境变量覆盖：

| 环境变量 | 默认值/用途 |
|---|---|
| `PYTHON_BIN` | `/data2/mzy/GAD/.miniconda3/envs/gad/bin/python` |
| `DATA_PATH` | `/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset`，应含 `cafe/` |
| `TRACKS_PATH` | `/data2/mzy/GAD/CAFE_codebase/.local_data/cafe/gt_tracks.pkl`，沿用现有轨迹资产 |
| `GROUNDTRUTH` | NAS evaluation 的 `gt_tracks.txt` |
| `SAM_REPO` | 本 worktree `.local_deps/segment-anything`，仅本地官方源码 |
| `SAM_CHECKPOINT` | 本 worktree `.local_data/sam/checkpoints/sam_vit_b_01ec64.pth` |
| `SAM_CHECKPOINT_SHA256` | 可选预期SHA256；`--run`前校验并将实测值传给模型 |
| `CHECKPOINT_ROOT` | 本 worktree `.local_data/sam_encoder/checkpoints` |
| `LAUNCH_ROOT` | 本 worktree `.local_data/sam_encoder/launches` |

默认 place、TRAIN→VAL、四卡 Host DataParallel、全局 batch16、VAL batch4、T5/K12、30 epochs，学习率与 ResNet18研究协议一致：Adam、`1e-5→1e-4`、5 epoch warmup、linear schedule、weight_decay0。`BACKBONE_LR_SCALE=1.0` 与view-final B0记录一致；SAM 完全冻结，该倍率不会让它更新。TRAIN 默认随机分段采帧，VAL 沿用中心帧；`FIXED_CENTER_TRAIN=1` 是显式的额外协议选择，配对B0必须一致。

所有示例首先使用 `--dry-run` 检查命令。只有明确的 `--run` 才会执行。以下命令将执行对应训练，**本次交付没有运行它们**。

单卡有界 pilot（2个 optimizer steps；VAL 范围仍取决于 manifest）：

```bash
WARMUP_EPOCHS=1 bash scripts/train_cafe_sam_encoder.sh --run \
  --device 0 --split place --epochs 1 --max-train-steps 2 \
  --batch 1 --test-batch 1 --tag sam_encoder_all_place_smoke
```

可进一步提供固定 clip manifest 限制 TRAIN/VAL：

```bash
CLIP_MANIFEST=/absolute/path/to/pilot_clips.json WARMUP_EPOCHS=1 \
bash scripts/train_cafe_sam_encoder.sh --dry-run \
  --device 0 --epochs 1 --max-train-steps 20 --batch 1 --test-batch 1
```

`MAX_TRAIN_STEPS` / `--max-train-steps` 是**每 epoch**上限，0表示整轮；不能把 epochs30、上限20解释成总共20步。有界预算同时指定 epochs 和步骤，完整VAL评估另有成本。warmup须在1到epochs之间；上述一轮示例显式设为1，也可用 `--warmup-epochs 1`。`--split view` 使用view划分。

clip manifest 是单个JSON对象，含 `train` 和 `val` 数组；每项用 `{"clip": [video_id, clip_id]}` 选择样本。须选择属于当前place/view划分的真实clip；它不是旧SAM逐帧mask的JSONL manifest。

四卡延用当前 Host DataParallel 传输方式：

```bash
bash scripts/train_cafe_sam_encoder.sh --dry-run --device 0,1,2,3 --split place
```

可选原生 DDP 命令（与 Host DP 分开比较）：

```bash
bash scripts/train_cafe_sam_encoder.sh --dry-run --device 0,1,2,3 \
  --distributed 1 --parallel-transport native
```

同架构恢复会读入模型、optimizer、scheduler和epoch状态；恢复时沿用原始split、SAM权重、输出网格、batch、学习率计划和采样协议：

```bash
bash scripts/train_cafe_sam_encoder.sh --dry-run \
  --resume /absolute/path/to/sam_encoder_run/epoch1.pth \
  --tag sam_encoder_all_place_resume
```

不要把先前 ResNet 最优模型作为该例的路径。这个入口没有 warm-start、fusion 或 LoRA 模式。

## 显式切换到view-final B0同协议

launcher默认始终为development。`--final-test` 映射训练参数 `--final_test`，改为TRAIN+VAL训练、最后epoch评估一次TEST；另加 `--best-on-test` 才映射 `--best_on_test`，按每epoch的TEST mAP@1.0选 `best_test.pth`。后者要求同时 `--final-test`、`TEST_FREQ=1`，当前训练协议用于view。TEST指标在日志/summary中标为best-on-TEST，与独立留出评价区分。

`--keep-all-checkpoints`（别名 `--save-every-epoch`）映射实际训练参数 `--keep_all_checkpoints`，保留所有epoch checkpoint。保存最新epoch本来就是训练器默认行为，因此即使不传此参数也保留最终epoch；不存在另一个 `--save_last_epoch` 训练参数。

与已冻结view-final B0共同配置一致的命令预览：

```bash
bash scripts/train_cafe_sam_encoder.sh --dry-run --split view \
  --device 0,1,2,3 --final-test --best-on-test --keep-all-checkpoints \
  --tag sam_encoder_all_view_trainval_dp4_lr1_noclip_seed1
```

final标签会自动带上 `final_test` 或 `best_on_test`，包括对自定义tag添加后缀，以免与development结果混淆。只预览不会创建队列或启动训练。实际顺序由外部队列安排；不能同时手工运行上面命令来占用队列等待的GPU。

共同部分保持view、TRAIN+VAL→TEST、每epochTEST/同主指标选模、所有epoch保留、batch16/test_batch4/micro_batch0、T5/K12、随机分段、Host DP、seed1、Adam/linear/warmup5、无梯度裁剪、相同GT轨迹/评价器。必需差异是 `resnet18→sam_vit_b`、冻结SAM、SAM资产参数、256通道投影及独立输出目录/tag；从新run开始，不能使用B0的resume路径。模型输入逻辑尺寸仍声明720×1280，SAM loader按其比例送入576×1024；有效特征网格保持23×40，不沿用mask cache参数。

独立验证SAM训练检查点时，同样使用SAM backbone和对应输入尺寸，例如：

```bash
/data2/mzy/GAD/.miniconda3/envs/gad/bin/python test.py \
  --backbone sam_vit_b --research_b0_repro --no_mae --val_mode --split place \
  --image_width 1280 --image_height 720 --num_frame 5 --test_batch 4 \
  --data_path /nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset \
  --tracks_path /data2/mzy/GAD/CAFE_codebase/.local_data/cafe/gt_tracks.pkl \
  --groundtruth /nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/evaluation/gt_tracks.txt \
  --model_path /absolute/path/to/sam_encoder_run/best_val.pth --device 0
```

SAM源码、原始预训练权重路径有变化时，另传`--sam_repo`和`--sam_checkpoint`；恢复检查会核对架构及可用的权重来源hash。

## CPU验收与GPU smoke

CPU单元测试使用伪encoder验证契约，不加载GPU、不启动训练：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  /data2/mzy/GAD/.miniconda3/envs/gad/bin/python \
  -m unittest discover -s tests/sam_encoder -p 'test_sam_encoder.py'
```

真实 SAM 前向/GAD反传 smoke 的入口为 `tools/smoke_sam_encoder.py`，与完整训练入口分开。默认 `--device cpu`，真实checkpoint单帧前向和真实GAD反传；可选 `--image` 读取真实RGB。其参数以该脚本自身帮助为准：

```bash
/data2/mzy/GAD/.miniconda3/envs/gad/bin/python tools/smoke_sam_encoder.py --help
```

下例是待执行的单卡smoke命令，不会启动完整dataset训练；它会实际使用GPU：

```bash
CUDA_VISIBLE_DEVICES=0 /data2/mzy/GAD/.miniconda3/envs/gad/bin/python \
  tools/smoke_sam_encoder.py --device cuda:0 \
  --sam_repo .local_deps/segment-anything \
  --sam_checkpoint .local_data/sam/checkpoints/sam_vit_b_01ec64.pth \
  --output .local_data/sam_encoder/smoke_cuda.json
```

验收重点：真实有效视场和人物框坐标对应正确；输出256×23×40；SAM无梯度、GAD/投影有非零有限梯度；actor和scene确实依赖SAM；没有实例化ResNet或调用mask decoder。GPU smoke和训练成绩须实际运行后报告，不能用伪encoder CPU测试替代。

## 产物与比较

仅 `--run` 会保存独立的launch目录，包含 `launch.json` 和精确的 `command.sh`。记录命令、环境、原始数据/轨迹/GT路径、SAM checkpoint SHA256、manifest/resume SHA256，以及GAD和SAM源码SHA256。不会自动下载权重或提交后台任务。

`train.py` 保留完整解析参数 `result/<experiment>/args.json`、日志和summary，checkpoint写入新 `CHECKPOINT_ROOT` 下按时间/新tag建立的实验目录；既有B0和MaskRoI结果不覆盖。冻结SAM权重仍属于模型state_dict，全参数checkpoint会明显大于原ResNet模型；每个epoch仍保存optimizer/scheduler以恢复训练。默认development保留最新epoch与best_val；best-on-TEST保存best_test；传 `--keep-all-checkpoints` 才保留每个历史epoch。最终epoch始终保留。

与B0比较时固定place/view、数据协议、GT actor tracklets、T5/K12、batch、采样、任务损失、Group Transformer、评价器和同一选模规则。development主指标为完整VAL `group_mAP_1.0`；显式best-on-TEST协议则在同一主指标选中epoch报告TEST mAP@1.0、mAP@0.5及outlier mIoU，不分别挑次指标峰值。配套记录成员错误和吞吐/显存。有界pilot与完整评估分开标注。本实现验证“整套SAM视觉表示替换是否有效”；它同时改变人物/场景表示，不能据此单独归因人物分支，也不直接证明分割预训练优于其他同架构预训练。
