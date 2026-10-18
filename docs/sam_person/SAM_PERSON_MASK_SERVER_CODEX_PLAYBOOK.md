# SAM 人物 Mask 验证：服务器 Codex 执行手册

日期：2026-10-18  
对象：能够访问现有 CAFE / GAD 项目、数据、环境和实验产物的服务器 Codex  
性质：待执行的实施与验收协议；不是已完成的服务器实验报告

## 0. 任务与研究边界

请在现有项目上实现一个独立、可关闭的 SAM 人物 mask 实验支线，并完成预算内的最小验证闭环。不要只给计划。优先回答两个问题：

1. 用项目已有的人物框作为提示，冻结的 SAM 能否在真实数据上生成正确对应到人物的 masks？困难在哪里，覆盖率和代价如何？
2. 这些 masks 用于过滤人物 RoI 特征后，相比普通 RoI 特征，是否改善人物表示及 GAD 下游表现？

先完成分割诊断，再开展下游小实验。暂不复现整套 SAM-CP，不新增其 unified affinity decoder、DCA、AR、CLIP 分类器或文本分支，也不微调 SAM。SAM 不直接预测 group 或 activity。

借鉴 SAM-CP 的部分仅为：复用已有分割区域；利用 mask 过滤 RoI 内的背景和邻人干扰。本文的 box-prompt 人物提取、缓存协议、GAD 接线和实验编号是本项目的设计，不是 SAM-CP 原实验。

默认主线：

```text
原始 RGB 帧 + 该帧已有 actor boxes
  -> 冻结 SAM，逐人物生成 mask
  -> 按原始帧/人物身份离线缓存
  -> mask 与图像、框同步增强
  -> 现有 backbone 的人物 RoI 特征 × 对应局部 mask
  -> 现有 actor encoder / group reasoning / activity / membership
  -> 不变的 GAD evaluator
```

### 0.1 与此前 heatmap 实验的关系

- 本地 2026-10-14 周报记录 H1-Sup 有单 seed 的改善、H2-S 明显下降，并提示部分跨机对照不完全受控。这些是待服务器核验的历史线索，不是当前 checkout 的已知成绩。
- 因此先在确认的 B0 上只替换 actor RoI 的取特征方式；不默认以 H2-S 为起点，不把 SAM 与新的 grouping / matching 同时加入。
- 可选第二支线是在确认的 H1-Sup 上只替换 heatmap target，见第 8 节。它不是首轮必做，也不与主线捆绑。
- 旧 heatmap 指南中“首轮不加入 SAM”的限制，本次仅对这个独立支线更新；其他公平比较、数据协议和结果保存约定仍保留。

### 0.2 权限与资源

遵守项目适用的 `AGENTS.md`。保留用户已有 diff、checkpoint、缓存和结果，不强制 reset、不覆盖旧 run、不终止其他人的任务。不自动租用算力、外传数据或启动大规模完整训练。网络受限时使用已批准的官方代码/权重缓存，报告缺项。

所有操作使用当前项目可用、已分配的资源。算力预算不明确时，先完成 audit、实现、CPU 合成测试；GPU 步骤只在已分配的可用额度内执行第 10 节的有界试验。完整训练需已有明确预算，否则交付耗时估计和待运行命令。

## 1. 执行顺序与阶段出口

| 顺序 | 执行内容 | 最小交付 | 何时继续 |
| --- | --- | --- | --- |
| P0 | 检查项目、输入框来源、B0 和资源 | audit、真实数据流、环境/配置快照 | 框与原图可准确对应，基线定义明确 |
| P1 | 独立接通 SAM 单图推理 | 8 帧以内 smoke、overlay、耗时 | 人物 ID、坐标、输出尺寸检查通过 |
| P2 | 固定诊断集，生成缓存并做 QC | QC 表、固定成功/失败画廊、缓存校验 | 无系统性错位/错人，失败有显式标记 |
| P3 | 接通可关闭的 MaskRoI | 代码、测试、关闭/全 1 mask 等价性 | 不变的 B0 路径可恢复，梯度及数据隔离正常 |
| P4 | 小集 overfit + 成对 pilot | B0 / S2F 同预算记录、实际配置与命令 | 技术健康且有进一步实验依据 |
| P5 | 扩大缓存、完整受控实验 | 多 seed 结果、误差归因、资源开销 | 只在预算已明确时执行 |
| 可选 | H1-Sup 的 SAM target 支线 | 单独实验编号与对照 | 不与人物特征改动合并归因 |

不要越过 P1/P2，直接对全数据生成 masks 或启动长训练。一个阶段失败时，先保留产物并诊断；能继续完成的独立实现和 CPU 测试仍继续。

## 2. P0：先查清项目事实

### 2.1 只读检查

先定位 dataloader、人物框处理、backbone、RoIAlign、actor encoder、group matcher、criterion、decoder、evaluator 和真实启动入口。先用 `rg` / `rg --files` 搜索，不依赖猜测文件名。

此前线索包括 `models/models.py`、`models/group_transformer.py`、`models/group_matcher.py`、`models/criterion.py`；仅作为检索线索。文件不存在时按职责搜索，禁止创建空壳来冒充原实现。

在 `docs/sam_person/project_audit.md` 记录：

| 类别 | 必须确认 |
| --- | --- |
| 数据协议 | 数据集版本、place/view、train/val/test 清单及 hash、是否存在跨 split 重复帧 |
| 输入 | 原始图像读取方法、RGB/BGR、帧命名、原视频帧号与 clip 内采样序号的映射 |
| 人物 | actor/track ID 的作用域、有效性、padding、缺失帧、是否含 individual |
| 人物框 | GT / detector / tracker 来源，xyxy/xywh、像素/归一化、可见框/全身框、边界约定 |
| 增强 | resize/crop/flip 的调用顺序，是否使用 clip 一致增强，变换参数如何复用 |
| 模型 | 实际 T、N、backbone、RoI 输出尺寸、stride/坐标约定、actor 特征维度 |
| GAD | slot 数、membership 归一化、matching、activity、no-group、解码阈值和 evaluator |
| 训练 | B0 初始化、AMP、seed、batch、optimizer、scheduler、恢复策略和选模规则 |
| 资产 | 可复评 checkpoint、已有 masks/缓存、SAM 源码、权重及其真实路径 |
| 资源 | 项目 Python、GPU/驱动/显存、空闲且已分配 GPU、磁盘可用量、允许的预算 |

若项目是“给定人物框”的评估协议，沿用它，并在结果中声明。若真实部署依赖检测框，不得在新实验中悄悄换成 GT 框。SAM 用 GT 框得到好 mask，不等于模型能自动找到人。

### 2.2 基线与随机过程

保存代码 revision、用户未提交 diff、展开后的配置、依赖版本、split/data/evaluator hash。存在可信 B0 checkpoint 时复评一次；缺失则标记“未复评”，不要填入历史分数冒充当前结果。

新代码关闭 SAM 功能后，应与原 B0 的前向输出、loss 和导出一致，容差按原精度和非确定性声明。新模块不应通过额外消耗随机数改变共同参数初始化。paired pilot 的两个配置使用可核对的共同初始化和数据顺序。

## 3. SAM 版本与环境：优先离线生成，不污染训练环境

本手册统称 SAM 家族。优先复用服务器已有、可信且能工作的 SAM 2 / SAM 2.1 安装和权重；没有时，建议用官方 SAM 2.1 Hiera-Small 作为首次验证模型：

```text
checkpoint 文件名：sam2.1_hiera_small.pt
官方配置标识：configs/sam2.1/sam2.1_hiera_s.yaml
```

这是首轮资源折中，不是精度最优结论。权重与配置必须成对核验，不能用 SAM 2 权重搭配 SAM 2.1 配置。若仅已有原版 SAM，允许先完成同样的 box-prompt 验证，但明确实验变体和 API 差异，不冒充 SAM 2 结果。

执行要求：

1. 记录官方源码 commit、checkpoint SHA256、配置内容/hash、设备、精度及后处理选项；不要只写“latest”。
2. 先检查已安装版本。当前官方 SAM 2 文档列出的推荐/要求环境包括 Python 3.10+、PyTorch 2.5.1+ 及匹配的 torchvision；执行时以选定 commit 为准。[官方安装说明](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)
3. 不为 SAM 在现有 GAD 环境直接执行会升级核心依赖的安装。发生版本冲突时，在项目允许的位置建独立生成环境，离线写出通用 mask 缓存，原训练环境只读缓存。
4. 自定义 CUDA 扩展是否可用、后处理是否跳过，都写入 manifest；同一对照使用相同设置。若扩展妨碍首次验证，可按官方安装选项禁用，但不能静默改变结果来源。
5. 优先只取一个模型权重，不下载整套 checkpoint；先测磁盘与样本耗时，再决定扩大。
6. 首次用 FP32 跑通；若使用 AMP，先核验设备支持，做少量结果/稳定性对比并记录精度。不要无条件在不支持 BF16 的设备上启用 BF16。

## 4. P1：用已有 actor box 得到对应人物 mask

### 4.1 首选 box-prompt，而非自动 segment-everything

我们已有人物位置，首轮直接问“这个框对应的人在哪些像素”，而不是先生成所有区域再解决复杂匹配。与 SAM-CP 的全图候选区域再组合相比，这能隔离“人物 mask 是否有用”的问题。

provider 的语义接口如下，允许按项目规范命名：

```text
输入：
  image_rgb       uint8[H0,W0,3]，未经模型归一化的原始 RGB
  boxes_xyxy      float32[N_valid,4]，与该 RGB 对应的原图像素坐标
  actor_uids      N_valid 个稳定身份键
输出：
  每位 actor 一个原图分辨率 mask、SAM 质量预测、有效性与失败原因
```

实现步骤：

1. 读取原始 RGB；核对不是 OpenCV BGR，也不是模型的标准化 tensor。
2. 将有效人物框变换到这个原图坐标。检查 finite、顺序、边界和退化框；padding/dummy 不发给 SAM。
3. 同一帧只调用一次 `set_image`，复用图像 embedding。逐 actor 预测先作为正确性基线，之后才按已安装 API 优化批处理。
4. 用 `SAM2ImagePredictor.predict` 传入单个 `XYXY` box，首轮 `multimask_output=False`、无 point prompt，保持官方默认二值 mask 阈值。当前默认坐标处理接收原图像素框；不要预先缩放到 SAM 内部尺寸后又重复转换。[官方接口源码](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py)
5. 显式检查返回维度，取回原图 H0×W0 的 mask；不要把低分辨率的第三个返回值当最终图。用显式索引处理候选轴，不对所有轴盲目 `squeeze()`。
6. 记录该 actor 的输出，即使失败也保留记录；不能过滤掉失败人物，改变 GAD 的输入人数。
7. mask 与当前输入框的对应由调用时的 actor UID 保证，不用 activity、GT membership 或“与真值群组最像”来选择 mask。

首轮保留 raw mask；不默认强裁到框内、不做全图 mask NMS、不强制不同 actor 的像素互斥、不只留最大连通域。遮挡人物可能是不连通的，这些后处理都可能改变实例与上下文，应后续独立验证。

### 4.2 可选提示消融，不自动展开大矩阵

主线固定 `box_only`。若失败主要来自邻人串入，最多先在固定训练诊断样本上比较一项自动提示改动：

- `box + center point`：框中心并不一定在可见人体上，必须观察其失败，不当作真值点。
- 或多候选输出：使用模型自身的质量预测选候选，记录规则；不使用像素 GT 选最优 mask。

负点只能来自推理时可获得的信息，并明确其可靠性；不因邻框中心出现就无条件视为背景。人工纠正点仅作为独立 oracle 诊断，不能混入自动方法成绩。

SAM 的质量预测不是已测得的 IoU，也不保证类别就是 person。用人的框提示得到的 mask 仍可能包含桌子、邻人或只覆盖身体的一部分。

### 4.3 Smoke 交付

最多 8 个独立原始帧，覆盖单人、重叠、遮挡、小人、边界/截断等情况。保存原图、box+actor ID、raw mask overlay 和失败原因。检查一帧多人物时所有输出均回到正确 UID。

计时区分：加载权重、首次热身、每帧 image encoder、每人物 prompt decoder、读写缓存；CUDA 计时正确同步，报告峰值显存和人物数。不能用单个 prompt 的耗时冒充整帧所有人的耗时。

## 5. P2：固定诊断集与离线缓存

### 5.1 固定清单

先冻结抽样种子和 manifest，再看结果。首轮建议总计不超过 200 个唯一原始帧，其中 train 120、val 80；同时设 actor prompt 上限。样本不足或预算不足时记录实际数量，不补造。

覆盖不同视角、人物尺度、人数、遮挡与近邻情况；包含 individual、同活动多组以及交错群组。不得只挑清晰大人物。训练与验证分开存表；真实 test 暂不读取、不用于提示或阈值选择。

诊断帧集不等于训练 clip 集。若下游 pilot 每个 clip 实际采样 T 帧，必须把它需要的全部帧和 actor 加入相应 pilot manifest；不能用缺 mask 的帧默默凑训练。随机采帧训练需预先列出该有界 run 的采帧清单，或缓存其全部可能采样帧；后一种超预算时不执行。

### 5.2 缓存键与格式

缓存位于新的独立目录，不写入或改动数据集原文件。建议按 cache version / video / frame 组织，在记录中显式保存 actor 身份；不依赖文件排列推断人物 ID。

每条记录至少包含：

```text
schema_version, cache_version
dataset, protocol, split, video_id, source_frame_index, source_image_key
source_image_hash, original_hw
actor_uid, actor_id_scope, source_box_xyxy, box_source, box_hash
sam_family, sam_commit, checkpoint_sha256, sam_config_hash
prompt_policy, prompt_hash, precision, postprocess_hash
mask_encoding, mask_shape, mask_checksum
predicted_quality, selected_candidate
generation_status, raw_mask_valid, quality_flags, failure_reason
```

缓存版本由模型、配置、prompt、阈值与后处理等共同决定；同一个版本内不混入不同策略。至少在构建 manifest 时校验源图与框的 fingerprint。model/框/源帧不匹配时必须拒绝旧缓存。

- 推荐 COCO-style RLE，并明确 H/W、列优先编码及 JSON counts 的保存方式；没有兼容依赖时可用压缩二值数组，但记录完整解码规范。
- 保存裁剪 ROI 而非整图时，必须保存原图尺寸、裁剪起点和范围，使完整 mask 可无损恢复。
- 首轮保留二值 raw mask；如另存 logits，标明其分辨率、尺度、阈值，不能冒充概率。
- 写入采用临时文件与原子完成标记。`--resume` 仅复用校验通过的完成项，不把半写文件当成功。
- 输入身份相同但重复出现在多个 clips 时允许复用；先审计跨 split 源数据重叠，不能靠缓存目录掩盖划分问题。

### 5.3 训练读取规则

SAM 不放入 DataLoader workers，也不在每个 epoch 重算。训练只读取缓存；同一帧的图像、框、mask 使用同一组几何增强参数。

原始二值 mask 的离散几何变换使用合适的 mask 插值策略并记录；投到特征网格时允许 area/bilinear 得到软覆盖权重。禁止每次增强后都再次硬阈值化，导致小人物消失而不记录。

区分以下状态：

| 情况 | 行为 |
| --- | --- |
| actor/frame 本来无效或 padding | 沿用 B0 有效性，不参与新增处理 |
| 应有缓存却缺失、身份/尺寸/版本不一致 | preflight 或运行时报错；不静默忽略 |
| 生成失败或 raw mask 空 | 显式标记；该 actor 特征回退普通 RoI，计数 |
| mask 变换后被 crop 掉，actor 同时无效 | 沿用原有效性，不误当 SAM 失败 |
| actor 仍有效，但特征网格 mask 近空 | 该 actor 特征回退普通 RoI，单独记录网格消失率 |
| 高置信但疑似错人/串人 | 保存 QC flag；不能假定 SAM 分数能自动识别错误 |

首轮自动 fallback 只覆盖明确的失败/空值/近空等结构性情况；area ratio、质量分数等阈值若用于拒绝 mask，须只在训练诊断集预先确定、配置化并冻结，同时保留未经该过滤的 QC 结果。

回退不改变 actor 数量、ID、group labels 或 evaluator。报告 raw success、usable coverage 和实际 fallback 比例，避免靠大量回退制造“稳定”。

## 6. QC：如何判断人物 mask 有效

### 6.1 有独立像素真值时

先确认标注是可见人体还是 amodal 全身，以及 ignore 区域语义。按对应 actor ID 计算 mask IoU、Dice、IoU 阈值成功率；必要时报告边界指标。失败和空 mask 不从分母删除。

按尺度、拥挤/框重叠、遮挡和视角分桶；同时报告 actor 加权和按 frame/clip 聚合的结果。若人工真值不存在，不创建“用 SAM 输出当 GT”的自我评价。

如果用户另行提供少量人工标注，可在固定 train-calibration / held-out QC 子集上定量检查；标注工作和人工点击不由服务器 Codex 假定已经完成。

### 6.2 没有像素真值时

仍可交付以下可核验诊断，但全部称为 proxies / QC，不叫真实分割精度：

- 同时列出 manifest 中有效 actor-frame 总数、已尝试、成功、失败、未处理数量。生成成功/空 mask 率按已尝试项计算，处理覆盖率按全 manifest 计算；不能把尚未处理项当生成失败，也不能隐去它们。
- 缓存完整的下游清单上，usable coverage 和 fallback 率按其中所有有效 actor-frame 计算；部分 QC 的比率明确标注作用域与分母。
- mask 面积 / prompt box 面积，mask 落在框内的比例、bbox 覆盖情况、连通域统计。
- 邻人物 masks 的重叠、同帧重复/近相同 mask、异常背景覆盖；这些只触发检查，不自动等于错误。
- 在训练特征网格上剩余的覆盖面积、近空率；不能只看原图高分辨率效果。
- 固定样本画廊：框/actor ID、各自 mask、邻人关系；困难样本与失败样本不能删掉。
- 固定 track 的时序可视化。运动/遮挡会改变 mask，不能把相邻帧原位 IoU 低直接称为身份漂移。

执行者可做视觉初筛并列出“疑似错人、串人、只分到物体、漏肢体、边界噪声”等类别；自身判断不冒充独立人工像素 GT。没有独立真值时，分割 IoU/Dice 留 null，并说明证据等级。

### 6.3 进入下游实验的门槛

必须通过坐标、身份、编码和增强测试，报告所有有效 actor 的生成状态，没有系统性偏移/换人，实际代价在预算内。若固定画廊暴露系统性错人或大量人物消失，先修复或比较一个有界提示变体，不靠长训练掩盖。

无像素 GT 时，只能认定“具备进入下游试验的工程条件”，不能写成“已证明高质量人物分割”。技术门槛通过后，可在明确这一限制的前提下开展有界 pilot。

## 7. P3/P4 主线：SAM-guided MaskRoI 人物特征

### 7.1 最小实现

设现有人物 RoI 为 `R_i = RoIAlign(F, box_i)`，局部 mask 权重为同一采样网格上的 `S_i`：

```text
普通 RoI：     actor_i = existing_actor_head(R_i)
SAM MaskRoI： actor_i = existing_actor_head(R_i * S_i)
```

这里使用原项目的 actor box，不改成 mask 新外接框；保留 RoI 输出形状、actor head 和输出维度。这样首轮只改变 RoI 内的空间过滤。

对齐实现需要明确：

1. 原图 mask 先随图像/框同步增强。
2. 将 mask 映射到与 F 相同的特征坐标网格，再以与 F 完全一致的框坐标、RoIAlign 参数和输出尺寸取样；或实现经过测试的等价公共采样网格。
3. 核验 `aligned`、`sampling_ratio`、`spatial_scale`、像素中心和图像 padding。不得对框先缩放后再次乘 stride。非方形输入和实际特征尺寸不能靠一个猜测 stride 推断。
4. 使用连续覆盖权重。若该有效 actor 的 S 近空则按预先固定 eps 回退原 R，并记录；eps 写入配置。
5. S 来自冻结缓存，无须对 SAM 求梯度；`R*S` 到 backbone/actor head 的梯度必须保留，不能把 R 或 actor features 一并 detach。

首轮不增加独立的归一化加权平均池化、新投影层、残差分支或特征 concat。若现有模型只有全局 actor vector 而拿不到 spatial RoI，先定位其生成位置；无法实现上述乘法时说明限制，另命名 masked-pooling 变体，不能伪称同一实验。

### 7.2 公平对照

| 编号 | 起点与改动 | 用途 |
| --- | --- | --- |
| B0 | 确认的当前基线，普通 RoI | 成对训练对照 |
| S2F-ONES | 新路径输入全 1 的有效 mask | 等价性测试，不必单独完整训练 |
| S2F | B0 + SAM MaskRoI；其他路径不变 | 首轮主实验 |
| S2F-SCALE（诊断项） | 不使用空间形状，只按对应 S 的平均值缩放 R | 若 S2F 有收益，排查纯特征幅度变化；非首轮必跑 |

S2F-SCALE 是针对性诊断，不等于完整排除所有正则化因素。所有配置同样保留 B0 的 backbone、T、slots、activity、matching、membership 归一化、损失、解码和置信度定义，不新增 SAM mask loss。

MaskRoI 会降低部分激活幅度，也可能去掉人与物体交互所需上下文；所以“背景更少”不保证 activity 更好。出现下降时先做这类归因，不自动同时加新 backbone、文本分支或 affinity decoder。

### 7.3 训练与推理的信息边界

S2F 在训练、验证和部署推理时都需要对应输入人物框的 SAM masks。原 evaluator 不变，不代表方法的推理输入没有变化。即使缓存把在线 GAD 前向加速，也须报告 masks 生成的总成本。

mask provider 和 feature forward 不能访问 GT group membership、activity 或评估答案。给定同样 RGB、输入框和模型，改变这些标签不得改变生成的 mask 或预测。GT box 的使用是否允许，以第 2 节核实的协议为准。

## 8. 可选支线：只用 SAM 改进 heatmap target

本支线应排在分割 QC 后，且已有 H1-Sup 可复现时才开展；不是执行首轮闭环的必要条件。

继承确认的 H1-Sup，保留原人物 RoI、membership、matching 和所有解码。训练 target 由真值 group 的人物 masks 并集构造：

```text
Y_g,t(x,y) = max over valid member i of M_i,t(x,y)
```

这是“成员身体支持区域”，不是完整的交互/活动语义区域；不填充凸包或把成员间所有空隙当正例。

- GT membership 只用于训练 target，推理不需要它，也不需要在线 SAM（前提是本支线未改人物特征）。
- target 使用训练人物的缓存；验证 mask 标签仅用于明确标注的监督损失诊断，不参与模型预测或 checkpoint 输入。
- 生成失败的 actor 必须按固定规则回退原 H1-Sup 的 actor target，或显式忽略其监督区域；选择一种并冻结。若采用回退，把结果标记为混合 target，报告每组/每帧覆盖率。
- 某组某帧无有效成员时，不当成真实空组；不把 padding 或缺标当负例。
- 降采样时保留小人物覆盖并明确插值；概率/软覆盖 target 的 loss 必须支持软标签。

实验命名为 `H1-GAUSS`（原 target）、`H1-SAM-TARGET`（仅 target 改变）。若要声称人体形状优于一般占据监督，还应加同分辨率的 box-occupancy target 控制；Gaussian 与 SAM target 的差别不只有边界，还包括面积、软硬程度及 loss 密度。

首轮不同时跑 `S2F + H1-SAM-TARGET`。只有两条支线的独立贡献有证据后，才考虑组合。

## 9. 接入训练前的必要测试

优先写小而有针对性的单元/诊断测试，使用项目现有测试框架；不用复制整套训练实现。

| 检查 | 验收证据 |
| --- | --- |
| 编码往返 | 非方形合成 mask 经保存/解码后逐像素一致，H/W 不交换 |
| 坐标 | 不同尺寸、归一化框、越界/退化框得到正确处理；overlay 对应原图 |
| 增强 | resize/crop/flip 后 mask、box、图像一致；翻转两次可恢复离散身份 |
| actor 身份 | actor 重排后 mask/features 可反向恢复；不同 frame/video 的同名 ID 不碰撞 |
| 同帧多人 | image embedding 复用，但输出与各自 prompt/UID 不串位 |
| 缓存完整性 | checkpoint/box/image 改变触发失效；坏文件/缺文件按契约报错 |
| 有效性 | padding、无有效人物、无群组、mask 空、全部 fallback 不产生 NaN |
| 关闭功能 | disabled 模式与原 B0 前向/loss/导出兼容 |
| 全 1 mask | eval 模式下 S2F-ONES 与普通 RoI 一致，报告误差和容差 |
| 梯度 | 固定非空 S 时，原 loss 到 backbone/actor head 有有限梯度 |
| 因果依赖 | 固定 F/boxes 时移动或置换 S，局部特征确实改变；关闭开关则不受缓存影响 |
| GT 隔离 | 改 group/activity labels 不改变 SAM 输出与固定模型预测 |
| 评价兼容 | actor ID、预测导出和原 evaluator 往返不变；没有新增隐藏过滤 |

不要求每个样本的最终 activity 都因换 mask 而变化，但必须证明新增空间过滤真正进入特征路径。单纯 loss 非零、梯度非零或图变好看，都不等于下游有效。

## 10. 预算内小实验与正式评价

下列是启动上限建议，不是已经获批的服务器 GPU 配额；始终取“本表上限”与“项目已分配剩余额度”中的较小者。

| 操作 | 默认上限/约束 |
| --- | --- |
| 首次分割 smoke | 8 帧、100 actor prompts、10 GPU 分钟，先到即停 |
| 首轮 QC 生成 | 总计 200 唯一帧、3,000 prompts、60 GPU 分钟，包含 smoke，先到即停 |
| GAD 技术 smoke | 每个配置最多 3 train batches |
| 小集 overfit | 8–16 train clips；每个配置最多 200 optimizer steps |
| paired pilot | 固定不超过 64 train clips / 32 val clips；B0 和 S2F 各最多 500 optimizer steps |
| 总训练预算 | 首轮 GPU 训练合计不超过 2 GPU 小时；多 GPU 计设备时数之和，先到即停 |
| pilot masks | 需要的所有帧另建清单并先估算；加上 QC 的首轮生成合计仍限 60 GPU 分钟，超额则缩小 pilot 或交付资源申请 |

每个预算都含失败重试；中断/失败记录保留。不要让命令因“继续研究”无限重启。资源不足时保留已完成产物、继续 CPU 检查，并明确未运行项。

### 10.1 成对协议

- B0 和 S2F 使用相同 train clips、采样帧、增强随机过程、共同初始化、有效 batch、optimizer、scheduler、steps 和验证清单。
- 不通过冻结部分 B0、仅 warm-start 新方法等方式形成不公平对照。若统一 warm start，两者来自同一 checkpoint 并接受相同追加训练预算。
- overfit 只做工程诊断；pilot 小验证集只做趋势诊断。不与完整 VAL 历史分数混在一张表中冒充提升。
- pilot 对照默认比较固定最后一步；若使用验证选模，两者同一预声明规则，另报固定步结果。
- 缓存或增强引入的新 RNG 不应改变共同权重和采帧；可复用已采样变换参数或独立 RNG。
- 不在 validation/test 上反复调 prompt、质量过滤、成员阈值和 mask 后处理。

### 10.2 下游指标

沿用实际 evaluator。若确认是此前 CAFE 协议，报告 group mAP@1.0、group mAP@0.5、outlier mIoU、per-class AP，补充 membership precision/recall/F1、漏/多成员、拆组/并组、活动错分、置信度错误和 fallback 分桶。

Group IoU 是成员集合 IoU，不是人物 mask 的像素 IoU。前者改善才能支持 grouping 收益；后者即使更好，也不能替代 GAD 指标。

完整实验只有在预算明确、技术验证通过后开展。核心结论优先至少 3 个配对 seeds，报告每个 seed、均值/标准差和配对差值。单 seed 或 pilot 正结果只称初步趋势，不称稳定或统计显著。

### 10.3 成本与决策

分别报告 SAM 离线生成的帧/人耗时、总 GPU 时数、峰值显存、磁盘；缓存读取开销；GAD 训练/前向吞吐。若方法推理需要 SAM，报告含 SAM 的端到端成本，并说明是否已知人物框。

- 坐标/身份出错：暂停扩大生成和长训练，修复后重跑同一诊断集。
- 原图 mask 尚可但特征网格消失：先检查分辨率/采样，不能直接宣称 SAM 无效。
- mask 可用而 S2F 下降：检查上下文丢失、幅度变化、fallback、时序闪烁、训练公平性；做至多一个有证据的诊断，不无限堆模块。
- mask 画廊更漂亮而下游不变：仅支持分割呈现/输入变化，尚不支持 GAD 改善。
- S2F 有正趋势：先核验成对协议，再决定扩大样本和训练；不立即归因于 SAM-CP affinity（本轮没有实现它）。
- 没有独立分割 GT 但下游改善：可以报告给定框协议下的任务收益，不能补写未经测量的分割精度。

## 11. 可执行接口契约与运行顺序

服务器结构未知，以下脚本是需要执行者实现或映射到现有工具的接口，**不是本机已经存在、已经跑通的服务器命令**。复用项目现有 CLI 优先；如映射到不同命名，必须生成 `commands_actual.md`，保存实际运行过的完整命令。

### 11.1 最小文件职责

```text
tools/sam_person_masks.py
  audit       项目/环境检查；不初始化 SAM 或启动训练
  manifest    使用真实 dataset adapter 导出有界原帧/actor 清单
  validate    检查配置、源图/框、模型/权重、缓存规则；未解析必填项则报错
  generate    单图 provider、缓存、计时、上限、resume
  qc          缓存一致性、覆盖/proxy 指标，可选独立 GT 指标
  visualize   固定画廊与失败图

项目规范内的 mask reader / 同步几何变换 / 可开关 MaskRoI
项目规范内的必要 tests
configs/sam_person/qc_resolved.yaml
configs/sam_person/pilot_resolved.yaml
docs/sam_person/{project_audit,decision_log,commands_actual,report}.md
```

新工具的数据 adapter 必须调用或与真实 dataloader 一致，不能另外编一套未经核验的 actor ID/帧号解释。

### 11.2 resolved 配置必须具备的内容

由 P0 自动/人工核验后填写真实值，不把占位路径直接提交运行：

```text
project_root, dataset_adapter, protocol, split_manifest_paths/hashes
source_frame_mapping, actor_box_source, box_coordinate_convention
sam_python, sam_repo, sam_commit, checkpoint, checkpoint_sha256, model_config
device, precision, prompt_policy=box_only, multimask_output=false
mask_threshold, postprocess_options, cache_root, cache_version
sample_seed, per_split_limits, max_frames, max_actor_prompts, max_gpu_minutes
train_mask_mode=disabled|sam_roi, empty_eps, fallback_policy
pilot_clip_manifests, sampled_frame_manifest, paired_init/seed, run_budget
```

校验必须拒绝未解析的必填路径、未知框格式、错误权重配对和超预算计划。路径有空格时正确引用；配置文件只存路径和公开运行参数，不输出凭据。

### 11.3 接口实现后的命令示例

以下相对路径均相对于服务器真实项目根目录。`python` 指激活后的正确环境；独立 SAM 环境运行 generate，GAD 环境运行训练，实际解释器绝对路径写入命令记录。

```bash
python tools/sam_person_masks.py audit --project-root . --out docs/sam_person/project_audit.md

# P0 后先生成两份已解析配置；manifest 在读取图像 embedding 前即可导出。
python tools/sam_person_masks.py manifest --config configs/sam_person/qc_resolved.yaml --out outputs/sam_person/qc_manifest.jsonl
python tools/sam_person_masks.py validate --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/qc_manifest.jsonl

# 小 smoke：触达任何上限即正常停止，结果标记 partial，不伪报全清单完成。
python tools/sam_person_masks.py generate --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/qc_manifest.jsonl --max-frames 8 --max-actor-prompts 100 --max-gpu-minutes 10
python tools/sam_person_masks.py qc --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/qc_manifest.jsonl --scope completed --out outputs/sam_person/smoke_qc.json
python tools/sam_person_masks.py visualize --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/qc_manifest.jsonl --scope completed --out outputs/sam_person/smoke_gallery

# 几何/身份验证通过后再扩大；生成预算按 ledger 累计，不能重置计时逃过总上限。
python tools/sam_person_masks.py generate --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/qc_manifest.jsonl --resume --max-frames 200 --max-actor-prompts 3000 --max-gpu-minutes 60
python tools/sam_person_masks.py qc --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/qc_manifest.jsonl --require-complete --out outputs/sam_person/qc_metrics.json
python tools/sam_person_masks.py visualize --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/qc_manifest.jsonl --out outputs/sam_person/qc_gallery
```

达到预算但未完成时，`--require-complete` 应失败并输出缺项表；保留 partial QC，不强行标记通过。边界/身份错误必须以非零退出码结束。

随后根据真实训练入口生成并执行 B0/S2F 的 smoke、overfit、pilot 命令。这里不虚构项目的训练参数名。最终 `commands_actual.md` 不允许只留下 `YOUR_PATH`、`TODO` 或未实现的脚本。

## 12. 产物、报告与完成标准

建议使用新命名空间：

```text
docs/sam_person/
configs/sam_person/
outputs/sam_person/<cache_or_run_id>/
results/sam_person/
```

每次运行记录 revision/diff hash、配置、真实命令、环境、输入 manifest hash、模型权重/缓存版本、seed、采样帧、预算 ledger、耗时/显存、状态和失败原因。训练记录初始化、有效更新数、选模规则、指标及预测导出。

QC 表保留所有 actor-frame 的状态；训练主表至少包括：

```text
run_id, variant, base_checkpoint_or_init_hash, code/config/evaluator_hash
protocol, split_hash, train/val_clip_manifest_hash, sampled_frame_hash
seed, steps, effective_batch, lr_schedule, checkpoint_selection_rule
sam_version, checkpoint_hash, prompt_policy, cache_version
raw_success_rate, usable_mask_coverage, fallback_rate, feature_grid_empty_rate
group_map_1_0, group_map_0_5, outlier_miou, membership_f1
sam_gpu_hours, train_gpu_hours, peak_memory, cache_bytes, status, failure_reason
```

无真值/未运行指标写 null，不写 0。失败、中断和负结果不删除。固定画廊同时包括成功和失败，overlay 提供 actor ID，便于查身份而非只看美观。

首次交付必须清楚区分：代码已实现、CPU/几何测试已通过、SAM smoke 已完成、QC 已完成、GAD pilot 已完成、完整训练未运行。最终报告回答：

1. 当前基线和人物框协议是什么？
2. 使用哪个 SAM、什么提示，在多少独立帧/人物上实际测过？
3. masks 的身份/几何是否正确？有没有独立真值；哪些是 proxy？
4. 遮挡、小人物、邻人、手持物和特征网格消失分别造成什么失败？
5. S2F 是否真正改变人物特征？相比同预算 B0 的结果如何？
6. 生成、存储和部署成本是什么？fallback 是否掩盖问题？
7. 现有证据支持什么，不支持什么？下一项最小实验及所需资源是什么？

## 13. 可直接发给服务器 Codex 的启动消息

> 请阅读并执行项目中的 `SAM_PERSON_MASK_SERVER_CODEX_PLAYBOOK.md`，在现有 CAFE/GAD 项目上开展独立的 SAM 人物 mask 验证。先按项目 AGENTS.md 核验代码、人物框来源与坐标、帧号/actor ID、B0、环境和已分配预算；复用已有资产。首轮用冻结的 SAM 2/2.1 和已有 actor box 逐人分割，同帧复用 image embedding，离线缓存原图 masks，先完成小 smoke、固定诊断集 QC、身份/几何/增强检查及失败画廊。无像素真值时只报告覆盖率、proxies 和可视化，不把 SAM quality score 当真实 IoU。再实现可关闭的 SAM-guided MaskRoI：只对原人物 RoI 特征做局部 mask 过滤，保留 actor head、group reasoning、matching、membership、activity、解码和 evaluator。通过关闭/全 1 mask 等价性、梯度与 GT 隔离测试后，在已分配预算内完成 B0/S2F 成对 smoke、overfit 和有界 pilot，不只输出计划。不得改动原结果、升级破坏训练环境、复现整套 SAM-CP、同时加入 H2/CLIP/FPN/新 grouping 或自动展开完整训练矩阵。资源不足时继续 CPU 可做的检查，保留 partial 产物与准确阻塞项。交付实际代码、配置、已运行命令、日志、QC、成本、下游结果及结论边界；已有明确预算才进一步完整训练。SAM heatmap target 支线作为后续独立实验，不与人物特征收益混在一起。

## 14. 来源与适用说明

- [SAM-CP 论文](https://arxiv.org/abs/2407.16682)：区域组合、MaskRoI 与 query–patch affinity；本手册仅选择其中区域特征过滤的启发，不声称复现整篇。
- [SAM 2 官方仓库](https://github.com/facebookresearch/sam2)：图像/视频 predictor 与官方权重。本次首轮采用图像 predictor，不涉及视频传播实验。
- [SAM 2 安装说明](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)：隔离环境、依赖与可选 CUDA 扩展。
- [SAM2ImagePredictor 源码](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py)：图像输入、框提示、输出及阈值语义。实现时核对服务器锁定版本。
- [SAM 2 坐标变换源码](https://github.com/facebookresearch/sam2/blob/main/sam2/utils/transforms.py)：用于核验归一化与原图尺寸映射，禁止重复变换。
- [Hiera-Small 配置](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml)：首轮建议模型的官方配置。
- 本地 `CAFE_HEATMAP_RESEARCH_EXECUTION_GUIDE.md` 与 `weekly_20261014_说明补充版.md`：提供此前路线、单 seed 观察和协议线索；服务器须重新核验对应版本及结果。

本手册写于本地，未登录服务器，未核实其当前仓库状态，也未执行任何 SAM 推理或 GAD 训练。不得将手册中的接口、预算建议或实验设计当作已完成事实。
