# CAFE 群组 Heatmap 研究执行指南

日期：2026-10-12  
适用对象：能够访问 CAFE/GAD 服务器项目、数据、现有训练环境和实验产物的 Codex  
文件性质：研究任务说明与实施验收标准，不是已完成实验的报告

## 0. 给执行者的任务

请在现有项目中开展 group heatmap 研究。先核验代码和基线，再实现最小方法、验证梯度与解码、进行短实验，最后在项目已有算力预算内开展完整对照实验。不要只输出一份计划后停止。

本文件已经给出了常规实现选择的处理原则。可以依据项目实际情况调整模块位置、配置字段和启动方式，记录理由后继续工作，无须为每个可逆实现选择再次询问。遵循用户当前要求及项目适用的 `AGENTS.md`。本地未核实的路径、配置、数据统计和指标都不能作为既成事实。

本轮首先回答：

> 在不增加人工像素标注的条件下，使用每个 group slot 的独立占据图，并从热图响应读取 actor membership，能否改善 CAFE 的精确群组检测？

随后回答：

> 使用预测群组区域池化视觉特征，并与成员特征对齐，是否改善群组活动识别及成员预测？

完成标准包括可运行代码、配置、验证证据、实际实验记录和结论边界。模型图和可视化本身不构成实验结论。

## 1. 研究范围与首轮边界

### 1.1 采用的表示

- 一个预测 slot 对应一个群组候选，同时拥有 activity、逐帧 heatmap、actor membership。
- 预测的是 instance-level group heatmap，而不是每个 activity 类别一张图。
- 使用独立 sigmoid occupancy。一个群组允许包含多个空间分离区域。
- 真实群组按成员区域构造目标，不使用单个群组中心、group bounding box 或 convex hull 替代成员集合。
- H2 的 membership 必须主要来自热图在 actor 区域内的响应。
- H3 的新增机制是 mask pooling 与 actor–group alignment。appearance residual 是另一个消融，不与 H3 默认捆绑。

### 1.2 首轮保持不变的项目设置

此前分析建议保留 ResNet18、5 帧、12 个 slots、现有 activity 路径、threshold-based individual 解码和 place protocol。先核实它们是否确实属于当前 B0；若项目实际基线不同，以被确认的 B0 配置为准并记录差异。

首轮不同时引入新 backbone、FPN、SAM 推理依赖、CLIP 文本分支、显式 individual 类、重采样或新的学习率策略。原有辅助损失若仍适用，应在 B0/H1/H2 中一致保留；如某项因预测路径更换而失去定义，必须说明并设置相应对照，不能静默删除。

### 1.3 资源与工作区

- 使用现有项目环境和已分配算力，不自动启动付费实例或抢占其他任务。
- 保留用户已有修改、checkpoint 和历史结果。新实验使用独立输出目录，不覆盖旧产物。
- 先检查已有研究代码和 run registry，复用已完成且配置一致的工作。
- 已有预算允许完整训练时，按阶段推进。预算不明确时，先完成代码、测试和有界短实验，交付具体配置及完整训练所需的估计算力，不擅自启动大规模矩阵。
- 默认有界短实验建议为：smoke 3 个训练 batch，overfit 最多 500 optimizer steps，paired pilot 每个配置最多 1000 optimizer steps。它们是本指南的启动预算，不是用户给定预算，也不能用于宣称最终性能。
- 运行完整训练前用短实验测量 steps/s、显存和预计总时长，记录实际可用 GPU、并发上限与 run 清单。

## 2. 阶段 0：核验项目，冻结可比较的 B0

### 2.1 查清真实数据流

读取实际的模型、dataloader、matcher、criterion、decoder、evaluator 和训练入口。此前服务器分析中的候选位置为：

```text
copyfrom_aixi_gadfortest-b0-repro/models/models.py
copyfrom_aixi_gadfortest-b0-repro/models/group_transformer.py
copyfrom_aixi_gadfortest-b0-repro/models/group_matcher.py
copyfrom_aixi_gadfortest-b0-repro/models/criterion.py
```

这些是搜索线索。若路径不存在，按职责搜索，不创建同名空壳，也不假定当前 checkout 就是该版本。

至少确认并记录：

| 项目 | 需要核实的内容 |
|---|---|
| 输入 | 视频 tensor、采样帧、resize/crop/flip、actor ID 和有效框标记 |
| 框坐标 | xyxy/xywh、归一化/像素坐标、变换前后尺寸、边界约定 |
| 场景特征 | 实际 stride、空间尺寸、通道数和时间布局 |
| group query | tensor 轴顺序、是否逐帧、clip-level 聚合方式 |
| `feature_att` | 是否存在、head/layer/时间轴、actor/group 切片顺序 |
| membership | logits 来源、softmax/sigmoid 轴、loss 接收 logits 还是概率 |
| 原辅助损失 | consistency 的对象、梯度路径和权重 |
| matcher | cost 定义、空目标处理、clip 级还是逐帧匹配 |
| decoder | no-group 处理顺序、阈值、单 actor 冲突处理、最小群组规则 |
| evaluator | 导出格式、score 定义、三个指标、异常/空结果处理 |
| 训练 | optimizer、初始化、AMP、epoch/step、有效 batch、checkpoint selection |

生成一张以代码符号和真实 tensor shape 为依据的数据流说明。不得用 PPT 中的画法代替代码核验。

### 2.2 核验基线

1. 保存当前代码 revision、未提交 diff、环境版本、配置和 evaluator hash。
2. 如果有被信任的 B0 checkpoint，先复评一次，并保存完整命令与输出。
3. 核对 split 文件、数据清单、预处理和 checkpoint 是否一致。不把历史笔记中的分数直接填入新结果表。
4. 用新代码关闭所有 heatmap 选项后，验证输出和指标与 B0 一致；容差依据计算精度和项目已有非确定性声明，不任意放宽。
5. 新增模块不能消耗随机数后悄悄改变 B0 共同参数的初始化。共同权重、预训练来源、seed 列表和数据顺序要可核对。

交付：`baseline_audit.md`、配置快照、基线复评记录、已有结果清单。

**通过条件：** 已知道当前 B0 的真实定义，且 heatmap-disabled 模式可复现。缺少 checkpoint 时明确标记未复评，先完成短基线；不得把其他配置的历史结果当成对照。

## 3. 阶段 1：attention probe 与目标/解码 oracle

正式训练前先分清两个问题：模型是否有空间线索，以及所选目标和读取方式是否有能力恢复成员。

### 3.1 只读 attention probe

若基线确有 group spatial attention，暴露其 group 部分并按当前匹配结果关联到 GT 群组。沿 actor tracks 读取成员与非成员位置的响应，保存分布和可视化。

- 不改变原预测、损失或解码。
- 保留 attention 的原始归一化，不把它直接命名为 sigmoid occupancy。
- 固定要观察的 layer/head 聚合方式和样本清单。若比较多种聚合，全部记录。
- attention 缺少定位能力不等于独立 mask head 不可行；它也不是后续方法训练的必要前提。

### 3.2 目标生成与坐标验证

从变换后的 actor boxes 和 GT membership 构造目标。目标在最终监督的特征网格上生成，并与预测图采用同一坐标约定。

对群组 g 的人物 i，使用自适应中心高斯：

```text
G[i,t,x,y] = exp(-0.5 * (((x-cx)/sigma_x)^2 + ((y-cy)/sigma_y)^2))
Y[g,t,x,y] = max over valid members i of G[i,t,x,y]
```

sigma 与框宽高关联，具体比例、最小值和最大值写入配置。为目标选择保留一个小的预注册候选集即可，例如框宽高比例 0.15/0.25；它们是待检验设置，不是已证明的最优值。

明确像素中心、连续坐标和边界截断约定。中心不落在网格中心时，高斯峰值可能低于 1，不得假定一定存在 `target == 1` 的像素。

有效性规则：

- padding、dummy、退化框和无效帧不参与目标或成员损失。
- 某帧所有成员均不可用时，将该群组该帧标为监督无效，不自动把它当成真实空群组。
- individual 是已标注群组的非成员，不单独生成一个大 background group mask。
- clip/group/frame 的有效性使用显式布尔标记，不靠全零 tensor 猜测。

### 3.3 Oracle：用理想目标图测试读取上限

将构造出的 Y 暂时作为“理想预测图”，按拟采用的 ROI pooling 和时间聚合读取所有 actor 的响应。比较真成员与非成员的可分性。

重点覆盖：相邻框、框重叠、群组空间交错、人物尺度变化、individual 紧邻群组、同活动的多个群组、边界框和缺失帧。

标定或阈值若需要拟合，只在固定训练子集上确定，然后冻结并在独立诊断集上观察；不要逐 clip 使用 GT 调阈值。验证集 GT 只可用于明确标注的 oracle 诊断，不能进入真实模型推理。

记录：成员/非成员响应分布、F1、集合 Group IoU、不同距离/尺度分桶结果、目标覆盖和热图质量。Oracle 不代表可实现的模型分数，不与 B0 的真实模型成绩混在一起比较。

**通过条件：** 无坐标错误，移动框能读到移动的目标响应；普通样本可恢复成员；困难样本的不可分情况有记录。若理想图已经难以区分近邻，先检查坐标、Gaussian 尺度和 pooling，再决定是否单独测试更高分辨率。不要让大型网络训练替几何错误背锅。

## 4. 模块接口与实现约定

遵循现有项目命名；下面是语义接口，不要求机械照搬字段名。

```text
scene_features       [B,T,D,H,W]
group_queries        [B,T,K,Dq]
group_mask_logits    [B,K,T,Hm,Wm]
group_masks          [B,K,T,Hm,Wm] = sigmoid(logits)
actor_boxes          [B,T,N,4]
actor_frame_valid    [B,T,N]
spatial_response     [B,K,T,N]
membership_logits   [B,K,N]
membership_probs    [B,K,N]
activity_logits     [B,K,C+1]      # 包含 no-group
mask_pooled_features [B,K,Dz]       # H3
```

GT 群组数 G 可变，targets 中保存 `group_activity`、`group_members`、`group_heatmaps` 和有效帧标记。预测路径不能依赖这些 GT 字段。

### 4.1 独立 sigmoid mask head

```text
P_t = Conv_pixel(F_t)
E_k,t = MLP_mask(q_k,t)
L_k,t(x,y) = dot(E_k,t, P_t(x,y))
M_k,t(x,y) = sigmoid(L_k,t(x,y))
```

- mask embedding 与 pixel embedding 的通道必须一致。
- 每个 slot 输出自己的图。空间维度不做 softmax，也不强制各 slots 的像素响应和为 1。
- mask logits 与概率分开命名，避免重复 sigmoid。
- mask logits 用于数值稳定的监督；ROI pooling/pooling feature 使用连续概率，不先硬阈值化。
- 记录 mask head 初始化和参数量。若使用负 bias 抑制初始满图，作为固定配置记录，不能用 validation 结果反复挑初始值。

### 4.2 从热图生成 membership

```text
r_k,i,t = ROI_Pool(M_k,t, box_i,t)
r_k,i = masked_mean_t(r_k,i,t)
ell_k,i = alpha * r_k,i + beta
```

首轮使用一种明确的 pooling，例如 ROIAlign 后对固定网格取平均。固定 output size、sampling ratio、aligned 和 spatial_scale，并核验服务器安装版本支持的接口。坐标只换算一次，不能先缩放框后又重复设置缩放因子。

alpha 保持正值，使空间响应越高，成员 logit 越高；可用正值参数化。初始标定依据训练 oracle 的响应范围，而不是假定框内平均值接近 1。记录 alpha、beta、响应直方图和饱和比例。

整个过程须可微：不允许在训练 membership 路径中 `.detach()`、转 NumPy、硬 top-k 或硬二值化。如果现有算子存在梯度限制，先验证再选替代实现。

### 4.3 归一化公平性：必须区分 H2-S 与 H2-I

如果 B0 在 K 个 slots 上做 softmax，则直接把新分数改成独立 sigmoid，会同时改变分组依据和概率归一化。保留阈值数值并不能消除这一混杂因素。

采用以下实验命名：

| 配置 | 分组依据 | 成员归一化 | 用途 |
|---|---|---|---|
| B0 | 现有 embedding | 原实现 | 原基线 |
| H2-S | heatmap ROI response | 保留 B0 的 group softmax/解码 | 首轮受控比较 |
| H2-I | heatmap ROI response | 独立 sigmoid + threshold | 独立分数版本，单独报告 |
| B0-I | 现有 embedding | 与 H2-I 相同的独立分数协议 | 若主张 H2-I 的收益，用于排除归一化影响 |

主线先做 H2-S；若 B0 本来就是独立 sigmoid，则相应合并配置并记录。H2-I 可作为后续分支，不能把它与 B0 的差异全部归因于 heatmap。

softmax 分支中共享 scalar beta 会抵消，不保留无效 bias 参数。loss 必须接收对应类型：独立 sigmoid 分支可对 logits 用 BCEWithLogits；保留 softmax 概率的分支按原 criterion 的语义处理，不对概率再调用 BCEWithLogits。

no-group 过滤顺序、阈值、actor 冲突决策和最小群组规则首轮保持与 B0 一致。任何改变均需配置化并独立记录。

### 4.4 H3 的 pooling 与 alignment

```text
z_k = sum_valid(t,x,y)(M_k,t * P_t) / (sum_valid(t,x,y)(M_k,t) + eps)
activity_input = q_k + Project(z_k)
```

q_k 采用当前 activity 路径的 clip-level 群组表示。必要时投影维度；不悄悄更换其时间聚合方法。

z 必须由预测 M 产生，不能用 GT target 替代。定义近空 mask 的数值稳定行为并记录其发生率。

对同一 Hungarian 匹配中的群组，计算成员 actor 平均特征：

```text
a_bar_g = mean of valid member actor features
L_align = 1 - cosine(Project_z(z_k), Project_a(a_bar_g))
```

为分清贡献，H3 至少分成 H3-P（仅 pooling）和 H3-A（pooling + cosine alignment）。简单 cosine 只拉近正配对，不能声称已显式排斥非成员。appearance residual、InfoNCE 和 temporal loss 都是独立后续项。

## 5. Matching 与训练损失

### 5.1 使用一个共享的 clip-level assignment

根据当前项目语义，用 activity、mask 和 membership 共同构成 K×G cost，再做一次 Hungarian matching：

```text
C = w_cls*C_activity + w_focal*C_focal + w_dice*C_dice + w_mem*C_membership
```

- 三个预测分支使用同一个 assignment。不得各自匹配后拼接损失。
- 跨帧 mask cost 聚合有效帧，避免每帧单独换组导致 slot 身份不一致。
- matching cost 在无梯度上下文计算，但 matched predictions 的训练损失保留梯度。
- cost 权重与 training loss 权重分开配置；记录各 cost 的量级。
- group/clip 无有效 mask 监督时，不产生 NaN，也不把缺失标签误当负例。
- H2 的 membership cost 来自热图路径；不能在该处继续使用一个独立 embedding head，使方法悄悄回到旧路径。

### 5.2 高斯目标是软标签

Gaussian Y 的取值在 [0,1]。不要直接套用依赖 `target == 1` 的硬正样本 focal 写法。

第一版可以采用明确、可测试的 soft-target focal 定义：

```text
p = sigmoid(logit)
p_t = y*p + (1-y)*(1-p)
alpha_t = alpha*y + (1-alpha)*(1-y)   # 若关闭 alpha balance，则令其为 1
L_focal = alpha_t * (1-p_t)^gamma * BCEWithLogits(logit, y)
```

这是一种本研究选定的软标签扩展，不等同于所有文献/库中同名 focal 实现。将公式与 gamma、alpha 写入配置和报告。增加 gamma=0、关闭 alpha balance 时等于 soft BCE 的验证，并保留 soft BCE + Dice 的诊断配置。

soft Dice 对 matched group 的有效帧计算。建议先逐有效帧平均像素 loss，再对帧和群组平均；不能直接把整个 batch 的正像素堆在一起导致大群组权重更高。

### 5.3 Loss 组成

```text
L = w_act*L_activity
  + w_focal*L_mask_focal + w_dice*L_mask_dice
  + w_mem*L_membership + w_empty*L_empty
  + baseline_aux_losses
  + w_align*L_alignment      # 仅 H3-A
```

- matched slots：正确 activity、群组 target 和成员向量。
- unmatched slots：no-group 与空热图；empty-mask loss 单独归一化，避免 12 个 slots 中大量空项淹没正群组。
- membership 的非成员包含其他群组的人和 individual，padding actor 不计入。
- loss/cost 权重先按数值尺度作小规模 train-only 校准。之后冻结配置，避免无限 validation 搜索。
- H3 的 activity loss 应能通过 z 传播到 mask head；H2 的 membership loss 应能传播到 mask head。
- 首轮不增加跨帧像素相等约束。后续 temporal ablation 若开展，应约束同一 actor track 的有效响应，而不是要求人物移动时整张图不变。

## 6. 分阶段实验矩阵

| 阶段/配置 | 新增或替换内容 | 需要回答的问题 |
|---|---|---|
| B0 | 无 | 当前实际基线能否复现 |
| Probe | 只读已有 attention | 是否已有可观察的空间线索 |
| H1-Sup | 新 mask head、heatmap loss、empty loss，保留 B0 matching/membership | 空间辅助监督本身是否有效 |
| H1-Match | 在 H1-Sup 上加入 mask matching cost | matching 改变的独立影响 |
| H2-S | 在 H1-Match 上把 membership/cost 改为 heatmap 来源，保留原归一化 | 空间读取能否替代 embedding grouping |
| H3-P | 在选定 H2 上加入预测 mask pooling | 活动分类使用区域证据是否有帮助 |
| H3-A | 在 H3-P 上加 cosine alignment | 成员特征对齐的额外作用 |
| H4-SAM | 固定其余项，仅替换/融合监督目标 | 更细的目标是否改善结果 |

H2-I/B0-I 属于归一化分支，不默认塞进主线。H3/H4 必须注明继承 H2-S 还是 H2-I。不能在一张结果表中把不同归一化分支当作完全相同的 H2。

此前方案把 auxiliary heatmap 与新增 matching cost 合称 H1，本指南拆成 H1-Sup/H1-Match，以分清变量；把 H3 拆成 H3-P/H3-A，以分清 pooling 与 alignment。保留和旧命名的映射。

执行顺序：

1. audit、B0 复评、probe、oracle。
2. 实现 H1/H2；完成第 7 节验证后，进行有界 paired pilot。
3. 技术健康且预算允许时，完成 B0、关键 H1、H2 的同协议完整实验，核心比较至少三个独立 seeds。
4. H2 表现不佳时先诊断。可做一次有界 H3-P pilot 检查缺少语义耦合是否是原因，但不能省略失败的 H2 结果或直接宣布方法有效。
5. 再推进 H3-A 和有明确假设的消融，不做所有参数的笛卡尔积。

pilot 负结果不自动否定方法；一次 pilot 正结果也不证明有效。是否推进由错误分析、训练健康度、实际预算和完整训练可比较性决定，理由写入 decision log。

## 7. 正式训练前必须通过的验证

这些是新研究路径的正确性检查，应以小型、针对性测试和诊断脚本实现，不需要复制整套训练实现写冗余测试。

| 检查 | 应得到的证据 |
|---|---|
| 坐标/变换 | 非方形图片、resize、crop、flip、边界框均能对应到目标网格 |
| 非连通目标 | 同组两个远离人物保留两个峰，中间非成员不因 hull 被覆盖 |
| 同活动多组 | 两个相同 activity 的群组仍对应独立 targets/slots |
| 有效性 | padding、dummy、缺失帧、空 GT、无效框不产生错误监督或 NaN |
| H2 梯度 | 单独反传 L_membership，mask head 得到有限且非零梯度 |
| H3 梯度 | 单独反传 activity loss，mask head 经 z 路径得到梯度 |
| 因果依赖 | 固定其他输出后移动/置换 M，membership 相应改变；替换 mask-pooling 输入改变 H3 活动 logits |
| slot/actor 置换 | 重排 slots 后集合损失/结果保持语义一致，重排 actor 索引后对应恢复 |
| GT 隔离 | 推理接口不需要 GT group/activity；改变评估 GT 不改变固定模型的预测 |
| 空图与 no-group | unmatched 图被抑制；近空 pooling 有稳定定义；individual 没有被固定强制纳入 |
| 关闭新模块 | B0 输出和评估兼容；没有额外阈值或过滤偷偷生效 |
| evaluator 往返 | 新输出可被原评价代码读取，actor ID/boxes/group IDs/activity/score 对应正确 |

H3 梯度非零只证明路径连通，不证明模型训练后充分使用了 z。后续还要做固定验证集上的 query-only、替换/打乱 mask 等机制诊断，报告其局限，不把人为破坏输入的分数当公平模型对照。

### 7.1 Smoke

少量 batch 完成 forward、loss、backward、optimizer step、保存/加载和 evaluation export。检查每个 loss、gradient norm、mask range、valid counts 和内存。

### 7.2 小样本 overfit

固定约 8–16 个训练 clips，包含多组和 individual。固定采样、降低不必要的随机增强，不调整 validation 来帮助 overfit。

观察热图定位、成员/非成员分离、成员集合恢复和 unmatched 抑制是否随训练改善。500 steps 预算用完仍失败时，记录 failure 而非静默增加成完整训练。

### 7.3 Paired pilot

采用固定训练子集和诊断样本；B0/候选配置使用相同 optimizer steps、样本顺序、初始化协议、有效 batch 和评估频率。额外模块会改变计算成本，因此同时记录 wall time，避免“同时间”和“同步数”混用。

## 8. 正式评价与结论规则

### 8.1 指标

主指标是 `group_mAP@1.0`，同时报告 `group_mAP@0.5`、`outlier_mIoU` 和 per-class AP。

Group IoU 比较成员集合，不是预测热图的像素 IoU：

```text
GroupIoU(G, G_pred) = |G intersection G_pred| / |G union G_pred|
```

1.0 要求成员集合一致；Group mAP 还结合 activity 分类分数。Outlier mIoU 比较每个 clip 的真实与预测 outlier 集合。定义以项目实际 evaluator 和 CAFE 论文为准，差异要报告。

补充诊断包括成员 precision/recall/F1、individual precision/recall、成员与非成员 ROI response 分布、错误类型、预测群组数量、重复群组率、no-group/空图比例、每类 AP、训练显存和吞吐。

只有指标的官方导出协议明确使用某种 confidence 时才沿用其定义。不要把热图整体均值当成新的 detection confidence 后仍声称只改了 grouping。

### 8.2 可比较性

- 固定同一 place/view protocol、split 清单、预处理、增强和 seed 列表。
- 完整比较至少三个独立 seeds，给出每个 seed、均值、标准差和配对差值。
- 使用相同训练预算、学习率策略、有效 batch 和 checkpoint selection rule。
- 同一个 seed 的共同模型初始化可核对；若 warm start，则各对照采用同一来源及相同追加训练预算。
- checkpoint 只按预先声明的 validation 指标选取，不能为不同指标各挑一份最好结果后拼成一行。
- test 不用于选阈值、超参数或 checkpoint；项目划分若没有单独 test，明确最终报告使用的 validation protocol，不冒充独立测试集。
- 阈值需要校准时，用相同搜索预算与协议校准所有相关对照；同时报告固定阈值结果。归一化不同必须标明。

三个 seeds 的正均值与标准差不是统计显著性的证明。先报告效应、方差、每类变化和失败样例，再决定是否追加 seeds。不得只保留最好的 seed。

### 8.3 正负结论

- 热图更好看但 Group mAP 未改善：只能说明可视化/空间拟合变化，不能宣称 grouping 提升。
- 只有 H1 改善：支持空间辅助监督，尚不能支持直接读取 membership 更优。
- H2 改善：仍需排除 matcher、归一化、阈值和训练预算差异。
- H3-P/H3-A 改善：分别判断区域 pooling 与 alignment 的贡献。
- 仅 SAM 版本改善：说明目标来源重要，不能归因于纯 actor-track 方法。
- 结果稳定不改善：保留负结果及诊断，提出有证据的后续方向，不改写评价定义来制造收益。

## 9. 常见失败的诊断顺序

| 现象 | 优先检查 | 下一步 |
|---|---|---|
| 图与人物整体错位 | resize/crop/flip、xyxy、spatial_scale、aligned | 修几何后重跑 oracle，暂停完整训练 |
| 热图全空或全满 | logits 初始化、正负权重、empty 比例、focal 软标签、梯度 | 看分项 loss 和梯度，做 BCE 诊断 |
| Oracle 都分不开近邻 | 分辨率、框重叠、Gaussian/pooling 尺度 | 单独比较目标/采样；必要时做 stride-16 分支 |
| 热图好看但成员差 | response 标定、group softmax、阈值、valid actor | 画响应分布，不先换 backbone |
| 图变化但 membership 不变 | embedding fallback、detach、缓存、decoder 用错字段 | 做固定输入依赖测试 |
| H3 对 mask 不敏感 | z 尺度、投影、q 路径主导、alignment 梯度 | 做 query-only 与替换 mask 诊断 |
| outlier 恶化 | 归一化变化、no-group 顺序、阈值、虚假 slot 响应 | 比较 H2-S/H2-I 及对应控制 |
| 多 slots 重复同一组 | cost 尺度、共同 assignment、empty/no-group 权重 | 查看匹配与每 slot 输出 |
| seed 方差大 | 数据顺序、初始化、matching 波动、样本量 | 保存配对结果，不挑 seed |
| 严格 mAP 不变而宽松指标提高 | 差一个成员、错误纳入 individual、活动类别 | 做集合级错误统计 |

## 10. 后续消融只按证据展开

优先选当前错误分析对应的一项，不同时改变多个因素：

- 目标：Gaussian、box core、SAM member-mask union。
- slots：4、6、12，保持其余配置和解码协议一致。
- 空间分辨率：现有特征与 stride-16/layer3/FPN；它属于额外模型改变。
- membership：heatmap-only、embedding-only、heatmap + appearance residual。
- alignment：无、cosine、带负样本的对比损失。
- temporal：无、track-response 一致性，不使用像素静止约束。
- outlier：现有 threshold 与 explicit background，单独比较。

SAM union 需要核验缓存帧号、actor ID、ROI 到全图映射、mask 尺寸和采样策略。缓存无法对应时先修复或报告覆盖率，不能默默把缺失 mask 当空 GT。

## 11. 实验产物与可复现记录

遵循项目已有目录规范；没有规范时可以采用以下结构。名称为建议，不要求创建第二套冲突目录。

```text
docs/heatmap/
  baseline_audit.md
  implementation_contract.md
  decision_log.md
  research_report.md
configs/heatmap/
  b0.* / h1_sup.* / h1_match.* / h2_s.* / h3_p.* / h3_a.*
tests_or_diagnostics/heatmap/
  coordinate / target / pooling / gradient / oracle / export checks
outputs/heatmap/<protocol>/<variant>/seed_<seed>/<run_id>/
  resolved_config.*
  command.txt
  code_revision.txt
  environment.txt
  train.log
  metrics.json
  prediction_export.*
  visualizations/
  checkpoints/
results/heatmap/
  run_registry.csv
  comparison.csv
  commands.md
```

所有运行必须保存展开后的真实配置及完整启动命令。先依据实际 CLI/配置系统生成命令，不在报告中留不存在的脚本名或不可运行的占位命令。

`run_registry` 至少记录：

```text
run_id, variant, protocol, split_hash, data_manifest_hash,
code_revision, dirty_diff_hash, config_hash, evaluator_hash,
seed, init_source, init_hash, membership_normalizer,
target_type, mask_resolution, K, T, threshold_protocol,
epochs_or_steps, effective_batch, precision,
checkpoint_path, checkpoint_selection_rule,
group_map_1_0, group_map_0_5, outlier_miou,
wall_time, peak_memory, status, failure_reason
```

尚未得到的指标写 null/未运行，不用 0 代替。失败、取消、预算用尽的 runs 仍保留在 registry 中。

可视化使用固定 clip 清单：原图与 actor ID、GT 成员、每个预测 slot 的连续热图、匹配关系、ROI response、最终成员及 individual、activity。包括成功与失败样例，不能只挑最好看的几张。

## 12. 首次执行的交付要求

收到本文件并被用户要求执行后，先完成以下最小闭环：

1. 项目 audit、B0 状态核验和真实数据流说明。
2. 可运行 Gaussian target、ROI/time pooling、oracle 与必要的坐标检查。
3. 可开关的独立 mask head，以及 H1/H2-S 的训练和评估接线。
4. 梯度、GT 隔离、有效框、关闭新功能和输出兼容性验证。
5. 有可用算力时完成 smoke、固定小集 overfit 和预算内的 paired pilot。
6. 输出实际结果、失败诊断、下一阶段配置、准确启动命令和资源估计。

上述步骤在资源缺失时分开标注：代码已实现、技术验证已通过、短实验已完成、完整实验未运行。不能把“代码能跑”写成“方法有效”。算力不可用时继续完成 CPU 可做的合成测试、配置和评估导出检查，不只留下等待说明。

完整研究报告至少回答：

- B0 的定义和复现情况是什么？
- oracle 揭示了哪些目标/分辨率限制？
- H1、H2、H3 各自改变了什么，哪些比较实际已完成？
- 严格 Group mAP、outlier 与 per-class AP 如何变化？
- 热图是否通过成员和活动路径实际影响预测？
- 哪些失败来自几何、优化、解码或表示本身？
- 现有证据支持什么结论，尚不支持什么结论？

## 13. 可直接发给服务器 Codex 的启动消息

> 请阅读本项目中的 `CAFE_HEATMAP_RESEARCH_EXECUTION_GUIDE.md`，按其中的分阶段协议执行 CAFE group heatmap 研究。先核验当前代码、B0 配置、数据划分、membership 归一化和评价流程，再完成 target/ROI oracle、可开关的 mask head、H1/H2-S 接线及必要验证。有可用的已分配算力时，继续完成有界 smoke、overfit 和 paired pilot，不要只给计划。首轮保持已确认 B0 的 backbone、帧数、slots、activity 路径和训练协议。H3 pooling/alignment、独立 sigmoid membership、SAM、FPN 和 explicit background 分开验证。保留已有代码和结果，保存真实命令、配置、日志、预测和失败记录。完整训练在项目既有预算内推进；预算不明确时，先交付最小闭环与明确的下一阶段算力需求。

## 14. 来源与适用边界

- 用户提供的 `CAFE_GROUP_HEATMAP_INSTANCE_ALIGNMENT_PLAN.md`：提供服务器项目分析、候选代码位置与初始方法设想。其路径、数据统计和缓存状态必须在实际服务器重新核实。
- 本次讨论确认的新版框架：H2 从群组热图读取 membership；H3 加入预测区域 pooling 与 cosine alignment；appearance residual 独立消融。
- [CAFE 论文](https://arxiv.org/html/2312.02878v2#S3.SS4)：Group IoU、Group mAP 和 Outlier mIoU 定义。
- [CAFE 官方代码](https://github.com/dk-kim/CAFE_codebase)：用于核对原始实现和评价协议；不代表当前项目与该仓库版本相同。
- [Torchvision ROIAlign 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html)：坐标缩放与采样参数。执行时以服务器已安装版本为准，不为匹配最新文档擅自升级环境。
- [PyTorch BCEWithLogitsLoss 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)：logits 与概率的区分及数值稳定计算；本文的 soft-target focal 是明确选定的研究定义。

本文件没有访问服务器代码，也没有执行研究训练。它规定的是可核验的实施和比较方式，不预设 heatmap 一定优于现有模型。
