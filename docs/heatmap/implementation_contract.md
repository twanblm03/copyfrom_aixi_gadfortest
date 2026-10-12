# Heatmap implementation contract

本轮只实现 B0、H1-Sup、H1-Match、H2-S。`configs/heatmap/*.json` 保存完整可运行配置；
`models.build_model` 显式选择 research B0。train.py/test.py 也接入相同的模型/criterion和有效性。
使用标准入口时必须同时传 `--research_b0_repro --no_mae` 及完整 ResNet B0 超参数。
推荐使用研究 runner 的配置入口，以免 main 的不同默认设置混入。

| 配置 | membership | assignment | 附加训练项 |
|---|---|---|---|
| off/B0 | embedding dot + K-softmax | 原 activity+L2 | 无 |
| H1-Sup | 同B0 | 同B0 | focal+Dice+unmatched empty |
| H1-Match | 同B0 | 原cost+focal+Dice | 同H1-Sup |
| H2-S | predicted mask ROI → positive alpha → K-softmax | activity+热图membership L2+mask cost | 同H1-Sup |

Mask head 是 Conv1×1 pixel embedding 与 MLP(group query) 的缩放点积加固定初始负 bias=-2，
64 channels，每 slot 独立 sigmoid occupancy；无 spatial/slot mask softmax。
head 在完整 B0 初始化之后、CPU fork_rng 作用域内创建，seed=B0 seed+104729，不推进共同 RNG。
activity 路径不读取新增 mask；H2-S 没有 appearance fallback，也没有无效 shared beta。
原 embedding heads 保留在 checkpoint 中，但 H2 membership loss 不经过它们。

目标：变换后 normalized track boxes → 同一23×40网格；高斯使用连续中心，网格采样点 x+.5/y+.5；
sigma=box_size*ratio，clamp[.5,4] grid units；同组成员逐像素 max，不是 hull/box/中心点。
训练 oracle 小候选集 ratio=.15/.25，alpha=5/10/20/40；最终按 train-only H2-S F1 选择 .15/20。
GT group/frame validity 显式表示；该帧无有效成员则不产生 matched mask loss。

ROI：预测连续 sigmoid map，ROIAlign output3×3、sampling_ratio=2、aligned=True、spatial_scale=1；
先把 normalized cxcywh 换为 grid xyxy 一次，并截取到图像边界。逐 ROI 均值后对有效 track 帧平均。
alpha=softplus(raw_alpha)+1e-6，初值20；membership=softmax_K(alpha*mean_response)。
padding、missing、退化和不相交框不参与新空间监督；H2 membership cost/loss 排除全帧不可用 actor。
H1 保留原 embedding membership 的标签语义，使辅助空间监督与替换 grouping 分开。

所有分支只用一次 clip-level Hungarian assignment。mask cost 权重 focal=1、Dice=1，与
loss 权重 focal=1、Dice=1、empty=.1 分开配置。soft focal 是指南的 soft-label 扩展，
gamma=2、alpha=.25；gamma0/alpha disabled 的 BCE+Dice 配置独立保留，未当作主线结果。
逐有效帧平均，再对有效 matched groups 平均；unmatched empty BCE 独立归一化。
原 activity/member/individual/consistency 系数仍为1/5/1/2。

预测 API 为 `forward(images, boxes, dummy_mask, mae_feats=None, actor_frame_valid=None)`。
不接收 GT members/activity/heatmaps；targets 只在 criterion/probe/oracle/诊断中使用。
原 activity logits、membership 和 actor_embeddings 的 keys/shape 保留；附加
group_mask_logits/group_masks [B,K,T,23,40]、spatial_response [B,K,T,N]、actor_frame_valid [B,T,N]。
membership_logits 仅 H2-S 输出。attention probe 是最后 decoder layer 的 native head average，
保留 spatial softmax，不称为 occupancy。

H3-P（仅 predicted-region pooling）、H3-A（再加 cosine alignment）尚未实现/训练；
H2-I/B0-I、SAM、FPN/stride16、explicit background、appearance residual 均独立后续，当前无隐式开关。
H3 梯度/因果检查须在相应阶段实现后单独通过，本轮不声称已验证。

各 run 采用独立目录、真实命令、展开配置/runtime、环境、revision/diff、源文件快照/hash、
数据与split hash、sample/frame顺序、init hash、loss/grad/cost尺度、checkpoint、预测、指标与失败文件。
历史运行不复写。初期微批 smoke 独立标为工程诊断；paired/full 使用 micro_batch=0。
完整训练入口会按 VAL mAP@1.0 保存/加载 best_val 后报告三指标，绝不拼接不同 epoch 的最好结果。

## Runner boundary safeguards

`tools/run_heatmap_research.py` freezes the first-round ResNet18/T5/K12/FP32/Adam
configuration. Full runs require unsplit batch16 and all VAL clips, record all
TRAIN clip identities, and restore the single best-VAL mAP@1.0 checkpoint before
the final export. Bounded runs count learning-rate epochs using full TRAIN
steps (402 at batch16), not subset passes. Runs below 402 steps keep initial LR.
These final safeguards do not retroactively change saved source snapshots.
