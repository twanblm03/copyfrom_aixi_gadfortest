# Baseline audit — 2026-10-12

已 fetch 并将 `main` 快进至 `0ffe57666cb972ac6dc2b9593df556045b727e44`。原工作区的
未跟踪代码、DINO 缓存、结果和 checkpoint 保留。研究基线来自
`exp/b0-repro = 9e6a83f5374f2d5a61a60ce3f44d17a2b55c75bc`。

## B0 身份与协议冲突

`main/BASELINE_PROTOCOL.md` 描述 DINOv2、8 帧、HOI、temporal encoder、VideoMAE、
AdamW/cyclic。它不是用户此前确认的 ResNet B0。研究代码通过显式
`research_b0_repro=true` 选择 `models/b0_repro.py` 和 `models/b0_transformer.py`，
这两个文件以该分支的实际源码为基础；main 原有模型仍由默认入口选择。

本轮冻结 ResNet18/ImageNet、720×1280、T=5、N=14、K=12、D=256、6 层/4 heads/FFN512、
5×5 actor ROI、distance threshold=.2、dropout=.1、frame-specific queries 和原 activity 路径。
Adam(.9,.999,eps=1e-8)、weight decay=0、有效 batch16、FP32、30 epochs；前5 epochs
LR 1e-5→1e-4，随后 linear decay。预定完整训练 seeds=1,2,3，TRAIN→VAL，按同一 epoch 的
Group mAP@1.0 选择 checkpoint。TEST 未加载或评价。

## 真实数据流

| 符号/位置 | 本轮实际 shape 与含义 |
|---|---|
| CafeDataset.images | [B,5,3,720,1280]，ImageNet normalize；训练分段随机采样、验证确定性中心 |
| targets.boxes | [5,14,4]，track 的 normalized cxcywh；只 Resize，无 crop/flip augmentation |
| gt_boxes | annotation key-frame pixel boxes 除原图尺寸；不是每帧 heatmap 的来源 |
| track ID | `_id-1` 对应 annotation 行，不是原始 annotation ID；999 clips的ID被重编号，详见下文 |
| padding | action=7，dummy_mask=true；individual action=6、membership=-1 |
| actor_frame_valid | 显式 track 命中且 finite、非退化、与图像相交；missing/sentinel/padding 均无效 |
| backbone layer4 | [B*5,512,23,40]，名义 stride32；纵向是 ceil(720/32)，不是强行22 |
| src=input_proj | [B,5,256,23,40]；normalized boxes 乘(40,23)，坐标换算一次 |
| actor ROI | [B*5*14,512,5,5]，aligned=True、sampling_ratio=1，并恢复旧算子的越界采样零外推 |
| group_query_emb | [12*5,256]，不是 main 的 [12,256] shared queries |
| hs | [1,B*5,14+12,256]；actor 后面是 group |
| feature_att | [1,B*5,26,920]，最后 decoder layer、PyTorch 原生 head 平均 spatial softmax |
| group_hs | [B*5,12,256]；activity=逐帧 Linear 后 mean_t，首轮不改路径 |
| inst_repr/group_repr | time mean 后 L2 normalize；membership 经过两个 Linear 再点积 |
| membership | [B,12,14]，softmax(dim=1)，每个 actor 的12个slot概率和为1 |
| criterion.group_code | BCE(probabilities, member indicators)，不是 BCEWithLogits |
| baseline matcher | clip-level Hungarian；-activity probability + membership L2 distance |

## 数据与评价核验

实际 read_dataset(place,val_mode=True)：TRAIN 6,444 clips、VAL 1,386 clips；TRAIN 最多12 actors。
TRAIN videos=1,2,3,4,9,10,11,12,17,18,19,20,21,22,23,24；
VAL=13,14,15,16；TEST=5,6,7,8。源码把无已标注 group 的 clips 排除：这是 B0 自带过滤，
本轮不修改数据划分；空 GT 的技术行为另以合成测试验证。

轨迹 SHA256 `98ccd12267bb62fa27040f11876bab5f1d38c96f4544fe552ca6f5dcc55ff072`。
实际数据 manifest（annotation 内容哈希、clip IDs、帧/组/actor数量）和 split 保存在各 run；
manifest hash=`f428e7a08b774fd6f0c9e3ce7402fb6b01846f81902dbabb0166df95c9c0a212`，
split hash=`b0a1621b1731e154aa3c992c6b46ce7c15aecd2736583e58d0325e4a0d27bad8`。
训练小集使用排序后的 manifest 等间隔取样，覆盖多个 video；真实 actor/采样帧顺序逐 step 保存。

进一步核验：原始 TRAIN 6,709 / VAL 1,423 个 annotation，B0 分别排除265/37个；
TRAIN/VAL clip 交集为0。保留 clips 中999个的原始 annotation actor ID 不等于1..N，
但 gt_tracks 已按 annotation 的行顺序重新编号。对全部7,830个保留 clips 做 keyframe几何核对：
7,829个所有行 normalized xyxy 与 annotation 完全一致（绝对误差0）；
`(19,374)` 的frame204缺 actor row10（annotation共10人，track只有1..9）。
这是缺失轨迹，不是整体坐标偏移；新增有效性会排除该缺失行。
原 B0 的 missing-track 特征填充行为继续保留，空间监督不把它当作真实有效框。

原 decoder 先对12 slots argmax，再检查其 membership max>.5；若 slot activity 不是 no-group
而 membership 不足阈值，则导出 individual(group=-1,activity=7)。no-group 的过滤顺序未改。
输出每 actor 每帧10列：video clip frame x1 y1 x2 y2 group activity score。score 始终是该 slot
所预测 activity 的 softmax confidence，不使用 mask 均值。实际 evaluator 用 clip 第一张被导出帧
的 actor 行号作为集合元素；不是通过显式 actor ID 列匹配。至少2人构成计入 mAP 的 group。

评价源码未修改，SHA256=`8d7f5d5f2bebaa911e30199036b20fed1cc9d1c31bfebb454b68f1901a50f181`。
Group IoU 是成员集合交并比；报告原实现 mAP@1.0/.5、per-class AP、outlier mIoU。
原 evaluator 仅遍历 prediction clip keys，因此研究 runner 验证选定 clip 覆盖，并截取对应 GT。
双方都没有 outlier 的 clip 不进入原 outlier 平均；空平均标 null，不填0。

## 已知基线局限与状态

原 consistency 是 actor embedding 上的正/负成员对比，temperature=.2、coef=2。
原函数在逐 batch-item 累积过程中反复除 group_count，导致与 batch 顺序/拆分相关。
普通批次保留该行为，B0/H1/H2 不静默修正；无组/singleton/全padding/非连续有效 actor
才使用显式有限值 fallback。本轮 paired 使用未拆分 batch16，避免微批改变这一损失。

环境 Python3.8.5、PyTorch1.7.1、torchvision0.8.2、SciPy1.7.1；不升级依赖。
沙箱看不到 NVIDIA 设备；宿主继承 CUDA_VISIBLE_DEVICES=1,2,3,4，PyTorch 实际见3张3090，
逻辑0,1,2对应有效的物理1,2,3。使用已分配范围，不使用物理0或新建实例。

关闭 heatmap 后与分支原模型逐元素一致（rtol=0,atol=0）；H1/H2 与 B0 共同权重、
初始化后的 CPU/CUDA RNG 逐位一致；测试见 `tests/heatmap/test_contract.py`。
本地 `cafe_place.pth` 是已发布参考，缺训练 args/val-selection 元数据，不能认证为本轮 B0。
第一次严格加载因多出的 bbox_emb 六个旧参数失败；仅参考复评允许显式忽略该不在 B0
forward 中的头，保存完整兼容清单与失败记录。复评结果以研究报告中的实测为准。
已有 DINO Stage2B 结果含 batch4、cyclic/weight decay、mAP@.5选模等差异，不作为本轮对照。
