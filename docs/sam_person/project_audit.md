# SAM 人物 Mask：项目审计（2026-10-18）

本轮操作根目录为 `/mzy/copyfrom_aixi_gadfortest`。起始 revision 为 `33ca78c`。
未发现适用于 `/mzy` 或项目的 AGENTS.md。原有未提交文件和 diff 保留，初始快照见
`outputs/sam_person/20261018_v1/user_initial*`；另一个 b0-repro checkout 的已有 SAM 改动未修改。

| 项目 | 核验结果 |
|---|---|
| B0 | 开发基线 `research_b0_repro=true`、`heatmap_variant=off`，`models/b0_repro.py`；不是 main 默认 DINO/HOI 模型，也不是官方 view→TEST 完整复现 |
| 数据 | CAFE split-by-place TRAIN→VAL；既有审计清单保留 6,444 / 1,386 clips。当前有界清单通过原解析器重新读取所选 ann，并对照既有清单核对 SHA256 |
| 划分 | TRAIN videos 1–4,9–12,17–24；VAL 13–16。不读真实TEST5–8的图像或逐clip ann。统一pickle及原GT文本物理含全数据，只索引/筛选所选TRAIN/VAL；不做TEST预测/评价/调参 |
| 重复 | 全 split clip key 交集和本轮源图内容 hash 交集检查；全数据跨 split 图像内容重复尚未穷举检查，不宣称全量去重 |
| 图像 | PIL 打开原 RGB，`video/clip/images/frames_<fid>.jpg`。模型 resize 至 720×1280，ToTensor、ImageNet normalize |
| 帧号 | 真实 CafeDataset segment centers，floor 后乘 ann.framesEach。fid 是 clip 内文件名坐标；无可靠全视频绝对帧号映射，缓存不猜测该映射 |
| actor | gt_tracks 的 ID=row+1，row 为该 clip annotation 行；不能直接用原 annotation ID。UID 显式包含 video/clip/row。individual 保留，padding action=7；缺帧按原有效性处理 |
| 框 | 给定 GT actor tracklets，normalized XYXY 经真实 loader 转 cxcywh，SAM 前还原原图 pixel XYXY；连续边界，无额外 +1。标注 visible/amodal 语义未独立确认。不是自动检测实验 |
| 增强 | 当前真实 B0 仅 Resize，无 crop/flip。mask 用 nearest 同步 resize，再 area 投影到特征网格；投影后不硬阈值。额外 crop/flip 复用接口仅作合成测试 |
| backbone | ResNet18 ImageNet、T=5、N=14、layer4 channels=512；720×1280 对应实际 23×40，不能用整数 stride 猜 22×40 |
| RoI | 原框乘实际 (40,23)，torchvision RoIAlign output=5×5，aligned=True、sampling_ratio=1、spatial_scale=1；保留 B0 的 longcw 越界整点零外推补偿 |
| actor | 原 512×5×5 flatten→fc_emb(256)、ReLU、dropout；S2F 只在 flatten 之前乘同网格 mask，无新参数、投影、归一化、loss |
| GAD | K=12、6 层/4 heads/FFN512、逐帧 group query、softmax_K membership、原 Hungarian、activity 路径和 no-group 顺序保持 |
| 导出/评价 | `util/group_export.py` 和 `evaluation/cafe_eval.py` 原实现；argmax slot 后 membership>.5；confidence 为 activity softmax，Group IoU 为成员集合 IoU |
| 训练 | FP32，Adam(.9,.999,eps=1e-8)、weight_decay=0。pilot 同一既有 B0 checkpoint warm start、同 seed、固定同一采帧和样本顺序；batch4 不拆 microbatch。此处 batch/有界时长与原完整 batch16 训练不同，不能与完整历史结果直接比较 |
| 基线局限 | 原 consistency 对 batch 组成/顺序敏感，保留；paired 两侧 batch 和顺序一致。开发 checkpoint 曾用 VAL 选模，本轮只报告开发子集诊断 |
| SAM | 已验证原版 SAM ViT-B，非 SAM2。项目资产未找到已核验能用的 SAM2 配置/权重对；不升级 GAD 环境、不下载其他模型 |
| SAM 来源 | 已安装 archive 的 direct_url 指向官方 commit `dca509fe793f601edb92606367a655c15ac00fdf`；配置记录此来源等级并锁定全部 Python 源码内容 hash |
| 权重 | `/mzy/cafe_runs/sam-probe/checkpoints/sam_vit_b_01ec64.pth`，SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| API | 原版 `SamPredictor`：同帧 set_image 一次，逐 actor predict 原图像素 box、单候选、无 points；显式检查最终 (1,H,W)，不用第三个低分辨率返回值 |
| 环境 | GAD Python3.8.20/PyTorch1.7.1/torchvision0.8.2，项目已有 numpy1.21.2/scipy1.7.1/Pillow8.4.0 overlay；SAM 独立已有 Python3.9.25/PyTorch1.13.0+cu117/torchvision0.14.0，无依赖安装 |
| 资源 | 8×RTX3090 24GB，驱动535.86.05。用户明确允许物理 GPU4/5/6/7。生成≤1 GPU小时、训练所有配置合计≤2 GPU小时（含失败），不开展完整训练 |
| 磁盘 | 初始 `/mzy` 约45GB空闲；所有新产物独立命名，不覆盖旧 run/cache/checkpoint |

现有旧 SAM 缓存只有 box crop 后 5×5 uint8 权重，不能无损还原原图 mask，也不证明与
当前实际 RoIAlign 公共网格等价。因此不作为本次 raw mask cache，不将旧 VAL alpha sweep
解释为成对训练收益。

首次全量目录扫描在共享盘 readdir 上长时间等待，保留日志后只终止本任务自己的只读进程。
有界 adapter 增加可选 clip_ids，默认原读取行为保持；实际解析、采帧、box 和有效性仍调用
原 dataloader。完整 split sampling manifest 及所选 ann 内容 fingerprint 都锁定。

SAM API 语义已对照[官方原版 predictor](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/predictor.py)；
SAM2 环境要求仅作后续参考，见[官方说明](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)。
实际实验成绩、未运行项和证据边界以 report.md 及机器可读日志为准。

补充核验（同日）：原 evaluator 的 make_groups 将 group score 存成单元素 set，
group_mAP_eval 直接用该 set 作为排序 key。CPU复核保持预测行完全相同、仅反转clip顺序，
80步B0的mAP@1.0从15.0463变成15.9722，S2F从7.4074变成4.6296；outlier mIoU不变。
证据见 `outputs/sam_person/20261018_v1/evaluator_order_probe.json` 及该目录独立复核脚本。
本轮没有修改 evaluator，SHA256仍为8d7f5d5f2bebaa911e30199036b20fed1cc9d1c31bfebb454b68f1901a50f181。
正式paired导出顺序一致，所有主表数字均为原evaluator结果；另报成员集合诊断。
该实现风险应在后续正式实验前独立核验，不能将修正评价器的变化归因于SAM。
