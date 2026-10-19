# H1-Sup 拥挤场景中心热图对照训练方案（2026-10-16）

## 研究问题

检验：在保持 H1-Sup 的 embedding membership、Hungarian matching、activity 路径和解码不变时，仅收紧拥挤区域的成员中心高斯目标，能否减少跨群组人物的成员误分，并保持完整 VAL 的群组检测成绩。

现有 H1-Sup 已使用每位有效组员的框中心生成高斯，组内逐点取最大值；它并非整框填充目标。当前目标的宽度由人物框宽高决定，`sigma_ratio=0.15`、`sigma_min=0.5`、`sigma_max=4`，输出网格为 23×40。此实验针对邻近人物的高斯尾部重叠，不改变推理时的 grouping 方法。

## 唯一实验变量

| 组别 | GT 热图目标 | 其他设置 |
|---|---|---|
| A：现有 H1-Sup | `sigma_a = clamp(0.15 * box_size_a * grid_size_a, 0.5, 4)` | 现有配置和源码快照 |
| B：邻近中心约束 H1-Sup | 先算 A 的 `sigma_a`；再令 `sigma'_a = max(0.5, min(sigma_a, 0.35 * d_nearest))` | 与 A 相同 |

`d_nearest` 是同帧所有**有效人物**中心在 23×40 网格坐标中的最近欧氏距离，不使用该邻居的 GT 群组标签。若没有其他有效人物，取 `+∞`，因此目标与 A 一致。每个成员仍生成二维高斯，同组逐点 `max`；有效帧、空图、focal、Dice、empty loss 的规则完全不变。`0.35` 在看 VAL 结果前固定；无下限约束时，相邻中心处的单峰响应约为 `exp[-1/(2×0.35²)]≈0.017`。实际下限 `0.5` 和离散网格会限制分离能力，不能承诺达到这个响应值。

代码实现应只增加一个 target 模式开关，在 `models/heatmap.py::gaussian_targets` 的 sigma 计算处接入；`prepare_targets` 从配置传入模式。预测头、loss 系数、matching cost、embedding membership、activity head、阈值和 evaluator 均不改。关闭新模式时，A 的 target 和损失应与现有实现逐元素一致。

## 训练与配对

1. 从现有 `configs/heatmap/h1_sup.json` 复制 B 配置，除 target 模式外，字段完全相同。A 的 seed1 完整运行已有严格配对记录；若设备、数据、评估器和训练传输协议一致，可以作为 A 的 seed1 锚点。任何一项关键条件不一致，则同机重训 A seed1。
2. 先做目标几何检查：在固定 TRAIN 样本上记录新模式实际收紧的比例、组员中心响应、异组或 individual 中心泄漏、目标总质量，以及中心重合和小框的极端样本。检查 23×40 网格下 B 确实改变拥挤样本的目标；若几乎没有改变，先修改实验定义并重新冻结，不直接开始完整训练。
3. 对 B 做 contract test、3-step smoke 和短 overfit，核验有限 loss/梯度、GT 不进入预测前向、关闭模式复现 A。短跑只作为技术检查，不拿其 mAP 选参数。
4. 固定 seeds `1,2,3`，A/B 同 seed 使用相同的 B0 与 mask head 初始权重、采样帧顺序、batch 顺序、LR 轨迹、双卡数量和通信方式。每次全新训练 30 epochs，batch16、FP32、Adam、5-epoch warmup、全 TRAIN=6,444 clips；每轮评估完整 VAL=1,386 clips。按各自 VAL `group_mAP@1.0` 选择单个 checkpoint；同一 checkpoint 报告其余指标。中断后若无法恢复 optimizer/LR/RNG/数据顺序，则该 run 不计入严格配对，重新跑或单列。
5. 不在实验中启用 H1-Match、H2 的热图 membership、额外 pooling、SAM、FPN 或阈值校准。B0 seed1 只作历史背景，不参与 A/B 的主要因果差值。

## 预先固定的评估

- **主要结果：** A/B 在完整 VAL 的 `group_mAP@1.0` 配对差值，报告每个 seed、均值与范围；同时报告 mAP@0.5、outlier mIoU 和逐类 AP，不以单 seed 的最高值代表效果。
- **拥挤诊断：** 仅用 GT 几何预先定义子集：五个采样帧中，任一有效的异组人物或 group–individual 人物对，其框 IoU≥0.3。对这一子集报告匹配后的成员 precision/recall/F1、成员集合 Group IoU、跨组误分率，以及完整 VAL 对应指标。另按中心距离 `<1`、`1–2`、`≥2` 个网格单位分桶，并分别报告样本数。
- **机制检查：** 对固定成功/失败样本保存人物 ID/框、GT 与预测图、原始 embedding membership、最终分组。比较 A/B 的异组人物中心响应与误分是否同时下降；如果目标更分离而 grouping 未改善，就不能把热图几何变化宣称为有效改进。
- **边界：** 两人中心映射到几乎同一网格位置时，单张空间图不能稳定区分二者；保留人物 embedding 分支正是为了覆盖这类情况。

## 计算预算与执行边界

已完成的 H1-Sup seed1 双 RTX A6000 运行约 11.75 小时墙钟、23.5 GPU 小时，包含 30 轮完整 VAL。若复用 A seed1，新增 B seed1 与 A/B seeds2、3 共 5 次完整训练，按同等环境估算约 59 小时墙钟、118 GPU 小时，另计几何检查、smoke 和诊断；若 A seed1 必须重训，则为 6 次完整训练，约 71 小时墙钟、141 GPU 小时。实际排程以设备和预热计时为准。

本文件是预注册方案；尚未修改研究分支代码或启动新训练。
