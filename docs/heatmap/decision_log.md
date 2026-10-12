# Decisions and failures — 2026-10-12

GPU恢复后按用户要求只用物理2/3双卡，继续有界实验；具体预算、隔离方式和新日志见
[dual_gpu_recovery_20261012.md](dual_gpu_recovery_20261012.md)。以下编号保留此前历史决策。

1. 更新 main，保留 exp/b0-repro 和用户未跟踪文件。没有强制覆盖当前研究代码/结果。
2. 审计发现 main 的增强模型与此前确认的 ResNet B0 不同。显式移植冻结 B0 模型/Transformer，
   保留 main 默认模型入口；用关闭功能的 bit-exact 对比检查防止悄悄换基线。
3. Sandbox CUDA=false 并非实际无算力；提升为只读宿主机设备检查后，确认继承范围有3张空闲3090。
   仅用逻辑0/1/2，不新增实例，不抢占其他进程。
4. 首次 CPU contract tests 在旧 PyTorch 的整数 tensor.any 上3项报错。显式 bool 后通过；
   最终扩展为13项，已有8项 DINO helper regression 也通过。
5. 发布 checkpoint 严格加载失败（六个 bbox_emb keys），失败 run 保留。
   核查冻结 B0 无此参数/forward 路径后，以显式兼容选项仅复评发布参考；不认证为 val-selected B0。
6. 初始 prefix oracle 只覆盖一个 video，补充 manifest 等间隔固定子集覆盖全 split；
   两种实验均保留。主 oracle 用64 TRAIN/32独立VAL clips，不用 VAL 选择 sigma/alpha。
7. raw ROI threshold 与 K-softmax 的读取上限不同。补测含12 slots和零占据 unmatched slots的
   H2-S oracle；按训练集 F1 冻结 ratio=.15/alpha20。VAL 仅诊断，未逐clip调阈值。
8. 初始 oracle actor bucket 把双方空集合计为0；修正诊断为跳过该 undefined 集合IoU，
   并标明是 actor-restricted bucket。旧产物保留，后续仅解释修正后的 oracle。
9. 单GPU微批 smoke 完成，但微批会改变 BN 和 B0 原 consistency reduction，不能作正式对照。
   实测单clip峰值约2.5GB后采用3GPU DataParallel的未拆分batch16，H2 smoke通过、峰值约13.7GB。
10. 完整研究预算未给出。本轮冻结有界工程预算：3-step smoke；16 clips / 32-step overfit；
    64 TRAIN / 32 VAL / 8-step paired B0/H1-Sup/H1-Match/H2-S，seed1。
    这些短预算只判断技术健康，不能检验最终收益；失败不自动追加到500/1000步。
11. 原 consistency 的 batch 顺序相关归一化缺陷保留于普通批次；修正该缺陷须独立控制。
    只对空/无效输入增加一致的有限值 guard。没有引入 H3、sigmoid membership、SAM、FPN 或新decoder。
12. 配对审查发现 torch.manual_seed 在 CPU-only fork_rng 中仍会改 CUDA RNG。首轮 pilot
    判为不可作配对结论，保留在 `paired_pilot8_seed1_20261012` 与 invalidated_runs.json。
    改用 torch.default_generator.manual_seed 后，14项宿主机测试（含CPU/CUDA RNG）全部通过。
    重跑 `paired_pilot8_rngfixed_seed1_20261012`：共同初始化、CPU/CUDA RNG、源码快照、
    样本/采样帧、data/evaluator、batch/steps全部配对检查通过。
13. 32-step overfit 未恢复 group，保留负结果。8-step paired pilot用于判断早期 no-group
    是否是各对照共同现象；四者均 mAP=0，不作为正式收益实验。修复 RNG 后的 overfit另行重跑。
    当前没有理由因这组短结果直接推进 H3 或完整多seed矩阵。
14. 当前环境缺 matplotlib；采用已有 Pillow 生成固定12-slot诊断面板、原始响应JSON/SVG直方图，
    不升级训练依赖。main 的可选曲线绘图改为依赖缺失时发警告，指标JSON仍正常保存；
    train.py/test.py入口导入已检查。原 train.py禁用默认help，--help返回错误属于既有CLI设置。
15. 全量 actor ID/几何审计确认 gt_tracks 按 annotation 行号重编号，不能将原始 actor ID
    当作 tensor 行；一个 keyframe 缺轨迹 actor 已记录，无需改 split 或猜填 heatmap标签。
16. RNG修复后的32步overfit完成：train group mAP两种阈值均0，192/192 slots为no-group，
    member/nonmember ROI均值 .114490/.114199。individual CE明显下降但mask/member未分离，
    gate仍未通过；不是数值崩溃，也不是充分预算下的不可收敛证明。
17. 收尾发现full的selected_clips元数据错误地沿用短跑默认64，已改为实际全TRAIN；
    full强制全VAL/batch16及既定selection metric。短跑超过402steps时按完整B0 epoch调LR，
    不按小集pass调度。所有已执行短跑少于402steps，数值不变；不重写旧源码快照。
    最终15项宿主机测试和8项既有DINO测试通过，原始日志与失败记录全部保留。
