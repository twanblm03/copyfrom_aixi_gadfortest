# 四卡续跑 — 2026-10-12

用户允许继续并明确四张卡都可用。启动前四张RTX3090 24GB均空闲。
继承的 `CUDA_VISIBLE_DEVICES=1,2,3,4` 包含不存在的物理4；按新授权显式设置物理0/1、2/3。
并发上限2 jobs，每job两卡DataParallel、全局batch16、FP32，不使用microbatch。

## 当前有界阶段

固定同一16 TRAIN clips、seed1、确定性5帧；B0与H2-S均从相同ImageNet/B0初始化训练500steps。
所有其他B0协议不变，LR按完整TRAIN的402steps/epoch计数。最终checkpoint按固定末步，
训练集每100步记录原始指标/预测，不能据VAL挑epoch或阈值。
两卡分片的BN统计与上一轮三卡分片不同；两组新对照保持一致，不把它与旧8步结果作正式增益比较。

只对确定性overfit输入做CPU内存缓存：16clips的图像884,736,000bytes，像素、boxes、labels、
帧ID和内容hash保存；不缓存随机训练帧，不缓存GT替代预测，不改preprocessing。
训练集诊断放在CPU/CUDA RNG保护内，不推进训练RNG。并发registry改用文件锁+原子替换。

初次 `overfit500_pair_seed1_20261012/{b0,h2_s}` 均在step0诊断JSON日志因NumPy数组序列化失败，
没有optimizer step；失败目录和registry保留。统一JSON转换并加回归测试后，重启到
`overfit500_pair_retry_seed1_20261012/{b0,h2_s}`。17项宿主机合同测试全部通过。

本阶段预算为两组各500steps；没有把“允许四卡”视为无限GPU-hours或自动批准完整多seed矩阵。
用户进一步明确“先完成有界实验，再决定完整训练”，本轮不启动full。
如果H2-S仍未通过小集gate，先根据分项梯度、mask/member响应和B0控制做train-only诊断。
H3、独立sigmoid、SAM、FPN、explicit background不混入此轮。

## 实际命令

```bash
CUDA_VISIBLE_DEVICES=0,1 .miniconda3/envs/gad/bin/python tools/run_heatmap_research.py \
  --stage overfit --config configs/heatmap/b0.json --steps 500 \
  --train-clips 16 --eval-clips 32 --workers 2 --cache-fixed-overfit --train-eval-every 100 \
  --device cuda:0 --parallel-gpus 0,1 \
  --output outputs/heatmap/place/overfit500_pair_retry_seed1_20261012/b0

CUDA_VISIBLE_DEVICES=2,3 .miniconda3/envs/gad/bin/python tools/run_heatmap_research.py \
  --stage overfit --config configs/heatmap/h2_s.json --steps 500 \
  --train-clips 16 --eval-clips 32 --workers 2 --cache-fixed-overfit --train-eval-every 100 \
  --device cuda:0 --parallel-gpus 0,1 \
  --output outputs/heatmap/place/overfit500_pair_retry_seed1_20261012/h2_s
```

## 真实结果与交接（13:17 +08:00）

H2-S已完成500步，wall931.20秒。固定16 TRAIN clips上的成员TP98/FP5/FN0，
F1=97.5124%；原始训练集mAP@1.0=66.5185、mAP@.5=74.0741、outlier mIoU=94.1071。
匹配成员/非成员ROI均值为.421766/.031414，group class error=0。
这说明比32步诊断更长的训练确实形成了成员分离，但不是完美过拟合，也不是方法增益。
16 clips不含Queueing类，原六类macro mAP的该子集理论上限为83.3333。
独立32 VAL clips诊断mAP@1.0=0、mAP@.5=.8333、outlier mIoU=45.7314，不能宣称泛化收益。

B0最后落盘210/500步，200步检查点及原始预测已保存；GPU0异常后数分钟无新step。
13:16对核验命令完全匹配的本次PID767765发SIGINT，未响应后发SIGTERM。
13:17:16仍残留`[python]`进程，不能声称已退出或资源已释放。没有重置显卡或重启机器。
两组预算未完整配对，因此没有生成500步paired通过声明，也不能计算B0/H2增益。

H2-S的500步checkpoint train-only分项梯度诊断已完成，没有优化更新：
加权mask-head梯度范数member=.09275、focal=.16229、Dice=.55172、empty=.00635；
member与focal/Dice的cosine为-.8671/-.8141。这仅是固定checkpoint的局部梯度冲突，
不构成训练损害的因果证明，不据此改权重或按VAL调参。
同assignment的soft-BCE反事实只算导数，没有训练新变体。
initial梯度诊断遇到`NCCL Error 2`失败，保留独立失败目录。

500步四配置paired pilot在物理2/3尝试启动，首个B0在任何optimizer step前报
`CUDA unavailable in this execution environment`，脚本随即退出；H1-Sup/H1-Match/H2-S未开始。
实际命令：

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_heatmap_paired_pilot.sh \
  --steps 500 --seed 1 --root outputs/heatmap/place/paired_pilot500_seed1_healthy23_20261012 \
  --gpus 0,1 --workers 4 --persistent-workers
```

最新宿主机19项合同测试全部通过（在故障发生前），日志为
`outputs/heatmap/place/contract_tests_bounded_continuation_cuda_20261012.log`。
新增确定性缓存、NumPy日志序列化、分项梯度和诊断整合回归检查。
只为后续pilot提供可选persistent workers，不改变已启动overfit的数据或采样。
各run自己的源快照保留，旧paired证据未覆盖；本轮汇总单独写入
`results/heatmap/continuation_20261012/`。
完整训练尚未启动，下一步必须先恢复健康CUDA，再补齐B0和四配置有界paired pilot。

## 硬件状态与可比较性

12:57的只读 `nvidia-smi -q -i 0 -d PERFORMANCE,TEMPERATURE,POWER,CLOCK` 显示
GPU0 `SW Thermal Slowdown: Active`、SM285MHz、core78°C、power约150W，软件热限频。
没有改风扇/功率/时钟设置，不将该两组实际wall time解释为模型效率优劣。
GPU0的内存温度接口不提供读数，不能凭core温度推断具体热源。
两组数值/数据/预算可配对，但完整训练吞吐须在健康硬件上重新profiling。

随后13:12及13:15的全卡查询均报：
`Unable to determine the device handle for GPU0000:01:00.0: Unknown Error`。
13:16单独查询物理1/2/3仍可访问；物理2/3空闲，但新PyTorch进程CUDA初始化失败。
不能由NVML查询可用就断言这些卡可训练。内核日志读取被系统拒绝，未确认Xid或具体故障根因。
热限频与设备失联的先后关系已记录，不能据此断言因果。
