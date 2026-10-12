# GPU恢复后的双卡有界续跑 — 2026-10-12

用户告知服务器已恢复，但GPU0仍未修好，要求优先双卡训练。
17:44的宿主机只读检查确认旧PID767765已消失，物理GPU1/2/3空闲。
显式按UUID选物理GPU2/3，映射到逻辑cuda:0/1，不使用GPU0或GPU1。
实际CUDA初始化、两卡DataParallel前向、反向和同步已通过；不只依据nvidia-smi判断可用。
没有由本代理执行重置、重启或风扇/功率设置变更。

## 实际启动确认

本轮持久工作流于17:50:40 +08:00启动，PID9204，执行会话95828。
恢复后的19项合同测试全部通过（7.955秒），B0在17:52:26已实际完成41/500步，
loss/gradient均有限，GPU2/3温度65/62°C，driver显存22,375/21,533MiB。
以上是该时刻的观察，不预填后续完成状态；pilot尚待B0完成后自动开始。
会话结束聊天时不主动终止该工作流。

## 本轮运行范围

1. 恢复后的两卡微型前向/反向与19项合同测试；失败即停止。
2. 从共同初始化重跑B0固定16 TRAIN clips、500步overfit；每100步保存checkpoint与训练指标。
   不将旧step200权重加载冒充optimizer/LR/RNG完整resume，旧故障目录保留。
3. 同一对卡顺序运行B0/H1-Sup/H1-Match/H2-S，各500 optimizer steps，
   seed1、64 TRAIN/32诊断VAL clips、batch16、micro_batch0、FP32、Adam。
4. 四组全部完成后检查共同初始化、CPU/CUDA RNG、源码、数据、逐样本/帧顺序、LR、步数与batch；
   保存独立summary及固定样本可视化。任何训练/配对失败都会使工作流报错，不自动扩预算。

B0 overfit最长30分钟；四配置pilot合计最长120分钟，另加短预检和汇总。
只有一个双卡训练任务并发，不启动full，不开展H3/sigmoid membership/SAM/FPN/background。
保留ResNet18、T5、K12、activity路径、B0原始训练与评价协议。
pilot是有界优化诊断，不把单seed短VAL分数解释为最终方法收益。

先前H2-S 500步overfit及梯度诊断直接保留，不重复训练。
新B0与旧H2有工具/环境快照差异，不能宣称完整源码一致的500步配对；
正式的本轮严格配对以四配置新pilot和其独立paired_checks.json为准。

## 启动与日志

```bash
bash scripts/run_heatmap_dual_gpu_bounded.sh \
  --root outputs/heatmap/place/dual_gpu_recovery_20261012_1752
```

通过持久执行会话运行以上工作流，不依赖立即退出的启动shell；实际PID和启动时间写入workflow.log。
新目录：`outputs/heatmap/place/dual_gpu_recovery_20261012_1752/`。
第一次nohup后台尝试`dual_gpu_recovery_20261012_1748`的PID8810未存活，
未执行预检/optimizer step。该目录及launcher.log保留，不将它记录为已启动训练。

- `workflow.log`：完整工作流、展开命令、CUDA UUID、预检和训练输出。
- `contract_tests.log`：本轮恢复后测试结果。
- `b0_overfit500/train.log`：B0 overfit实时日志。
- `paired_pilot500/{b0,h1_sup,h1_match,h2_s}/train.log`：各pilot配置日志，按顺序创建。
- `summary/paired_checks.json`：四组真正完成后才生成的配对证据。
- `workflow_status.txt`：退出后记录stage/exit_code/时间；不存在时需结合PID/日志判断，不能仅凭缺失认定运行健康。
- `handoff_status.json`：本次交接时的真实观察快照，不是后续完成结果。

各run独立保留command.txt、配置、source_snapshot、环境、数据清单、checkpoint、预测及失败记录。
工作流和pilot shell脚本在启动时也复制到本轮目录。
新工作流使用项目内双卡锁避免重复启动；registry保留历史行，不覆盖旧paired报告。

```bash
tail -f outputs/heatmap/place/dual_gpu_recovery_20261012_1752/workflow.log
```
