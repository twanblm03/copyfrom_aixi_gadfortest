# H1-Sup 拥挤约束：最小改动与 B-only 短测

2026-10-17用户已追加授权B seed1完整训练；本文件下述“暂不完整训练”为保留的历史准备边界。
此前短测因快照漏带参考C++而止于合同检查，没有B训练更新；新修复/短测/启动记录见
[B完整训练启动](h1_sup_crowding_full_launch_20261017.md)。

## 本轮边界

遵循用户最新选择：先准备代码和短测，暂不安排任何完整训练；不重跑 A、不排 seed2/3。后续若只做 B seed1 完整训练，只能作为探索性结果，与异机历史 A 作参考，不能声称严格配对因果收益。

本轮采用任务文件 `h1_center_crowding_control_20261016.md` 明确规定的 **place TRAIN→VAL 研究协议**（6444/1386、linear LR、weight decay=0）。这不是当前 view 官方参考对齐协议（TRAIN+VAL→TEST、CyclicLR、weight decay=1e-4）；两者不混用数据、checkpoint 或指标。

## 实现范围

- `models/heatmap.py` 增加 `heatmap_target_mode=box|nearest_center`，默认 `box`。B 严格使用预注册 `max(0.5,min(sigma_a,0.35*d_nearest))`，所有有效人物参与距离计算，不查询其群组标签；无邻居时不变。
- `configs/heatmap/h1_sup_crowding.json` 相比 `h1_sup.json` **只增加 target mode 一个字段**。
- 模型前向、embedding membership、Hungarian matcher、activity、loss 系数、阈值、evaluator、优化器与训练循环均不变。runner 仅添加模式检查和独立 run 标识 `h1_sup_crowding`，避免混入原 H1-Sup 汇总。
- 新增合同测试、TRAIN 几何检查工具与有界短测启动脚本；没有新增 full-training 队列。

## CPU 验证

输出目录：`outputs/heatmap/place/crowding_preparation_20261016/`。

`target_contract_cpu.log`：6 个新增测试通过。覆盖关闭模式与提交 `33ca78c` 的目标/损失逐元素一致；nearest-center 公式、无邻居、无效人物、中心重合、极小框、空组、人物排列、目标质量单调性和有限梯度；A/B 完整模型相同初始化/RNG、相同无 GT 前向和相同非 mask 损失。

`regression_cpu.log`：81 项测试，80 通过，1 个 CUDA 测试因本次 CPU-only 环境跳过。GPU 测试未在当前 view 任务占卡时执行。

`audit/`：实际 B 配置、source snapshot/manifest、真实命令、数据及 split 哈希；CPU 数据审计成功，TRAIN=6444、VAL=1386。不是一次训练结果。

## 固定 TRAIN64 几何结果

从排序 TRAIN 清单均匀抽取 64 clips，实际 loader + 固定五段中心帧；不加载 VAL/TEST、不训练、不搜索参数。原始人物 ID/框/有效性、两种目标及 sigma 保存在 `geometry_train64/targets/*.npz`；逐 clip CSV/JSON、源码快照、命令与汇总一并保留。

| 检查项 | A | B |
|---|---:|---:|
| 组员中心平均离散插值响应 | 0.754292 | 0.696181 |
| 异组人物中心平均响应 | 0.003406 | 0.002117 |
| individual 中心平均响应 | 0.007714 | 0.004746 |
| 目标总质量 | 6342.54 | 4900.70 |

1809 个有效组员帧中 1507 个（83.31%）至少一轴 sigma 收紧；63/64 clips 目标改变，20/20 个预定义 IoU 拥挤 clips 目标改变。总质量减少 22.73%，异组与 individual 中心响应各减少约 38%，同时组员中心响应减少约 7.70%。中心距离分桶（clip 内最小异组/group–individual 距离）为 `<1`:7、`1–2`:13、`≥2`:44；无完全重合有效人物对，11 个组员帧的 A 两轴均已在 sigma 下限。重合中心边界另由合成合同测试覆盖。

预先写入工具的几何工程门槛为：至少 1% 有效组员帧收紧，至少 2 个 IoU 拥挤 clips 改变。本次通过，未调整 0.35。**响应指 GT 目标，不是预测；几何门槛通过不代表 grouping 更好。83.3% 收紧说明此约束并不局限于少数高 IoU 拥挤样本。**

## B-only 有界 GPU 检查

启动器：`scripts/run_heatmap_crowding_checks.sh`，其冻结副本为输出目录下的 `bounded_workflow_script.sh`。

使用前台持久会话与现有 GPU2/3 项目锁。只有前一 view workflow 正常结束、epoch30/13680 更新完成、最终 TEST 指标/预测/所评 checkpoint 哈希均核验成功，且物理 GPU2/3 无其他进程，才启动 CUDA。前序失败、超时、几何门槛失败、源码漂移或外来占卡均停止，不抢占。

只运行：宿主合同测试 → B smoke 3 updates → B 固定 16 TRAIN clips overfit 100 updates（step0/50/100 TRAIN 诊断）。smoke/overfit 各从相同 seed1 初始化独立开始，不是连续训练；VAL 仅取固定 3 clips 做工程检查，短跑 AP 不用于选参数。前向和 loss 使用原始完整 batch16、双卡 DataParallel、FP32，无 microbatch。

日志：`bounded_workflow.log`、`contract_host.log`、`smoke3_b/train.log`、`overfit100_b/train.log`。退出状态：`bounded_workflow_status.txt`。每次 run 保留配置/命令/源码/初始化哈希/采样帧顺序/损失梯度/预测/checkpoint；失败也保留，不覆盖。**脚本没有 full 启动命令，短测通过也不会自动启动完整训练。**

2026-10-16 19:37 +08 已以前台持久会话 92489 启动等待，workflow PID 1890238；当时尚未执行任何 CUDA 短测。真实启动命令：

```bash
bash /home/zhenyu/Documents/CAFE_codebase-main/outputs/heatmap/place/crowding_preparation_20261016/bounded_workflow_script.sh \
  --repo /home/zhenyu/Documents/CAFE_codebase-main \
  --output-root /home/zhenyu/Documents/CAFE_codebase-main/outputs/heatmap/place/crowding_preparation_20261016 \
  --after-official-view /home/zhenyu/Documents/CAFE_codebase-main/outputs/heatmap/view/official_h1_sup_seed1_dualgpu23_20261016_1325
```
