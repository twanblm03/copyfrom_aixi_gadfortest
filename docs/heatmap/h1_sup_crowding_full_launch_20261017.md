# B 拥挤约束 H1-Sup seed1 完整训练 — 2026-10-17

## 授权与实验身份

用户于2026-10-17明确授权“现在可以启动B 拥挤约束的训练”。本轮只运行
**B seed1 的 place TRAIN→VAL 完整30轮**；不重跑A，不排seed2/3，不启动view、B0或H2。
沿用用户任务文件 `h1_center_crowding_control_20261016.md`，不是刚完成的view逐轮TEST协议。

这是**单组探索性运行**。目前没有本机同硬件/传输/随机轨迹的H1-Sup A完整对照，
异机历史A只能参考，不声称严格配对因果收益。模型初始化仍用seed1/ImageNet+B0 heads，
不从view/place已训练checkpoint或smoke/overfit权重续训。

## 冻结配置

- `configs/heatmap/h1_sup_crowding.json` 与原`h1_sup.json`仅差一个
  `heatmap_target_mode=nearest_center`。H1-Sup embedding membership、原Hungarian matching、
  activity、解码/evaluator/阈值、热图loss系数均不改。
- 目标：先按`.15 * box size * grid size`得sigma并clamp[.5,4]，再按
  `max(.5,min(sigma,.35*d_nearest))`收紧。最近距离来自同帧**全部有效人物**，包含同组、
  异组和individual，邻居选择不查询群组标签；无邻居不变。组内仍逐像素max。
- TRAIN=6444、VAL=1386；每轮全VAL，按VAL mAP@1.0选择单个best_val并同checkpoint报告
  mAP@0.5/outlier mIoU。TEST不加载、不选模型。这里不是三视角view的TRAIN+VAL→TEST。
- seed1、ResNet18/T5/K12、720×1280、D256、batch16、FP32、无microbatch，双3090
  DataParallel，物理GPU2/3固定UUID（GPU0/1禁止）。workers4、non-persistent。
- 原研究协议Adam WD=0、warmup5/linear30，lr=1e-5/max_lr=1e-4；末轮lr=0不静默修正。
  不是view新版的WD=1e-4/CyclicLR。每轮402更新、有效6432clips、随机丢12clips；
  30轮共12060更新。每轮训练原分段随机采帧，VAL为原研究协议确定性段中心采帧。

## 重新验证与失败保留

此前`outputs/heatmap/place/crowding_preparation_20261016/`的队列于10月16日20:10:40
在合同检查停止：冻结快照漏带用于来源核验的参考`crop_and_resize.cpp`，**未执行B训练更新**。
原目录、失败日志和原快照保留。本次将通用快照扩展为包含`.cpp`，补上冻结来源/manifest测试；
这只是来源打包修复，不改变任何训练/预测运算。

新输出根目录：`outputs/heatmap/place/full_h1_sup_crowding_seed1_dualgpu23_20261017_1000/`。
目录时间仅为标签，实际时间以workflow.log为准。

- CPU回归：87项，86通过、1 CUDA跳过；包括关闭模式target/loss bitexact、GT不进入前向、
  A/B初始化/RNG/非mask损失一致、几何极端情形以及参考C++打包回归。
- 新`geometry_train64/`实际重做固定TRAIN64，未读VAL/TEST；结果与之前一致：
  1507/1809有效组员帧（83.31%）收紧，63/64clips目标改变，20/20个预定义IoU拥挤clips改变，
  几何门槛通过。参数`.35`未更改，未用VAL选择参数。
- 组员中心目标响应0.754292→0.696181，异组0.003406→0.002117，individual
  0.007714→0.004746；目标总质量为A的77.27%。**这是目标几何，不是预测收益**；
  83.31%收紧说明影响并非只局限于少数IoU拥挤样本。
- 新`audit/`验证6444/1386及全部数据/source/config/split/evaluator哈希，并冻结含C++的完整源码。

## GPU门槛与完整训练

脚本`scripts/run_heatmap_h1_crowding_full.sh`、冻结副本`workflow_script.sh`。
整个workflow持有现有GPU2/3项目flock；实际空闲检查通过才分配CUDA，不抢占外来任务。

顺序：宿主合同→B smoke3（固定32 TRAIN、3 VAL工程样本）→B overfit100
（固定16 TRAIN并缓存CPU输入，step0/50/100作TRAIN诊断，3 VAL仅技术检查）→通过则B完整30轮。
短测各自从相同初始化独立开始，不串接权重；full也从全新解释器初始化。

在看短测结果前预先固定技术门槛：全部loss/梯度有限、mask梯度始终非零，
100步最后10步的加权总loss均值与mask Dice loss均值均低于前10步；
common/full初始化、初始CPU+CUDA RNG及source/data/split/evaluator哈希在两段短测一致。
**不按短跑AP决定sigma、阈值、loss或选择训练配置。** 失败即停止并保留记录，不自动放宽。

```bash
bash /home/zhenyu/Documents/CAFE_codebase-main/outputs/heatmap/place/full_h1_sup_crowding_seed1_dualgpu23_20261017_1000/workflow_script.sh \
  --repo /home/zhenyu/Documents/CAFE_codebase-main \
  --output-root /home/zhenyu/Documents/CAFE_codebase-main/outputs/heatmap/place/full_h1_sup_crowding_seed1_dualgpu23_20261017_1000
```

以前台持久会话启动；`workflow.log`记录真实命令、阶段，`workflow_status.txt`记录退出状态。
短测日志分别为`smoke3/train.log`、`overfit100/train.log`，技术门槛证据为`bounded_checks.json`。
正式主日志`h1_sup_crowding_full/train.log`；每轮`metrics_epochN.json`/预测，最终保存
`best_val.pth`和`last.pth`，另保留真实采样顺序、初始化/来源哈希、配置、环境与失败记录。
普通训练runner不是精确恢复入口，本轮不将任意中断恢复伪称严格配对。

预计短测约5–8分钟、完整训练约7–9小时，需按实际吞吐和全VAL时间修正。
完成后另做预注册的IoU拥挤子集、中心距离分桶和成员误分机制分析；仅目标更分离或
单组分数变化都不足以宣称相对A的grouping收益。

## 启动过程快照

workflow于 **2026-10-17 09:58:30 +08** 以前台持久会话 **46639** 启动，workflow PID
**2244472**；固定源码manifest SHA256为`15869a5965e5bfad552666d4fdd31cfc8054d6a089519a7ac0465f86f7a85494`。
宿主87项合同全部通过、无跳过（33.59秒）。smoke3已正常完成（50.12秒），
损失/梯度有限、mask梯度非零、保存/重载权重及重复VAL预测hash一致。
common/full初始权重哈希分别为`2a562905…479b` / `16cdabdf…1380b`，初始RNG为`0c1efac6…476d`。
10:00进入overfit100；此段是短测进度记录，正式训练是否开始以full日志为准。

**已实际启动完整训练，不是仅排队。** overfit100正常结束（232.06秒）；前/后10步加权
总loss均值4.99643→2.57712，mask Dice loss均值0.973589→0.971478，loss/梯度有限、
mask梯度非零，source/data/split/evaluator及common/full初始化/RNG均与smoke一致。
TRAIN group mAP在0/50/100步仍为0，100步末所有slots为no-group；因此本轮只确认
计算/梯度/轻量拟合技术门槛通过，**不声称已充分过拟合成员集合，更不声称B有效**。
成员/非成员平均热图响应之差由0.000382增至0.001349，但这也不是最终解码收益。
未因短跑AP调整任何设置，保留固定协议进入用户已授权的完整训练。

10:03结束短测并进入full（主训练PID **2247148**）；**10:04:55 +08**核验到第1轮
step **24/12060**，所有已记录loss/梯度有限、mask梯度持续非零，完整配置比A仅多target
模式字段；full源快照、数据与split内容均与smoke逐字节一致，CLI checkpoint=None。
宿主GPU2/3温度67/61°C，driver显存22429/21727MiB；torch主卡峰值17817.57MiB。
上述是启动状态快照，后续以主日志与逐轮metrics为准；本轮尚无完整VAL成绩。
