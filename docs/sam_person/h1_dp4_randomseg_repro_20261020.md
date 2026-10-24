# 四卡 DataParallel 复现历史 H1-Sup（2026-10-20）

目的：检验历史 H1-Sup 相对 B0 的优势在随机分段 TRAIN 采帧、历史完整 batch 热图损失求均值下能否复现。B0 与 H1-Sup 配对运行，各自从 ImageNet ResNet18 初始化，均使用四张 GPU 的 `HostStagedDataParallel`，全局 batch16、每卡 4 个 clip、30 epoch。VAL 均使用固定分段中心帧的完整 1,386 个 clip，按修正后的 group mAP@1.0 选模。

历史两卡 DataParallel 在模型汇集完整 batch16 输出之后调用 criterion。本轮四卡同样先汇集 batch16 输出，再调用完全相同的 `heatmap_losses`；`models/heatmap.py` 与历史实现 SHA256 均为 `8374dc0a22285ed707b53a3df0f569878236df097d7abb5d76a9cf84dad699cb`。`util/host_data_parallel.py` 也与历史实现 SHA256 相同，为 `12981f5c9039d322ac564a8d923bd9125bdc878bd7c56480a6d5cf97090fa908`。热图 focal/dice 对有效 matched group 求完整 batch 均值；empty 对有未匹配 slot 且有有效帧的 clip 求完整 batch 均值。此协议不启用 DDP 的每 rank 局部损失归一化。

与历史高分运行相同：TRAIN 每段随机一帧、VAL 每段中心帧、place split、seed1、ResNet18、T=5、GT actor tracks、batch16、Adam、初始/峰值 LR 1e-5/1e-4、5 epoch warmup、无梯度裁剪、H1-Sup sigma ratio 0.15。与历史仍不同：四卡而非两卡，每卡训练 batch4 而非8，故普通 BatchNorm 的本地统计可能不同；本轮 PyTorch 及评价器也与历史运行不同。B0 与 H1-Sup 在本轮共享这些设置，组内差值可用于判断相对优势是否复现。

先前的四卡 DDP 随机采帧 B0 与 H1-Sup 已按用户要求停止，断点保留在 `.local_data/historical_lr_randomsegments/`。这两项均未完成 30 epoch，不纳入本轮主对照。一次四卡 DDP 全局损失试启动也在数据准备阶段停止；本轮采用用户指定的 DataParallel。

两步 H1-Sup DataParallel 短验证通过：完整 batch16 的前向、反向及全部热图 loss 有限，日志显示 `Using host DataParallel on 4 GPUs`，GPU0 PyTorch 峰值 allocated/reserved 为 9,599/15,402 MiB。第一次短验证只因 `epochs=1` 与 `warmup_epochs=5` 的预检矛盾而未执行训练步；修正为 warmup1 后验证通过。完整训练恢复 30 epoch、warmup5。

B0 于 21:51:47、H1-Sup 于 21:52:05（北京时间）在独立会话启动，wrapper PID 分别为 2822423、2822567；输出标签分别为 `b0_randomseg_dp4_lr1_noclip_seed1`、`h1_sup_randomseg_dp4_lr1_noclip_seed1`，独立 checkpoint、日志和状态在 `.local_data/historical_lr_randomsegments_dp4/{b0,h1_sup}/`。两组训练全部完成后，后台汇总会检查各自完整 30 epoch、参数和 VAL clip 覆盖，输出 `results/sam_person/h1_dp4_randomseg_full.json` 及同名 `.md` 报告。

两组实际 `args.json` 已核对：`distributed=false`、`parallel_transport=host`、`fixed_center_train=false`、`random_sampling=false`、`batch=16`、`micro_batch=0`。除各自输出路径/标签、`heatmap_variant` 和对应 sigma ratio 外，配置字段完全相同。两项均进入第 1 轮训练步，未见异常。双任务并行时四卡显存占用约 35.5/29.9/29.9/29.9 GiB，分别尚有约 13.0/18.7/18.7/18.7 GiB 空闲。

监控命令：

```bash
cd /data2/mzy/GAD/CAFE_codebase/.worktrees/cafe-sam-person-mask-20261018
python - <<'PY'
from pathlib import Path
import re
root = Path('.local_data/historical_lr_randomsegments_dp4')
for variant in ('b0', 'h1_sup'):
    path = root / variant
    print(variant, (path / 'full_training.status').read_text().strip())
    log = (path / 'full_training.log').read_text(errors='replace')
    progress = re.findall(r'Train progress epoch #(\d+): step (\d+)/(\d+)', log)
    print('latest progress:', progress[-1] if progress else 'preparing data')
comparison = root / 'comparison.status'
print('comparison', comparison.read_text().strip() if comparison.exists() else 'not started')
PY
nvidia-smi
```
