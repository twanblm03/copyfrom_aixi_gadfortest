
```bash
/share/apps/anaconda3/envs/torch171/bin/python tools/sam_person_masks.py manifest --config configs/sam_person/qc_resolved.yaml --out outputs/sam_person/20261018_v1/qc_manifest.jsonl
```

```bash
/share/apps/anaconda3/envs/torch171/bin/python tools/sam_person_masks.py manifest --config configs/sam_person/qc_resolved.yaml --out outputs/sam_person/20261018_v1/qc_manifest.jsonl
```

```bash
/share/apps/anaconda3/bin/python tools/sam_person_masks.py generate --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/20261018_v1/smoke_manifest.jsonl --max-frames 8 --max-actor-prompts 100 --max-gpu-minutes 10
```

```bash
/share/apps/anaconda3/bin/python tools/sam_person_masks.py audit --config configs/sam_person/qc_resolved.yaml --out outputs/sam_person/20261018_v1/audit_snapshot.json
```

```bash
CUDA_VISIBLE_DEVICES='' /share/apps/anaconda3/bin/python tools/sam_person_masks.py visualize --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/20261018_v1/smoke_manifest.jsonl --out outputs/sam_person/20261018_v1/smoke_gallery
```

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 /share/apps/anaconda3/envs/torch171/bin/python tools/sam_person_masks.py qc --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/20261018_v1/smoke_manifest.jsonl --require-complete --out outputs/sam_person/20261018_v1/smoke_qc.json
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-d2b35225-e4f2-e4d6-e4e1-d289490a03ab OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /share/apps/anaconda3/bin/python tools/sam_person_masks.py generate --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/20261018_v1/qc_manifest.jsonl --resume --max-frames 200 --max-actor-prompts 3000 --max-gpu-minutes 60
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-d2b35225-e4f2-e4d6-e4e1-d289490a03ab OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /share/apps/anaconda3/bin/python tools/sam_person_masks.py generate --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/20261018_v1/qc_manifest.jsonl --resume --max-frames 200 --max-actor-prompts 3000 --max-gpu-minutes 60
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-d2b35225-e4f2-e4d6-e4e1-d289490a03ab OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /share/apps/anaconda3/bin/python /mzy/copyfrom_aixi_gadfortest/tools/sam_person_masks.py generate --config /mzy/copyfrom_aixi_gadfortest/configs/sam_person/qc_resolved.yaml --manifest /mzy/copyfrom_aixi_gadfortest/outputs/sam_person/20261018_v1/qc_manifest.jsonl --resume --max-frames 200 --max-actor-prompts 3000 --max-gpu-minutes 60
```

```bash
CUDA_VISIBLE_DEVICES='' /share/apps/anaconda3/bin/python tools/sam_person_masks.py visualize --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/20261018_v1/qc_manifest.jsonl --out outputs/sam_person/20261018_v1/qc_gallery
```

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 /share/apps/anaconda3/envs/torch171/bin/python tools/sam_person_masks.py qc --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/20261018_v1/qc_manifest.jsonl --require-complete --out outputs/sam_person/20261018_v1/qc_metrics.json
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-d2b35225-e4f2-e4d6-e4e1-d289490a03ab OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 TORCH_HOME=/mzy/copyfrom_aixi_gadfortest/.local_data/torch171 /share/apps/anaconda3/envs/torch171/bin/python tools/run_sam_person_pilot.py --config configs/sam_person/pilot_resolved.yaml --variant B0
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-37772aef-f317-5c32-ed2d-168988272873 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 TORCH_HOME=/mzy/copyfrom_aixi_gadfortest/.local_data/torch171 /share/apps/anaconda3/envs/torch171/bin/python tools/run_sam_person_pilot.py --config configs/sam_person/pilot_resolved.yaml --variant S2F
```

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 /share/apps/anaconda3/envs/torch171/bin/python tools/sam_person_masks.py qc --config configs/sam_person/qc_resolved.yaml --manifest outputs/sam_person/20261018_v1/qc_manifest.jsonl --require-complete --out outputs/sam_person/20261018_v1/qc_metrics_stable.json
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-d2b35225-e4f2-e4d6-e4e1-d289490a03ab OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 TORCH_HOME=/mzy/copyfrom_aixi_gadfortest/.local_data/torch171 /share/apps/anaconda3/envs/torch171/bin/python tools/run_sam_person_pilot.py --config configs/sam_person/pilot_resolved.yaml --variant B0 --stages reference overfit pilot
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-37772aef-f317-5c32-ed2d-168988272873 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 TORCH_HOME=/mzy/copyfrom_aixi_gadfortest/.local_data/torch171 /share/apps/anaconda3/envs/torch171/bin/python tools/run_sam_person_pilot.py --config configs/sam_person/pilot_resolved.yaml --variant S2F --stages reference overfit pilot
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-37772aef-f317-5c32-ed2d-168988272873 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 TORCH_HOME=/mzy/copyfrom_aixi_gadfortest/.local_data/torch171 /share/apps/anaconda3/envs/torch171/bin/python tools/run_sam_person_pilot.py --config configs/sam_person/pilot_extended_resolved.yaml --variant S2F --stages reference pilot
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-d2b35225-e4f2-e4d6-e4e1-d289490a03ab OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 TORCH_HOME=/mzy/copyfrom_aixi_gadfortest/.local_data/torch171 /share/apps/anaconda3/envs/torch171/bin/python tools/run_sam_person_pilot.py --config configs/sam_person/pilot_extended_resolved.yaml --variant B0 --stages reference pilot
```


## 最终汇总与交付核验（已执行，无模型运行）

工作目录 `/mzy`；使用绝对路径。下列命令只读取已完成实验并生成报告，不导入模型或启动GPU。

```bash
/share/apps/anaconda3/bin/python /mzy/copyfrom_aixi_gadfortest/outputs/sam_person/20261018_v1/collect_final.py > /mzy/copyfrom_aixi_gadfortest/outputs/sam_person/20261018_v1/collect_final.log
/share/apps/anaconda3/bin/python /mzy/copyfrom_aixi_gadfortest/outputs/sam_person/20261018_v1/write_report.py
/share/apps/anaconda3/bin/python /mzy/copyfrom_aixi_gadfortest/outputs/sam_person/20261018_v1/verify_delivery.py > /mzy/copyfrom_aixi_gadfortest/outputs/sam_person/20261018_v1/verify_delivery.log
```

正式80步和300步的源码清单不同：80步使用 `source_archives/0a3fb1b2cc1d9a7db0cea0de38b1007d807a941c323664e026f17000d86fda11/` 中的冻结源码，300步使用 `source_delivery/`。来源核验会拒绝在不匹配源码下直接重放旧配置；不得为重跑而清零已有GPU累计ledger或覆盖旧run。

早期自动记录的SAM smoke命令未打印完整环境前缀；实际生成配置锁定物理GPU4的UUID，后续同环境命令已打印完整前缀。SAM与GAD使用独立现有Python环境；CPU验收日志为 `outputs/sam_person/20261018_v1/cpu_acceptance_stable.log`。

CPU测试的等价复核入口如下，仅供复核，收尾没有重复运行已通过的6项测试：

```bash
cd /mzy/copyfrom_aixi_gadfortest
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 TORCH_HOME=/mzy/copyfrom_aixi_gadfortest/.local_data/torch171 /share/apps/anaconda3/envs/torch171/bin/python -m unittest discover -s tests/sam_person
```

全量4卡S2F/DDP训练未运行，现有入口只支持有界验证，没有已验证的全量启动命令。后续P5需独立预算、全量缓存/数据流和DDP入口验证，本轮不启动。


## GitHub分支整理复核（2026-10-18）

工作目录 `/mzy/copyfrom_aixi_gadfortest-sam-person-20261018`。活动等价性测试固定到实验前B0 revision，GPU禁用；没有增加训练。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mzy/copyfrom_aixi_gadfortest/.local_data/heatmap_py38_20261013 TORCH_HOME=/mzy/copyfrom_aixi_gadfortest/.local_data/torch171 /share/apps/anaconda3/envs/torch171/bin/python -m unittest discover -s tests/sam_person > results/sam_person/20261018_v1/github_cpu_acceptance.log 2>&1
CUDA_VISIBLE_DEVICES='' /share/apps/anaconda3/bin/python tools/collect_sam_person_results.py > results/sam_person/20261018_v1/github_collection.log
/share/apps/anaconda3/bin/python results/sam_person/20261018_v1/github_verify.py > results/sam_person/20261018_v1/github_verify.log
```

画廊、逐actor QC和轻量结果从服务器outputs命名空间复制到Git跟踪的results/sam_person/20261018_v1，报告链接已调整。GPU原始结果及源码快照保持原值；服务器绝对路径是历史来源，clone后的汇总入口使用新的便携收集器。
