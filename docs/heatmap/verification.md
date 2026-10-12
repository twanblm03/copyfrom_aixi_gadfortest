# Executed verification — 2026-10-12

以下命令已执行；正式训练和500/1000步后续命令没有执行。
工作目录 `/home/zhenyu/Documents/CAFE_codebase-main`，main revision
`0ffe57666cb972ac6dc2b9593df556045b727e44`，代码修改尚未commit/push。

```bash
# 宿主机、继承CUDA allocation：15 tests通过，无skip。
set -o pipefail
.miniconda3/envs/gad/bin/python tests/heatmap/test_contract.py 2>&1 | \
  tee outputs/heatmap/place/contract_tests_cuda_runnerfinal_20261012.log

# 原有DINO helper回归：8 tests通过。
.miniconda3/envs/gad/bin/python -m unittest discover -s tests -p 'test_dino*.py' -v 2>&1 | \
  tee outputs/heatmap/place/dino_regression_final_20261012.log

# 编译和语法检查：exit0。
.miniconda3/envs/gad/bin/python -m py_compile \
  models/b0_repro.py models/b0_transformer.py models/heatmap.py \
  models/heatmap_criterion.py models/__init__.py dataloader/cafe.py train.py test.py \
  util/group_export.py util/experiment.py tools/run_heatmap_research.py \
  tools/report_heatmap_research.py tests/heatmap/test_contract.py
bash -n scripts/run_heatmap_paired_pilot.sh
git diff --check

# 汇总及固定样本可视化：四配置八项配对控制全部通过。
.miniconda3/envs/gad/bin/python tools/report_heatmap_research.py \
  --paired-root outputs/heatmap/place/paired_pilot8_rngfixed_seed1_20261012 --visualize

# 只读检查：实验结束时没有遗留compute process。
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

所有真实训练/oracle/evaluate命令在 `results/heatmap/commands.md`，
各自输出目录的 `command.txt`、`resolved_config.json`、源码快照和日志保留可重放的实际状态。
工程运行完成不等于通过性能gate；见 `results/heatmap/gate_assessment.json`。
