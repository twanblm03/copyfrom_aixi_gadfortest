# CAFE group heatmap research

This branch publishes the runnable ResNet B0 / H1 / H2-S research path, not a copy
of the original server's environment, dataset or training artifacts. Start with
[the execution guide](CAFE_HEATMAP_RESEARCH_EXECUTION_GUIDE.md) and
[the baseline audit](docs/heatmap/baseline_audit.md).
Publication checks and intentionally excluded local files are recorded in
[publication verification](docs/heatmap/publication_verification.md).

## Get the code and reference used by tests

Use a new directory; do not overwrite another dirty checkout.

```bash
git clone --branch exp/cafe-heatmap \
  https://github.com/twanblm03/copyfrom_aixi_gadfortest.git CAFE_heatmap
cd CAFE_heatmap
git fetch origin exp/b0-repro:refs/heads/exp/b0-repro
git rev-parse HEAD
git rev-parse exp/b0-repro
```

The frozen reference is `9e6a83f5374f2d5a61a60ce3f44d17a2b55c75bc`.
The contract tests compare against `git show exp/b0-repro:models/models.py`;
if the fetched reference differs, investigate before training. A single-branch
clone without that additional reference is not enough for this test.

This research branch is based on main
`0ffe57666cb972ac6dc2b9593df556045b727e44`. The default main model and defaults
are not the frozen B0: use `tools/run_heatmap_research.py` with the supplied
`configs/heatmap/*.json`, not an unqualified `train.py` launch.

## Environment and external assets

The verified environment is Python 3.8.5, PyTorch 1.7.1 (CUDA 11.0),
torchvision 0.8.2, NumPy 1.21.2, SciPy 1.7.1 and Pillow 8.4.0. Reuse a compatible
environment and run the tests; do not silently upgrade dependencies or copy the
old server's entire `.miniconda3`. Matplotlib is optional.

Provide the CAFE dataset and official evaluation ground truth separately:

```text
Dataset/cafe/<video>/<clip>/ann.json
Dataset/cafe/<video>/<clip>/images/frames_<fid>.jpg
Dataset/cafe/gt_tracks.pkl
evaluation/gt_tracks.txt
```

`Dataset` may point to this server's dataset root. The old server's absolute
symlink targets must not be reused. `evaluation/gt_tracks.txt` is also an
external asset and is intentionally not in Git. The tracked label map is
`label_map/group_action_list.pbtxt`.

Expected place-protocol retained sample counts are TRAIN=6,444 and VAL=1,386;
TEST is not used for development or selection. Check annotations, track mapping,
data/split manifests and the evaluator hash as described in the baseline audit,
not only sample counts. ResNet18 ImageNet weights (`resnet18-5c106cde.pth`) must
be available through the torchvision cache/download mechanism. Do not warm-start
from the published CAFE checkpoint or from a smoke/overfit/pilot checkpoint.

## GPU selection and preflight

Inspect allocation, utilization and CUDA health first. Explicitly set
`CUDA_VISIBLE_DEVICES` to this server's two allocated, healthy GPU UUIDs, and use
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. They become logical GPUs `0,1`. Never copy the
original server's physical indices or UUIDs. The original machine has a faulty
GPU0, but that says nothing about this machine's GPU0.

The commands below assume the verified Python is at
`.miniconda3/envs/gad/bin/python`; replace that executable path if necessary.
Run from the project root and use new output directories for every attempt.

```bash
.miniconda3/envs/gad/bin/python -m unittest discover \
  -s tests/heatmap -p test_contract.py -v

.miniconda3/envs/gad/bin/python tools/run_heatmap_research.py \
  --stage audit --config configs/heatmap/b0.json \
  --output outputs/heatmap/place/remote_b0_audit_run1

.miniconda3/envs/gad/bin/python tools/run_heatmap_research.py \
  --stage smoke --config configs/heatmap/b0.json \
  --steps 3 --train-clips 16 --eval-clips 16 --workers 4 \
  --device cuda:0 --parallel-gpus 0,1 \
  --output outputs/heatmap/place/remote_b0_smoke_run1

.miniconda3/envs/gad/bin/python tools/run_heatmap_research.py \
  --stage pilot --config configs/heatmap/b0.json \
  --steps 20 --seed 1 --train-clips 64 --eval-clips 32 \
  --workers 4 --persistent-workers --device cuda:0 --parallel-gpus 0,1 \
  --output outputs/heatmap/place/remote_b0_profile_run1
```

There are 19 contract tests. A CPU-only run skips the CUDA RNG test and is not
a replacement for actual GPU preflight. Confirm finite forward/loss/backward,
serialization/reload and evaluation export on the selected GPUs. Measure warm
steps, per-GPU memory and evaluation time; do not extrapolate the cold first step.

## Frozen full-training protocol and authorization

Use ResNet18, T=5, K=12, 720x1280, global batch16, micro_batch0, FP32 and the
current DataParallel path. Keep original activity, K-softmax membership,
threshold/no-group order and auxiliary losses. Adam has weight decay0; LR starts
at1e-5, peaks at1e-4 with the original five-epoch warmup and linear schedule.
Keep the same GPU count, sharding and worker policy across formal comparisons.
Do not silently substitute AMP, gradient accumulation, DDP, frozen BN or smaller
images to fit resources.

Full training has 30 epochs, 402 updates/epoch, 12,060 updates total and full
1,386-clip VAL evaluation after every epoch. Select a single best checkpoint
by VAL `group_mAP@1.0`; also report mAP@0.5, outlier mIoU and per-class AP.
Estimate training plus 30 evaluations, serialization and I/O before requesting
the user's GPU-hours budget. Do not start a multi-seed/configuration matrix just
because the server is idle.

Only after the user approves the budget for B0 seed1, run:

```bash
.miniconda3/envs/gad/bin/python tools/run_heatmap_research.py \
  --stage full --config configs/heatmap/b0.json --seed 1 --eval-clips 0 \
  --workers 4 --persistent-workers --device cuda:0 --parallel-gpus 0,1 \
  --output outputs/heatmap/place/remote_b0_seed1_full_run1
```

`--eval-clips 0` means all VAL. The full stage uses all TRAIN and ignores the
default bounded subset size. Use the server's persistent job mechanism and
confirm actual step progress before handing off logs. `--checkpoint` loads model
weights only; it is not a complete optimizer/LR/RNG resume interface.

## Artifacts and research boundaries

Runs save commands, expanded configuration, environment, source snapshots,
manifests, steps, predictions, checkpoints, metrics and failures. Keep them on
the server; they are not part of this source publication. Existing reports link
to original-server artifacts that will not appear in a fresh clone.

The original server's 19 CUDA tests passed; B0 and H2-S fixed-small-set 500-step
overfits completed. Four-variant 500-step paired pilot was still running at
publication preparation; no full-training or final method-improvement claim is
made. Check dated reports rather than treating old negative/positive diagnostics
as final results. H1/H2 full training and additional seeds require a separate
decision. H3, independent sigmoid membership, SAM, FPN and explicit background
remain separate experiments.

`scripts/run_heatmap_paired_pilot.sh` is configurable, but
`scripts/run_heatmap_dual_gpu_bounded.sh` is deliberately original-server-specific
and hard-codes its GPU2/3 UUIDs. **Do not run that recovery wrapper on another
server.** Use the checked Python commands with locally verified GPU selection.

Unrelated, previously untracked DINO tools, local environments, datasets, results
and third-party caches are not added by this research publication. Upstream
already-tracked files are preserved. The training Python files are not changed
while publishing; a Git revision/dirty-diff metadata change during an in-flight
pilot does not replace the exact source snapshot saved separately for each run.
