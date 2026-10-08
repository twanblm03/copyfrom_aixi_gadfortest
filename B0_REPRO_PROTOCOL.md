# Café B0-repro protocol

`b0-repro` is the untouched upstream snapshot at commit
`caee21a8ccdd5faadecbc5a5034107084a7cd270`.  Experiments run from
`exp/b0-repro`, which keeps the upstream model and losses while adding only
runtime compatibility and a leakage-free evaluation protocol.

## Frozen model

- ImageNet-pretrained ResNet-18, trained end to end.
- Five frames per clip with segment-based sampling.
- Ground-truth actor tracklets, at most 14 actors.
- Input resolution 1280x720.
- RoIAlign crop size 5x5 and actor/group hidden dimension 256.
- Six-layer Grouping Transformer with four heads and FFN dimension 512.
- Twelve frame-specific group tokens and distance-mask threshold 0.2.
- Original individual classification, group classification, membership, and
  group-consistency losses. Their coefficients are 1, 1, 5, and 2; the
  consistency temperature is 0.2.
- No DINOv2, VideoMAE, extra HOI graph, mask input, heatmap head, or explicit
  temporal encoder.

The upstream `longcw/RoIAlign.pytorch` extension is unavailable under the
workspace's PyTorch version. `exp/b0-repro` therefore implements its 5x5
contract with `torchvision.ops.RoIAlign`: aligned coordinates, one center
sample per bin, plus explicit zero extrapolation for out-of-range centers.
This matches the upstream reference example, but the compatibility
substitution must still be recorded in reports and checked against the
released checkpoint or published range before calling the full model
numerically exact.

## Optimization

The primary run follows the paper rather than the released code's conflicting
scheduler default:

- Adam with beta1=0.9, beta2=0.999, epsilon=1e-8 and no weight decay;
- global batch size 16 for 30 epochs;
- learning rate 1e-5 to 1e-4 over the first five epochs, then linear decay;
- random seeds 1, 2, and 3.

`--lr_schedule cyclic --weight_decay 1e-4` is retained only for a separately
named released-code diagnostic. It must not be mixed into B0-repro results.

## Evaluation

- Development: optimize TRAIN, evaluate VAL, and select `best_val.pth` by
  Group mAP@1.0.
- Final report: freeze the configuration, train on TRAIN+VAL, and evaluate
  TEST exactly once using `--final_test`.
- Validation uses deterministic temporal segment centers and a sequential
  sampler. Repeated evaluation of a fixed checkpoint must produce identical
  predictions and metrics.
- Report Group mAP@1.0, Group mAP@0.5, and Outlier mIoU from the same
  primary-selected epoch, for every seed plus mean and population standard
  deviation.

## Commands

Inspect commands without starting training:

```bash
bash scripts/run_b0_repro_val.sh --dry-run
```

Run the place split with three seeds:

```bash
bash scripts/run_b0_repro_val.sh --split place --seeds 1,2,3 --gpus 0,1,2,3
```

Summarize completed runs:

```bash
python scripts/summarize_b0_repro.py /mzy/cafe_runs/b0-repro/results \
  --split place --seeds 1,2,3
```
