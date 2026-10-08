# Cafe B0 baseline protocol

This document freezes the mask-free baseline used before any SAM, mask
pooling, group heatmap, or heatmap-membership consistency experiment.

## Task and data contract

- Dataset: Cafe.
- Development protocol: train on the official TRAIN videos and select models
  on the official VAL videos (`--val_mode`).
- Final-test protocol: TEST remains untouched during development. Train on
  TRAIN+VAL and evaluate TEST only after the configuration and epoch budget
  have been frozen.
- Actor input: `gt_tracks.pkl` track boxes (`eval_type=gt_base`). This is the
  oracle-track setting and must not be described as detector-based inference.
- Labels: clip-level group activity and membership, with actor identities
  supplied by the tracklets.
- Output: unordered group-query predictions containing an activity class and
  a membership score for every actor.

## Frozen B0 model and optimization

- No SAM masks or mask-derived features.
- DINOv2 ViT-B/14 backbone; only the last two blocks and final norm are
  unfrozen, with backbone learning rate scaled by 0.1.
- Existing RoIAlign actor extraction, HOI graph, group queries, Hungarian
  activity/membership matching, membership head, consistency loss, and
  temporal encoder remain enabled and unchanged.
- Eight sampled frames per clip.
- VideoMAE-v2 cached clip features enabled.
- AdamW, weight decay `1e-4`, cyclic schedule with steps 4/25.
- Global batch size 16, 30 epochs, validation every epoch.

Any change to this section creates a different baseline variant and must not
be mixed into a mask ablation.

## Metrics and model selection

- Primary selection and reporting metric: group mAP at group IoU 1.0.
- Secondary metrics: group mAP at group IoU 0.5 and outlier mIoU, both
  measured at the checkpoint selected by the primary metric. Do not select a
  separate epoch for each secondary metric.
- Run seeds 1, 2, and 3 with identical data and hyperparameters. Report
  per-seed values plus mean and standard deviation.
- Validation frame indices are deterministic temporal segment centers.
- A fixed checkpoint must produce identical prediction content and metrics on
  two repeated evaluations.
- Historical metrics obtained by evaluating TEST every epoch are reference
  diagnostics only; they are not valid B0 validation results.

## Commands

Inspect the three resolved commands without starting training:

```bash
bash scripts/run_cafe_baseline_val.sh --dry-run
```

Run the place-split validation baseline:

```bash
bash scripts/run_cafe_baseline_val.sh \
  --split place \
  --seeds 1,2,3 \
  --gpus 0,1,2,3
```

Resume one interrupted seed from its latest `epochN.pth`:

```bash
bash scripts/run_cafe_baseline_val.sh \
  --split place \
  --seed 1 \
  --gpus 0,1,2,3 \
  --resume-path /path/to/epoch10.pth
```

After all seeds finish, summarize metrics from the common primary-selection
epoch rule:

```bash
python scripts/summarize_cafe_baseline.py result \
  --split place \
  --seeds 1,2,3
```

Environment-specific paths and GPU IDs may be overridden by the launcher's
documented options. Model or optimization overrides must be recorded as a new
variant rather than silently replacing B0.

## Exit criteria

B0 is considered established only when:

1. all three seeds finish without non-finite losses;
2. each run retains its best validation checkpoint and full resolved args;
3. repeated evaluation of one checkpoint is identical;
4. the three-seed primary metric mean and variation are recorded; and
5. no TEST result was used for checkpoint, threshold, or hyperparameter
   selection.

Before an official three-seed run, record the exact code revision (or commit
the reviewed baseline changes) so later mask experiments use the same B0
implementation.
