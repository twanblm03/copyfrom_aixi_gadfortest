# Cafe B0 baseline protocol

> **Current split-specific rule (2026-10-23, user instruction).** All future
> **view** experiments train from the common ImageNet initialization on the
> official **TRAIN+VAL** videos (three camera views) for 30 epochs and evaluate
> the fourth camera view, **TEST**, after every epoch. Retain every epoch's
> checkpoint, select **best_test.pth** by TEST **group mAP@1.0**, and report
> all secondary metrics from that same checkpoint. Also retain **epoch30.pth**.
> Label these results **best-on-TEST**. This is the user-authorized TEST
> selection protocol, rather than an independent VAL-selected result.
> Each new run starts afresh; do not initialize it from the two-view B0.
> This rule overrides earlier TRAIN-to-VAL descriptions for future view runs.
> The existing two-view B0 may finish as a development control; its VAL scores
> are not paper TEST results. The **place development protocol is unchanged**:
> TRAIN training, fixed-center VAL evaluation and VAL checkpoint selection.
> Preserve the current ResNet18/DP4 optimization settings and corrected
> evaluator in both splits. See the
> [view protocol and paper comparison](../../reports/heatmap/view_protocol_vs_paper_20261023.md)
> for exact scope and remaining implementation differences.

> Current research rule (2026-10-21): use the ResNet18 random-segment TRAIN,
> fixed-center VAL, four-GPU Host DataParallel protocol described in
> [the current B0 audit](../../reports/heatmap/b0_baseline_vs_cafe_paper_20261021.md).
> The DINOv2/VideoMAE protocol below is a historical, separate model variant.
> The user requires the corrected evaluator for every future evaluation and
> checkpoint selection; never execute the legacy set-ranking implementation.
> Use numeric confidence ranking and the explicit evaluation clip list.
> The three report labels `new_dp4_corrected_evaluator`,
> `old_reselected_corrected_evaluator`, and `old_original_corrected_evaluator`
> all use the same corrected metric implementation; only the first names the
> recommended new training run.
> New paired experiments use the same ImageNet initialization rule. Do not
> resume legacy best/selection scores without verifying the evaluator version
> and re-evaluating any unknown historical records under the corrected metric.

This document freezes the mask-free baseline used before any SAM, mask
pooling, group heatmap, or heatmap-membership consistency experiment.

## Task and data contract

- Dataset: Cafe.
- Place development protocol, and historical view development controls: train
  on the official TRAIN videos and select models on official VAL (`--val_mode`).
- Future view protocol: start a fresh 30-epoch TRAIN+VAL run from ImageNet and
  evaluate TEST every epoch. Keep all epoch checkpoints, select best_test.pth
  by TEST group mAP@1.0, retain epoch30.pth, and label results best-on-TEST.
- Place final-test protocol: TEST remains untouched during place development.
  Train on TRAIN+VAL and evaluate TEST after the configuration and epoch
  budget have been frozen; this place rule does not override the view rule.
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

- Future view runs use `best_on_test=true`, `keep_all_checkpoints=true`,
  `protocol=best_on_test`, and `eval_split=test`. Keep all epoch predictions
  and metrics as well as checkpoints; the selected checkpoint is best_test.pth.
- Place development continues to select best_val.pth on VAL.
- Primary selection and reporting metric: group mAP at group IoU 1.0.
- Secondary metrics: group mAP at group IoU 0.5 and outlier mIoU, both
  measured at the checkpoint selected by the primary metric. Do not select a
  separate epoch for each secondary metric.
- Run seeds 1, 2, and 3 with identical data and hyperparameters. Report
  per-seed values plus mean and standard deviation.
- Validation frame indices are deterministic temporal segment centers.
- A fixed checkpoint must produce identical prediction content and metrics on
  two repeated evaluations.
- TEST-evaluated or TEST-selected metrics must retain their TEST label.
  Future view best-on-TEST results are not independent VAL-selected results.

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
2. each run retains its selected checkpoint and full resolved args (place:
   best_val.pth; future view: all epoch checkpoints plus best_test.pth);
3. repeated evaluation of one checkpoint is identical;
4. the three-seed primary metric mean and variation are recorded; and
5. selection metadata records the intended split: VAL for place development,
   and explicitly labelled best-on-TEST for future view runs.

Before an official three-seed run, record the exact code revision (or commit
the reviewed baseline changes) so later mask experiments use the same B0
implementation.
