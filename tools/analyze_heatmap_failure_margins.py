"""Frozen H2-S failure decomposition. No optimizer, calibration or model changes.

GT is used only after model inference for explicitly labelled diagnostic matching.
Primary assignment is the saved training matcher; a set-IoU-only assignment is
also reported to expose sensitivity to GT/slot association. Neither is inference.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stable_softmax(x, axis=0):
    shifted = np.asarray(x, dtype=np.float64) - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=axis, keepdims=True)


def logsumexp(x):
    maximum = float(np.max(x))
    return maximum + math.log(float(np.exp(x - maximum).sum()))


def decode_slots(probability, activity, threshold=.5):
    """Logical outlier=-1; preserve argmax -> no-group -> threshold ordering."""
    winner = probability.argmax(0)
    maximum = probability.max(0)
    no_group = activity.shape[1] - 1
    classes = activity.argmax(1)
    accepted = (classes[winner] != no_group) & (maximum > threshold)
    return winner, maximum, np.where(accepted, winner, -1), classes


def set_iou_matrix(prediction, truth):
    intersection = np.logical_and(prediction[:, None], truth[None]).sum(-1)
    union = np.logical_or(prediction[:, None], truth[None]).sum(-1)
    return intersection / np.maximum(union, 1)


def routing_category(winner, maximum, classes, expected, assigned_slots, no_group, threshold):
    if winner == expected:
        if classes[winner] == no_group:
            return 'correct_slot_no_group'
        return 'correct_slot_pass' if maximum > threshold else 'correct_slot_threshold_reject'
    if classes[winner] == no_group:
        return 'wrong_slot_no_group'
    return 'wrong_slot_other_gt_group' if winner in assigned_slots else 'wrong_slot_unmatched'


def geometry(boxes, valid, members):
    """Descriptive geometry, not a learned feature or a calibration decision."""
    n = boxes.shape[1]
    area = np.where(valid, boxes[..., 2] * boxes[..., 3], 0).sum(0) / np.maximum(valid.sum(0), 1)
    labels = np.full(n, -1, dtype=int)
    for group, membership in enumerate(members):
        labels[membership] = group
    nearest = np.full(n, np.nan)
    overlap = np.zeros(n)
    for i in range(n):
        distances = []
        for j in range(n):
            if i == j or (labels[i] >= 0 and labels[i] == labels[j]):
                continue
            frames = valid[:, i] & valid[:, j]
            if not frames.any():
                continue
            first, second = boxes[frames, i], boxes[frames, j]
            distances.append(float(np.linalg.norm(first[:, :2] - second[:, :2], axis=1).mean()))
            lo = np.maximum(first[:, :2] - first[:, 2:] / 2, second[:, :2] - second[:, 2:] / 2)
            hi = np.minimum(first[:, :2] + first[:, 2:] / 2, second[:, :2] + second[:, 2:] / 2)
            intersection = np.maximum(hi - lo, 0).prod(1)
            union = first[:, 2:].prod(1) + second[:, 2:].prod(1) - intersection
            overlap[i] = max(overlap[i], float((intersection / np.maximum(union, 1e-12)).mean()))
        if distances:
            nearest[i] = min(distances)
    return area, nearest, overlap


def analyze_clip(probability, logits, response, activity, members, group_activity,
                 assigned_slots, valid, present, clip, boxes=None, frame_valid=None,
                 threshold=.5):
    """Pure diagnostic summaries. All arrays use annotation-row actor indices."""
    probability, logits, response, activity = [np.asarray(x) for x in (probability, logits, response, activity)]
    members, valid, present = np.asarray(members, dtype=bool), np.asarray(valid, dtype=bool), np.asarray(present, dtype=bool)
    if any(not np.isfinite(x).all() for x in (probability, logits, response, activity)):
        raise FloatingPointError('nonfinite diagnostic input')
    if (members.sum(0) > 1).any():
        raise ValueError('an actor cannot be a member of two labelled groups')
    k, n = probability.shape
    if k < 2 or len(assigned_slots) != len(members):
        raise ValueError('invalid diagnostic dimensions/assignment')
    winner, maximum, decoded, classes = decode_slots(probability, activity, threshold)
    no_group = activity.shape[1] - 1
    assigned_slots = np.asarray(assigned_slots, dtype=int)
    truth = members & valid[None]
    threshold_sets = (probability > threshold) & valid[None]
    ious = set_iou_matrix(threshold_sets, truth)
    alternative = np.full(len(truth), -1, dtype=int)
    if len(truth):
        slots, groups = linear_sum_assignment(-ious)
        alternative[groups] = slots
    visible = valid & present
    decoded_sets = np.array([(decoded == slot) & visible for slot in range(k)])
    detected = [slot for slot in range(k) if decoded_sets[slot].sum() >= 2]
    gt_actor_group = np.full(n, -1, dtype=int)
    for group, membership in enumerate(members):
        gt_actor_group[membership] = group
    if boxes is not None:
        area, nearest, overlap = geometry(boxes, frame_valid, members)
    else:
        area, nearest, overlap = np.zeros(n), np.full(n, np.nan), np.zeros(n)
    actors = []
    for actor in range(n):
        # Padding is not an actor. Real but unavailable tracks are retained
        # separately by caller-provided annotation count after this function.
        if not valid[actor]:
            continue
        ordered = np.argsort(-logits[:, actor], kind='stable')
        group = int(gt_actor_group[actor])
        expected = int(assigned_slots[group]) if group >= 0 else -1
        alternate = int(alternative[group]) if group >= 0 else -1
        if group >= 0:
            category = routing_category(int(winner[actor]), maximum[actor], classes, expected,
                set(assigned_slots.tolist()), no_group, threshold)
            alt_category = routing_category(int(winner[actor]), maximum[actor], classes, alternate,
                set(alternative.tolist()), no_group, threshold)
            others = np.arange(k) != expected
            true_margin = float(response[expected, actor] - response[others, actor].max())
            logit_margin = float(logits[expected, actor] - logits[others, actor].max())
            denominator_margin = float(logits[expected, actor] - logsumexp(logits[others, actor]))
        else:
            if classes[winner[actor]] == no_group:
                category = 'individual_rejected_no_group'
            elif maximum[actor] <= threshold:
                category = 'individual_rejected_threshold'
            elif decoded_sets[winner[actor]].sum() < 2:
                category = 'individual_singleton_prediction'
            else:
                category = 'individual_false_group_member'
            alt_category, true_margin, logit_margin, denominator_margin = None, None, None, None
        actors.append(dict(video=int(clip[0]), clip=int(clip[1]), actor_index=actor,
            actor_id_annotation_row1=actor+1, gt_group=group, gt_activity=int(group_activity[group])+1 if group >= 0 else None,
            expected_slot=expected, expected_slot_set_iou_assignment=alternate,
            assignment_disagrees=bool(expected != alternate) if group >= 0 else False,
            winner_slot=int(winner[actor]), winner_activity=int(classes[winner[actor]])+1,
            winner_is_no_group=bool(classes[winner[actor]] == no_group),
            max_probability=float(maximum[actor]), true_slot_probability=float(probability[expected, actor]) if group >= 0 else None,
            decoded_slot=int(decoded[actor]), decoder_group_size=int(decoded_sets[winner[actor]].sum()),
            first_export_frame_present=bool(present[actor]),
            category=category, category_set_iou_assignment=alt_category,
            true_minus_best_other_roi=true_margin, true_minus_best_other_logit=logit_margin,
            true_vs_other_logsumexp=denominator_margin,
            top1_top2_roi_margin=float(response[ordered[0], actor]-response[ordered[1], actor]),
            top1_top2_logit_margin=float(logits[ordered[0], actor]-logits[ordered[1], actor]),
            threshold_rejected=bool(maximum[actor] <= threshold),
            activity_wrong_on_expected_slot=bool(classes[expected] != group_activity[group]) if group >= 0 else None,
            box_area=float(area[actor]), nearest_nonmember_center_distance=float(nearest[actor]) if np.isfinite(nearest[actor]) else None,
            max_nonmember_box_iou=float(overlap[actor])))
    groups = []
    tp = fp = fn = 0
    for group, slot in enumerate(assigned_slots):
        actual_truth = truth[group]
        predicted_threshold = threshold_sets[slot]
        group_tp = int((actual_truth & predicted_threshold).sum())
        group_fp = int((~actual_truth & predicted_threshold).sum())
        group_fn = int((actual_truth & ~predicted_threshold).sum())
        tp += group_tp; fp += group_fp; fn += group_fn
        # Set-level topology uses only actors the first export frame contains.
        target = actual_truth & present
        candidate_ious = [(int(s), float((target & decoded_sets[s]).sum()) /
            max(int((target | decoded_sets[s]).sum()), 1)) for s in detected]
        best_slot, best_iou = max(candidate_ious, key=lambda x:(x[1], -x[0])) if candidate_ious else (-1, 0.)
        exact_slots = [s for s in detected if target.sum() >= 2 and np.array_equal(decoded_sets[s], target)]
        exact_correct = [s for s in exact_slots if classes[s] == group_activity[group]]
        if exact_correct:
            category = 'exact_set_correct_activity'
        elif exact_slots:
            category = 'exact_set_wrong_activity'
        elif best_iou <= 0:
            category = 'no_overlapping_detected_group'
        else:
            prediction = decoded_sets[best_slot]
            missing, extra = int((target & ~prediction).sum()), int((~target & prediction).sum())
            category = 'missing_members_only' if missing and not extra else 'extra_members_only' if extra and not missing else 'missing_and_extra_members'
        occupied = sorted(set(decoded[target].tolist()) - {-1})
        fragments_detected = [s for s in occupied if s in detected]
        touching = [s for s in detected if (target & decoded_sets[s]).any()]
        merges = [s for s in touching if sum(bool((other & decoded_sets[s]).any()) for other in truth) >= 2]
        contaminations = [s for s in touching if ((gt_actor_group < 0) & decoded_sets[s]).any()]
        groups.append(dict(video=int(clip[0]),clip=int(clip[1]),gt_group=group,gt_activity=int(group_activity[group])+1,
            expected_slot=int(slot),expected_slot_set_iou_assignment=int(alternative[group]),
            gt_members_count=int(actual_truth.sum()),gt_members_first_export_frame=int(target.sum()),
            gt_members_missing_first_export_frame=int((actual_truth & ~present).sum()),
            matched_members_tp=group_tp,matched_members_fp=group_fp,matched_members_fn=group_fn,
            matched_threshold_set_iou=float(ious[slot,group]),
            matched_slot_activity=int(classes[slot])+1,matched_slot_is_no_group=bool(classes[slot]==no_group),
            matched_slot_activity_correct=bool(classes[slot]==group_activity[group]),
            best_decoded_slot=best_slot,best_decoded_set_iou=best_iou,
            exact_set_any_activity=bool(exact_slots),exact_set_correct_activity=bool(exact_correct),
            category=category,split_over_accepted_slots=len(occupied)>=2,
            split_over_detected_groups=len(fragments_detected)>=2,merged_with_other_gt_group=bool(merges),
            contaminated_by_individual=bool(contaminations),
            accepted_fragment_count=len(occupied),detected_fragment_count=len(fragments_detected),
            gt_actor_indices=np.where(actual_truth)[0].tolist(),
            predicted_actor_indices=np.where(decoded_sets[best_slot])[0].tolist() if best_slot>=0 else []))
    return actors, groups, dict(tp=tp,fp=fp,fn=fn,matched_exact_sets=sum(g['matched_threshold_set_iou']==1 for g in groups),
        assignment_disagreements=int((assigned_slots != alternative).sum()),
        no_group_slots=int((classes==no_group).sum()),detected_groups=len(detected))


def describe(values):
    values = np.asarray([v for v in values if v is not None],dtype=float)
    if not len(values):
        return dict(count=0,mean=None,p10=None,p50=None,p90=None)
    return dict(count=len(values),mean=float(values.mean()),p10=float(np.quantile(values,.1)),
                p50=float(np.quantile(values,.5)),p90=float(np.quantile(values,.9)))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def summarize(actors, groups, clips):
    members=[r for r in actors if r['gt_group']>=0]
    individuals=[r for r in actors if r['gt_group']<0]
    counts=Counter(r['category'] for r in members)
    group_counts=Counter(r['category'] for r in groups)
    fields=('true_minus_best_other_roi','true_minus_best_other_logit','true_vs_other_logsumexp',
            'top1_top2_roi_margin','max_probability','true_slot_probability')
    by_error={name:{field:describe([r[field] for r in members if r['category']==name]) for field in fields}
              for name in sorted(counts)}
    margins=[r['true_minus_best_other_roi'] for r in members]
    edges=[-float('inf'),-.10,-.05,0,.02,.05,.10,.20,float('inf')]
    bins=[]
    for lower,upper in zip(edges,edges[1:]):
        sample=[r for r in members if lower<=r['true_minus_best_other_roi']<upper]
        bins.append(dict(lower=None if not math.isfinite(lower) else lower,upper=None if not math.isfinite(upper) else upper,
            count=len(sample),correct_slot_pass=sum(r['category']=='correct_slot_pass' for r in sample)))
    tp=sum(r['matched_members_tp'] for r in groups);fp=sum(r['matched_members_fp'] for r in groups);fn=sum(r['matched_members_fn'] for r in groups)
    return dict(clips=len(clips),valid_actors=len(actors),gt_member_actors=len(members),gt_individual_actors=len(individuals),
        gt_groups=len(groups),member_error_counts=dict(counts),individual_error_counts=dict(Counter(r['category'] for r in individuals)),
        member_error_counts_set_iou_assignment=dict(Counter(r['category_set_iou_assignment'] for r in members)),
        group_error_counts=dict(group_counts),
        group_topology_flags={key:sum(bool(r[key]) for r in groups) for key in ('split_over_accepted_slots','split_over_detected_groups','merged_with_other_gt_group','contaminated_by_individual','exact_set_any_activity','exact_set_correct_activity')},
        matched_membership=dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
            mean_set_iou=float(np.mean([r['matched_threshold_set_iou'] for r in groups])) if groups else None,exact_sets=sum(r['matched_threshold_set_iou']==1 for r in groups)),
        assignment_disagreement_groups=sum(c['assignment_disagreements'] for c in clips),
        assigned_slot_activity_counts=dict(Counter('no_group' if r['matched_slot_is_no_group'] else 'correct' if r['matched_slot_activity_correct'] else 'wrong_activity' for r in groups)),
        correct_slot_pass_activity_wrong=sum(r['category']=='correct_slot_pass' and r['activity_wrong_on_expected_slot'] for r in members),
        signed_true_roi_margin=describe(margins),margin_bins=bins,margins_by_member_error=by_error,
        missing_first_export_frame_members=sum(r['gt_members_missing_first_export_frame'] for r in groups))


def audit_group_targets(groundtruth_path, raw_outputs_path):
    """Compare annotation-row group sets/classes with saved official GT.

    This is an audit, never a label correction or an alternate official score.
    """
    from evaluation.cafe_eval import read_text_file, make_groups
    with Path(groundtruth_path).open() as stream:
        boxes, ids, labels, _, scores = read_text_file(stream, 'gt_base', 'gt')
    official, activities, _ = make_groups(boxes, ids, labels, scores)
    mismatches, counts, clips = [], Counter(), 0
    with Path(raw_outputs_path).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = ','.join(map(str, row['clip']))
            expected = []
            for gid, actor_ids in official[key][0].items():
                label = next(iter(activities[key][0][gid]))
                if 1 <= label <= 6 and len(actor_ids) >= 2:
                    expected.append((label, tuple(sorted(actor_ids))))
                    counts[label] += 1
            observed = [(int(label)+1, tuple(i for i,value in enumerate(mask) if value))
                        for label,mask in zip(row['target_activities'],row['target_members'])]
            if sorted(expected) != sorted(observed):
                mismatches.append(dict(clip=row['clip'],official=expected,targets=observed))
            clips += 1
    return dict(clips=clips,official_gt_groups=sum(counts.values()),
                official_groups_per_class=dict(counts),
                target_annotation_vs_official_gt_mismatch_count=len(mismatches),mismatches=mismatches)


def make_report(out, summary, record, verification):
    m=summary['matched_membership'];total=summary['gt_member_actors']
    lines=['# H2-S frozen failure / slot-margin diagnosis', '',
        'No optimizer updates, threshold tuning, new ROI or altered decoder. GT-assisted diagnostic assignments are not inference predictions.', '',
        '## Verified frozen run', '',
        f"- Source: `{record['source_run']}`; best epoch={record['checkpoint_epoch']}; alpha={record['learned_alpha']:.6f}.",
        f"- Clips={summary['clips']}; official prediction bytes match source: {verification['prediction_hash_matches_source_subset']}.",
        f"- Model state and checkpoint unchanged: {verification['model_state_unchanged']} / {verification['checkpoint_file_unchanged']}.",
        f"- Original matched-member TP/FP/FN reproduced: {verification['original_matched_counts_reproduced']}.", '',
        '## Mutually exclusive GT-member routing errors', '',
        '| category | actors | % of valid GT members |','|---|---:|---:|']
    for name,count in sorted(summary['member_error_counts'].items(),key=lambda x:-x[1]):
        lines.append(f'| {name} | {count} | {100*count/max(total,1):.2f} |')
    lines.extend(['',
        'Primary expected slots are the unchanged training Hungarian matcher (activity + membership + mask).',
        'Wrong-slot categories take priority even if the winning probability is below threshold. Threshold/no-group flags are also in actors.csv.',
        '`correct_slot_pass` includes wrong activity classifications; `correct_slot_no_group` respects original no-group precedence.',
        'Alternative set-IoU-only Hungarian assignments are a sensitivity check, not a learned replacement or calibration.', '',
        f"Matched-member precision/recall/F1 = {100*m['precision']:.2f}/{100*m['recall']:.2f}/{100*m['f1']:.2f}%.", '',
        '## GT-group set / activity errors', '',
        '| category | GT groups |','|---|---:|'])
    for name,count in sorted(summary['group_error_counts'].items(),key=lambda x:-x[1]):
        lines.append(f'| {name} | {count} |')
    lines.extend(['',
        'Groups are compared to decoded active groups with at least two visible actors. Best-overlap matches are descriptive, not confidence-ranked AP matching.',
        'Set diagnostics use first exported sampled-frame visibility, as the official evaluator does; they do not imply extra keyframe sampling.',
        'Split/merge/outlier-contamination flags overlap and must not be summed as an exclusive decomposition.', '',
        '## Slot margins', '',
        '`true_minus_best_other_roi = ROI(expected GT slot) - max ROI(other slots)`; negative values mean GT-slot ranking failure.',
        '`true_vs_other_logsumexp = logit(expected) - logsumexp(logits(other slots))`;',
        'at the unchanged 0.5 threshold it is positive exactly when the expected-slot probability exceeds 0.5 (apart from floating-point boundary ties).',
        'Positive alpha cannot change slot ranking. Low confidence with correct ranking and wrong-slot ranking therefore require different remedies.',
        'Aggregate member/nonmember means are not per-actor competition margins.', '',
        '## Artifacts / boundaries', '',
        '- `actors.csv`: valid actor rows, exclusive categories, competing-slot margins and geometry.',
        '- `groups.csv`: matched set errors, activity and overlapping topology flags.',
        '- `clip_outputs.jsonl`: all slots, probabilities/logits/ROI, GT and current matching for each clip; GT appended only after forward.',
        '- `summary.json`, `verification.json`, `per_class.csv`, `per_video.csv`: machine-readable results.',
        '- `prediction_export_val.txt`, `metrics_val.json`: unchanged original decoder/evaluator export and metrics.',
        '- `command.txt`, configs/environment/source snapshots, `train.log`, `run_record.json`: provenance (zero training steps).',
        '- Diagnostic outputs must not replace the published full-run metrics or be used to tune thresholds on VAL.',
        '- Annotation-row actor IDs are not raw external person IDs. Padding and unavailable tracks are excluded and recorded.', ''])
    consistency_path=out/'official_gt_consistency.json'
    if consistency_path.exists():
        consistency=json.loads(consistency_path.read_text())
        lines.extend(['## Annotation / official-GT consistency audit','',
            f"Mismatching clips={consistency['target_annotation_vs_official_gt_mismatch_count']}/{consistency['clips']}; see official_gt_consistency.json.",
            'Routing and topology diagnostics use dataset annotation targets; official AP still uses the unchanged saved official GT.',
            'Annotation disagreements are retained, not silently corrected. Official metric/prediction checks remain independent.', ''])
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--eval-clips',type=int,default=0,help='0 means all source VAL; prefix subsets only for engineering verification')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--parallel-gpus',default='0,1')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--use-saved-source',action='store_true',
        help='import the verified training source_snapshot instead of the modified working tree')
    cli=parser.parse_args()
    import os
    import hashlib
    import shutil
    source=Path(cli.run).resolve();out=Path(cli.output).resolve()
    inference_root=source/'source_snapshot' if cli.use_saved_source else ROOT
    if cli.use_saved_source:
        manifest=json.loads((source/'source_manifest.json').read_text())
        # Verify before importing ANY model/loader/evaluator module from it.
        for relative,expected in manifest.items():
            path=inference_root/relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('saved training source is incomplete/changed: '+relative)
        sys.path.insert(0,str(inference_root))
    import torch
    from torch.utils.data import DataLoader, Subset
    from tools.run_heatmap_research import (collate,get_batch,read_dataset,build_model,seed_all,record_context,
        dataset_manifest,json_write,file_hash,tensor_hash)
    from models.heatmap import prepare_targets
    from models.heatmap_criterion import HeatmapMatcher
    from util.group_export import make_txt
    from evaluation.cafe_eval import GAD_Evaluation
    os.chdir(str(ROOT))
    source_record=json.loads((source/'run_record.json').read_text())
    args=argparse.Namespace(**{k:v for k,v in json.loads((source/'resolved_config.json').read_text()).items() if k!='runtime'})
    for key in ('data_path','groundtruth','labelmap'):
        setattr(args,key,str((ROOT/getattr(args,key)).resolve()))
    if source_record['status']!='completed' or args.heatmap_variant!='h2_s' or not args.val_mode or args.split!='place':
        raise ValueError('completed place TRAIN->VAL H2-S source required')
    if args.test_batch!=1 or cli.eval_clips<0 or cli.workers<0:
        raise ValueError('preserve original eval batch1 and nonnegative limits')
    cli.checkpoint=str(source/'checkpoints/best_val.pth');cli.stage='failure_margin';cli.steps=0
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'diagnostic_driver.py')
    log=(out/'train.log').open('x',buffering=1)
    class Tee:
        def __init__(self,stream):self.stream=stream
        def write(self,value):self.stream.write(value);log.write(value)
        def flush(self):self.stream.flush();log.flush()
    sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr)
    record=record_context(out,args,cli)
    record.update(source_run=str(source),checkpoint_path=cli.checkpoint,checkpoint_selection_rule='source best VAL mAP@1.0, frozen',optimizer_updates=0,
        inference_source_root=str(inference_root),frozen_saved_source=cli.use_saved_source)
    json_write(out/'run_record.json',record)
    started=time.monotonic()
    try:
        manifest=json.loads((source/'source_manifest.json').read_text())
        tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only',source_record['code_revision']]).decode().splitlines())
        verified_paths=manifest if cli.use_saved_source else {p:h for p,h in manifest.items() if p in tracked}
        if any(file_hash(inference_root/p)!=h for p,h in verified_paths.items()):
            raise RuntimeError('original tracked inference/training sources changed')
        json_write(out/'verified_original_source_manifest.json',verified_paths)
        seed_all(args.random_seed);torch.set_num_threads(4)
        train_set,val_set=read_dataset(args);train_set.frames.sort();val_set.frames.sort()
        hashes=dataset_manifest(train_set,val_set,out)
        record['data_manifest_hash'],record['split_hash']=hashes
        if hashes!=(source_record['data_manifest_hash'],source_record['split_hash']):
            raise RuntimeError('source data/split manifest changed')
        selected=json.loads((source/'selected_clips.json').read_text())['val']
        if cli.eval_clips:selected=selected[:cli.eval_clips]
        lookup={tuple(clip):i for i,clip in enumerate(val_set.frames)}
        indices=[lookup[tuple(clip)] for clip in selected]
        json_write(out/'selected_clips.json',dict(train=[],val=selected))
        device=torch.device(cli.device)
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        model,_=build_model(args)
        checkpoint_hash=file_hash(cli.checkpoint)
        saved=torch.load(cli.checkpoint,map_location='cpu');model.load_state_dict(saved['state_dict'],strict=True)
        record.update(checkpoint_sha256=checkpoint_hash,checkpoint_epoch=saved['epoch'],learned_alpha=float(model.mask_head.alpha),
            source_model_state_hash=tensor_hash(model,common=False))
        del saved
        model.to(device);model.eval()
        if cli.parallel_gpus:
            devices=[int(x) for x in cli.parallel_gpus.split(',')]
            if device.type!='cuda' or devices[0]!=(device.index or 0):raise ValueError('invalid GPU mapping')
            model=torch.nn.DataParallel(model,device_ids=devices)
        core=model.module if isinstance(model,torch.nn.DataParallel) else model
        record['inference_batch']=1
        record['parallel_gpus']=[int(x) for x in cli.parallel_gpus.split(',')] if cli.parallel_gpus else []
        matcher=HeatmapMatcher(args)
        loader=DataLoader(Subset(val_set,indices),batch_size=1,shuffle=False,num_workers=cli.workers,collate_fn=collate)
        prediction=out/'prediction_export_val.txt';prediction.touch(exist_ok=False)
        raw=(out/'clip_outputs.jsonl').open('x',buffering=1)
        actors,groups,clip_rows=[],[],[]
        unavailable=[]
        original_diag=json.loads((source/'heatmap_diagnostics_val.json').read_text())
        for batch_index,batch in enumerate(loader):
            with torch.no_grad():
                x,targets,infos,boxes,dummy,valid=get_batch(batch,device)
                outputs=model(x,boxes,dummy,actor_frame_valid=valid)
                make_txt(boxes,infos,outputs,{i:i for i in range(1,25)},str(prediction),args)
                # Only after the target-free model forward may GT be used.
                for target in targets:target['dummy_idx']=(target['actions']!=args.num_class+1).long()
                prepared=prepare_targets(outputs,targets,args)
                slots,gt_indices=matcher(outputs,prepared)[0]
                assignment=np.full(len(targets[0]['activities'][0]),-1,dtype=int)
                assignment[gt_indices.cpu().numpy()]=slots.cpu().numpy()
                if (assignment<0).any():raise RuntimeError('unassigned GT group')
                actual_valid=outputs['actor_frame_valid']
                pooled=outputs['spatial_response'].sum(2)/actual_valid.sum(1)[:,None].clamp(min=1)
                p=outputs['membership'][0].cpu().numpy();l=outputs['membership_logits'][0].cpu().numpy();r=pooled[0].cpu().numpy()
                activity=outputs['pred_activities'][0].cpu().numpy()
                if not np.allclose(p,stable_softmax(l,axis=0),rtol=1e-5,atol=1e-7):raise AssertionError('membership softmax mismatch')
                if not np.allclose(l,record['learned_alpha']*r,rtol=1e-5,atol=1e-6):raise AssertionError('ROI/logit mismatch')
                truth=targets[0]['members'][0].bool().cpu().numpy()
                labels=targets[0]['activities'][0].cpu().numpy()
                actor_valid=valid[0].any(0).cpu().numpy()
                frame_valid=actual_valid[0].cpu().numpy();b=boxes[0].cpu().numpy()
                present=(b[0]!=0).any(-1)
                real=(~dummy[0]).cpu().numpy()
                info=infos[0];clip=[info['vid'],info['sid']]
                actor_rows,group_rows,cr=analyze_clip(p,l,r,activity,truth,labels,assignment,
                    actor_valid,present,clip,b,frame_valid,args.group_threshold)
                for i in np.where(real & ~actor_valid)[0]:unavailable.append(dict(video=clip[0],clip=clip[1],actor_index=int(i)))
                cr.update(video=clip[0],clip=clip[1])
                prior=original_diag[batch_index]
                if any(cr[k]!=prior['member_'+k] for k in ('tp','fp','fn')):raise AssertionError('original matched-member counts not reproduced')
                if cr['no_group_slots']!=prior['no_group_slots']:raise AssertionError('original no-group count changed')
                actors.extend(actor_rows);groups.extend(group_rows);clip_rows.append(cr)
                raw.write(json.dumps(dict(clip=clip,frame_ids=info['fid'],annotation_keyframe=info['key_frame'],
                    probability=p.tolist(),membership_logits=l.tolist(),roi_response=r.tolist(),activity_logits=activity.tolist(),
                    boxes=b.tolist(),frame_valid=frame_valid.tolist(),input_track_valid=valid[0].cpu().tolist(),
                    real_actor=real.tolist(),target_members=truth.tolist(),target_activities=labels.tolist(),
                    training_matcher_assigned_slots=assignment.tolist()),allow_nan=False)+'\n')
            if batch_index%100==0:print('DIAG_PROGRESS',batch_index+1,'/',len(indices),'seconds',round(time.monotonic()-started,1),flush=True)
        raw.close()
        json_write(out/'official_gt_consistency.json',audit_group_targets(source/'groundtruth_val.txt',out/'clip_outputs.jsonl'))
        expected={tuple(clip) for clip in selected}
        lines=(source/'prediction_export_val.txt').read_bytes().splitlines(keepends=True)
        source_subset=b''.join(line for line in lines if tuple(map(int,line.split()[:2])) in expected)
        import hashlib
        expected_hash=hashlib.sha256(source_subset).hexdigest()
        if file_hash(prediction)!=expected_hash:raise AssertionError('frozen predictions differ from original export')
        gt_path=out/'groundtruth_val.txt'
        gt_lines=(source/'groundtruth_val.txt').read_bytes().splitlines(keepends=True)
        gt_path.write_bytes(b''.join(line for line in gt_lines if tuple(map(int,line.split()[:2])) in expected))
        with gt_path.open() as gt,Path(args.labelmap).open() as labelmap,prediction.open() as pred:
            metrics=GAD_Evaluation(argparse.Namespace(eval_type=args.eval_type,groundtruth=gt,labelmap=labelmap)).evaluate(pred)
        metrics.update(clip_count=len(indices),prediction_hash=file_hash(prediction))
        json_write(out/'metrics_val.json',metrics)
        state_after=tensor_hash(core,common=False)
        verification=dict(prediction_hash_matches_source_subset=True,prediction_hash=metrics['prediction_hash'],
            full_source_val=len(indices)==len(val_set),original_matched_counts_reproduced=True,
            model_state_unchanged=state_after==record['source_model_state_hash'],
            checkpoint_file_unchanged=file_hash(cli.checkpoint)==checkpoint_hash,
            original_tracked_sources_unchanged=True,data_split_unchanged=True,
            optimizer_updates=0,threshold_calibration_performed=False,gt_only_after_forward=True)
        if not verification['model_state_unchanged'] or not verification['checkpoint_file_unchanged']:raise AssertionError('frozen state changed')
        summary=summarize(actors,groups,clip_rows);summary['unavailable_annotated_tracks']=unavailable
        json_write(out/'summary.json',summary);json_write(out/'verification.json',verification)
        write_csv(out/'actors.csv',actors);write_csv(out/'groups.csv',groups);write_csv(out/'clips.csv',clip_rows)
        for name,field in (('per_class','gt_activity'),('per_video','video')):
            rows=[]
            for key in sorted(set(r[field] for r in groups)):
                subset=[r for r in groups if r[field]==key]
                rows.append(dict(key=key,gt_groups=len(subset),exact_set_any_activity=sum(r['exact_set_any_activity'] for r in subset),
                    exact_set_correct_activity=sum(r['exact_set_correct_activity'] for r in subset),
                    split=sum(r['split_over_accepted_slots'] for r in subset),merge=sum(r['merged_with_other_gt_group'] for r in subset),
                    mean_best_set_iou=float(np.mean([r['best_decoded_set_iou'] for r in subset]))))
            write_csv(out/(name+'.csv'),rows)
        make_report(out,summary,record,verification)
        record.update(status='completed',group_map_1_0=float(metrics['group_mAP_1.0']),group_map_0_5=float(metrics['group_mAP_0.5']),
            outlier_miou=float(metrics['outlier_mIoU']),diagnostic_clips=len(indices),failure_reason=None)
        print('DIAG_SUMMARY',json.dumps({k:summary[k] for k in ('clips','gt_member_actors','member_error_counts','group_error_counts','group_topology_flags','matched_membership')},allow_nan=False),flush=True)
    except BaseException as error:
        record.update(status='failed',failure_reason=type(error).__name__+': '+str(error))
        (out/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
        raise
    finally:
        record['wall_time']=time.monotonic()-started
        json_write(out/'run_record.json',record)
        print('DIAG_EXIT',record['status'],record.get('failure_reason'),flush=True)


if __name__=='__main__':
    main()
