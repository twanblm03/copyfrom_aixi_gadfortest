"""CPU-only A/B diagnosis of saved H1-Sup exports; never trains or changes decoding.

Crowding subsets use only frozen GT geometry. Set matching below ignores activity
and scores, and is a descriptive one-to-one IoU assignment, NOT official AP.
"""
import argparse
from collections import Counter, defaultdict
import difflib
import json
from pathlib import Path
import shlex
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from evaluation.cafe_eval import GAD_Evaluation
from tools.analyze_heatmap_saved_sets import export_rows, decoded_from_export, summary
from tools.analyze_heatmap_failure_margins import write_csv
from tools.run_heatmap_research import file_hash, json_write

METRICS = ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU')
CLASSES = ('Queueing', 'Ordering', 'Eating/Drinking', 'Working/Studying', 'Fighting', 'TakingSelfie')


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def geometry(raw):
    """Prespecified: any sampled-frame cross-group/group-individual IoU >= .3."""
    boxes = np.asarray(raw['boxes'], dtype=np.float32)
    valid = np.asarray(raw['input_track_valid'], dtype=bool)
    members = np.asarray(raw['target_members'], dtype=bool)
    grouped = members.any(0)
    labels = np.full(boxes.shape[1], -1, dtype=int)
    for g, mask in enumerate(members):
        labels[mask] = g
    pairs = valid[:, :, None] & valid[:, None, :] & ~np.eye(len(labels), dtype=bool)[None]
    relevant = pairs & (labels[:, None] != labels[None, :])[None] & (grouped[:, None] | grouped[None, :])[None]
    centers = boxes[..., :2] * np.array([40, 23], dtype=np.float32)
    distances = np.sqrt(((centers[:, :, None] - centers[:, None, :])**2).sum(-1))
    lo, hi = boxes[..., :2] - boxes[..., 2:]/2, boxes[..., :2] + boxes[..., 2:]/2
    inter = np.maximum(np.minimum(hi[:, :, None], hi[:, None, :]) -
                       np.maximum(lo[:, :, None], lo[:, None, :]), 0).prod(-1)
    area = boxes[..., 2:].prod(-1)
    iou = inter / np.maximum(area[:, :, None] + area[:, None, :] - inter, 1e-12)
    distance = float(distances[relevant].min()) if relevant.any() else None
    bucket = 'no_relevant_pair' if distance is None else '<1' if distance < 1 else '1-2' if distance < 2 else '>=2'
    return dict(crowded_iou_0_3=bool(((iou >= .3) & relevant).any()), distance_bucket=bucket,
                min_cross_group_or_individual_grid_distance=distance)


def membership_counts(truth, predicted):
    """Disjoint actor sets; unmatched predicted groups count FP, unmatched GT FN.

    Cross-group errors count false-positive actor assignments from another GT
    group, divided by GT-member actors in matched predicted groups. Unmatched
    predictions have no assigned GT identity, so are reported separately.
    """
    truth, predicted = [set(s) for s in truth], [set(s) for s in predicted]
    require(sum(map(len, truth)) == len(set().union(*truth)), 'GT groups overlap')
    require(sum(map(len, predicted)) == len(set().union(*predicted)), 'predicted groups overlap')
    grouped = set().union(*truth)
    matrix = np.array([[len(t & p)/max(len(t | p), 1) for p in predicted] for t in truth], dtype=float)
    matches = []
    if truth and predicted:
        gi, pi = linear_sum_assignment(-matrix)
        matches = [(int(g), int(p)) for g, p in zip(gi, pi) if matrix[g, p] > 0]
    used = {p for g, p in matches}
    tp = sum(len(truth[g] & predicted[p]) for g, p in matches)
    ntruth, npred = sum(map(len, truth)), sum(map(len, predicted))
    wrong = sum(len((predicted[p] - truth[g]) & grouped) for g, p in matches)
    return dict(tp=tp, fp=npred-tp, fn=ntruth-tp, gt_members=ntruth, predicted_members=npred,
                gt_groups=len(truth), predicted_groups=len(predicted), positive_matched_groups=len(matches),
                group_iou_sum=sum(float(matrix[g, p]) for g, p in matches),
                cross_group_wrong_assignments=wrong,
                matched_predicted_gt_member_assignments=sum(len(predicted[p] & grouped) for g, p in matches),
                unmatched_predicted_gt_member_assignments=sum(len(p & grouped) for i, p in enumerate(predicted) if i not in used),
                individual_false_assignments=sum(len(p-grouped) for p in predicted))


def aggregate_counts(rows):
    counts = {k:sum(r[k] for r in rows) for k in rows[0]} if rows else {}
    ratio = lambda a, b: counts.get(a, 0)/counts[b] if counts.get(b, 0) else None
    counts.update(member_precision=ratio('tp', 'predicted_members'), member_recall=ratio('tp', 'gt_members'),
                  member_f1=2*counts['tp']/(2*counts['tp']+counts['fp']+counts['fn']) if rows else None,
                  mean_group_iou_over_gt=ratio('group_iou_sum', 'gt_groups'),
                  cross_group_misassignment_rate=ratio('cross_group_wrong_assignments', 'matched_predicted_gt_member_assignments'))
    return counts


def predicted_sets(actors):
    groups = defaultdict(list)
    for actor in actors:
        if actor['category'] in ('accepted_detected_group', 'individual_false_group_member'):
            groups[actor['decoded_slot']].append(actor['actor_index'])
    return [groups[k] for k in sorted(groups)]


def filter_export(source, destination, clips):
    with source.open() as inp, destination.open('x') as out:
        for line in inp:
            if tuple(map(int, line.split()[:2])) in clips:
                out.write(line)


def replay(gt_path, prediction, labelmap, eval_type):
    with gt_path.open() as gt, prediction.open() as pred, labelmap.open() as labels:
        return GAD_Evaluation(argparse.Namespace(eval_type=eval_type, groundtruth=gt, labelmap=labels)).evaluate(pred)


def audit_runs(paths, inputs, out):
    configs = [read(p/'resolved_config.json') for p in paths]
    records = [read(p/'run_record.json') for p in paths]
    manifests = [read(p/'source_manifest.json') for p in paths]
    model_differences = {k:[c.get(k) for c in configs] for k in configs[0].keys() | configs[1].keys()
                         if k != 'runtime' and configs[0].get(k) != configs[1].get(k)}
    require(model_differences == {'heatmap_target_mode':[None, 'nearest_center']}, 'unexpected configuration change')
    require(all(c['split'] == 'place' and c['val_mode'] and c['heatmap_variant'] == 'h1_sup' for c in configs), 'wrong task')
    for name, key in (('data_manifest.json', 'data_manifest_hash'), ('split.json', 'split_hash')):
        hashes = [file_hash(p/name) for p in (*paths, inputs)]
        require(len(set(hashes)) == 1, 'frozen data/split mismatch')
        require(all(r[key] == h for r, h in zip(records, hashes)), 'data record hash mismatch')
    require(file_hash(paths[0]/'groundtruth_val.txt') == file_hash(paths[1]/'groundtruth_val.txt'), 'official GT changed')
    history = []; runs = []
    for name, path, record, config, manifest in zip(('A', 'B'), paths, records, configs, manifests):
        require(record['status'] == 'completed' and record['epochs_or_steps'] == 30, 'run incomplete')
        for filename, key in (('source_manifest.json', 'source_manifest_hash'), ('resolved_config.json', 'config_hash')):
            require(file_hash(path/filename) == record[key], 'record hash mismatch: '+filename)
        for filename, expected in manifest.items():
            require(file_hash(path/'source_snapshot'/filename) == expected, 'source mismatch: '+filename)
        require(file_hash(ROOT/'evaluation/cafe_eval.py') == record['evaluator_hash'], 'evaluator mismatch')
        saved = read(path/'metrics_val.json')
        require(file_hash(path/'prediction_export_val.txt') == saved['prediction_hash'], 'prediction hash mismatch')
        epochs = [dict(run=name, epoch=e, **read(path/('metrics_epoch%d.json'%e))) for e in range(1, 31)]
        best = max(epochs, key=lambda r:r[METRICS[0]])
        require(all(np.allclose(best[k], saved[k], rtol=0, atol=1e-12) for k in METRICS), 'best metrics mixed')
        checkpoint = path/'checkpoints/best_val.pth'
        ck = torch.load(checkpoint, map_location='cpu')
        require(ck['epoch'] == best['epoch'], 'checkpoint epoch differs')
        require(all(np.allclose(ck['metrics'][k], saved[k], rtol=0, atol=1e-12) for k in METRICS), 'checkpoint metrics differ')
        del ck
        steps = [json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
        require([s['step'] for s in steps] == list(range(1, 12061)), 'training steps not contiguous')
        require(Counter(s['epoch'] for s in steps) == Counter({e:402 for e in range(30)}), 'epoch budget differs')
        env = read(path/'environment.json')
        runs.append(dict(name=name, path=str(path), best_epoch=best['epoch'], checkpoint_sha256=file_hash(checkpoint),
                         metrics=saved, runtime=config['runtime'], environment=env,
                         lr_trace=[s['lr'] for s in steps],
                         sample_order_hash=file_hash(path/'sample_order.json'),
                         initial_full_state_hash=record.get('initial_full_state_hash')))
        history.extend(epochs)
    orders = [read(p/'sample_order.json') for p in paths]
    require(all(len(order) == 12060 for order in orders), 'incomplete sample order')
    first = next((i+1 for i, (a, b) in enumerate(zip(*orders)) if a != b), None)
    clip_first = next((i+1 for i, (a, b) in enumerate(zip(*orders))
                       if [s[:2] for s in a] != [s[:2] for s in b]), None)
    common_diff = [k for k in manifests[0] if k in manifests[1] and manifests[0][k] != manifests[1][k]]
    for filename in common_diff:
        a, b = [p/'source_snapshot'/filename for p in paths]
        diff = ''.join(difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True), fromfile='A/'+filename, tofile='B/'+filename))
        dest = out/'source_diffs'/(filename+'.diff'); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(diff)
    same_init = {k:records[0].get(k) == records[1].get(k) for k in ('init_hash', 'initial_rng_hash')}
    lr_equal = runs[0].pop('lr_trace') == runs[1].pop('lr_trace')
    result = dict(runs=runs, config_difference_excluding_runtime=model_differences,
                  source_common_changed_files=common_diff, same_recorded_initialization=same_init,
                  same_lr_every_step=lr_equal, first_different_training_sample_step=first,
                  first_different_training_clip_order_step=clip_first,
                  strict_training_pair=False,
                  caveat='A6000/host transport/persistent workers vs 3090/native DataParallel/nonpersistent workers; '
                         'sample traces diverge; A lacks full-state initialization hash. Same-sample evaluation is valid, '
                         'but target-only causal attribution and statistical significance are not established.')
    write_csv(out/'epoch_metrics.csv', [{k:r[k] for k in ('run', 'epoch', *METRICS)} for r in history])
    json_write(out/'audit.json', result)
    return result, configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--input-diagnostic', required=True)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    paths = [Path(cli.baseline).resolve(), Path(cli.candidate).resolve()]
    inputs, out = Path(cli.input_diagnostic).resolve(), Path(cli.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'command.txt').write_text(shlex.join([sys.executable]+sys.argv)+'\n')
    torch.set_num_threads(2)
    def log(message):
        print(message, flush=True)
        with (out/'analysis.log').open('a') as stream:
            stream.write(message+'\n')
    try:
        snapshots = ('tools/analyze_heatmap_crowding_control.py', 'tools/analyze_heatmap_saved_sets.py',
                     'tools/analyze_heatmap_failure_margins.py', 'tools/run_heatmap_research.py',
                     'evaluation/cafe_eval.py', 'h1_center_crowding_control_20261016.md',
                     'tests/heatmap/test_crowding_analysis.py')
        for filename in snapshots:
            dest = out/'source_snapshot'/filename; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/filename, dest)
        json_write(out/'source_manifest.json', {f:file_hash(ROOT/f) for f in snapshots})
        audit, configs = audit_runs(paths, inputs, out)
        log('AUDIT_OK best_epochs='+str([r['best_epoch'] for r in audit['runs']])+
            ' first_different_training_sample_step='+str(audit['first_different_training_sample_step']))
        # Read ONLY input/GT fields from the old H2 diagnostic. No H2 scores or slots used.
        keys = ('clip', 'frame_ids', 'boxes', 'input_track_valid', 'target_members', 'target_activities')
        raw = [{k:r[k] for k in keys} for r in map(json.loads, (inputs/'clip_outputs.jsonl').read_text().splitlines())]
        clips = {tuple(r['clip']) for r in raw}
        exports = [export_rows(p/'prediction_export_val.txt') for p in paths]
        require(len(raw) == len(clips) == 1386 and all(set(e) == clips for e in exports), 'full VAL coverage differs')
        geometry_rows = {tuple(r['clip']):dict(video=r['clip'][0], clip=r['clip'][1], **geometry(r)) for r in raw}
        write_csv(out/'geometry.csv', list(geometry_rows.values()))
        bad = {tuple(r['clip']) for r in read(inputs/'official_gt_consistency.json')['mismatches']}
        subsets = dict(full_val=clips,
                       crowded={c for c, g in geometry_rows.items() if g['crowded_iou_0_3']},
                       noncrowded={c for c, g in geometry_rows.items() if not g['crowded_iou_0_3']},
                       exclude_two_gt_mismatch_clips=clips-bad)
        for bucket in ('<1', '1-2', '>=2', 'no_relevant_pair'):
            subsets['distance_'+bucket] = {c for c, g in geometry_rows.items() if g['distance_bucket'] == bucket}
        results = {}; group_tables = {}; actor_tables = {}; membership_tables = {}
        for name, path, export in zip(('A', 'B'), paths, exports):
            actors, groups, counts = [], [], []
            for index, row in enumerate(raw):
                clip = tuple(row['clip'])
                a, g = decoded_from_export(row, export[clip])
                actors.extend(a); groups.extend(g)
                truth = [s['gt_actor_indices'] for s in g]
                counts.append(dict(video=clip[0], clip=clip[1], **membership_counts(truth, predicted_sets(a))))
                if (index+1) % 300 == 0:
                    log(name+' verified_frames_boxes_and_decoded_clips='+str(index+1))
            directory = out/name; directory.mkdir()
            write_csv(directory/'actors.csv', actors); write_csv(directory/'groups.csv', groups)
            write_csv(directory/'clip_membership_counts.csv', counts)
            results[name] = {}
            group_tables[name], actor_tables[name], membership_tables[name] = groups, actors, counts
            for subset, selected in subsets.items():
                select = lambda rows:[r for r in rows if (r['video'], r['clip']) in selected]
                result = summary(select(actors), select(groups))
                result.update(clips=len(selected), membership=aggregate_counts([
                    {k:v for k, v in r.items() if k not in ('video', 'clip')} for r in select(counts)]))
                if selected:
                    folder = directory/('subset_'+str(list(subsets).index(subset))); folder.mkdir()
                    pred, gt = folder/'predictions.txt', folder/'groundtruth.txt'
                    filter_export(path/'prediction_export_val.txt', pred, selected)
                    filter_export(path/'groundtruth_val.txt', gt, selected)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        result['official_metrics'] = replay(gt, pred, ROOT/configs[0]['labelmap'], configs[0]['eval_type'])
                    result['official_subset_note'] = 'Original evaluator; zero-positive classes retain its convention. Subset AP is not additive.'
                    if subset == 'full_val':
                        saved = read(path/'metrics_val.json')
                        require(all(np.allclose(result['official_metrics'][k], saved[k], rtol=0, atol=1e-12)
                                    for k in result['official_metrics']), 'full CPU replay differs')
                results[name][subset] = result
                log(name+' subset='+subset+' clips='+str(len(selected))+' exact_correct='+str(
                    result['group_topology_flags']['exact_set_correct_activity']))
            json_write(directory/'summary.json', results[name])
        transitions = []
        require([(r['video'], r['clip'], r['gt_group']) for r in group_tables['A']] ==
                [(r['video'], r['clip'], r['gt_group']) for r in group_tables['B']], 'GT group row pairing differs')
        for a, b in zip(group_tables['A'], group_tables['B']):
            clip = a['video'], a['clip']
            transitions.append(dict(video=clip[0], clip=clip[1], gt_group=a['gt_group'], gt_activity=a['gt_activity'],
                                    crowded=geometry_rows[clip]['crowded_iou_0_3'], distance_bucket=geometry_rows[clip]['distance_bucket'],
                                    category_a=a['category'], category_b=b['category'],
                                    set_iou_a=a['best_decoded_set_iou'], set_iou_b=b['best_decoded_set_iou']))
        write_csv(out/'paired_group_transitions.csv', transitions)
        paired = {}
        for subset, selected in subsets.items():
            rows = [r for r in transitions if (r['video'], r['clip']) in selected]
            correct = 'exact_set_correct_activity'
            paired[subset] = dict(improved_to_exact_correct=sum(r['category_a'] != correct and r['category_b'] == correct for r in rows),
                                  regressed_from_exact_correct=sum(r['category_a'] == correct and r['category_b'] != correct for r in rows),
                                  transition_counts=[dict(category_a=a, category_b=b, count=n) for (a, b), n in
                                                     sorted(Counter((r['category_a'], r['category_b']) for r in rows).items())])
        # Fixed, deterministic examples, not a performance-selected evaluation subset.
        chosen = []
        for tag, predicate in (
                ('crowded_improved', lambda r:r['crowded'] and r['category_a'] != 'exact_set_correct_activity' and r['category_b'] == 'exact_set_correct_activity'),
                ('crowded_regressed', lambda r:r['crowded'] and r['category_a'] == 'exact_set_correct_activity' and r['category_b'] != 'exact_set_correct_activity')):
            for row in [r for r in transitions if predicate(r)][:3]:
                chosen.append(dict(tag=tag, **row))
        raw_by_clip = {tuple(r['clip']):r for r in raw}
        for case in chosen:
            clip = case['video'], case['clip']; row = raw_by_clip[clip]
            ann_path = ROOT/configs[0]['data_path']/'cafe'/str(clip[0])/str(clip[1])/'ann.json'
            ids = [f['id'] for f in read(ann_path)['figures']]
            case.update(actor_ids=ids, frame_ids=row['frame_ids'], boxes=row['boxes'], gt_members=row['target_members'],
                        gt_activities=row['target_activities'], annotation_hash=file_hash(ann_path),
                        prediction_a=[g for g in group_tables['A'] if (g['video'], g['clip']) == clip],
                        prediction_b=[g for g in group_tables['B'] if (g['video'], g['clip']) == clip])
        json_write(out/'cases.json', chosen)
        json_write(out/'paired_summary.json', paired)
        json_write(out/'comparison.json', results)
        write_csv(out/'per_class_ap.csv', [dict(activity=c, **{
            name+'_'+key:float(results[name]['full_val']['official_metrics'][key][i])
            for name in ('A', 'B') for key in ('group_APs_1.0', 'group_APs_0.5')}) for i, c in enumerate(CLASSES)])
        json_write(out/'verification.json', dict(status='completed', full_val_clips=1386,
            all_sampled_frames_and_actor_box_rows_verified_both_runs=True,
            official_metrics_cpu_replayed=True, data_split_gt_unchanged=True,
            input_record_sha256=file_hash(inputs/'clip_outputs.jsonl'),
            original_files_overwritten=False, model_forwards=0, optimizer_updates=0,
            h2_predictions_used=False, cuda_initialized=torch.cuda.is_initialized(),
            description='GT geometry strata; final decoded-set errors and one-to-one membership; no raw slot margin reconstruction.',
            limitations=['Not a strict training pair; one seed only.',
                         'Two pre-existing annotation/official-GT inconsistencies kept in main metrics; exclusion sensitivity reported.',
                         'Full-VAL raw heatmaps/embedding probabilities are not in saved exports; need separate model forwards for mechanism confirmation.']))
        log('COMPLETED '+str(out))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
