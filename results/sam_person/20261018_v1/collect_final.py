"""Collect completed artifacts only; no model forward, training, or source mutation."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path('/mzy/copyfrom_aixi_gadfortest')
OUT = ROOT / 'outputs/sam_person/20261018_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.finalizing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def export_geometry(path):
    return [line.split()[:7] for line in path.read_text().splitlines() if line.strip()]


def main():
    registry = ROOT / 'results/sam_person'
    for name in ['run_registry.csv', 'run_registry.json']:
        old = registry / name
        backup = OUT / ('before_extended_' + name)
        if not backup.exists():
            shutil.copyfile(old, backup)
    old_cost = OUT / 'before_extended_cost_summary.json'
    if not old_cost.exists():
        shutil.copyfile(OUT / 'cost_summary.json', old_cost)
    qc = read(OUT / 'qc_metrics_stable.json')
    costs = read(old_cost)
    costs['training_ledgers'] = {v: read(OUT / 'training_ledgers' / (v + '.json')) for v in ['B0', 'S2F', 'diagnostic']}
    costs['train_gpu_hours_including_evaluation'] = sum(v['seconds'] for v in costs['training_ledgers'].values()) / 3600
    costs['sam_gpu_hours_including_failed_attempts'] = read(OUT / 'generation_budget.json')['seconds'] / 3600
    costs['accounting'] = 'Cumulative allocated device-wall hours, includes failures, I/O and conservative interruption recovery gaps; not kernel-only time.'
    costs['budgets_gpu_hours'] = {'generation': 1, 'all_training_evaluation_diagnosis': 2}
    costs['within_budget'] = costs['sam_gpu_hours_including_failed_attempts'] <= 1 and costs['train_gpu_hours_including_evaluation'] <= 2
    assert costs['within_budget']
    assert all('active_since' not in v for v in costs['training_ledgers'].values())
    records = []
    checks = []
    cohorts = {}
    correction = {r['run_id']: r for r in read(OUT / 'diagnostic_corrections.json')['corrections']}
    masks = [json.loads(line) for line in (OUT / 'qc_manifest.jsonl').open()]
    assert len(masks) == 200 and sum(len(r['actors']) for r in masks) == 2054
    assert len({r['source_image_key'] for r in masks}) == 200
    assert len({r['source_image_hash'] for r in masks}) == 200
    count_by_clip = collections.Counter()
    for r in masks:
        count_by_clip[(r['video_id'], r['clip_id'])] += len(r['actors'])
    for root_name, ntrain, nval, nsteps in [('paired_stable', 16, 8, 80), ('paired_extended', 24, 16, 300)]:
        base = OUT / root_name
        pair = {v: read(base / v / 'results.json') for v in ['B0', 'S2F']}
        assert read(base / 'B0/input_manifest.json') == read(base / 'S2F/input_manifest.json')
        inputs = read(base / 'B0/input_manifest.json')
        assert len(inputs['train']) == ntrain and len(inputs['val']) == nval
        for stage in ['reference', 'overfit', 'pilot']:
            rs = [next((r for r in pair[v] if r['stage'] == stage), None) for v in ['B0', 'S2F']]
            if stage == 'overfit' and root_name == 'paired_extended':
                continue
            assert all(rs)
            keys = ['base_checkpoint_or_init_hash', 'base_checkpoint_sha256', 'code_hash', 'config_hash', 'evaluator_hash', 'input_manifest_hash', 'split_hash', 'train_clip_manifest_hash', 'val_clip_manifest_hash', 'sampled_frame_hash', 'seed', 'effective_batch', 'steps', 'checkpoint_selection_rule', 'lr_schedule']
            if stage != 'reference':
                keys.append('sample_order_hash')
            tested = {k: rs[0].get(k) is not None and rs[0].get(k) == rs[1].get(k) for k in keys}
            assert all(tested.values()), (root_name, stage, tested)
            if stage != 'overfit':
                suffix = 'reference' if stage == 'reference' else 'val'
                tested['export_clip_frame_boxes_actor_order'] = export_geometry(base / ('B0/%s/prediction_%s.txt' % (stage, suffix))) == export_geometry(base / ('S2F/%s/prediction_%s.txt' % (stage, suffix)))
                assert tested['export_clip_frame_boxes_actor_order']
            for v, r in zip(['B0', 'S2F'], rs):
                row = {k: val for k, val in r.items() if not isinstance(val, (list, dict))}
                metrics = r.get('metrics') or {}
                row.update(cohort=root_name, included_in_final_pair=True,
                    train_clips=ntrain if stage != 'overfit' else 8, val_clips=nval if stage != 'overfit' else None,
                    group_map_1_0=metrics.get('group_mAP_1.0'), group_map_0_5=metrics.get('group_mAP_0.5'),
                    outlier_miou=metrics.get('outlier_mIoU'), membership_f1=metrics.get('membership_f1'),
                    membership_precision=metrics.get('membership_precision'), membership_recall=metrics.get('membership_recall'),
                    raw_success_rate=qc['all']['raw_success_rate'], usable_mask_coverage=qc['all']['usable_coverage'],
                    feature_grid_empty_rate=0., qc_fallback_rate=qc['all']['fallback_rate'],
                    segmentation_pixel_iou=None, segmentation_dice=None,
                    coverage_scope='global frozen 200-frame QC; structural usability only',
                    sam_gpu_hours=costs['sam_gpu_hours_including_failed_attempts'],
                    sam_cost_scope='shared cache cumulative cost; do not sum this value across rows',
                    cache_bytes=costs['cache_bytes'],
                    metrics_artifact=str(base / v / stage / 'result.json'))
                if stage != 'reference':
                    steps = [json.loads(line) for line in (base / v / stage / 'steps.jsonl').open()]
                    assert len(steps) == r['steps'] and steps[-1]['step'] == r['steps']
                    assert all(math.isfinite(s['loss']) and math.isfinite(s['grad_norm']) for s in steps)
                    order = read(base / v / stage / 'sample_order.json')
                    assert len(order) == r['steps']
                    actual_actors = sum(count_by_clip[(item[0], item[1])] for batch in order for item in batch)
                    row['actual_valid_train_actor_frames'] = actual_actors
                    row['actual_sam_roi_train_actor_frames'] = actual_actors if v == 'S2F' else 0
                    row['fallback_rate'] = r['mask_counts']['fallback'] / actual_actors if v == 'S2F' else None
                    if v == 'S2F':
                        assert r['mask_counts']['valid'] == actual_actors == r['mask_counts']['used']
                        assert r['mask_counts']['fallback'] == 0
                    row['step_seconds_median'] = statistics.median(s['seconds'] for s in steps)
                    row['step_clips_per_second_median'] = 4 / row['step_seconds_median']
                    row['step_clips_per_second_aggregate'] = 4 * len(steps) / sum(s['seconds'] for s in steps)
                    row['logged_mask_counts_corrected'] = r['run_id'] in correction
                    if r['run_id'] in correction:
                        assert correction[r['run_id']]['actual_ordinary_roi_actor_frames'] == actual_actors
                if stage == 'reference' and v == 'B0':
                    assert all(x == 0 for x in r['real_gpu_ones_max_error'].values())
                records.append(row)
            checks.append({'cohort': root_name, 'stage': stage, 'checks': tested})
        pilots = {v: next(r for r in pair[v] if r['stage'] == 'pilot') for v in ['B0', 'S2F']}
        assert pilots['B0']['steps'] == pilots['S2F']['steps'] == nsteps
        vals = {v: {k: pilots[v]['metrics'][k] for k in ['group_mAP_1.0','group_mAP_0.5','outlier_mIoU','membership_precision','membership_recall','membership_f1','group_APs_1.0','group_APs_0.5']} for v in ['B0', 'S2F']}
        delta = {k: vals['S2F'][k] - vals['B0'][k] for k in ['group_mAP_1.0','group_mAP_0.5','outlier_mIoU','membership_f1']}
        err = read(OUT / ('error_analysis_extended.json' if root_name == 'paired_extended' else 'error_analysis.json'))
        errors = {}
        for v in ['B0','S2F']:
            e = err[v]
            m = pilots[v]['metrics']
            errors[v] = {k: e[k] for k in ['gt_groups','pred_groups','exact_compatible_gt_sets','split_overlap_events','merge_overlap_events']}
            errors[v].update(high_confidence_incompatible=len(e['high_confidence_incompatible_predictions']),
                matched_missing_members=sum(d['missing'] for d in m['matched_group_details']),
                matched_extra_members=sum(d['extra'] for d in m['matched_group_details']),
                matched_group_activity_errors=m['matched_group_activity_errors'],
                gt_groups_by_class=dict(collections.Counter(str(d['activity']) for d in e['gt_group_details'])))
        cohorts[root_name] = {'train_clips': ntrain, 'val_clips': nval, 'steps': nsteps, 'seed': 1, 'metrics': vals, 'delta_S2F_minus_B0': delta, 'error_diagnostics': errors}
    # Retain all failed/superseded model and generation attempts in the registry.
    for v in ['B0', 'S2F']:
        initial = OUT / 'paired' / v
        failed = initial / 'failure.json'
        if failed.exists():
            records.append({'run_id': str(initial), 'variant': v, 'status': 'failed_numerical_gate', 'included_in_final_pair': False, 'failure_reason': read(failed)['error']})
        elif (initial / 'results.json').exists():
            for r in read(initial / 'results.json'):
                row = {k: val for k,val in r.items() if not isinstance(val,(dict,list))}
                row.update(status='superseded_before_pilot', included_in_final_pair=False,
                    failure_reason='paired B0 GPU ones gate failed; original completed reference/smoke and source retained')
                records.append(row)
    generation = read(OUT / 'generation_budget.json')
    for i, session in enumerate(generation['sessions']):
        records.append({'run_id': 'generation_session_%d' % i, 'variant': 'SAM-QC', 'status': session['status'],
            'failure_reason': session.get('failure_reason'), 'frames_completed': len(session['frames']),
            'peak_memory_bytes': session.get('peak_memory_bytes'), 'included_in_final_pair': False})
    fields = sorted({k for r in records for k in r})
    with (registry / 'run_registry.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: ('null' if row.get(k) is None else row[k]) for k in fields} for row in records)
    write(registry / 'run_registry.json', records)
    write(OUT / 'paired_checks_final.json', checks)
    write(OUT / 'cost_summary_final.json', costs)
    write(OUT / 'cost_summary.json', costs)
    write(OUT / 'final_summary.json', {'cohorts':cohorts, 'costs': costs, 'qc_all':qc['all'],
        'qc_scale_buckets':qc['scale_buckets'], 'full_training_run':False,
        'decision':'No evidence of downstream benefit for current SAM-v1 box-only MaskRoI; do not start four-GPU full training.',
        'limits':['one seed; developmental TRAIN->VAL subsets and selected warm start',
            'original evaluator has set-score sorting/order sensitivity; unchanged, deterministic paired export order',
            'full current split rescan and global image-content duplicate audit incomplete',
            'no independent person pixel GT, no statistical significance or full-dataset score',
            '80/300 diagnostics use different cohort and update count; not an isolated duration ablation'],
        'pilot_updates_per_variant':380, 'overfit_updates_per_variant_accepted':40, 'superseded_smoke_updates':{'B0':0,'S2F':3}, 'all_optimizer_updates_including_superseded':{'B0':420,'S2F':423}, 'own_gpu_training_complete':True})
    print(json.dumps({'cohorts':cohorts,'sam_gpu_hours':costs['sam_gpu_hours_including_failed_attempts'],
        'training_gpu_hours':costs['train_gpu_hours_including_evaluation'],'registry_rows':len(records),'fairness_checks_passed':all(all(c['checks'].values()) for c in checks)},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
