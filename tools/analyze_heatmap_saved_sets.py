"""CPU-only decoded-set audit from saved exports and verified frozen input records.

No probabilities/margins are reconstructed. Source annotation targets come from
the previous frozen diagnostic, whose data/frames/box order must match both runs.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_heatmap_failure_margins import write_csv
from tools.run_heatmap_research import file_hash,json_write


def export_rows(path):
    result=defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        fields=line.split()
        clip=tuple(map(int,fields[:2]));frame=int(fields[2])
        result[clip].setdefault(frame,[]).append(fields)
    return result


def decoded_groups(members,activities,valid,present,group_ids,labels,clip):
    truth=np.asarray(members,dtype=bool)&valid[None]
    visible=valid&present
    active=(labels>=1)&(labels<=6)&(group_ids>=0)&visible
    sets={int(s):(group_ids==s)&active for s in np.unique(group_ids[active])}
    classes={s:int(labels[mask][0]) for s,mask in sets.items()}
    if any(len(set(labels[mask].tolist()))!=1 for mask in sets.values()):
        raise ValueError('one predicted slot has inconsistent activity')
    detected={s:mask for s,mask in sets.items() if mask.sum()>=2}
    gt_actor_group=np.full(len(valid),-1,dtype=int)
    for g,mask in enumerate(members):gt_actor_group[np.asarray(mask,dtype=bool)]=g
    groups=[]
    for g,target in enumerate(truth):
        target=target&present
        candidates=[(s,float((target&mask).sum())/max(int((target|mask).sum()),1)) for s,mask in detected.items()]
        best_slot,best_iou=max(candidates,key=lambda x:(x[1],-x[0])) if candidates else (-1,0.)
        exact=[s for s,mask in detected.items() if target.sum()>=2 and np.array_equal(mask,target)]
        correct=[s for s in exact if classes[s]==int(activities[g])+1]
        if correct:category='exact_set_correct_activity'
        elif exact:category='exact_set_wrong_activity'
        elif best_iou<=0:category='no_overlapping_detected_group'
        else:
            pred=detected[best_slot]
            missing,extra=int((target&~pred).sum()),int((~target&pred).sum())
            category='missing_members_only' if missing and not extra else 'extra_members_only' if extra and not missing else 'missing_and_extra_members'
        occupied=[s for s,mask in sets.items() if (target&mask).any()]
        touching=[s for s in occupied if s in detected]
        merges=[s for s in touching if sum(bool((other&detected[s]).any()) for other in truth)>=2]
        individuals=[s for s in touching if ((gt_actor_group<0)&detected[s]).any()]
        groups.append(dict(video=clip[0],clip=clip[1],gt_group=g,gt_activity=int(activities[g])+1,
            category=category,exact_set_any_activity=bool(exact),exact_set_correct_activity=bool(correct),
            best_decoded_set_iou=best_iou,best_decoded_slot=best_slot,
            split_over_accepted_slots=len(occupied)>=2,split_over_detected_groups=len(touching)>=2,
            merged_with_other_gt_group=bool(merges),contaminated_by_individual=bool(individuals),
            gt_actor_indices=np.where(target)[0].tolist(),
            predicted_actor_indices=np.where(detected[best_slot])[0].tolist() if best_slot>=0 else []))
    actors=[]
    for i in np.where(valid)[0]:
        group=int(gt_actor_group[i]);label=int(labels[i]);slot=int(group_ids[i])
        if not present[i]:category='not_present_first_export_frame'
        elif label==7:category='rejected_threshold' if slot<0 else 'rejected_no_group'
        else:category='accepted_detected_group' if slot in detected else 'accepted_singleton'
        if group<0:
            category={'rejected_threshold':'individual_rejected_threshold','rejected_no_group':'individual_rejected_no_group',
                'accepted_detected_group':'individual_false_group_member','accepted_singleton':'individual_singleton_prediction'}.get(category,category)
        actors.append(dict(video=clip[0],clip=clip[1],actor_index=int(i),gt_group=group,
            category=category,decoded_slot=slot if active[i] else -1,activity=label))
    return actors,groups


def summary(actors,groups):
    return dict(valid_actors=len(actors),gt_member_actors=sum(r['gt_group']>=0 for r in actors),
        gt_individual_actors=sum(r['gt_group']<0 for r in actors),gt_groups=len(groups),
        individual_error_counts=dict(Counter(r['category'] for r in actors if r['gt_group']<0)),
        gt_member_decoded_counts=dict(Counter(r['category'] for r in actors if r['gt_group']>=0)),
        group_error_counts=dict(Counter(r['category'] for r in groups)),
        group_topology_flags={k:sum(r[k] for r in groups) for k in ('split_over_accepted_slots','split_over_detected_groups',
            'merged_with_other_gt_group','contaminated_by_individual','exact_set_any_activity','exact_set_correct_activity')})


def decoded_from_export(raw,rows):
    boxes=torch.tensor(raw['boxes'],dtype=torch.float32)
    gids=np.full(boxes.shape[1],-1,dtype=int);labels=np.full(boxes.shape[1],7,dtype=int)
    if set(rows)!=set(raw['frame_ids']):raise ValueError('sampled frames changed')
    for t,fid in enumerate(raw['frame_ids']):
        corners=torch.cat((boxes[t,:,:2]-boxes[t,:,2:]/2,boxes[t,:,:2]+boxes[t,:,2:]/2),-1)
        corners=corners*torch.tensor([1280,720,1280,720])
        ids=np.where((corners.numpy()!=0).any(-1))[0]
        if len(ids)!=len(rows[fid]):raise ValueError('actor-row coverage changed')
        for i,fields in zip(ids,rows[fid]):
            if [int(v) for v in corners[i]]!=list(map(int,fields[3:7])):
                raise ValueError('box order/geometry changed')
            if t==0:gids[i],labels[i]=int(fields[7]),int(fields[8])
    valid=np.asarray(raw['input_track_valid'],dtype=bool).any(0)
    present=(boxes[0].numpy()!=0).any(-1)
    return decoded_groups(raw['target_members'],raw['target_activities'],valid,present,gids,labels,raw['clip'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True);parser.add_argument('--candidate',required=True)
    parser.add_argument('--input-diagnostic',required=True);parser.add_argument('--output',required=True)
    cli=parser.parse_args();torch.set_num_threads(2)
    baseline,candidate,inputs,out=(Path(getattr(cli,k)) for k in ('baseline','candidate','input_diagnostic','output'))
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'source.py')
    import shlex
    (out/'command.txt').write_text(shlex.join([sys.executable]+sys.argv)+'\n')
    records=[json.loads((p/'run_record.json').read_text()) for p in (baseline,candidate,inputs)]
    legacy_nulls=[]
    for filename,key in (('data_manifest.json','data_manifest_hash'),('split.json','split_hash')):
        actuals=[file_hash(p/filename) for p in (baseline,candidate,inputs)]
        if len(set(actuals))!=1:raise ValueError('input data/split files differ')
        for path,record,actual in zip((baseline,candidate,inputs),records,actuals):
            if record.get(key) is None:legacy_nulls.append(dict(run=str(path),field=key,verified_file_hash=actual))
            elif record[key]!=actual:raise ValueError('recorded data/split hash differs from saved file')
    if file_hash(baseline/'groundtruth_val.txt')!=file_hash(candidate/'groundtruth_val.txt'):raise ValueError('official GT changed')
    exports=[export_rows(p/'prediction_export_val.txt') for p in (baseline,candidate)]
    raw=[json.loads(line) for line in (inputs/'clip_outputs.jsonl').read_text().splitlines()]
    if len(raw)!=1386 or any(set(x)!={tuple(r['clip']) for r in raw} for x in exports):raise ValueError('full VAL coverage required')
    results={}
    mismatches={tuple(r['clip']) for r in json.loads((inputs/'official_gt_consistency.json').read_text())['mismatches']}
    for name,path,export in zip(('h2_s','h2_s_center'),(baseline,candidate),exports):
        record=json.loads((path/'run_record.json').read_text())
        if file_hash(path/'prediction_export_val.txt')!=json.loads((path/'metrics_val.json').read_text())['prediction_hash']:raise ValueError('prediction hash mismatch')
        actors=[];groups=[]
        for row in raw:
            a,g=decoded_from_export(row,export[tuple(row['clip'])]);actors.extend(a);groups.extend(g)
        result=summary(actors,groups)
        directory=out/name;directory.mkdir()
        write_csv(directory/'actors.csv',actors);write_csv(directory/'groups.csv',groups)
        json_write(directory/'summary.json',result)
        result['exclude_two_gt_mismatch_clips']=summary(
            [r for r in actors if (r['video'],r['clip']) not in mismatches],
            [r for r in groups if (r['video'],r['clip']) not in mismatches])
        results[name]=result
    prior=json.loads((inputs/'summary.json').read_text())
    for key in ('individual_error_counts','group_error_counts','group_topology_flags'):
        if results['h2_s'][key]!=prior[key]:raise AssertionError('baseline diagnostic not reproduced: '+key)
    json_write(out/'comparison.json',results)
    json_write(out/'verification.json',dict(full_val_clips=1386,baseline_group_and_individual_diagnostics_reproduced=True,
        all_sampled_frames_and_actor_box_rows_verified=True,input_data_split_gt_unchanged=True,
        old_probabilities_used_for_new_model=False,new_model_forwards=0,optimizer_updates=0,
        legacy_null_metadata_verified_from_files=legacy_nulls,
        scope='decoded sets and rejection reasons only; new-model slot margins, full-slot classes and matcher assignment unavailable'))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
