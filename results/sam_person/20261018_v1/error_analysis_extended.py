import sys,json
from pathlib import Path
root=Path('/mzy/copyfrom_aixi_gadfortest');sys.path.insert(0,str(root))
from evaluation.cafe_eval import read_text_file,make_groups
from util.sam_person import atomic_json
out=root/'outputs/sam_person/20261018_v1'
def groups(path,mode):
 with open(path) as stream:b,g,a,_,s=read_text_file(stream,'gt_base',mode)
 ids,acts,scores=make_groups(b,g,a,s);result={}
 for clip,entry in ids.items():
  result[clip]=[dict(id=int(gid),members=set(m),activity=next(iter(acts[clip][0][gid])),confidence=next(iter(scores[clip][0][gid]))) for gid,m in entry[0].items() if len(m)>=2 and len(acts[clip][0][gid])==1 and 7 not in acts[clip][0][gid]]
 return result
result={}
for variant in ('B0','S2F'):
 run=out/'paired_extended'/variant/'pilot';gt=groups(run/'groundtruth_val.txt','gt');pred=groups(run/'prediction_val.txt','pred')
 rows=[];pred_rows=[];splits=merges=0
 for clip,truth in gt.items():
  guesses=pred.get(clip,[])
  for g in truth:
   intersect=[p for p in guesses if g['members']&p['members']];splits+=int(len(intersect)>=2)
   same=[p for p in guesses if p['activity']==g['activity']]
   if same:
    best=max(same,key=lambda p:len(g['members']&p['members'])/len(g['members']|p['members']))
    iou=len(g['members']&best['members'])/len(g['members']|best['members']);missing=len(g['members']-best['members']);extra=len(best['members']-g['members'])
   else:iou=0.;missing=len(g['members']);extra=0
   rows.append(dict(clip=clip,gt_group=g['id'],activity=g['activity'],best_same_activity_member_iou=iou,missing=missing,extra=extra,gt_members=sorted(g['members'])))
  for p in guesses:
   intersect=[g for g in truth if p['members']&g['members']];merges+=int(len(intersect)>=2)
   compatible=[g for g in truth if g['activity']==p['activity']]
   best=max((len(g['members']&p['members'])/len(g['members']|p['members']) for g in compatible),default=0.)
   pred_rows.append(dict(clip=clip,pred_group=p['id'],activity=p['activity'],members=sorted(p['members']),confidence=p['confidence'],best_same_activity_iou=best))
 result[variant]=dict(scope='16 VAL clips; unchanged evaluator parsing; overlap diagnostics, not replacement AP matching; no thresholds selected',gt_groups=len(rows),pred_groups=len(pred_rows),split_overlap_events=splits,merge_overlap_events=merges,exact_compatible_gt_sets=sum(r['best_same_activity_member_iou']==1 for r in rows),high_confidence_incompatible_predictions=[r for r in pred_rows if r['confidence']>=.9 and r['best_same_activity_iou']<.5],gt_group_details=rows,pred_group_details=pred_rows)
atomic_json(out/'error_analysis_extended.json',result)
print(json.dumps({v:{k:x for k,x in r.items() if k not in ('gt_group_details','pred_group_details')} for v,r in result.items()},indent=2))
