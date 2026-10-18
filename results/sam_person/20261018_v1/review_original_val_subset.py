import json,sys,argparse
from pathlib import Path
root=Path('/mzy/copyfrom_aixi_gadfortest');sys.path.insert(0,str(root))
from evaluation.cafe_eval import GAD_Evaluation
from util.sam_person import atomic_json
out=root/'outputs/sam_person/20261018_v1';keys={tuple(r['clip']) for r in json.loads((out/'paired_stable/B0/input_manifest.json').read_text())['val']};result={}
for v in ('B0','S2F'):
 p=out/'paired_extended'/v/'pilot';target=p/'prediction_original8_val.txt'
 lines=(p/'prediction_val.txt').read_bytes().splitlines(keepends=True)
 target.write_bytes(b''.join(line for line in lines if tuple(map(int,line.split()[:2])) in keys))
 with open(out/'paired_stable'/v/'pilot/groundtruth_val.txt') as gt,open(root/'label_map/group_action_list.pbtxt') as lm,open(target) as pred:
  result[v]=GAD_Evaluation(argparse.Namespace(groundtruth=gt,labelmap=lm,eval_type='gt_base')).evaluate(pred)
result['scope']='fixed step300 evaluated on the original fixed8 VAL clips by slicing saved predictions; no new model or validation selection; no direct comparison across 8/16-cohort scores'
atomic_json(out/'extended_on_original8_val.json',result)
print(json.dumps({v:{k:r[k] for k in ('group_mAP_1.0','group_mAP_0.5','outlier_mIoU')} for v,r in result.items() if v in ('B0','S2F')},indent=2))
