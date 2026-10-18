import sys,json,argparse,collections
from pathlib import Path
root=Path('/mzy/copyfrom_aixi_gadfortest');sys.path.insert(0,str(root))
from evaluation.cafe_eval import GAD_Evaluation
from util.sam_person import atomic_json
out=root/'outputs/sam_person/20261018_v1';result={}
for variant in ('B0','S2F'):
 run=out/'paired_stable'/variant/'pilot';original=run/'prediction_val.txt';lines=original.read_bytes().splitlines(keepends=True);blocks=collections.OrderedDict()
 for line in lines:
  parts=line.split();key=(parts[0],parts[1]);blocks.setdefault(key,[]).append(line)
 reversed_lines=[line for key in reversed(blocks) for line in blocks[key]]
 assert collections.Counter(lines)==collections.Counter(reversed_lines)
 target=out/(variant+'_prediction_reverse_clip_order_probe.txt');target.write_bytes(b''.join(reversed_lines))
 scores={}
 for order,path in [('original',original),('reversed_clip_order',target)]:
  with open(run/'groundtruth_val.txt') as gt,open(root/'label_map/group_action_list.pbtxt') as labels,open(path) as pred:
   metrics=GAD_Evaluation(argparse.Namespace(groundtruth=gt,labelmap=labels,eval_type='gt_base')).evaluate(pred)
  scores[order]={k:metrics[k] for k in ('group_mAP_1.0','group_mAP_0.5','outlier_mIoU')}
 result[variant]=scores
result['scope']='CPU probe only; same exported rows, clip order reversed; primary evaluation files unchanged. make_groups stores singleton score sets and group_mAP_eval sorts those sets directly.'
atomic_json(out/'evaluator_order_probe.json',result);print(json.dumps(result,indent=2))
