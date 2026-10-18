import sys,json,argparse
from pathlib import Path
root=Path('/mzy/copyfrom_aixi_gadfortest');sys.path.insert(0,str(root))
import torch
from models import build_model
from util.sam_person import file_hash,atomic_json,checkpoint_state
c=json.loads((root/'configs/sam_person/pilot_resolved.yaml').read_text())
checkpoint=torch.load(c['checkpoint'],map_location='cpu')
state,key_mapping=checkpoint_state(checkpoint)
a=argparse.Namespace(**json.loads((root/'configs/heatmap/b0.json').read_text()));model,criterion=build_model(a)
compat=model.load_state_dict(state,strict=True)
meta=dict(checkpoint=c['checkpoint'],sha256=file_hash(c['checkpoint']),top_level_keys=list(checkpoint),state_keys=len(state),checkpoint_key_mapping=key_mapping,epoch=checkpoint.get('epoch'),best_metric=checkpoint.get('best_metric'),strict_model_compatibility=str(compat),checkpoint_args=vars(checkpoint['args']) if isinstance(checkpoint.get('args'),argparse.Namespace) else checkpoint.get('args'),historical_metric_not_current_reevaluation=True)
atomic_json(root/'outputs/sam_person/20261018_v1/baseline_checkpoint_snapshot.json',meta)
print(json.dumps({k:v for k,v in meta.items() if k!='checkpoint_args'},indent=2))
