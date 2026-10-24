import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path('/data2/mzy/GAD/CAFE_heatmap_research_20261012')
sys.path.insert(0, str(ROOT))
import torch
import torchvision
from models import build_model

torch.set_num_threads(2)
torch.manual_seed(17)
args = argparse.Namespace(**json.loads((ROOT / 'configs/heatmap/h1_sup.json').read_text()))
original_resnet18 = torchvision.models.resnet18

def offline_resnet18(*positional, **kwargs):
    kwargs['pretrained'] = False
    return original_resnet18(*positional, **kwargs)

with patch('torchvision.models.resnet18', side_effect=offline_resnet18):
    model, criterion = build_model(args)
model.eval()
images = torch.randn(1, 5, 3, 96, 160)
boxes = torch.zeros(1, 5, 14, 4)
boxes[:, :, 0] = torch.tensor([.25, .5, .2, .4])
boxes[:, :, 1] = torch.tensor([.4, .5, .2, .4])
boxes[:, :, 2] = torch.tensor([.8, .5, .1, .3])
dummy = torch.ones(1, 14, dtype=torch.bool)
dummy[:, :3] = False
valid = (~dummy[:, None, :]).expand(1, 5, 14)
actions = torch.full((1, 14), 7, dtype=torch.long)
actions[0, :3] = torch.tensor([0, 0, 6])
members = torch.zeros(1, 1, 14)
members[0, 0, :2] = 1
membership = torch.full((1, 14), -1, dtype=torch.long)
membership[0, :2] = 0
targets = [dict(boxes=boxes[0], actions=actions, activities=torch.tensor([[0]]),
                members=members, membership=membership)]
outputs = model(images, boxes, dummy, actor_frame_valid=valid)
terms = criterion(outputs, targets, log=False)
mask_loss = sum(terms[key] * criterion.weight_dict[key]
                for key in ('loss_mask_focal', 'loss_mask_dice', 'loss_mask_empty'))
mask_loss.backward()

modules = ['backbone', 'input_proj', 'fc_emb', 'box_pos_emb', 'group_transformer',
           'group_query_emb', 'mask_head', 'actor_match_emb', 'group_match_emb',
           'group_emb', 'class_emb']
gradients = {}
for name in modules:
    grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
    gradients[name] = dict(parameters_with_grad=len(grads),
        gradient_l2=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt()) if grads else None)
for name in modules[:7]:
    assert gradients[name]['gradient_l2'] > 0, name
for name in modules[7:]:
    assert gradients[name]['parameters_with_grad'] == 0, name

head = model.mask_head
model.mask_head = None
with torch.no_grad():
    without_heatmaps = model(images, boxes, dummy, actor_frame_valid=valid)
model.mask_head = head
differences = {key: float((outputs[key].detach() - without_heatmaps[key]).abs().max())
               for key in ('membership', 'pred_activities', 'pred_actions', 'actor_embeddings')}
assert all(value == 0 for value in differences.values())
print(json.dumps(dict(
    description='CPU synthetic dependency check; untrained ResNet18; 96x160 input; eval mode; no optimizer step. Not an accuracy experiment.',
    torch_version=torch.__version__, seed=17,
    mask_head_parameters=sum(p.numel() for p in head.parameters()),
    output_shapes={key:list(value.shape) for key,value in outputs.items()},
    heatmap_only_backward=gradients,
    max_absolute_difference_after_removing_mask_head=differences), indent=2))
