import argparse
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F

from models import build_model
from models.heatmap import (gaussian_targets,roi_responses,frame_validity,soft_focal,
                            GroupMaskHead,prepare_targets,heatmap_losses)
from models.heatmap_criterion import HeatmapMatcher
from util.group_export import make_txt
from evaluation.cafe_eval import GAD_Evaluation


def args_for(variant='h2_s'):
    args=argparse.Namespace(**json.loads((ROOT/'configs/heatmap/b0.json').read_text()))
    args.heatmap_variant=variant
    return args


class Geometry(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.boxes=torch.tensor([[[.2,.5,.12,.4],[.8,.5,.12,.4],[.5,.5,.1,.3]]]*2)
        self.valid=torch.ones(2,3,dtype=torch.bool)
        self.members=torch.tensor([[1.,1.,0.],[0.,0.,1.]])

    def test_disconnected_same_activity_targets_and_roi(self):
        maps,valid=gaussian_targets(self.boxes,self.members,self.valid,23,40)
        self.assertEqual(tuple(maps.shape),(2,2,23,40))
        self.assertTrue(valid.all())
        _,response=roi_responses(maps[None],self.boxes[None],self.valid[None])
        self.assertGreater(float(response[0,0,0]),float(response[0,0,2])+.1)
        self.assertGreater(float(response[0,0,1]),float(response[0,0,2])+.1)
        self.assertFalse(torch.equal(maps[0],maps[1]))
        self.assertLess(float(maps[0,:,11,20].max()),.01)

    def test_motion_resize_flip_crop_boundary(self):
        original=torch.tensor([[[.2,.4,.2,.3]],[[.7,.6,.2,.3]]])
        valid=torch.ones(2,1,dtype=torch.bool)
        members=torch.ones(1,1)
        maps,_=gaussian_targets(original,members,valid,23,40)
        _,moving=roi_responses(maps[None],original[None],valid[None])
        wrong=original.flip(0)
        _,stationary=roi_responses(maps[None],wrong[None],valid[None])
        self.assertGreater(float(moving.mean()),float(stationary.mean())+.1)
        flipped=original.clone(); flipped[...,0]=1-flipped[...,0]
        flip_maps,_=gaussian_targets(flipped,members,valid,23,40)
        torch.testing.assert_allclose(flip_maps,maps.flip(-1),rtol=1e-5,atol=1e-6)
        # Non-square resize leaves normalized boxes unchanged. A crop's affine
        # coordinate transform is checked independently (B0 has no crop aug).
        pixels=original*torch.tensor([1280,720,1280,720])
        normalized=pixels*torch.tensor([640/1280,360/720,640/1280,360/720])/torch.tensor([640,360,640,360])
        torch.testing.assert_allclose(normalized,original)
        cropped=original.clone(); cropped[...,0]=(cropped[...,0]-.1)/.8; cropped[...,2]/=.8
        crop_maps,_=gaussian_targets(cropped,members,valid,23,32)
        _,cropped_read=roi_responses(crop_maps[None],cropped[None],valid[None])
        self.assertGreater(float(cropped_read.mean()),.1)
        edge=original.clone(); edge[...,0]=.01
        edge_maps,_=gaussian_targets(edge,members,valid,23,40)
        _,edge_read=roi_responses(edge_maps[None],edge[None],valid[None])
        self.assertTrue(torch.isfinite(edge_read).all())
        self.assertGreater(float(edge_read.mean()),0)

    def test_validity_empty_missing_nan_padding(self):
        boxes=self.boxes.clone(); boxes[0,0]=float('nan'); boxes[1,0,2:]=1e-8
        dummy=torch.tensor([[False,False,True]])
        valid=frame_validity(boxes[None],dummy)[0]
        self.assertFalse(valid[:,0].any()); self.assertFalse(valid[:,2].any())
        maps,group_valid=gaussian_targets(boxes,torch.tensor([[1.,0.,0.]]),valid,23,40)
        self.assertFalse(group_valid.any()); self.assertEqual(float(maps.sum()),0)
        _,response=roi_responses(maps[None],boxes[None],valid[None])
        self.assertTrue(torch.isfinite(response).all())
        empty,empty_valid=gaussian_targets(boxes,torch.zeros(0,3),valid,23,40)
        self.assertEqual(tuple(empty.shape),(0,2,23,40))
        self.assertEqual(tuple(empty_valid.shape),(0,2))

    def test_soft_focal_is_bce_at_gamma_zero_without_balance(self):
        logits=torch.randn(3,4,requires_grad=True); target=torch.rand(3,4)*.9
        torch.testing.assert_allclose(soft_focal(logits,target,0,-1),F.binary_cross_entropy_with_logits(logits,target,reduction='none'),rtol=0,atol=0)

    def test_identical_overlapping_tracks_are_oracle_limit(self):
        boxes=torch.tensor([[[.4,.5,.2,.4],[.4,.5,.2,.4]]])
        valid=torch.ones(1,2,dtype=torch.bool)
        maps,_=gaussian_targets(boxes,torch.tensor([[1.,0.]]),valid,23,40)
        _,response=roi_responses(maps[None],boxes[None],valid[None])
        torch.testing.assert_allclose(response[:,:,0],response[:,:,1],rtol=0,atol=0)

    def test_soft_peak_and_missing_group_frame(self):
        boxes=self.boxes.clone(); boxes[...,0]+=.003
        valid=self.valid.clone(); valid[1,:2]=False
        maps,group_valid=gaussian_targets(boxes,self.members,valid,23,40)
        self.assertFalse(group_valid[0,1]); self.assertEqual(float(maps[0,1].sum()),0)
        self.assertLess(float(maps[0,0].max()),1)

    def test_mask_dependency_actor_permutation_and_gradient(self):
        maps,_=gaussian_targets(self.boxes,self.members,self.valid,23,40)
        maps=maps[None].requires_grad_()
        _,mean=roi_responses(maps,self.boxes[None],self.valid[None])
        _,moved=roi_responses(maps.flip(-1),self.boxes[None],self.valid[None])
        # Asymmetric third actor supplies a non-invariant causal check.
        shifted=maps.roll(8,-1)
        _,shift_response=roi_responses(shifted,self.boxes[None],self.valid[None])
        self.assertGreater(float((mean-shift_response).abs().max()),.1)
        permutation=torch.tensor([2,0,1])
        _,permuted=roi_responses(maps,self.boxes[None,:,permutation],self.valid[None,:,permutation])
        torch.testing.assert_allclose(permuted,mean[:,:,permutation])
        mean.sum().backward()
        self.assertTrue(torch.isfinite(maps.grad).all()); self.assertGreater(float(maps.grad.abs().sum()),0)


class ModelContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.args=args_for('off')
        torch.manual_seed(1)
        cls.off,cls.criterion=build_model(cls.args)
        cls.off.eval()
        cls.images=torch.randn(1,5,3,96,160)
        cls.boxes=torch.zeros(1,5,14,4)
        cls.boxes[:,:,0]=torch.tensor([.25,.5,.2,.4]); cls.boxes[:,:,1]=torch.tensor([.45,.5,.2,.4])
        cls.boxes[:,:,2]=torch.tensor([.8,.5,.1,.3])
        cls.dummy=torch.ones(1,14,dtype=torch.bool); cls.dummy[:,:3]=False
        cls.valid=~cls.dummy[:,None,:].expand(1,5,14)

    def target(self,empty=False):
        members=torch.zeros(1,0 if empty else 1,14)
        if not empty: members[0,0,:2]=1
        actions=torch.full((1,14),7,dtype=torch.long); actions[0,:3]=torch.tensor([0,0,6])
        membership=torch.full((1,14),-1,dtype=torch.long)
        if not empty: membership[0,:2]=0
        return dict(boxes=self.boxes[0].clone(),actions=actions,members=members,
            activities=torch.zeros((1,0 if empty else 1),dtype=torch.long),membership=membership,
            actor_frame_valid=self.valid[0].clone())

    def test_read_only_gradient_diagnosis_whole_model(self):
        from tools.diagnose_heatmap_checkpoint import diagnose
        args=args_for('h2_s'); model,criterion=build_model(args)
        info={'vid':1,'sid':1,'fid':[0,1,2,3,4]}
        result=diagnose(model,criterion,(self.images,[self.target()],[info]),args,torch.device('cpu'))
        self.assertGreater(result['gradient_scales']['loss_group_code']['weighted_gradient_norm'],0)
        self.assertEqual(result['gradient_scales']['loss_group_ce']['weighted_gradient_norm'],0)
        self.assertGreater(result['gradient_scales']['diagnostic_soft_bce']['weighted_gradient_norm'],0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(result['total_slots'],12)

    def test_disabled_exact_original_revision(self):
        source=subprocess.check_output(['git','show','exp/b0-repro:models/models.py'],cwd=str(ROOT)).decode()
        source=source.replace('from .group_transformer import','from .b0_transformer import')
        module=types.ModuleType('models._b0_reference'); module.__package__='models'
        exec(compile(source,'exp/b0-repro:models/models.py','exec'),module.__dict__)
        reference=module.GADTR(self.args).eval()
        reference.load_state_dict(self.off.state_dict(),strict=True)
        with torch.no_grad():
            expected=reference(self.images,self.boxes,self.dummy)
            actual=self.off(self.images,self.boxes,self.dummy)
        for key in expected:
            torch.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=0)

    def test_common_initialization_rng_preserved_and_h1_same_predictions(self):
        torch.manual_seed(19); off,_=build_model(args_for('off')); rng=torch.get_rng_state()
        torch.manual_seed(19); h1,_=build_model(args_for('h1_sup'))
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for key,value in off.state_dict().items(): self.assertTrue(torch.equal(value,h1.state_dict()[key]),key)
        off.eval(); h1.eval()
        with torch.no_grad():
            b0=off(self.images,self.boxes,self.dummy)
            candidate=h1(self.images,self.boxes,self.dummy)
        for key in b0: torch.testing.assert_allclose(b0[key],candidate[key],rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable in sandbox; run on allocated host GPU')
    def test_cuda_rng_unchanged_by_new_head(self):
        torch.cuda.init()
        torch.manual_seed(37); off,_=build_model(args_for('off'))
        cpu=torch.get_rng_state(); cuda=torch.cuda.get_rng_state_all()
        torch.manual_seed(37); h2,_=build_model(args_for('h2_s'))
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))

    def test_h2_membership_only_gradient_and_no_gt_forward_dependency(self):
        model,criterion=build_model(args_for('h2_s')); model.eval()
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device); criterion.to(device)
        images,boxes,dummy,valid=(tensor.to(device) for tensor in (self.images,self.boxes,self.dummy,self.valid))
        target={key:value.to(device) for key,value in self.target().items()}
        output=model(images,boxes,dummy,actor_frame_valid=valid)
        terms=criterion(output,[target])
        terms['loss_group_code'].backward()
        gradients=[p.grad for p in model.mask_head.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)
        self.assertIsNone(model.group_match_emb.weight.grad)
        torch.testing.assert_allclose(output['membership'].sum(1),torch.ones(1,14,device=device))
        with torch.no_grad():
            before=model(images,boxes,dummy,actor_frame_valid=valid)
            changed={key:value.to(device) for key,value in self.target().items()}
            changed['members'].zero_(); changed['activities'][:]=4
            criterion(before,[changed])
            after=model(images,boxes,dummy,actor_frame_valid=valid)
        for key in before: torch.testing.assert_allclose(before[key],after[key],rtol=0,atol=0)

    def test_empty_gt_and_all_padding_losses_finite(self):
        model,criterion=build_model(args_for('h2_s')); model.eval()
        output=model(self.images,self.boxes,self.dummy,actor_frame_valid=self.valid)
        terms=criterion(output,[self.target(empty=True)])
        self.assertTrue(all(torch.isfinite(v).all() for v in terms.values()))
        dummy=torch.ones_like(self.dummy)
        valid=torch.zeros_like(self.valid)
        output=model(self.images,torch.zeros_like(self.boxes),dummy,actor_frame_valid=valid)
        target=self.target(empty=True); target['actions'][:]=7; target['boxes'].zero_(); target['actor_frame_valid'].zero_()
        terms=criterion(output,[target]); self.assertTrue(all(torch.isfinite(v).all() for v in terms.values()))

    def test_shared_assignment_slot_permutation_and_empty_suppression(self):
        args=args_for('h2_s'); model,criterion=build_model(args); model.eval()
        outputs=model(self.images,self.boxes,self.dummy,actor_frame_valid=self.valid)
        targets=[self.target()]
        terms=criterion(outputs,copy.deepcopy(targets))
        permutation=torch.arange(11,-1,-1)
        permuted=dict(outputs)
        for key in ('membership','membership_logits','group_masks','group_mask_logits','spatial_response','pred_activities'):
            permuted[key]=outputs[key][:,permutation]
        permuted_terms=criterion(permuted,copy.deepcopy(targets))
        for key in criterion.weight_dict:
            torch.testing.assert_allclose(terms[key],permuted_terms[key],rtol=1e-5,atol=1e-6)
        prepared=prepare_targets(outputs,targets,args)
        logits=outputs['group_mask_logits'].detach().requires_grad_()
        losses=heatmap_losses(dict(outputs,group_mask_logits=logits),prepared,[(torch.tensor([0]),torch.tensor([0]))],args)
        losses['loss_mask_empty'].backward()
        self.assertEqual(float(logits.grad[:,0].abs().sum()),0)
        self.assertGreater(float(logits.grad[:,1:].min()),0)

    def test_export_original_decoder_and_evaluator_roundtrip(self):
        args=args_for('off')
        outputs=dict(pred_activities=torch.full((1,12,7),-10.),membership=torch.zeros(1,12,14))
        outputs['pred_activities'][0,0,0]=10
        outputs['pred_activities'][0,1,1]=10
        outputs['membership'][0,0,:2]=1
        outputs['membership'][0,1,2]=.1
        infos=[dict(vid=1,sid=1,fid=list(range(5)))]
        # Compile the actual original decoder, then compare byte for byte.
        source=subprocess.check_output(['git','show','HEAD:train.py'],cwd=str(ROOT)).decode()
        import ast
        tree=ast.parse(source)
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='make_txt')
        scope=dict(args=args,torch=torch,F=F)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'original_decoder','exec'),scope)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prediction.txt'; expected=Path(directory)/'expected.txt'
            make_txt(self.boxes,infos,outputs,{1:1},str(path),args)
            scope['make_txt'](self.boxes,infos,outputs,{1:1},str(expected))
            self.assertEqual(path.read_bytes(),expected.read_bytes())
            rows=path.read_text().splitlines()
            self.assertEqual(len(rows),15)
            self.assertEqual(rows[2].split()[7:9],['-1','7'])
            gt=Path(directory)/'gt.txt'; gt.write_text('\n'.join(' '.join(row.split()[:9]) for row in rows)+'\n')
            with open(gt) as groundtruth,open(ROOT/args.labelmap) as labels,open(path) as predictions:
                result=GAD_Evaluation(argparse.Namespace(groundtruth=groundtruth,labelmap=labels,eval_type='gt_base')).evaluate(predictions)
            self.assertAlmostEqual(result['group_mAP_1.0'],100/6)
            self.assertAlmostEqual(result['outlier_mIoU'],100)


class RunnerProtocol(unittest.TestCase):
    def test_read_only_diagnostic_gradient_vector_and_unused_parameters(self):
        from tools.diagnose_heatmap_checkpoint import gradient_vector
        first=torch.nn.Parameter(torch.tensor([1.,2.]))
        unused=torch.nn.Parameter(torch.tensor([3.]))
        result=gradient_vector(first.square().sum(),[first,unused])
        torch.testing.assert_allclose(result,torch.tensor([2.,4.,0.]),rtol=0,atol=0)
        self.assertIsNone(first.grad)
        self.assertEqual(float(gradient_vector(torch.tensor(1.),[first,unused]).abs().sum()),0)

    def test_evaluator_numpy_metrics_json_serialization(self):
        import numpy as np
        from tools.run_heatmap_research import json_default
        metric={'AP':np.array([0.,1.]),'mAP':np.float32(.5),'path':Path('test')}
        self.assertEqual(json.loads(json.dumps(metric,default=json_default)),
                         {'AP':[0.,1.],'mAP':.5,'path':'test'})

    def test_fixed_input_cache_preserves_samples_and_rejects_training_augmentation(self):
        from tools.run_heatmap_research import cache_fixed_samples,collate
        class TinyDataset:
            is_training=False
            def __getitem__(self,index):
                return torch.ones(2,3,4,5)*index,{'boxes':torch.tensor([index])},{'vid':1,'sid':index,'fid':[2,4]}
        dataset=TinyDataset()
        cached,metadata=cache_fixed_samples(dataset,[1,3])
        raw=[dataset[i] for i in [1,3]]
        torch.testing.assert_allclose(collate(cached)[0],collate(raw)[0],rtol=0,atol=0)
        self.assertEqual(collate(cached)[2],collate(raw)[2])
        self.assertEqual(metadata['image_bytes'],2*2*3*4*5*4)
        self.assertEqual(metadata,cache_fixed_samples(dataset,[1,3])[1])
        dataset.is_training=True
        with self.assertRaises(ValueError): cache_fixed_samples(dataset,[1])

    def test_original_learning_rate_and_full_epoch_step_boundaries(self):
        from tools.run_heatmap_research import b0_learning_rate
        args=args_for('off')
        steps_per_epoch=6444//args.batch
        self.assertEqual(steps_per_epoch,402)
        for step,expected in ((0,1e-5),(401,1e-5),(402,3.25e-5),(803,3.25e-5),
                              (804,5.5e-5),(1608,1e-4),(2010,9.6e-5),(11658,0.)):
            self.assertAlmostEqual(b0_learning_rate(args,step//steps_per_epoch),expected)


if __name__=='__main__':
    unittest.main(verbosity=2)
