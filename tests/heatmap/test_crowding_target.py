"""Single target-variable H1-Sup control; no new prediction or matching path."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from models import build_model
from models.heatmap import gaussian_targets, prepare_targets, heatmap_losses
from tools.run_heatmap_research import (validate_development_protocol, tensor_hash, seed_all,
                                       record_context, file_hash)


def config(crowding=False):
    name = 'h1_sup_crowding' if crowding else 'h1_sup'
    return argparse.Namespace(**json.loads((ROOT/'configs/heatmap'/('%s.json' % name)).read_text()))


class CrowdingTarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        source = subprocess.check_output(['git', 'show', '33ca78c:models/heatmap.py'], cwd=str(ROOT)).decode()
        cls.reference = types.ModuleType('original_heatmap_targets')
        exec(compile(source, '33ca78c:models/heatmap.py', 'exec'), cls.reference.__dict__)

    def fixture(self):
        # Large close boxes, a far actor, and a padded/invalid actor.
        boxes = torch.tensor([[[.4,.5,.3,.7],[.44,.5,.3,.7],[.85,.5,.1,.2],[.401,.5,.3,.7]]]*2)
        valid = torch.tensor([[True, True, True, False], [True, False, True, False]])
        members = torch.tensor([[1.,0.,0.,0.],[0.,1.,1.,0.]])
        return boxes, members, valid

    def test_config_is_exactly_one_variable(self):
        base, candidate = vars(config()), vars(config(True)).copy()
        self.assertEqual(candidate.pop('heatmap_target_mode'), 'nearest_center')
        self.assertEqual(base, candidate)
        validate_development_protocol(config(True), argparse.Namespace(checkpoint=None))
        for field, value in (('split','view'), ('heatmap_variant','h2_s')):
            args = config(True); setattr(args, field, value)
            with self.assertRaisesRegex(ValueError, 'limited to place H1-Sup'):
                validate_development_protocol(args, argparse.Namespace(checkpoint=None))

    def test_frozen_source_includes_reference_cpp_and_manifest_hash(self):
        cli=argparse.Namespace(stage='audit',steps=3,checkpoint=None)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);record=record_context(out,config(True),cli)
            relative='tools/official_cafe_reference/crop_and_resize.cpp'
            manifest=json.loads((out/'source_manifest.json').read_text())
            self.assertIn(relative,manifest)
            self.assertEqual(file_hash(out/'source_snapshot'/relative),file_hash(ROOT/relative))
            self.assertEqual(file_hash(out/'source_manifest.json'),record['source_manifest_hash'])

    def test_disabled_targets_and_losses_are_bit_exact_to_original(self):
        boxes, members, valid = self.fixture()
        boxes[0,3] = float('nan')
        old = self.reference.gaussian_targets(boxes, members, valid, 23, 40, .15, .5, 4.)
        for mode in ({}, {'target_mode':'box'}):
            actual = gaussian_targets(boxes, members, valid, 23, 40, .15, .5, 4., **mode)
            for a, b in zip(actual, old): self.assertTrue(torch.equal(a, b))
        outputs = dict(group_mask_logits=torch.randn(1,3,2,23,40), actor_frame_valid=valid[None])
        target = dict(boxes=boxes, members=members[None])
        old_targets = self.reference.prepare_targets(outputs, [target], config())
        new_targets = prepare_targets(outputs, [target], config())
        indices = [(torch.tensor([0,2]), torch.tensor([1,0]))]
        expected = self.reference.heatmap_losses(outputs, old_targets, indices, config())
        actual = heatmap_losses(outputs, new_targets, indices, config())
        for key in expected: self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_exact_formula_includes_nonmember_neighbor_without_group_label(self):
        boxes, members, valid = self.fixture()
        actual, group_valid = gaussian_targets(boxes, members, valid, 23, 40, .15, .5, 4., 'nearest_center')
        # Independently evaluate each actor as a singleton target with the
        # prescribed sigma cap, including same-group / other-group / individual.
        reference = torch.zeros_like(actual)
        for ti in range(2):
            for actor in range(4):
                if not valid[ti,actor]: continue
                center = boxes[ti,actor,:2]*torch.tensor([40,23])
                neighbor = [float(torch.norm(center-boxes[ti,j,:2]*torch.tensor([40,23])))
                            for j in range(4) if j != actor and valid[ti,j]]
                cap = .35*min(neighbor) if neighbor else float('inf')
                sx, sy = (boxes[ti,actor,2:]*torch.tensor([40,23])*.15).clamp(.5,4.).tolist()
                sx, sy = max(.5,min(sx,cap)), max(.5,min(sy,cap))
                xx = (torch.arange(40)+.5-center[0])/sx
                yy = (torch.arange(23)+.5-center[1])/sy
                peak = torch.exp(-.5*(xx[None].square()+yy[:,None].square()))
                for group in range(2):
                    if members[group,actor]: reference[group,ti] = torch.maximum(reference[group,ti],peak)
        torch.testing.assert_allclose(actual, reference, atol=2e-7, rtol=2e-6)
        # Relabeling all other actors cannot affect this group's target.
        isolated = members[:1].clone()
        same, _ = gaussian_targets(boxes, isolated, valid, 23, 40, .15, .5, 4., 'nearest_center')
        self.assertTrue(torch.equal(same[0], actual[0]))
        self.assertTrue(group_valid.all())

    def test_isolated_invalid_coincident_and_small_boxes(self):
        boxes, members, valid = self.fixture()
        valid[:] = False; valid[:,0] = True
        a = gaussian_targets(boxes, members, valid, 23, 40, .15, .5, 4.)
        b = gaussian_targets(boxes, members, valid, 23, 40, .15, .5, 4., 'nearest_center')
        self.assertTrue(torch.equal(a[0], b[0]))
        for width in (.3, 1e-5):
            boxes[:,1] = boxes[:,0]; boxes[... ,2:] = width
            valid[:,:2] = True
            candidate, _ = gaussian_targets(boxes, members, valid, 23, 40, .15, .5, 4., 'nearest_center')
            floor, _ = gaussian_targets(boxes, members, valid, 23, 40, .15, .5, .5)
            self.assertTrue(torch.equal(candidate, floor))
            self.assertTrue(torch.isfinite(candidate).all())
        empty, frames = gaussian_targets(boxes, members[:0], valid, 23, 40, target_mode='nearest_center')
        self.assertEqual(tuple(empty.shape), (0,2,23,40)); self.assertEqual(tuple(frames.shape), (0,2))
        valid[:] = False
        blank, frame_valid = gaussian_targets(boxes, members, valid, 23, 40, target_mode='nearest_center')
        self.assertEqual(float(blank.sum()), 0); self.assertFalse(frame_valid.any())
        with self.assertRaises(ValueError): gaussian_targets(boxes,members,valid,23,40,target_mode='typo')

    def test_permutation_monotonicity_and_finite_mask_gradients(self):
        boxes, members, valid = self.fixture()
        base, _ = gaussian_targets(boxes,members,valid,23,40,.15,.5,4.)
        candidate, _ = gaussian_targets(boxes,members,valid,23,40,.15,.5,4.,'nearest_center')
        self.assertTrue((candidate <= base).all()); self.assertLess(float(candidate.sum()), float(base.sum()))
        permutation = torch.tensor([2,0,3,1])
        permuted, _ = gaussian_targets(boxes[:,permutation],members[:,permutation],valid[:,permutation],23,40,.15,.5,4.,'nearest_center')
        self.assertTrue(torch.equal(permuted,candidate))
        logits = torch.randn(1,3,2,23,40,requires_grad=True)
        outputs = dict(group_mask_logits=logits,actor_frame_valid=valid[None])
        targets = prepare_targets(outputs,[dict(boxes=boxes,members=members[None])],config(True))
        terms = heatmap_losses(outputs,targets,[(torch.tensor([0,1]),torch.tensor([0,1]))],config(True))
        sum(terms.values()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all()); self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_full_model_forward_initialization_rng_and_membership_unchanged(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        seed_all(1); base, criterion_a = build_model(config())
        state_hash = tensor_hash(base,common=False); rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        seed_all(1); candidate, criterion_b = build_model(config(True))
        self.assertEqual(state_hash,tensor_hash(candidate,common=False))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        if gpu_rng: self.assertTrue(all(torch.equal(a,b) for a,b in zip(gpu_rng,torch.cuda.get_rng_state_all())))
        base.to(device).eval(); candidate.to(device).eval(); criterion_a.to(device); criterion_b.to(device)
        images = torch.randn(1,5,3,96,160,device=device)
        boxes = torch.zeros(1,5,14,4,device=device)
        boxes[:,:,:3] = torch.tensor([[.4,.5,.3,.7],[.44,.5,.3,.7],[.8,.5,.1,.2]],device=device)
        dummy = torch.ones(1,14,dtype=torch.bool,device=device); dummy[:,:3] = False
        valid = (~dummy[:,None]).expand(1,5,14)
        actions = torch.full((1,14),7,dtype=torch.long,device=device); actions[0,:3] = torch.tensor([0,0,6],device=device)
        members = torch.zeros(1,1,14,device=device); members[0,0,:2] = 1
        membership = torch.full((1,14),-1,dtype=torch.long,device=device); membership[0,:2] = 0
        target = dict(boxes=boxes[0],actions=actions,members=members,membership=membership,
                      activities=torch.zeros(1,1,dtype=torch.long,device=device),actor_frame_valid=valid[0])
        with torch.no_grad():
            a = base(images,boxes,dummy,actor_frame_valid=valid)
            b = candidate(images,boxes,dummy,actor_frame_valid=valid)
        for key in a: self.assertTrue(torch.equal(a[key],b[key]),key)
        loss_a = criterion_a(a,[copy.deepcopy(target)])
        loss_b = criterion_b(b,[copy.deepcopy(target)])
        for key in loss_a:
            if not key.startswith('loss_mask_'): self.assertTrue(torch.equal(loss_a[key],loss_b[key]),key)
        # Supervision may change mask losses, never the target-free forward.
        with torch.no_grad():
            after = candidate(images,boxes,dummy,actor_frame_valid=valid)
        for key in b: self.assertTrue(torch.equal(b[key],after[key]),key)


if __name__ == '__main__': unittest.main()
