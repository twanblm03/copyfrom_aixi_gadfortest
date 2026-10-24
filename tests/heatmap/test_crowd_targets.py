"""CPU checks for the crowd-constrained H1 auxiliary supervision target."""
import argparse
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from models.heatmap import (add_heatmap_arguments, frame_validity,
                            gaussian_targets, heatmap_losses, prepare_targets)


class CrowdTargets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        # The two nearby centers fall exactly on samples in the 23 x 40 grid.
        self.boxes = torch.tensor([[[18.5/40, 11.5/23, .5, .8],
                                    [20.5/40, 11.5/23, .5, .8],
                                    [35.5/40, 18.5/23, .2, .3]]])
        self.valid = torch.ones(1, 3, dtype=torch.bool)
        self.members = torch.tensor([[1., 0., 0.], [0., 1., 0.]])

    def target(self, boxes=None, members=None, valid=None, **kwargs):
        return gaussian_targets(self.boxes if boxes is None else boxes,
                                self.members if members is None else members,
                                self.valid if valid is None else valid,
                                23, 40, sigma_ratio=.15, **kwargs)

    def test_legacy_target_is_bitwise_unchanged_and_rng_unchanged(self):
        boxes = torch.tensor([[[.3, .4, .6, .8], [.7, .6, .4, .6], [float('nan')]*4],
                              [[.4, .6, .8, .6], [.6, .4, .6, .8], [0., 0., 0., 0.]]])
        valid = torch.tensor([[True, True, False], [True, False, False]])
        members = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        # Captured from gaussian_targets before adding the crowd mode. Mixed
        # axis widths, invalid NaNs and an absent group frame are intentional.
        expected = torch.tensor([
            [[[.7115725874900818, .8175944089889526, .46909764409065247, .13439874351024628],
              [.8175944089889526, .9394130706787109, .5389915108680725, .15442368388175964],
              [.46909764409065247, .5389915108680725, .3092482388019562, .08860112726688385]],
             [[.2781658172607422, .35163453221321106, .3007689416408539, .17407171428203583],
              [.7468553781509399, .9441136717796326, .8075432181358337, .4673701524734497],
              [.58344966173172, .7375494241714478, .6308594346046448, .3651134669780731]]],
            [[[.0056504360400140285, .09408825635910034, .328400194644928, .2402627170085907],
              [.01517102587968111, .25262027978897095, .8817311525344849, .645088255405426],
              [.011851726099848747, .19734905660152435, .688815712928772, .5039482712745667]],
             [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]]])
        rng = torch.get_rng_state().clone()
        legacy, group_valid = gaussian_targets(boxes, members, valid, 3, 4, .5, .5, 4.)
        explicit, _ = gaussian_targets(boxes, members, valid, 3, 4, .5, .5, 4., target_mode='box')
        self.assertTrue(torch.equal(legacy, expected))
        self.assertTrue(torch.equal(explicit, expected))
        self.assertTrue(torch.equal(group_valid, torch.tensor([[True, True], [True, False]])))
        self.target(target_mode='crowd')
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_overlapping_nonmember_center_leakage_is_reduced(self):
        legacy, legacy_valid = self.target()
        crowd, crowd_valid = self.target(target_mode='crowd')
        self.assertTrue(torch.equal(crowd_valid, legacy_valid))
        self.assertEqual(float(crowd[0, 0, 11, 18]), 1.)
        self.assertGreater(float(legacy[0, 0, 11, 20]), .8)
        self.assertLess(float(crowd[0, 0, 11, 20]), .02)
        self.assertAlmostEqual(float(crowd[0, 0, 11, 20]), math.exp(-1/(2*.35**2)), places=6)
        self.assertTrue((crowd <= legacy).all())
        # The nearest actor still caps the target when it has no group label.
        ungrouped_neighbour, _ = self.target(members=self.members[:1], target_mode='crowd')
        self.assertTrue(torch.equal(ungrouped_neighbour, crowd[:1]))
        union, _ = self.target(members=torch.tensor([[1., 1., 0.]]), target_mode='crowd')
        self.assertTrue(torch.equal(union[0], crowd.max(0).values))

    def test_invalid_padding_and_missing_frames_do_not_constrain(self):
        boxes = self.boxes[:, :1].expand(3, 1, 4).clone()
        boxes = torch.cat((boxes, boxes.clone(), boxes.clone(), boxes.clone()), 1)
        boxes[:, 1] = float('nan')
        boxes[:, 2, 2:] = 0.
        boxes[1, 0, 2:] = 0.
        boxes[2, 0] = float('nan')
        # The fourth, identical actor is padded; it must not force the floor.
        dummy = torch.tensor([[False, False, False, True]])
        valid = frame_validity(boxes[None], dummy)[0]
        members = torch.tensor([[1., 0., 0., 1.], [0., 1., 1., 0.]])
        crowd, group_valid = self.target(boxes, members, valid, target_mode='crowd')
        legacy, _ = self.target(boxes, members, valid)
        self.assertTrue(torch.equal(crowd, legacy))
        self.assertTrue(torch.isfinite(crowd).all())
        self.assertTrue(torch.equal(group_valid, torch.tensor([[True, False, False], [False, False, False]])))
        self.assertEqual(float(crowd[:, 1:].sum()), 0.)
        empty, empty_valid = self.target(boxes, members[:0], valid, target_mode='crowd')
        self.assertEqual(tuple(empty.shape), (0, 3, 23, 40))
        self.assertEqual(tuple(empty_valid.shape), (0, 3))

    def test_valid_neighbours_are_selected_per_frame(self):
        boxes = self.boxes.expand(2, 3, 4).clone()
        valid = self.valid.expand(2, 3).clone()
        valid[1, 1:] = False
        crowd, group_valid = self.target(boxes, valid=valid, target_mode='crowd')
        legacy, _ = self.target(boxes, valid=valid)
        self.assertLess(float(crowd[0, 0].sum()), float(legacy[0, 0].sum()))
        self.assertTrue(torch.equal(crowd[:, 1], legacy[:, 1]))
        self.assertTrue(torch.equal(group_valid, torch.tensor([[True, True], [True, False]])))

    def test_single_and_identical_centers_preserve_finite_floor_targets(self):
        isolated, _ = self.target(self.boxes[:, :1], self.members[:1, :1], self.valid[:, :1],
                                  target_mode='crowd')
        legacy, _ = self.target(self.boxes[:, :1], self.members[:1, :1], self.valid[:, :1])
        self.assertTrue(torch.equal(isolated, legacy))
        boxes = self.boxes[:, :1].expand(1, 2, 4).clone()
        members = torch.eye(2)
        valid = torch.ones(1, 2, dtype=torch.bool)
        crowd, _ = self.target(boxes, members, valid, target_mode='crowd')
        floor, _ = self.target(boxes, members, valid, sigma_min=.5, sigma_max=.5)
        self.assertTrue(torch.isfinite(crowd).all())
        self.assertTrue(torch.equal(crowd, floor))
        self.assertTrue(torch.equal(crowd[0], crowd[1]))

    def test_grid_distances_and_axis_scales_match_continuous_geometry(self):
        boxes = torch.tensor([[[.3125, .425, .8, .8], [.3625, .525, .8, .8]]], dtype=torch.float64)
        members = torch.tensor([[1., 0.]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        for height, width in ((20, 40), (40, 40), (40, 80)):
            with self.subTest(height=height, width=width):
                target, _ = gaussian_targets(boxes, members, valid, height, width,
                                              .15, .5, 4., target_mode='crowd')
                distance = math.hypot(.05*width, .1*height)
                sigma_x = max(.5, min(.8*width*.15, 4., .35*distance))
                sigma_y = max(.5, min(.8*height*.15, 4., .35*distance))
                rows = torch.arange(height, dtype=torch.float64)[:, None] + .5
                columns = torch.arange(width, dtype=torch.float64)[None] + .5
                expected = torch.exp(-.5*((columns-.3125*width).square()/sigma_x**2
                                          + (rows-.425*height).square()/sigma_y**2))
                torch.testing.assert_close(target[0, 0], expected, rtol=1e-12, atol=1e-14)
                # Pixel-space resizing leaves normalized geometry unchanged.
                pixels = boxes * torch.tensor([1280., 720., 1280., 720.], dtype=boxes.dtype)
                resized = pixels * torch.tensor([.5, 2., .5, 2.], dtype=boxes.dtype)
                normalized = resized / torch.tensor([640., 1440., 640., 1440.], dtype=boxes.dtype)
                resized_target, _ = gaussian_targets(normalized, members, valid, height, width,
                                                      .15, .5, 4., target_mode='crowd')
                torch.testing.assert_close(resized_target, target, rtol=1e-12, atol=1e-14)

    def test_parser_and_old_config_defaults_and_configured_cap(self):
        parser = argparse.ArgumentParser()
        add_heatmap_arguments(parser)
        parsed = parser.parse_args([])
        self.assertEqual(parsed.heatmap_target_mode, 'box')
        self.assertEqual(parsed.heatmap_crowd_cap_ratio, .35)
        args = argparse.Namespace(heatmap_sigma_ratio=.15, heatmap_sigma_min=.5, heatmap_sigma_max=4.)
        outputs = {'group_mask_logits': torch.zeros(1, 3, 1, 23, 40),
                   'actor_frame_valid': self.valid[None]}
        target = {'boxes': self.boxes, 'members': self.members[None]}
        original = prepare_targets(outputs, [target], args)[0]
        legacy, valid = self.target()
        self.assertTrue(torch.equal(original['group_heatmaps'], legacy))
        self.assertTrue(torch.equal(original['group_frame_valid'], valid))
        args.heatmap_target_mode = 'crowd'
        args.heatmap_crowd_cap_ratio = .45
        prepared = prepare_targets(outputs, [target], args)[0]
        expected, _ = self.target(target_mode='crowd', crowd_cap_ratio=.45)
        self.assertTrue(torch.equal(prepared['group_heatmaps'], expected))
        self.assertNotIn('group_heatmaps', target)

    def test_crowd_supervision_losses_and_gradients_are_finite(self):
        boxes = self.boxes.expand(3, 3, 4).clone()
        valid = self.valid.expand(3, 3).clone()
        valid[1, 1:] = False
        valid[2] = False
        args = argparse.Namespace(heatmap_sigma_ratio=.15, heatmap_sigma_min=.5, heatmap_sigma_max=4.,
                                  heatmap_target_mode='crowd', heatmap_crowd_cap_ratio=.35,
                                  heatmap_focal_gamma=2., heatmap_focal_alpha=.25)
        logits = torch.linspace(-2., 2., 3*3*23*40).reshape(1, 3, 3, 23, 40).requires_grad_()
        outputs = {'group_mask_logits': logits, 'actor_frame_valid': valid[None]}
        targets = prepare_targets(outputs, [{'boxes': boxes, 'members': self.members[None]}], args)
        losses = heatmap_losses(outputs, targets, [(torch.tensor([0, 1]), torch.tensor([0, 1]))], args)
        self.assertTrue(all(torch.isfinite(term) and term > 0 for term in losses.values()))
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        self.assertEqual(float(logits.grad[:, :, 2].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[:, 1, 1].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[:, 2, :2].abs().sum()), 0.)

    def test_invalid_mode_and_crowd_ratio_rejected(self):
        with self.assertRaises(ValueError):
            self.target(target_mode='unknown')
        for ratio in (0., -.1, float('nan'), float('inf')):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.target(target_mode='crowd', crowd_cap_ratio=ratio)


if __name__ == '__main__':
    unittest.main()
