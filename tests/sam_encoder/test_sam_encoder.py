"""CPU contract tests for the SAM encoder path; no SAM weights or downloads.

Only SAM's expensive image encoder is substituted.  The integration test runs
the actual GAD model, RoIAlign, transformer, matcher, and training criterion.
Run with: python tests/sam_encoder/test_sam_encoder.py
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from models import build_model
from models.sam_backbone import SAMImageBackbone, align_sam_features
from util.sam_encoder import (
    SAMRGBTransform, add_sam_encoder_arguments, is_sam_backbone,
    sam_resume_config, validate_sam_encoder_args, validate_sam_resume,
)


def setUpModule():
    global _previous_threads
    _previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)


def tearDownModule():
    torch.set_num_threads(_previous_threads)


class TinyEncoder(nn.Module):
    """Spatially varying, stateful stand-in with SAM's post-neck contract."""

    def __init__(self):
        super().__init__()
        # Random initialization also lets the test detect GAD RNG pollution.
        self.scale = nn.Parameter(torch.randn(256))
        self.bias = nn.Parameter(torch.randn(256))
        self.dropout = nn.Dropout(.5)
        self.calls = []

    def forward(self, pixels):
        self.calls.append(dict(
            batch=len(pixels), training=self.training,
            grad_enabled=torch.is_grad_enabled(),
            top_left=pixels[:, :, 0, 0].detach().clone(),
            bottom_right=pixels[:, :, -1, -1].detach().clone(),
        ))
        spatial = F.avg_pool2d(pixels, 16).mean(1, keepdim=True)
        return self.dropout(spatial * self.scale[None, :, None, None]
                            + self.bias[None, :, None, None])


def arguments(repo, checkpoint, **overrides):
    config = json.loads((ROOT / 'configs/heatmap/b0.json').read_text())
    parser = argparse.ArgumentParser()
    add_sam_encoder_arguments(parser)
    config.update(vars(parser.parse_args([])))
    config.update(backbone='sam_vit_b', train_mask_mode='disabled',
                  sam_repo=str(repo), sam_checkpoint=str(checkpoint),
                  no_mae=True, use_mae=False, criterion_on_cpu=True,
                  unfreeze_blocks=0)
    config.update(overrides)
    return argparse.Namespace(**config)


class LocalAssets(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.repo = self.directory / 'sam'
        for name in ('__init__.py', 'modeling/image_encoder.py', 'build_sam.py'):
            path = self.repo / 'segment_anything' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# CPU fixture: the encoder constructor is mocked.\n')
        tiny = TinyEncoder()
        self.encoder_state = {
            'scale': torch.linspace(.5, 1.5, 256),
            'bias': torch.linspace(-.2, .2, 256),
        }
        tiny.load_state_dict(self.encoder_state, strict=True)
        self.checkpoint = self.directory / 'sam_vit_b.pth'
        self.write_checkpoint()
        self.args = arguments(self.repo, self.checkpoint)
        self.constructor = mock.patch(
            'models.sam_backbone._build_image_encoder',
            side_effect=lambda repo: TinyEncoder())
        self.builder = self.constructor.start()
        self.addCleanup(self.constructor.stop)

    def write_checkpoint(self, state=None):
        if state is None:
            state = {'image_encoder.' + key: value
                     for key, value in self.encoder_state.items()}
            # Official full SAM files also contain unrelated frozen modules.
            state['prompt_encoder.fixture'] = torch.ones(1)
            state['mask_decoder.fixture'] = torch.ones(1)
        torch.save(state, self.checkpoint)


class Geometry(unittest.TestCase):
    @staticmethod
    def coordinate_field():
        rows = torch.arange(64, dtype=torch.float64)[:, None].expand(64, 64)
        columns = torch.arange(64, dtype=torch.float64)[None, :].expand(64, 64)
        return torch.stack((rows, columns))[None]

    def test_16_by_9_view_crops_36_rows_and_excludes_padding(self):
        features = self.coordinate_field()
        features[:, :, 36:, :] = 1e6  # Padding must never become scene content.
        actual = align_sam_features(features, (576, 1024), (23, 40))
        row_centers = (torch.arange(23, dtype=torch.float64) + .5) * 36 / 23 - .5
        column_centers = (torch.arange(40, dtype=torch.float64) + .5) * 64 / 40 - .5
        torch.testing.assert_close(actual[0, 0], row_centers[:, None].expand(23, 40))
        torch.testing.assert_close(actual[0, 1], column_centers[None, :].expand(23, 40))
        self.assertLess(float(actual.max()), 64)

    def test_fractional_view_uses_original_pixel_centers_without_rounding(self):
        # 603 / 16 = 37.6875 valid token rows: neither 37 nor 38 is correct.
        actual = align_sam_features(self.coordinate_field(), (603, 1024), (23, 40))
        original_row_centers = (torch.arange(23, dtype=torch.float64) + .5) * 603 / 23
        expected = (original_row_centers / 16 - .5).clamp(0, 63)
        torch.testing.assert_close(actual[0, 0], expected[:, None].expand(23, 40),
                                   rtol=0, atol=1e-12)
        rounded_crop = F.interpolate(self.coordinate_field()[..., :38, :],
                                     size=(23, 40), mode='bilinear', align_corners=False)
        self.assertGreater(float((actual[:, :1] - rounded_crop[:, :1]).abs().max()), .2)

    def test_invalid_view_or_native_grid_is_rejected(self):
        for resized, output in (((0, 1024), (23, 40)), ((1025, 1024), (23, 40)),
                                ((576, 1024), (0, 40))):
            with self.subTest(resized=resized, output=output), self.assertRaises(ValueError):
                align_sam_features(self.coordinate_field(), resized, output)
        with self.assertRaisesRegex(ValueError, 'full padded image grid'):
            align_sam_features(torch.zeros(1, 256, 36, 64), (576, 1024), (23, 40))


class InputAndConfiguration(LocalAssets):
    def test_uint8_transform_preserves_rgb_and_configured_geometry(self):
        # Source dimensions differ deliberately: B0's configured geometry wins.
        source = Image.new('RGB', (37, 59), color=(11, 47, 203))
        transform = SAMRGBTransform((720, 1280))
        image = transform(source)
        self.assertEqual(transform.resized_hw, (576, 1024))
        self.assertEqual(tuple(image.shape), (3, 576, 1024))
        self.assertEqual(image.dtype, torch.uint8)
        self.assertTrue(image.is_contiguous())
        self.assertTrue(torch.equal(image[:, 123, 456], torch.tensor([11, 47, 203], dtype=torch.uint8)))
        self.assertEqual(SAMRGBTransform((753, 1280)).resized_hw, (602, 1024))

    def test_argument_registration_does_not_require_assets_and_rejects_wrong_modes(self):
        parser = argparse.ArgumentParser()
        add_sam_encoder_arguments(parser)
        defaults = parser.parse_args([])
        self.assertEqual(defaults.sam_encoder_chunk_size, 1)
        self.assertIn('--sam_checkpoint', parser.format_help())
        self.assertTrue(is_sam_backbone({'backbone': 'sam_vit_b'}))
        ordinary = argparse.Namespace(backbone='resnet18')
        validate_sam_encoder_args(ordinary)  # Legacy runs require no SAM paths.
        self.assertFalse(hasattr(ordinary, 'freeze_backbone'))
        validate_sam_encoder_args(self.args)
        self.assertTrue(self.args.freeze_backbone)
        for name, value in (('research_b0_repro', False), ('no_mae', False),
                            ('heatmap_variant', 'h1_sup'), ('train_mask_mode', 'sam_roi'),
                            ('dilation', True), ('unfreeze_blocks', 1),
                            ('sam_encoder_chunk_size', 0), ('sam_output_height', 0),
                            ('image_width', -1), ('sam_checkpoint_sha256', 'bad-digest')):
            candidate = copy.copy(self.args)
            setattr(candidate, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_sam_encoder_args(candidate)

    def test_missing_assets_and_stale_sha_fail_before_encoder_construction(self):
        missing = copy.copy(self.args)
        missing.sam_checkpoint = str(self.directory / 'missing.pth')
        with self.assertRaises(FileNotFoundError):
            SAMImageBackbone(missing)
        self.builder.assert_not_called()
        self.args.sam_checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        validate_sam_encoder_args(self.args)
        self.write_checkpoint({'scale': torch.zeros(256), 'bias': torch.zeros(256)})
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            validate_sam_encoder_args(self.args)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            SAMImageBackbone(self.args)
        self.builder.assert_not_called()
        (self.repo / 'segment_anything/build_sam.py').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'source'):
            validate_sam_encoder_args(arguments(self.repo, self.checkpoint))

    def test_resume_rejects_cross_backbone_and_changed_gad_geometry(self):
        sam_state = {'backbone.0.image_encoder.scale': torch.ones(256)}
        checkpoint = dict(state_dict=sam_state, model_config=sam_resume_config(self.args))
        validate_sam_resume(self.args, checkpoint)
        validate_sam_resume(self.args, {'state_dict': sam_state})  # Historical file.
        other = copy.copy(self.args)
        other.backbone = 'resnet18'
        with self.assertRaisesRegex(ValueError, 'backbone'):
            validate_sam_resume(other, checkpoint)
        with self.assertRaisesRegex(ValueError, 'backbone'):
            validate_sam_resume(self.args, {'state_dict': {'backbone.0.body.weight': torch.ones(1)}})
        other = copy.copy(self.args)
        other.sam_output_width += 1
        with self.assertRaisesRegex(ValueError, 'sam_output_width'):
            validate_sam_resume(other, checkpoint)
        with self.assertRaisesRegex(ValueError, 'no SAM image_encoder'):
            validate_sam_resume(self.args, dict(model_config=sam_resume_config(self.args), state_dict={}))


class FrozenBackbone(LocalAssets):
    def test_strict_pretrained_loading_formats_and_invalid_tensor_sets(self):
        full = {'image_encoder.' + key: value for key, value in self.encoder_state.items()}
        formats = (
            ('full_sam', full),
            ('image_encoder_only', self.encoder_state),
            ('full_sam', {'state_dict': {'module.' + key: value for key, value in full.items()}}),
        )
        for expected_format, state in formats:
            with self.subTest(format=expected_format, wrapped='state_dict' in state):
                self.write_checkpoint(state)
                backbone = SAMImageBackbone(self.args)
                self.assertEqual(backbone.metadata['checkpoint_format'], expected_format)
                for name, value in backbone.image_encoder.state_dict().items():
                    torch.testing.assert_close(value, self.encoder_state[name], rtol=0, atol=0)
                self.assertEqual(backbone.metadata['loaded_encoder_tensors'], 2)
                self.assertEqual(backbone.metadata['checkpoint_sha256'], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        for invalid in ({'scale': torch.ones(256)},
                        {'scale': torch.ones(128), 'bias': torch.ones(256)},
                        dict(self.encoder_state, unexpected=torch.ones(1))):
            with self.subTest(keys=list(invalid)), self.assertRaises(RuntimeError):
                self.write_checkpoint(invalid)
                SAMImageBackbone(self.args)

    def test_chunking_preserves_frame_order_normalization_rng_and_frozen_eval(self):
        frames = torch.tensor([7, 53, 119, 211], dtype=torch.uint8)
        images = frames[:, None, None, None].expand(-1, 3, 576, 1024).contiguous()
        rng = torch.get_rng_state().clone()
        backbone = SAMImageBackbone(self.args)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(backbone.chunk_size, 1)
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.image_encoder.parameters()))
        backbone.train()
        self.assertTrue(backbone.training)
        self.assertFalse(backbone.image_encoder.training)
        reference = backbone(images)
        self.assertFalse(reference.requires_grad)
        self.assertEqual(tuple(reference.shape), (4, 256, 23, 40))
        self.assertEqual([call['batch'] for call in backbone.image_encoder.calls], [1, 1, 1, 1])
        normalized = (frames.float()[:, None] - backbone.pixel_mean.flatten()) / backbone.pixel_std.flatten()
        torch.testing.assert_close(torch.cat([call['top_left'] for call in backbone.image_encoder.calls]), normalized)
        for call in backbone.image_encoder.calls:
            self.assertFalse(call['training'])
            self.assertFalse(call['grad_enabled'])
            self.assertEqual(float(call['bottom_right'].abs().sum()), 0)  # pad after normalizing
        expected = normalized.mean(1) * self.encoder_state['scale'][0] + self.encoder_state['bias'][0]
        torch.testing.assert_close(reference[:, 0, 0, 0], expected)
        for chunk_size, expected_batches in ((2, [2, 2]), (3, [3, 1])):
            backbone.chunk_size = chunk_size
            backbone.image_encoder.calls.clear()
            actual = backbone(images)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            self.assertEqual([call['batch'] for call in backbone.image_encoder.calls], expected_batches)
        self.assertEqual(backbone.last_feature_shapes['valid_feature_hw'], [36., 64.])

    def test_legacy_normalized_float_and_wrong_shapes_cannot_enter_sam(self):
        backbone = SAMImageBackbone(self.args)
        for images, expected in (
            (torch.zeros(1, 3, 576, 1024), 'RGB uint8'),
            (torch.zeros(1, 3, 720, 1280, dtype=torch.uint8), 'longest side'),
            (torch.zeros(0, 3, 576, 1024, dtype=torch.uint8), 'nonempty'),
            (torch.zeros(1, 1, 576, 1024, dtype=torch.uint8), 'nonempty'),
        ):
            with self.subTest(shape=images.shape, dtype=images.dtype), self.assertRaisesRegex(ValueError, expected):
                backbone(images)
        self.assertEqual(backbone.image_encoder.calls, [])


class GADIntegration(LocalAssets):
    def test_real_model_loss_backward_actor_equivariance_and_checkpoint_roundtrip(self):
        # The production model is built unchanged, including all six GAD layers.
        # A ResNet constructor or a network weight fetch would fail this test.
        with mock.patch('models.backbone.torchvision.models.resnet18',
                        side_effect=AssertionError('SAM path constructed ResNet')), \
                mock.patch('torch.hub.load_state_dict_from_url',
                           side_effect=AssertionError('CPU contract tests must not download')):
            model, criterion = build_model(self.args)
        self.assertEqual(model.backbone.num_channels, 256)
        self.assertEqual(model.fc_emb.in_features, 256 * 5 * 5)
        self.assertEqual(model.input_proj.in_channels, 256)
        self.assertFalse(any(module.__class__.__name__ == 'ResNet' for module in model.modules()))
        self.assertFalse(hasattr(model.backbone[0], 'prompt_encoder'))
        self.assertFalse(hasattr(model.backbone[0], 'mask_decoder'))

        h, w, t, n = 576, 1024, self.args.num_frame, self.args.num_boxes
        pattern = (torch.arange(h)[:, None] + 2 * torch.arange(w)[None, :]) % 192
        images = torch.stack([(pattern + 8 * frame).to(torch.uint8).expand(3, -1, -1)
                              for frame in range(t)])[None].contiguous()
        boxes = torch.zeros(1, t, n, 4)
        boxes[:, :, :4] = torch.tensor([[.2, .4, .14, .3], [.35, .45, .16, .35],
                                         [.65, .55, .14, .3], [.8, .6, .12, .25]])
        dummy = torch.ones(1, n, dtype=torch.bool)
        dummy[:, :4] = False
        members = torch.zeros(1, 2, n)
        members[0, 0, :2] = 1
        members[0, 1, 2:4] = 1
        actions = torch.full((1, n), self.args.num_class + 1, dtype=torch.long)
        actions[0, :4] = torch.tensor([0, 0, 1, 1])
        target = dict(actions=actions, activities=torch.tensor([[0, 1]]), members=members,
                      membership=torch.tensor([[0, 0, 1, 1] + [-1] * (n - 4)]),
                      boxes=boxes[0])

        model.train()
        outputs = model(images, boxes, dummy)
        self.assertFalse(model.backbone[0].image_encoder.training)
        expected_shapes = dict(pred_actions=(1, n, 7), pred_activities=(1, 12, 7),
                               membership=(1, 12, n), actor_embeddings=(1, n, 256))
        self.assertEqual({key: tuple(value.shape) for key, value in outputs.items()}, expected_shapes)
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))
        terms = criterion(outputs, [copy.deepcopy(target)])
        self.assertTrue(all(torch.isfinite(value).all() for value in terms.values()))
        loss = sum(terms[key] * weight for key, weight in criterion.weight_dict.items())
        loss.backward()
        for prefix in ('input_proj.', 'fc_emb.', 'group_transformer.'):
            gradients = [parameter.grad for name, parameter in model.named_parameters()
                         if name.startswith(prefix) and parameter.grad is not None]
            self.assertTrue(gradients, prefix)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients), prefix)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0, prefix)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad
                            for parameter in model.backbone[0].image_encoder.parameters()))
        optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            reference = model(images, boxes, dummy)
            permutation = torch.tensor([2, 0, 13, 1, 3] + list(range(4, 13)))
            permuted = model(images, boxes[:, :, permutation], dummy[:, permutation])
        torch.testing.assert_close(permuted['pred_activities'], reference['pred_activities'], rtol=2e-5, atol=2e-5)
        for name, axis in (('pred_actions', 1), ('membership', 2), ('actor_embeddings', 1)):
            torch.testing.assert_close(permuted[name], reference[name].index_select(axis, permutation),
                                       rtol=2e-5, atol=2e-5)

        buffer = io.BytesIO()
        torch.save(dict(state_dict=model.state_dict(), model_config=sam_resume_config(self.args),
                        optimizer=optimizer.state_dict(), sam_encoder=model.backbone[0].metadata), buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)
        validate_sam_resume(self.args, checkpoint)
        restored, _ = build_model(self.args)
        restored.load_state_dict(checkpoint['state_dict'], strict=True)
        restored.eval()
        with torch.no_grad():
            actual = restored(images, boxes, dummy)
        for key in reference:
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        self.assertEqual(restored.backbone[0].metadata['checkpoint_sha256'],
                         checkpoint['sam_encoder']['checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
