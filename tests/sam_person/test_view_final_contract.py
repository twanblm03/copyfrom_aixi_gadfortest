import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.validate_view_final_contract import (
    ContractError, EVALUATION_POLICY, EVALUATOR_SHA256, ROOT, VARIANTS,
    validate_contract, validate_runtime_policy,
)


def reviewed_args():
    folder = ROOT / 'configs/view_final_dp4'
    plan = json.loads((folder / 'contract.plan.json').read_text())
    references, planned = {}, {}
    for variant in VARIANTS:
        references[variant] = json.loads(Path(plan['runs'][variant]['reference_args']).read_text())
        planned[variant] = json.loads((folder / (variant + '.args.json')).read_text())
    return references, planned


class FinalViewContract(unittest.TestCase):
    def validate(self, references, planned, policy=EVALUATION_POLICY):
        return validate_contract(references, planned, EVALUATOR_SHA256, policy)

    def test_reviewed_final_runs_preserve_105_original_shared_fields(self):
        references, planned = reviewed_args()
        result = self.validate(references, planned)
        self.assertEqual(result['shared_original_fields_checked'], 105)
        self.assertEqual(result['shared_fields_checked'], 107)
        self.assertEqual(result['evaluation_policy']['epochs'], list(range(1, 31)))
        self.assertEqual(result['evaluation_policy']['checkpoint'], 'best_test.pth')
        for variant in VARIANTS:
            self.assertEqual(len(references[variant]), 115)
            self.assertEqual(len(planned[variant]), 117)

    def test_previous_once_only_policy_is_rejected(self):
        references, planned = reviewed_args()
        once_policy = dict(EVALUATION_POLICY, epochs=[30], checkpoint='epoch30.pth')
        with self.assertRaisesRegex(ContractError, 'best-on-TEST'):
            self.validate(references, planned, once_policy)

    def test_from_scratch_test_selection_and_retention_cannot_be_disabled(self):
        for field, value in [('load_model', True), ('model_path', 'epoch4.pth'),
                             ('best_on_test', False), ('keep_all_checkpoints', False),
                             ('val_mode', True), ('protocol', 'final_test'),
                             ('test_freq', 2), ('epochs', 10)]:
            with self.subTest(field=field):
                references, planned = reviewed_args()
                planned['b0'][field] = value
                with self.assertRaises(ContractError):
                    self.validate(references, planned)

    def test_original_training_configuration_cannot_drift(self):
        for field, value in [('weight_decay', 1e-4), ('gar_ffn_dim', 1024),
                             ('batch', True), ('heatmap_target_mode', 'crowd')]:
            with self.subTest(field=field):
                references, planned = reviewed_args()
                planned['b0'][field] = value
                with self.assertRaises(ContractError):
                    self.validate(references, planned)

    def test_matching_reference_drift_cannot_hide_method_common_difference(self):
        references, planned = reviewed_args()
        references['sam']['drop_rate'] = 0.2
        planned['sam']['drop_rate'] = 0.2
        with self.assertRaisesRegex(ContractError, 'common training configuration'):
            self.validate(references, planned)

    def test_sam_cache_must_remain_independent_and_use_identical_model(self):
        for field, value in [('sam_cache_root', 'nested'),
                             ('sam_mask_manifest', 'same'),
                             ('sam_cache_version', 'new_model')]:
            with self.subTest(field=field):
                references, planned = reviewed_args()
                if value == 'nested':
                    value = str(Path(references['sam'][field]) / 'final')
                elif value == 'same':
                    value = references['sam'][field]
                planned['sam'][field] = value
                with self.assertRaises(ContractError):
                    self.validate(references, planned)

    def test_stale_evaluator_or_output_collision_is_rejected(self):
        references, planned = reviewed_args()
        with self.assertRaisesRegex(ContractError, 'evaluator SHA256'):
            validate_contract(references, planned, 'legacy', EVALUATION_POLICY)
        planned['sam']['checkpoint_root'] = planned['b0']['checkpoint_root']
        with self.assertRaises(ContractError):
            self.validate(references, planned)

    def test_ast_rejects_silent_once_only_or_ungated_test_selection(self):
        # Exercise the launch gate against the actual trainer structure, with
        # one guard changed at a time. No trainer function is executed.
        source = (ROOT / '.local_data/sam/dp4_randomsegments_lr1/recovery_e27_20261022/source_snapshot/train.py').read_text()
        source = source.replace(
            'should_evaluate = (epoch % args.test_freq == 0) if args.val_mode else (epoch == args.epochs)',
            'should_evaluate = (epoch % args.test_freq == 0) if (args.val_mode or args.best_on_test) else (epoch == args.epochs)')
        source = source.replace('                if args.val_mode:\n                    current_value',
                                '                if args.val_mode or args.best_on_test:\n                    current_value')
        source += "\n# AST-only declaration of independently checked retention fields.\n"
        source += "_contract_test_checkpoint = 'best_test.pth'\n"
        source += "def _contract_retention(args):\n    return args.keep_all_checkpoints\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.py'
            path.write_text(source)
            self.assertEqual(validate_runtime_policy(path)['status'], 'passed')
            for old, replacement in [
                    ('if (args.val_mode or args.best_on_test) else', 'if args.val_mode else'),
                    ('if args.val_mode or args.best_on_test:\n                    current_value',
                     'if True:\n                    current_value')]:
                with self.subTest(replacement=replacement):
                    path.write_text(source.replace(old, replacement))
                    with self.assertRaises(ContractError):
                        validate_runtime_policy(path)


if __name__ == '__main__':
    unittest.main()
