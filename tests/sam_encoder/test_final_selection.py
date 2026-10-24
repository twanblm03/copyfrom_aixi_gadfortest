"""Run the active trainer's selection policy on CPU, without importing it.

Importing train.py would parse process arguments and import GPU model code.
Compile its real policy statements instead, then exercise complete metric
sequences, CLI constraints and retention decisions with synthetic inputs.
"""
import argparse
import ast
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock


ROOT = Path(__file__).resolve().parents[2]
TREE = ast.parse((ROOT / 'train.py').read_text())
MAIN = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
SPEC = importlib.util.spec_from_file_location(
    'active_checkpoint_selection', str(ROOT / 'util/checkpoint_selection.py'))
SELECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTION)


def statement_code(nodes):
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                   '<active trainer policy>', 'exec')


def assigned_expression(name):
    values = [node.value for node in ast.walk(MAIN)
              if isinstance(node, ast.Assign) and any(
                  isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(values) != 1:
        raise AssertionError('expected one assignment: ' + name)
    return compile(ast.Expression(values[0]), '<active trainer policy>', 'eval')


def policy_parser():
    parser = argparse.ArgumentParser()
    scope = dict(parser=parser, protocol_group=parser.add_mutually_exclusive_group())
    required = {'--val_mode', '--final_test', '--best_on_test', '--keep_all_checkpoints',
                '--split', '--test_freq', '--selection_metric'}
    for node in TREE.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument'
                and call.args and isinstance(call.args[0], ast.Constant)
                and call.args[0].value in required):
            exec(statement_code([node]), scope)
        if (isinstance(call.func, ast.Attribute) and call.func.attr == 'set_defaults'
                and any(item.arg == 'val_mode' for item in call.keywords)):
            exec(statement_code([node]), scope)
    return parser


def apply_protocol(args):
    nodes = []
    for node in MAIN.body:
        if (isinstance(node, ast.If) and node.body and isinstance(node.body[0], ast.Raise)
                and any(isinstance(item, ast.Attribute) and item.attr == 'best_on_test'
                        for item in ast.walk(node.test))):
            nodes.append(node)
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr in ('protocol', 'eval_split')
                for target in node.targets):
            nodes.append(node)
    exec(statement_code(nodes), dict(args=args))
    return args


def selection_scope(args):
    args = apply_protocol(args)
    scope = dict(args=args, selection_mode='max', os=os,
                 checkpoint_dir='/tmp/selection-fixture', best={},
                 selected_epoch_metrics=SELECTION.selected_epoch_metrics,
                 experiment=SimpleNamespace(update_best=Mock(return_value={'legacy': 'VAL behavior'})))
    scope['selection_state'] = eval(assigned_expression('selection_state'), scope)
    return scope


def select(scope, epoch, metrics):
    blocks = [node for node in ast.walk(MAIN) if isinstance(node, ast.If) and node.body
              and isinstance(node.body[0], ast.Assign) and any(
                  isinstance(target, ast.Name) and target.id == 'current_value'
                  for target in node.body[0].targets)]
    if len(blocks) != 1:
        raise AssertionError('expected one primary selection block')
    scope.update(epoch=epoch, eval_metrics=metrics, is_new_best=False)
    exec(statement_code(blocks), scope)


class ActiveFinalSelection(unittest.TestCase):
    def best_test_args(self, extra=()):
        return policy_parser().parse_args(
            ['--split', 'view', '--final_test', '--best_on_test', '--keep_all_checkpoints'] + list(extra))

    def test_default_stays_train_to_val(self):
        args = apply_protocol(policy_parser().parse_args([]))
        self.assertTrue(args.val_mode)
        self.assertFalse(args.best_on_test)
        self.assertFalse(args.keep_all_checkpoints)
        self.assertEqual((args.protocol, args.eval_split), ('development', 'val'))

    def test_best_on_test_rejects_ambiguous_or_different_protocols(self):
        for argv in (['--best_on_test'],
                     ['--best_on_test', '--final_test'],
                     ['--best_on_test', '--final_test', '--split', 'view', '--test_freq', '2'],
                     ['--best_on_test', '--final_test', '--split', 'view', '--selection_metric', 'loss']):
            with self.subTest(argv=argv), self.assertRaisesRegex(ValueError, 'best-on-TEST requires'):
                apply_protocol(policy_parser().parse_args(argv))

    def test_best_on_test_evaluates_all_epochs_and_default_final_only_once(self):
        expression = assigned_expression('should_evaluate')
        scenarios = [
            (self.best_test_args(), list(range(1, 31))),
            (policy_parser().parse_args(['--final_test']), [30]),
            (policy_parser().parse_args(['--test_freq', '2']), list(range(2, 31, 2))),
        ]
        for args, expected in scenarios:
            args.epochs = 30
            apply_protocol(args)
            actual = [epoch for epoch in range(1, 31)
                      if eval(expression, dict(args=args, epoch=epoch))]
            self.assertEqual(actual, expected)

    def test_secondary_peaks_and_primary_ties_do_not_change_selected_checkpoint(self):
        scope = selection_scope(self.best_test_args())
        first = {'group_mAP_1.0': 20., 'group_mAP_0.5': 35., 'outlier_mIoU': 60., 'loss': 1.2}
        select(scope, 6, first)
        original = scope['best']
        select(scope, 7, dict(first, **{'group_mAP_1.0': 19., 'group_mAP_0.5': 50.,
                                     'outlier_mIoU': 75., 'loss': .7}))
        self.assertIs(scope['best'], original)
        self.assertFalse(scope['is_new_best'])
        select(scope, 8, dict(first, **{'group_mAP_0.5': 55.}))
        self.assertIs(scope['best'], original)
        self.assertEqual(scope['selection_state']['epoch'], 6)
        self.assertEqual(scope['selection_state']['value'], 20.)
        self.assertEqual(scope['selection_state']['selection_label'], 'best-on-TEST')
        self.assertEqual(scope['selection_state']['split'], 'test')
        self.assertTrue(scope['selection_state']['checkpoint_path'].endswith('/best_test.pth'))
        scope['experiment'].update_best.assert_not_called()

    def test_primary_improvement_moves_every_reported_metric_to_same_epoch(self):
        scope = selection_scope(self.best_test_args())
        metrics = {'group_mAP_1.0': 20., 'group_mAP_0.5': 35., 'outlier_mIoU': 60., 'loss': 1.2}
        select(scope, 6, metrics)
        improved = dict(metrics, **{'group_mAP_1.0': 21., 'group_mAP_0.5': 32., 'loss': 1.4})
        select(scope, 9, improved)
        self.assertTrue(scope['is_new_best'])
        self.assertEqual(scope['selection_state']['epoch'], 9)
        self.assertEqual(scope['best'], {key: {'epoch': 9, 'value': value}
                                        for key, value in improved.items()})

    def test_val_preserves_original_metric_tracking_and_final_once_does_not_select(self):
        metrics = {'group_mAP_1.0': 20., 'group_mAP_0.5': 35., 'outlier_mIoU': 60., 'loss': 1.2}
        val_scope = selection_scope(policy_parser().parse_args([]))
        select(val_scope, 1, metrics)
        val_scope['experiment'].update_best.assert_called_once_with({}, 1, metrics)
        self.assertEqual(val_scope['best'], {'legacy': 'VAL behavior'})
        self.assertTrue(val_scope['selection_state']['checkpoint_path'].endswith('/best_val.pth'))
        final_scope = selection_scope(policy_parser().parse_args(['--final_test']))
        select(final_scope, 30, metrics)
        self.assertFalse(final_scope['is_new_best'])
        self.assertEqual(final_scope['best'], {})
        self.assertIsNone(final_scope['selection_state']['epoch'])

    def test_keep_all_flag_retains_epoch_files_without_calling_cleanup(self):
        expression = assigned_expression('removed_paths')
        cleanup_node = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
                            and node.name == 'remove_older_checkpoints')
        cleanup_scope = {'os': os}
        exec(statement_code([cleanup_node]), cleanup_scope)
        cleanup = Mock(side_effect=cleanup_scope['remove_older_checkpoints'])
        with tempfile.TemporaryDirectory() as folder:
            for name in ('epoch1.pth', 'epoch2.pth', 'epoch3.pth', 'best_test.pth'):
                (Path(folder) / name).write_bytes(b'checkpoint')
            scope = dict(args=SimpleNamespace(keep_all_checkpoints=True),
                         remove_older_checkpoints=cleanup, checkpoint_dir=folder,
                         checkpoint_path=str(Path(folder) / 'epoch3.pth'))
            self.assertEqual(eval(expression, scope), [])
            cleanup.assert_not_called()
            self.assertEqual(len(list(Path(folder).glob('epoch*.pth'))), 3)
            scope['args'].keep_all_checkpoints = False
            self.assertEqual(len(eval(expression, scope)), 2)
            self.assertEqual(sorted(path.name for path in Path(folder).iterdir()),
                             ['best_test.pth', 'epoch3.pth'])

    def test_curve_reporting_includes_test_loss_and_both_test_maps(self):
        spec = importlib.util.spec_from_file_location(
            'active_experiment_curves', str(ROOT / 'util/experiment.py'))
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        experiment.plt = MagicMock()
        experiment.plot_curves('/tmp/unused-final-selection.png', {
            'train': [{'epoch': 1, 'loss': 2.}],
            'val': [],
            'test': [{'epoch': 1, 'loss': 1.5, 'group_mAP_1.0': 20., 'group_mAP_0.5': 35.}],
        })
        plots = {call.kwargs['label']: call.args for call in experiment.plt.plot.call_args_list}
        self.assertEqual(plots['test_loss'], ([1], [1.5]))
        self.assertEqual(plots['test_mAP@1.0'], ([1], [20.]))
        self.assertEqual(plots['test_mAP@0.5'], ([1], [35.]))


if __name__ == '__main__':
    unittest.main()
