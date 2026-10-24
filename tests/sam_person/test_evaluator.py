import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace

from evaluation.cafe_eval import GAD_Evaluation, group_mAP_eval, outlier_metric


class NumericConfidenceOrder(unittest.TestCase):
    def test_evaluator_restricts_full_gt_to_expected_split(self):
        with tempfile.TemporaryDirectory() as directory:
            groundtruth = Path(directory) / 'gt.txt'
            groundtruth.write_text(
                '1 1 0 0 0 10 10 1 0 1\n'
                '13 2 0 0 0 10 10 1 0 1\n')
            with open(groundtruth) as gt, open('label_map/group_action_list.pbtxt') as labels:
                args = SimpleNamespace(groundtruth=gt, labelmap=labels, eval_type='gt_base')
                evaluator = GAD_Evaluation(args, clip_ids=[(13, 2)])
                self.assertEqual(set(evaluator.gt_groups_ids), {'13,2'})
                with self.assertRaisesRegex(ValueError, 'missing 1 expected clips'):
                    GAD_Evaluation(args, clip_ids=[(13, 3)])

    def test_ap_is_invariant_to_clip_input_order(self):
        gt_ids = {'1,1': [{1: [0, 1]}], '1,2': [{1: [2, 3]}]}
        pred_ids = {'1,1': [{1: [0, 1]}], '1,2': [{2: [0, 1]}]}
        gt_activities = {key: [{1: {0}}] for key in gt_ids}
        pred_activities = {'1,1': [{1: {0}}], '1,2': [{2: {0}}]}
        scores = {'1,1': [{1: {0.9}}], '1,2': [{2: {0.1}}]}
        category = [{'id': 0, 'name': 'activity'}]
        forward = group_mAP_eval(gt_ids, gt_activities, pred_ids,
                                 pred_activities, scores, category, 1.0)
        reverse = group_mAP_eval(gt_ids, gt_activities,
                                 dict(reversed(list(pred_ids.items()))),
                                 pred_activities, scores, category, 1.0)
        self.assertEqual(forward[0], reverse[0])
        self.assertEqual(forward[0], 50.0)

    def test_missing_prediction_clip_keeps_gt_recall_denominator(self):
        gt_ids = {'1,1': [{1: [0, 1]}], '1,2': [{1: [0, 1]}]}
        gt_activities = {key: [{1: {0}}] for key in gt_ids}
        pred_ids = {'1,1': [{1: [0, 1]}]}
        pred_activities = {'1,1': [{1: {0}}]}
        scores = {'1,1': [{1: {0.9}}]}
        category = [{'id': 0, 'name': 'activity'}, {'id': 1, 'name': 'absent'}]
        mean, per_class = group_mAP_eval(gt_ids, gt_activities, pred_ids,
                                          pred_activities, scores, category, 1.0)
        self.assertEqual(per_class.tolist(), [50.0, 0.0])
        self.assertEqual(mean, 25.0)

    def test_missing_prediction_clip_counts_as_outlier_miss(self):
        gt_ids = {'1,1': [{7: [0]}], '1,2': [{7: [0]}]}
        gt_activities = {key: [{7: {7}}] for key in gt_ids}
        pred_ids = {'1,1': [{7: [0]}]}
        pred_activities = {'1,1': [{7: {7}}]}
        self.assertEqual(outlier_metric(gt_ids, gt_activities, pred_ids,
                                        pred_activities, 6), 50.0)


if __name__ == '__main__':
    unittest.main()
