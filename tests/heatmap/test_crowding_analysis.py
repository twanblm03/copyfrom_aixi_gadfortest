"""Saved-export crowding analysis: geometry and one-to-one membership accounting."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analyze_heatmap_crowding_control import geometry, membership_counts, aggregate_counts


class CrowdingAnalysis(unittest.TestCase):
    def raw(self):
        return dict(boxes=[[[.5, .5, .1, .1], [.51, .5, .1, .1]]],
                    input_track_valid=[[True, True]], target_members=[[True, False]])

    def test_group_individual_overlap_and_distance(self):
        result = geometry(self.raw())
        self.assertTrue(result['crowded_iou_0_3'])
        self.assertEqual(result['distance_bucket'], '<1')

    def test_same_group_and_individual_pair_not_relevant(self):
        raw = self.raw(); raw['target_members'] = [[True, True]]
        self.assertEqual(geometry(raw)['distance_bucket'], 'no_relevant_pair')
        self.assertFalse(geometry(raw)['crowded_iou_0_3'])
        raw['target_members'] = [[False, False]]
        self.assertFalse(geometry(raw)['crowded_iou_0_3'])

    def test_not_simultaneously_valid(self):
        raw = self.raw()
        raw['boxes'] *= 2; raw['input_track_valid'] = [[True, False], [False, True]]
        self.assertFalse(geometry(raw)['crowded_iou_0_3'])

    def test_one_to_one_unmatched_errors(self):
        r = membership_counts([[0, 1, 2, 3]], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (2, 4, 2))
        self.assertEqual(r['cross_group_wrong_assignments'], 0)
        self.assertEqual(r['unmatched_predicted_gt_member_assignments'], 2)
        self.assertEqual(r['individual_false_assignments'], 2)
        self.assertEqual(r['group_iou_sum'], .5)

    def test_permutation_and_no_predictions(self):
        r = membership_counts([[0, 1], [2, 3]], [[2, 3], [0, 1]])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (4, 0, 0))
        r = membership_counts([[0, 1]], [])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (0, 0, 2))
        self.assertEqual(aggregate_counts([r])['member_f1'], 0)
        self.assertIsNone(aggregate_counts([r])['member_precision'])

    def test_reject_overlapping_groups(self):
        with self.assertRaises(AssertionError):
            membership_counts([[0, 1]], [[0, 1], [1, 2]])


if __name__ == '__main__':
    unittest.main()
