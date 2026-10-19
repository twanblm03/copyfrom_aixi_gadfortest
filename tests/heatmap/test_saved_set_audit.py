"""Saved exports support decoded-set diagnostics, not raw slot margins."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.analyze_heatmap_saved_sets import decoded_groups,summary


class SavedSetAudit(unittest.TestCase):
    def diagnose(self,gids,labels):
        return decoded_groups([[1,1,0]],[1],np.ones(3,dtype=bool),np.ones(3,dtype=bool),
            np.array(gids),np.array(labels),(13,0))

    def test_individual_contamination_not_cross_gt_merge(self):
        actors,groups=self.diagnose([0,0,0],[2,2,2])
        result=summary(actors,groups)
        self.assertEqual(result['individual_error_counts']['individual_false_group_member'],1)
        self.assertEqual(groups[0]['category'],'extra_members_only')
        self.assertTrue(groups[0]['contaminated_by_individual'])
        self.assertFalse(groups[0]['merged_with_other_gt_group'])

    def test_original_rejection_order_and_exact_set_wrong_activity(self):
        actors,groups=self.diagnose([0,0,-1],[3,3,7])
        self.assertEqual(groups[0]['category'],'exact_set_wrong_activity')
        self.assertEqual(actors[2]['category'],'individual_rejected_threshold')
        actors,groups=self.diagnose([0,0,2],[2,2,7])
        self.assertEqual(groups[0]['category'],'exact_set_correct_activity')
        self.assertEqual(actors[2]['category'],'individual_rejected_no_group')


if __name__=='__main__':unittest.main()
