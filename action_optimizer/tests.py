"""
Run with:

    python action_optimizer/tests.py

"""

import unittest
from pprint import pprint

from optimizer import Optimizer


class Tests(unittest.TestCase):

    def test_causal_trend(self):
        """
        An action is performed (supp_alpha) consecutively that gradually improves the score,
        then that action is halted and the score gradually diminishes.
        Confirm we detect this causal relation.
        """
        o = Optimizer(fn='fixtures/test-trend.ods', yes=True, stop_on_error=True)

        final_recommendations, final_scores = o.analyze(save=False)
        print('final_recommendations:')
        pprint(final_recommendations, indent=4)
        print('final_scores:')
        pprint(final_scores, indent=4)
        print('column_predictables:')
        pprint(o.column_predictables, indent=4)

        # Metrics that aren't listed as features to predict shouldn't be marked as predictable.
        self.assertEqual(o.column_predictables['metric_a_strong'], 0)
        self.assertEqual(o.column_predictables['metric_b_weak'], 0)
        self.assertEqual(o.column_predictables['metric_c_none'], 0)

        # Metrics we explicitly want to predict should be marked as predictable.
        self.assertEqual(o.column_predictables['supp_alpha'], 1)
        self.assertEqual(o.column_predictables['supp_beta'], 1)

        # supp_alpha has a direct proportional positive improvement on the score, so algorithm should detect this and recommend futher use.
        # supp_beta has no effect on score, so algorithm should not recommend it more highly than supp_alpha.
        self.assertEqual(sorted(final_scores.keys()), ['bed', 'sleep_hours', 'supp_alpha', 'supp_beta'])
        self.assertTrue(final_scores['supp_alpha'][0] > final_scores['supp_beta'][0])
        self.assertTrue(final_scores['supp_alpha'][0] > final_scores['sleep_hours'][0])
        self.assertTrue(final_scores['supp_alpha'][0] > final_scores['bed'][0])

if __name__ == '__main__':
    unittest.main()
