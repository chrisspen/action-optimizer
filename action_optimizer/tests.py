"""
Run with:

    python action_optimizer/tests.py Tests.test_foo

"""

import os
import unittest
from pprint import pprint

from optimizer import Optimizer

SHOW_GRAPH = int(os.environ.get('SHOW_GRAPH', 0))


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

    def test_curvefit_linear(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        from sklearn.metrics import r2_score

        from optimizer import fit_linear, linear_func, sigmoid_func, guassian_func

        plt.clf()

        # Generate pure sigmoid curve.
        x = np.linspace(-10, 10, 100)
        pure = linear_func(x)
        # pure = linear_func(x+15) # Add horizontal offset.
        # pure = linear_func(x) + 10 # Add vertical offset.
        pure = linear_func(x + 15) + 10 # Add horizontal and vertical offset.
        plt.plot(x, pure, label='Pure')

        # Add noise to guassian curve.
        signal = pure + np.random.normal(scale=1, size=len(x))
        plt.scatter(x, signal, label='Pure + Noise', color='red', marker='.')

        # Estimate the original curve from the noise.
        estimate = fit_linear(x, signal)
        plt.plot(x, estimate, linewidth=2, label='Fit')

        if SHOW_GRAPH:
            plt.legend()
            plt.show()

        # Calculate error.
        cod = r2_score(pure, estimate)
        # print('cod:', cod)
        self.assertEqual(round(cod), 1.0)

        # Confirm no other curves fit as well.
        for _func in [sigmoid_func, guassian_func]:
            other_cod = r2_score(pure, _func(x, signal))
            # print('func:', _func, other_cod)
            self.assertNotEqual(round(other_cod), 1.0)

    def test_curvefit_sigmoid(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        from sklearn.metrics import r2_score

        from optimizer import fit_sigmoid, linear_func, sigmoid_func, guassian_func

        plt.clf()

        # Generate pure sigmoid curve.
        x = np.linspace(-100, 100, 1000)
        # pure = sigmoid_func(x)
        # pure = sigmoid_func(x+15) # Add horizontal offset.
        # pure = sigmoid_func(x) + 10 # Add vertical offset.
        pure = sigmoid_func(x + 15) + 10 # Add horizontal and vertical offset.
        plt.plot(x, pure, label='Pure')

        # Add noise to guassian curve.
        signal = pure + np.random.normal(scale=0.05, size=len(x))
        plt.scatter(x, signal, label='Pure + Noise', color='red', marker='.')

        # Estimate the original curve from the noise.
        estimate = fit_sigmoid(x, signal)
        plt.plot(x, estimate, linewidth=2, label='Fit')

        if SHOW_GRAPH:
            plt.legend()
            plt.show()

        # Calculate error.
        cod = r2_score(pure, estimate)
        self.assertEqual(round(cod), 1.0)

        # Confirm no other curves fit as well.
        for _func in [linear_func, guassian_func]:
            other_cod = r2_score(pure, _func(x, signal))
            self.assertNotEqual(round(other_cod), 1.0)

    def test_curvefit_guassian(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        from sklearn.metrics import r2_score

        from optimizer import fit_guassian, linear_func, sigmoid_func, guassian_func

        plt.clf()

        # Generate pure guassian curve.
        x = np.linspace(-10, 10, 100)
        pure = guassian_func(x + 5) + 10 # Add horizontal and vertical offset.
        plt.plot(x, pure, label='Pure')

        # Add noise to guassian curve.
        signal = pure + np.random.normal(scale=0.05, size=len(x))
        plt.scatter(x, signal, label='Pure + Noise', color='red', marker='.')

        # Estimate the original curve from the noise.
        estimate = fit_guassian(x, signal)
        plt.plot(x, estimate, linewidth=2, label='Fit')

        # Calculate error.
        cod = r2_score(pure, estimate)
        self.assertEqual(round(cod), 1.0)

        # Confirm no other curves fit as well. These numbers should have high absolute values.
        for _func in [linear_func, sigmoid_func]:
            other_cod = r2_score(pure, _func(x, signal))
            self.assertNotEqual(round(other_cod), 1.0)

        if SHOW_GRAPH:
            plt.legend()
            plt.show()


if __name__ == '__main__':
    unittest.main()
