"""不联网的规则测试：命名、持续期、滞后仓位和业绩对照。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "参考答案"))
import markov_regime_csi300 as model


class RegimeRuleTests(unittest.TestCase):
    def test_higher_variance_is_named_turbulent(self):
        params = pd.Series({"sigma2[0]": 0.0004, "sigma2[1]": 0.0001})
        names = model.name_regimes(params)
        self.assertEqual(names["动荡"], 0)
        self.assertEqual(names["平稳"], 1)

    def test_equal_variance_is_rejected(self):
        params = pd.Series({"sigma2[0]": 0.0002, "sigma2[1]": 0.0002})
        with self.assertRaises(RuntimeError):
            model.name_regimes(params)

    def test_duration_matches_stay_probability(self):
        self.assertAlmostEqual(model.expected_duration(0.98), 50.0)
        self.assertTrue(np.isinf(model.expected_duration(1.0)))

    def test_transition_rows_sum_to_one(self):
        params = pd.Series({"p[0->0]": 0.90, "p[1->0]": 0.20})
        matrix = model.transition_matrix(params)
        self.assertAlmostEqual(matrix.loc[0, 0], 0.90)
        self.assertAlmostEqual(matrix.loc[0, 1], 0.10)
        self.assertAlmostEqual(matrix.loc[1, 0], 0.20)
        self.assertAlmostEqual(matrix.loc[1, 1], 0.80)

    def test_trailing_true_count_stops_at_the_first_false(self):
        mask = pd.Series([True, False, True, True])
        self.assertEqual(model.trailing_true_count(mask), 2)
        self.assertEqual(model.trailing_true_count(pd.Series([False, False])), 0)

    def test_position_uses_yesterday_probability(self):
        probability = pd.Series([0.1, 0.8, 0.9, 0.2], index=pd.bdate_range("2024-01-02", periods=4))
        position = model.position_from_filtered(probability, threshold=0.5)
        self.assertListEqual(position.tolist(), [1.0, 1.0, 0.5, 0.5])

    def test_performance_compounds_half_position(self):
        returns = pd.Series([0.10, -0.10, 0.10])
        position = pd.Series([1.0, 1.0, 0.5])
        stats = model.performance(returns, position)
        self.assertAlmostEqual(stats["cumulative"], 1.1 * 0.9 * 1.05 - 1.0)
        self.assertAlmostEqual(stats["max_drawdown"], -0.10)
        self.assertAlmostEqual(stats["average_position"], 5.0 / 6.0)

    def test_risk_clause_does_not_claim_a_missing_improvement(self):
        rule = {"ann_vol": 0.20, "max_drawdown": -0.40, "ann_return": 0.05}
        hold = {"ann_vol": 0.18, "max_drawdown": -0.30, "ann_return": 0.08}
        text = model.risk_clause(rule, hold)
        self.assertIn("都没有改善", text)
        self.assertIn("年化收益低于一直满仓", text)

    def test_two_state_fit_separates_a_simulated_market(self):
        rng = np.random.default_rng(7)
        observations = 900
        state = np.zeros(observations, dtype=int)
        for time in range(1, observations):
            stay = 0.97 if state[time - 1] == 0 else 0.90
            state[time] = state[time - 1] if rng.random() < stay else 1 - state[time - 1]
        mean = np.array([0.0004, -0.0008])
        scale = np.array([0.004, 0.02])
        returns = pd.Series(mean[state] + scale[state] * rng.normal(size=observations))
        result = model.fit_two_state(returns)
        names = model.name_regimes(result.params)
        calm = float(result.params[f"sigma2[{names['平稳']}]"])
        turbulent = float(result.params[f"sigma2[{names['动荡']}]"])
        self.assertGreater(turbulent, calm * 4)
        filtered = model.state_probability(result, names["动荡"], "filtered")
        smoothed = model.state_probability(result, names["动荡"], "smoothed")
        self.assertAlmostEqual(float(filtered.iloc[-1]), float(smoothed.iloc[-1]), places=6)
        self.assertGreaterEqual(float(filtered.min()), 0.0)
        self.assertLessEqual(float(filtered.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
