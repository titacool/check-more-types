"""参考答案里纯计算部分的测试。不访问行情。"""

import math
import statistics
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution"))

import bsm_50etf as bsm


class BlackScholesMertonTest(unittest.TestCase):
    def test_textbook_prices_without_dividend(self):
        args = (100, 100, 1, 0.05, 0.2, 0)
        self.assertAlmostEqual(bsm.bsm_price(*args, "call"), 10.4506, places=4)
        self.assertAlmostEqual(bsm.bsm_price(*args, "put"), 5.5735, places=4)
        self.assertAlmostEqual(bsm.bsm_delta(*args, "call"), 0.6368, places=4)

    def test_textbook_prices_with_dividend_yield(self):
        args = (100, 100, 1, 0.05, 0.2, 0.02)
        self.assertAlmostEqual(bsm.bsm_price(*args, "call"), 9.2270, places=4)
        self.assertAlmostEqual(bsm.bsm_price(*args, "put"), 6.3301, places=4)
        self.assertAlmostEqual(bsm.bsm_delta(*args, "call"), 0.5869, places=4)

    def test_put_call_parity_holds_for_the_model(self):
        args = dict(
            spot=2.917,
            strike=2.9,
            year_fraction=15 / 365,
            rate=math.log(1.0135),
            volatility=0.18,
            dividend_yield=0.0,
        )
        call = bsm.bsm_price(*args.values(), "call")
        put = bsm.bsm_price(*args.values(), "put")
        gap = bsm.put_call_parity_gap(
            call,
            put,
            args["spot"],
            args["strike"],
            args["year_fraction"],
            args["rate"],
            args["dividend_yield"],
        )
        self.assertAlmostEqual(gap, 0.0, places=10)

    def test_call_price_falls_when_dividend_yield_rises(self):
        base = (2.917, 2.9, 43 / 365, 0.014, 0.2)
        cheap_dividend = bsm.bsm_price(*base, 0.0, "call")
        rich_dividend = bsm.bsm_price(*base, 0.08, "call")
        self.assertGreater(cheap_dividend, rich_dividend)

    def test_call_price_rises_with_volatility(self):
        base = (2.917, 2.9, 43 / 365, 0.014, 0.0)
        low_vol = bsm.bsm_price(base[0], base[1], base[2], base[3], 0.1, base[4], "call")
        high_vol = bsm.bsm_price(base[0], base[1], base[2], base[3], 0.3, base[4], "call")
        self.assertGreater(high_vol, low_vol)

    def test_historical_volatility_uses_sample_standard_deviation(self):
        returns = [0.01, -0.02, 0.015, 0.0, -0.005]
        price = 100.0
        closes = [price]
        for value in returns:
            price *= math.exp(value)
            closes.append(price)
        expected = statistics.stdev(returns) * math.sqrt(252)
        self.assertAlmostEqual(
            bsm.annualized_volatility(pd.Series(closes), window=len(returns)),
            expected,
            places=10,
        )

    def test_shibor_tenor_picks_the_nearest_maturity(self):
        self.assertEqual(bsm.choose_shibor_tenor(15), ("2W", "2W-定价"))
        self.assertEqual(bsm.choose_shibor_tenor(43), ("1M", "1M-定价"))
        self.assertEqual(bsm.choose_shibor_tenor(1), ("O/N", "O/N-定价"))

    def test_shibor_percent_quote_becomes_continuous_rate(self):
        self.assertAlmostEqual(bsm.shibor_to_continuous(1.35), math.log(1.0135), places=12)

    def test_no_dividend_yield_when_next_ex_date_is_after_expiry(self):
        dividends = pd.DataFrame(
            {"日期": ["2024-12-02", "2025-12-17"], "累计分红": [0.717, 0.797]}
        )
        result = bsm.dividend_yield_for_contract(
            dividends,
            spot=2.917,
            as_of=date(2026, 10, 13),
            expiry=date(2026, 10, 28),
            year_fraction=15 / 365,
        )
        self.assertEqual(result["last_ex_date"], date(2025, 12, 17))
        self.assertEqual(result["next_ex_date"], date(2026, 12, 17))
        self.assertAlmostEqual(result["cash_dividend"], 0.08)
        self.assertFalse(result["dividend_before_expiry"])
        self.assertEqual(result["dividend_yield"], 0.0)

    def test_dividend_yield_when_ex_date_falls_inside_the_contract(self):
        dividends = pd.DataFrame(
            {"日期": ["2024-12-02", "2025-12-17"], "累计分红": [0.717, 0.797]}
        )
        year_fraction = 22 / 365
        result = bsm.dividend_yield_for_contract(
            dividends,
            spot=2.917,
            as_of=date(2026, 12, 1),
            expiry=date(2026, 12, 23),
            year_fraction=year_fraction,
        )
        expected = math.log(2.917 / (2.917 - 0.08)) / year_fraction
        self.assertTrue(result["dividend_before_expiry"])
        self.assertAlmostEqual(result["dividend_yield"], expected, places=10)

    def test_choose_expiry_skips_a_contract_that_is_too_close(self):
        contracts = [
            {"month": "202610", "expiry": date(2026, 10, 16)},
            {"month": "202611", "expiry": date(2026, 11, 25)},
        ]
        selected = bsm.choose_expiry(contracts, date(2026, 10, 13), min_days=7)
        self.assertEqual(selected["month"], "202611")
        self.assertEqual(selected["calendar_days"], 43)

    def test_pick_lower_strike_when_distance_ties(self):
        board = pd.DataFrame(
            {
                "合约交易代码": [
                    "510050C2610M02800",
                    "510050C2610M03000",
                    "510050P2610M02800",
                ],
                "当前价": [0.12, 0.02, 0.01],
                "前结价": [0.12, 0.02, 0.01],
                "行权价": [2.8, 3.0, 2.8],
            }
        )
        chosen = bsm.pick_nearest_contract(board, spot=2.9, right="C")
        self.assertEqual(chosen["合约交易代码"], "510050C2610M02800")

    def test_zero_last_price_falls_back_to_previous_settlement(self):
        row = pd.Series({"合约交易代码": "510050C2610M02900", "当前价": 0.0, "前结价": 0.0524})
        self.assertAlmostEqual(bsm.quoted_premium(row), 0.0524)

    def test_conclusion_names_the_market_as_cheaper_when_model_is_higher(self):
        text = bsm.build_conclusion(0.05, 0.04, 0.18, 0.0, "2W")
        self.assertIn("市场报价更便宜", text)
        self.assertIn("不是交易建议", text)

    def test_conclusion_calls_a_tiny_gap_close(self):
        text = bsm.build_conclusion(0.0455, 0.0458, 0.1508, 0.0, "2W")
        self.assertIn("非常接近", text)
        self.assertIn("模型略低", text)
        self.assertNotIn("更贵", text)


if __name__ == "__main__":
    unittest.main()
