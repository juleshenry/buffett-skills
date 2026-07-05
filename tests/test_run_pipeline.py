import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import run


class TestRunPipeline(unittest.TestCase):
    def test_prepare_evaluator_inputs_marks_market_forecasting_forecast_as_missing(self):
        class MarketForecasting:
            def evaluate(self, forecast_return=None, actual_return=None, ticker=""):
                return {
                    "forecast_return": forecast_return,
                    "actual_return": actual_return,
                    "ticker": ticker,
                }

        evaluator = MarketForecasting()
        kwargs, missing = run._prepare_evaluator_inputs(evaluator, {"ticker": "YUM", "stock_cagr": 0.12})
        self.assertEqual(kwargs, {"ticker": "YUM"})
        self.assertEqual(missing, ["forecast_return"])

    def test_prepare_evaluator_inputs_marks_missing_required_values_for_non_principle_evaluator(self):
        evaluator = run.valuation_capital.MarginOfSafety()
        kwargs, missing = run._prepare_evaluator_inputs(evaluator, {"ticker": "YUM"})
        self.assertEqual(kwargs, {"ticker": "YUM"})
        self.assertEqual(missing, [])

    def test_prepare_evaluator_inputs_marks_missing_required_values(self):
        # We need an evaluator that actually has required arguments without defaults.
        evaluator = run.valuation_capital.IntrinsicValueEstimation()
        kwargs, missing = run._prepare_evaluator_inputs(evaluator, {"ticker": "YUM"})
        # "ticker" gets passed in since it's default "", "years" gets passed in from defaults
        self.assertEqual(kwargs, {"ticker": "YUM"})
        self.assertEqual(missing, [])

    def test_prepare_evaluator_inputs_uses_real_context_values(self):
        evaluator = run.valuation_capital.MarginOfSafety()
        kwargs, missing = run._prepare_evaluator_inputs(
            evaluator,
            {"intrinsic_value": 120.0, "market_price": 100.0},
        )
        self.assertEqual(missing, [])
        self.assertEqual(kwargs["intrinsic_value"], 120.0)
        self.assertEqual(kwargs["market_price"], 100.0)

    def test_make_json_safe_coerces_numpy_scalar_types(self):
        value = {
            "flag": np.bool_(True),
            "count": np.int64(3),
            "ratio": np.float64(1.5),
        }

        result = run._make_json_safe(value)

        self.assertEqual(result, {"flag": True, "count": 3, "ratio": 1.5})

    @patch("run.business_moat.fetch_cpi_inflation_data")
    @patch("run.business_moat.fetch_historical_margins")
    @patch("run.financial_metrics.fetch_deep_financials")
    @patch("run.business_moat.infer_business_details")
    @patch("run.investment_philosophy.analyze_investment_philosophy")
    @patch("run.valuation_capital.fetch_management_commentary")
    @patch("run.thinking_frameworks.fetch_company_info")
    @patch("run.yf.Ticker")
    def test_build_real_context_collects_real_sources(
        self,
        mock_ticker,
        mock_company_info,
        mock_commentary,
        mock_compounding,
        mock_moat_details,
        mock_financials,
        mock_margins,
        mock_inflation,
    ):
        mock_ticker.return_value.info = {
            "freeCashflow": 500.0,
            "sharesOutstanding": 1000,
            "totalCash": 300.0,
            "totalDebt": 450.0,
            "currentPrice": 42.0,
            "trailingPE": 18.0,
            "heldPercentInsiders": 0.06,
            "debtToEquity": 80.0,
        }
        mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [40.0, 42.0]})
        mock_company_info.return_value = {"name": "Yum", "description": "SEC description"}
        mock_commentary.return_value = "SEC commentary"
        mock_compounding.return_value = {
            "ticker": "YUM",
            "period_years": 10.0,
            "stock_cagr": 0.12,
            "benchmark_cagr": 0.10,
        }
        mock_moat_details.return_value = {"products": ["Chicken"], "competitors": ["Peer"]}
        mock_financials.return_value = {
            "total_revenue": 1000.0,
            "net_income": 120.0,
            "operating_cash_flow": 180.0,
            "capex_total": -50.0,
        }
        mock_margins.return_value = pd.DataFrame(
            [{"Year": 2023, "Gross_Margin": 0.40}, {"Year": 2024, "Gross_Margin": 0.42}]
        )
        mock_inflation.return_value = pd.DataFrame([{"Year": 2024, "Inflation_Rate": 3.0}])

        context = run._build_real_context("YUM")

        self.assertEqual(context["company_name"], "Yum")
        self.assertEqual(context["description"], "SEC description")
        self.assertEqual(context["commentary"], "SEC commentary")
        self.assertEqual(context["stock_cagr"], 0.12)
        self.assertEqual(context["benchmark_cagr"], 0.10)
        self.assertEqual(context["products"], ["Chicken"])
        self.assertEqual(context["competitors"], ["Peer"])
        self.assertEqual(context["net_debt"], 150.0)
        self.assertEqual(context["debt_to_equity"], 0.8)
        self.assertEqual(context["gross_margin"], 0.42)

if __name__ == "__main__":
    unittest.main()
