import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from financial_metrics import KeyFinancialMetrics
from financial_metrics import LookthroughEarningsPrinciple
from business_moat import BusinessModelTypes, EconomicMoat, TheDurabilityOfCompetitiveAdvantage
from business_moat import GoodwillEconomicGoodwillVsAccountingGoodwill
from investment_philosophy import IntrinsicValue, UndervaluedMarginOfSafety
from management_governance import CorporateGovernanceAndShareholderOrientation
from management_governance import AcquisitionLogicAcquisitionCriteria, CorporateCulture
from thinking_frameworks import MrMarket
from valuation_capital import CapitalAllocationAnalysis, MarginOfSafety, TheRelationshipBetweenPurchasePriceAndIntrinsicValue
from valuation_capital import DividendsRetainedEarningsAndTaxEfficiency
from valuation_capital import ShareBuybackAnalysis, SpecialInvestmentInstruments
from evaluator_config import (
    DEFAULT_INTRINSIC_VALUE_YEARS,
    _resolve_ollama_model,
    aggregate_panel_judgment,
    call_ollama_panel_json,
)
from sec_data import extract_keyword_context
from risk_behavior import ValueTraps
from risk_behavior import DerivativesRisk
from financial_metrics import CorePrincipleSeeThroughAccountingToEconomicReality, OwnerEarnings, normalize_capex_breakdown
# industry_playbooks was deliberately removed from the universal ticker pipeline
# (industry-specific evaluators require inputs no single pipeline run can supply
# for every ticker) -- see plan.md. Its tests were removed along with it below.
import investment_philosophy
import management_governance
import thinking_frameworks
import valuation_capital
from management_governance import ManagementEvaluation
from risk_behavior import LeverageRisk
from risk_behavior import normalize_footnote_analysis
from valuation_capital import normalize_buyback_analysis
from principles_bot import (
    CommonBehavioralBiasesPrinciple,
    CompoundingPrinciple,
    EfficientMarketTheoryPrinciple,
    FocusInvestingPrinciple,
    IndependentThinkingPrinciple,
    LongtermOrientationPrinciple,
    MarketForecastingPrinciple,
    OpportunityCostAwarenessPrinciple,
    PatienceAsEdgePrinciple,
    WhenToSellPrinciple,
)


class TestMinimalHeuristics(unittest.TestCase):
    def test_ollama_model_aliases(self):
        self.assertEqual(_resolve_ollama_model("llama3"), "llama3")
        self.assertEqual(_resolve_ollama_model("ollama3"), "llama3")
        self.assertEqual(_resolve_ollama_model("gemma"), "gemma4:26b")
        self.assertEqual(_resolve_ollama_model("gemma4"), "gemma4:26b")

    def test_panel_judgment_aggregates_majority_and_average_confidence(self):
        result = aggregate_panel_judgment([
            {"inside_circle": True, "confidence": 80, "explanation": "Simple business.", "_panel_model": "qwen2.5:7b"},
            {"inside_circle": True, "confidence": 70, "explanation": "Understandable.", "_panel_model": "llama3"},
            {"inside_circle": False, "confidence": 40, "explanation": "Too complex.", "_panel_model": "gemma4:26b"},
        ])
        self.assertTrue(result["inside_circle"])
        self.assertEqual(result["confidence"], 63)
        self.assertEqual(result["panel_vote_split"]["inside_circle"], 2)
        self.assertIn("qwen2.5:7b", result["panel_judgments"])
        self.assertEqual(result["panel_judgments"]["llama3"]["confidence"], 70)

    def test_normalize_footnote_analysis_merges_partial_panel_outputs(self):
        result = normalize_footnote_analysis(
            {
                "_panel_model": "qwen2.5:7b",
                "panel_judgments": {
                    "qwen2.5:7b": {},
                    "llama3": {
                        "operating_lease_obligations": None,
                        "pension_underfunding": None,
                    },
                },
                "panel_models": ["qwen2.5:7b", "llama3"],
            }
        )

        self.assertEqual(result["operating_lease_obligations"], "Not found")
        self.assertEqual(result["pension_underfunding"], "Not found")
        self.assertEqual(result["toxic_derivative_exposure"], "None mentioned")
        self.assertEqual(result["panel_judgments"]["qwen2.5:7b"]["operating_lease_obligations"], "Not found")

    def test_normalize_footnote_analysis_accepts_legacy_field_names(self):
        result = normalize_footnote_analysis(
            {
                "total_operating_lease_obligations": 50000000,
                "pension_plan_underfunding_amount": 0,
                "toxic_derivative_exposure": "None",
            }
        )

        self.assertEqual(result["operating_lease_obligations"], 50000000)
        self.assertEqual(result["pension_underfunding"], 0)
        self.assertEqual(result["toxic_derivative_exposure"], "None")

    @patch("thinking_frameworks.fetch_filing_section")
    @patch("thinking_frameworks.call_ollama_panel_json")
    def test_circle_of_competence_uses_panel_of_judges(self, mock_panel_call, mock_fetch_section):
        mock_fetch_section.return_value = "Makes branded beverages and distributes them globally."
        mock_panel_call.return_value = {
            "inside_circle": True,
            "confidence": 63,
            "explanation": "Simple business.",
            "panel_judgments": {
                "qwen2.5:7b": {"inside_circle": True, "confidence": 80, "explanation": "Simple business."},
                "llama3": {"inside_circle": True, "confidence": 70, "explanation": "Easy to understand."},
            },
            "panel_vote_split": {"inside_circle": 2, "outside_circle": 1},
        }

        result = thinking_frameworks.CircleOfCompetence().evaluate("KO")

        self.assertTrue(result["inside_circle"])
        self.assertEqual(result["confidence"], 63)
        self.assertEqual(result["panel_vote_split"]["inside_circle"], 2)
        self.assertIn("qwen2.5:7b", result["panel_judgments"])

    @patch("evaluator_config.time.sleep")
    @patch("evaluator_config.requests.post")
    @patch.dict(os.environ, {"OLLAMA_PANEL_SLEEP_SECONDS": "0.25"}, clear=False)
    def test_call_ollama_panel_json_sleeps_between_models(self, mock_post, mock_sleep):
        responses = [
            {"response": '{"inside_circle": true, "confidence": 80, "explanation": "A"}'},
            {"response": '{"inside_circle": true, "confidence": 70, "explanation": "B"}'},
            {"response": '{"inside_circle": false, "confidence": 40, "explanation": "C"}'},
        ]
        mock_post.side_effect = [MagicMock(json=MagicMock(return_value=payload), raise_for_status=MagicMock()) for payload in responses]

        result = call_ollama_panel_json(
            "prompt",
            model="qwen2.5:7b",
            aggregator=aggregate_panel_judgment,
        )

        self.assertTrue(result["inside_circle"])
        self.assertEqual(mock_sleep.call_count, 2)

    def test_margin_of_safety(self):
        result = MarginOfSafety().evaluate(intrinsic_value=100.0, market_price=70.0)
        self.assertAlmostEqual(result["margin_of_safety"], 0.30)
        self.assertTrue(result["is_discount"])

    def test_margin_of_safety_handles_negative_intrinsic_value(self):
        # Real, observed case: companies in a heavy growth-capex phase can
        # have genuinely negative recent FCF (APD -$3.5B, AES -$3.0B), which
        # produces a nonsensical negative DCF "intrinsic value." This must
        # return the same "not applicable" shape every other evaluator uses
        # for missing/inapplicable data, not raise ValueError.
        result = MarginOfSafety().evaluate(intrinsic_value=-413.12, market_price=160.44)
        self.assertEqual(
            result,
            {"applicable": False, "reason": "Computed intrinsic value is not positive (likely driven by negative free cash flow); margin of safety is not meaningful here"},
        )

    def test_purchase_price_relationship(self):
        result = TheRelationshipBetweenPurchasePriceAndIntrinsicValue().evaluate(
            intrinsic_value=100.0,
            market_price=72.0,
        )
        self.assertAlmostEqual(result["margin_of_safety"], 0.28)
        self.assertEqual(result["purchase_price_verdict"], "deep_discount")

    def test_purchase_price_relationship_handles_negative_intrinsic_value(self):
        result = TheRelationshipBetweenPurchasePriceAndIntrinsicValue().evaluate(
            intrinsic_value=-413.12,
            market_price=160.44,
        )
        self.assertEqual(result, {"applicable": False, "reason": "Computed intrinsic value is not positive (likely driven by negative free cash flow); margin of safety is not meaningful here"})

    def test_value_trap_screen(self):
        result = ValueTraps().evaluate(
            pe_ratio=8.0,
            revenue_growth=-0.03,
            free_cash_flow_growth=-0.10,
            debt_to_equity=1.4,
            return_on_capital=0.05,
        )
        self.assertTrue(result["is_value_trap"])
        self.assertIn("cheap_multiple", result["risk_flags"])

    @patch("risk_behavior.fetch_value_trap_metrics")
    def test_value_traps_fetch_real_metrics_for_ticker(self, mock_fetch_metrics):
        mock_fetch_metrics.return_value = {
            "pe_ratio": 8.0,
            "revenue_growth": -0.03,
            "free_cash_flow_growth": -0.10,
            "debt_to_equity": 1.4,
            "return_on_capital": 0.05,
        }
        result = ValueTraps().evaluate(ticker="AAPL")
        self.assertTrue(result["is_value_trap"])
        mock_fetch_metrics.assert_called_once_with("AAPL")

    @patch("risk_behavior.fetch_earnings_call_risk_commentary")
    def test_value_traps_adds_transcript_risk_flags(self, mock_risk_commentary):
        mock_risk_commentary.return_value = "Management called demand challenging and discussed debt refinancing needs."

        result = ValueTraps().evaluate(
            pe_ratio=8.0,
            revenue_growth=-0.03,
            free_cash_flow_growth=-0.10,
            debt_to_equity=1.4,
            return_on_capital=0.05,
            ticker="AAPL",
        )

        self.assertIn("management_flags_headwinds", result["risk_flags"])
        self.assertIn("management_flags_balance_sheet_risk", result["risk_flags"])
        self.assertIn("earnings_call_risk_commentary", result)

    def test_when_to_sell(self):
        result = WhenToSellPrinciple().evaluate(
            thesis_broken=False,
            better_opportunity_available=True,
            extreme_overvaluation=False,
            balance_sheet_deterioration=True,
        )
        self.assertTrue(result["sell"])
        self.assertEqual(result["primary_reason"], "better_opportunity_available")

    def test_key_financial_metrics(self):
        result = KeyFinancialMetrics().evaluate(
            total_revenue=1000.0,
            net_income=120.0,
            operating_cash_flow=180.0,
            capex_total=-50.0,
        )
        self.assertAlmostEqual(result["profit_margin"], 0.12)
        self.assertAlmostEqual(result["cash_conversion"], 1.5)
        self.assertAlmostEqual(result["free_cash_flow"], 130.0)

    @patch("financial_metrics.fetch_deep_financials")
    def test_key_financial_metrics_fetches_real_financials(self, mock_fetch_financials):
        mock_fetch_financials.return_value = {
            "total_revenue": 1000.0,
            "net_income": 120.0,
            "operating_cash_flow": 180.0,
            "capex_total": -50.0,
        }

        result = KeyFinancialMetrics().evaluate(ticker="AAPL")
        self.assertAlmostEqual(result["profit_margin"], 0.12)
        mock_fetch_financials.assert_called_once_with("AAPL")

    def test_long_term_orientation(self):
        result = LongtermOrientationPrinciple().evaluate(stock_cagr=0.14, benchmark_cagr=0.09, years=7)
        self.assertEqual(result["long_term_orientation"], "strong")
        self.assertAlmostEqual(result["excess_return"], 0.05)

    def test_opportunity_cost_awareness(self):
        result = OpportunityCostAwarenessPrinciple().evaluate(
            candidate_return=0.15,
            hurdle_return=0.10,
            alternative_return=0.12,
        )
        self.assertTrue(result["clears_opportunity_cost"])
        self.assertAlmostEqual(result["excess_return_vs_best_alternative"], 0.03)

    def test_capital_allocation_analysis(self):
        result = CapitalAllocationAnalysis().evaluate(
            recent_free_cash_flow=300.0,
            total_debt=800.0,
            cash_and_equivalents=500.0,
            commentary="",
        )
        self.assertEqual(result["balance_sheet_position"], "net_debt")
        self.assertAlmostEqual(result["fcf_to_debt"], 0.375)
        self.assertEqual(result["capital_allocation_discipline"], "strong")

    @patch("valuation_capital.ShareBuybackAnalysis.evaluate")
    def test_capital_allocation_analysis_fetches_buyback_commentary_for_ticker(self, mock_evaluate):
        mock_evaluate.return_value = {
            "buyback_strategy": "Opportunistic/Value-Based",
            "mentions_intrinsic_value": True,
            "analysis_summary": "Repurchases occur when shares are undervalued.",
        }

        result = CapitalAllocationAnalysis().evaluate(
            recent_free_cash_flow=300.0,
            total_debt=800.0,
            cash_and_equivalents=500.0,
            ticker="AAPL",
        )

        mock_evaluate.assert_called_once_with("AAPL")
        self.assertEqual(result["buyback_analysis"]["buyback_strategy"], "Opportunistic/Value-Based")

    @patch("valuation_capital.load_cached_transcript_keyword_context")
    @patch("valuation_capital.fetch_filing_section")
    def test_fetch_management_commentary_merges_sec_and_earnings_call_context(self, mock_fetch_section, mock_transcript_context):
        mock_fetch_section.return_value = "SEC MD&A discusses capital allocation and buybacks."
        mock_transcript_context.return_value = "On the call, management discussed acquisitions and repurchases."

        result = valuation_capital.fetch_management_commentary.__wrapped__("YUM")

        self.assertIn("SEC commentary", result)
        self.assertIn("Earnings call commentary", result)
        self.assertIn("acquisitions and repurchases", result)

    @patch("valuation_capital.ShareBuybackAnalysis.evaluate")
    def test_capital_allocation_analysis_prefers_provided_commentary(self, mock_evaluate):
        mock_evaluate.return_value = {
            "buyback_strategy": "Systematic",
            "mentions_intrinsic_value": False,
            "analysis_summary": "Management uses a standing authorization.",
        }

        result = CapitalAllocationAnalysis().evaluate(
            recent_free_cash_flow=300.0,
            total_debt=800.0,
            cash_and_equivalents=500.0,
            ticker="AAPL",
            commentary="Board approved a standing repurchase program.",
        )

        mock_evaluate.assert_called_once_with("AAPL", "Board approved a standing repurchase program.")
        self.assertEqual(result["buyback_analysis"]["buyback_strategy"], "Systematic")

    def test_lookthrough_earnings(self):
        result = LookthroughEarningsPrinciple().evaluate(
            ownership_percentage=0.25,
            investee_net_income=200.0,
            dividends_received=20.0,
        )
        self.assertAlmostEqual(result["lookthrough_earnings"], 50.0)
        self.assertAlmostEqual(result["retained_earnings_share"], 30.0)

    @patch("financial_metrics.fetch_lookthrough_commentary")
    def test_lookthrough_earnings_fetches_real_investee_inputs_for_ticker(self, mock_fetch_commentary):
        mock_fetch_commentary.return_value = (
            "The company holds a 25% ownership interest in the investee. "
            "Equity in earnings was $200 million. Dividends received totaled $20 million."
        )

        result = LookthroughEarningsPrinciple().evaluate(ticker="AAPL")

        self.assertAlmostEqual(result["ownership_percentage"], 0.25)
        self.assertAlmostEqual(result["investee_net_income"], 200_000_000.0)
        self.assertAlmostEqual(result["dividends_received"], 20_000_000.0)
        self.assertAlmostEqual(result["lookthrough_earnings"], 50_000_000.0)
        self.assertAlmostEqual(result["retained_earnings_share"], 30_000_000.0)

    @patch("financial_metrics.fetch_lookthrough_commentary")
    def test_lookthrough_earnings_returns_not_applicable_when_no_investee_evidence_found(self, mock_fetch_commentary):
        mock_fetch_commentary.return_value = ""

        result = LookthroughEarningsPrinciple().evaluate(ticker="AAPL")

        self.assertFalse(result["applicable"])
        self.assertIn("No material equity investee evidence", result["reason"])

    @patch("financial_metrics.fetch_lookthrough_commentary")
    def test_lookthrough_earnings_returns_not_applicable_when_metrics_cannot_be_extracted(self, mock_fetch_commentary):
        mock_fetch_commentary.return_value = "The company discusses equity method investments, but does not quantify ownership, earnings, or dividends received."

        result = LookthroughEarningsPrinciple().evaluate(ticker="AAPL")

        self.assertFalse(result["applicable"])
        self.assertIn("could not be extracted reliably", result["reason"])

    def test_business_model_type(self):
        result = BusinessModelTypes().evaluate(
            recurring_revenue_ratio=0.8,
            gross_margin=0.6,
            capital_intensity=0.04,
        )
        self.assertEqual(result["business_model_type"], "recurring_revenue")

    @patch("sec_data.fetch_filing_section")
    @patch("business_moat.fetch_deep_financials")
    @patch("business_moat.fetch_historical_margins")
    def test_business_model_type_classifies_from_margins_when_recurring_revenue_ratio_is_unavailable(self, mock_margins, mock_financials, mock_filing_section):
        mock_margins.return_value = pd.DataFrame([{"Year": 2024, "Gross_Margin": 0.7}])
        mock_financials.return_value = {"total_revenue": 1000.0, "capex_total": -50.0}
        # No business description available, so recurring_revenue_ratio can't be
        # inferred -- BusinessModelTypes should still classify from gross margin
        # and capital intensity alone rather than refusing to answer, since
        # those two are always-available financial-statement figures.
        mock_filing_section.return_value = ""

        result = BusinessModelTypes().evaluate(ticker="AAPL")
        self.assertIsNone(result["recurring_revenue_ratio"])
        self.assertEqual(result["business_model_type"], "asset_light")

    def test_durability_of_competitive_advantage(self):
        result = TheDurabilityOfCompetitiveAdvantage().evaluate(
            gross_margin_trend=0.02,
            market_share_trend=0.01,
            return_on_capital=0.15,
        )
        self.assertEqual(result["durability_score"], 3)
        self.assertEqual(result["durability_assessment"], "strong")

    @patch("business_moat.fetch_durability_metrics")
    def test_durability_of_competitive_advantage_fetches_real_company_data(self, mock_fetch_metrics):
        mock_fetch_metrics.return_value = {
            "gross_margin_trend": 0.02,
            "market_share_trend": 0.01,
            "return_on_capital": 0.15,
        }
        result = TheDurabilityOfCompetitiveAdvantage().evaluate(ticker="AAPL")
        self.assertEqual(result["durability_assessment"], "strong")
        mock_fetch_metrics.assert_called_once_with("AAPL")

    @patch("business_moat.fetch_historical_margins")
    def test_durability_of_competitive_advantage_handles_missing_margins(self, mock_margins):
        # Real, observed case: banks/insurers (AFL, AIG, ALL, AXP, BAC, ...)
        # don't report "Total Revenue"/"Gross Profit" the way yfinance
        # expects, so fetch_historical_margins legitimately returns an empty
        # DataFrame. This must produce the same "not applicable" shape every
        # other evaluator uses for missing data, not raise ValueError.
        mock_margins.return_value = pd.DataFrame()
        result = TheDurabilityOfCompetitiveAdvantage().evaluate(ticker="AFL")
        self.assertEqual(result.get("applicable"), False)

    @patch("business_moat.call_ollama_panel_json")
    @patch("business_moat.fetch_moat_commentary")
    @patch("sec_data.fetch_filing_section")
    def test_economic_moat_includes_earnings_call_commentary_when_available(self, mock_fetch_section, mock_fetch_moat_commentary, mock_panel):
        mock_fetch_section.return_value = "A software platform with sticky enterprise workflows."
        mock_fetch_moat_commentary.return_value = "Management cited pricing power, customer retention, and competitive wins."
        mock_panel.return_value = {
            "moat_type": "Switching Costs",
            "justification": "Sticky workflows create friction. Retention supports durability.",
        }

        EconomicMoat().evaluate("AAPL")

        prompt = mock_panel.call_args.args[0]
        self.assertIn("Relevant Earnings Call Commentary", prompt)
        self.assertIn("pricing power", prompt)

    @patch("business_moat.call_ollama_panel_json")
    @patch("business_moat.fetch_moat_commentary")
    @patch("sec_data.fetch_filing_section")
    def test_economic_moat_rejects_ungrounded_llm_output(self, mock_fetch_section, mock_fetch_moat_commentary, mock_panel):
        mock_fetch_section.return_value = "Aflac sells supplemental health and life insurance in Japan and the United States."
        mock_fetch_moat_commentary.return_value = "Management discussed underwriting margins and distribution partnerships."
        mock_panel.return_value = {
            "moat_type": "Network Effect",
            "justification": "The business grows as more lawyers and clients join the referral marketplace, reinforcing the platform.",
        }

        result = EconomicMoat().evaluate("AFL")

        self.assertEqual(result.get("applicable"), False)
        self.assertIn("grounded", result.get("reason", ""))

    def test_focus_investing(self):
        result = FocusInvestingPrinciple().evaluate([40.0, 25.0, 15.0, 10.0, 10.0])
        self.assertEqual(result["position_count"], 5)
        self.assertTrue(result["is_focus_investing"])
        self.assertAlmostEqual(result["top_three_weight"], 0.8)

    def test_efficient_market_theory(self):
        result = EfficientMarketTheoryPrinciple().evaluate(
            stock_cagr=0.09,
            benchmark_cagr=0.08,
            tracking_error=0.03,
        )
        self.assertTrue(result["market_efficiency_supported"])

    def test_market_forecasting(self):
        result = MarketForecastingPrinciple().evaluate(
            forecast_return=0.10,
            actual_return=0.12,
        )
        self.assertAlmostEqual(result["forecast_error"], 0.02)
        self.assertTrue(result["forecast_was_useful"])

    def test_undervalued_margin_of_safety(self):
        result = UndervaluedMarginOfSafety().evaluate(
            intrinsic_value=100.0,
            market_price=70.0,
            minimum_margin=0.25,
        )
        self.assertTrue(result["is_undervalued"])
        self.assertAlmostEqual(result["margin_of_safety"], 0.30)

    @patch("investment_philosophy.IntrinsicValue.evaluate")
    def test_undervalued_margin_of_safety_fetches_real_intrinsic_value_for_ticker(self, mock_intrinsic):
        mock_intrinsic.return_value = {"intrinsic_value_per_share": 100.0, "market_price": 70.0}
        result = UndervaluedMarginOfSafety().evaluate(ticker="AAPL")
        self.assertTrue(result["is_undervalued"])
        mock_intrinsic.assert_called_once_with(ticker="AAPL")

    @patch("investment_philosophy.IntrinsicValue.evaluate")
    def test_undervalued_margin_of_safety_handles_negative_intrinsic_value(self, mock_intrinsic):
        mock_intrinsic.return_value = {"intrinsic_value_per_share": -413.12, "market_price": 160.44}
        result = UndervaluedMarginOfSafety().evaluate(ticker="APD")
        self.assertEqual(result, {"applicable": False, "reason": "Computed intrinsic value is not positive (likely driven by negative free cash flow); margin of safety is not meaningful here"})

    @patch("investment_philosophy.IntrinsicValue.evaluate")
    def test_undervalued_margin_of_safety_handles_inapplicable_intrinsic_value(self, mock_intrinsic):
        # Banks/insurers: IntrinsicValue() itself returns {"applicable":
        # False, ...} when FCF is None (JPM, BAC, C, BK, USB). Direct-
        # indexing intrinsic_result["intrinsic_value_per_share"] would raise
        # KeyError here; must fall through to the same clean shape instead.
        mock_intrinsic.return_value = {"applicable": False, "reason": "Missing required metrics: fcf, growth_rate, discount_rate, shares_outstanding, and net_debt are required"}
        result = UndervaluedMarginOfSafety().evaluate(ticker="JPM")
        self.assertEqual(result, {"applicable": False, "reason": "Missing required metrics: intrinsic_value and market_price are required"})

    def test_intrinsic_value_wrapper(self):
        result = IntrinsicValue().evaluate(
            fcf=100.0,
            growth_rate=0.05,
            discount_rate=0.10,
            terminal_growth_rate=0.02,
            shares_outstanding=10,
            net_debt=50.0,
            years=5,
        )
        self.assertIn("intrinsic_value_per_share", result)
        self.assertGreater(result["intrinsic_value_per_share"], 0)

    @patch("investment_philosophy.IntrinsicValueEstimation.evaluate")
    def test_intrinsic_value_fetches_real_inputs_for_ticker(self, mock_estimate):
        mock_estimate.return_value = {"intrinsic_value_per_share": 123.0, "market_price": 100.0}
        result = IntrinsicValue().evaluate(ticker="AAPL")
        self.assertEqual(result["intrinsic_value_per_share"], 123.0)
        mock_estimate.assert_called_once_with(ticker="AAPL", terminal_growth_rate=0.02, years=DEFAULT_INTRINSIC_VALUE_YEARS)

    @patch("valuation_capital.fetch_current_market_price")
    @patch("valuation_capital.fetch_risk_free_rate")
    @patch("valuation_capital.fetch_financial_data")
    def test_intrinsic_value_estimation_handles_none_free_cash_flow(self, mock_financials, mock_risk_free, mock_market_price):
        # Banks/insurers: yfinance's freeCashflow is routinely None for these
        # (JPM, BAC, C, BK, USB all observed None). This must return the same
        # clean "not applicable" shape every other evaluator uses for missing
        # data, not raise float(None) -> TypeError.
        mock_financials.return_value = {
            "recent_free_cash_flow": None,
            "historical_fcf_growth_rate": 0.05,
            "shares_outstanding": 10,
            "cash_and_equivalents": 20.0,
            "total_debt": 50.0,
        }
        mock_risk_free.return_value = 0.04
        mock_market_price.return_value = 80.0

        result = valuation_capital.IntrinsicValueEstimation().evaluate(ticker="JPM")

        self.assertEqual(result, {"applicable": False, "reason": "Missing required metrics: fcf, growth_rate, discount_rate, shares_outstanding, and net_debt are required"})

    def test_intrinsic_value_estimation_rejects_non_positive_free_cash_flow(self):
        result = valuation_capital.IntrinsicValueEstimation().evaluate(
            fcf=-100.0,
            growth_rate=0.05,
            discount_rate=0.10,
            terminal_growth_rate=0.02,
            shares_outstanding=10,
            net_debt=50.0,
            years=5,
        )

        self.assertEqual(
            result,
            {"applicable": False, "reason": "Intrinsic value is not meaningful with non-positive free cash flow"},
        )

    @patch("valuation_capital.fetch_current_market_price")
    @patch("valuation_capital.fetch_risk_free_rate")
    @patch("valuation_capital.fetch_financial_data")
    def test_margin_of_safety_handles_inapplicable_intrinsic_value(self, mock_financials, mock_risk_free, mock_market_price):
        mock_financials.return_value = {
            "recent_free_cash_flow": None,
            "historical_fcf_growth_rate": 0.05,
            "shares_outstanding": 10,
            "cash_and_equivalents": 20.0,
            "total_debt": 50.0,
        }
        mock_risk_free.return_value = 0.04
        mock_market_price.return_value = 80.0

        result = MarginOfSafety().evaluate(ticker="JPM")

        self.assertEqual(result, {"applicable": False, "reason": "Missing required metrics: intrinsic_value and market_price are required"})

    @patch("valuation_capital.fetch_financial_data")
    def test_capital_allocation_analysis_handles_none_free_cash_flow(self, mock_financials):
        mock_financials.return_value = {
            "recent_free_cash_flow": None,
            "total_debt": 800.0,
            "cash_and_equivalents": 500.0,
        }

        result = CapitalAllocationAnalysis().evaluate(ticker="JPM")

        self.assertEqual(result, {"applicable": False, "reason": "Missing required metrics: recent_free_cash_flow, total_debt, and cash_and_equivalents are required"})

    @patch("valuation_capital.fetch_current_market_price")
    @patch("valuation_capital.fetch_risk_free_rate")
    @patch("valuation_capital.fetch_financial_data")
    def test_intrinsic_value_estimation_fetches_real_company_data(self, mock_financials, mock_risk_free, mock_market_price):
        mock_financials.return_value = {
            "recent_free_cash_flow": 100.0,
            "historical_fcf_growth_rate": 0.05,
            "shares_outstanding": 10,
            "cash_and_equivalents": 20.0,
            "total_debt": 50.0,
        }
        mock_risk_free.return_value = 0.04
        mock_market_price.return_value = 80.0

        result = valuation_capital.IntrinsicValueEstimation().evaluate(ticker="AAPL", years=5)

        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["market_price"], 80.0)
        self.assertIn("intrinsic_value_per_share", result)

    @patch("valuation_capital.IntrinsicValueEstimation.evaluate")
    def test_margin_of_safety_fetches_real_values_for_ticker(self, mock_estimate):
        mock_estimate.return_value = {"intrinsic_value_per_share": 100.0, "market_price": 70.0}
        result = MarginOfSafety().evaluate(ticker="AAPL")
        self.assertAlmostEqual(result["margin_of_safety"], 0.30)
        mock_estimate.assert_called_once_with(ticker="AAPL")

    @patch("valuation_capital.MarginOfSafety.evaluate")
    def test_purchase_price_relationship_fetches_real_values_for_ticker(self, mock_mos):
        mock_mos.side_effect = [
            {"intrinsic_value": 100.0, "market_price": 72.0},
            {"margin_of_safety": 0.28},
        ]
        result = TheRelationshipBetweenPurchasePriceAndIntrinsicValue().evaluate(ticker="AAPL")
        self.assertEqual(result["purchase_price_verdict"], "deep_discount")

    @patch("valuation_capital.fetch_financial_data")
    @patch("valuation_capital.ShareBuybackAnalysis.evaluate")
    def test_capital_allocation_analysis_fetches_real_financials_for_ticker(self, mock_evaluate, mock_financials):
        mock_financials.return_value = {
            "recent_free_cash_flow": 300.0,
            "total_debt": 800.0,
            "cash_and_equivalents": 500.0,
        }
        mock_evaluate.return_value = {
            "buyback_strategy": "Opportunistic/Value-Based",
            "mentions_intrinsic_value": True,
            "analysis_summary": "Repurchases occur when shares are undervalued.",
        }

        result = CapitalAllocationAnalysis().evaluate(ticker="AAPL")

        self.assertEqual(result["balance_sheet_position"], "net_debt")
        self.assertAlmostEqual(result["fcf_to_debt"], 0.375)
        mock_financials.assert_called_once_with("AAPL")

    def test_calculate_dcf_uses_shared_default_horizon(self):
        self.assertEqual(
            valuation_capital.calculate_dcf.__defaults__[-1],
            DEFAULT_INTRINSIC_VALUE_YEARS,
        )

    def test_goodwill_quality(self):
        result = GoodwillEconomicGoodwillVsAccountingGoodwill().evaluate(
            goodwill=400.0,
            acquired_earnings=100.0,
            return_on_tangible_assets=0.15,
        )
        self.assertEqual(result["goodwill_quality"], "economic_goodwill")

    @patch("business_moat.fetch_goodwill_metrics")
    def test_goodwill_quality_fetches_real_company_data(self, mock_fetch_metrics):
        mock_fetch_metrics.return_value = {
            "goodwill": 400.0,
            "acquired_earnings": 100.0,
            "return_on_tangible_assets": 0.15,
        }
        result = GoodwillEconomicGoodwillVsAccountingGoodwill().evaluate(ticker="AAPL")
        self.assertEqual(result["goodwill_quality"], "economic_goodwill")
        mock_fetch_metrics.assert_called_once_with("AAPL")

    def test_shareholder_orientation(self):
        result = CorporateGovernanceAndShareholderOrientation().evaluate(
            insider_ownership=0.08,
            roic_linked_pay=True,
            dual_class_structure=False,
            buybacks_below_intrinsic_value=True,
        )
        self.assertEqual(result["governance_score"], 4)
        self.assertEqual(result["shareholder_orientation"], "strong")

    @patch("valuation_capital.fetch_management_commentary")
    @patch("management_governance.fetch_latest_filing_text")
    @patch("management_governance.yf.Ticker")
    def test_shareholder_orientation_fetches_real_governance_inputs(self, mock_ticker, mock_proxy_text, mock_commentary):
        mock_ticker.return_value.info = {"heldPercentInsiders": 0.08}
        mock_proxy_text.return_value = (
            "Executive compensation incentives are tied to return on invested capital. "
            "The company has Class A common stock and Class B common stock."
        )
        mock_commentary.return_value = "The board approved share repurchases when shares trade below intrinsic value."

        result = CorporateGovernanceAndShareholderOrientation().evaluate(ticker="AAPL")

        self.assertEqual(result["insider_ownership"], 0.08)
        self.assertTrue(result["roic_linked_pay"])
        self.assertTrue(result["dual_class_structure"])
        self.assertTrue(result["buybacks_below_intrinsic_value"])
        self.assertEqual(result["governance_score"], 3)
        self.assertEqual(result["shareholder_orientation"], "strong")

    @patch("valuation_capital.fetch_management_commentary")
    @patch("management_governance.fetch_latest_filing_text")
    @patch("management_governance.yf.Ticker")
    def test_shareholder_orientation_handles_missing_proxy_filing(self, mock_ticker, mock_proxy_text, mock_commentary):
        # Real, observed case: some tickers (e.g. BX -- Blackstone) have no
        # DEF 14A on file the way this fetcher expects. This must produce the
        # same "not applicable" shape every other evaluator uses for missing
        # data, not raise ValueError.
        mock_ticker.return_value.info = {"heldPercentInsiders": 0.05}
        mock_proxy_text.side_effect = ValueError("No DEF 14A filing found for ticker BX")
        mock_commentary.return_value = ""

        result = CorporateGovernanceAndShareholderOrientation().evaluate(ticker="BX")

        self.assertEqual(result.get("applicable"), False)

    def test_economic_reality(self):
        result = CorePrincipleSeeThroughAccountingToEconomicReality().evaluate(
            net_income=100.0,
            operating_cash_flow=130.0,
            capex_total=-20.0,
        )
        self.assertAlmostEqual(result["free_cash_flow"], 110.0)
        self.assertEqual(result["economic_reality_assessment"], "cash_backed")

    @patch("financial_metrics.fetch_deep_financials")
    def test_economic_reality_fetches_real_financials(self, mock_fetch_financials):
        mock_fetch_financials.return_value = {
            "net_income": 100.0,
            "operating_cash_flow": 130.0,
            "capex_total": -20.0,
        }

        result = CorePrincipleSeeThroughAccountingToEconomicReality().evaluate(ticker="AAPL")
        self.assertAlmostEqual(result["free_cash_flow"], 110.0)
        mock_fetch_financials.assert_called_once_with("AAPL")

    @patch.object(OwnerEarnings, "_query_ollama_capex_breakdown")
    @patch.object(OwnerEarnings, "_fetch_mda_section")
    @patch.object(OwnerEarnings, "_fetch_deep_financials")
    def test_owner_earnings_uses_reported_depreciation_when_available(self, mock_financials, mock_mda, mock_capex_breakdown):
        # Mirrors a real AAPL-shaped period where operating cash flow dips
        # slightly below net income (working-capital timing), which used to
        # make the OCF-NI proxy go negative even though real reported D&A is
        # a large positive number.
        mock_financials.return_value = {
            "net_income": 112.0,
            "operating_cash_flow": 111.5,
            "capex_total": -12.7,
            "depreciation_and_amortization": 11.7,
        }
        mock_mda.return_value = ""
        mock_capex_breakdown.return_value = {"maintenance_percentage": 0.35}

        result = OwnerEarnings().evaluate("AAPL")

        self.assertEqual(result["depreciation_amortization_estimate"], 11.7)
        self.assertEqual(result["depreciation_amortization_source"], "reported")
        self.assertAlmostEqual(result["owner_earnings"], 112.0 + 11.7 - (12.7 * 0.35))

    @patch.object(OwnerEarnings, "_query_ollama_capex_breakdown")
    @patch.object(OwnerEarnings, "_fetch_mda_section")
    @patch.object(OwnerEarnings, "_fetch_deep_financials")
    def test_owner_earnings_floors_ocf_minus_ni_proxy_at_zero(self, mock_financials, mock_mda, mock_capex_breakdown):
        # No reported D&A available, and OCF fell below NI for the period --
        # the fallback proxy must not go negative (D&A cannot be negative).
        mock_financials.return_value = {
            "net_income": 112.0,
            "operating_cash_flow": 111.5,
            "capex_total": -12.7,
            "depreciation_and_amortization": None,
        }
        mock_mda.return_value = ""
        mock_capex_breakdown.return_value = {"maintenance_percentage": 0.35}

        result = OwnerEarnings().evaluate("AAPL")

        self.assertEqual(result["depreciation_amortization_estimate"], 0)
        self.assertEqual(result["depreciation_amortization_source"], "ocf_minus_ni_proxy")

    def test_dividends_tax_efficiency(self):
        result = DividendsRetainedEarningsAndTaxEfficiency().evaluate(
            dividend_payout_ratio=0.30,
            retained_return_on_equity=0.18,
            tax_rate_on_dividends=0.20,
        )
        self.assertEqual(result["capital_return_preference"], "distribute")

    def test_derivatives_risk(self):
        result = DerivativesRisk().evaluate(
            notional_exposure=250.0,
            equity_capital=100.0,
            level_3_assets_ratio=0.06,
        )
        self.assertEqual(result["derivatives_risk"], "moderate")

    @patch("risk_behavior.yf.Ticker")
    @patch("risk_behavior.LeverageRisk.evaluate")
    def test_derivatives_risk_fetches_real_footnote_signal_for_ticker(self, mock_leverage_risk, mock_ticker):
        mock_leverage_risk.return_value = {
            "toxic_derivative_exposure": "No material exposure"
        }
        mock_ticker.return_value.balance_sheet = pd.DataFrame(
            {pd.Timestamp("2024-12-31"): [500.0]},
            index=["Stockholders Equity"],
        )

        result = DerivativesRisk().evaluate(ticker="AAPL")
        self.assertEqual(result["derivatives_risk"], "low")
        self.assertEqual(result["derivative_exposure_summary"], "No material exposure")

    @patch("risk_behavior.yf.Ticker")
    @patch("risk_behavior.LeverageRisk.evaluate")
    def test_derivatives_risk_returns_not_applicable_when_footnotes_do_not_yield_derivative_summary(self, mock_leverage_risk, mock_ticker):
        mock_leverage_risk.return_value = {}
        mock_ticker.return_value.balance_sheet = pd.DataFrame(
            {pd.Timestamp("2024-12-31"): [500.0]},
            index=["Stockholders Equity"],
        )

        result = DerivativesRisk().evaluate(ticker="AAPL")

        self.assertFalse(result["applicable"])
        self.assertIn("No material derivative exposure found", result["reason"])

    @patch("business_moat.fetch_cpi_inflation_data")
    @patch("business_moat.fetch_historical_margins")
    def test_impact_of_inflation_fetches_real_inputs_for_ticker(self, mock_margins, mock_inflation):
        mock_margins.return_value = pd.DataFrame(
            [
                {"Year": 2023, "Gross_Margin": 0.40, "Operating_Margin": 0.20},
                {"Year": 2024, "Gross_Margin": 0.41, "Operating_Margin": 0.21},
            ]
        )
        mock_inflation.return_value = pd.DataFrame(
            [
                {"Year": 2023, "Inflation_Rate": 3.5},
                {"Year": 2024, "Inflation_Rate": 2.8},
            ]
        )

        from risk_behavior import TheImpactOfInflation
        result = TheImpactOfInflation().evaluate(ticker="AAPL")
        self.assertFalse(result.empty)
        self.assertIn("Pricing_Power_Assessment", result.columns)

    def test_independent_thinking(self):
        result = IndependentThinkingPrinciple().evaluate(
            thesis_differs_from_consensus=True,
            evidence_strength=0.8,
            valuation_gap=0.20,
        )
        self.assertEqual(result["independent_thinking_score"], 3)
        self.assertEqual(result["independent_thinking"], "strong")

    def test_corporate_culture(self):
        result = CorporateCulture().evaluate(
            employee_turnover=0.05,
            insider_ownership=0.12,
            restructurings_per_5y=0,
        )
        self.assertAlmostEqual(result["culture_score"], 2.83, places=2)
        self.assertEqual(result["culture_assessment"], "strong")

    @patch("management_governance._semantic_polarity_score")
    @patch("management_governance.yf.Ticker")
    @patch("management_governance.fetch_filing_keyword_context")
    @patch("management_governance.fetch_filing_section")
    def test_corporate_culture_fetches_real_inputs_for_ticker(self, mock_fetch_section, mock_fetch_context, mock_ticker, mock_semantic_score):
        mock_ticker.return_value.info = {"heldPercentInsiders": 0.06}
        mock_fetch_section.return_value = "10% employee turnover was reported. Retention remained strong."
        mock_fetch_context.return_value = "The company announced a restructuring and a reorganization program."
        # Regex directly extracts the turnover rate above, so this mocked semantic
        # score is only consumed for the restructuring-severity read: 0.2 means
        # "fairly close to the negative (significant restructuring) anchor".
        mock_semantic_score.return_value = 0.2

        result = CorporateCulture().evaluate(ticker="AAPL")

        self.assertEqual(result["insider_ownership"], 0.06)
        self.assertEqual(result["employee_turnover"], 0.10)
        self.assertAlmostEqual(result["restructurings_per_5y"], 2.4, places=4)
        self.assertEqual(result["signals_available"], 3)
        self.assertEqual(result["culture_assessment"], "stable")

    @patch("management_governance._semantic_polarity_score")
    @patch("management_governance.yf.Ticker")
    @patch("management_governance.fetch_filing_keyword_context")
    @patch("management_governance.fetch_filing_section")
    def test_corporate_culture_returns_not_applicable_without_enough_signals(self, mock_fetch_section, mock_fetch_context, mock_ticker, mock_semantic_score):
        mock_ticker.return_value.info = {}
        mock_fetch_section.return_value = "The company discussed employees and retention qualitatively."
        mock_fetch_context.return_value = ""
        # Simulates the embedding pipeline being unavailable/inconclusive, so
        # there's genuinely zero usable signal (not even a semantic estimate).
        mock_semantic_score.return_value = None

        result = CorporateCulture().evaluate(ticker="AAPL")

        self.assertFalse(result["applicable"])
        self.assertIn("at least one culture proxy", result["reason"])

    def test_acquisition_logic(self):
        result = AcquisitionLogicAcquisitionCriteria().evaluate(
            purchase_multiple=10.0,
            return_on_invested_capital=0.14,
            debt_funded=False,
        )
        self.assertEqual(result["acquisition_score"], 3)
        self.assertEqual(result["acquisition_discipline"], "disciplined")

    @patch("management_governance.yf.Ticker")
    @patch("management_governance.fetch_filing_keyword_context")
    def test_acquisition_logic_fetches_real_inputs_for_ticker(self, mock_fetch_context, mock_ticker):
        mock_fetch_context.return_value = "The company acquired a target for 10.0x EBITDA and financed the deal with cash on hand."

        income_stmt = pd.DataFrame({
            pd.Timestamp("2024-12-31"): {"Operating Income": 140.0}
        })
        balance_sheet = pd.DataFrame({
            pd.Timestamp("2024-12-31"): {
                "Total Debt": 200.0,
                "Stockholders Equity": 800.0,
                "Cash And Cash Equivalents": 100.0,
            }
        })
        mock_ticker.return_value.income_stmt = income_stmt
        mock_ticker.return_value.balance_sheet = balance_sheet

        result = AcquisitionLogicAcquisitionCriteria().evaluate(ticker="AAPL")

        self.assertEqual(result["purchase_multiple"], 10.0)
        self.assertAlmostEqual(result["return_on_invested_capital"], 140.0 / 900.0)
        self.assertFalse(result["debt_funded"])
        self.assertEqual(result["acquisition_discipline"], "disciplined")

    @patch("management_governance.yf.Ticker")
    @patch("management_governance.fetch_filing_keyword_context")
    def test_acquisition_logic_returns_not_applicable_without_purchase_multiple(self, mock_fetch_context, mock_ticker):
        mock_fetch_context.return_value = "Management discussed an acquisition and financing structure without disclosing a purchase multiple."

        income_stmt = pd.DataFrame({
            pd.Timestamp("2024-12-31"): {"Operating Income": 140.0}
        })
        balance_sheet = pd.DataFrame({
            pd.Timestamp("2024-12-31"): {
                "Total Debt": 200.0,
                "Stockholders Equity": 800.0,
                "Cash And Cash Equivalents": 100.0,
            }
        })
        mock_ticker.return_value.income_stmt = income_stmt
        mock_ticker.return_value.balance_sheet = balance_sheet

        result = AcquisitionLogicAcquisitionCriteria().evaluate(ticker="AAPL")

        self.assertFalse(result["applicable"])
        self.assertIn("No recent acquisition evidence found", result["reason"])

    def test_special_investment_instruments(self):
        result = SpecialInvestmentInstruments().evaluate(
            coupon_rate=0.09,
            conversion_discount=0.12,
            collateral_coverage=1.2,
        )
        self.assertEqual(result["instrument_score"], 3)
        self.assertEqual(result["instrument_attractiveness"], "high")

    @patch("valuation_capital.fetch_special_instrument_commentary")
    def test_special_investment_instruments_fetches_real_filing_context_for_ticker(self, mock_fetch_commentary):
        mock_fetch_commentary.return_value = (
            "The company issued convertible preferred stock with a dividend rate of 9%. "
            "Investors received a conversion discount of 12%. The instrument is secured by collateral."
        )

        result = SpecialInvestmentInstruments().evaluate(ticker="AAPL")

        self.assertEqual(result["instrument_score"], 3)
        self.assertEqual(result["instrument_attractiveness"], "high")
        mock_fetch_commentary.assert_called_once_with("AAPL")

    @patch("valuation_capital.fetch_special_instrument_commentary")
    def test_special_investment_instruments_returns_not_applicable_when_no_evidence_found(self, mock_fetch_commentary):
        mock_fetch_commentary.return_value = ""

        result = SpecialInvestmentInstruments().evaluate(ticker="AAPL")

        self.assertFalse(result["applicable"])
        self.assertIn("No special investment instrument evidence", result["reason"])

    def test_behavioral_biases(self):
        result = CommonBehavioralBiasesPrinciple().evaluate(
            thesis_changes_after_price_move=True,
            avg_holding_period_years=0.5,
            adds_to_losers_without_new_evidence=False,
        )
        self.assertEqual(result["bias_count"], 2)
        self.assertEqual(result["behavioral_risk"], "high")

    def test_patience_as_edge(self):
        result = PatienceAsEdgePrinciple().evaluate(
            avg_holding_period_years=5,
            turnover_ratio=0.20,
            forced_activity=False,
        )
        self.assertEqual(result["patience_score"], 3)
        self.assertEqual(result["patience_as_edge"], "strong")

    @patch("thinking_frameworks.fetch_filing_section")
    @patch("thinking_frameworks.yf.Ticker")
    def test_fetch_company_info(self, mock_ticker, mock_fetch_section):
        mock_fetch_section.side_effect = Exception("Network error")
        mock_ticker.return_value.info = {
            "longName": "Yum! Brands, Inc.",
            "longBusinessSummary": "Operates restaurant brands.",
        }
        result = thinking_frameworks.fetch_company_info("YUM")
        self.assertEqual(result["name"], "Yum! Brands, Inc.")
        self.assertEqual(result["description"], "Operates restaurant brands.")

    @patch("management_governance.ManagementEvaluation.evaluate")
    def test_analyze_management_governance(self, mock_evaluate):
        mock_evaluate.return_value = "analysis"
        self.assertEqual(management_governance.analyze_management_governance("YUM"), "analysis")

    @patch("management_governance.ManagementEvaluation.evaluate")
    def test_analyze_management_governance_passes_explicit_inputs(self, mock_evaluate):
        mock_evaluate.return_value = "analysis"
        management_governance.analyze_management_governance(
            "YUM",
            transcript="Transcript text",
            proxy_statement="Proxy text",
        )
        _, kwargs = mock_evaluate.call_args
        self.assertEqual(kwargs["transcript"], "Transcript text")
        self.assertEqual(kwargs["proxy_stmt"], "Proxy text")

    @patch("management_governance.load_cached_transcript_text")
    def test_management_evaluation_requires_cached_transcript(self, mock_load_cached):
        mock_load_cached.return_value = ""
        with self.assertRaises(RuntimeError):
            ManagementEvaluation()._fetch_earnings_call_transcript("YUM")

    @patch("management_governance.ManagementEvaluation._fetch_earnings_call_transcript")
    def test_management_evaluation_handles_missing_transcript_gracefully(self, mock_fetch_transcript):
        # Real, observed case: some tickers (GOOG, BRK-B, BF-B) have no
        # cached or auto-fetchable earnings call transcripts. evaluate()
        # must catch that and return the same "not applicable" shape every
        # other evaluator uses for missing data, not let the exception
        # propagate up to the generic per-evaluator catch in run.py.
        mock_fetch_transcript.side_effect = RuntimeError(
            "No cached earnings call transcripts found for GOOG."
        )
        result = ManagementEvaluation().evaluate("GOOG")
        self.assertEqual(result.get("applicable"), False)
        self.assertIn("No cached earnings call transcripts", result.get("reason", ""))

    @patch("management_governance.ManagementEvaluation._fetch_sec_def_14a")
    @patch("management_governance.ManagementEvaluation._fetch_earnings_call_transcript")
    @patch("management_governance.ManagementEvaluation._call_ollama")
    def test_management_evaluation_rejects_ungrounded_llm_output(self, mock_call_ollama, mock_fetch_transcript, mock_fetch_proxy):
        mock_fetch_transcript.return_value = "Management discussed underwriting, pricing, and claims reserves."
        mock_fetch_proxy.return_value = "Executive compensation is tied to ROE and underwriting profitability."
        mock_call_ollama.return_value = {
            "response": "Management benefits from marketplace flywheel dynamics as more lawyers and clients join the platform, improving lead generation efficiency."
        }

        result = ManagementEvaluation().evaluate("AFL")

        self.assertEqual(result.get("applicable"), False)
        self.assertIn("grounded", result.get("reason", ""))

    @patch("management_governance.load_cached_transcript_text")
    def test_management_evaluation_reads_cached_transcript(self, mock_load_cached):
        mock_load_cached.return_value = "CEO: We repurchased shares opportunistically."
        result = ManagementEvaluation()._fetch_earnings_call_transcript("YUM")
        self.assertIn("repurchased shares", result)

    @patch("thinking_frameworks.fetch_filing_section")
    @patch("thinking_frameworks.yf.Ticker")
    def test_fetch_company_info_prefers_sec_business_description(self, mock_ticker, mock_fetch_section):
        mock_ticker.return_value.info = {
            "longName": "Yum! Brands, Inc.",
            "longBusinessSummary": "Fallback summary.",
        }
        mock_fetch_section.return_value = "SEC business description"
        result = thinking_frameworks.fetch_company_info("YUM")
        self.assertEqual(result["description"], "SEC business description")
        self.assertEqual(result["description_source"], "sec_10k_item_1")

    @patch("valuation_capital.fetch_filing_keyword_context")
    def test_fetch_management_commentary_prefers_sec_keyword_context(self, mock_fetch_context):
        mock_fetch_context.return_value = "The company repurchased shares opportunistically."
        result = valuation_capital.fetch_management_commentary.__wrapped__("YUM")
        self.assertIn("repurchased shares", result)

    @patch("valuation_capital.fetch_management_commentary")
    @patch("valuation_capital.call_ollama_panel_json")
    def test_share_buyback_analysis_uses_combined_management_commentary(self, mock_panel, mock_fetch_commentary):
        mock_fetch_commentary.return_value = "SEC commentary: buybacks. Earnings call commentary: intrinsic value repurchases."
        mock_panel.return_value = {
            "buyback_strategy": "Opportunistic/Value-Based",
            "mentions_intrinsic_value": True,
            "analysis_summary": "Management ties repurchases to value.",
        }

        result = ShareBuybackAnalysis().evaluate("YUM")

        self.assertTrue(result["mentions_intrinsic_value"])
        prompt = mock_panel.call_args.args[0]
        self.assertIn("Earnings call commentary", prompt)

    def test_extract_keyword_context_returns_real_snippets(self):
        text = "Alpha. The company expanded its share repurchase program materially this quarter. Omega."
        result = extract_keyword_context(text, keywords=("share repurchase",), context_chars=60)
        self.assertIn("share repurchase program", result)

    @patch("management_governance.fetch_filing_section")
    def test_management_evaluation_fetches_real_proxy_section(self, mock_fetch_section):
        mock_fetch_section.return_value = "Executive compensation ties bonuses to ROIC."
        result = ManagementEvaluation()._fetch_sec_def_14a("YUM")
        self.assertEqual(result, "Executive compensation ties bonuses to ROIC.")

    @patch("risk_behavior.fetch_sec_10k_footnotes")
    def test_leverage_risk_uses_sec_fetcher(self, mock_fetch):
        mock_fetch.return_value = "Leases and pension disclosures"
        result = LeverageRisk()._fetch_sec_10k_footnotes("YUM")
        self.assertEqual(result, "Leases and pension disclosures")

    @patch("risk_behavior.fetch_earnings_call_risk_commentary")
    @patch("risk_behavior._extract_footnote_risk_signals")
    @patch("risk_behavior.fetch_sec_10k_footnotes")
    def test_leverage_risk_attaches_earnings_call_context_to_deterministic_result(self, mock_fetch_footnotes, mock_extract_signals, mock_risk_commentary):
        mock_fetch_footnotes.return_value = "Footnotes"
        mock_risk_commentary.return_value = "Management discussed refinancing risk."
        mock_extract_signals.return_value = {
            "operating_lease_obligations": "$500 million",
            "pension_underfunding": "Not found",
            "toxic_derivative_exposure": "None mentioned",
        }

        result = LeverageRisk().evaluate("YUM")

        self.assertEqual(result["earnings_call_risk_commentary"], "Management discussed refinancing risk.")

    @patch("financial_metrics.fetch_mda_section")
    def test_owner_earnings_uses_real_mda_fetcher(self, mock_fetch_mda):
        mock_fetch_mda.return_value = "Item 7. Management discussion"
        result = OwnerEarnings()._fetch_mda_section("YUM")
        self.assertEqual(result, "Item 7. Management discussion")

    def test_normalize_capex_breakdown(self):
        normalized = normalize_capex_breakdown({
            "maintenance_capex_percentage": 70,
            "growth_capex_percentage": 30,
            "analysis": "Derived from MD&A",
        })
        self.assertEqual(normalized["maintenance_percentage"], 70)
        self.assertEqual(normalized["growth_percentage"], 30)
        self.assertEqual(normalized["reasoning"], "Derived from MD&A")

    def test_normalize_buyback_analysis(self):
        normalized = normalize_buyback_analysis({
            "systematic_buybacks": True,
            "intrinsic_value_mentioned": True,
            "analysis": "Management discusses intrinsic value explicitly.",
        })
        self.assertEqual(normalized["buyback_strategy"], "Systematic")
        self.assertTrue(normalized["mentions_intrinsic_value"])
        self.assertEqual(normalized["analysis_summary"], "Management discusses intrinsic value explicitly.")

    def test_compounding_principle(self):
        result = CompoundingPrinciple().evaluate(annual_return=0.10, years=10, initial_capital=1.0)
        self.assertGreater(result["ending_value"], 2.5)

    @patch("thinking_frameworks.yf.Ticker")
    def test_mr_market_price_regime(self, mock_ticker):
        close = [100.0] * 210 + [80.0, 79.0, 78.0, 77.0, 76.0]
        history = pd.DataFrame({"Close": close})
        mock_ticker.return_value.history.return_value = history

        result = MrMarket().evaluate("AOS")
        self.assertEqual(result["mr_market_mood"], "fear")
        self.assertLess(result["drawdown_from_high"], -0.20)

    @patch("valuation_capital.yf.Ticker")
    def test_fetch_financial_data_uses_full_history_cagr_not_single_year(self, mock_ticker):
        # FCF was flat for 3 years then jumped in the most recent year. A
        # naive single-year-over-year comparison (old behavior) reads this
        # as +30% -> capped to 15%, extrapolating one lucky year across the
        # entire DCF horizon. Real example that motivated this: AOS's FCF
        # was 321.1/597.7/473.8/546.0 across 2022-2025 -- any single
        # adjacent-year pair swings from -21% to +86%. The full-history CAGR
        # is a much more modest ~9% here, which is what should actually
        # drive the growth assumption.
        dates = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])
        cashflow = pd.DataFrame(
            {dates[0]: [130.0], dates[1]: [100.0], dates[2]: [100.0], dates[3]: [100.0]},
            index=["Free Cash Flow"],
        )
        mock_ticker.return_value.cashflow = cashflow
        mock_ticker.return_value.info = {
            "sharesOutstanding": 10_000_000,
            "totalCash": 50.0,
            "totalDebt": 20.0,
        }

        financials = valuation_capital.fetch_financial_data("TEST")

        expected_growth = (130.0 / 100.0) ** (1 / 3) - 1
        self.assertAlmostEqual(financials["historical_fcf_growth_rate"], expected_growth, places=6)
        self.assertLess(financials["historical_fcf_growth_rate"], 0.15)


if __name__ == "__main__":
    unittest.main()
