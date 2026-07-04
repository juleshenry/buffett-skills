import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from financial_metrics import KeyFinancialMetrics
from financial_metrics import LookthroughEarnings
from business_moat import BusinessModelTypes, TheDurabilityOfCompetitiveAdvantage
from business_moat import GoodwillEconomicGoodwillVsAccountingGoodwill
from investment_philosophy import EfficientMarketTheory, FocusInvesting, IntrinsicValue, MarketForecasting, UndervaluedMarginOfSafety
from management_governance import CorporateGovernanceAndShareholderOrientation
from management_governance import AcquisitionLogicAcquisitionCriteria, CorporateCulture
from thinking_frameworks import IndependentThinking, LongtermOrientation, MrMarket, MungersLatticeOfMentalModels, OpportunityCostAwareness, PatienceAsEdge
from valuation_capital import CapitalAllocationAnalysis, MarginOfSafety, TheRelationshipBetweenPurchasePriceAndIntrinsicValue
from valuation_capital import DividendsRetainedEarningsAndTaxEfficiency
from valuation_capital import SpecialInvestmentInstruments
from evaluator_config import DEFAULT_INTRINSIC_VALUE_YEARS
from sec_data import extract_keyword_context
from risk_behavior import ValueTraps, WhenToSellClearCriteria
from risk_behavior import CommonBehavioralBiasesPsychologicalTrapsInInvesting, DerivativesRisk
from financial_metrics import CorePrincipleSeeThroughAccountingToEconomicReality, OwnerEarnings, normalize_capex_breakdown
from industry_playbooks import ConsumerBrandsRetail, EnergyUtilities, IndustriesToAvoidCounterexamples, InsuranceFloat, MediaPublishing, Railways, TechnologyInternet, UnderwritingDiscipline
import investment_philosophy
import management_governance
import thinking_frameworks
import valuation_capital
from management_governance import ManagementEvaluation
from risk_behavior import LeverageRisk
from valuation_capital import normalize_buyback_analysis


class TestMinimalHeuristics(unittest.TestCase):
    def test_margin_of_safety(self):
        result = MarginOfSafety().evaluate(intrinsic_value=100.0, market_price=70.0)
        self.assertAlmostEqual(result["margin_of_safety"], 0.30)
        self.assertTrue(result["is_discount"])

    def test_purchase_price_relationship(self):
        result = TheRelationshipBetweenPurchasePriceAndIntrinsicValue().evaluate(
            intrinsic_value=100.0,
            market_price=72.0,
        )
        self.assertAlmostEqual(result["margin_of_safety"], 0.28)
        self.assertEqual(result["purchase_price_verdict"], "deep_discount")

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

    def test_when_to_sell(self):
        result = WhenToSellClearCriteria().evaluate(
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
        result = LongtermOrientation().evaluate(stock_cagr=0.14, benchmark_cagr=0.09, years=7)
        self.assertEqual(result["long_term_orientation"], "strong")
        self.assertAlmostEqual(result["excess_return"], 0.05)

    @patch("thinking_frameworks.fetch_price_comparison_data")
    def test_long_term_orientation_fetches_real_price_data_for_ticker(self, mock_price_data):
        mock_price_data.return_value = {
            "stock_cagr": 0.14,
            "benchmark_cagr": 0.09,
            "period_years": 7.0,
        }

        result = LongtermOrientation().evaluate(ticker="AAPL")
        self.assertEqual(result["long_term_orientation"], "strong")
        mock_price_data.assert_called_once()

    def test_opportunity_cost_awareness(self):
        result = OpportunityCostAwareness().evaluate(
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
        result = LookthroughEarnings().evaluate(
            ownership_percentage=0.25,
            investee_net_income=200.0,
            dividends_received=20.0,
        )
        self.assertAlmostEqual(result["lookthrough_earnings"], 50.0)
        self.assertAlmostEqual(result["retained_earnings_share"], 30.0)

    def test_business_model_type(self):
        result = BusinessModelTypes().evaluate(
            recurring_revenue_ratio=0.8,
            gross_margin=0.6,
            capital_intensity=0.04,
        )
        self.assertEqual(result["business_model_type"], "recurring_revenue")

    @patch("business_moat.fetch_deep_financials")
    @patch("business_moat.fetch_historical_margins")
    @patch("business_moat.fetch_company_info")
    def test_business_model_type_fetches_real_company_data(self, mock_company_info, mock_margins, mock_financials):
        mock_company_info.return_value = {
            "description": "The company sells software as a service subscriptions with recurring renewals."
        }
        mock_margins.return_value = pd.DataFrame([{"Year": 2024, "Gross_Margin": 0.7}])
        mock_financials.return_value = {"total_revenue": 1000.0, "capex_total": -50.0}

        result = BusinessModelTypes().evaluate(ticker="AAPL")
        self.assertEqual(result["business_model_type"], "recurring_revenue")

    def test_durability_of_competitive_advantage(self):
        result = TheDurabilityOfCompetitiveAdvantage().evaluate(
            gross_margin_trend=0.02,
            market_share_trend=0.01,
            return_on_capital=0.15,
        )
        self.assertEqual(result["durability_score"], 3)
        self.assertEqual(result["durability_assessment"], "strong")

    def test_underwriting_discipline(self):
        result = UnderwritingDiscipline().evaluate(combined_ratio=97.0)
        self.assertEqual(result["underwriting_discipline"], "disciplined")
        self.assertTrue(result["profitable_underwriting"])

    def test_insurance_float(self):
        result = InsuranceFloat().evaluate(
            current_float=1200.0,
            prior_float=1000.0,
            combined_ratio=96.0,
        )
        self.assertEqual(result["float_growth"], 200.0)
        self.assertEqual(result["float_quality"], "valuable")

    def test_focus_investing(self):
        result = FocusInvesting().evaluate([40.0, 25.0, 15.0, 10.0, 10.0])
        self.assertEqual(result["position_count"], 5)
        self.assertTrue(result["is_focus_investing"])
        self.assertAlmostEqual(result["top_three_weight"], 0.8)

    def test_efficient_market_theory(self):
        result = EfficientMarketTheory().evaluate(
            stock_cagr=0.09,
            benchmark_cagr=0.08,
            tracking_error=0.03,
        )
        self.assertTrue(result["market_efficiency_supported"])

    @patch("investment_philosophy.fetch_price_comparison_data")
    def test_efficient_market_theory_fetches_real_price_data_for_ticker(self, mock_price_data):
        mock_price_data.return_value = {
            "stock_cagr": 0.09,
            "benchmark_cagr": 0.08,
            "tracking_error": 0.03,
        }
        result = EfficientMarketTheory().evaluate(ticker="AAPL")
        self.assertTrue(result["market_efficiency_supported"])
        mock_price_data.assert_called_once()

    def test_market_forecasting(self):
        result = MarketForecasting().evaluate(
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
        result = IndependentThinking().evaluate(
            thesis_differs_from_consensus=True,
            evidence_strength=0.8,
            valuation_gap=0.20,
        )
        self.assertEqual(result["independent_thinking_score"], 3)
        self.assertEqual(result["independent_thinking"], "strong")

    def test_corporate_culture(self):
        result = CorporateCulture().evaluate(
            employee_turnover=0.10,
            insider_ownership=0.06,
            restructurings_per_5y=1,
        )
        self.assertEqual(result["culture_score"], 3)
        self.assertEqual(result["culture_assessment"], "strong")

    def test_acquisition_logic(self):
        result = AcquisitionLogicAcquisitionCriteria().evaluate(
            purchase_multiple=10.0,
            return_on_invested_capital=0.14,
            debt_funded=False,
        )
        self.assertEqual(result["acquisition_score"], 3)
        self.assertEqual(result["acquisition_discipline"], "disciplined")

    def test_special_investment_instruments(self):
        result = SpecialInvestmentInstruments().evaluate(
            coupon_rate=0.09,
            conversion_discount=0.12,
            collateral_coverage=1.2,
        )
        self.assertEqual(result["instrument_score"], 3)
        self.assertEqual(result["instrument_attractiveness"], "high")

    def test_behavioral_biases(self):
        result = CommonBehavioralBiasesPsychologicalTrapsInInvesting().evaluate(
            thesis_changes_after_price_move=True,
            avg_holding_period_years=0.5,
            adds_to_losers_without_new_evidence=False,
        )
        self.assertEqual(result["bias_count"], 2)
        self.assertEqual(result["behavioral_risk"], "high")

    def test_mungers_lattice(self):
        result = MungersLatticeOfMentalModels().evaluate(
            economics_score=0.8,
            psychology_score=0.7,
            accounting_score=0.9,
        )
        self.assertEqual(result["lattice_assessment"], "broad")

    def test_patience_as_edge(self):
        result = PatienceAsEdge().evaluate(
            avg_holding_period_years=5,
            turnover_ratio=0.20,
            forced_activity=False,
        )
        self.assertEqual(result["patience_score"], 3)
        self.assertEqual(result["patience_as_edge"], "strong")

    def test_consumer_brands_retail(self):
        result = ConsumerBrandsRetail().evaluate(
            gross_margin=0.40,
            same_store_sales_growth=0.03,
            brand_share_trend=0.01,
        )
        self.assertEqual(result["consumer_brand_quality"], "strong")

    def test_media_publishing(self):
        result = MediaPublishing().evaluate(
            subscription_revenue_ratio=0.7,
            ad_revenue_ratio=0.3,
            churn_rate=0.08,
        )
        self.assertEqual(result["media_quality"], "strong")

    def test_energy_utilities(self):
        result = EnergyUtilities().evaluate(
            regulated_asset_ratio=0.8,
            debt_to_ebitda=4.5,
            allowed_return_on_equity=0.10,
        )
        self.assertEqual(result["utility_quality"], "strong")

    def test_railways(self):
        result = Railways().evaluate(
            operating_ratio=0.62,
            volume_growth=0.01,
            maintenance_capex_ratio=0.5,
        )
        self.assertEqual(result["railway_quality"], "strong")

    def test_technology_internet(self):
        result = TechnologyInternet().evaluate(
            recurring_revenue_ratio=0.75,
            net_revenue_retention=1.1,
            stock_comp_ratio=0.10,
        )
        self.assertEqual(result["technology_quality"], "strong")

    def test_industries_to_avoid(self):
        result = IndustriesToAvoidCounterexamples().evaluate(
            commodity_exposure=0.8,
            leverage_ratio=4.5,
            pricing_power=0.2,
        )
        self.assertEqual(result["red_flag_count"], 3)
        self.assertTrue(result["avoid_industry"])

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

    @patch("management_governance.fetch_filing_keyword_context")
    def test_management_evaluation_fetches_sec_commentary(self, mock_fetch_context):
        mock_fetch_context.return_value = "Management discussed capital allocation and operating conditions."
        result = ManagementEvaluation()._fetch_earnings_call_transcript("YUM")
        self.assertIn("capital allocation", result)

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
        result = valuation_capital.fetch_management_commentary("YUM")
        self.assertIn("repurchased shares", result)

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

    @patch("investment_philosophy.Compounding.evaluate")
    def test_analyze_investment_philosophy(self, mock_evaluate):
        mock_evaluate.return_value = pd.DataFrame([
            {
                "Ticker": "YUM",
                "Period (Yrs)": 10.0,
                "Stock CAGR": 0.12,
                "Benchmark CAGR": 0.10,
                "Outperformed?": "Yes",
            }
        ])
        result = investment_philosophy.analyze_investment_philosophy("YUM")
        self.assertEqual(result["ticker"], "YUM")
        self.assertTrue(result["outperformed_benchmark"])

    @patch("thinking_frameworks.yf.Ticker")
    def test_mr_market_price_regime(self, mock_ticker):
        close = [100.0] * 210 + [80.0, 79.0, 78.0, 77.0, 76.0]
        history = pd.DataFrame({"Close": close})
        mock_ticker.return_value.history.return_value = history

        result = MrMarket().evaluate("AOS")
        self.assertEqual(result["mr_market_mood"], "fear")
        self.assertLess(result["drawdown_from_high"], -0.20)


if __name__ == "__main__":
    unittest.main()
