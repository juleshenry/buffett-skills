import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

# Add src to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import thinking_frameworks
import investment_philosophy
import business_moat
import management_governance
import financial_metrics
import valuation_capital
import risk_behavior
import industry_playbooks

class TestAOSIntegration(unittest.TestCase):
    
    def setUp(self):
        self.ticker = "AOS"
        self.company_name = "A. O. Smith Corporation"
    
    @patch('thinking_frameworks.call_ollama_panel_json')
    def test_01_thinking_frameworks(self, mock_panel_call):
        mock_panel_call.return_value = {
            "inside_circle": True,
            "confidence": 85,
            "explanation": "AOS manufactures water heaters and boilers, a very straightforward business."
        }

        with patch('thinking_frameworks.fetch_company_info') as mock_info:
            mock_info.return_value = {
                "name": self.company_name,
                "description": "Manufactures commercial and residential water heaters.",
            }

            info = thinking_frameworks.fetch_company_info(self.ticker)
            result = thinking_frameworks.evaluate_simplicity_with_ollama(info['name'], info['description'])

            self.assertIn("inside_circle", result)
            self.assertTrue(result["inside_circle"])
            self.assertIn("explanation", result)

    @patch('yfinance.Ticker')
    def test_02_investment_philosophy(self, mock_yf_ticker):
        # Mock yfinance to prevent network calls and return dummy pandas DataFrames
        import pandas as pd
        import numpy as np
        
        # Create a dummy dataframe for 10 years of trading days
        dates = pd.date_range(start="2014-01-01", end="2024-01-01", freq='B')
        
        # Mock Stock Data
        mock_stock = MagicMock()
        mock_stock.history.return_value = pd.DataFrame(
            index=dates, 
            data={'Close': np.linspace(20, 80, len(dates))}
        )
        
        # Mock Benchmark Data
        mock_bench = MagicMock()
        mock_bench.history.return_value = pd.DataFrame(
            index=dates, 
            data={'Close': np.linspace(1500, 4500, len(dates))}
        )
        
        # yf.Ticker() will return mock_stock first time, mock_bench second time
        mock_yf_ticker.side_effect = [mock_stock, mock_bench]
        
        result = investment_philosophy.analyze_investment_philosophy(self.ticker, years=10)
        
        self.assertEqual(result["ticker"], self.ticker)
        self.assertIn("stock_cagr", result)
        self.assertIn("benchmark_cagr", result)
        self.assertIn("outperformed_benchmark", result)

    @patch('business_moat.call_ollama_panel_json')
    def test_03_business_moat(self, mock_panel_call):
        mock_panel_call.return_value = {
            "moat_type": "Brand",
            "justification": "AOS has a very strong brand presence in the water heater market."
        }

        products = ["Water Heaters", "Boilers"]
        competitors = ["Rheem", "Bradford White"]
        
        result = business_moat.classify_moat_with_ollama(self.company_name, products, competitors)
        
        self.assertEqual(result.get("moat_type"), "Brand")

    @patch('management_governance.fetch_filing_section')
    @patch('management_governance.ManagementEvaluation._call_ollama')
    @patch('management_governance.ManagementEvaluation._fetch_earnings_call_transcript')
    def test_04_management_governance(self, mock_transcript, mock_call_ollama, mock_fetch_section):
        mock_transcript.return_value = "Management discussed capital allocation candidly."
        mock_fetch_section.return_value = "Executive compensation ties bonuses to ROIC."
        mock_call_ollama.return_value = {"response": "Honest management with aligned incentives."}

        result_str = management_governance.analyze_management_governance(self.ticker)

        self.assertEqual(result_str, "Honest management with aligned incentives.")
        mock_transcript.assert_called_once_with(self.ticker)
        mock_fetch_section.assert_called_once()

    @patch('management_governance.ManagementEvaluation._call_ollama')
    def test_04_management_governance_with_explicit_inputs(self, mock_call_ollama):
        mock_call_ollama.return_value = {"response": "Explicit-input management analysis."}

        result_str = management_governance.analyze_management_governance(
            self.ticker,
            transcript="Provided transcript",
            proxy_statement="Provided proxy",
        )

        self.assertEqual(result_str, "Explicit-input management analysis.")

    @patch('financial_metrics.call_ollama_panel_json')
    def test_05_financial_metrics(self, mock_panel_call):
        mock_panel_call.return_value = {
            "maintenance_capex_percentage": 70,
            "growth_capex_percentage": 30,
            "extracted_numbers": ["$50M maintenance", "$20M growth"],
            "reasoning": "Based on MD&A text."
        }
        
        with patch('financial_metrics.fetch_mda_section') as mock_mda:
            mock_mda.return_value = "We spent a lot on maintenance."
            mda_text = financial_metrics.fetch_mda_section(self.ticker)
            result = financial_metrics.query_ollama_capex_breakdown(mda_text)
            
            self.assertEqual(result.get("maintenance_percentage"), 70)

    @patch('valuation_capital.call_ollama_panel_json')
    def test_06_valuation_capital(self, mock_panel_call):
        mock_panel_call.return_value = {
            "systematic_buybacks": True,
            "intrinsic_value_mentioned": True,
            "analysis": "Management explicitly states they only buy below intrinsic value."
        }

        # The valuation capital script requires specific keys to run
        with patch('valuation_capital.fetch_financial_data') as mock_fin:
            mock_fin.return_value = {
                "shares_outstanding": 150_000_000,
                "recent_free_cash_flow": 400_000_000,
                "historical_fcf_growth_rate": 0.08,
                "terminal_multiple": 15,
                "total_debt": 200_000_000,
                "cash_and_equivalents": 50_000_000
            }
            
            result = valuation_capital.process_company(self.ticker)
            
            self.assertEqual(result["ticker"], self.ticker)
            self.assertIn("intrinsic_value_per_share", result["valuation"])
            self.assertTrue(result["capital_allocation_analysis"]["mentions_intrinsic_value"])
            self.assertEqual(result["capital_allocation_analysis"]["buyback_strategy"], "Systematic")

    @patch('risk_behavior.call_ollama_panel_json')
    def test_07_risk_behavior(self, mock_panel_call):
        mock_panel_call.return_value = {
            "total_operating_lease_obligations": 50000000,
            "pension_plan_underfunding_amount": 0,
            "toxic_derivative_exposure": "None",
            "summary": "Clean balance sheet."
        }
        
        with patch('risk_behavior.fetch_sec_10k_footnotes') as mock_fetch:
            mock_fetch.return_value = "Standard lease footnotes for AOS."
            
            footnotes = risk_behavior.fetch_sec_10k_footnotes(self.ticker)
            result = risk_behavior.analyze_footnotes_with_ollama(footnotes)
            
            self.assertIsNotNone(result)
            self.assertEqual(result.get("total_operating_lease_obligations"), 50000000)

    @patch('industry_playbooks.query_ollama')
    def test_08_industry_playbooks(self, mock_query_ollama):
        mock_query_ollama.return_value = {
            "capital_adequate": True,
            "risk_summary": "Adequate"
        }
        
        # AOS isn't a bank or insurance, but let's test the bank logic anyway 
        # to ensure the function works. We'll capture stdout to verify it runs.
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            industry_playbooks.analyze_bank(self.ticker)
            
        output = f.getvalue()
        self.assertIn(self.ticker, output)
        self.assertIn("Adequate", output)

if __name__ == "__main__":
    unittest.main()
