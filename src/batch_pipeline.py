import json
import logging
import sys
from typing import Dict, Any

# Import all 8 workflow modules
import thinking_frameworks
import financial_metrics
import business_moat
import valuation_capital
import management_governance
import risk_behavior
import industry_playbooks
import investment_philosophy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_company(ticker: str) -> dict:
    """
    Runs all 8 workflows on a single ticker, wrapping each step in a try/except 
    block to gracefully handle missing data or connection errors (e.g., from Ollama).
    """
    results = {"ticker": ticker}

    # 1. Thinking Frameworks (Simplicity Analysis)
    logger.info(f"Running Thinking Frameworks for {ticker}...")
    try:
        info = thinking_frameworks.fetch_company_info(ticker)
        name = info.get("name", ticker)
        desc = info.get("description", "")
        simplicity = thinking_frameworks.CircleOfCompetence().evaluate(name, desc)
        results["thinking_frameworks"] = {
            "info": info,
            "simplicity_analysis": simplicity
        }
    except Exception as e:
        logger.error(f"Error in thinking_frameworks for {ticker}: {e}")
        results["thinking_frameworks"] = {"error": str(e)}

    # 2. Financial Metrics (Deep Financials & CapEx Breakdown)
    logger.info(f"Running Financial Metrics for {ticker}...")
    try:
        fin_data = financial_metrics.fetch_deep_financials(ticker)
        mda_text = financial_metrics.fetch_mda_section(ticker)
        capex_analysis = financial_metrics.query_ollama_capex_breakdown(mda_text)
        results["financial_metrics"] = {
            "financials": fin_data,
            "capex_analysis": capex_analysis
        }
    except Exception as e:
        logger.error(f"Error in financial_metrics for {ticker}: {e}")
        results["financial_metrics"] = {"error": str(e)}

    # 3. Business Moat (Moat Classification & Products)
    logger.info(f"Running Business Moat for {ticker}...")
    try:
        details = business_moat.infer_business_details(ticker)
        moat_analysis = business_moat.classify_moat_with_ollama(
            company_name=details.get("company_name", ticker),
            products=details.get("products", []),
            competitors=details.get("competitors", [])
        )
        results["business_moat"] = {
            "details": details,
            "moat_analysis": moat_analysis
        }
    except Exception as e:
        logger.error(f"Error in business_moat for {ticker}: {e}")
        results["business_moat"] = {"error": str(e)}

    # 4. Valuation Capital (DCF & Buyback analysis)
    logger.info(f"Running Valuation Capital for {ticker}...")
    try:
        val_data = valuation_capital.process_company(ticker)
        results["valuation_capital"] = val_data
    except Exception as e:
        logger.error(f"Error in valuation_capital for {ticker}: {e}")
        results["valuation_capital"] = {"error": str(e)}

    # 5. Management Governance (Earnings call & proxy statement NLP)
    logger.info(f"Running Management Governance for {ticker}...")
    try:
        gov_data_str = management_governance.ManagementEvaluation().evaluate(ticker)
        # Attempt to parse it if it returns a JSON string as expected
        try:
            gov_data = json.loads(gov_data_str) if gov_data_str else {}
        except json.JSONDecodeError:
            gov_data = {"raw_output": gov_data_str}
        results["management_governance"] = gov_data
    except Exception as e:
        logger.error(f"Error in management_governance for {ticker}: {e}")
        results["management_governance"] = {"error": str(e)}

    # 6. Risk Behavior (10-K Footnotes Analysis)
    logger.info(f"Running Risk Behavior for {ticker}...")
    try:
        footnotes = risk_behavior.fetch_sec_10k_footnotes(ticker)
        risk_analysis = risk_behavior.analyze_footnotes_with_ollama(footnotes)
        results["risk_behavior"] = risk_analysis
    except Exception as e:
        logger.error(f"Error in risk_behavior for {ticker}: {e}")
        results["risk_behavior"] = {"error": str(e)}

    # 7. Industry Playbooks (Banking/Insurance standard metrics fetcher)
    logger.info(f"Running Industry Playbooks for {ticker}...")
    try:
        # Calling fetch_industry_data generically
        ind_data = industry_playbooks.fetch_industry_data(ticker, "general")
        results["industry_playbooks"] = ind_data
    except Exception as e:
        logger.error(f"Error in industry_playbooks for {ticker}: {e}")
        results["industry_playbooks"] = {"error": str(e)}

    # 8. Investment Philosophy (CAGR & Performance vs S&P 500)
    logger.info(f"Running Investment Philosophy for {ticker}...")
    try:
        inv_phil = investment_philosophy.analyze_investment_philosophy(ticker)
        if "outperformed_benchmark" in inv_phil:
            inv_phil["outperformed_benchmark"] = bool(inv_phil["outperformed_benchmark"])
        results["investment_philosophy"] = inv_phil
    except Exception as e:
        logger.error(f"Error in investment_philosophy for {ticker}: {e}")
        results["investment_philosophy"] = {"error": str(e)}

    return results

if __name__ == "__main__":
    ticker_symbol = "ZTS"
    logger.info(f"Starting batch pipeline for {ticker_symbol}...")
    
    final_output = analyze_company(ticker_symbol)
    
    print("\n--- FINAL BATCH PIPELINE RESULTS ---\n")
    print(json.dumps(final_output, indent=2))

    import os
    os.makedirs("output", exist_ok=True)
    output_filename = f"output/{ticker_symbol}_analysis.json"
    with open(output_filename, "w") as f:
        json.dump(final_output, f, indent=2)
    logger.info(f"Successfully saved analysis to {output_filename}")
