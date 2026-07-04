import json
import logging
import inspect
import math
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import business_moat
from . import financial_metrics
from . import industry_playbooks
from . import investment_philosophy
from . import management_governance
from . import risk_behavior
from . import thinking_frameworks
from . import valuation_capital


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def get_all_heuristic_classes():
    modules = [
        thinking_frameworks,
        investment_philosophy,
        business_moat,
        management_governance,
        financial_metrics,
        valuation_capital,
        risk_behavior,
        industry_playbooks,
    ]

    classes = {}
    for mod in modules:
        mod_name = mod.__name__.split(".")[-1]
        classes[mod_name] = []
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__ and obj.__doc__ and "Heuristic:" in obj.__doc__:
                classes[mod_name].append(obj)
    return classes


def _make_json_safe(value):
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze_company(ticker: str) -> dict:
    results = {"ticker": ticker}

    logger.info(f"Fetching context data for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        info = t.info
        company_name = info.get("shortName", ticker)
        description = info.get("longBusinessSummary", "")
        fcf = info.get("freeCashflow", 0)
        shares = info.get("sharesOutstanding", 1)
        net_debt = info.get("totalDebt", 0) - info.get("totalCash", 0)
    except Exception as e:
        logger.error(f"Error fetching base yfinance info: {e}")
        company_name = ticker
        description = "Information unavailable."
        fcf = 0
        shares = 1
        net_debt = 0

    results["company_name"] = company_name
    results["description"] = description

    history = None
    try:
        history = t.history(period="10y") if "t" in locals() else None
    except Exception:
        history = None

    current_price = 0.0
    benchmark_cagr = 0.10
    stock_cagr = 0.08
    if history is not None and not history.empty and "Close" in history:
        close = history["Close"].dropna()
        if not close.empty:
            current_price = float(close.iloc[-1])
            years_observed = max((close.index[-1] - close.index[0]).days / 365.25, 1)
            if close.iloc[0] > 0:
                stock_cagr = (float(close.iloc[-1]) / float(close.iloc[0])) ** (1 / years_observed) - 1

    moat_details = {"products": [], "competitors": []}
    try:
        moat_details = business_moat.infer_business_details(ticker)
    except Exception:
        pass

    try:
        fin_data = financial_metrics.fetch_deep_financials(ticker)
    except Exception:
        fin_data = {}

    total_revenue = float(fin_data.get("total_revenue") or 0)
    net_income = float(fin_data.get("net_income") or 0)
    operating_cash_flow = float(fin_data.get("operating_cash_flow") or 0)
    capex_total = float(fin_data.get("capex_total") or 0)

    heuristic_modules = get_all_heuristic_classes()

    for mod_name, class_list in heuristic_modules.items():
        results[mod_name] = {}
        for cls in class_list:
            cls_name = cls.__name__
            logger.info(f"Evaluating {cls_name}...")
            try:
                instance = cls()
                params = inspect.signature(instance.evaluate).parameters

                kwargs = {}
                if "ticker" in params: kwargs["ticker"] = ticker
                if "company_name" in params: kwargs["company_name"] = company_name
                if "description" in params: kwargs["description"] = description
                if "commentary" in params: kwargs["commentary"] = description
                if "fcf" in params: kwargs["fcf"] = fcf
                if "shares_outstanding" in params: kwargs["shares_outstanding"] = shares
                if "net_debt" in params: kwargs["net_debt"] = net_debt
                if "growth_rate" in params: kwargs["growth_rate"] = 0.08
                if "discount_rate" in params: kwargs["discount_rate"] = 0.10
                if "terminal_growth_rate" in params: kwargs["terminal_growth_rate"] = 0.02
                if "years" in params: kwargs["years"] = 10
                if "tickers" in params: kwargs["tickers"] = [ticker]
                if "products" in params: kwargs["products"] = moat_details.get("products", [])
                if "competitors" in params: kwargs["competitors"] = moat_details.get("competitors", [])
                if "positions" in params: kwargs["positions"] = [40.0, 25.0, 15.0, 10.0, 10.0]
                if "stock_cagr" in params: kwargs["stock_cagr"] = stock_cagr
                if "benchmark_cagr" in params: kwargs["benchmark_cagr"] = benchmark_cagr
                if "tracking_error" in params: kwargs["tracking_error"] = 0.03
                if "forecast_return" in params: kwargs["forecast_return"] = benchmark_cagr
                if "actual_return" in params: kwargs["actual_return"] = stock_cagr
                if "candidate_return" in params: kwargs["candidate_return"] = 0.15
                if "hurdle_return" in params: kwargs["hurdle_return"] = 0.10
                if "alternative_return" in params: kwargs["alternative_return"] = 0.12
                if "intrinsic_value" in params: kwargs["intrinsic_value"] = max(current_price * 1.2, 1.0)
                if "market_price" in params: kwargs["market_price"] = current_price or 1.0
                if "recurring_revenue_ratio" in params: kwargs["recurring_revenue_ratio"] = 0.7
                if "gross_margin" in params: kwargs["gross_margin"] = 0.4
                if "capital_intensity" in params: kwargs["capital_intensity"] = 0.08
                if "goodwill" in params: kwargs["goodwill"] = 500.0
                if "acquired_earnings" in params: kwargs["acquired_earnings"] = 100.0
                if "return_on_tangible_assets" in params: kwargs["return_on_tangible_assets"] = 0.12
                if "gross_margin_trend" in params: kwargs["gross_margin_trend"] = 0.0
                if "market_share_trend" in params: kwargs["market_share_trend"] = 0.0
                if "return_on_capital" in params: kwargs["return_on_capital"] = 0.12
                if "purchase_multiple" in params: kwargs["purchase_multiple"] = 12.0
                if "return_on_invested_capital" in params: kwargs["return_on_invested_capital"] = 0.12
                if "debt_funded" in params: kwargs["debt_funded"] = False
                if "employee_turnover" in params: kwargs["employee_turnover"] = 0.12
                if "insider_ownership" in params: kwargs["insider_ownership"] = 0.06
                if "restructurings_per_5y" in params: kwargs["restructurings_per_5y"] = 1
                if "roic_linked_pay" in params: kwargs["roic_linked_pay"] = True
                if "dual_class_structure" in params: kwargs["dual_class_structure"] = False
                if "buybacks_below_intrinsic_value" in params: kwargs["buybacks_below_intrinsic_value"] = False
                if "total_revenue" in params: kwargs["total_revenue"] = total_revenue
                if "net_income" in params: kwargs["net_income"] = net_income
                if "operating_cash_flow" in params: kwargs["operating_cash_flow"] = operating_cash_flow
                if "capex_total" in params: kwargs["capex_total"] = capex_total
                if "ownership_percentage" in params: kwargs["ownership_percentage"] = 0.25
                if "investee_net_income" in params: kwargs["investee_net_income"] = max(net_income, 100.0)
                if "dividends_received" in params: kwargs["dividends_received"] = 20.0
                if "recent_free_cash_flow" in params: kwargs["recent_free_cash_flow"] = float(fcf or 0)
                if "total_debt" in params: kwargs["total_debt"] = max(net_debt + 1000.0, 0.0)
                if "cash_and_equivalents" in params: kwargs["cash_and_equivalents"] = max(kwargs.get("total_debt", 0.0) - net_debt, 0.0)
                if "dividend_payout_ratio" in params: kwargs["dividend_payout_ratio"] = 0.3
                if "retained_return_on_equity" in params: kwargs["retained_return_on_equity"] = 0.15
                if "tax_rate_on_dividends" in params: kwargs["tax_rate_on_dividends"] = 0.2
                if "coupon_rate" in params: kwargs["coupon_rate"] = 0.08
                if "conversion_discount" in params: kwargs["conversion_discount"] = 0.1
                if "collateral_coverage" in params: kwargs["collateral_coverage"] = 1.0
                if "thesis_changes_after_price_move" in params: kwargs["thesis_changes_after_price_move"] = False
                if "avg_holding_period_years" in params: kwargs["avg_holding_period_years"] = 3.0
                if "turnover_ratio" in params: kwargs["turnover_ratio"] = 0.2
                if "forced_activity" in params: kwargs["forced_activity"] = False
                if "adds_to_losers_without_new_evidence" in params: kwargs["adds_to_losers_without_new_evidence"] = False
                if "notional_exposure" in params: kwargs["notional_exposure"] = 100.0
                if "equity_capital" in params: kwargs["equity_capital"] = 100.0
                if "level_3_assets_ratio" in params: kwargs["level_3_assets_ratio"] = 0.02
                if "margins_df" in params: kwargs["margins_df"] = pd.DataFrame([{"Year": 2023, "Gross_Margin": 0.4, "Operating_Margin": 0.2}, {"Year": 2024, "Gross_Margin": 0.41, "Operating_Margin": 0.21}])
                if "inflation_df" in params: kwargs["inflation_df"] = pd.DataFrame([{"Year": 2023, "Inflation_Rate": 3.5}, {"Year": 2024, "Inflation_Rate": 2.8}])
                if "pe_ratio" in params: kwargs["pe_ratio"] = 15.0
                if "revenue_growth" in params: kwargs["revenue_growth"] = 0.03
                if "free_cash_flow_growth" in params: kwargs["free_cash_flow_growth"] = 0.04
                if "debt_to_equity" in params: kwargs["debt_to_equity"] = 0.8
                if "thesis_broken" in params: kwargs["thesis_broken"] = False
                if "better_opportunity_available" in params: kwargs["better_opportunity_available"] = False
                if "extreme_overvaluation" in params: kwargs["extreme_overvaluation"] = False
                if "balance_sheet_deterioration" in params: kwargs["balance_sheet_deterioration"] = False
                if "combined_ratio" in params: kwargs["combined_ratio"] = 98.0
                if "current_float" in params: kwargs["current_float"] = 1200.0
                if "prior_float" in params: kwargs["prior_float"] = 1000.0
                if "same_store_sales_growth" in params: kwargs["same_store_sales_growth"] = 0.02
                if "brand_share_trend" in params: kwargs["brand_share_trend"] = 0.01
                if "subscription_revenue_ratio" in params: kwargs["subscription_revenue_ratio"] = 0.6
                if "ad_revenue_ratio" in params: kwargs["ad_revenue_ratio"] = 0.4
                if "churn_rate" in params: kwargs["churn_rate"] = 0.08
                if "regulated_asset_ratio" in params: kwargs["regulated_asset_ratio"] = 0.8
                if "debt_to_ebitda" in params: kwargs["debt_to_ebitda"] = 4.0
                if "allowed_return_on_equity" in params: kwargs["allowed_return_on_equity"] = 0.1
                if "operating_ratio" in params: kwargs["operating_ratio"] = 0.65
                if "volume_growth" in params: kwargs["volume_growth"] = 0.01
                if "maintenance_capex_ratio" in params: kwargs["maintenance_capex_ratio"] = 0.5
                if "net_revenue_retention" in params: kwargs["net_revenue_retention"] = 1.05
                if "stock_comp_ratio" in params: kwargs["stock_comp_ratio"] = 0.1
                if "commodity_exposure" in params: kwargs["commodity_exposure"] = 0.2
                if "leverage_ratio" in params: kwargs["leverage_ratio"] = 2.0
                if "pricing_power" in params: kwargs["pricing_power"] = 0.7
                if "thesis_differs_from_consensus" in params: kwargs["thesis_differs_from_consensus"] = True
                if "evidence_strength" in params: kwargs["evidence_strength"] = 0.75
                if "valuation_gap" in params: kwargs["valuation_gap"] = 0.2
                if "economics_score" in params: kwargs["economics_score"] = 0.7
                if "psychology_score" in params: kwargs["psychology_score"] = 0.7
                if "accounting_score" in params: kwargs["accounting_score"] = 0.7

                output = instance.evaluate(**kwargs)
                if isinstance(output, pd.DataFrame):
                    output = output.to_dict(orient="records")

                results[mod_name][cls_name] = _make_json_safe(output)
            except Exception as e:
                logger.error(f"Error in {cls_name}: {e}")
                results[mod_name][cls_name] = {"error": str(e)}

    return results


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args(argv)

    ticker_symbol = args.ticker.upper()
    logger.info(f"Starting exhaustive 49-heuristic pipeline for {ticker_symbol}...")

    final_output = _make_json_safe(analyze_company(ticker_symbol))
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_filename = OUTPUT_DIR / f"{ticker_symbol}_analysis.json"
    with output_filename.open("w") as f:
        json.dump(final_output, f, indent=2)
    logger.info(f"Successfully saved full analysis to {output_filename}")


if __name__ == "__main__":
    main()
