import json
import logging
import inspect
import math
from pathlib import Path

import pandas as pd
import yfinance as yf

import business_moat
import financial_metrics
import industry_playbooks
import investment_philosophy
import management_governance
import risk_behavior
import thinking_frameworks
import valuation_capital


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_real_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, pd.DataFrame):
        return not value.empty
    return True


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


def _build_real_context(ticker: str) -> dict:
    context = {
        "ticker": ticker,
        "tickers": [ticker],
        "company_name": ticker,
        "description": "",
        "commentary": "",
    }

    info = {}
    ticker_obj = None
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info or {}
    except Exception as e:
        logger.error(f"Error fetching base yfinance info for {ticker}: {e}")

    try:
        company_info = thinking_frameworks.fetch_company_info(ticker)
        context["company_name"] = company_info.get("name") or info.get("shortName") or ticker
        context["description"] = company_info.get("description", "")
    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {e}")
        context["company_name"] = info.get("shortName") or ticker
        context["description"] = info.get("longBusinessSummary", "")

    context["fcf"] = _coerce_float(info.get("freeCashflow"))
    context["recent_free_cash_flow"] = context["fcf"]
    context["shares_outstanding"] = _coerce_float(info.get("sharesOutstanding"))
    context["cash_and_equivalents"] = _coerce_float(info.get("totalCash"))
    context["total_debt"] = _coerce_float(info.get("totalDebt"))
    if context["cash_and_equivalents"] is not None and context["total_debt"] is not None:
        context["net_debt"] = context["total_debt"] - context["cash_and_equivalents"]
    else:
        context["net_debt"] = None
    context["market_price"] = _coerce_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    context["pe_ratio"] = _coerce_float(info.get("trailingPE") or info.get("forwardPE"))
    context["insider_ownership"] = _coerce_float(info.get("heldPercentInsiders"))

    debt_to_equity = _coerce_float(info.get("debtToEquity"))
    context["debt_to_equity"] = debt_to_equity / 100.0 if debt_to_equity is not None else None

    try:
        context["commentary"] = valuation_capital.fetch_management_commentary(ticker)
    except Exception as e:
        logger.error(f"Error fetching management commentary for {ticker}: {e}")

    try:
        history = ticker_obj.history(period="10y") if ticker_obj is not None else None
    except Exception as e:
        logger.error(f"Error fetching price history for {ticker}: {e}")
        history = None

    if history is not None and not history.empty and "Close" in history:
        close = history["Close"].dropna()
        if not close.empty:
            context["market_price"] = _coerce_float(close.iloc[-1])

    try:
        compounding = investment_philosophy.analyze_investment_philosophy(ticker)
        if "error" not in compounding:
            context["stock_cagr"] = _coerce_float(compounding.get("stock_cagr"))
            context["benchmark_cagr"] = _coerce_float(compounding.get("benchmark_cagr"))
            context["long_term_years"] = _coerce_float(compounding.get("period_years"))
    except Exception as e:
        logger.error(f"Error fetching compounding data for {ticker}: {e}")

    try:
        moat_details = business_moat.infer_business_details(ticker)
        context["products"] = moat_details.get("products", [])
        context["competitors"] = moat_details.get("competitors", [])
    except Exception as e:
        logger.error(f"Error inferring moat details for {ticker}: {e}")
        context["products"] = []
        context["competitors"] = []

    try:
        financials = financial_metrics.fetch_deep_financials(ticker)
    except Exception as e:
        logger.error(f"Error fetching deep financials for {ticker}: {e}")
        financials = {}

    context["total_revenue"] = _coerce_float(financials.get("total_revenue"))
    context["net_income"] = _coerce_float(financials.get("net_income"))
    context["operating_cash_flow"] = _coerce_float(financials.get("operating_cash_flow"))
    context["capex_total"] = _coerce_float(financials.get("capex_total"))

    if context["total_revenue"] and context["capex_total"] is not None:
        context["capital_intensity"] = abs(context["capex_total"]) / context["total_revenue"]

    try:
        margins_df = business_moat.fetch_historical_margins(ticker)
    except Exception as e:
        logger.error(f"Error fetching historical margins for {ticker}: {e}")
        margins_df = pd.DataFrame()

    context["margins_df"] = margins_df
    if not margins_df.empty:
        latest_margin_row = margins_df.sort_values("Year").iloc[-1]
        context["gross_margin"] = _coerce_float(latest_margin_row.get("Gross_Margin"))
        if len(margins_df) >= 2:
            sorted_margins = margins_df.sort_values("Year")
            last_two = sorted_margins["Gross_Margin"].dropna().tail(2)
            if len(last_two) == 2:
                context["gross_margin_trend"] = _coerce_float(last_two.iloc[-1] - last_two.iloc[-2])

        start_year = int(margins_df["Year"].min())
        end_year = int(margins_df["Year"].max())
        try:
            context["inflation_df"] = business_moat.fetch_cpi_inflation_data(start_year, end_year)
        except Exception as e:
            logger.error(f"Error fetching CPI data for {ticker}: {e}")
            context["inflation_df"] = pd.DataFrame()
    else:
        context["inflation_df"] = pd.DataFrame()

    return context


def _resolve_param_value(class_name: str, param_name: str, context: dict):
    if class_name == "LongtermOrientation" and param_name == "years":
        return context.get("long_term_years")
    return context.get(param_name)


def _prepare_evaluator_inputs(instance, context: dict) -> tuple[dict, list[str]]:
    params = inspect.signature(instance.evaluate).parameters
    kwargs = {}
    missing = []

    for param_name, param in params.items():
        if param_name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        value = _resolve_param_value(instance.__class__.__name__, param_name, context)
        if _has_real_value(value):
            kwargs[param_name] = value
        elif param.default is inspect.Parameter.empty:
            missing.append(param_name)

    return kwargs, missing


def analyze_company(ticker: str) -> dict:
    results = {"ticker": ticker}

    logger.info(f"Fetching context data for {ticker}...")
    context = _build_real_context(ticker)

    results["company_name"] = context.get("company_name", ticker)
    results["description"] = context.get("description", "")

    heuristic_modules = get_all_heuristic_classes()

    for mod_name, class_list in heuristic_modules.items():
        results[mod_name] = {}
        for cls in class_list:
            cls_name = cls.__name__
            logger.info(f"Evaluating {cls_name}...")
            try:
                instance = cls()
                kwargs, missing = _prepare_evaluator_inputs(instance, context)
                if missing:
                    results[mod_name][cls_name] = {
                        "error": "Missing real inputs",
                        "missing_inputs": missing,
                    }
                    continue

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
