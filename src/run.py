import json
import logging
import inspect
import math
from pathlib import Path

import pandas as pd
import yfinance as yf

import business_moat
import comparison_scoring
import financial_metrics
# import industry_playbooks (Removed: Industry playbooks are not universal)
import investment_philosophy
import management_governance
import risk_behavior
import thinking_frameworks
import valuation_capital
from earnings_calls import load_sp500_tickers


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


PIPELINE_REQUIRED_REAL_INPUTS = {
    "MarketForecasting": {"forecast_return"},
}


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
    if type(value).__module__ == 'numpy':
        if hasattr(value, "item"):
            try:
                return _make_json_safe(value.item())
            except Exception:
                pass
        # Fallback for numpy types
        if 'bool' in type(value).__name__.lower():
            return bool(value)
        if 'int' in type(value).__name__.lower():
            return int(value)
        if 'float' in type(value).__name__.lower():
            return float(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _make_json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    # Catch any remaining numpy bools
    if 'bool' in type(value).__name__.lower() and not isinstance(value, bool):
        return bool(value)
    return value


def _write_analysis_output(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(_make_json_safe(data), f, indent=2)


def _build_real_context(ticker: str) -> dict:
    context = {
        "ticker": ticker,
        "tickers": [ticker],
        "company_name": ticker,
        "sector": "",
        "industry": "",
        "description": "",
        "display_description": "",
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
        context["display_description"] = company_info.get("display_description", context["description"])
    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {e}")
        context["company_name"] = info.get("shortName") or ticker
        context["description"] = info.get("longBusinessSummary", "")
        context["display_description"] = context["description"]

    context["sector"] = info.get("sector") or ""
    context["industry"] = info.get("industry") or ""

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
        logger.error(f"Error fetching compounding data for {ticker}: {e}", exc_info=True)

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
    required_real_inputs = PIPELINE_REQUIRED_REAL_INPUTS.get(instance.__class__.__name__, set())

    for param_name, param in params.items():
        if param_name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        value = _resolve_param_value(instance.__class__.__name__, param_name, context)
        if _has_real_value(value):
            kwargs[param_name] = value
        elif param.default is inspect.Parameter.empty or param_name in required_real_inputs:
            missing.append(param_name)

    return kwargs, missing


def analyze_company(ticker: str) -> dict:
    results = {"ticker": ticker}

    logger.info(f"Fetching context data for {ticker}...")
    context = _build_real_context(ticker)

    results["company_name"] = context.get("company_name", ticker)
    results["sector"] = context.get("sector", "")
    results["industry"] = context.get("industry", "")
    results["description"] = context.get("display_description") or context.get("description", "")

    output_filename = OUTPUT_DIR / f"{ticker}_analysis.json"
    _write_analysis_output(output_filename, results)

    heuristic_modules = get_all_heuristic_classes()

    for mod_name, class_list in heuristic_modules.items():
        results[mod_name] = {}
        _write_analysis_output(output_filename, results)
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
                    _write_analysis_output(output_filename, results)
                    continue

                output = instance.evaluate(**kwargs)
                if isinstance(output, pd.DataFrame):
                    output = output.to_dict(orient="records")

                results[mod_name][cls_name] = _make_json_safe(output)
            except Exception as e:
                logger.error(f"Error in {cls_name}: {e}")
                results[mod_name][cls_name] = {"error": str(e)}
            _write_analysis_output(output_filename, results)

    return results


def _is_complete_analysis(output_filename: Path) -> bool:
    """A file only counts as "done" if every top-level heuristic category
    ran. analyze_company() writes incrementally (results[mod_name] = {} is
    written *before* that category's evaluators run), so a hard crash
    mid-run (observed: Ollama OOM-kills the whole process, not just one
    request) leaves a real, parseable, but truncated JSON file on disk --
    missing whichever categories hadn't started yet. Without this check,
    run_continuous_all's resume-by-skip treats that file as finished
    forever. A per-evaluator {"error": ...} is fine and expected; a missing
    top-level category key is not.
    """
    try:
        data = json.loads(output_filename.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all(category in data for category in get_all_heuristic_classes())


def run_continuous_all() -> None:
    """
    Runs the full 49-heuristic pipeline over every S&P 500 ticker, resuming
    automatically: any ticker that already has a *complete*
    output/<TICKER>_analysis.json is skipped, so re-running after a
    crash/interrupt just picks up where it left off -- including redoing
    any ticker whose file was left truncated by a mid-run crash. Failures
    are logged and skipped per-ticker instead of aborting the whole run. No
    cross-company comparison is built (not meaningful at 500-company scale)
    -- use --compare-existing on a handful of tickers for that.
    """
    tickers = [ticker.upper() for ticker in load_sp500_tickers()]
    total = len(tickers)
    completed = 0
    skipped = 0
    failures: dict[str, str] = {}

    for index, ticker_symbol in enumerate(tickers, start=1):
        output_filename = OUTPUT_DIR / f"{ticker_symbol}_analysis.json"
        if output_filename.exists() and _is_complete_analysis(output_filename):
            skipped += 1
            continue

        logger.info(f"[{index}/{total}] Starting exhaustive 49-heuristic pipeline for {ticker_symbol}...")
        try:
            analyze_company(ticker_symbol)
            completed += 1
            logger.info(f"[{index}/{total}] Saved {output_filename}")
        except Exception as e:
            failures[ticker_symbol] = str(e)
            logger.error(f"[{index}/{total}] Failed {ticker_symbol}: {e}")

    logger.info(
        f"continuous-all done: {completed} completed, {skipped} already cached, "
        f"{len(failures)} failed, out of {total} total"
    )
    if failures:
        logger.info(f"Failures: {json.dumps(failures, indent=2)}")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*")
    parser.add_argument(
        "--compare-existing",
        action="store_true",
        help="Treat positional arguments as paths to existing *_analysis.json files and only build a comparison output.",
    )
    parser.add_argument(
        "--continuous-all",
        action="store_true",
        help="Run the full pipeline over every S&P 500 ticker (sp500_tickers.json), resuming "
        "automatically by skipping tickers that already have an output/<TICKER>_analysis.json.",
    )
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.continuous_all:
        run_continuous_all()
        return

    if not args.tickers:
        parser.error("tickers required unless --continuous-all is passed")

    if args.compare_existing:
        analyses = comparison_scoring.load_analysis_files(args.tickers)
        tickers = [analysis.get("ticker", "UNKNOWN") for analysis in analyses]
        if len(analyses) < 2:
            raise ValueError("Need at least two valid analysis files to build a comparison")
    else:
        tickers = [ticker.upper() for ticker in args.tickers]
        for ticker_symbol in tickers:
            logger.info(f"Starting exhaustive 49-heuristic pipeline for {ticker_symbol}...")
            analyze_company(ticker_symbol)
            output_filename = OUTPUT_DIR / f"{ticker_symbol}_analysis.json"
            logger.info(f"Successfully saved full analysis to {output_filename}")

        analyses = comparison_scoring.load_analysis_files(
            [OUTPUT_DIR / f"{ticker_symbol}_analysis.json" for ticker_symbol in tickers]
        )

    if len(analyses) > 1:
        comparison = comparison_scoring.build_comparison(analyses)
        comparison_name = "_vs_".join(tickers)
        comparison_filename = OUTPUT_DIR / f"{comparison_name}_comparison.json"
        with comparison_filename.open("w") as f:
            json.dump(_make_json_safe(comparison), f, indent=2)
        logger.info(f"Successfully saved comparison to {comparison_filename}")


if __name__ == "__main__":
    main()
