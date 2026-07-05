from cache_utils import disk_cache
import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Optional

from evaluator_config import DEFAULT_BENCHMARK, DEFAULT_INTRINSIC_VALUE_YEARS, DEFAULT_LOOKBACK_YEARS
from evaluator_thresholds import (
    UNDERVALUED_MARGIN_MIN,
)
from valuation_capital import fetch_risk_free_rate
from valuation_capital import IntrinsicValueEstimation, MarginOfSafety


def _calculate_cagr(start_value: float, end_value: float, years: float) -> float:
    if start_value <= 0 or years <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1


def fetch_price_comparison_data(
    ticker: str,
    years: int = DEFAULT_LOOKBACK_YEARS,
    benchmark: str = DEFAULT_BENCHMARK,
) -> dict:
    result_df = fetch_batch_cagrs([ticker], years=years, benchmark=benchmark)
    if result_df.empty:
        raise ValueError(f"No price comparison data fetched for {ticker}")

    row = result_df.iloc[0].to_dict()
    if "Error" in row:
        raise ValueError(f"Price comparison data unavailable for {ticker}: {row['Error']}")

    end_date = datetime.today()
    start_date = end_date - relativedelta(years=years)
    prices_raw = yf.download(
        [ticker, benchmark],
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False,
    )
    prices = prices_raw['Close'] if 'Close' in prices_raw else prices_raw
    aligned_prices = prices[[ticker, benchmark]].dropna()
    if len(aligned_prices) < 2:
        raise ValueError(f"Insufficient aligned price data for {ticker}")

    stock_returns = aligned_prices[ticker].pct_change().dropna()
    benchmark_returns = aligned_prices[benchmark].pct_change().dropna()
    active_returns = stock_returns - benchmark_returns

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "period_years": row["Period (Yrs)"],
        "stock_cagr": row["Stock CAGR"],
        "benchmark_cagr": row["Benchmark CAGR"],
        "tracking_error": float(active_returns.std() * (252 ** 0.5)) if not active_returns.empty else 0.0,
    }


def analyze_investment_philosophy(ticker: str, years: int = DEFAULT_LOOKBACK_YEARS, benchmark: str = DEFAULT_BENCHMARK) -> dict:
    result_df = fetch_batch_cagrs([ticker], years=years, benchmark=benchmark)
    if result_df.empty:
        return {"ticker": ticker, "error": "No data fetched"}

    row = result_df.iloc[0].to_dict()
    if "Error" in row:
        return {"ticker": ticker, "error": row["Error"]}

    return {
        "ticker": row["Ticker"],
        "period_years": row["Period (Yrs)"],
        "stock_cagr": row["Stock CAGR"],
        "benchmark_cagr": row["Benchmark CAGR"],
        "outperformed_benchmark": row["Outperformed?"] == "Yes"
    }



if __name__ == "__main__":
    tickers_to_test = ["AOS", "AAPL", "KO"]
    print("Analyzing 10-year performance against S&P 500 (^GSPC)...\n")
    
    try:
        df_results = analyze_investment_philosophy(tickers_to_test, years=10)
        
        # Clean console formatting using Pandas formatters
        format_dict = {"Stock CAGR": "{:.2%}", "Benchmark CAGR": "{:.2%}"}
        print(df_results.to_string(formatters=format_dict, index=False))
        
    except Exception as e:
        print(f"An error occurred: {e}")

class UndervaluedMarginOfSafety:
    """
    Heuristic: Undervalued & Margin of Safety
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        intrinsic_value: Optional[float] = None,
        market_price: Optional[float] = None,
        minimum_margin: float = UNDERVALUED_MARGIN_MIN,
        ticker: str = "",
    ) -> dict:
        if ticker and (intrinsic_value is None or market_price is None):
            intrinsic_result = IntrinsicValue().evaluate(ticker=ticker)
            intrinsic_value = intrinsic_result["intrinsic_value_per_share"]
            market_price = intrinsic_result["market_price"]

        if intrinsic_value is None or market_price is None:
            return {"applicable": False, "reason": "Missing required metrics: intrinsic_value and market_price are required"}

        result = MarginOfSafety().evaluate(intrinsic_value, market_price)
        result["minimum_margin"] = minimum_margin
        result["is_undervalued"] = result["margin_of_safety"] >= minimum_margin
        return result

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

@disk_cache()
def fetch_batch_cagrs(tickers: List[str], years: int = DEFAULT_LOOKBACK_YEARS, benchmark: str = DEFAULT_BENCHMARK) -> pd.DataFrame:
    """
    Fetches price history and calculates CAGR.
    Uses batch downloading for network efficiency and exact date alignment.
    """
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=years)

    all_symbols = tickers + [benchmark]
    df_raw = yf.download(
        all_symbols,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False,
    )

    if df_raw.empty:
        raise ValueError("No data fetched. Check your network or ticker symbols.")

    prices = df_raw['Close'] if 'Close' in df_raw else df_raw
    results = []

    for ticker in tickers:
        if ticker not in prices.columns:
            results.append({"Ticker": ticker, "Error": "Data not found"})
            continue

        aligned_data = prices[[ticker, benchmark]].dropna()
        if len(aligned_data) < 2:
            results.append({"Ticker": ticker, "Error": "Insufficient data"})
            continue

        actual_years = (aligned_data.index[-1] - aligned_data.index[0]).days / 365.25
        if actual_years <= 0:
            continue

        stock_cagr = _calculate_cagr(aligned_data[ticker].iloc[0], aligned_data[ticker].iloc[-1], actual_years)
        bench_cagr = _calculate_cagr(aligned_data[benchmark].iloc[0], aligned_data[benchmark].iloc[-1], actual_years)

        results.append(
            {
                "Ticker": ticker,
                "Period (Yrs)": round(actual_years, 2),
                "Stock CAGR": stock_cagr,
                "Benchmark CAGR": bench_cagr,
                "Outperformed?": "Yes" if stock_cagr > bench_cagr else "No",
            }
        )

    return pd.DataFrame(results)


class IntrinsicValue:
    """
    Heuristic: Intrinsic Value
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        fcf: Optional[float] = None,
        growth_rate: Optional[float] = None,
        discount_rate: Optional[float] = None,
        terminal_growth_rate: float = 0.02,
        shares_outstanding: Optional[int] = None,
        net_debt: Optional[float] = None,
        years: int = DEFAULT_INTRINSIC_VALUE_YEARS,
        ticker: str = "",
    ) -> dict:
        if ticker and any(value is None for value in (fcf, growth_rate, discount_rate, shares_outstanding, net_debt)):
            return IntrinsicValueEstimation().evaluate(ticker=ticker, terminal_growth_rate=terminal_growth_rate, years=years)

        if any(value is None for value in (fcf, growth_rate, discount_rate, shares_outstanding, net_debt)):
            return {"applicable": False, "reason": "Missing required metrics: fcf, growth_rate, discount_rate, shares_outstanding, and net_debt are required"}

        return IntrinsicValueEstimation().evaluate(
            fcf=fcf,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
            years=years,
        )

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
