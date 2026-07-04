import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List

from evaluator_config import DEFAULT_BENCHMARK, DEFAULT_INTRINSIC_VALUE_YEARS, DEFAULT_LOOKBACK_YEARS
from evaluator_thresholds import (
    EFFICIENT_MARKET_EXCESS_RETURN_MAX,
    EFFICIENT_MARKET_TRACKING_ERROR_MAX,
    FOCUS_INVESTING_TOP_THREE_WEIGHT_MIN,
    MARKET_FORECAST_USEFUL_ERROR_MAX,
    UNDERVALUED_MARGIN_MIN,
)
from valuation_capital import IntrinsicValueEstimation, MarginOfSafety


def analyze_investment_philosophy(ticker: str, years: int = DEFAULT_LOOKBACK_YEARS, benchmark: str = DEFAULT_BENCHMARK) -> dict:
    result_df = Compounding().evaluate([ticker], years=years, benchmark=benchmark)
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

class FocusInvesting:
    """
    Heuristic: Focus Investing
    """
    def __init__(self):
        pass

    def evaluate(self, positions: List[float]) -> dict:
        if not positions:
            raise ValueError("positions must not be empty")

        total = sum(positions)
        if total <= 0:
            raise ValueError("positions must sum to a positive number")

        weights = [position / total for position in positions]
        top_position_weight = max(weights)
        top_three_weight = sum(sorted(weights, reverse=True)[:3])

        return {
            "position_count": len(positions),
            "top_position_weight": top_position_weight,
            "top_three_weight": top_three_weight,
            "is_focus_investing": top_three_weight >= FOCUS_INVESTING_TOP_THREE_WEIGHT_MIN
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class EfficientMarketTheory:
    """
    Heuristic: Efficient Market Theory
    """
    def __init__(self):
        pass

    def evaluate(self, stock_cagr: float, benchmark_cagr: float, tracking_error: float) -> dict:
        excess_return = stock_cagr - benchmark_cagr
        market_efficiency_supported = abs(excess_return) <= EFFICIENT_MARKET_EXCESS_RETURN_MAX and tracking_error <= EFFICIENT_MARKET_TRACKING_ERROR_MAX

        return {
            "stock_cagr": stock_cagr,
            "benchmark_cagr": benchmark_cagr,
            "tracking_error": tracking_error,
            "excess_return": excess_return,
            "market_efficiency_supported": market_efficiency_supported
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class MarketForecasting:
    """
    Heuristic: Market Forecasting
    """
    def __init__(self):
        pass

    def evaluate(self, forecast_return: float, actual_return: float) -> dict:
        forecast_error = actual_return - forecast_return
        return {
            "forecast_return": forecast_return,
            "actual_return": actual_return,
            "forecast_error": forecast_error,
            "forecast_was_useful": abs(forecast_error) <= MARKET_FORECAST_USEFUL_ERROR_MAX
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class UndervaluedMarginOfSafety:
    """
    Heuristic: Undervalued & Margin of Safety
    """
    def __init__(self):
        pass

    def evaluate(self, intrinsic_value: float, market_price: float, minimum_margin: float = UNDERVALUED_MARGIN_MIN) -> dict:
        result = MarginOfSafety().evaluate(intrinsic_value, market_price)
        result["minimum_margin"] = minimum_margin
        result["is_undervalued"] = result["margin_of_safety"] >= minimum_margin
        return result

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class Compounding:
    """
    Heuristic: Compounding
    """
    def __init__(self):
        pass

    def evaluate(self, tickers: List[str], years: int = DEFAULT_LOOKBACK_YEARS, benchmark: str = DEFAULT_BENCHMARK) -> pd.DataFrame:
        """
        Fetches price history and calculates CAGR.
        Uses batch downloading for network efficiency and exact date alignment.
        """
        end_date = datetime.today()
        start_date = end_date - relativedelta(years=years)
        
        # 1. Combine all symbols for a single batch download
        all_symbols = tickers + [benchmark]
        
        # yf.download is significantly faster and inherently aligns dates
        df_raw = yf.download(
            all_symbols,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if df_raw.empty:
            raise ValueError("No data fetched. Check your network or ticker symbols.")
            
        # 2. Extract 'Close' prices (yf.download returns a MultiIndex DataFrame)
        prices = df_raw['Close'] if 'Close' in df_raw else df_raw
            
        results = []
        
        # 3. Process each ticker independently against the benchmark
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
                
            stock_cagr = self._calculate_cagr(aligned_data[ticker].iloc[0], aligned_data[ticker].iloc[-1], actual_years)
            bench_cagr = self._calculate_cagr(aligned_data[benchmark].iloc[0], aligned_data[benchmark].iloc[-1], actual_years)
            
            results.append({
                "Ticker": ticker,
                "Period (Yrs)": round(actual_years, 2),
                "Stock CAGR": stock_cagr,
                "Benchmark CAGR": bench_cagr,
                "Outperformed?": "Yes" if stock_cagr > bench_cagr else "No"
            })
            
        return pd.DataFrame(results)

    def _calculate_cagr(self, start_value: float, end_value: float, years: float) -> float:
        """Calculates the Compound Annual Growth Rate."""
        if start_value <= 0 or years <= 0:
            return 0.0
        return (end_value / start_value) ** (1 / years) - 1

class IntrinsicValue:
    """
    Heuristic: Intrinsic Value
    """
    def __init__(self):
        pass

    def evaluate(self, fcf: float, growth_rate: float, discount_rate: float, terminal_growth_rate: float, shares_outstanding: int, net_debt: float, years: int = DEFAULT_INTRINSIC_VALUE_YEARS) -> dict:
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
