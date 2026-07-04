import json
import requests
import yfinance as yf
from typing import Dict, Any, List
from evaluator_config import DEFAULT_INTRINSIC_VALUE_YEARS, DEFAULT_OLLAMA_MODEL, OLLAMA_GENERATE_URL, RISK_FREE_RATE_FALLBACK
from evaluator_thresholds import (
    CAPITAL_ALLOCATION_STRONG_FCF_TO_DEBT_MIN,
    DEEP_DISCOUNT_MARGIN_MIN,
    DISCOUNT_MARGIN_MIN,
    SPECIAL_INSTRUMENT_COLLATERAL_COVERAGE_MIN,
    SPECIAL_INSTRUMENT_CONVERSION_DISCOUNT_MIN,
    SPECIAL_INSTRUMENT_COUPON_RATE_MIN,
)


def normalize_buyback_analysis(result: Dict[str, Any] | None) -> Dict[str, Any]:
    result = result or {}

    buyback_strategy = result.get("buyback_strategy")
    if buyback_strategy is None and "systematic_buybacks" in result:
        buyback_strategy = "Systematic" if result.get("systematic_buybacks") else "Opportunistic/Value-Based"

    mentions_intrinsic_value = result.get("mentions_intrinsic_value")
    if mentions_intrinsic_value is None:
        mentions_intrinsic_value = bool(result.get("intrinsic_value_mentioned", False))

    analysis_summary = result.get("analysis_summary") or result.get("analysis") or ""

    return {
        "buyback_strategy": buyback_strategy or "Unknown",
        "mentions_intrinsic_value": mentions_intrinsic_value,
        "analysis_summary": analysis_summary,
    }

# --- API FETCHING (using yfinance) ---

def fetch_risk_free_rate() -> float:
    """
    Fetches the current Risk-Free Rate (10-Year Treasury Yield) using ^TNX.
    """
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if not hist.empty:
            # ^TNX is quoted in percentage (e.g., 4.2 for 4.2%)
            return hist['Close'].iloc[-1] / 100.0
    except Exception as e:
        print(f"Error fetching ^TNX: {e}")
    return RISK_FREE_RATE_FALLBACK

def fetch_financial_data(ticker_symbol: str) -> Dict[str, Any]:
    """
    Fetches shares outstanding, historical cash flows, and balance sheet items via yfinance.
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    
    shares_outstanding = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding", 100_000_000)
    
    cashflow = ticker.cashflow
    fcf = 0.0
    fcf_growth = 0.08
    
    if not cashflow.empty:
        if "Free Cash Flow" in cashflow.index:
            fcfs = cashflow.loc["Free Cash Flow"].dropna()
        else:
            op_cf = cashflow.loc["Operating Cash Flow"] if "Operating Cash Flow" in cashflow.index else cashflow.loc.get("Total Cash From Operating Activities", 0)
            capex = cashflow.loc["Capital Expenditure"] if "Capital Expenditure" in cashflow.index else 0
            fcfs = op_cf + capex  # capex is typically negative
            
        if isinstance(fcfs, float) or isinstance(fcfs, int):
            fcf = float(fcfs)
        elif not fcfs.empty:
            recent_fcf = fcfs.iloc[0]
            if len(fcfs) >= 2:
                prev_fcf = fcfs.iloc[1]
                if prev_fcf > 0 and recent_fcf > 0:
                    fcf_growth = (recent_fcf / prev_fcf) - 1
            fcf = recent_fcf

    if fcf <= 0:
        fcf = info.get("freeCashflow", 500_000_000)
        
    cash = info.get("totalCash", 1_000_000_000)
    debt = info.get("totalDebt", 2_000_000_000)
    
    return {
        "ticker": ticker_symbol,
        "shares_outstanding": shares_outstanding,
        "recent_free_cash_flow": fcf,
        "historical_fcf_growth_rate": fcf_growth,
        "cash_and_equivalents": cash,
        "total_debt": debt
    }

def fetch_management_commentary(ticker_symbol: str) -> str:
    """
    Fetches recent management commentary (using longBusinessSummary as a fallback).
    """
    ticker = yf.Ticker(ticker_symbol)
    summary = ticker.info.get("longBusinessSummary", "")
    if summary:
        return summary
    return "No management commentary or business summary available."


# --- DCF VALUATION ---

def calculate_dcf(
    fcf: float, 
    growth_rate: float, 
    discount_rate: float, 
    terminal_growth_rate: float, 
    shares_outstanding: int, 
    net_debt: float, 
    years: int = 10
) -> Dict[str, float]:
    """
    Calculates Intrinsic Value using a pure Python Discounted Cash Flow (DCF) model.
    """
    projected_fcfs = []
    current_fcf = fcf
    
    # Project FCFs
    for year in range(1, years + 1):
        current_fcf *= (1 + growth_rate)
        projected_fcfs.append(current_fcf)
        
    # Discount projected FCFs
    discounted_fcfs = [
        cf / ((1 + discount_rate) ** idx) 
        for idx, cf in enumerate(projected_fcfs, start=1)
    ]
    pv_of_fcf = sum(discounted_fcfs)
    
    # Calculate Terminal Value (Gordon Growth Model)
    final_year_fcf = projected_fcfs[-1]
    terminal_value = (final_year_fcf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    pv_of_tv = terminal_value / ((1 + discount_rate) ** years)
    
    # Enterprise Value to Equity Value
    enterprise_value = pv_of_fcf + pv_of_tv
    equity_value = enterprise_value - net_debt
    
    # Value per share
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
    
    return {
        "pv_of_fcf": pv_of_fcf,
        "pv_of_terminal_value": pv_of_tv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": value_per_share
    }


# --- OLLAMA INTEGRATION ---

def analyze_buyback_commentary(ticker: str, commentary: str) -> Dict[str, Any]:
    """
    Uses local Ollama LLM to analyze management's approach to share repurchases.
    """
    prompt = f"""
    You are an expert value investor analyzing management's capital allocation strategy for {ticker}.
    
    Read the following management commentary or business summary on share repurchases:
    "{commentary}"
    
    Answer the following questions:
    1. Is management buying back stock systematically regardless of price, or only when they believe it is cheap?
    2. Do they explicitly mention "Intrinsic Value" (or similar concepts) as a benchmark for repurchases?
    
    Return your analysis strictly as a JSON object with the following keys:
    - "buyback_strategy": A short string describing the strategy (e.g., "Systematic", "Opportunistic/Value-Based").
    - "mentions_intrinsic_value": A boolean (true or false).
    - "analysis_summary": A one-sentence explanation.
    """
    
    url = OLLAMA_GENERATE_URL
    payload = {
        "model": DEFAULT_OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Parse the JSON string returned by Ollama
        result = json.loads(data.get("response", "{}"))
        return normalize_buyback_analysis(result)
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return normalize_buyback_analysis({
            "buyback_strategy": "Unknown",
            "mentions_intrinsic_value": False,
            "analysis_summary": f"Failed to analyze: {str(e)}"
        })


# --- MAIN EXECUTION ---

def process_company(ticker: str) -> Dict[str, Any]:
    print(f"Processing Valuation & Capital Allocation for {ticker}...")
    
    # 1. Fetch Quantitative Data
    print("Fetching risk-free rate...")
    risk_free_rate = fetch_risk_free_rate()
    print(f"Risk-free rate: {risk_free_rate:.2%}")
    
    print("Fetching financial data...")
    financials = fetch_financial_data(ticker)
    
    # 2. Perform DCF Calculation
    # Buffett typically uses a flat discount rate (e.g., 9-10%) or Risk-Free Rate + Equity Risk Premium
    discount_rate = max(risk_free_rate + 0.05, 0.09) # Minimum 9% hurdle rate
    net_debt = financials["total_debt"] - financials["cash_and_equivalents"]
    
    dcf_results = calculate_dcf(
        fcf=financials["recent_free_cash_flow"],
        growth_rate=financials["historical_fcf_growth_rate"],
        discount_rate=discount_rate,
        terminal_growth_rate=0.02, # 2% long-term growth assumption
        shares_outstanding=financials["shares_outstanding"],
        net_debt=net_debt,
        years=DEFAULT_INTRINSIC_VALUE_YEARS
    )
    
    print(f"\n--- DCF Valuation for {ticker} ---")
    print(f"Intrinsic Value per Share: ${dcf_results['intrinsic_value_per_share']:.2f}")
    
    # 3. Fetch Qualitative Data (Commentary)
    print("\nFetching management commentary...")
    commentary = fetch_management_commentary(ticker)
    
    # 4. Analyze Commentary with Ollama
    print("\nAnalyzing Management Commentary with Ollama...")
    analysis = analyze_buyback_commentary(ticker, commentary)
    
    # 5. Combine and Output
    final_output = {
        "ticker": ticker,
        "valuation": dcf_results,
        "capital_allocation_analysis": analysis
    }
    
    print("\n--- Final Output ---")
    print(json.dumps(final_output, indent=2))
    
    return final_output


if __name__ == "__main__":
    # Example usage:
    # process_company("AAPL")
    pass


class IntrinsicValueEstimation:
    """
    Heuristic: Intrinsic Value Estimation
    """
    def __init__(self):
        pass

    def evaluate(self, fcf: float, growth_rate: float, discount_rate: float, terminal_growth_rate: float, shares_outstanding: int, net_debt: float, years: int = DEFAULT_INTRINSIC_VALUE_YEARS) -> dict:
        projected_fcfs = []
        current_fcf = fcf
        
        for year in range(1, years + 1):
            current_fcf *= (1 + growth_rate)
            projected_fcfs.append(current_fcf)
            
        discounted_fcfs = [cf / ((1 + discount_rate) ** idx) for idx, cf in enumerate(projected_fcfs, start=1)]
        pv_of_fcf = sum(discounted_fcfs)
        
        final_year_fcf = projected_fcfs[-1]
        terminal_value = (final_year_fcf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
        pv_of_tv = terminal_value / ((1 + discount_rate) ** years)
        
        enterprise_value = pv_of_fcf + pv_of_tv
        equity_value = enterprise_value - net_debt
        value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
        
        return {
            "pv_of_fcf": pv_of_fcf,
            "pv_of_terminal_value": pv_of_tv,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "intrinsic_value_per_share": value_per_share
        }

    def _helper_method(self):
        pass

class MarginOfSafety:
    """
    Heuristic: Margin of Safety
    """
    def __init__(self):
        pass

    def evaluate(self, intrinsic_value: float, market_price: float) -> dict:
        if intrinsic_value <= 0:
            raise ValueError("intrinsic_value must be positive")

        margin = (intrinsic_value - market_price) / intrinsic_value
        return {
            "intrinsic_value": intrinsic_value,
            "market_price": market_price,
            "margin_of_safety": margin,
            "is_discount": market_price < intrinsic_value
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class TheRelationshipBetweenPurchasePriceAndIntrinsicValue:
    """
    Heuristic: The Relationship Between Purchase Price and Intrinsic Value
    """
    def __init__(self):
        pass

    def evaluate(self, intrinsic_value: float, market_price: float) -> dict:
        margin = MarginOfSafety().evaluate(intrinsic_value, market_price)["margin_of_safety"]

        verdict = "fairly_priced"
        if margin >= DEEP_DISCOUNT_MARGIN_MIN:
            verdict = "deep_discount"
        elif margin >= DISCOUNT_MARGIN_MIN:
            verdict = "discount"
        elif margin < 0:
            verdict = "premium"

        return {
            "intrinsic_value": intrinsic_value,
            "market_price": market_price,
            "margin_of_safety": margin,
            "purchase_price_verdict": verdict
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class CapitalAllocationAnalysis:
    """
    Heuristic: Capital Allocation Analysis
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        recent_free_cash_flow: float,
        total_debt: float,
        cash_and_equivalents: float,
        commentary: str = "",
    ) -> dict:
        net_cash = cash_and_equivalents - total_debt
        balance_sheet = "net_cash" if net_cash >= 0 else "net_debt"
        fcf_to_debt = recent_free_cash_flow / total_debt if total_debt > 0 else None

        buyback_analysis = ShareBuybackAnalysis().evaluate("N/A", commentary) if commentary else {
            "buyback_strategy": "Unknown",
            "mentions_intrinsic_value": False,
            "analysis_summary": "No commentary provided."
        }

        discipline = "moderate"
        if recent_free_cash_flow > 0 and (fcf_to_debt is None or fcf_to_debt >= CAPITAL_ALLOCATION_STRONG_FCF_TO_DEBT_MIN):
            discipline = "strong"
        elif recent_free_cash_flow <= 0:
            discipline = "weak"

        return {
            "recent_free_cash_flow": recent_free_cash_flow,
            "net_cash": net_cash,
            "balance_sheet_position": balance_sheet,
            "fcf_to_debt": fcf_to_debt,
            "capital_allocation_discipline": discipline,
            "buyback_analysis": buyback_analysis
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class ShareBuybackAnalysis:
    """
    Heuristic: Share Buyback Analysis
    """
    def __init__(self):
        pass

    def evaluate(self, ticker: str, commentary: str) -> dict:
        prompt = f"""
        You are an expert value investor analyzing management's capital allocation strategy for {ticker}.
        Read the following management commentary or business summary on share repurchases: "{commentary}"
        
        Answer the following questions:
        1. Is management buying back stock systematically regardless of price, or only when they believe it is cheap?
        2. Do they explicitly mention "Intrinsic Value" (or similar concepts) as a benchmark for repurchases?
        
        Return your analysis strictly as a JSON object with the following keys:
        - "buyback_strategy": A short string describing the strategy (e.g., "Systematic", "Opportunistic/Value-Based").
        - "mentions_intrinsic_value": A boolean (true or false).
        - "analysis_summary": A one-sentence explanation.
        """
        import requests
        import json
        url = OLLAMA_GENERATE_URL
        payload = {"model": DEFAULT_OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False}
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return normalize_buyback_analysis(json.loads(data.get("response", "{}")))
        except Exception as e:
            return normalize_buyback_analysis({"buyback_strategy": "Unknown", "mentions_intrinsic_value": False, "analysis_summary": f"Failed: {str(e)}"})

    def _helper_method(self):
        pass

class DividendsRetainedEarningsAndTaxEfficiency:
    """
    Heuristic: Dividends, Retained Earnings, and Tax Efficiency
    """
    def __init__(self):
        pass

    def evaluate(self, dividend_payout_ratio: float, retained_return_on_equity: float, tax_rate_on_dividends: float) -> dict:
        retained_earnings_ratio = 1 - dividend_payout_ratio
        after_tax_dividend_value = dividend_payout_ratio * (1 - tax_rate_on_dividends)
        retained_value_creation = retained_earnings_ratio * retained_return_on_equity

        preference = "retain"
        if after_tax_dividend_value >= retained_value_creation:
            preference = "distribute"

        return {
            "dividend_payout_ratio": dividend_payout_ratio,
            "retained_earnings_ratio": retained_earnings_ratio,
            "after_tax_dividend_value": after_tax_dividend_value,
            "retained_value_creation": retained_value_creation,
            "capital_return_preference": preference
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class SpecialInvestmentInstruments:
    """
    Heuristic: Special Investment Instruments
    """
    def __init__(self):
        pass

    def evaluate(self, coupon_rate: float, conversion_discount: float, collateral_coverage: float) -> dict:
        score = 0
        if coupon_rate >= SPECIAL_INSTRUMENT_COUPON_RATE_MIN:
            score += 1
        if conversion_discount >= SPECIAL_INSTRUMENT_CONVERSION_DISCOUNT_MIN:
            score += 1
        if collateral_coverage >= SPECIAL_INSTRUMENT_COLLATERAL_COVERAGE_MIN:
            score += 1

        attractiveness = "low"
        if score == 3:
            attractiveness = "high"
        elif score == 2:
            attractiveness = "moderate"

        return {
            "coupon_rate": coupon_rate,
            "conversion_discount": conversion_discount,
            "collateral_coverage": collateral_coverage,
            "instrument_score": score,
            "instrument_attractiveness": attractiveness
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
