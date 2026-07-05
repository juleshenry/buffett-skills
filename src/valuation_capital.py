import json
import re
import requests
import yfinance as yf
from typing import Dict, Any, List, Optional
from evaluator_config import DEFAULT_INTRINSIC_VALUE_YEARS, DEFAULT_OLLAMA_MODEL, OLLAMA_GENERATE_URL, RISK_FREE_RATE_FALLBACK, call_ollama_panel_json
from sec_data import fetch_filing_keyword_context, fetch_filing_section
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


def fetch_current_market_price(ticker_symbol: str) -> float:
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info or {}
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if current_price:
        return float(current_price)

    history = ticker.history(period="5d")
    if not history.empty and "Close" in history:
        return float(history["Close"].dropna().iloc[-1])

    raise ValueError(f"No current market price available for {ticker_symbol}")

def fetch_management_commentary(ticker_symbol: str) -> str:
    """
    Fetches real SEC management commentary relevant to repurchases or capital allocation.
    """
    keyword_sets = (
        ("10-K", ("share repurchase", "share repurchases", "stock repurchase", "repurchase program", "capital allocation", "returned to shareholders", "intrinsic value")),
        ("10-Q", ("share repurchase", "share repurchases", "stock repurchase", "repurchase program", "capital allocation", "returned to shareholders", "intrinsic value")),
        ("8-K", ("share repurchase", "earnings release", "results of operations and financial condition", "capital allocation")),
    )

    for form, keywords in keyword_sets:
        try:
            commentary = fetch_filing_keyword_context(
                ticker_symbol,
                form=form,
                keywords=keywords,
                context_chars=1600,
                max_matches=3,
                max_chars=8000,
            )
            if commentary:
                return commentary
        except Exception:
            continue

    try:
        return fetch_filing_section(
            ticker_symbol,
            form="10-K",
            start_markers=("item 7.", "management's discussion and analysis", "management s discussion and analysis"),
            end_markers=("item 7a.", "item 8."),
            max_chars=12000,
        )
    except Exception:
        return "No SEC management commentary available."


def fetch_special_instrument_commentary(ticker_symbol: str) -> str:
    keyword_sets = (
        (
            "8-K",
            (
                "convertible notes",
                "convertible preferred",
                "preferred stock",
                "warrant",
                "financing",
                "collateral",
                "secured",
            ),
        ),
        (
            "10-Q",
            (
                "convertible notes",
                "convertible preferred",
                "preferred stock",
                "warrant",
                "collateral",
                "secured debt",
            ),
        ),
        (
            "10-K",
            (
                "convertible notes",
                "convertible preferred",
                "preferred stock",
                "warrant",
                "collateral",
                "secured debt",
            ),
        ),
    )

    for form, keywords in keyword_sets:
        try:
            commentary = fetch_filing_keyword_context(
                ticker_symbol,
                form=form,
                keywords=keywords,
                context_chars=1800,
                max_matches=4,
                max_chars=12000,
            )
            if commentary:
                return commentary
        except Exception:
            continue

    return ""


def _parse_percentage_value(text: str, patterns: list[str]) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
    return None


def _parse_special_instrument_metrics(commentary: str) -> dict[str, Any]:
    normalized = commentary.lower()
    has_special_instrument = any(
        token in normalized
        for token in (
            "convertible note",
            "convertible notes",
            "convertible preferred",
            "preferred stock",
            "warrant",
        )
    )

    coupon_rate = _parse_percentage_value(
        commentary,
        [
            r"coupon\s+rate[^\d]{0,20}(\d+(?:\.\d+)?)\s*%",
            r"interest\s+rate[^\d]{0,20}(\d+(?:\.\d+)?)\s*%",
            r"bearing\s+interest\s+at\s+(\d+(?:\.\d+)?)\s*%",
            r"dividend\s+rate[^\d]{0,20}(\d+(?:\.\d+)?)\s*%",
        ],
    )
    conversion_discount = _parse_percentage_value(
        commentary,
        [
            r"conversion\s+discount[^\d]{0,20}(\d+(?:\.\d+)?)\s*%",
            r"discount\s+to\s+(?:the\s+)?conversion\s+price[^\d]{0,20}(\d+(?:\.\d+)?)\s*%",
        ],
    )

    collateral_coverage = None
    if re.search(r"\b(secured|collateral|first lien|asset-backed)\b", commentary, flags=re.IGNORECASE):
        collateral_coverage = 1.0
    elif has_special_instrument:
        collateral_coverage = 0.0

    return {
        "has_special_instrument": has_special_instrument,
        "coupon_rate": coupon_rate,
        "conversion_discount": conversion_discount,
        "collateral_coverage": collateral_coverage,
    }


# --- DCF VALUATION ---

def calculate_dcf(
    fcf: float, 
    growth_rate: float, 
    discount_rate: float, 
    terminal_growth_rate: float, 
    shares_outstanding: int, 
    net_debt: float, 
    years: int = DEFAULT_INTRINSIC_VALUE_YEARS
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
    
    try:
        return normalize_buyback_analysis(call_ollama_panel_json(prompt, model=DEFAULT_OLLAMA_MODEL))
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
        current_market_price = None
        if ticker and any(value is None for value in (fcf, growth_rate, discount_rate, shares_outstanding, net_debt)):
            financials = fetch_financial_data(ticker)
            risk_free_rate = fetch_risk_free_rate()
            fcf = float(financials["recent_free_cash_flow"])
            growth_rate = float(financials["historical_fcf_growth_rate"])
            discount_rate = max(risk_free_rate + 0.05, 0.09)
            shares_outstanding = int(financials["shares_outstanding"])
            net_debt = float(financials["total_debt"] - financials["cash_and_equivalents"])
            current_market_price = fetch_current_market_price(ticker)

        if any(value is None for value in (fcf, growth_rate, discount_rate, shares_outstanding, net_debt)):
            return {"applicable": False, "reason": "Missing required metrics: fcf, growth_rate, discount_rate, shares_outstanding, and net_debt are required"}

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
        
        result = {
            "pv_of_fcf": pv_of_fcf,
            "pv_of_terminal_value": pv_of_tv,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "intrinsic_value_per_share": value_per_share
        }
        if ticker:
            result["ticker"] = ticker
        if current_market_price is not None:
            result["market_price"] = current_market_price
        return result

    def _helper_method(self):
        pass

class MarginOfSafety:
    """
    Heuristic: Margin of Safety
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        intrinsic_value: Optional[float] = None,
        market_price: Optional[float] = None,
        ticker: str = "",
    ) -> dict:
        if ticker and (intrinsic_value is None or market_price is None):
            intrinsic_result = IntrinsicValueEstimation().evaluate(ticker=ticker)
            intrinsic_value = intrinsic_result["intrinsic_value_per_share"]
            market_price = intrinsic_result.get("market_price") or fetch_current_market_price(ticker)

        if intrinsic_value is None or market_price is None:
            return {"applicable": False, "reason": "Missing required metrics: intrinsic_value and market_price are required"}

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

    def evaluate(
        self,
        intrinsic_value: Optional[float] = None,
        market_price: Optional[float] = None,
        ticker: str = "",
    ) -> dict:
        if ticker and (intrinsic_value is None or market_price is None):
            mos = MarginOfSafety().evaluate(ticker=ticker)
            intrinsic_value = mos["intrinsic_value"]
            market_price = mos["market_price"]

        if intrinsic_value is None or market_price is None:
            return {"applicable": False, "reason": "Missing required metrics: intrinsic_value and market_price are required"}

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
        recent_free_cash_flow: Optional[float] = None,
        total_debt: Optional[float] = None,
        cash_and_equivalents: Optional[float] = None,
        ticker: str = "",
        commentary: str = "",
    ) -> dict:
        if ticker and any(value is None for value in (recent_free_cash_flow, total_debt, cash_and_equivalents)):
            financials = fetch_financial_data(ticker)
            recent_free_cash_flow = float(financials["recent_free_cash_flow"])
            total_debt = float(financials["total_debt"])
            cash_and_equivalents = float(financials["cash_and_equivalents"])

        if recent_free_cash_flow is None or total_debt is None or cash_and_equivalents is None:
            return {"applicable": False, "reason": "Missing required metrics: recent_free_cash_flow, total_debt, and cash_and_equivalents are required"}

        net_cash = cash_and_equivalents - total_debt
        balance_sheet = "net_cash" if net_cash >= 0 else "net_debt"
        fcf_to_debt = recent_free_cash_flow / total_debt if total_debt > 0 else None

        buyback_analysis = {
            "buyback_strategy": "Unknown",
            "mentions_intrinsic_value": False,
            "analysis_summary": "No commentary provided."
        }
        if commentary:
            buyback_analysis = ShareBuybackAnalysis().evaluate(ticker or "N/A", commentary)
        elif ticker:
            buyback_analysis = ShareBuybackAnalysis().evaluate(ticker)

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

    def evaluate(self, ticker: str, commentary: str = "") -> dict:
        if not commentary:
            from sec_data import fetch_filing_section
            try:
                commentary = fetch_filing_section(
                    ticker,
                    form="10-K",
                    start_markers=("item 7.", "management's discussion"),
                    end_markers=("item 8.", "financial statements"),
                    max_chars=15000
                )
            except Exception as e:
                return normalize_buyback_analysis({"buyback_strategy": "Unknown", "mentions_intrinsic_value": False, "analysis_summary": f"Failed to fetch MD&A: {e}"})

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
        try:
            return normalize_buyback_analysis(call_ollama_panel_json(prompt, model=DEFAULT_OLLAMA_MODEL))
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

    def evaluate(self, dividend_payout_ratio: float | None = None, retained_return_on_equity: float | None = None, tax_rate_on_dividends: float = 0.15, ticker: str = "") -> dict:
        if ticker and (dividend_payout_ratio is None or retained_return_on_equity is None):
            import yfinance as yf
            info = yf.Ticker(ticker).info
            dividend_payout_ratio = dividend_payout_ratio if dividend_payout_ratio is not None else info.get("payoutRatio", 0.0)
            retained_return_on_equity = retained_return_on_equity if retained_return_on_equity is not None else info.get("returnOnEquity", 0.0)
            
        if dividend_payout_ratio is None or retained_return_on_equity is None:
            raise ValueError("All metrics must be provided or fetchable via ticker")

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

    def evaluate(self, coupon_rate: float | None = None, conversion_discount: float | None = None, collateral_coverage: float | None = None, ticker: str = "") -> dict:
        if ticker and (coupon_rate is None or conversion_discount is None or collateral_coverage is None):
            instrument_metrics = self._fetch_special_instrument_metrics(ticker)
            if not instrument_metrics["has_special_instrument"]:
                return {
                    "ticker": ticker,
                    "applicable": False,
                    "reason": "No special investment instrument evidence found in recent filings.",
                }
            if coupon_rate is None:
                coupon_rate = instrument_metrics["coupon_rate"]
            if conversion_discount is None:
                conversion_discount = instrument_metrics["conversion_discount"]
            if collateral_coverage is None:
                collateral_coverage = instrument_metrics["collateral_coverage"]
            
        if coupon_rate is None or conversion_discount is None or collateral_coverage is None:
            raise ValueError("All metrics must be provided")

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

    def _fetch_special_instrument_metrics(self, ticker: str) -> dict[str, Any]:
        commentary = fetch_special_instrument_commentary(ticker)
        if not commentary:
            return {
                "has_special_instrument": False,
                "coupon_rate": None,
                "conversion_discount": None,
                "collateral_coverage": None,
            }
        return _parse_special_instrument_metrics(commentary)

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
