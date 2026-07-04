import json
import requests
import yfinance as yf
from evaluator_config import DEFAULT_OLLAMA_MODEL, OLLAMA_GENERATE_URL
from sec_data import fetch_filing_section


def normalize_capex_breakdown(result: dict | None) -> dict:
    result = result or {}
    maintenance_percentage = result.get("maintenance_percentage")
    if maintenance_percentage is None:
        maintenance_percentage = result.get("maintenance_capex_percentage")

    growth_percentage = result.get("growth_percentage")
    if growth_percentage is None:
        growth_percentage = result.get("growth_capex_percentage")

    return {
        "maintenance_percentage": maintenance_percentage,
        "growth_percentage": growth_percentage,
        "maintenance_capex_amount": result.get("maintenance_capex_amount"),
        "reasoning": result.get("reasoning") or result.get("analysis") or "",
    }

def fetch_deep_financials(ticker: str) -> dict:
    """Fetch deep financials using yfinance."""
    print(f"Fetching deep financials for {ticker} using yfinance...")
    stock = yf.Ticker(ticker)
    
    # Try to get cashflow and income statement
    try:
        cashflow = stock.cashflow
        income_stmt = stock.income_stmt
        
        # yfinance returns DataFrames where columns are dates. Get the most recent one.
        recent_cf = cashflow.iloc[:, 0] if not cashflow.empty else {}
        recent_is = income_stmt.iloc[:, 0] if not income_stmt.empty else {}
        
        # Attempt to extract some relevant metrics safely
        capex = recent_cf.get("Capital Expenditure")
        if capex is None:
            capex = recent_cf.get("CapitalExpenditure", None)
            
        ocf = recent_cf.get("Operating Cash Flow", None)
        
        return {
            "capex_total": capex,
            "operating_cash_flow": ocf,
            "net_income": recent_is.get("Net Income", None),
            "total_revenue": recent_is.get("Total Revenue", None)
        }
    except Exception as e:
        print(f"Error fetching financials for {ticker}: {e}")
        return {}

def fetch_mda_section(ticker: str) -> str:
    """Attempt fetching MD&A section from SEC EDGAR."""
    print(f"Attempting to fetch MD&A section for {ticker} from SEC EDGAR...")

    try:
        section = fetch_filing_section(
            ticker,
            form="10-K",
            start_markers=("item 7.", "management's discussion and analysis", "management s discussion and analysis"),
            end_markers=("item 7a.", "item 8."),
            max_chars=25000,
        )
        if section:
            return section
        raise ValueError(f"MD&A section not found for {ticker}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch from SEC EDGAR: {e}")
        raise RuntimeError(f"Failed to fetch MD&A section for {ticker}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to fetch MD&A section for {ticker}: {e}") from e

def query_ollama_capex_breakdown(mda_text: str) -> dict:
    """Query local Ollama to estimate maintenance vs growth capex from MD&A text."""
    prompt = f"""
    Read this MD&A and estimate what percentage of Capital Expenditures (Capex) was spent on 'Maintenance' (upkeeping current operations) versus 'Growth' (expanding operations). 
    Extract any specific numbers mentioned regarding maintenance capex.
    
    Return ONLY a valid JSON object with the following keys:
    - maintenance_percentage (float, 0-100)
    - growth_percentage (float, 0-100)
    - maintenance_capex_amount (string, specific amount mentioned, or null if none)
    - reasoning (string, brief justification)

    MD&A Text:
    {mda_text}
    """
    
    url = OLLAMA_GENERATE_URL
    data = {
        "model": DEFAULT_OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    print("Querying Ollama for Capex breakdown...")
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result_text = response.json().get("response", "{}")
        return normalize_capex_breakdown(json.loads(result_text))
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return {}

def main():
    tickers = ["AAPL", "MSFT"]
    
    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")
        financials = fetch_deep_financials(ticker)
        print("Financials:", financials)
        
        mda_text = fetch_mda_section(ticker)
        
        capex_breakdown = query_ollama_capex_breakdown(mda_text)
        
        print("\nExtracted Capex Data:")
        print(json.dumps(capex_breakdown, indent=2))
        print("---------------------------\n")

if __name__ == "__main__":
    main()


class CorePrincipleSeeThroughAccountingToEconomicReality:
    """
    Heuristic: Core Principle: See Through Accounting to Economic Reality
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        net_income: float | None = None,
        operating_cash_flow: float | None = None,
        capex_total: float | None = None,
        ticker: str = "",
    ) -> dict:
        if ticker and (net_income is None or operating_cash_flow is None or capex_total is None):
            financials = fetch_deep_financials(ticker)
            net_income = financials.get("net_income")
            operating_cash_flow = financials.get("operating_cash_flow")
            capex_total = financials.get("capex_total")

        if net_income is None or operating_cash_flow is None or capex_total is None:
            raise ValueError("net_income, operating_cash_flow, and capex_total are required")

        free_cash_flow = (operating_cash_flow or 0) - abs(capex_total or 0)
        accrual_gap = (net_income or 0) - (operating_cash_flow or 0)

        assessment = "cash_backed"
        if free_cash_flow < 0 or accrual_gap > 0:
            assessment = "accounting_risk"

        return {
            "net_income": net_income,
            "operating_cash_flow": operating_cash_flow,
            "capex_total": capex_total,
            "free_cash_flow": free_cash_flow,
            "accrual_gap": accrual_gap,
            "economic_reality_assessment": assessment
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class OwnerEarnings:
    """
    Heuristic: Owner Earnings
    """
    def __init__(self):
        pass

    def evaluate(self, ticker: str) -> dict:
        financials = self._fetch_deep_financials(ticker)
        mda_text = self._fetch_mda_section(ticker)
        capex_breakdown = self._query_ollama_capex_breakdown(mda_text)
        
        net_income = financials.get("net_income", 0)
        depreciation = financials.get("operating_cash_flow", 0) - net_income if financials.get("operating_cash_flow") and net_income else 0
        total_capex = abs(financials.get("capex_total", 0)) if financials.get("capex_total") else 0
        
        maintenance_pct = capex_breakdown.get("maintenance_percentage", 100) / 100.0
        maintenance_capex = total_capex * maintenance_pct
        
        owner_earnings = net_income + depreciation - maintenance_capex
        
        return {
            "net_income": net_income,
            "depreciation_amortization_estimate": depreciation,
            "total_capex": total_capex,
            "maintenance_capex_estimate": maintenance_capex,
            "owner_earnings": owner_earnings
        }

    def _fetch_deep_financials(self, ticker: str) -> dict:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        try:
            cashflow = stock.cashflow
            income_stmt = stock.income_stmt
            recent_cf = cashflow.iloc[:, 0] if not cashflow.empty else {}
            recent_is = income_stmt.iloc[:, 0] if not income_stmt.empty else {}
            capex = recent_cf.get("Capital Expenditure", recent_cf.get("CapitalExpenditure"))
            ocf = recent_cf.get("Operating Cash Flow")
            return {
                "capex_total": capex,
                "operating_cash_flow": ocf,
                "net_income": recent_is.get("Net Income"),
                "total_revenue": recent_is.get("Total Revenue")
            }
        except Exception:
            return {}

    def _fetch_mda_section(self, ticker: str) -> str:
        return fetch_mda_section(ticker)

    def _query_ollama_capex_breakdown(self, mda_text: str) -> dict:
        import requests, json
        prompt = f"""Read this MD&A and estimate what percentage of Capital Expenditures (Capex) was spent on 'Maintenance' vs 'Growth'. Return ONLY JSON: {{"maintenance_percentage": float, "growth_percentage": float, "maintenance_capex_amount": string, "reasoning": string}}. MD&A Text: {mda_text}"""
        try:
            response = requests.post(OLLAMA_GENERATE_URL, json={"model": DEFAULT_OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}, timeout=60)
            return normalize_capex_breakdown(json.loads(response.json().get("response", "{}")))
        except Exception:
            return {}


class KeyFinancialMetrics:
    """
    Heuristic: Key Financial Metrics
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        total_revenue: float | None = None,
        net_income: float | None = None,
        operating_cash_flow: float | None = None,
        capex_total: float | None = None,
        ticker: str = "",
    ) -> dict:
        if ticker and (total_revenue is None or net_income is None or operating_cash_flow is None or capex_total is None):
            financials = fetch_deep_financials(ticker)
            total_revenue = financials.get("total_revenue")
            net_income = financials.get("net_income")
            operating_cash_flow = financials.get("operating_cash_flow")
            capex_total = financials.get("capex_total")

        if total_revenue is None or net_income is None or operating_cash_flow is None or capex_total is None:
            raise ValueError("total_revenue, net_income, operating_cash_flow, and capex_total are required")

        profit_margin = (net_income / total_revenue) if total_revenue else None
        cash_conversion = (operating_cash_flow / net_income) if net_income else None
        free_cash_flow = (operating_cash_flow or 0) - abs(capex_total or 0)

        return {
            "total_revenue": total_revenue,
            "net_income": net_income,
            "operating_cash_flow": operating_cash_flow,
            "capex_total": capex_total,
            "profit_margin": profit_margin,
            "cash_conversion": cash_conversion,
            "free_cash_flow": free_cash_flow
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class LookthroughEarnings:
    """
    Heuristic: Look-Through Earnings
    """
    def __init__(self):
        pass

    def evaluate(self, ownership_percentage: float, investee_net_income: float, dividends_received: float) -> dict:
        if not 0 <= ownership_percentage <= 1:
            raise ValueError("ownership_percentage must be between 0 and 1")

        proportional_earnings = ownership_percentage * investee_net_income
        retained_earnings_share = proportional_earnings - dividends_received

        return {
            "ownership_percentage": ownership_percentage,
            "investee_net_income": investee_net_income,
            "dividends_received": dividends_received,
            "lookthrough_earnings": proportional_earnings,
            "retained_earnings_share": retained_earnings_share
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
