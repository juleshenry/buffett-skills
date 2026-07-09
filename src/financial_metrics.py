from cache_utils import disk_cache
import json
import re
import requests
import yfinance as yf
from evaluator_config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_STRUCTURED_EXTRACTION_TEMPERATURE,
    OLLAMA_GENERATE_URL,
    call_ollama_panel_json,
)
from sec_data import fetch_filing_keyword_context, fetch_filing_section


def _coerce_percentage(value) -> float | None:
    """Best-effort coercion of an LLM-returned percentage into a bare float.

    Local panel models are told to "return ONLY JSON" with a float field but
    routinely answer with strings like "62.8%" anyway. Strip the decoration
    here, at the normalization boundary, so a formatting quirk in one judge's
    response doesn't raise ValueError deep inside OwnerEarnings.evaluate().
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group(0))
    return None


def normalize_capex_breakdown(result: dict | None) -> dict:
    result = result or {}
    maintenance_percentage = result.get("maintenance_percentage")
    if maintenance_percentage is None:
        maintenance_percentage = result.get("maintenance_capex_percentage")

    growth_percentage = result.get("growth_percentage")
    if growth_percentage is None:
        growth_percentage = result.get("growth_capex_percentage")

    return {
        "maintenance_percentage": _coerce_percentage(maintenance_percentage),
        "growth_percentage": _coerce_percentage(growth_percentage),
        "maintenance_capex_amount": result.get("maintenance_capex_amount"),
        "reasoning": result.get("reasoning") or result.get("analysis") or "",
    }

@disk_cache()
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

@disk_cache()
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
    
    print("Querying Ollama for Capex breakdown...")
    try:
        return normalize_capex_breakdown(call_ollama_panel_json(prompt, model=DEFAULT_OLLAMA_MODEL, options={"temperature": DEFAULT_STRUCTURED_EXTRACTION_TEMPERATURE}))
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return {}


@disk_cache()
def fetch_lookthrough_commentary(ticker: str) -> str:
    keyword_sets = (
        ("10-K", ("equity method", "equity in earnings", "dividends received", "ownership interest", "investee")),
        ("10-Q", ("equity method", "equity in earnings", "dividends received", "ownership interest", "investee")),
    )

    for form, keywords in keyword_sets:
        try:
            commentary = fetch_filing_keyword_context(
                ticker,
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


def _parse_lookthrough_inputs(commentary: str) -> dict:
    ownership_percentage = None
    ownership_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s+(?:ownership\s+interest|interest|stake)",
        commentary,
        flags=re.IGNORECASE,
    )
    if ownership_match:
        ownership_percentage = float(ownership_match.group(1)) / 100.0

    amounts = []
    for match in re.finditer(r"\$\s*(\d+(?:\.\d+)?)\s*(million|billion)?", commentary, flags=re.IGNORECASE):
        value = float(match.group(1))
        unit = (match.group(2) or "").lower()
        if unit == "billion":
            value *= 1_000_000_000
        elif unit == "million":
            value *= 1_000_000
        amounts.append((match.start(), value))

    investee_net_income = None
    dividends_received = None

    income_match = re.search(
        r"equity\s+in\s+earnings[^\$]{0,120}\$\s*(\d+(?:\.\d+)?)\s*(million|billion)?",
        commentary,
        flags=re.IGNORECASE,
    )
    if income_match:
        investee_net_income = float(income_match.group(1))
        unit = (income_match.group(2) or "").lower()
        if unit == "billion":
            investee_net_income *= 1_000_000_000
        elif unit == "million":
            investee_net_income *= 1_000_000

    dividends_match = re.search(
        r"dividends\s+received[^\$]{0,120}\$\s*(\d+(?:\.\d+)?)\s*(million|billion)?",
        commentary,
        flags=re.IGNORECASE,
    )
    if dividends_match:
        dividends_received = float(dividends_match.group(1))
        unit = (dividends_match.group(2) or "").lower()
        if unit == "billion":
            dividends_received *= 1_000_000_000
        elif unit == "million":
            dividends_received *= 1_000_000

    has_investee_evidence = any(token in commentary.lower() for token in ("equity method", "equity in earnings", "investee"))

    return {
        "has_investee_evidence": has_investee_evidence,
        "ownership_percentage": ownership_percentage,
        "investee_net_income": investee_net_income,
        "dividends_received": dividends_received,
        "has_complete_metrics": all(
            value is not None
            for value in (ownership_percentage, investee_net_income, dividends_received)
        ),
    }

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
            return {"applicable": False, "reason": "Missing required metrics: net_income, operating_cash_flow, and capex_total are required"}

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
        try:
            mda_text = self._fetch_mda_section(ticker)
        except RuntimeError as e:
            return {
                "applicable": False,
                "reason": str(e),
            }
        capex_breakdown = self._query_ollama_capex_breakdown(mda_text)
        
        net_income = financials.get("net_income") or 0
        operating_cash_flow = financials.get("operating_cash_flow") or 0
        total_capex = abs(financials.get("capex_total") or 0)

        depreciation = financials.get("depreciation_and_amortization")
        depreciation_source = "reported"
        if depreciation is None:
            # Fallback only when the cash flow statement doesn't disclose D&A
            # directly. OCF - NI conflates D&A with every other non-cash and
            # working-capital adjustment and can go negative (e.g. when
            # receivables/inventory swings outpace D&A for the period), which
            # is economically meaningless -- D&A is never negative -- so the
            # proxy is floored at 0 instead of silently understating owner
            # earnings.
            proxy = operating_cash_flow - net_income if operating_cash_flow and net_income else 0
            depreciation = max(proxy, 0)
            depreciation_source = "ocf_minus_ni_proxy"

        maintenance_pct_raw = capex_breakdown.get("maintenance_percentage")
        if maintenance_pct_raw is None:
            maintenance_pct = 1.0
        else:
            maintenance_pct = float(maintenance_pct_raw)
            if maintenance_pct > 1.0:
                maintenance_pct /= 100.0
        
        maintenance_capex = total_capex * maintenance_pct
        
        owner_earnings = net_income + depreciation - maintenance_capex
        
        return {
            "net_income": net_income,
            "depreciation_amortization_estimate": depreciation,
            "depreciation_amortization_source": depreciation_source,
            "total_capex": total_capex,
            "maintenance_capex_estimate": maintenance_capex,
            "owner_earnings": owner_earnings
        }

    def _fetch_deep_financials(self, ticker: str) -> dict:
        import pandas as pd
        import yfinance as yf
        stock = yf.Ticker(ticker)
        try:
            cashflow = stock.cashflow
            income_stmt = stock.income_stmt
            recent_cf = cashflow.iloc[:, 0] if not cashflow.empty else {}
            recent_is = income_stmt.iloc[:, 0] if not income_stmt.empty else {}
            capex = recent_cf.get("Capital Expenditure", recent_cf.get("CapitalExpenditure"))
            ocf = recent_cf.get("Operating Cash Flow")
            depreciation = recent_cf.get(
                "Depreciation And Amortization",
                recent_cf.get("Depreciation Amortization Depletion"),
            )
            if depreciation is not None and pd.isna(depreciation):
                depreciation = None
            return {
                "capex_total": capex,
                "operating_cash_flow": ocf,
                "net_income": recent_is.get("Net Income"),
                "total_revenue": recent_is.get("Total Revenue"),
                "depreciation_and_amortization": float(depreciation) if depreciation is not None else None,
            }
        except Exception:
            return {}

    def _fetch_mda_section(self, ticker: str) -> str:
        return fetch_mda_section(ticker)

    def _query_ollama_capex_breakdown(self, mda_text: str) -> dict:
        prompt = f"""Read this MD&A and estimate what percentage of Capital Expenditures (Capex) was spent on 'Maintenance' vs 'Growth'. Return ONLY JSON: {{"maintenance_percentage": float, "growth_percentage": float, "maintenance_capex_amount": string, "reasoning": string}}. MD&A Text: {mda_text}"""
        try:
            return normalize_capex_breakdown(call_ollama_panel_json(prompt, model=DEFAULT_OLLAMA_MODEL, options={"temperature": DEFAULT_STRUCTURED_EXTRACTION_TEMPERATURE}))
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
            return {"applicable": False, "reason": "Missing required metrics: total_revenue, net_income, operating_cash_flow, and capex_total are required"}

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

class LookthroughEarningsPrinciple:
    """
    Principle: Look-Through Earnings
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        ownership_percentage: float | None = None,
        investee_net_income: float | None = None,
        dividends_received: float | None = None,
        ticker: str = "",
    ) -> dict:
        if ticker and (ownership_percentage is None or investee_net_income is None or dividends_received is None):
            lookthrough_inputs = self._fetch_lookthrough_inputs(ticker)
            if not lookthrough_inputs["has_investee_evidence"]:
                return {
                    "ticker": ticker,
                    "applicable": False,
                    "reason": "No material equity investee evidence found in recent filings.",
                }
            if not lookthrough_inputs["has_complete_metrics"]:
                return {
                    "ticker": ticker,
                    "applicable": False,
                    "reason": "Investee-related disclosures were found in recent filings, but ownership percentage, investee earnings, and dividends received could not be extracted reliably.",
                }
            if ownership_percentage is None:
                ownership_percentage = lookthrough_inputs["ownership_percentage"]
            if investee_net_income is None:
                investee_net_income = lookthrough_inputs["investee_net_income"]
            if dividends_received is None:
                dividends_received = lookthrough_inputs["dividends_received"]

        if ownership_percentage is None or investee_net_income is None or dividends_received is None:
            return {"applicable": False, "reason": "Not applicable: No material equity investee evidence found or missing required metrics"}
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

    def _fetch_lookthrough_inputs(self, ticker: str) -> dict:
        commentary = fetch_lookthrough_commentary(ticker)
        if not commentary:
            return {
                "has_investee_evidence": False,
                "ownership_percentage": None,
                "investee_net_income": None,
                "dividends_received": None,
            }
        return _parse_lookthrough_inputs(commentary)

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
