import os
import json
import requests
from typing import Dict, Any, Optional
import yfinance as yf
from evaluator_config import DEFAULT_OLLAMA_MODEL, OLLAMA_GENERATE_URL
from evaluator_thresholds import (
    BEHAVIORAL_HIGH_RISK_FLAG_COUNT_MIN,
    BEHAVIORAL_HOLDING_PERIOD_MIN_YEARS,
    DERIVATIVES_EXPOSURE_HIGH_MIN,
    DERIVATIVES_EXPOSURE_MODERATE_MIN,
    INFLATION_MARGIN_CHANGE_FLOOR,
    INFLATION_SPIKE_RATE_MIN,
    LEVEL3_ASSETS_HIGH_MIN,
    LEVEL3_ASSETS_MODERATE_MIN,
    VALUE_TRAP_DEBT_TO_EQUITY_MAX,
    VALUE_TRAP_FLAG_COUNT_MIN,
    VALUE_TRAP_PE_RATIO_MAX,
    VALUE_TRAP_RETURN_ON_CAPITAL_MIN,
)
from sec_data import fetch_filing_section, get_cik_from_ticker as get_cik_from_ticker_from_sec

# Constants
SEC_API_KEY = os.environ.get("SEC_API_KEY", "your_sec_api_key_here")
OLLAMA_API_URL = OLLAMA_GENERATE_URL
MODEL_NAME = DEFAULT_OLLAMA_MODEL

def get_cik_from_ticker(ticker: str, headers: dict) -> str:
    """Fetches the CIK for a given ticker using the SEC company tickers JSON."""
    return get_cik_from_ticker_from_sec(ticker)

def fetch_sec_10k_footnotes(ticker: str) -> str:
    """
    Fetches the 10-K Footnotes (specifically Leases, Pensions, Commitments) 
    from SEC EDGAR directly.
    """
    print(f"Fetching 10-K footnotes for {ticker} from SEC EDGAR...")

    try:
        footnotes = fetch_filing_section(
            ticker,
            form="10-K",
            start_markers=(
                "notes to consolidated financial statements",
                "notes to financial statements",
                "notes to consolidated and combined financial statements",
                "note 1. business and summary of significant accounting policies",
            ),
            end_markers=("item 9", "changes in and disagreements with accountants"),
            max_chars=20000,
            prefer_notes_body=True,
        )
        if footnotes:
            return footnotes
        raise ValueError(f"Footnotes section not found for {ticker}")

    except Exception as e:
        raise RuntimeError(f"Failed to fetch 10-K footnotes for {ticker}: {e}") from e

def analyze_footnotes_with_ollama(footnotes_text: str) -> Optional[Dict[str, Any]]:
    """
    Feeds the footnotes to Ollama to act as a 'Footnote Detective'.
    Forces JSON output extracting specific liability metrics.
    """
    print("Analyzing footnotes with Ollama...")
    
    prompt = f"""You are a 'Footnote Detective' analyzing SEC 10-K financial footnotes for hidden liabilities.
Scan the following footnotes and extract the requested information.

Footnotes text:
{footnotes_text}

Return a JSON extracting exactly these three fields:
1) "operating_lease_obligations": Total operating lease obligations (include units if specified, e.g., "$450 million"). If not found, return "Not found".
2) "pension_underfunding": Pension plan underfunding amount. If not found, return "Not found".
3) "toxic_derivative_exposure": Any toxic derivative exposure or significant complex derivative risks mentioned. Describe briefly or state "None mentioned" / "No material exposure".

Respond ONLY with valid JSON. Do not include markdown formatting like ```json or any other text.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Parse the JSON response from Ollama
        json_output = json.loads(result.get("response", "{}"))
        return json_output
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding Ollama JSON response: {e}")
        print("Raw response:", result.get("response"))
        return None


def _get_statement_value(statement, names: tuple[str, ...], column_index: int = 0) -> Optional[float]:
    if statement is None or getattr(statement, "empty", True):
        return None
    for name in names:
        if name in statement.index:
            value = statement.loc[name].iloc[column_index]
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def fetch_value_trap_metrics(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    income_stmt = stock.income_stmt
    cashflow = stock.cashflow
    balance_sheet = stock.balance_sheet

    pe_ratio = info.get("trailingPE") or info.get("forwardPE")
    debt_to_equity = info.get("debtToEquity")
    if debt_to_equity is not None:
        debt_to_equity = float(debt_to_equity) / 100.0

    revenue_growth = None
    if income_stmt is not None and not income_stmt.empty and income_stmt.shape[1] >= 2:
        recent_revenue = _get_statement_value(income_stmt, ("Total Revenue", "Operating Revenue"), 0)
        prior_revenue = _get_statement_value(income_stmt, ("Total Revenue", "Operating Revenue"), 1)
        if recent_revenue is not None and prior_revenue not in (None, 0):
            revenue_growth = (recent_revenue - prior_revenue) / abs(prior_revenue)

    free_cash_flow_growth = None
    if cashflow is not None and not cashflow.empty and cashflow.shape[1] >= 2:
        recent_fcf = _get_statement_value(cashflow, ("Free Cash Flow",), 0)
        prior_fcf = _get_statement_value(cashflow, ("Free Cash Flow",), 1)
        if recent_fcf is None or prior_fcf is None:
            recent_ocf = _get_statement_value(cashflow, ("Operating Cash Flow", "Total Cash From Operating Activities"), 0)
            prior_ocf = _get_statement_value(cashflow, ("Operating Cash Flow", "Total Cash From Operating Activities"), 1)
            recent_capex = _get_statement_value(cashflow, ("Capital Expenditure",), 0)
            prior_capex = _get_statement_value(cashflow, ("Capital Expenditure",), 1)
            if None not in (recent_ocf, recent_capex):
                recent_fcf = recent_ocf + recent_capex
            if None not in (prior_ocf, prior_capex):
                prior_fcf = prior_ocf + prior_capex
        if recent_fcf is not None and prior_fcf not in (None, 0):
            free_cash_flow_growth = (recent_fcf - prior_fcf) / abs(prior_fcf)

    return_on_capital = None
    earnings = _get_statement_value(income_stmt, ("Operating Income", "Net Income"), 0)
    total_debt = _get_statement_value(balance_sheet, ("Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"), 0)
    equity = _get_statement_value(balance_sheet, ("Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"), 0)
    cash = _get_statement_value(balance_sheet, ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments"), 0)
    if None not in (earnings, total_debt, equity):
        invested_capital = total_debt + equity - (cash or 0.0)
        if invested_capital > 0:
            return_on_capital = earnings / invested_capital

    return {
        "pe_ratio": float(pe_ratio) if pe_ratio is not None else None,
        "revenue_growth": revenue_growth,
        "free_cash_flow_growth": free_cash_flow_growth,
        "debt_to_equity": debt_to_equity,
        "return_on_capital": return_on_capital,
    }

def main():
    ticker = "AAPL"
    
    # 1. Fetch SEC 10-K Footnotes
    footnotes = fetch_sec_10k_footnotes(ticker)
    
    # 2. Analyze with Ollama (Footnote Detective)
    analysis_result = analyze_footnotes_with_ollama(footnotes)
    
    # 3. Output results
    if analysis_result:
        print("\n--- Footnote Detective Findings ---")
        print(json.dumps(analysis_result, indent=2))
    else:
        print("Failed to analyze footnotes.")

if __name__ == "__main__":
    main()


class WhenToSellClearCriteria:
    """
    Heuristic: When to Sell (Clear Criteria)
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        thesis_broken: bool,
        better_opportunity_available: bool,
        extreme_overvaluation: bool,
        balance_sheet_deterioration: bool,
    ) -> dict:
        reasons = []
        if thesis_broken:
            reasons.append("thesis_broken")
        if better_opportunity_available:
            reasons.append("better_opportunity_available")
        if extreme_overvaluation:
            reasons.append("extreme_overvaluation")
        if balance_sheet_deterioration:
            reasons.append("balance_sheet_deterioration")

        return {
            "sell": bool(reasons),
            "reasons": reasons,
            "primary_reason": reasons[0] if reasons else None
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class ValueTraps:
    """
    Heuristic: Value Traps
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        pe_ratio: Optional[float] = None,
        revenue_growth: Optional[float] = None,
        free_cash_flow_growth: Optional[float] = None,
        debt_to_equity: Optional[float] = None,
        return_on_capital: Optional[float] = None,
        ticker: str = "",
    ) -> dict:
        if ticker and any(value is None for value in (pe_ratio, revenue_growth, free_cash_flow_growth, debt_to_equity, return_on_capital)):
            metrics = fetch_value_trap_metrics(ticker)
            pe_ratio = metrics["pe_ratio"]
            revenue_growth = metrics["revenue_growth"]
            free_cash_flow_growth = metrics["free_cash_flow_growth"]
            debt_to_equity = metrics["debt_to_equity"]
            return_on_capital = metrics["return_on_capital"]

        if any(value is None for value in (pe_ratio, revenue_growth, free_cash_flow_growth, debt_to_equity, return_on_capital)):
            raise ValueError("pe_ratio, revenue_growth, free_cash_flow_growth, debt_to_equity, and return_on_capital are required")

        flags = []

        if pe_ratio <= VALUE_TRAP_PE_RATIO_MAX:
            flags.append("cheap_multiple")
        if revenue_growth < 0:
            flags.append("shrinking_revenue")
        if free_cash_flow_growth < 0:
            flags.append("shrinking_free_cash_flow")
        if debt_to_equity > VALUE_TRAP_DEBT_TO_EQUITY_MAX:
            flags.append("high_leverage")
        if return_on_capital < VALUE_TRAP_RETURN_ON_CAPITAL_MIN:
            flags.append("weak_returns_on_capital")

        is_value_trap = "cheap_multiple" in flags and len(flags) >= VALUE_TRAP_FLAG_COUNT_MIN
        return {
            "is_value_trap": is_value_trap,
            "risk_flags": flags,
            "flag_count": len(flags)
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class LeverageRisk:
    """
    Heuristic: Leverage Risk
    """
    def __init__(self):
        pass

    def evaluate(self, ticker: str) -> dict:
        footnotes = self._fetch_sec_10k_footnotes(ticker)
        analysis = self._analyze_footnotes_with_ollama(footnotes)
        return analysis if analysis else {}

    def _fetch_sec_10k_footnotes(self, ticker: str) -> str:
        return fetch_sec_10k_footnotes(ticker)

    def _analyze_footnotes_with_ollama(self, footnotes_text: str) -> dict:
        import requests, json
        prompt = f"""Analyze footnotes. Return JSON with 'operating_lease_obligations', 'pension_underfunding', 'toxic_derivative_exposure'. Footnotes: {footnotes_text}"""
        try:
            res = requests.post(OLLAMA_API_URL, json={"model": MODEL_NAME, "prompt": prompt, "format": "json", "stream": False}, timeout=60)
            return json.loads(res.json().get("response", "{}"))
        except Exception:
            return {}


class TheImpactOfInflation:
    """
    Heuristic: The Impact of Inflation
    """
    def __init__(self):
        pass

    def evaluate(self, margins_df=None, inflation_df=None, ticker: str = ""):
        from business_moat import fetch_historical_margins, fetch_cpi_inflation_data
        import pandas as pd

        if ticker and (margins_df is None or inflation_df is None):
            margins_df = fetch_historical_margins(ticker)
            if not margins_df.empty:
                inflation_df = fetch_cpi_inflation_data(int(margins_df["Year"].min()), int(margins_df["Year"].max()))

        if margins_df is None or inflation_df is None:
            raise ValueError("margins_df and inflation_df are required")

        if margins_df.empty or inflation_df.empty:
            return pd.DataFrame()
            
        merged_df = pd.merge(margins_df, inflation_df, on="Year", how="inner")
        merged_df["Is_Inflation_Spike"] = merged_df["Inflation_Rate"] > INFLATION_SPIKE_RATE_MIN
        merged_df["Inflation_YoY_Change"] = merged_df["Inflation_Rate"].diff()
        merged_df["Gross_Margin_YoY_Change"] = merged_df["Gross_Margin"].diff()
        
        if "Operating_Margin" in merged_df.columns:
            merged_df["Operating_Margin_YoY_Change"] = merged_df["Operating_Margin"].diff()
        
        def assess_pricing_power(row: pd.Series) -> str:
            if pd.isna(row["Gross_Margin_YoY_Change"]): return "N/A"
            if row["Is_Inflation_Spike"]:
                return "Strong (Maintained Margin)" if row["Gross_Margin_YoY_Change"] >= INFLATION_MARGIN_CHANGE_FLOOR else "Weak (Margin Degraded)"
            return "Normal Environment"
            
        merged_df["Pricing_Power_Assessment"] = merged_df.apply(assess_pricing_power, axis=1)
        return merged_df

    def _helper_method(self):
        pass


class DerivativesRisk:
    """
    Heuristic: Derivatives Risk
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        notional_exposure: Optional[float] = None,
        equity_capital: Optional[float] = None,
        level_3_assets_ratio: Optional[float] = None,
        ticker: str = "",
    ) -> dict:
        if ticker and any(value is None for value in (notional_exposure, equity_capital, level_3_assets_ratio)):
            footnote_analysis = LeverageRisk().evaluate(ticker)
            toxic_summary = str(footnote_analysis.get("toxic_derivative_exposure") or "").strip()

            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet
            equity_capital = _get_statement_value(
                balance_sheet,
                ("Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"),
            )

            normalized_summary = toxic_summary.lower()
            if normalized_summary:
                if any(token in normalized_summary for token in ("none", "no material", "not found")):
                    risk = "low"
                elif any(token in normalized_summary for token in ("significant", "material", "substantial", "toxic")):
                    risk = "high"
                else:
                    risk = "moderate"

                return {
                    "notional_exposure": notional_exposure,
                    "equity_capital": equity_capital,
                    "exposure_ratio": None,
                    "level_3_assets_ratio": level_3_assets_ratio,
                    "derivative_exposure_summary": toxic_summary,
                    "derivatives_risk": risk,
                }

        if notional_exposure is None or equity_capital is None or level_3_assets_ratio is None:
            raise ValueError("notional_exposure, equity_capital, and level_3_assets_ratio are required")

        exposure_ratio = (notional_exposure / equity_capital) if equity_capital > 0 else None

        risk = "low"
        if exposure_ratio is None or exposure_ratio > DERIVATIVES_EXPOSURE_HIGH_MIN or level_3_assets_ratio > LEVEL3_ASSETS_HIGH_MIN:
            risk = "high"
        elif exposure_ratio > DERIVATIVES_EXPOSURE_MODERATE_MIN or level_3_assets_ratio > LEVEL3_ASSETS_MODERATE_MIN:
            risk = "moderate"

        return {
            "notional_exposure": notional_exposure,
            "equity_capital": equity_capital,
            "exposure_ratio": exposure_ratio,
            "level_3_assets_ratio": level_3_assets_ratio,
            "derivatives_risk": risk
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class CommonBehavioralBiasesPsychologicalTrapsInInvesting:
    """
    Heuristic: Common Behavioral Biases (Psychological Traps in Investing)
    """
    def __init__(self):
        pass

    def evaluate(self, thesis_changes_after_price_move: bool, avg_holding_period_years: float, adds_to_losers_without_new_evidence: bool) -> dict:
        flags = []
        if thesis_changes_after_price_move:
            flags.append("recency_bias")
        if avg_holding_period_years < BEHAVIORAL_HOLDING_PERIOD_MIN_YEARS:
            flags.append("impatience")
        if adds_to_losers_without_new_evidence:
            flags.append("commitment_bias")

        return {
            "bias_flags": flags,
            "bias_count": len(flags),
            "behavioral_risk": "high" if len(flags) >= BEHAVIORAL_HIGH_RISK_FLAG_COUNT_MIN else "moderate" if len(flags) == 1 else "low"
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
