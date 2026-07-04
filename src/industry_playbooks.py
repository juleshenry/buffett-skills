import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional
import yfinance as yf
import requests
from evaluator_config import DEFAULT_OLLAMA_MODEL, OLLAMA_GENERATE_URL
from sec_data import fetch_filing_keyword_context
from evaluator_thresholds import (
    AVOID_INDUSTRY_COMMODITY_EXPOSURE_MIN,
    AVOID_INDUSTRY_LEVERAGE_RATIO_MIN,
    AVOID_INDUSTRY_PRICING_POWER_MAX,
    AVOID_INDUSTRY_RED_FLAG_COUNT_MIN,
    CONSUMER_BRAND_GROSS_MARGIN_MIN,
    MEDIA_AD_REVENUE_MAX,
    MEDIA_CHURN_MAX,
    MEDIA_SUBSCRIPTION_REVENUE_MIN,
    RAILWAY_MAINTENANCE_CAPEX_RATIO_MAX,
    RAILWAY_OPERATING_RATIO_MAX,
    TECH_NET_REVENUE_RETENTION_MIN,
    TECH_RECURRING_REVENUE_MIN,
    TECH_STOCK_COMP_RATIO_MAX,
    UNDERWRITING_DISCIPLINED_COMBINED_RATIO_MAX,
    UNDERWRITING_EXCELLENT_COMBINED_RATIO_MAX,
    UTILITY_ALLOWED_ROE_MIN,
    UTILITY_DEBT_TO_EBITDA_MAX,
    UTILITY_REGULATED_ASSET_RATIO_MIN,
)

OLLAMA_URL = OLLAMA_GENERATE_URL

def query_ollama(prompt: str, context_text: str, model: str = DEFAULT_OLLAMA_MODEL) -> Optional[Dict[str, Any]]:
    """Query the local Ollama API to analyze text and return structured JSON."""
    full_prompt = f"{prompt}\n\nContext:\n{context_text}"
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json"
    }
    
    req = urllib.request.Request(
        OLLAMA_URL, 
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            return json.loads(result.get("response", "{}"))
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None

def fetch_industry_data(ticker: str, industry_type: str) -> Dict[str, Any]:
    """
    Fetch industry-specific data using yfinance and SEC EDGAR.
    """
    print(f"Fetching {industry_type} data for {ticker}...")
    
    result = {}
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result = {
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            "totalRevenue": info.get('totalRevenue'),
            "ebitdaMargins": info.get('ebitdaMargins'),
        }
    except Exception as e:
        print(f"Error fetching yfinance data for {ticker}: {e}")

    sec_text = ""
    keyword_map = {
        "bank": ("tier 1 capital", "common equity tier 1", "stress test", "capital adequacy", "regulatory capital"),
        "banking": ("tier 1 capital", "common equity tier 1", "stress test", "capital adequacy", "regulatory capital"),
        "insurance": ("reserve development", "loss reserves", "claims reserves", "adverse development", "favorable development", "incurred but not reported"),
        "general": (),
    }

    keywords = keyword_map.get(industry_type.lower(), ())
    if keywords:
        for form in ("10-K", "10-Q"):
            try:
                sec_text = fetch_filing_keyword_context(
                    ticker,
                    form=form,
                    keywords=keywords,
                    context_chars=1600,
                    max_matches=3,
                    max_chars=8000,
                )
                if sec_text:
                    break
            except Exception as e:
                print(f"SEC EDGAR fetch failed for {ticker} {form}: {e}")

    if industry_type.lower() == "bank":
        result["regulatory_text"] = sec_text
    elif industry_type.lower() == "insurance":
        result["reserve_development_text"] = sec_text
        
    return result

def analyze_bank(ticker: str) -> None:
    data = fetch_industry_data(ticker, "bank")
    text = data.get("regulatory_text", "")
    
    prompt = (
        "You are a bank analyst. Read the following regulatory and stress-test commentary. "
        "Summarize the tier 1 capital adequacy risks. Return a JSON object with two keys: "
        "'capital_adequate' (boolean) and 'risk_summary' (string under 50 words)."
    )
    
    print(f"\n--- Analyzing Bank: {ticker} ---")
    print(f"Sector: {data.get('sector')}, Industry: {data.get('industry')}")
    print(f"Total Revenue: {data.get('totalRevenue')}, EBITDA Margin: {data.get('ebitdaMargins')}")
    result = query_ollama(prompt, text)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Analysis failed.")

def analyze_insurance(ticker: str) -> None:
    data = fetch_industry_data(ticker, "insurance")
    text = data.get("reserve_development_text", "")
    
    prompt = (
        "You are an insurance analyst. Read the following 'Reserve Development' text. "
        "Determine if the company is consistently underestimating future payouts "
        "(unfavorable/adverse development), which is a sign of a bad operator. "
        "Return a JSON object with two keys: 'underestimating_payouts' (boolean) and "
        "'development_summary' (string under 50 words)."
    )
    
    print(f"\n--- Analyzing Insurance: {ticker} ---")
    print(f"Sector: {data.get('sector')}, Industry: {data.get('industry')}")
    print(f"Total Revenue: {data.get('totalRevenue')}, EBITDA Margin: {data.get('ebitdaMargins')}")
    result = query_ollama(prompt, text)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Analysis failed.")

if __name__ == "__main__":
    # Example usage:
    analyze_bank("JPM")
    analyze_insurance("BRK.A")


class Insurance:
    """
    Heuristic: Insurance
    """
    def __init__(self):
        pass

    def evaluate(self, ticker: str) -> dict:
        data = self._fetch_industry_data(ticker, "insurance")
        prompt = f"""You are Warren Buffett analyzing an insurance company.
        Focus on: 1. Combined Ratio (Underwriting profitability, < 100%). 2. Float growth.
        Analyze: {data}
        Return JSON: {{"underwriting_discipline_assessment": string, "float_quality": string, "verdict": "Investable" | "Too Risky"}}
        """
        return self._query_ollama(prompt, str(data)) or {}

    def _fetch_industry_data(self, ticker: str, industry_type: str) -> dict:
        return fetch_industry_data(ticker, industry_type)

    def _query_ollama(self, prompt: str, context: str) -> dict:
        import requests, json
        try:
            res = requests.post(OLLAMA_URL, json={"model": DEFAULT_OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False}, timeout=60)
            return json.loads(res.json().get("response", "{}"))
        except Exception:
            return {}


class UnderwritingDiscipline:
    """
    Heuristic: Underwriting Discipline
    """
    def __init__(self):
        pass

    def evaluate(self, combined_ratio: float) -> dict:
        if combined_ratio < UNDERWRITING_EXCELLENT_COMBINED_RATIO_MAX:
            assessment = "excellent"
        elif combined_ratio < UNDERWRITING_DISCIPLINED_COMBINED_RATIO_MAX:
            assessment = "disciplined"
        else:
            assessment = "undisciplined"

        return {
            "combined_ratio": combined_ratio,
            "underwriting_discipline": assessment,
            "profitable_underwriting": combined_ratio < UNDERWRITING_DISCIPLINED_COMBINED_RATIO_MAX
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class InsuranceFloat:
    """
    Heuristic: Insurance Float
    """
    def __init__(self):
        pass

    def evaluate(self, current_float: float, prior_float: float, combined_ratio: float) -> dict:
        float_growth = current_float - prior_float
        if float_growth > 0 and combined_ratio < UNDERWRITING_DISCIPLINED_COMBINED_RATIO_MAX:
            quality = "valuable"
        elif float_growth > 0:
            quality = "growing_but_costly"
        else:
            quality = "shrinking"

        return {
            "current_float": current_float,
            "prior_float": prior_float,
            "float_growth": float_growth,
            "combined_ratio": combined_ratio,
            "float_quality": quality
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class Banking:
    """
    Heuristic: Banking
    """
    def __init__(self):
        pass

    def evaluate(self, ticker: str) -> dict:
        data = self._fetch_industry_data(ticker, "banking")
        prompt = f"""You are Warren Buffett analyzing a bank.
        Focus on: 1. Return on Assets (ROA > 1%). 2. Return on Equity (ROE > 10%). 3. Deposit base stickiness.
        Analyze: {data}
        Return JSON: {{"roa_assessment": string, "roe_assessment": string, "deposit_franchise_quality": string, "verdict": "Investable" | "Too Risky"}}
        """
        return self._query_ollama(prompt, str(data)) or {}

    def _fetch_industry_data(self, ticker: str, industry_type: str) -> dict:
        return fetch_industry_data(ticker, industry_type)

    def _query_ollama(self, prompt: str, context: str) -> dict:
        import requests, json
        try:
            res = requests.post(OLLAMA_URL, json={"model": DEFAULT_OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False}, timeout=60)
            return json.loads(res.json().get("response", "{}"))
        except Exception:
            return {}


class ConsumerBrandsRetail:
    """
    Heuristic: Consumer Brands & Retail
    """
    def __init__(self):
        pass

    def evaluate(self, gross_margin: float | None = None, same_store_sales_growth: float | None = None, brand_share_trend: float | None = None, ticker: str = "") -> dict:
        if ticker and (gross_margin is None or same_store_sales_growth is None or brand_share_trend is None):
            import yfinance as yf
            info = yf.Ticker(ticker).info
            gross_margin = gross_margin if gross_margin is not None else info.get("grossMargins", 0.0)
            same_store_sales_growth = same_store_sales_growth if same_store_sales_growth is not None else info.get("revenueGrowth", 0.0)
            brand_share_trend = brand_share_trend if brand_share_trend is not None else 0.0 # Approximation if not fetchable
            
        if gross_margin is None or same_store_sales_growth is None or brand_share_trend is None:
            raise ValueError("All metrics must be provided or fetchable via ticker")

        score = 0
        if gross_margin >= CONSUMER_BRAND_GROSS_MARGIN_MIN:
            score += 1
        if same_store_sales_growth >= 0:
            score += 1
        if brand_share_trend >= 0:
            score += 1

        return {
            "gross_margin": gross_margin,
            "same_store_sales_growth": same_store_sales_growth,
            "brand_share_trend": brand_share_trend,
            "consumer_brand_score": score,
            "consumer_brand_quality": "strong" if score >= 3 else "mixed" if score == 2 else "weak"
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class MediaPublishing:
    """
    Heuristic: Media & Publishing
    """
    def __init__(self):
        pass

    def evaluate(self, subscription_revenue_ratio: float, ad_revenue_ratio: float, churn_rate: float) -> dict:
        score = 0
        if subscription_revenue_ratio >= MEDIA_SUBSCRIPTION_REVENUE_MIN:
            score += 1
        if ad_revenue_ratio <= MEDIA_AD_REVENUE_MAX:
            score += 1
        if churn_rate <= MEDIA_CHURN_MAX:
            score += 1

        return {
            "subscription_revenue_ratio": subscription_revenue_ratio,
            "ad_revenue_ratio": ad_revenue_ratio,
            "churn_rate": churn_rate,
            "media_score": score,
            "media_quality": "strong" if score >= 3 else "mixed" if score == 2 else "weak"
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class EnergyUtilities:
    """
    Heuristic: Energy & Utilities
    """
    def __init__(self):
        pass

    def evaluate(self, regulated_asset_ratio: float | None = None, debt_to_ebitda: float | None = None, allowed_return_on_equity: float | None = None, ticker: str = "") -> dict:
        if ticker and (regulated_asset_ratio is None or debt_to_ebitda is None or allowed_return_on_equity is None):
            import yfinance as yf
            info = yf.Ticker(ticker).info
            regulated_asset_ratio = regulated_asset_ratio if regulated_asset_ratio is not None else 0.8 # Conservative fallback for utilities if unfetchable
            ebitda = info.get("ebitda")
            total_debt = info.get("totalDebt")
            computed_debt_to_ebitda = (total_debt / ebitda) if total_debt and ebitda else 0.0
            debt_to_ebitda = debt_to_ebitda if debt_to_ebitda is not None else computed_debt_to_ebitda
            allowed_return_on_equity = allowed_return_on_equity if allowed_return_on_equity is not None else info.get("returnOnEquity", 0.0)

        if regulated_asset_ratio is None or debt_to_ebitda is None or allowed_return_on_equity is None:
            raise ValueError("All metrics must be provided or fetchable via ticker")

        score = 0
        if regulated_asset_ratio >= UTILITY_REGULATED_ASSET_RATIO_MIN:
            score += 1
        if debt_to_ebitda <= UTILITY_DEBT_TO_EBITDA_MAX:
            score += 1
        if allowed_return_on_equity >= UTILITY_ALLOWED_ROE_MIN:
            score += 1

        return {
            "regulated_asset_ratio": regulated_asset_ratio,
            "debt_to_ebitda": debt_to_ebitda,
            "allowed_return_on_equity": allowed_return_on_equity,
            "utility_score": score,
            "utility_quality": "strong" if score >= 3 else "mixed" if score == 2 else "weak"
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class Railways:
    """
    Heuristic: Railways
    """
    def __init__(self):
        pass

    def evaluate(self, operating_ratio: float, volume_growth: float, maintenance_capex_ratio: float) -> dict:
        score = 0
        if operating_ratio <= RAILWAY_OPERATING_RATIO_MAX:
            score += 1
        if volume_growth >= 0:
            score += 1
        if maintenance_capex_ratio <= RAILWAY_MAINTENANCE_CAPEX_RATIO_MAX:
            score += 1

        return {
            "operating_ratio": operating_ratio,
            "volume_growth": volume_growth,
            "maintenance_capex_ratio": maintenance_capex_ratio,
            "railway_score": score,
            "railway_quality": "strong" if score >= 3 else "mixed" if score == 2 else "weak"
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class TechnologyInternet:
    """
    Heuristic: Technology & Internet
    """
    def __init__(self):
        pass

    def evaluate(self, recurring_revenue_ratio: float | None = None, net_revenue_retention: float | None = None, stock_comp_ratio: float | None = None, ticker: str = "") -> dict:
        if ticker and (recurring_revenue_ratio is None or net_revenue_retention is None or stock_comp_ratio is None):
            import yfinance as yf
            info = yf.Ticker(ticker).info
            # Fallbacks when hard to extract without NLP
            recurring_revenue_ratio = recurring_revenue_ratio if recurring_revenue_ratio is not None else 0.5
            net_revenue_retention = net_revenue_retention if net_revenue_retention is not None else 1.0
            # Rough proxy: we don't have stock comp natively in info easily, default to conservative
            stock_comp_ratio = stock_comp_ratio if stock_comp_ratio is not None else 0.10

        if recurring_revenue_ratio is None or net_revenue_retention is None or stock_comp_ratio is None:
            raise ValueError("All metrics must be provided or fetchable via ticker")

        score = 0
        if recurring_revenue_ratio >= TECH_RECURRING_REVENUE_MIN:
            score += 1
        if net_revenue_retention >= TECH_NET_REVENUE_RETENTION_MIN:
            score += 1
        if stock_comp_ratio <= TECH_STOCK_COMP_RATIO_MAX:
            score += 1

        return {
            "recurring_revenue_ratio": recurring_revenue_ratio,
            "net_revenue_retention": net_revenue_retention,
            "stock_comp_ratio": stock_comp_ratio,
            "technology_score": score,
            "technology_quality": "strong" if score >= 3 else "mixed" if score == 2 else "weak"
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class IndustriesToAvoidCounterexamples:
    """
    Heuristic: Industries to Avoid (Counter-Examples)
    """
    def __init__(self):
        pass

    def evaluate(self, commodity_exposure: float, leverage_ratio: float, pricing_power: float) -> dict:
        red_flags = 0
        if commodity_exposure >= AVOID_INDUSTRY_COMMODITY_EXPOSURE_MIN:
            red_flags += 1
        if leverage_ratio >= AVOID_INDUSTRY_LEVERAGE_RATIO_MIN:
            red_flags += 1
        if pricing_power <= AVOID_INDUSTRY_PRICING_POWER_MAX:
            red_flags += 1

        return {
            "commodity_exposure": commodity_exposure,
            "leverage_ratio": leverage_ratio,
            "pricing_power": pricing_power,
            "red_flag_count": red_flags,
            "avoid_industry": red_flags >= AVOID_INDUSTRY_RED_FLAG_COUNT_MIN
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
