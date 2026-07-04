import argparse
import json
import logging
import requests
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime
from typing import Dict, List, Any, Optional
from evaluator_config import DEFAULT_OLLAMA_MODEL, OLLAMA_GENERATE_URL
from evaluator_thresholds import (
    BUSINESS_MODEL_ASSET_HEAVY_CAPITAL_INTENSITY_MIN,
    BUSINESS_MODEL_ASSET_HEAVY_GROSS_MARGIN_MAX,
    BUSINESS_MODEL_ASSET_LIGHT_GROSS_MARGIN_MIN,
    BUSINESS_MODEL_RECURRING_REVENUE_MIN,
    DURABILITY_RETURN_ON_CAPITAL_MIN,
    GOODWILL_ECONOMIC_MULTIPLE_MAX,
    GOODWILL_ECONOMIC_RETURN_ON_TANGIBLE_ASSETS_MIN,
    GOODWILL_MIXED_MULTIPLE_MAX,
    GOODWILL_MIXED_RETURN_ON_TANGIBLE_ASSETS_MIN,
    INFLATION_MARGIN_CHANGE_FLOOR,
    INFLATION_SPIKE_RATE_MIN,
)
from thinking_frameworks import fetch_company_info
from financial_metrics import fetch_deep_financials

# --- Configuration ---
OLLAMA_API_URL = OLLAMA_GENERATE_URL
DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def fetch_goodwill_metrics(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    income_stmt = stock.income_stmt

    goodwill = _get_statement_value(balance_sheet, ("Goodwill", "Goodwill And Other Intangible Assets"))
    intangible_assets = _get_statement_value(
        balance_sheet,
        (
            "Other Intangible Assets",
            "Other Intangible Assets Excluding Goodwill",
            "Net Tangible Assets",
        ),
    )
    total_assets = _get_statement_value(balance_sheet, ("Total Assets",))
    net_income = _get_statement_value(income_stmt, ("Net Income", "Operating Income"))

    tangible_assets = None
    if total_assets is not None:
        tangible_assets = total_assets - (goodwill or 0.0)
        if intangible_assets is not None and intangible_assets > 0:
            tangible_assets -= intangible_assets

    return_on_tangible_assets = None
    if net_income is not None and tangible_assets not in (None, 0):
        return_on_tangible_assets = net_income / tangible_assets

    return {
        "goodwill": goodwill,
        # Net income is the closest broadly available real earnings figure in public statements.
        "acquired_earnings": net_income,
        "return_on_tangible_assets": return_on_tangible_assets,
    }

def _query_ollama_json(prompt: str, model: str = DEFAULT_MODEL) -> dict:
    """Helper function to query the local Ollama API and return parsed JSON."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return json.loads(data.get("response", "{}"))
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error communicating with Ollama: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Ollama response: {e}")
    
    return {}

def fetch_historical_margins(ticker: str) -> pd.DataFrame:
    """Fetches historical Gross and Operating margins for a given company using yfinance."""
    logger.info(f"Fetching historical financial margins for {ticker}...")
    
    try:
        t = yf.Ticker(ticker)
        financials = t.financials
        
        if financials.empty:
            logger.warning(f"No financial data found for {ticker}.")
            return pd.DataFrame()

        # Transpose so dates are rows
        df = financials.T
        
        # Standardize column names (yfinance returns different names sometimes)
        revenue_cols = [col for col in df.columns if "Total Revenue" in col or "Operating Revenue" in col]
        gross_profit_cols = [col for col in df.columns if "Gross Profit" in col]
        operating_income_cols = [col for col in df.columns if "Operating Income" in col]
        
        if not revenue_cols or not gross_profit_cols:
            logger.error(f"Missing required financial columns (Revenue or Gross Profit) for {ticker}")
            return pd.DataFrame()
            
        rev_col = revenue_cols[0]
        gp_col = gross_profit_cols[0]
        
        # Calculate margins
        df["Gross_Margin"] = df[gp_col] / df[rev_col]
        
        if operating_income_cols:
            df["Operating_Margin"] = df[operating_income_cols[0]] / df[rev_col]
        else:
            df["Operating_Margin"] = pd.NA
            
        df = df.reset_index().rename(columns={"index": "Date"})
        df["Year"] = df["Date"].dt.year
        
        return df[["Year", "Gross_Margin", "Operating_Margin"]].sort_values("Year").dropna(subset=["Gross_Margin"])
        
    except Exception as e:
        logger.error(f"Error fetching data from yfinance for {ticker}: {e}")
        return pd.DataFrame()

def fetch_cpi_inflation_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetches CPI inflation data using pandas_datareader from FRED."""
    logger.info(f"Fetching CPI inflation data from FRED ({start_year}-{end_year})...")
    start = datetime.datetime(start_year - 1, 1, 1) # Get previous year for YoY calculation
    end = datetime.datetime(end_year, 12, 31)
    
    try:
        cpi = web.DataReader('CPIAUCSL', 'fred', start, end)
        
        # Resample to Annual Average
        cpi_annual = cpi.resample('YE').mean()
        
        # Calculate Year-over-Year Percentage Change
        cpi_annual['Inflation_Rate'] = cpi_annual['CPIAUCSL'].pct_change() * 100
        cpi_annual = cpi_annual.reset_index()
        cpi_annual['Year'] = cpi_annual['DATE'].dt.year
        
        # Filter for requested years
        df = cpi_annual[(cpi_annual['Year'] >= start_year) & (cpi_annual['Year'] <= end_year)].copy()
        
        return df[['Year', 'Inflation_Rate']].dropna()
    except Exception as e:
        logger.error(f"Error fetching FRED data via pandas_datareader: {e}")
        return pd.DataFrame()

def overlay_margins_on_inflation(margins_df: pd.DataFrame, inflation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mathematically overlay margins onto inflation spikes to test for actual pricing power.
    Calculates the change in margin relative to the change in inflation.
    """
    if margins_df.empty or inflation_df.empty:
        logger.warning("Insufficient data to overlay margins on inflation.")
        return pd.DataFrame()
        
    logger.info("Analyzing pricing power during inflation spikes...")
    
    merged_df = pd.merge(margins_df, inflation_df, on="Year", how="inner")
    
    # Identify inflation spikes (e.g., inflation > 3.0%)
    merged_df["Is_Inflation_Spike"] = merged_df["Inflation_Rate"] > INFLATION_SPIKE_RATE_MIN
    
    # Calculate Year-over-Year changes securely
    merged_df["Inflation_YoY_Change"] = merged_df["Inflation_Rate"].diff()
    merged_df["Gross_Margin_YoY_Change"] = merged_df["Gross_Margin"].diff()
    
    if "Operating_Margin" in merged_df.columns:
        merged_df["Operating_Margin_YoY_Change"] = merged_df["Operating_Margin"].diff()
    
    def assess_pricing_power(row: pd.Series) -> str:
        if pd.isna(row["Gross_Margin_YoY_Change"]):
            return "N/A"
        if row["Is_Inflation_Spike"]:
            if row["Gross_Margin_YoY_Change"] >= INFLATION_MARGIN_CHANGE_FLOOR: # Less than 1% margin degradation
                return "Strong (Maintained Margin)"
            return "Weak (Margin Degraded)"
        return "Normal Environment"
        
    merged_df["Pricing_Power_Assessment"] = merged_df.apply(assess_pricing_power, axis=1)
    return merged_df

def infer_business_details(ticker: str, model: str = DEFAULT_MODEL) -> dict:
    """Use SEC business description where available, then infer products and competitors with Ollama."""
    logger.info(f"Extracting business details for {ticker}...")

    try:
        company_info = fetch_company_info(ticker)
        company_name = company_info.get("name", ticker)
        business_summary = company_info.get("description", "")
    except Exception as e:
        logger.error(f"Error fetching ticker info for {ticker}: {e}")
        return {"company_name": ticker, "products": [], "competitors": []}
    
    if not business_summary:
        logger.warning(f"No business summary found for {ticker}.")
        return {"company_name": company_name, "products": [], "competitors": []}
        
    prompt = f"""
    Based on the following business summary for {company_name}, extract the core products/services and the top competitors.
    Business Summary: {business_summary}
    
    Output strictly as a JSON object with this schema:
    {{
        "products": ["product1", "product2"],
        "competitors": ["comp1", "comp2"]
    }}
    Do not include any markdown or text outside the JSON.
    """
    
    data = _query_ollama_json(prompt, model)
    return {
        "company_name": company_name,
        "business_description": business_summary,
        "products": data.get("products", []),
        "competitors": data.get("competitors", [])
    }

def classify_moat_with_ollama(company_name: str, products: List[str], competitors: List[str], model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Constructs an Ollama prompt to classify the moat (Brand, Switching Costs, Network Effect, 
    Low-Cost Producer, or Commodity) and returns JSON.
    """
    logger.info(f"Classifying business moat for {company_name}...")
    
    products_str = ", ".join(products) if products else "Unknown"
    competitors_str = ", ".join(competitors) if competitors else "Unknown"
    
    prompt = f"""
You are an expert value investor analyzing {company_name}. 

Company Products/Services: {products_str}
Top 5 Competitors: {competitors_str}

Based on these products and competitors, classify the company's business moat into exactly one of these buckets: 
- Brand
- Switching Costs
- Network Effect
- Low-Cost Producer
- Commodity (No Moat)

Justify your reasoning in exactly 2 sentences. 

Output your response STRICTLY as a JSON object with the following schema:
{{
    "moat_type": "string",
    "justification": "string"
}}
Do not include markdown blocks or any other text outside the JSON.
"""
    return _query_ollama_json(prompt, model)

def main():
    parser = argparse.ArgumentParser(description="Analyze business moats and pricing power.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to use")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    
    # 1. Fetch Margins and Inflation
    margins_df = fetch_historical_margins(ticker)
    
    if not margins_df.empty:
        start_year = int(margins_df["Year"].min())
        end_year = int(margins_df["Year"].max())
        inflation_df = fetch_cpi_inflation_data(start_year, end_year)
        
        # 2. Overlay Margins on Inflation
        analysis_df = overlay_margins_on_inflation(margins_df, inflation_df)
        
        if not analysis_df.empty:
            print(f"\n--- Pricing Power Analysis for {ticker} ---")
            columns_to_display = ["Year", "Inflation_Rate", "Gross_Margin", "Pricing_Power_Assessment"]
            print(analysis_df[columns_to_display].tail(5).to_markdown(index=False))
        else:
            logger.warning("Could not complete pricing power analysis due to missing data.")
    else:
        logger.error(f"Could not fetch historical margins for {ticker}.")
        
    # 3. Infer Business Details
    details = infer_business_details(ticker, model=args.model)
    company_name = details["company_name"]
    products = details["products"]
    competitors = details["competitors"]
    
    print(f"\n--- Extracted Details for {company_name} ---")
    print(f"Products:    {', '.join(products) if products else 'None found'}")
    print(f"Competitors: {', '.join(competitors) if competitors else 'None found'}")
    
    # 4. Classify Moat using Local LLM
    moat_classification = classify_moat_with_ollama(company_name, products, competitors, model=args.model)
    
    print("\n--- Qualitative Moat Classification ---")
    print(json.dumps(moat_classification, indent=4))

if __name__ == "__main__":
    main()

class EconomicMoat:
    """
    Heuristic: Economic Moat
    """
    def __init__(self):
        self.OLLAMA_API_URL = OLLAMA_GENERATE_URL
        self.DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL

    def evaluate(self, ticker: str, company_name: str = "", products: list = None, competitors: list = None, model: str = DEFAULT_OLLAMA_MODEL) -> dict:
        company_name = company_name or ticker
        
        context_str = ""
        if products and competitors:
            products_str = ", ".join(products)
            competitors_str = ", ".join(competitors)
            context_str = f"Company Products/Services: {products_str}\nTop 5 Competitors: {competitors_str}"
        else:
            from sec_data import fetch_filing_section
            try:
                description = fetch_filing_section(
                    ticker,
                    form="10-K",
                    start_markers=("item 1.", "item 1. business"),
                    end_markers=("item 1a.", "risk factors"),
                    max_chars=8000
                )
                context_str = f"Business Description from SEC 10-K:\n{description}"
            except Exception as e:
                return {"moat_type": "Unknown", "justification": f"Failed to fetch SEC description: {e}"}

        prompt = f"""
        You are an expert value investor analyzing {company_name}. 

        {context_str}

        Based on this information, classify the company's business moat into exactly one of these buckets: 
        - Brand
        - Switching Costs
        - Network Effect
        - Low-Cost Producer
        - Commodity (No Moat)

        Justify your reasoning in exactly 2 sentences. 

        Output your response STRICTLY as a JSON object with the following schema:
        {{
            "moat_type": "string",
            "justification": "string"
        }}
        Do not include markdown blocks or any other text outside the JSON.
        """
        return self._query_ollama_json(prompt, model)

    def _query_ollama_json(self, prompt: str, model: str) -> dict:
        import requests
        import json
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            response = requests.post(self.OLLAMA_API_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return json.loads(data.get("response", "{}"))
        except Exception as e:
            return {}

class BusinessModelTypes:
    """
    Heuristic: Business Model Types
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        recurring_revenue_ratio: Optional[float] = None,
        gross_margin: Optional[float] = None,
        capital_intensity: Optional[float] = None,
        ticker: str = "",
    ) -> dict:
        if ticker and any(value is None for value in (recurring_revenue_ratio, gross_margin, capital_intensity)):
            company_info = fetch_company_info(ticker)
            description = (company_info.get("description") or "").lower()
            recurring_keywords = (
                "subscription",
                "subscriptions",
                "recurring",
                "renewal",
                "renewals",
                "membership",
                "software as a service",
                "saas",
            )
            keyword_hits = sum(1 for keyword in recurring_keywords if keyword in description)
            recurring_revenue_ratio = min(1.0, max(0.1, keyword_hits / 4))

            if gross_margin is None:
                margins_df = fetch_historical_margins(ticker)
                if not margins_df.empty:
                    gross_margin = float(margins_df.sort_values("Year").iloc[-1]["Gross_Margin"])

            if capital_intensity is None:
                financials = fetch_deep_financials(ticker)
                total_revenue = financials.get("total_revenue")
                capex_total = financials.get("capex_total")
                if total_revenue and capex_total is not None:
                    capital_intensity = abs(float(capex_total)) / float(total_revenue)

        if recurring_revenue_ratio is None or gross_margin is None or capital_intensity is None:
            raise ValueError("recurring_revenue_ratio, gross_margin, and capital_intensity are required")

        if recurring_revenue_ratio >= BUSINESS_MODEL_RECURRING_REVENUE_MIN:
            model_type = "recurring_revenue"
        elif gross_margin <= BUSINESS_MODEL_ASSET_HEAVY_GROSS_MARGIN_MAX and capital_intensity >= BUSINESS_MODEL_ASSET_HEAVY_CAPITAL_INTENSITY_MIN:
            model_type = "asset_heavy"
        elif gross_margin >= BUSINESS_MODEL_ASSET_LIGHT_GROSS_MARGIN_MIN:
            model_type = "asset_light"
        else:
            model_type = "hybrid"

        return {
            "recurring_revenue_ratio": recurring_revenue_ratio,
            "gross_margin": gross_margin,
            "capital_intensity": capital_intensity,
            "business_model_type": model_type
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class GoodwillEconomicGoodwillVsAccountingGoodwill:
    """
    Heuristic: Goodwill: Economic Goodwill vs. Accounting Goodwill
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        goodwill: Optional[float] = None,
        acquired_earnings: Optional[float] = None,
        return_on_tangible_assets: Optional[float] = None,
        ticker: str = "",
    ) -> dict:
        if ticker and any(value is None for value in (goodwill, acquired_earnings, return_on_tangible_assets)):
            metrics = fetch_goodwill_metrics(ticker)
            goodwill = metrics["goodwill"]
            acquired_earnings = metrics["acquired_earnings"]
            return_on_tangible_assets = metrics["return_on_tangible_assets"]

        if goodwill is None or acquired_earnings is None or return_on_tangible_assets is None:
            raise ValueError("goodwill, acquired_earnings, and return_on_tangible_assets are required")

        goodwill_multiple = (goodwill / acquired_earnings) if acquired_earnings > 0 else None

        quality = "accounting_heavy"
        if goodwill_multiple is not None and goodwill_multiple <= GOODWILL_ECONOMIC_MULTIPLE_MAX and return_on_tangible_assets >= GOODWILL_ECONOMIC_RETURN_ON_TANGIBLE_ASSETS_MIN:
            quality = "economic_goodwill"
        elif goodwill_multiple is not None and goodwill_multiple <= GOODWILL_MIXED_MULTIPLE_MAX and return_on_tangible_assets >= GOODWILL_MIXED_RETURN_ON_TANGIBLE_ASSETS_MIN:
            quality = "mixed"

        return {
            "goodwill": goodwill,
            "acquired_earnings": acquired_earnings,
            "goodwill_multiple": goodwill_multiple,
            "return_on_tangible_assets": return_on_tangible_assets,
            "goodwill_quality": quality
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class TheDurabilityOfCompetitiveAdvantage:
    """
    Heuristic: The Durability of Competitive Advantage
    """
    def __init__(self):
        pass

    def evaluate(self, gross_margin_trend: float, market_share_trend: float, return_on_capital: float) -> dict:
        score = 0
        if gross_margin_trend >= 0:
            score += 1
        if market_share_trend >= 0:
            score += 1
        if return_on_capital >= DURABILITY_RETURN_ON_CAPITAL_MIN:
            score += 1

        durability = "weak"
        if score == 3:
            durability = "strong"
        elif score == 2:
            durability = "moderate"

        return {
            "gross_margin_trend": gross_margin_trend,
            "market_share_trend": market_share_trend,
            "return_on_capital": return_on_capital,
            "durability_score": score,
            "durability_assessment": durability
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
