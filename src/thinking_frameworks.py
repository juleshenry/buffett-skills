import yfinance as yf
import requests
import json
from typing import Dict, Any
import pandas as pd
from investment_philosophy import fetch_price_comparison_data
from sec_data import fetch_filing_section
from valuation_capital import fetch_risk_free_rate
from evaluator_config import (
    DEFAULT_CIRCLE_OF_COMPETENCE_TEMPERATURE,
    DEFAULT_INVERSION_TEMPERATURE,
    DEFAULT_MR_MARKET_PERIOD,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_GENERATE_URL,
)
from evaluator_thresholds import (
    INDEPENDENT_THINKING_EVIDENCE_MIN,
    INDEPENDENT_THINKING_VALUATION_GAP_MIN,
    LATTICE_BROAD_SCORE_MIN,
    LATTICE_DEVELOPING_SCORE_MIN,
    LONG_TERM_ACCEPTABLE_YEARS_MIN,
    LONG_TERM_STRONG_EXCESS_RETURN_MIN,
    LONG_TERM_STRONG_YEARS_MIN,
    MR_MARKET_FEAR_DISTANCE_FROM_MA_MAX,
    MR_MARKET_FEAR_DRAWDOWN_MAX,
    MR_MARKET_GREED_DISTANCE_FROM_MA_MIN,
    MR_MARKET_GREED_DRAWDOWN_MIN,
    PATIENCE_HOLDING_PERIOD_MIN_YEARS,
    PATIENCE_TURNOVER_RATIO_MAX,
    TRADING_DAYS_PER_YEAR,
)


def fetch_company_info(ticker: str) -> Dict[str, Any]:
    info = yf.Ticker(ticker).info
    description = ""
    description_source = "yfinance"

    try:
        description = fetch_filing_section(
            ticker,
            form="10-K",
            start_markers=("item 1.", "business", "our business"),
            end_markers=("item 1a.", "risk factors"),
            max_chars=12000,
        )
        if description:
            description_source = "sec_10k_item_1"
    except Exception:
        description = ""

    if not description:
        description = info.get("longBusinessSummary", "")

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "description": description,
        "description_source": description_source,
    }


def evaluate_simplicity_with_ollama(company_name: str, description: str, model: str = DEFAULT_OLLAMA_MODEL) -> dict:
    return CircleOfCompetence().evaluate(ticker=company_name, company_name=company_name, description=description, model=model)

class CircleOfCompetence:
    """
    Heuristic: Circle of Competence
    """
    def __init__(self):
        pass

    def evaluate(self, ticker: str, company_name: str = "", description: str = "", model: str = DEFAULT_OLLAMA_MODEL) -> dict:
        company_name = company_name or ticker
        if not description:
            try:
                description = fetch_filing_section(
                    ticker,
                    form="10-K",
                    start_markers=("item 1.", "item 1. business"),
                    end_markers=("item 1a.", "risk factors"),
                    max_chars=8000
                )
            except Exception as e:
                return {"inside_circle": False, "confidence": 0, "explanation": f"Failed to fetch SEC description: {e}"}

        prompt = f"""
        You are a disciplined value investor applying Buffett's circle of competence.
        Company: {company_name}

        Business description:
        {description}

        Return JSON with exactly these keys:
        - inside_circle (boolean)
        - confidence (integer from 0 to 100)
        - explanation (string, under 40 words)
        Judge whether the business is simple enough for a generalist value investor to understand.
        """
        url = OLLAMA_GENERATE_URL
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": DEFAULT_CIRCLE_OF_COMPETENCE_TEMPERATURE}
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = json.loads(response.json().get("response", "{}"))
            confidence = int(result.get("confidence", 0) or 0)
            return {
                "inside_circle": bool(result.get("inside_circle", False)),
                "confidence": max(0, min(confidence, 100)),
                "explanation": result.get("explanation", "")
            }
        except Exception as e:
            return {
                "inside_circle": False,
                "confidence": 0,
                "explanation": f"Failed to evaluate: {e}"
            }

    def _helper_method(self):
        pass

class Inversion:
    """
    Heuristic: Inversion
    """
    def __init__(self):
        pass

    def evaluate(self, ticker: str, company_name: str = "", description: str = "", model: str = DEFAULT_OLLAMA_MODEL) -> str:
        """
        Forces the LLM to act as a skeptic and find the 3 most likely ways this company fails.
        """
        company_name = company_name or ticker
        if not description:
            try:
                description = fetch_filing_section(
                    ticker,
                    form="10-K",
                    start_markers=("item 1.", "item 1. business"),
                    end_markers=("item 1a.", "risk factors"),
                    max_chars=8000
                )
            except Exception as e:
                return f"Failed to fetch SEC description: {e}"

        prompt = f"""
        You are Charlie Munger, a brilliant, skeptical value investor. 
        I am considering investing in {company_name}. 
        
        Business Description:
        {description}
        
        Invert, always invert. I don't want to know why it will succeed. Tell me the top 3 most likely ways this investment could permanently lose money over the next 10 years. Focus on moat erosion, technological disruption, or structural flaws. 
        
        Format as a brief, punchy bulleted list. No intro, no fluff.
        """

        url = OLLAMA_GENERATE_URL
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": DEFAULT_INVERSION_TEMPERATURE} # Low temp for analytical focus
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "No response generated.")
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

    def _helper_method(self):
        pass

class MrMarket:
    """
    Heuristic: Mr. Market
    """
    def __init__(self):
        pass

    def evaluate(self, ticker: str, period: str = DEFAULT_MR_MARKET_PERIOD) -> dict:
        history = yf.Ticker(ticker).history(period=period)
        if history.empty or "Close" not in history:
            return {"ticker": ticker, "error": "No price history available"}

        close = history["Close"].dropna()
        if close.empty:
            return {"ticker": ticker, "error": "No closing prices available"}

        high_52w = float(close.max())
        current_price = float(close.iloc[-1])
        drawdown_from_high = (current_price - high_52w) / high_52w if high_52w else 0.0

        ma_200 = float(close.tail(min(len(close), 200)).mean())
        distance_from_200d_ma = (current_price - ma_200) / ma_200 if ma_200 else 0.0

        daily_returns = close.pct_change().dropna()
        annualized_volatility = float(daily_returns.std() * (TRADING_DAYS_PER_YEAR ** 0.5)) if not daily_returns.empty else 0.0

        mood = "neutral"
        if drawdown_from_high <= MR_MARKET_FEAR_DRAWDOWN_MAX and distance_from_200d_ma <= MR_MARKET_FEAR_DISTANCE_FROM_MA_MAX:
            mood = "fear"
        elif drawdown_from_high >= MR_MARKET_GREED_DRAWDOWN_MIN and distance_from_200d_ma >= MR_MARKET_GREED_DISTANCE_FROM_MA_MIN:
            mood = "greed"

        return {
            "ticker": ticker,
            "current_price": current_price,
            "high_52w": high_52w,
            "drawdown_from_high": drawdown_from_high,
            "moving_average_200d": ma_200,
            "distance_from_200d_ma": distance_from_200d_ma,
            "annualized_volatility": annualized_volatility,
            "mr_market_mood": mood
        }

    def _helper_method(self):
        pass

class LongtermOrientation:
    """
    Heuristic: Long-term Orientation
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        stock_cagr: float | None = None,
        benchmark_cagr: float | None = None,
        years: float | None = None,
        ticker: str = "",
        benchmark: str = "^GSPC",
    ) -> dict:
        if ticker and (stock_cagr is None or benchmark_cagr is None or years is None):
            price_data = fetch_price_comparison_data(ticker, years=int(years or LONG_TERM_STRONG_YEARS_MIN), benchmark=benchmark)
            stock_cagr = price_data["stock_cagr"]
            benchmark_cagr = price_data["benchmark_cagr"]
            years = price_data["period_years"]

        if stock_cagr is None or benchmark_cagr is None or years is None:
            raise ValueError("stock_cagr, benchmark_cagr, and years are required")

        if years <= 0:
            raise ValueError("years must be positive")

        excess_return = stock_cagr - benchmark_cagr
        if years >= LONG_TERM_STRONG_YEARS_MIN and excess_return > LONG_TERM_STRONG_EXCESS_RETURN_MIN:
            assessment = "strong"
        elif years >= LONG_TERM_ACCEPTABLE_YEARS_MIN and excess_return >= 0:
            assessment = "acceptable"
        else:
            assessment = "weak"

        return {
            "years": years,
            "stock_cagr": stock_cagr,
            "benchmark_cagr": benchmark_cagr,
            "excess_return": excess_return,
            "long_term_orientation": assessment
        }

    def _helper_method(self):
        pass

class MungersLatticeOfMentalModels:
    """
    Heuristic: Munger's Lattice of Mental Models
    """
    def __init__(self):
        pass

    def evaluate(self, economics_score: float, psychology_score: float, accounting_score: float) -> dict:
        average_score = (economics_score + psychology_score + accounting_score) / 3

        assessment = "narrow"
        if average_score >= LATTICE_BROAD_SCORE_MIN:
            assessment = "broad"
        elif average_score >= LATTICE_DEVELOPING_SCORE_MIN:
            assessment = "developing"

        return {
            "economics_score": economics_score,
            "psychology_score": psychology_score,
            "accounting_score": accounting_score,
            "average_score": average_score,
            "lattice_assessment": assessment
        }

    def _helper_method(self):
        pass

class IndependentThinking:
    """
    Heuristic: Independent Thinking
    """
    def __init__(self):
        pass

    def evaluate(self, thesis_differs_from_consensus: bool, evidence_strength: float, valuation_gap: float) -> dict:
        score = 0
        if thesis_differs_from_consensus:
            score += 1
        if evidence_strength >= INDEPENDENT_THINKING_EVIDENCE_MIN:
            score += 1
        if abs(valuation_gap) >= INDEPENDENT_THINKING_VALUATION_GAP_MIN:
            score += 1

        assessment = "weak"
        if score >= 3:
            assessment = "strong"
        elif score == 2:
            assessment = "moderate"

        return {
            "thesis_differs_from_consensus": thesis_differs_from_consensus,
            "evidence_strength": evidence_strength,
            "valuation_gap": valuation_gap,
            "independent_thinking_score": score,
            "independent_thinking": assessment
        }

    def _helper_method(self):
        pass

class OpportunityCostAwareness:
    """
    Heuristic: Opportunity Cost Awareness
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        candidate_return: float | None = None,
        hurdle_return: float | None = None,
        alternative_return: float | None = None,
        ticker: str = "",
        benchmark: str = "^GSPC",
        years: int = 10,
    ) -> dict:
        if ticker and (candidate_return is None or hurdle_return is None or alternative_return is None):
            price_data = fetch_price_comparison_data(ticker, years=years, benchmark=benchmark)
            if candidate_return is None:
                candidate_return = price_data["stock_cagr"]
            if alternative_return is None:
                alternative_return = price_data["benchmark_cagr"]
            if hurdle_return is None:
                hurdle_return = max(fetch_risk_free_rate() + 0.05, 0.09)

        if candidate_return is None or hurdle_return is None or alternative_return is None:
            raise ValueError("candidate_return, hurdle_return, and alternative_return are required")

        best_alternative = max(hurdle_return, alternative_return)
        spread = candidate_return - best_alternative
        clears_hurdle = candidate_return >= best_alternative

        return {
            "candidate_return": candidate_return,
            "hurdle_return": hurdle_return,
            "alternative_return": alternative_return,
            "best_alternative": best_alternative,
            "excess_return_vs_best_alternative": spread,
            "clears_opportunity_cost": clears_hurdle
        }

    def _helper_method(self):
        pass

class PatienceAsEdge:
    """
    Heuristic: Patience as Edge
    """
    def __init__(self):
        pass

    def evaluate(self, avg_holding_period_years: float, turnover_ratio: float, forced_activity: bool) -> dict:
        score = 0
        if avg_holding_period_years >= PATIENCE_HOLDING_PERIOD_MIN_YEARS:
            score += 1
        if turnover_ratio <= PATIENCE_TURNOVER_RATIO_MAX:
            score += 1
        if not forced_activity:
            score += 1

        patience = "weak"
        if score == 3:
            patience = "strong"
        elif score == 2:
            patience = "moderate"

        return {
            "avg_holding_period_years": avg_holding_period_years,
            "turnover_ratio": turnover_ratio,
            "forced_activity": forced_activity,
            "patience_score": score,
            "patience_as_edge": patience
        }

    def _helper_method(self):
        pass

def main():
    pass

if __name__ == "__main__":
    main()
