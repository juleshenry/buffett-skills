from cache_utils import disk_cache
import yfinance as yf
import requests
import json
from typing import Dict, Any
import pandas as pd
from investment_philosophy import fetch_price_comparison_data
from sec_data import fetch_filing_section
from valuation_capital import fetch_risk_free_rate
from evaluator_config import (
    call_ollama_panel_json,
    call_ollama_panel_text,
    DEFAULT_CIRCLE_OF_COMPETENCE_TEMPERATURE,
    DEFAULT_INVERSION_TEMPERATURE,
    DEFAULT_MR_MARKET_PERIOD,
    DEFAULT_OLLAMA_MODEL,
)
from evaluator_thresholds import (
    MR_MARKET_FEAR_DISTANCE_FROM_MA_MAX,
    MR_MARKET_FEAR_DRAWDOWN_MAX,
    MR_MARKET_GREED_DISTANCE_FROM_MA_MIN,
    MR_MARKET_GREED_DRAWDOWN_MIN,
    TRADING_DAYS_PER_YEAR,
)
import re


def _make_display_description(sec_description: str, fallback_summary: str) -> str:
    text_source = fallback_summary or sec_description or ""
    text = re.sub(r"\s+", " ", text_source.strip())
    if not text:
        return ""

    text = re.sub(r"^item\s+1\s*[\.-–:]\s*business\s*", "", text, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    filtered_sentences = []

    boilerplate_patterns = (
        r"^as used in this annual report",
        r"^the following discussion should be read in conjunction",
        r"^references to the",
    )

    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(re.match(pattern, lowered) for pattern in boilerplate_patterns):
            continue
        filtered_sentences.append(stripped)
        if len(" ".join(filtered_sentences)) >= 420 or len(filtered_sentences) >= 3:
            break

    summary = " ".join(filtered_sentences) if filtered_sentences else text
    if len(summary) <= 420:
        return summary

    trimmed = summary[:420].rsplit(" ", 1)[0].strip()
    return trimmed or summary[:420].strip()


def _fetch_company_info_cached(ticker: str) -> Dict[str, Any]:
    info = yf.Ticker(ticker).info
    description = ""
    description_source = "yfinance"
    fallback_summary = info.get("longBusinessSummary", "")

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
        description = fallback_summary

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "description": description,
        "display_description": _make_display_description(description, fallback_summary),
        "description_source": description_source,
    }


def fetch_company_info(ticker: str) -> Dict[str, Any]:
    return _fetch_company_info_cached(ticker)


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
        aggregated = call_ollama_panel_json(
            prompt,
            model=model,
            options={"temperature": DEFAULT_CIRCLE_OF_COMPETENCE_TEMPERATURE},
            aggregator=self._aggregate_circle_results,
        )
        if aggregated:
            return aggregated

        return {
            "inside_circle": False,
            "confidence": 0,
            "explanation": "Failed to evaluate: no valid panel responses"
        }

    def _aggregate_circle_results(self, results: list[dict]) -> dict:
        valid_results = [result for result in results if result]
        if not valid_results:
            return {}

        inside_votes = sum(1 for result in valid_results if bool(result.get("inside_circle", False)))
        inside_circle = inside_votes >= ((len(valid_results) // 2) + 1)
        confidences = [int(result.get("confidence", 0) or 0) for result in valid_results]
        explanations = [
            result.get("explanation", "")
            for result in valid_results
            if bool(result.get("inside_circle", False)) == inside_circle and result.get("explanation")
        ]
        return {
            "inside_circle": inside_circle,
            "confidence": max(0, min(round(sum(confidences) / len(confidences)), 100)),
            "explanation": explanations[0] if explanations else valid_results[0].get("explanation", ""),
            "panel_models": [result.get("_panel_model") for result in valid_results if result.get("_panel_model")],
            "panel_vote_split": {
                "inside_circle": inside_votes,
                "outside_circle": len(valid_results) - inside_votes,
            },
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

        result = call_ollama_panel_text(
            prompt,
            model=model,
            options={"temperature": DEFAULT_INVERSION_TEMPERATURE},
        )
        return result.get("response", "No response generated.")

    def _helper_method(self):
        pass

class MrMarket:
    """
    Heuristic: Mr. Market
    """
    def __init__(self):
        pass

    @staticmethod
    def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
        return max(low, min(value, high))

    def _compute_contrarian_score(
        self,
        drawdown_from_high: float,
        distance_from_200d_ma: float,
        annualized_volatility: float,
    ) -> tuple[float, float]:
        # More negative drawdown and distance imply more fear and therefore a more positive contrarian setup.
        drawdown_component = self._clamp((-drawdown_from_high - 0.05) / 0.25)
        distance_component = self._clamp((-distance_from_200d_ma) / 0.15)
        volatility_component = self._clamp((annualized_volatility - 0.20) / 0.25, 0.0, 1.0)

        raw_score = (drawdown_component * 0.5) + (distance_component * 0.35) + (volatility_component * 0.15)

        # Mild premium conditions should lean negative rather than snap to neutral.
        if drawdown_from_high > -0.03 and distance_from_200d_ma > 0.03:
            raw_score -= min((drawdown_from_high + distance_from_200d_ma) * 4.0, 0.6)

        contrarian_score = self._clamp(raw_score)
        signal_strength = abs(contrarian_score)
        return contrarian_score, signal_strength

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

        contrarian_score, signal_strength = self._compute_contrarian_score(
            drawdown_from_high=drawdown_from_high,
            distance_from_200d_ma=distance_from_200d_ma,
            annualized_volatility=annualized_volatility,
        )

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
            "contrarian_score": contrarian_score,
            "signal_strength": signal_strength,
            "mr_market_mood": mood
        }

    def _helper_method(self):
        pass

def main():
    pass

if __name__ == "__main__":
    main()
