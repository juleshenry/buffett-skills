import json
import urllib.request
import urllib.error
import re
from typing import Dict, Any, Optional
from transformers import pipeline
from evaluator_config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL
from evaluator_thresholds import (
    ACQUISITION_PURCHASE_MULTIPLE_MAX,
    ACQUISITION_ROIC_MIN,
    CULTURE_EMPLOYEE_TURNOVER_MAX,
    CULTURE_INSIDER_OWNERSHIP_MIN,
    CULTURE_RESTRUCTURINGS_MAX,
    GOVERNANCE_INSIDER_OWNERSHIP_MIN,
    GOVERNANCE_STRONG_SCORE_MIN,
)
from sec_data import fetch_filing_keyword_context, fetch_filing_section


def analyze_management_governance(ticker: str, transcript: Optional[str] = None, proxy_statement: Optional[str] = None) -> str:
    return ManagementEvaluation().evaluate(ticker, transcript=transcript, proxy_stmt=proxy_statement)

class ManagementGovernanceAnalyzer:
    """
    Analyzer class to evaluate corporate management and governance.
    Models are initialized once upon instantiation to prevent memory overhead
    during repeated function calls.
    """
    def __init__(self, ollama_model: str = DEFAULT_OLLAMA_MODEL, ollama_host: str = DEFAULT_OLLAMA_HOST):
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        
        print("Initializing NLP models... (This may take a moment)")
        # Load models once. 
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='ProsusAI/finbert')
        self.zero_shot_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        print("Models loaded successfully.")

    def _call_ollama(self, prompt: str, json_format: bool = True, timeout: int = 45) -> Dict[str, Any]:
        """Calls local Ollama API using standard library with timeouts and robust JSON parsing."""
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        if json_format:
            payload["format"] = "json"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                response_text = result.get("response", "")
                
                if json_format:
                    # Robust JSON cleanup: Strip markdown formatting if the model hallucinates it
                    cleaned_text = re.sub(r"^```json\s*", "", response_text)
                    cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()
                    try:
                        return json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        return {"error": "Failed to parse Ollama JSON output", "raw_output": response_text}
                
                return {"response": response_text}
                
        except urllib.error.URLError as e:
            return {"error": f"Network/Timeout error communicating with Ollama: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def _fetch_earnings_call_transcript(self, ticker: str) -> str:
        keyword_sets = (
            ("8-K", ("conference call", "earnings call", "prepared remarks", "question-and-answer", "results of operations and financial condition")),
            ("10-Q", ("results of operations", "liquidity and capital resources", "capital allocation")),
            ("10-K", ("results of operations", "liquidity and capital resources", "capital allocation")),
        )

        for form, keywords in keyword_sets:
            try:
                commentary = fetch_filing_keyword_context(
                    ticker,
                    form=form,
                    keywords=keywords,
                    context_chars=1600,
                    max_matches=3,
                    max_chars=9000,
                )
                if commentary:
                    return commentary
            except Exception:
                continue

        raise RuntimeError(f"No SEC management commentary available for {ticker}.")

    def _fetch_sec_def_14a(self, ticker: str) -> str:
        return fetch_filing_section(
            ticker,
            form="DEF 14A",
            start_markers=("executive compensation", "compensation discussion and analysis", "director compensation"),
            end_markers=("security ownership", "certain relationships and related transactions", "proposal"),
            max_chars=25000,
        )

    def analyze(self, ticker: str, transcript: Optional[str] = None, proxy_statement: Optional[str] = None) -> str:
        """Runs the management governance analysis workflow and returns a JSON string."""
        
        # 1. Fetch raw data
        transcript = transcript or self._fetch_earnings_call_transcript(ticker)
        proxy_statement = proxy_statement or self._fetch_sec_def_14a(ticker)

        # 2. HF Task 1: Earnings Call Sentiment & Candor
        # CRITICAL: Added truncation=True to prevent crashes on long transcripts
        sentiment_result = self.sentiment_analyzer(transcript, truncation=True, max_length=512)

        candidate_labels = [
            "blaming macroeconomic factors", 
            "blaming supply chain", 
            "taking accountability", 
            "admitting mistakes"
        ]
        candor_result = self.zero_shot_classifier(
            transcript, 
            candidate_labels=candidate_labels, 
            truncation=True, 
            max_length=512
        )

        # 3. Ollama Task 2: Proxy Parser (Incentive Structure)
        proxy_prompt = f"""
        Read the following "Executive Compensation" section of a DEF 14A proxy statement.
        Are executive bonuses tied to ROIC/Return on Equity, or are they tied purely to Revenue Growth/Adjusted EBITDA? 
        
        Return a JSON object strictly matching this structure:
        {{
            "incentive_alignment": "strong" | "moderate" | "weak",
            "reasoning": "brief 1-2 sentence explanation"
        }}

        DEF 14A Excerpt:
        {proxy_statement}
        """
        proxy_result = self._call_ollama(proxy_prompt, json_format=True)

        # 4. Synthesize Results
        final_analysis = {
            "ticker": ticker,
            "management_governance": {
                "earnings_call_analysis": {
                    "sentiment": sentiment_result[0] if isinstance(sentiment_result, list) else sentiment_result,
                    "candor_classification": {
                        "top_label": candor_result["labels"][0],
                        "top_score": round(candor_result["scores"][0], 4),
                        "all_scores": dict(zip(candor_result["labels"], [round(s, 4) for s in candor_result["scores"]]))
                    }
                },
                "incentive_structure": proxy_result
            }
        }
        
        return json.dumps(final_analysis, indent=2)

if __name__ == "__main__":
    # Initialize the analyzer once (loads models into memory)
    analyzer = ManagementGovernanceAnalyzer()
    
    # Run analysis (much faster on subsequent runs)
    print("\n--- Analyzing AAPL ---")
    result_json = analyzer.analyze("AAPL")
    print(result_json)

class ManagementEvaluation:
    """
    Heuristic: Management Evaluation
    """
    def __init__(self, ollama_model: str = DEFAULT_OLLAMA_MODEL, ollama_host: str = DEFAULT_OLLAMA_HOST):
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host

    def evaluate(self, ticker: str, transcript: Optional[str] = None, proxy_stmt: Optional[str] = None) -> str:
        transcript = transcript or self._fetch_earnings_call_transcript(ticker)
        proxy_stmt = proxy_stmt or self._fetch_sec_def_14a(ticker)

        prompt = f"""
        You are Warren Buffett analyzing the management of {ticker}. 
        Assess management's candor, capital allocation skills, and shareholder orientation based on these excerpts.
        
        Earnings Call Transcript Excerpt:
        {transcript}
        
        SEC DEF 14A (Proxy Statement) Excerpt - Executive Comp:
        {proxy_stmt}
        
        Provide a concise, 3-paragraph evaluation. Focus on:
        1. Honesty in reporting (do they admit mistakes?).
        2. Rationality in capital allocation (dividends vs buybacks vs acquisitions).
        3. Alignment of executive compensation with shareholder returns.
        """

        result = self._call_ollama(prompt, json_format=False)
        return result.get("response", "Analysis failed.")

    def _call_ollama(self, prompt: str, json_format: bool = True, timeout: int = 45) -> dict:
        import requests
        url = f"{self.ollama_host}/api/generate"
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        if json_format:
            payload["format"] = "json"
            
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "response": f"Error connecting to Ollama: {e}"}

    def _fetch_earnings_call_transcript(self, ticker: str) -> str:
        return ManagementGovernanceAnalyzer(
            ollama_model=self.ollama_model,
            ollama_host=self.ollama_host,
        )._fetch_earnings_call_transcript(ticker)

    def _fetch_sec_def_14a(self, ticker: str) -> str:
        return fetch_filing_section(
            ticker,
            form="DEF 14A",
            start_markers=("executive compensation", "compensation discussion and analysis", "director compensation"),
            end_markers=("security ownership", "certain relationships and related transactions", "proposal"),
            max_chars=25000,
        )


class CorporateCulture:
    """
    Heuristic: Corporate Culture
    """
    def __init__(self):
        pass

    def evaluate(self, employee_turnover: float, insider_ownership: float, restructurings_per_5y: int) -> dict:
        score = 0
        if employee_turnover <= CULTURE_EMPLOYEE_TURNOVER_MAX:
            score += 1
        if insider_ownership >= CULTURE_INSIDER_OWNERSHIP_MIN:
            score += 1
        if restructurings_per_5y <= CULTURE_RESTRUCTURINGS_MAX:
            score += 1

        culture = "weak"
        if score == 3:
            culture = "strong"
        elif score == 2:
            culture = "stable"

        return {
            "employee_turnover": employee_turnover,
            "insider_ownership": insider_ownership,
            "restructurings_per_5y": restructurings_per_5y,
            "culture_score": score,
            "culture_assessment": culture
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class AcquisitionLogicAcquisitionCriteria:
    """
    Heuristic: Acquisition Logic (Acquisition Criteria)
    """
    def __init__(self):
        pass

    def evaluate(self, purchase_multiple: float, return_on_invested_capital: float, debt_funded: bool) -> dict:
        score = 0
        if purchase_multiple <= ACQUISITION_PURCHASE_MULTIPLE_MAX:
            score += 1
        if return_on_invested_capital >= ACQUISITION_ROIC_MIN:
            score += 1
        if not debt_funded:
            score += 1

        discipline = "weak"
        if score == 3:
            discipline = "disciplined"
        elif score == 2:
            discipline = "mixed"

        return {
            "purchase_multiple": purchase_multiple,
            "return_on_invested_capital": return_on_invested_capital,
            "debt_funded": debt_funded,
            "acquisition_score": score,
            "acquisition_discipline": discipline
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

class CorporateGovernanceAndShareholderOrientation:
    """
    Heuristic: Corporate Governance and Shareholder Orientation
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        insider_ownership: float,
        roic_linked_pay: bool,
        dual_class_structure: bool,
        buybacks_below_intrinsic_value: bool,
    ) -> dict:
        score = 0
        if insider_ownership >= GOVERNANCE_INSIDER_OWNERSHIP_MIN:
            score += 1
        if roic_linked_pay:
            score += 1
        if not dual_class_structure:
            score += 1
        if buybacks_below_intrinsic_value:
            score += 1

        orientation = "weak"
        if score >= GOVERNANCE_STRONG_SCORE_MIN:
            orientation = "strong"
        elif score == 2:
            orientation = "moderate"

        return {
            "insider_ownership": insider_ownership,
            "roic_linked_pay": roic_linked_pay,
            "dual_class_structure": dual_class_structure,
            "buybacks_below_intrinsic_value": buybacks_below_intrinsic_value,
            "governance_score": score,
            "shareholder_orientation": orientation
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
