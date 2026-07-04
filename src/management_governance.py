import json
import urllib.request
import urllib.error
import re
from typing import Dict, Any, Optional
import yfinance as yf
from transformers import pipeline
from evaluator_config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, call_ollama_panel_json, call_ollama_panel_text
from evaluator_thresholds import (
    ACQUISITION_PURCHASE_MULTIPLE_MAX,
    ACQUISITION_ROIC_MIN,
    CULTURE_EMPLOYEE_TURNOVER_MAX,
    CULTURE_INSIDER_OWNERSHIP_MIN,
    CULTURE_RESTRUCTURINGS_MAX,
    GOVERNANCE_INSIDER_OWNERSHIP_MIN,
    GOVERNANCE_STRONG_SCORE_MIN,
)
from sec_data import fetch_filing_keyword_context, fetch_filing_section, fetch_latest_filing_text


def analyze_management_governance(ticker: str, transcript: Optional[str] = None, proxy_statement: Optional[str] = None) -> str:
    return ManagementEvaluation().evaluate(ticker, transcript=transcript, proxy_stmt=proxy_statement)


def _get_statement_value(statement, labels: tuple[str, ...], column_index: int = 0):
    if statement is None or getattr(statement, "empty", True) or statement.shape[1] <= column_index:
        return None

    for label in labels:
        if label in statement.index:
            value = statement.loc[label].iloc[column_index]
            if value is not None:
                return float(value)
    return None

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
        try:
            if json_format:
                return call_ollama_panel_json(prompt, model=self.ollama_model, timeout=timeout, host=self.ollama_host)
            return call_ollama_panel_text(prompt, model=self.ollama_model, timeout=timeout, host=self.ollama_host)
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
        try:
            if json_format:
                return call_ollama_panel_json(prompt, model=self.ollama_model, timeout=timeout, host=self.ollama_host)
            return call_ollama_panel_text(prompt, model=self.ollama_model, timeout=timeout, host=self.ollama_host)
        except Exception as e:
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

    def evaluate(self, employee_turnover: float | None = None, insider_ownership: float | None = None, restructurings_per_5y: int | None = None, ticker: str = "") -> dict:
        if ticker and (employee_turnover is None or insider_ownership is None or restructurings_per_5y is None):
            culture_inputs = self._fetch_culture_inputs(ticker)
            if insider_ownership is None:
                insider_ownership = culture_inputs["insider_ownership"]
            if employee_turnover is None:
                employee_turnover = culture_inputs["employee_turnover"]
            if restructurings_per_5y is None:
                restructurings_per_5y = culture_inputs["restructurings_per_5y"]
            
        if employee_turnover is None or insider_ownership is None or restructurings_per_5y is None:
            raise ValueError("All metrics must be provided or fetchable via ticker")

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

    def _fetch_culture_inputs(self, ticker: str) -> dict:
        info = yf.Ticker(ticker).info or {}
        commentary = self._fetch_culture_commentary(ticker)
        return {
            "insider_ownership": float(info.get("heldPercentInsiders") or 0.0),
            "employee_turnover": self._estimate_employee_turnover(commentary),
            "restructurings_per_5y": self._count_restructuring_signals(commentary),
        }

    def _fetch_culture_commentary(self, ticker: str) -> str:
        keyword_sets = (
            ("10-K", ("employees", "employee retention", "turnover", "restructuring", "workforce reduction", "layoff")),
            ("10-Q", ("restructuring", "workforce reduction", "layoff", "retention")),
            ("8-K", ("restructuring", "workforce reduction", "layoff")),
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

    def _estimate_employee_turnover(self, commentary: str) -> float:
        text = commentary.lower()
        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:employee\s+)?turnover", text)
        if percent_match:
            return float(percent_match.group(1)) / 100.0

        if any(term in text for term in ("high turnover", "elevated attrition", "retention challenges")):
            return 0.20
        if any(term in text for term in ("low turnover", "strong retention", "retention remained strong")):
            return 0.10
        return 0.15

    def _count_restructuring_signals(self, commentary: str) -> int:
        text = commentary.lower()
        keywords = ("restructuring", "workforce reduction", "layoff", "reorganization", "severance")
        matches = sum(text.count(keyword) for keyword in keywords)
        return min(matches, 5)

class AcquisitionLogicAcquisitionCriteria:
    """
    Heuristic: Acquisition Logic (Acquisition Criteria)
    """
    def __init__(self):
        pass

    def evaluate(self, purchase_multiple: float | None = None, return_on_invested_capital: float | None = None, debt_funded: bool | None = None, ticker: str = "") -> dict:
        if ticker and (purchase_multiple is None or return_on_invested_capital is None or debt_funded is None):
            acquisition_inputs = self._fetch_acquisition_inputs(ticker)
            if purchase_multiple is None:
                purchase_multiple = acquisition_inputs["purchase_multiple"]
            if return_on_invested_capital is None:
                return_on_invested_capital = acquisition_inputs["return_on_invested_capital"]
            if debt_funded is None:
                debt_funded = acquisition_inputs["debt_funded"]
            
        if purchase_multiple is None or return_on_invested_capital is None or debt_funded is None:
            raise ValueError("All metrics must be provided or fetchable via ticker")

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

    def _fetch_acquisition_inputs(self, ticker: str) -> dict:
        commentary = self._fetch_acquisition_commentary(ticker)
        return {
            "purchase_multiple": self._parse_purchase_multiple(commentary),
            "return_on_invested_capital": self._fetch_return_on_invested_capital(ticker),
            "debt_funded": self._parse_debt_funding(commentary),
        }

    def _fetch_acquisition_commentary(self, ticker: str) -> str:
        keyword_sets = (
            ("8-K", ("acquisition", "merger agreement", "purchase price", "financing", "term loan")),
            ("10-K", ("acquisition", "business combination", "purchase price", "acquired", "financing")),
            ("10-Q", ("acquisition", "business combination", "purchase price", "acquired", "financing")),
        )

        for form, keywords in keyword_sets:
            try:
                commentary = fetch_filing_keyword_context(
                    ticker,
                    form=form,
                    keywords=keywords,
                    context_chars=1800,
                    max_matches=3,
                    max_chars=10000,
                )
                if commentary:
                    return commentary
            except Exception:
                continue

        return ""

    def _parse_purchase_multiple(self, commentary: str) -> float:
        text = commentary.lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(?:ebitda|earnings)", text)
        if match:
            return float(match.group(1))

        if "disciplined acquisition" in text or "value creation" in text:
            return float(ACQUISITION_PURCHASE_MULTIPLE_MAX)

        return float(ACQUISITION_PURCHASE_MULTIPLE_MAX + 2)

    def _parse_debt_funding(self, commentary: str) -> bool:
        text = commentary.lower()
        debt_terms = ("debt financing", "term loan", "bridge facility", "notes offering", "borrowings")
        return any(term in text for term in debt_terms)

    def _fetch_return_on_invested_capital(self, ticker: str) -> float:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        earnings = _get_statement_value(income_stmt, ("Operating Income", "Net Income"), 0)
        total_debt = _get_statement_value(balance_sheet, ("Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"), 0)
        equity = _get_statement_value(balance_sheet, ("Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"), 0)
        cash = _get_statement_value(balance_sheet, ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments"), 0)

        if None in (earnings, total_debt, equity):
            return 0.0

        invested_capital = total_debt + equity - (cash or 0.0)
        if invested_capital <= 0:
            return 0.0

        return float(earnings / invested_capital)

class CorporateGovernanceAndShareholderOrientation:
    """
    Heuristic: Corporate Governance and Shareholder Orientation
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        insider_ownership: Optional[float] = None,
        roic_linked_pay: Optional[bool] = None,
        dual_class_structure: Optional[bool] = None,
        buybacks_below_intrinsic_value: Optional[bool] = None,
        ticker: str = "",
    ) -> dict:
        if ticker and any(value is None for value in (insider_ownership, roic_linked_pay, dual_class_structure, buybacks_below_intrinsic_value)):
            proxy_text = self._fetch_proxy_text(ticker)
            proxy_flags = self._parse_proxy_governance(proxy_text)

            if insider_ownership is None:
                insider_ownership = self._fetch_insider_ownership(ticker)
            if roic_linked_pay is None:
                roic_linked_pay = proxy_flags["roic_linked_pay"]
            if dual_class_structure is None:
                dual_class_structure = proxy_flags["dual_class_structure"]
            if buybacks_below_intrinsic_value is None:
                buybacks_below_intrinsic_value = self._fetch_buyback_orientation(ticker)

        if insider_ownership is None or roic_linked_pay is None or dual_class_structure is None or buybacks_below_intrinsic_value is None:
            raise ValueError(
                "insider_ownership, roic_linked_pay, dual_class_structure, and buybacks_below_intrinsic_value are required"
            )

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

    def _fetch_insider_ownership(self, ticker: str) -> float:
        info = yf.Ticker(ticker).info or {}
        return float(info.get("heldPercentInsiders") or 0.0)

    def _fetch_proxy_text(self, ticker: str) -> str:
        return fetch_latest_filing_text(ticker, form="DEF 14A")

    def _parse_proxy_governance(self, proxy_text: str) -> dict:
        lower_text = proxy_text.lower()
        compensation_terms = ("compensation", "bonus", "incentive", "incentives")
        performance_terms = ("return on invested capital", "roic", "return on equity", "roe")

        roic_linked_pay = any(term in lower_text for term in compensation_terms) and any(
            term in lower_text for term in performance_terms
        )

        dual_class_structure = "dual-class" in lower_text or (
            ("class a common stock" in lower_text or "class a shares" in lower_text)
            and ("class b common stock" in lower_text or "class b shares" in lower_text)
        )

        return {
            "roic_linked_pay": roic_linked_pay,
            "dual_class_structure": dual_class_structure,
        }

    def _fetch_buyback_orientation(self, ticker: str) -> bool:
        from valuation_capital import fetch_management_commentary

        commentary = fetch_management_commentary(ticker)
        lower_text = commentary.lower()
        mentions_repurchase = any(
            term in lower_text
            for term in ("share repurchase", "share repurchases", "stock repurchase", "buyback", "buybacks")
        )
        mentions_intrinsic_value = any(
            term in lower_text
            for term in ("intrinsic value", "below intrinsic value", "undervalued")
        )
        return mentions_repurchase and mentions_intrinsic_value

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
