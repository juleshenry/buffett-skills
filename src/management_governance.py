import json
import math
import urllib.request
import urllib.error
import re
from typing import Dict, Any, Optional
import yfinance as yf
from transformers import pipeline
from earnings_calls import load_cached_transcript_text
from evaluator_config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_STRUCTURED_EXTRACTION_TEMPERATURE,
    call_ollama_panel_json,
    call_ollama_panel_text,
)
from evaluator_thresholds import (
    ACQUISITION_PURCHASE_MULTIPLE_MAX,
    ACQUISITION_ROIC_MIN,
    CULTURE_EMPLOYEE_TURNOVER_MAX,
    CULTURE_INSIDER_OWNERSHIP_MIN,
    CULTURE_RESTRUCTURINGS_MAX,
    GOVERNANCE_INSIDER_OWNERSHIP_MIN,
    GOVERNANCE_STRONG_SCORE_MIN,
)
from sec_data import fetch_filing_section, fetch_filing_keyword_context, fetch_latest_filing_text


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
        options = {"temperature": DEFAULT_STRUCTURED_EXTRACTION_TEMPERATURE}
        try:
            if json_format:
                return call_ollama_panel_json(prompt, model=self.ollama_model, timeout=timeout, host=self.ollama_host, options=options)
            return call_ollama_panel_text(prompt, model=self.ollama_model, timeout=timeout, host=self.ollama_host, options=options)
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def _fetch_earnings_call_transcript(self, ticker: str) -> str:
        transcript_text = load_cached_transcript_text(ticker)
        if transcript_text:
            return transcript_text
        raise RuntimeError(
            f"No cached earnings call transcripts found for {ticker}. "
            "Run src/earnings_calls.py first to populate output/earnings_calls/."
        )

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
        options = {"temperature": DEFAULT_STRUCTURED_EXTRACTION_TEMPERATURE}
        try:
            if json_format:
                return call_ollama_panel_json(prompt, model=self.ollama_model, timeout=timeout, host=self.ollama_host, options=options)
            return call_ollama_panel_text(prompt, model=self.ollama_model, timeout=timeout, host=self.ollama_host, options=options)
        except Exception as e:
            return {"error": str(e), "response": f"Error connecting to Ollama: {e}"}

    def _fetch_earnings_call_transcript(self, ticker: str) -> str:
        transcript_text = load_cached_transcript_text(ticker)
        if transcript_text:
            return transcript_text
        # Instead of crashing the whole pipeline, let's trigger the fetch directly
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"  -> [ManagementGovernance] No cached earnings call transcripts found for {ticker}. Auto-fetching now via earnings call API...")
        try:
            from earnings_calls import fetch_transcripts_for_ticker
            fetch_transcripts_for_ticker(ticker, limit=4)
            transcript_text = load_cached_transcript_text(ticker)
            if transcript_text:
                return transcript_text
        except Exception as e:
            logger.error(f"  -> [ManagementGovernance] Failed to auto-fetch transcripts: {e}")
            
        raise RuntimeError(
            f"No cached earnings call transcripts found for {ticker}. "
            "Run src/earnings_calls.py first to populate output/earnings_calls/."
        )

    def _fetch_sec_def_14a(self, ticker: str) -> str:
        return fetch_filing_section(
            ticker,
            form="DEF 14A",
            start_markers=("executive compensation", "compensation discussion and analysis", "director compensation"),
            end_markers=("security ownership", "certain relationships and related transactions", "proposal"),
            max_chars=25000,
        )


_SHARED_PIPELINES: Dict[str, Any] = {}


def _get_shared_pipeline(task: str, model: str):
    """
    Lazily creates and caches HuggingFace pipelines at module level.

    Evaluator classes are instantiated fresh per ticker in the batch pipeline
    (run.py calls `cls()` once per heuristic per ticker), so without this cache
    every one of the 500 S&P tickers would reload the same multi-hundred-MB
    model weights from disk again just to score a couple of sentences.
    """
    key = f"{task}:{model}"
    if key not in _SHARED_PIPELINES:
        _SHARED_PIPELINES[key] = pipeline(task, model=model)
    return _SHARED_PIPELINES[key]


def _embed_text(text: str) -> Optional[list[float]]:
    """
    Dependency-free sentence embedding: mean-pools FinBERT's token-level hidden
    states into a single fixed-size vector. Reuses a model already loaded
    elsewhere in this codebase instead of adding a new embedding dependency.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        extractor = _get_shared_pipeline("feature-extraction", "ProsusAI/finbert")
        token_vectors = extractor(cleaned[:2000])[0]
        if not token_vectors:
            return None
        hidden_size = len(token_vectors[0])
        pooled = [0.0] * hidden_size
        for token_vector in token_vectors:
            for i, value in enumerate(token_vector):
                pooled[i] += value
        token_count = len(token_vectors)
        return [value / token_count for value in pooled]
    except Exception:
        return None


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# Cosine similarities between sentence embeddings tend to cluster tightly
# (rarely below ~0.5 even for unrelated finance text), so a small difference in
# raw similarity is amplified before the softmax-style normalization below.
# Otherwise almost everything would land near 0.5 regardless of content.
SEMANTIC_TEMPERATURE = 12.0


def _semantic_polarity_score(text: str, positive_anchor: str, negative_anchor: str) -> Optional[float]:
    """
    Embeds `text` plus two reference anchor phrases, then converts the pair of
    cosine similarities into a single continuous [0, 1] score via relative
    (softmax-style) normalization: 1.0 means the text sits closer to
    `positive_anchor` in vector space, 0.0 means it sits closer to
    `negative_anchor`.

    This replaces literal keyword-hit counting (e.g. counting occurrences of
    "restructuring" or "layoffs") with an actual vector-space semantic
    comparison, so a hedged/negated mention like "no layoffs are planned"
    lands near the positive anchor instead of tripping a naive keyword
    counter that only knows the word "layoffs" appeared.
    """
    text_vector = _embed_text(text)
    if text_vector is None:
        return None
    positive_vector = _embed_text(positive_anchor)
    negative_vector = _embed_text(negative_anchor)
    if positive_vector is None or negative_vector is None:
        return None

    positive_similarity = _cosine_similarity(text_vector, positive_vector)
    negative_similarity = _cosine_similarity(text_vector, negative_vector)

    exp_positive = math.exp(positive_similarity * SEMANTIC_TEMPERATURE)
    exp_negative = math.exp(negative_similarity * SEMANTIC_TEMPERATURE)
    return exp_positive / (exp_positive + exp_negative)


def _normalize_signal(value: float, good_at: float, bad_at: float) -> float:
    """
    Linearly projects `value` onto a continuous [0, 1] scale where `good_at`
    maps to 1.0 and `bad_at` maps to 0.0 (clamped outside that range), instead
    of collapsing it to a hard pass/fail boolean at a single cutoff. Works
    whether "higher is better" (good_at > bad_at) or "lower is better"
    (good_at < bad_at). A value exactly at the historical pass/fail threshold
    now scores a neutral 0.5 instead of flipping between two extremes.
    """
    if good_at == bad_at:
        return 0.5
    fraction = (value - bad_at) / (good_at - bad_at)
    return max(0.0, min(1.0, fraction))


RESTRUCTURING_POSITIVE_ANCHOR = (
    "The company has a stable workforce with no significant restructuring, "
    "layoffs, or workforce reduction activity."
)
RESTRUCTURING_NEGATIVE_ANCHOR = (
    "The company announced significant restructuring charges, workforce "
    "reductions, and layoffs."
)
TURNOVER_POSITIVE_ANCHOR = (
    "The company reports low employee turnover and strong employee retention."
)
TURNOVER_NEGATIVE_ANCHOR = (
    "The company reports high employee turnover and workforce instability."
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

        # Vector normalization: each raw signal (whatever its native unit or
        # shape) gets projected onto a comparable [0, 1] "goodness" scale
        # first, using the historical thresholds as the neutral midpoint.
        normalized_signals = {}
        if employee_turnover is not None:
            normalized_signals["employee_turnover"] = _normalize_signal(
                employee_turnover, good_at=0.0, bad_at=CULTURE_EMPLOYEE_TURNOVER_MAX * 2
            )
        if insider_ownership is not None:
            normalized_signals["insider_ownership"] = _normalize_signal(
                insider_ownership, good_at=CULTURE_INSIDER_OWNERSHIP_MIN * 2, bad_at=0.0
            )
        if restructurings_per_5y is not None:
            normalized_signals["restructurings_per_5y"] = _normalize_signal(
                restructurings_per_5y, good_at=0.0, bad_at=CULTURE_RESTRUCTURINGS_MAX * 3
            )

        # Require at least ONE signal instead of two, since explicit culture metrics are rare in 10Ks
        if len(normalized_signals) < 1:
            return {
                "ticker": ticker,
                "applicable": False,
                "reason": "Insufficient explicit culture evidence found; need at least one culture proxy (e.g. insider ownership).",
            }

        # Combine the normalized vector by averaging its components, instead
        # of counting how many raw values individually clear a cutoff.
        normalized_vector = list(normalized_signals.values())
        culture_score_unit = sum(normalized_vector) / len(normalized_vector)
        culture_score = round(culture_score_unit * 3, 2)

        culture = "weak"
        if culture_score_unit >= 0.66:
            culture = "strong"
        elif culture_score_unit >= 0.33:
            culture = "stable"

        return {
            "employee_turnover": employee_turnover,
            "insider_ownership": insider_ownership,
            "restructurings_per_5y": restructurings_per_5y,
            "signals_available": len(normalized_signals),
            "signal_confidence": round(len(normalized_signals) / 3, 2),
            "culture_score": culture_score,
            "culture_assessment": culture
        }

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass

    def _fetch_culture_inputs(self, ticker: str) -> dict:
        info = yf.Ticker(ticker).info or {}
        employee_commentary = self._fetch_employee_commentary(ticker)
        restructuring_commentary = self._fetch_restructuring_commentary(ticker)
        return {
            "insider_ownership": float(info.get("heldPercentInsiders")) if info.get("heldPercentInsiders") is not None else None,
            "employee_turnover": self._estimate_employee_turnover(employee_commentary),
            "restructurings_per_5y": self._estimate_restructuring_severity(restructuring_commentary),
        }

    def _fetch_employee_commentary(self, ticker: str) -> str:
        try:
            return fetch_filing_section(
                ticker,
                form="10-K",
                start_markers=("item 1.", "business", "human capital", "employees"),
                end_markers=("item 1a.", "risk factors"),
                max_chars=16000,
            )
        except Exception:
            return ""

    def _fetch_restructuring_commentary(self, ticker: str) -> str:
        keyword_sets = (
            ("10-K", ("restructuring", "workforce reduction", "layoff", "reorganization", "severance")),
            ("10-Q", ("restructuring", "workforce reduction", "layoff", "reorganization", "severance")),
            ("8-K", ("restructuring", "workforce reduction", "layoff", "reorganization", "severance")),
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

    def _estimate_employee_turnover(self, commentary: str) -> float | None:
        if not commentary:
            return None
        text = commentary.lower()
        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:employee\s+)?turnover", text)
        if percent_match:
            return float(percent_match.group(1)) / 100.0

        # No explicit disclosed rate. Fall back to a semantic (vector-space)
        # read of the commentary instead of asking a generative LLM to guess
        # a bare number -- the same qualitative text now maps deterministically
        # onto a continuous turnover estimate via embedding similarity.
        stability_score = _semantic_polarity_score(commentary, TURNOVER_POSITIVE_ANCHOR, TURNOVER_NEGATIVE_ANCHOR)
        if stability_score is None:
            return None
        return round((1.0 - stability_score) * CULTURE_EMPLOYEE_TURNOVER_MAX * 2, 4)

    @staticmethod
    def _looks_like_filing_noise(text: str) -> bool:
        noise_markers = ("us-gaap:", "member", "2025-12-31", "2024-12-31", "2023-12-31")
        return sum(marker in text for marker in noise_markers) >= 2

    def _estimate_restructuring_severity(self, commentary: str) -> float | None:
        text = (commentary or "").strip()
        if not text:
            return None
        if self._looks_like_filing_noise(text.lower()):
            return None

        # Semantic (vector-space) severity read instead of literal keyword-hit
        # counting: negated/hedged mentions ("no restructuring is planned")
        # land near the positive anchor rather than tripping a naive counter
        # of words like "restructuring" or "layoffs".
        stability_score = _semantic_polarity_score(text, RESTRUCTURING_POSITIVE_ANCHOR, RESTRUCTURING_NEGATIVE_ANCHOR)
        if stability_score is None:
            return None
        return round((1.0 - stability_score) * CULTURE_RESTRUCTURINGS_MAX * 3, 4)

class AcquisitionLogicAcquisitionCriteria:
    """
    Heuristic: Acquisition Logic (Acquisition Criteria)
    """
    def __init__(self):
        pass

    def evaluate(self, purchase_multiple: float | None = None, return_on_invested_capital: float | None = None, debt_funded: bool | None = None, ticker: str = "") -> dict:
        fetched_inputs = None
        if ticker and (purchase_multiple is None or return_on_invested_capital is None or debt_funded is None):
            acquisition_inputs = self._fetch_acquisition_inputs(ticker)
            fetched_inputs = acquisition_inputs
            if purchase_multiple is None:
                purchase_multiple = acquisition_inputs["purchase_multiple"]
            if return_on_invested_capital is None:
                return_on_invested_capital = acquisition_inputs["return_on_invested_capital"]
            if debt_funded is None:
                debt_funded = acquisition_inputs["debt_funded"]

        if fetched_inputs and not fetched_inputs.get("has_acquisition_evidence"):
            return {
                "ticker": ticker,
                "applicable": False,
                "reason": "No recent acquisition evidence found in SEC filings.",
            }
             
        # Instead of failing on missing exact figures, default them conservatively so the qualitative boolean (debt_funded) can still carry weight
        if purchase_multiple is None:
            purchase_multiple = 15.0 # Neutral penalty
        if return_on_invested_capital is None:
            return_on_invested_capital = 0.05 # Low neutral return
        if debt_funded is None:
            debt_funded = True # Assume debt if unknown to be conservative

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
        purchase_multiple = self._parse_purchase_multiple(commentary)
        debt_funded = self._parse_debt_funding(commentary)
        has_acquisition_evidence = self._has_material_acquisition_evidence(
            commentary,
            purchase_multiple=purchase_multiple,
            debt_funded=debt_funded,
        )
        return {
            "has_acquisition_evidence": has_acquisition_evidence,
            "purchase_multiple": purchase_multiple,
            "return_on_invested_capital": self._fetch_return_on_invested_capital(ticker),
            "debt_funded": debt_funded,
        }

    def _has_material_acquisition_evidence(
        self,
        commentary: str,
        purchase_multiple: float | None = None,
        debt_funded: bool = False,
    ) -> bool:
        text = (commentary or "").lower()
        if not text.strip():
            return False

        if purchase_multiple is not None or debt_funded:
            return True

        explicit_deal_markers = (
            "merger agreement",
            "business combination",
            "purchase price of",
            "acquisition closed",
            "definitive agreement",
        )
        return any(marker in text for marker in explicit_deal_markers)

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
        return None

    def _parse_debt_funding(self, commentary: str) -> bool:
        text = commentary.lower()
        debt_terms = ("debt financing", "term loan", "bridge facility", "notes offering", "borrowings")
        return any(term in text for term in debt_terms)

    def _fetch_return_on_invested_capital(self, ticker: str) -> float | None:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        earnings = _get_statement_value(income_stmt, ("Operating Income", "Net Income"), 0)
        total_debt = _get_statement_value(balance_sheet, ("Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"), 0)
        equity = _get_statement_value(balance_sheet, ("Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"), 0)
        cash = _get_statement_value(balance_sheet, ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments"), 0)

        if None in (earnings, total_debt, equity):
            return None

        invested_capital = total_debt + equity - (cash or 0.0)
        if invested_capital <= 0:
            return None

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
