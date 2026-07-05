import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)


CATEGORY_ALIASES = {
    "thinking_frameworks": "thinking_frameworks",
    "business_moat": "moat",
    "management_governance": "management",
    "financial_metrics": "financial_quality",
    "valuation_capital": "valuation_capital",
    "risk_behavior": "risk",
    "industry_playbooks": "industry",
}


CATEGORY_WEIGHTS = {
    "financial_quality": 0.22,
    "moat": 0.22,
    "risk": 0.18,
    "valuation_capital": 0.18,
    "management": 0.14,
    "thinking_frameworks": 0.06,
}


BUSINESS_QUALITY_WEIGHTS = {
    "financial_quality": 0.28,
    "moat": 0.28,
    "risk": 0.20,
    "management": 0.18,
    "thinking_frameworks": 0.06,
}


OPPORTUNITY_WEIGHTS = {
    "valuation_capital": 0.50,
    "risk": 0.15,
    "moat": 0.12,
    "financial_quality": 0.10,
    "thinking_frameworks": 0.08,
    "management": 0.05,
}


COMPOSITE_SCORE_WEIGHTS = {
    "business_quality_score": 0.55,
    "opportunity_score": 0.45,
}


MIN_PEER_GROUP_SIZE = 5


PEER_SENSITIVE_METRICS = {
    "profit_margin",
    "cash_conversion",
    "free_cash_flow",
    "owner_earnings",
    "net_income",
    "operating_cash_flow",
    "depreciation_amortization_estimate",
    "maintenance_capex_estimate",
    "total_capex",
    "gross_margin",
    "gross_margin_trend",
    "market_share_trend",
    "return_on_capital",
    "return_on_tangible_assets",
    "recurring_revenue_ratio",
    "capital_intensity",
    "margin_of_safety",
    "fcf_to_debt",
    "dividend_payout_ratio",
    "retained_earnings_ratio",
    "stock_cagr",
    "benchmark_cagr",
    "tracking_error",
    "annualized_volatility",
    "debt_to_equity",
    "combined_ratio",
    "subscription_revenue_ratio",
    "ad_revenue_ratio",
    "churn_rate",
    "regulated_asset_ratio",
    "allowed_roe",
    "operating_ratio",
    "maintenance_capex_ratio",
    "net_revenue_retention",
    "stock_comp_ratio",
    "commodity_exposure",
    "leverage_ratio",
    "pricing_power_score",
    "purchase_multiple",
    "employee_turnover",
    "insider_ownership",
    "restructurings_per_5y",
}


SCORE_MAX_BY_KEY = {
    "durability_score": 3.0,
    "culture_score": 3.0,
    "acquisition_score": 3.0,
    "governance_score": 4.0,
    "consumer_brand_score": 3.0,
    "media_score": 3.0,
    "utility_score": 3.0,
    "railway_score": 3.0,
    "technology_score": 3.0,
    "confidence_score": 100.0,
}


PEER_NUMERIC_RULES = {
    "profit_margin": "higher",
    "cash_conversion": "higher",
    "free_cash_flow": "higher",
    "owner_earnings": "higher",
    "net_income": "higher",
    "operating_cash_flow": "higher",
    "depreciation_amortization_estimate": "higher",
    "maintenance_capex_estimate": "lower",
    "total_capex": "lower",
    "gross_margin": "higher",
    "gross_margin_trend": "higher",
    "market_share_trend": "higher",
    "return_on_capital": "higher",
    "return_on_tangible_assets": "higher",
    "recurring_revenue_ratio": "higher",
    "capital_intensity": "lower",
    "margin_of_safety": "higher",
    "fcf_to_debt": "higher",
    "dividend_payout_ratio": "lower",
    "retained_earnings_ratio": "higher",
    "stock_cagr": "higher",
    "benchmark_cagr": "higher",
    "tracking_error": "lower",
    "annualized_volatility": "lower",
    "debt_to_equity": "lower",
    "combined_ratio": "lower",
    "subscription_revenue_ratio": "higher",
    "ad_revenue_ratio": "lower",
    "churn_rate": "lower",
    "regulated_asset_ratio": "higher",
    "allowed_roe": "higher",
    "operating_ratio": "lower",
    "maintenance_capex_ratio": "lower",
    "net_revenue_retention": "higher",
    "stock_comp_ratio": "lower",
    "commodity_exposure": "lower",
    "leverage_ratio": "lower",
    "pricing_power_score": "higher",
    "purchase_multiple": "lower",
    "employee_turnover": "lower",
    "insider_ownership": "higher",
    "restructurings_per_5y": "lower",
}


BOOLEAN_RULES = {
    "inside_circle": "positive",
    "outperformed_benchmark": "positive",
    "is_discount": "positive",
    "is_undervalued": "positive",
    "mentions_intrinsic_value": "positive",
    "roic_linked_pay": "positive",
    "buybacks_below_intrinsic_value": "positive",
    "dual_class_structure": "negative",
    "debt_funded": "negative",
    "capital_adequate": "positive",
    "underestimating_payouts": "negative",
    "is_value_trap": "negative",
}


LABEL_SCORE_MAP = {
    "cash_backed": 85.0,
    "accounting_risk": 20.0,
    "economic_goodwill": 90.0,
    "mixed_goodwill": 60.0,
    "accounting_goodwill": 25.0,
    "strong": 85.0,
    "moderate": 60.0,
    "mixed": 55.0,
    "weak": 25.0,
    "excellent": 95.0,
    "disciplined": 80.0,
    "neutral": 50.0,
    "fear": 65.0,
    "greed": 35.0,
    "deep_discount": 95.0,
    "discount": 75.0,
    "fair": 50.0,
    "premium": 25.0,
    "net_cash": 80.0,
    "net_debt": 40.0,
    "distribute": 45.0,
    "retain": 65.0,
    "opportunistic/value-based": 75.0,
    "systematic": 55.0,
    "brand": 80.0,
    "low-cost producer": 75.0,
    "switching costs": 80.0,
    "network effects": 90.0,
    "regulatory": 70.0,
    "unknown": 50.0,
    "high": 85.0,
    "medium": 60.0,
    "low": 30.0,
    "strong": 85.0,
    "moderate": 60.0,
    "weak": 30.0,
}


POSITIVE_SENTIMENT_TERMS = (
    "candor",
    "honesty",
    "transparent",
    "transparency",
    "prudent",
    "discipline",
    "disciplined",
    "aligned",
    "alignment",
    "shareholder",
    "long-term",
    "rational",
    "prudence",
    "value creation",
    "healthy balance sheet",
)


NEGATIVE_SENTIMENT_TERMS = (
    "weak",
    "misaligned",
    "dilutive",
    "promotional",
    "opaque",
    "overpaid",
    "aggressive",
    "questionable",
    "poor",
    "short-term",
)


INVERSION_THEME_RULES = {
    "moat_erosion": (
        "moat erosion",
        "market share",
        "pricing power",
        "new entrants",
        "competition",
        "eroding",
    ),
    "technological_disruption": (
        "technological disruption",
        "obsolete",
        "new technologies",
        "heat pump",
        "solar",
        "decentralized",
        "biometric",
        "blockchain",
        "innovation",
    ),
    "structural_flaws": (
        "structural flaws",
        "structural flaw",
        "concentrated",
        "concentration",
        "china",
        "channel partners",
        "regulatory",
        "trade tensions",
        "over-reliance",
        "volatile",
    ),
}


SEVERE_RISK_TERMS = (
    "significantly",
    "permanently",
    "heavily",
    "disproportionate",
    "eroding",
    "volatile",
    "regulatory changes",
    "trade tensions",
    "obsolete",
)


SPECULATIVE_TERMS = (
    "could",
    "may",
    "might",
    "potentially",
)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(value, high))


def _round(value):
    return round(value, 2) if isinstance(value, float) else value


def _score_boolean(key: str, value: bool) -> float | None:
    direction = BOOLEAN_RULES.get(key)
    if direction == "positive":
        return 100.0 if value else 0.0
    if direction == "negative":
        return 0.0 if value else 100.0
    return None


def _score_label(value: str) -> float | None:
    normalized = value.strip().lower()
    return LABEL_SCORE_MAP.get(normalized)


def _score_fixed_scale(key: str, value) -> float | None:
    max_value = SCORE_MAX_BY_KEY.get(key)
    if max_value is None or not _is_number(value):
        return None
    return _clamp((float(value) / max_value) * 100.0)


def _score_inflation_assessment(rows: list[dict]) -> float | None:
    label_scores = {
        "strong (maintained margin)": 85.0,
        "weak (margin compression)": 20.0,
        "normal environment": 55.0,
        "n/a": None,
    }
    scores = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("Pricing_Power_Assessment", "")).strip().lower()
        score = label_scores.get(label)
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _split_bullets(text: str) -> list[str]:
    normalized = text.replace("\r", "\n").strip()
    if not normalized:
        return []

    matches = re.findall(r"(?:^|\n)\s*[•\-*]\s+(.*?)(?=(?:\n\s*[•\-*]\s+)|$)", normalized, flags=re.DOTALL)
    if matches:
        return [re.sub(r"\s+", " ", match).strip() for match in matches if match.strip()]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    return [re.sub(r"\s+", " ", part.lstrip("•-* ")).strip() for part in paragraphs]


def _classify_inversion_theme(text: str) -> str:
    lowered = text.lower()
    if "moat erosion" in lowered:
        return "moat_erosion"
    if "technological disruption" in lowered:
        return "technological_disruption"
    if "structural flaws" in lowered or "structural flaw" in lowered:
        return "structural_flaws"
    for theme, keywords in INVERSION_THEME_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            return theme
    return "general_inversion_risk"


def _score_inversion_claim(text: str) -> dict:
    lowered = text.lower()
    theme = _classify_inversion_theme(text)

    severity = 45.0
    severity += sum(8.0 for term in SEVERE_RISK_TERMS if term in lowered)
    severity += 6.0 if ":" in text else 0.0
    severity -= sum(3.0 for term in SPECULATIVE_TERMS if term in lowered)
    severity = _clamp(severity, 20.0, 95.0)

    confidence = 50.0
    if "**" in text or ":" in text:
        confidence += 10.0
    if any(token in lowered for token in ("north america", "rest of world", "china", "channel partners", "heat pump", "solar")):
        confidence += 12.0
    if any(term in lowered for term in SPECULATIVE_TERMS):
        confidence -= 6.0
    confidence = _clamp(confidence, 25.0, 90.0)

    risk_score = _clamp((severity * 0.7) + (confidence * 0.3))
    return {
        "theme": theme,
        "claim": text,
        "severity_score": _round(severity),
        "confidence_score": _round(confidence),
        "risk_score": _round(risk_score),
    }


def _score_inversion_theme(theme: str, claims: list[dict]) -> float | None:
    relevant_claims = [claim for claim in claims if claim.get("theme") == theme]
    if not relevant_claims:
        return None

    theme_keywords = INVERSION_THEME_RULES.get(theme, ())
    theme_scores = []
    for claim in relevant_claims:
        text = str(claim.get("claim", "")).lower()
        # Remove the leading theme heading so labels do not count as evidence.
        text = re.sub(r"^\*\*[^*]+\*\*:\s*", "", text).strip()

        matched_keywords = set()
        for keyword in theme_keywords:
            if keyword in text:
                normalized_keyword = keyword.rstrip("s")
                if any(
                    normalized_keyword in existing or existing in normalized_keyword
                    for existing in matched_keywords
                ):
                    continue
                matched_keywords.add(normalized_keyword)

        keyword_hits = len(matched_keywords)
        keyword_intensity = min(keyword_hits * 6.0, 24.0)
        score = float(claim.get("risk_score", 0.0)) + keyword_intensity
        theme_scores.append(_clamp(score, 0.0, 100.0))

    if not theme_scores:
        return None
    return _round(sum(theme_scores) / len(theme_scores))


def score_inversion_text(text: str) -> dict:
    bullets = _split_bullets(text)
    claims = [_score_inversion_claim(bullet) for bullet in bullets if bullet]
    if not claims:
        return {"claims": [], "composite_risk_score": None, "confidence_score": None}

    composite_risk = sum(claim["risk_score"] for claim in claims) / len(claims)
    confidence = sum(claim["confidence_score"] for claim in claims) / len(claims)
    themes = {claim["theme"] for claim in claims}
    theme_scores = {}
    for theme in themes:
        score = _score_inversion_theme(theme, claims)
        if score is not None:
            theme_scores[theme] = score

    return {
        "claims": claims,
        "theme_scores": theme_scores,
        "composite_risk_score": _round(composite_risk),
        "confidence_score": _round(confidence),
    }


def score_management_text(text: str) -> dict:
    lowered = text.lower()
    positive_hits = sum(1 for term in POSITIVE_SENTIMENT_TERMS if term in lowered)
    negative_hits = sum(1 for term in NEGATIVE_SENTIMENT_TERMS if term in lowered)
    sentiment = 50.0 + (positive_hits * 7.5) - (negative_hits * 9.0)
    confidence = 45.0 + (positive_hits * 4.0) + (negative_hits * 2.0)
    return {
        "sentiment_score": _round(_clamp(sentiment, 5.0, 95.0)),
        "confidence_score": _round(_clamp(confidence, 25.0, 85.0)),
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
    }


def _record_metric(company: dict, category: str, source_path: str, key: str, value, direct_score=None, direction=None):
    evidence_type, evidence_weight = _evidence_profile(source_path)
    metric_basis = "peer_sensitive" if key in PEER_SENSITIVE_METRICS or direction is not None else "intrinsic"
    metric = {
        "ticker": company.get("ticker"),
        "company_name": company.get("company_name") or company.get("ticker"),
        "sector": company.get("sector") or "",
        "industry": company.get("industry") or "",
        "category": category,
        "source_path": source_path,
        "metric_name": key,
        "raw_value": value,
        "evidence_type": evidence_type,
        "evidence_weight": evidence_weight,
        "metric_basis": metric_basis,
    }
    if direct_score is not None:
        metric["normalized_score"] = _round(direct_score)
    if direction:
        metric["direction"] = direction
    return metric


def _evidence_profile(source_path: str) -> tuple[str, float]:
    llm_text_prefixes = (
        "thinking_frameworks.Inversion",
        "management_governance.ManagementEvaluation",
    )
    llm_panel_prefixes = (
        "thinking_frameworks.CircleOfCompetence",
        "business_moat.EconomicMoat",
        "valuation_capital.ShareBuybackAnalysis",
        "valuation_capital.CapitalAllocationAnalysis.buyback_analysis",
        "risk_behavior.LeverageRisk",
    )

    if source_path.startswith(llm_text_prefixes):
        return "llm_text", 0.45
    if source_path.startswith(llm_panel_prefixes):
        return "llm_panel", 0.65
    return "data_backed", 1.0


def _should_skip_metric_path(source_path: str, key: str) -> bool:
    path = f"{source_path}.{key}" if source_path else key
    scenario_markers = ("valuation_scenarios.bear", "valuation_scenarios.base", "valuation_scenarios.bull")
    if any(marker in path for marker in scenario_markers):
        return True
    return False


def _extract_metrics_from_mapping(company: dict, category: str, source_path: str, value) -> list[dict]:
    metrics = []
    if isinstance(value, dict):
        if value.get("applicable") is False:
            return metrics
        for key, item in value.items():
            if key in {"panel_judgments", "panel_models", "panel_vote_split"}:
                continue
            if _should_skip_metric_path(source_path, key):
                continue
            child_path = f"{source_path}.{key}" if source_path else key
            if isinstance(item, dict):
                metrics.extend(_extract_metrics_from_mapping(company, category, child_path, item))
                continue
            if isinstance(item, list):
                if key == "TheImpactOfInflation" or child_path.endswith("TheImpactOfInflation"):
                    inflation_score = _score_inflation_assessment(item)
                    if inflation_score is not None:
                        metrics.append(_record_metric(company, category, child_path, "inflation_pricing_power", inflation_score, direct_score=inflation_score))
                continue
            if isinstance(item, bool):
                score = _score_boolean(key, item)
                if score is not None:
                    metrics.append(_record_metric(company, category, child_path, key, item, direct_score=score))
                continue
            if isinstance(item, str):
                score = _score_label(item)
                if score is not None:
                    metrics.append(_record_metric(company, category, child_path, key, item, direct_score=score))
                continue
            if _is_number(item):
                fixed_score = _score_fixed_scale(key, item)
                if fixed_score is not None:
                    metrics.append(_record_metric(company, category, child_path, key, float(item), direct_score=fixed_score))
                    continue
                direction = PEER_NUMERIC_RULES.get(key)
                if direction:
                    metrics.append(_record_metric(company, category, child_path, key, float(item), direction=direction))
        return metrics

    return metrics


def extract_company_metrics(company: dict) -> tuple[list[dict], dict]:
    metrics = []
    qualitative = {}

    for top_level_key, section in company.items():
        if top_level_key in {"ticker", "company_name", "description"}:
            continue

        category = CATEGORY_ALIASES.get(top_level_key)
        if not category or not isinstance(section, dict):
            continue

        metrics.extend(_extract_metrics_from_mapping(company, category, top_level_key, section))

    inversion_text = (((company.get("thinking_frameworks") or {}).get("Inversion")) or "")
    if isinstance(inversion_text, str) and inversion_text.strip():
        qualitative["inversion"] = score_inversion_text(inversion_text)
        if qualitative["inversion"].get("composite_risk_score") is not None:
            metrics.append(
                _record_metric(
                    company,
                    "risk",
                    "thinking_frameworks.Inversion",
                    "inversion_risk_score",
                    qualitative["inversion"]["composite_risk_score"],
                    direct_score=100.0 - qualitative["inversion"]["composite_risk_score"],
                )
            )

    management_text = (((company.get("management_governance") or {}).get("ManagementEvaluation")) or "")
    if isinstance(management_text, str) and management_text.strip():
        qualitative["management_sentiment"] = score_management_text(management_text)
        metrics.append(
            _record_metric(
                company,
                "management",
                "management_governance.ManagementEvaluation",
                "management_sentiment_score",
                qualitative["management_sentiment"]["sentiment_score"],
                direct_score=qualitative["management_sentiment"]["sentiment_score"],
            )
        )

    return metrics, qualitative


def _dedupe_metrics(metrics: list[dict]) -> list[dict]:
    deduped = {}
    for metric in metrics:
        dedupe_key = (
            metric.get("ticker"),
            metric.get("category"),
            metric.get("metric_name"),
            metric.get("source_path", "").split(".")[-1],
        )
        existing = deduped.get(dedupe_key)
        if existing is None or len(metric.get("source_path", "")) < len(existing.get("source_path", "")):
            deduped[dedupe_key] = metric
    return list(deduped.values())


def _candidate_normalization_buckets(metric: dict) -> list[tuple[str, str, str]]:
    metric_name = metric["metric_name"]
    industry = str(metric.get("industry") or "").strip()
    sector = str(metric.get("sector") or "").strip()

    candidates = []
    if industry:
        candidates.append((metric_name, "industry", industry))
    if sector:
        candidates.append((metric_name, "sector", sector))
    candidates.append((metric_name, "all", "all"))
    return candidates


def _apply_peer_normalization(metrics: list[dict]) -> None:
    grouped = {}
    for metric in metrics:
        if "normalized_score" in metric:
            continue
        direction = metric.get("direction")
        if not direction:
            continue
        for bucket in _candidate_normalization_buckets(metric):
            grouped.setdefault(bucket, []).append(metric)

    for metric in metrics:
        if "normalized_score" in metric:
            continue
        direction = metric.get("direction")
        if not direction:
            continue

        items = None
        selected_bucket = None
        for bucket in _candidate_normalization_buckets(metric):
            bucket_items = grouped.get(bucket, [])
            unique_tickers = {item.get("ticker") for item in bucket_items}
            if len(unique_tickers) >= 2:
                items = bucket_items
                selected_bucket = bucket
                break

        if not items:
            continue

        values = [item["raw_value"] for item in items if _is_number(item.get("raw_value"))]
        if not values:
            continue
        low = min(values)
        high = max(values)
        value = metric["raw_value"]
        if high == low:
            score = 50.0
        else:
            ratio = (float(value) - low) / (high - low)
            if metric.get("direction") == "lower":
                ratio = 1.0 - ratio
            score = ratio * 100.0
        metric["normalized_score"] = _round(_clamp(score))
        metric["peer_group_size"] = len({item.get("ticker") for item in items})
        metric["normalization_scope"] = selected_bucket[1] if selected_bucket else "all"


def _category_reliability_factor(diagnostic: dict | None) -> float:
    if not diagnostic:
        return 0.0

    evidence = float(diagnostic.get("average_evidence_weight") or 0.0)
    peer_metrics = int(diagnostic.get("peer_metrics") or 0)
    weak_peer_metrics = int(diagnostic.get("weak_peer_metrics") or 0)

    if peer_metrics == 0:
        peer_reliability = 1.0
    else:
        peer_reliability = max(0.35, 1.0 - (weak_peer_metrics / max(peer_metrics, 1)) * 0.45)

    return _round((0.6 * evidence) + (0.4 * peer_reliability))


def _compute_weighted_score(category_scores: dict, category_diagnostics: dict, weights: dict, score_name: str) -> dict:
    total_weight = sum(weights.values())
    weighted_total = 0.0
    used_weight = 0.0
    reliable_weight = 0.0

    for category, weight in weights.items():
        score = category_scores.get(category)
        if score is None:
            continue
        weighted_total += float(score) * weight
        used_weight += weight
        reliable_weight += weight * _category_reliability_factor(category_diagnostics.get(category))

    raw_score = None if used_weight == 0 else (weighted_total / used_weight)
    structural_coverage = 0.0 if total_weight == 0 else (used_weight / total_weight)
    reliability_adjusted_coverage = 0.0 if total_weight == 0 else (reliable_weight / total_weight)
    coverage = min(structural_coverage, reliability_adjusted_coverage)
    score = None if raw_score is None else _round(raw_score * coverage)

    return {
        f"raw_{score_name}": _round(raw_score) if raw_score is not None else None,
        score_name: score,
        f"{score_name}_coverage": _round(coverage * 100.0),
        f"{score_name}_structural_coverage": _round(structural_coverage * 100.0),
        f"{score_name}_reliability_coverage": _round(reliability_adjusted_coverage * 100.0),
        f"{score_name}_categories": [category for category in weights if category in category_scores],
    }


def _compute_basis_weighted_score(metrics: list[dict], category_diagnostics: dict, weights: dict, score_name: str, preferred_basis: str, basis_mix: float) -> dict:
    total_weight = sum(weights.values())
    weighted_total = 0.0
    used_weight = 0.0
    reliable_weight = 0.0

    for category, category_weight in weights.items():
        category_metrics = [metric for metric in metrics if metric.get("category") == category and metric.get("normalized_score") is not None]
        if not category_metrics:
            continue

        preferred_scores = [metric for metric in category_metrics if metric.get("metric_basis") == preferred_basis]
        fallback_scores = [metric for metric in category_metrics if metric.get("metric_basis") != preferred_basis]

        chosen = preferred_scores or fallback_scores
        if not chosen:
            continue

        blended_scores = []
        blended_weights = []
        for metric in chosen:
            score = float(metric["normalized_score"])
            evidence_weight = float(metric.get("evidence_weight", 1.0))
            basis_weight = basis_mix if metric.get("metric_basis") == preferred_basis else (1.0 - basis_mix)
            blended_scores.append(score * evidence_weight * basis_weight)
            blended_weights.append(evidence_weight * basis_weight)

        total_metric_weight = sum(blended_weights)
        if total_metric_weight <= 0:
            continue

        category_score = sum(blended_scores) / total_metric_weight
        weighted_total += category_score * category_weight
        used_weight += category_weight
        reliable_weight += category_weight * _category_reliability_factor(category_diagnostics.get(category))

    raw_score = None if used_weight == 0 else (weighted_total / used_weight)
    structural_coverage = 0.0 if total_weight == 0 else (used_weight / total_weight)
    reliability_adjusted_coverage = 0.0 if total_weight == 0 else (reliable_weight / total_weight)
    coverage = min(structural_coverage, reliability_adjusted_coverage)
    score = None if raw_score is None else _round(raw_score * coverage)

    return {
        f"raw_{score_name}": _round(raw_score) if raw_score is not None else None,
        score_name: score,
        f"{score_name}_coverage": _round(coverage * 100.0),
        f"{score_name}_structural_coverage": _round(structural_coverage * 100.0),
        f"{score_name}_reliability_coverage": _round(reliability_adjusted_coverage * 100.0),
        f"{score_name}_basis": preferred_basis,
    }


def _build_category_diagnostics(metrics: list[dict]) -> dict:
    diagnostics = {}
    by_category = {}
    for metric in metrics:
        score = metric.get("normalized_score")
        category = metric.get("category")
        if score is None or not category:
            continue
        by_category.setdefault(category, []).append(metric)

    for category, items in by_category.items():
        evidence_weights = [float(item.get("evidence_weight", 1.0)) for item in items]
        data_backed_count = sum(1 for item in items if item.get("evidence_type") == "data_backed")
        llm_only_count = len(items) - data_backed_count
        peer_group_sizes = [int(item.get("peer_group_size")) for item in items if item.get("peer_group_size") is not None]
        weak_peer_metrics = sum(1 for size in peer_group_sizes if size < MIN_PEER_GROUP_SIZE)

        warning_parts = []
        if peer_group_sizes and weak_peer_metrics:
            warning_parts.append(
                f"{weak_peer_metrics}/{len(peer_group_sizes)} peer metrics compared against fewer than {MIN_PEER_GROUP_SIZE} tickers"
            )
        if data_backed_count == 0 and llm_only_count:
            warning_parts.append("category score relies entirely on LLM-derived signals")

        diagnostics[category] = {
            "scored_metrics": len(items),
            "average_evidence_weight": _round(sum(evidence_weights) / len(evidence_weights)),
            "data_backed_metrics": data_backed_count,
            "llm_only_metrics": llm_only_count,
            "peer_metrics": len(peer_group_sizes),
            "weak_peer_metrics": weak_peer_metrics,
            "min_peer_group_size": min(peer_group_sizes) if peer_group_sizes else None,
            "warning": "; ".join(warning_parts) if warning_parts else None,
        }

    return diagnostics


def _aggregate_company_scores(metrics: list[dict], qualitative: dict) -> dict:
    category_buckets = {}
    category_weight_buckets = {}
    for metric in metrics:
        score = metric.get("normalized_score")
        category = metric.get("category")
        if score is None or not category:
            continue
        weight = float(metric.get("evidence_weight", 1.0))
        category_buckets.setdefault(category, []).append(float(score) * weight)
        category_weight_buckets.setdefault(category, []).append(weight)

    category_scores = {}
    for category, weighted_scores in category_buckets.items():
        total_weight = sum(category_weight_buckets.get(category, []))
        if total_weight <= 0:
            continue
        category_scores[category] = _round(sum(weighted_scores) / total_weight)

    category_diagnostics = _build_category_diagnostics(metrics)

    weighted_total = 0.0
    used_weight = 0.0
    for category, score in category_scores.items():
        weight = CATEGORY_WEIGHTS.get(category, 0.0)
        if weight <= 0:
            continue
        weighted_total += score * weight
        used_weight += weight

    raw_overall_score = None if used_weight == 0 else (weighted_total / used_weight)

    weighted_category_coverage = used_weight / sum(CATEGORY_WEIGHTS.values())
    metric_coverage = min(len([metric for metric in metrics if metric.get("normalized_score") is not None]) / 18.0, 1.0)
    coverage_penalty_factor = (0.65 * weighted_category_coverage) + (0.35 * metric_coverage)
    overall_score = None if raw_overall_score is None else _round(raw_overall_score * coverage_penalty_factor)

    confidence_parts = []
    scored_metrics = len([metric for metric in metrics if metric.get("normalized_score") is not None])
    if scored_metrics:
        confidence_parts.append(min(scored_metrics / 25.0, 1.0) * 100.0)
    inversion_confidence = ((qualitative.get("inversion") or {}).get("confidence_score"))
    if inversion_confidence is not None:
        confidence_parts.append(float(inversion_confidence))
    management_confidence = ((qualitative.get("management_sentiment") or {}).get("confidence_score"))
    if management_confidence is not None:
        confidence_parts.append(float(management_confidence))
    confidence_score = _round(sum(confidence_parts) / len(confidence_parts)) if confidence_parts else None

    return {
        "category_scores": category_scores,
        "category_diagnostics": category_diagnostics,
        "overall_score": overall_score,
        "raw_overall_score": _round(raw_overall_score) if raw_overall_score is not None else None,
        "coverage_penalty_factor": _round(coverage_penalty_factor),
        **_compute_basis_weighted_score(metrics, category_diagnostics, BUSINESS_QUALITY_WEIGHTS, "business_quality_score", preferred_basis="intrinsic", basis_mix=0.8),
        **_compute_basis_weighted_score(metrics, category_diagnostics, OPPORTUNITY_WEIGHTS, "opportunity_score", preferred_basis="peer_sensitive", basis_mix=0.8),
        "confidence_score": confidence_score,
        "coverage": {
            "scored_metrics": scored_metrics,
            "categories_with_scores": len(category_scores),
            "weighted_category_coverage": _round(weighted_category_coverage * 100.0),
            "metric_coverage": _round(metric_coverage * 100.0),
        },
    }


def _build_rankings(company_packets: list[dict], score_key: str, raw_score_key: str | None = None, coverage_key: str | None = None) -> list[dict]:
    rankings = []
    for packet in company_packets:
        row = {
            "ticker": packet["ticker"],
            "company_name": packet["company_name"],
            score_key: packet.get(score_key),
        }
        if raw_score_key:
            row[raw_score_key] = packet.get(raw_score_key)
        if coverage_key:
            row[coverage_key] = packet.get(coverage_key)
        rankings.append(row)

    rankings.sort(key=lambda item: (item.get(score_key) is None, -(item.get(score_key) or 0.0), item["ticker"]))
    for index, ranking in enumerate(rankings, start=1):
        ranking["rank"] = index
    return rankings


def _compute_composite_score(packet: dict) -> dict:
    business_quality_score = packet.get("business_quality_score")
    opportunity_score = packet.get("opportunity_score")
    business_quality_coverage = float(packet.get("business_quality_score_coverage") or 0.0) / 100.0
    opportunity_coverage = float(packet.get("opportunity_score_coverage") or 0.0) / 100.0

    weighted_total = 0.0
    used_weight = 0.0
    coverage_weighted_total = 0.0

    for score_key, base_weight in COMPOSITE_SCORE_WEIGHTS.items():
        score = packet.get(score_key)
        if score is None:
            continue
        weighted_total += float(score) * base_weight
        used_weight += base_weight

    if business_quality_score is not None:
        coverage_weighted_total += COMPOSITE_SCORE_WEIGHTS["business_quality_score"] * business_quality_coverage
    if opportunity_score is not None:
        coverage_weighted_total += COMPOSITE_SCORE_WEIGHTS["opportunity_score"] * opportunity_coverage

    raw_composite_score = None if used_weight == 0 else (weighted_total / used_weight)
    composite_coverage = 0.0 if used_weight == 0 else (coverage_weighted_total / used_weight)
    composite_score = None if raw_composite_score is None else _round(raw_composite_score * composite_coverage)

    return {
        "legacy_overall_score": packet.get("overall_score"),
        "legacy_raw_overall_score": packet.get("raw_overall_score"),
        "composite_score": composite_score,
        "raw_composite_score": _round(raw_composite_score) if raw_composite_score is not None else None,
        "composite_score_coverage": _round(composite_coverage * 100.0),
    }


def build_comparison(companies: list[dict]) -> dict:
    company_packets = []
    all_metrics = []

    for company in companies:
        metrics, qualitative = extract_company_metrics(company)
        metrics = _dedupe_metrics(metrics)
        packet = {
            "ticker": company.get("ticker"),
            "company_name": company.get("company_name") or company.get("ticker"),
            "qualitative_scores": qualitative,
            "metric_scores": metrics,
        }
        company_packets.append(packet)
        all_metrics.extend(metrics)

    _apply_peer_normalization(all_metrics)

    for packet in company_packets:
        metrics = [metric for metric in all_metrics if metric.get("ticker") == packet["ticker"]]
        packet["metric_scores"] = metrics
        packet.update(_aggregate_company_scores(metrics, packet.get("qualitative_scores") or {}))
        packet.update(_compute_composite_score(packet))

    rankings = _build_rankings(company_packets, "composite_score", raw_score_key="raw_composite_score", coverage_key="composite_score_coverage")
    for ranking in rankings:
        ticker = ranking["ticker"]
        packet = next(packet for packet in company_packets if packet["ticker"] == ticker)
        ranking["overall_score"] = ranking.pop("composite_score")
        ranking["raw_overall_score"] = ranking.pop("raw_composite_score")
        ranking["overall_score_coverage"] = ranking.pop("composite_score_coverage")
        ranking["confidence_score"] = packet.get("confidence_score")
        ranking["legacy_overall_score"] = packet.get("legacy_overall_score")
        ranking["legacy_raw_overall_score"] = packet.get("legacy_raw_overall_score")

    business_quality_rankings = _build_rankings(
        company_packets,
        "business_quality_score",
        raw_score_key="raw_business_quality_score",
        coverage_key="business_quality_score_coverage",
    )
    opportunity_rankings = _build_rankings(
        company_packets,
        "opportunity_score",
        raw_score_key="raw_opportunity_score",
        coverage_key="opportunity_score_coverage",
    )

    category_rankings = {}
    for category in CATEGORY_WEIGHTS:
        rows = []
        for packet in company_packets:
            score = packet.get("category_scores", {}).get(category)
            if score is None:
                continue
            rows.append({
                "ticker": packet["ticker"],
                "company_name": packet["company_name"],
                "score": score,
                "warning": ((packet.get("category_diagnostics") or {}).get(category) or {}).get("warning"),
            })
        rows.sort(key=lambda item: (-item["score"], item["ticker"]))
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        if rows:
            category_rankings[category] = rows

    category_warnings = {}
    for category, rows in category_rankings.items():
        warnings = [
            {
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "warning": row["warning"],
            }
            for row in rows
            if row.get("warning")
        ]
        if warnings:
            category_warnings[category] = warnings

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "company_count": len(company_packets),
        "rankings": rankings,
        "business_quality_rankings": business_quality_rankings,
        "opportunity_rankings": opportunity_rankings,
        "category_rankings": category_rankings,
        "category_warnings": category_warnings,
        "companies": company_packets,
    }


def load_analysis_files(paths: list[str | Path]) -> list[dict]:
    analyses = []
    for path in paths:
        file_path = Path(path)
        try:
            with file_path.open() as handle:
                analyses.append(json.load(handle))
        except json.JSONDecodeError as exc:
            logger.warning(f"Skipping malformed analysis file {file_path}: {exc}")
        except OSError as exc:
            logger.warning(f"Skipping unreadable analysis file {file_path}: {exc}")
    return analyses
