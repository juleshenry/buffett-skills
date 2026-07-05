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
    "financial_quality": 0.18,
    "moat": 0.18,
    "risk": 0.18,
    "valuation_capital": 0.18,
    "management": 0.10,
    "investment_performance": 0.08,
    "thinking_frameworks": 0.05,
    "industry": 0.05,
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
    metric = {
        "ticker": company.get("ticker"),
        "company_name": company.get("company_name") or company.get("ticker"),
        "sector": company.get("sector") or "",
        "industry": company.get("industry") or "",
        "category": category,
        "source_path": source_path,
        "metric_name": key,
        "raw_value": value,
    }
    if direct_score is not None:
        metric["normalized_score"] = _round(direct_score)
    if direction:
        metric["direction"] = direction
    return metric


def _extract_metrics_from_mapping(company: dict, category: str, source_path: str, value) -> list[dict]:
    metrics = []
    if isinstance(value, dict):
        if value.get("applicable") is False:
            return metrics
        for key, item in value.items():
            if key in {"panel_judgments", "panel_models", "panel_vote_split"}:
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
        for bucket in _candidate_normalization_buckets(metric):
            bucket_items = grouped.get(bucket, [])
            unique_tickers = {item.get("ticker") for item in bucket_items}
            if len(unique_tickers) >= 2:
                items = bucket_items
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


def _aggregate_company_scores(metrics: list[dict], qualitative: dict) -> dict:
    category_buckets = {}
    for metric in metrics:
        score = metric.get("normalized_score")
        category = metric.get("category")
        if score is None or not category:
            continue
        category_buckets.setdefault(category, []).append(float(score))

    category_scores = {category: _round(sum(scores) / len(scores)) for category, scores in category_buckets.items() if scores}

    weighted_total = 0.0
    used_weight = 0.0
    for category, score in category_scores.items():
        weight = CATEGORY_WEIGHTS.get(category, 0.0)
        if weight <= 0:
            continue
        weighted_total += score * weight
        used_weight += weight

    raw_overall_score = None if used_weight == 0 else (weighted_total / used_weight)

    weighted_category_coverage = used_weight
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
        "overall_score": overall_score,
        "raw_overall_score": _round(raw_overall_score) if raw_overall_score is not None else None,
        "coverage_penalty_factor": _round(coverage_penalty_factor),
        "confidence_score": confidence_score,
        "coverage": {
            "scored_metrics": scored_metrics,
            "categories_with_scores": len(category_scores),
            "weighted_category_coverage": _round(weighted_category_coverage * 100.0),
            "metric_coverage": _round(metric_coverage * 100.0),
        },
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

    packets_by_ticker = {packet["ticker"]: packet for packet in company_packets}
    for packet in company_packets:
        metrics = [metric for metric in all_metrics if metric.get("ticker") == packet["ticker"]]
        packet["metric_scores"] = metrics
        packet.update(_aggregate_company_scores(metrics, packet.get("qualitative_scores") or {}))

    rankings = []
    for packet in company_packets:
        rankings.append(
            {
                "ticker": packet["ticker"],
                "company_name": packet["company_name"],
                "overall_score": packet.get("overall_score"),
                "confidence_score": packet.get("confidence_score"),
            }
        )
    rankings.sort(key=lambda item: (item.get("overall_score") is None, -(item.get("overall_score") or 0.0), item["ticker"]))

    for index, ranking in enumerate(rankings, start=1):
        ranking["rank"] = index

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
            })
        rows.sort(key=lambda item: (-item["score"], item["ticker"]))
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        if rows:
            category_rankings[category] = rows

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "company_count": len(company_packets),
        "rankings": rankings,
        "category_rankings": category_rankings,
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
