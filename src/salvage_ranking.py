import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
SOURCE_COMPARISON = OUTPUT_DIR / "sp500_ranking_comparison.json"
SALVAGED_COMPARISON = OUTPUT_DIR / "salvaged_sp500_comparison.json"
SALVAGED_TOP20 = OUTPUT_DIR / "salvaged_top20_review.json"


COMMODITY_INDUSTRIES = {
    "gold",
    "oil & gas e&p",
    "oil & gas integrated",
    "oil & gas midstream",
    "oil & gas refining & marketing",
    "coking coal",
    "steel",
    "copper",
    "silver",
    "uranium",
    "aluminum",
}


COMPLEX_SECTORS = {
    "biotechnology",
}


def _read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _round(value):
    return round(float(value), 2)


def _load_analyses() -> dict:
    analyses = {}
    for path in OUTPUT_DIR.glob("*_analysis.json"):
        ticker = path.name.split("_")[0]
        analyses[ticker] = _read_json(path)
    return analyses


def _get_margin_of_safety(analysis: dict) -> dict:
    return (((analysis.get("investment_philosophy") or {}).get("UndervaluedMarginOfSafety")) or {})


def _get_intrinsic_value(analysis: dict) -> dict:
    return (((analysis.get("investment_philosophy") or {}).get("IntrinsicValue")) or {})


def _get_circle(analysis: dict) -> dict:
    return (((analysis.get("thinking_frameworks") or {}).get("CircleOfCompetence")) or {})


def _get_moat(analysis: dict) -> dict:
    return (((analysis.get("business_moat") or {}).get("EconomicMoat")) or {})


def _get_durability(analysis: dict) -> dict:
    return (((analysis.get("business_moat") or {}).get("TheDurabilityOfCompetitiveAdvantage")) or {})


def _get_management_eval(analysis: dict) -> dict:
    raw = (((analysis.get("management_governance") or {}).get("ManagementEvaluation")) or {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return {"applicable": True, "text": raw}
    return {}


def _get_governance(analysis: dict) -> dict:
    return (((analysis.get("management_governance") or {}).get("CorporateGovernanceAndShareholderOrientation")) or {})


def _get_capital_allocation(analysis: dict) -> dict:
    return (((analysis.get("valuation_capital") or {}).get("CapitalAllocationAnalysis")) or {})


def _industry_key(analysis: dict) -> str:
    return str(analysis.get("industry") or "").strip().lower()


def _sector_key(analysis: dict) -> str:
    return str(analysis.get("sector") or "").strip().lower()


def salvage_company(row: dict, analysis: dict) -> dict:
    score = float(row.get("overall_score") or 0.0)
    penalties = []
    bonuses = []

    circle = _get_circle(analysis)
    moat = _get_moat(analysis)
    durability = _get_durability(analysis)
    management_eval = _get_management_eval(analysis)
    governance = _get_governance(analysis)
    capital_allocation = _get_capital_allocation(analysis)
    margin_of_safety = _get_margin_of_safety(analysis)
    intrinsic_value = _get_intrinsic_value(analysis)
    industry = _industry_key(analysis)
    sector = _sector_key(analysis)

    if circle.get("inside_circle") is False:
        score -= 12.0
        penalties.append("Outside circle of competence")

    if industry in COMMODITY_INDUSTRIES:
        score -= 10.0
        penalties.append(f"Commodity-linked industry: {analysis.get('industry')}")

    if industry in COMPLEX_SECTORS:
        score -= 8.0
        penalties.append(f"High-complexity industry: {analysis.get('industry')}")

    if moat.get("applicable") is False:
        score -= 6.0
        penalties.append("Moat section ungrounded")

    if management_eval.get("applicable") is False:
        score -= 6.0
        penalties.append("Management evaluation unavailable or ungrounded")

    if durability.get("applicable") is False:
        score -= 4.0
        penalties.append("Durability section incomplete")

    if margin_of_safety.get("is_undervalued") is False:
        score -= 8.0
        penalties.append("Fails current margin-of-safety test")

    valuation_confidence = intrinsic_value.get("valuation_confidence") or {}
    if valuation_confidence.get("confidence_label") == "low":
        score -= 3.0
        penalties.append("Low valuation confidence")

    if governance.get("shareholder_orientation") == "strong":
        score += 2.0
        bonuses.append("Strong shareholder orientation")

    if capital_allocation.get("capital_allocation_discipline") == "strong":
        score += 2.0
        bonuses.append("Strong capital allocation discipline")

    if circle.get("inside_circle") is True:
        score += 2.0
        bonuses.append("Inside circle of competence")

    if margin_of_safety.get("is_undervalued") is True:
        score += 2.0
        bonuses.append("Passes current margin-of-safety test")

    score = max(0.0, min(100.0, score))

    major_red_flags = sum(
        1
        for condition in (
            circle.get("inside_circle") is False,
            industry in COMMODITY_INDUSTRIES,
            industry in COMPLEX_SECTORS,
            margin_of_safety.get("is_undervalued") is False,
            moat.get("applicable") is False,
            management_eval.get("applicable") is False,
        )
        if condition
    )

    label = "questionable"
    if score >= 68.0 and major_red_flags <= 1:
        label = "keep"
    elif score < 58.0 or major_red_flags >= 4:
        label = "reject"

    return {
        "ticker": row.get("ticker"),
        "company_name": row.get("company_name"),
        "sector": analysis.get("sector"),
        "industry": analysis.get("industry"),
        "original_rank": row.get("rank"),
        "original_score": row.get("overall_score"),
        "original_confidence_score": row.get("confidence_score"),
        "salvaged_score": _round(score),
        "review_label": label,
        "penalties": penalties,
        "bonuses": bonuses,
        "salvage_notes": {
            "inside_circle": circle.get("inside_circle"),
            "moat_grounded": moat.get("applicable") is not False,
            "management_grounded": management_eval.get("applicable") is not False,
            "durability_grounded": durability.get("applicable") is not False,
            "is_undervalued": margin_of_safety.get("is_undervalued"),
            "valuation_confidence_label": valuation_confidence.get("confidence_label"),
            "shareholder_orientation": governance.get("shareholder_orientation"),
            "capital_allocation_discipline": capital_allocation.get("capital_allocation_discipline"),
        },
    }


def build_salvaged_outputs() -> tuple[dict, dict]:
    comparison = _read_json(SOURCE_COMPARISON)
    analyses = _load_analyses()

    salvaged_rows = []
    for row in comparison.get("rankings", []):
        ticker = row.get("ticker")
        analysis = analyses.get(ticker)
        if not analysis:
            continue
        salvaged_rows.append(salvage_company(row, analysis))

    salvaged_rows.sort(key=lambda item: (-item["salvaged_score"], item["ticker"]))
    for index, row in enumerate(salvaged_rows, start=1):
        row["rank"] = index

    top20 = salvaged_rows[:20]

    comparison_payload = {
        "generated_at": comparison.get("generated_at"),
        "company_count": len(salvaged_rows),
        "method": "non_destructive_salvage_overlay",
        "notes": [
            "Derived entirely from existing *_analysis.json files and the original comparison ranking.",
            "Applies explicit penalties for ungrounded moat/management sections, outside-circle names, commodity exposure, and failed margin-of-safety tests.",
            "Does not modify source analysis JSON files.",
        ],
        "rankings": [
            {
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "rank": row["rank"],
                "overall_score": row["salvaged_score"],
                "raw_overall_score": row["salvaged_score"],
                "overall_score_coverage": None,
                "confidence_score": row["original_confidence_score"],
                "review_label": row["review_label"],
                "legacy_rank": row["original_rank"],
                "legacy_overall_score": row["original_score"],
            }
            for row in salvaged_rows
        ],
        "category_rankings": {
            "salvaged_score": [
                {
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "score": row["salvaged_score"],
                    "rank": row["rank"],
                    "warning": "; ".join(row["penalties"]) if row["penalties"] else None,
                }
                for row in salvaged_rows[:25]
            ]
        },
        "companies": [
            {
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "overall_score": row["salvaged_score"],
                "confidence_score": row["original_confidence_score"],
                "qualitative_scores": {
                    "inversion": {
                        "composite_risk_score": None,
                        "theme_scores": {},
                    },
                    "management_sentiment": {
                        "sentiment_score": None,
                    },
                },
                "salvage_review": {
                    "review_label": row["review_label"],
                    "penalties": row["penalties"],
                    "bonuses": row["bonuses"],
                    "original_rank": row["original_rank"],
                    "original_score": row["original_score"],
                },
            }
            for row in salvaged_rows
        ],
    }

    top20_payload = {
        "generated_at": comparison.get("generated_at"),
        "method": comparison_payload["method"],
        "top20": top20,
        "summary": {
            "keep": sum(1 for row in top20 if row["review_label"] == "keep"),
            "questionable": sum(1 for row in top20 if row["review_label"] == "questionable"),
            "reject": sum(1 for row in top20 if row["review_label"] == "reject"),
        },
    }
    return comparison_payload, top20_payload


def main() -> None:
    comparison_payload, top20_payload = build_salvaged_outputs()
    SALVAGED_COMPARISON.write_text(json.dumps(comparison_payload, indent=2))
    SALVAGED_TOP20.write_text(json.dumps(top20_payload, indent=2))
    print(f"Wrote {SALVAGED_COMPARISON}")
    print(f"Wrote {SALVAGED_TOP20}")


if __name__ == "__main__":
    main()
