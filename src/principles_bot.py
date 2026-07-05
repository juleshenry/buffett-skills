from evaluator_thresholds import (
    BEHAVIORAL_HIGH_RISK_FLAG_COUNT_MIN,
    BEHAVIORAL_HOLDING_PERIOD_MIN_YEARS,
    EFFICIENT_MARKET_EXCESS_RETURN_MAX,
    EFFICIENT_MARKET_TRACKING_ERROR_MAX,
    FOCUS_INVESTING_TOP_THREE_WEIGHT_MIN,
    INDEPENDENT_THINKING_EVIDENCE_MIN,
    INDEPENDENT_THINKING_VALUATION_GAP_MIN,
    LONG_TERM_ACCEPTABLE_YEARS_MIN,
    LONG_TERM_STRONG_EXCESS_RETURN_MIN,
    LONG_TERM_STRONG_YEARS_MIN,
    MARKET_FORECAST_USEFUL_ERROR_MAX,
    PATIENCE_HOLDING_PERIOD_MIN_YEARS,
    PATIENCE_TURNOVER_RATIO_MAX,
)


class FocusInvestingPrinciple:
    name = "Focus Investing"

    def evaluate(self, positions: list[float]) -> dict:
        if not positions:
            raise ValueError("positions must not be empty")

        total = sum(positions)
        if total <= 0:
            raise ValueError("positions must sum to a positive number")

        weights = [position / total for position in positions]
        top_position_weight = max(weights)
        top_three_weight = sum(sorted(weights, reverse=True)[:3])

        return {
            "position_count": len(positions),
            "top_position_weight": top_position_weight,
            "top_three_weight": top_three_weight,
            "is_focus_investing": top_three_weight >= FOCUS_INVESTING_TOP_THREE_WEIGHT_MIN,
        }


class PatienceAsEdgePrinciple:
    name = "Patience as Edge"

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
            "patience_as_edge": patience,
        }


class CommonBehavioralBiasesPrinciple:
    name = "Common Behavioral Biases (Psychological Traps in Investing)"

    def evaluate(
        self,
        thesis_changes_after_price_move: bool,
        avg_holding_period_years: float,
        adds_to_losers_without_new_evidence: bool,
    ) -> dict:
        flags = []
        if thesis_changes_after_price_move:
            flags.append("recency_bias")
        if avg_holding_period_years < BEHAVIORAL_HOLDING_PERIOD_MIN_YEARS:
            flags.append("impatience")
        if adds_to_losers_without_new_evidence:
            flags.append("commitment_bias")

        return {
            "bias_flags": flags,
            "bias_count": len(flags),
            "behavioral_risk": "high" if len(flags) >= BEHAVIORAL_HIGH_RISK_FLAG_COUNT_MIN else "moderate" if len(flags) == 1 else "low",
        }


class EfficientMarketTheoryPrinciple:
    name = "Efficient Market Theory"

    def evaluate(self, stock_cagr: float, benchmark_cagr: float, tracking_error: float) -> dict:
        excess_return = stock_cagr - benchmark_cagr
        return {
            "stock_cagr": stock_cagr,
            "benchmark_cagr": benchmark_cagr,
            "tracking_error": tracking_error,
            "excess_return": excess_return,
            "market_efficiency_supported": abs(excess_return) <= EFFICIENT_MARKET_EXCESS_RETURN_MAX and tracking_error <= EFFICIENT_MARKET_TRACKING_ERROR_MAX,
        }


class MarketForecastingPrinciple:
    name = "Market Forecasting"

    def evaluate(self, forecast_return: float, actual_return: float) -> dict:
        forecast_error = abs(actual_return - forecast_return)
        return {
            "forecast_return": forecast_return,
            "actual_return": actual_return,
            "forecast_error": forecast_error,
            "forecast_was_useful": forecast_error <= MARKET_FORECAST_USEFUL_ERROR_MAX,
        }


class CompoundingPrinciple:
    name = "Compounding"

    def evaluate(self, annual_return: float, years: float, initial_capital: float = 1.0) -> dict:
        if years <= 0:
            raise ValueError("years must be positive")
        ending_value = initial_capital * ((1 + annual_return) ** years)
        return {
            "annual_return": annual_return,
            "years": years,
            "initial_capital": initial_capital,
            "ending_value": ending_value,
            "total_return_multiple": ending_value / initial_capital if initial_capital else None,
        }


class LongtermOrientationPrinciple:
    name = "Long-term Orientation"

    def evaluate(self, stock_cagr: float, benchmark_cagr: float, years: float) -> dict:
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
            "long_term_orientation": assessment,
        }


class IndependentThinkingPrinciple:
    name = "Independent Thinking"

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
            "independent_thinking": assessment,
        }


class OpportunityCostAwarenessPrinciple:
    name = "Opportunity Cost Awareness"

    def evaluate(self, candidate_return: float, hurdle_return: float, alternative_return: float) -> dict:
        best_alternative = max(hurdle_return, alternative_return)
        spread = candidate_return - best_alternative
        return {
            "candidate_return": candidate_return,
            "hurdle_return": hurdle_return,
            "alternative_return": alternative_return,
            "best_alternative": best_alternative,
            "excess_return_vs_best_alternative": spread,
            "clears_opportunity_cost": candidate_return >= best_alternative,
        }


class WhenToSellPrinciple:
    name = "When to Sell (Clear Criteria)"

    def evaluate(
        self,
        thesis_broken: bool,
        better_opportunity_available: bool,
        extreme_overvaluation: bool,
        balance_sheet_deterioration: bool,
    ) -> dict:
        reasons = []
        if thesis_broken:
            reasons.append("thesis_broken")
        if better_opportunity_available:
            reasons.append("better_opportunity_available")
        if extreme_overvaluation:
            reasons.append("extreme_overvaluation")
        if balance_sheet_deterioration:
            reasons.append("balance_sheet_deterioration")

        return {
            "sell": bool(reasons),
            "reasons": reasons,
            "primary_reason": reasons[0] if reasons else None,
        }


def get_principles() -> dict[str, object]:
    return {
        "focus_investing": FocusInvestingPrinciple(),
        "patience_as_edge": PatienceAsEdgePrinciple(),
        "behavioral_biases": CommonBehavioralBiasesPrinciple(),
        "efficient_market_theory": EfficientMarketTheoryPrinciple(),
        "market_forecasting": MarketForecastingPrinciple(),
        "compounding": CompoundingPrinciple(),
        "longterm_orientation": LongtermOrientationPrinciple(),
        "independent_thinking": IndependentThinkingPrinciple(),
        "opportunity_cost_awareness": OpportunityCostAwarenessPrinciple(),
        "when_to_sell": WhenToSellPrinciple(),
    }
