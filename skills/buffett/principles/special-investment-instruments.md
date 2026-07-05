# Special Investment Instruments

These heuristics are not universal and only apply to specific situations (e.g., preferred stock, convertibles).

```python
class SpecialInvestmentInstruments:
    """
    Heuristic: Special Investment Instruments
    """
    def __init__(self):
        pass

    def evaluate(self, coupon_rate: float | None = None, conversion_discount: float | None = None, collateral_coverage: float | None = None, ticker: str = "") -> dict:
        if ticker and (coupon_rate is None or conversion_discount is None or collateral_coverage is None):
            instrument_metrics = self._fetch_special_instrument_metrics(ticker)
            if not instrument_metrics["has_special_instrument"]:
                return {
                    "ticker": ticker,
                    "applicable": False,
                    "reason": "No special investment instrument evidence found in recent filings.",
                }
            if coupon_rate is None:
                coupon_rate = instrument_metrics["coupon_rate"]
            if conversion_discount is None:
                conversion_discount = instrument_metrics["conversion_discount"]
            if collateral_coverage is None:
                collateral_coverage = instrument_metrics["collateral_coverage"]
            
        if coupon_rate is None or conversion_discount is None or collateral_coverage is None:
            return {"applicable": False, "reason": "Missing required metrics: All metrics must be provided"}

        score = 0
        if coupon_rate >= SPECIAL_INSTRUMENT_COUPON_RATE_MIN:
            score += 1
        if conversion_discount >= SPECIAL_INSTRUMENT_CONVERSION_DISCOUNT_MIN:
            score += 1
        if collateral_coverage >= SPECIAL_INSTRUMENT_COLLATERAL_COVERAGE_MIN:
            score += 1

        attractiveness = "low"
        if score == 3:
            attractiveness = "high"
        elif score == 2:
            attractiveness = "moderate"

        return {
            "coupon_rate": coupon_rate,
            "conversion_discount": conversion_discount,
            "collateral_coverage": collateral_coverage,
            "instrument_score": score,
            "instrument_attractiveness": attractiveness
        }

    def _fetch_special_instrument_metrics(self, ticker: str) -> dict[str, Any]:
        commentary = fetch_special_instrument_commentary(ticker)
        if not commentary:
            return {
                "has_special_instrument": False,
                "coupon_rate": None,
                "conversion_discount": None,
                "collateral_coverage": None,
            }
        return _parse_special_instrument_metrics(commentary)

    def _helper_method(self):
        """
        Example helper method. All internal logic should be _ prefixed.
        """
        pass
```
