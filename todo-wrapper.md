# Input Wrapper Conversion Checklist

Goal: convert the current `input_wrapper` evaluators into real end-to-end evaluators that fetch or derive their own evidence from a ticker, without placeholder defaults.

## Definition of done

- `evaluate(ticker=...)` works without fabricated defaults.
- Real evidence is fetched or derived inside the evaluator or a dedicated helper.
- Missing evidence fails explicitly instead of silently substituting guessed values.
- `python3 src/coverage_check.py` reclassifies the evaluator from `input_wrapper` to `end_to_end`.
- A focused test covers the happy path and the missing-data path.

## 1. Focus Investing

File: `src/investment_philosophy.py`

- [ ] Decide the real evidence source for portfolio concentration.
- [ ] If this heuristic is meant for an investor portfolio rather than a company ticker, decide whether it should stay caller-input only.
- [ ] If it should remain caller-input only, document that and exclude it from end-to-end coverage goals.
- [ ] If it should become end-to-end, add a real fetch path for holdings data.
- [ ] Remove the current `[1.0]` fallback for `ticker`.
- [ ] Add tests for real holdings input or explicit unsupported-ticker failure.

## 2. Independent Thinking

File: `src/thinking_frameworks.py`

- [ ] Define a real source for consensus view.
- [ ] Define how to measure `thesis_differs_from_consensus` from real evidence.
- [ ] Define how to derive `evidence_strength` from filings, metrics, or research artifacts.
- [ ] Define how to calculate `valuation_gap` from market price versus intrinsic value output.
- [ ] Remove placeholder defaults for `False`, `0.5`, and `0.0`.
- [ ] Add tests for a fully derived path and explicit failure when consensus evidence is unavailable.

## 3. Patience as Edge

File: `src/thinking_frameworks.py`

- [ ] Decide whether this heuristic is evaluating an investor behavior record or a company.
- [ ] If investor-only, document that and consider excluding it from ticker end-to-end coverage.
- [ ] If ticker-based, define real proxies for `avg_holding_period_years`, `turnover_ratio`, and `forced_activity`.
- [ ] Remove the current generic defaults of `5.0`, `0.20`, and `False`.
- [ ] Fail explicitly when no real proxy exists.
- [ ] Add tests for the chosen path.

## 4. Look-Through Earnings

File: `src/financial_metrics.py`

- [ ] Add a ticker-based fetch path for material equity investees where available.
- [ ] Define the source for `ownership_percentage`.
- [ ] Define the source for `investee_net_income`.
- [ ] Define the source for `dividends_received`.
- [ ] If company-level automatic derivation is not reliable, document that this heuristic requires structured investee inputs.
- [ ] Add tests for both derived or explicitly-required-input behavior.

## 5. Special Investment Instruments

File: `src/valuation_capital.py`

- [ ] Define the real evidence source for preferreds, converts, warrants, rescue financings, or structured deals.
- [ ] Add a ticker-based fetch or filing parser for `coupon_rate`, `conversion_discount`, and `collateral_coverage` when such instruments exist.
- [ ] Remove the current `0.05`, `0.0`, and `1.0` defaults.
- [ ] Return an explicit "not applicable" or raise a clear error when the company has no such instrument.
- [ ] Add tests for applicable and not-applicable cases.

## 6. When to Sell (Clear Criteria)

File: `src/risk_behavior.py`

- [ ] Define which signals can be inferred automatically versus which remain investor-judgment inputs.
- [ ] Add a real derivation path for `extreme_overvaluation` using current price versus intrinsic value output.
- [ ] Add a real derivation path for `balance_sheet_deterioration` using fetched leverage, liquidity, or coverage trends.
- [ ] Decide whether `thesis_broken` and `better_opportunity_available` can be made evidence-based or should remain explicit caller inputs.
- [ ] Remove the current all-`False` fallback defaults.
- [ ] Add tests for derived sell flags and required-manual-input behavior.

## 7. Common Behavioral Biases (Psychological Traps in Investing)

File: `src/risk_behavior.py`

- [ ] Decide whether this is fundamentally an investor-behavior heuristic rather than a company heuristic.
- [ ] If investor-only, document that and exclude it from ticker end-to-end goals.
- [ ] If an automated path is still desired, define a real event/history source for price-driven thesis changes and averaging-down behavior.
- [ ] Remove the current defaults for `False`, `5.0`, and `False`.
- [ ] Fail explicitly when behavioral evidence is unavailable.
- [ ] Add tests for whichever scope is chosen.

## Cross-cutting decisions

- [ ] Decide which wrappers should truly become ticker end-to-end evaluators versus remain caller-supplied investor heuristics.
- [ ] For investor-only heuristics, decide whether `coverage_check.py` should classify them separately from `input_wrapper`.
- [ ] Add a small status note to `plan.md` after each wrapper is resolved.
- [ ] Re-run `python3 src/coverage_check.py` after each conversion.
