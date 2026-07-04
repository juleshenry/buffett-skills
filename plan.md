# Hybrid API + Local LLM (Ollama) Execution Plan for 500 Companies

The core philosophy of this plan is **"Compute vs. Data."** We use APIs to fetch the raw financial data and SEC text, but we offload all the heavy natural language processing (NLP), summarization, and qualitative reasoning to Ollama running locally. Because we feed the local LLM the retrieved data as context, its internal knowledge cutoff doesn't matter.

This strategy saves massive amounts of premium API tokens while scaling easily to 500 companies.

## 01-thinking-frameworks.md (Circle of Competence & Mr. Market)
*   **API Task:** Fetch the company profile, SEC business description (Item 1 of 10-K), and segment breakdown (e.g., via SEC-API or FMP).
*   **Ollama Task (The "Simplicity" Test):** Feed the 10-K business description to Ollama and prompt it: *"Explain how this company makes money in exactly one paragraph. If the business model relies on complex derivatives, unproven biotech, or obscure financial engineering, flag it as 'Too Hard'."*
*   **Token Savings:** Reading and summarizing 500 dense SEC business descriptions would burn massive tokens. Ollama does this for free and outputs a boolean `is_simple` flag.

## 02-investment-philosophy.md (Compounding & Concentration)
*   **API/Code Task:** Fetch price history, dividends, and benchmark data (e.g., S&P 500, Yahoo Finance, Alpha Vantage). Calculate the 10-year CAGR.
*   **Ollama Task:** None needed here. This is pure math. Stick to Python scripts (Pandas) to compare the company's CAGR vs the S&P 500.

## 03-business-moat.md (Franchise vs. Commodity & Pricing Power)
*   **API Task:** Fetch 15-year Gross and Operating margins, plus CPI inflation data (QuickFS, FRED).
*   **Ollama Task (Qualitative Moat Classification):** Provide Ollama with a list of the company's products/services and its top 5 competitors. Prompt it: *"Based on these products and competitors, classify the moat into one of these buckets: Brand, Switching Costs, Network Effect, Low-Cost Producer, or Commodity (No Moat). Justify in 2 sentences."*
*   **Code Task:** Overlay margins onto inflation spikes mathematically to test for actual pricing power.

## 04-management-governance.md (Integrity & Capital Allocation)
*(This is where local AI processing saves you the most money and adds the most rigor)*
*   **API Task:** Scrape/fetch Earnings Call Transcripts and the SEC DEF 14A (Proxy Statement).
*   **Specialized NLP Task (Earnings Call Sentiment & Candor):** You are completely right—using a generative LLM for tone classification is a kludge. Instead, run specialized, lightweight models locally via Hugging Face `transformers`:
    *   Use **FinBERT** (ProsusAI/finbert) to analyze the sentiment (bullish/bearish/neutral) of management's answers during the Q&A.
    *   Use a **Zero-Shot Classifier** (e.g., `facebook/bart-large-mnli`) against specific labels like `"blaming macroeconomic factors"`, `"blaming supply chain"`, `"taking accountability"`, or `"admitting mistakes"`. This provides mathematically sound confidence scores rather than LLM hallucinations.
*   **Ollama Task (Proxy Parser):** Feed the "Executive Compensation" section of the DEF 14A to a local LLM. Prompt: *"Are executive bonuses tied to ROIC/Return on Equity, or are they tied purely to Revenue Growth/Adjusted EBITDA? Return a JSON evaluating the incentive structure."*

## 05-financial-metrics.md (Owner Earnings & ROIC)
*   **API Task:** Fetch deep financials and the Management Discussion & Analysis (MD&A) section of the 10-K (QuickFS).
*   **Ollama Task (Capex Extraction):** Standard financial APIs group all Capex together. Feed the MD&A text to Ollama and prompt: *"Read this MD&A and estimate what percentage of Capital Expenditures (Capex) was spent on 'Maintenance' (upkeeping current operations) versus 'Growth' (expanding operations). Extract any specific numbers mentioned regarding maintenance capex."*

## 06-valuation-capital.md (Intrinsic Value & Margin of Safety)
*   **API/Code Task:** Fetch Risk-Free Rate, shares outstanding, and historical cash flows.
*   **Ollama Task (Share Repurchase Context):** Instead of just looking at share counts, feed Ollama the recent management commentary on buybacks. Prompt: *"Is management buying back stock systematically, or only when the stock is cheap? Do they explicitly mention 'Intrinsic Value' as a benchmark for repurchases?"* The actual DCF valuation should be done in pure Python.

## 07-risk-behavior.md (Leverage, Survival, & Value Traps)
*   **API Task:** Fetch SEC 10-K Footnotes (specifically the ones on Leases, Pensions, and Commitments).
*   **Ollama Task (Footnote Detective):** Footnotes are notoriously long and boring. Batch feed them to Ollama. Prompt: *"Scan these footnotes for hidden liabilities. Return a JSON extracting: 1) Total operating lease obligations, 2) Pension plan underfunding amount, 3) Any toxic derivative exposure."*

## 08-industry-playbooks.md (Banks, Insurance, Utilities, Railroads, etc.)
*   **API Task:** Fetch industry-specific data (e.g., Insurance filings, Bank NIM).
*   **Ollama Task (Regulatory Parsing):** For banks, feed the latest regulatory/stress-test commentary to Ollama to summarize tier 1 capital adequacy risks. For insurance, feed the "Reserve Development" text to see if they are consistently underestimating future payouts (a sign of a bad insurance operator).

---

## Execution Strategy for 500 Companies

1. **The Retrieval Script (Python):** Write a script that hits the SEC EDGAR API, QuickFS, and SeekingAlpha APIs. It downloads the raw text (10-Ks, Transcripts) and saves them locally as `.txt` or `.json` files.
2. **The Local AI Pipeline (Ollama + Hugging Face):** Run a local script that iterates through the 500 companies. 
    *   For generative extraction (e.g., finding Maintenance Capex or classifying Moats), use the Ollama REST API (`localhost:11434`) with a model like `llama3`.
    *   For rigorous classification (Sentiment, Blame-shifting), use local Hugging Face pipelines (`FinBERT`, `BART-MNLI`) to score the transcripts mathematically.
3. **Structured Outputs:** Force Ollama and the NLP classifiers to output structured JSON. This allows you to append findings (e.g., `{"finbert_sentiment": 0.8, "blame_shift_score": 0.9, "moat_type": "Brand"}`) directly into a Pandas DataFrame alongside your quantitative API data.
4. **Final Premium Synthesis (Optional):** Once Ollama has processed the massive walls of text and you have your neat, structured DataFrame for all 500 companies, you can take the top 50 surviving companies and feed their condensed profiles to Claude 3.5 Sonnet or GPT-4o for a final, high-level Warren Buffett-style investment memo.

---

## Evaluator Hard-Coding Audit Plan

### Goal

Audit which evaluator implementations rely on hard-coded values, distinguish acceptable heuristic thresholds from risky embedded defaults or mock data, and define a cleanup order.

### What Counts As Hard-Coded

- Operational defaults embedded in evaluator code: model names, API hosts, time periods, benchmark symbols, temperatures.
- Stubbed or simulated data returned from evaluators.
- Fixed heuristic thresholds embedded inline instead of centralized in one place.
- Prompt criteria that bake scoring cutoffs into evaluator text.

### Highest Priority Issues

1. Evaluators using simulated or stubbed business data while presenting real analysis.
2. Repeated hard-coded Ollama URLs and model names across files.
3. Inline heuristic thresholds scattered across evaluator classes with no shared config.

### Evaluators With Hard-Coded Operational Defaults Or Mock Data

- `src/risk_behavior.py`
  - `LeverageRisk`: stubbed footnote text in `_fetch_sec_10k_footnotes` with fixed `$450 million` and `$300 million`.
  - `LeverageRisk`: hard-coded Ollama URL and model in `_analyze_footnotes_with_ollama`.

- `src/business_moat.py`
  - `EconomicMoat`: hard-coded Ollama URL and default model in `__init__`.
  - `EconomicMoat.evaluate`: default `model="qwen2.5:7b"`.

- `src/industry_playbooks.py`
  - `Insurance`: fixed prompt criteria including `Combined Ratio (< 100%)` and float growth.
  - `Insurance._query_ollama`: hard-coded Ollama URL and model.
  - `Banking`: fixed prompt criteria including `ROA > 1%` and `ROE > 10%`.
  - `Banking._query_ollama`: hard-coded Ollama URL and model.

- `src/financial_metrics.py`
  - `OwnerEarnings._fetch_mda_section`: simulated MD&A text with fixed `60%` growth / `40%` maintenance framing.
  - `OwnerEarnings._query_ollama_capex_breakdown`: hard-coded Ollama URL and model.
  - `OwnerEarnings._query_ollama_capex_breakdown`: hard-coded fallback `{maintenance_percentage: 100, growth_percentage: 0}`.

- `src/thinking_frameworks.py`
  - `CircleOfCompetence.evaluate`: default model, hard-coded Ollama URL, fixed `temperature: 0.2`.
  - `Inversion.evaluate`: default model, hard-coded Ollama URL, fixed `temperature: 0.3`.
  - `MrMarket.evaluate`: default period `"1y"`.

- `src/valuation_capital.py`
  - `IntrinsicValueEstimation.evaluate`: default `years=10`.
  - `ShareBuybackAnalysis.evaluate`: hard-coded Ollama URL and model.

### Summary of Implementation Status (The "Pure Math Wrapper" Problem)

Beyond simply moving hard-coded values into `evaluator_config.py` and `evaluator_thresholds.py`, a deeper audit reveals a more fundamental issue: **Out of 49 evaluators, ~38 are currently just "pure math wrappers" or stubs**. 

- They do not fetch SEC data.
- They do not parse text.
- They do not use Ollama or local LLMs.
- They are simply Python functions that require the caller to pass in exact, pre-calculated metrics (e.g., `evaluate(self, employee_turnover: float, insider_ownership: float, restructurings_per_5y: int)`).

Only about **7 evaluators** (such as `LeverageRisk` and `MrMarket`) are implemented end-to-end with actual data fetching. The rest are hard-coded heuristics waiting for a real data pipeline to feed them. Furthermore, the `coverage_check.py` script is fundamentally flawed as it only checks for the presence of an `evaluate` method, not whether that method actually implements the required fetching and NLP logic.

- `src/management_governance.py`
  - `ManagementEvaluation.__init__`: hard-coded model and host defaults.
  - `ManagementEvaluation._fetch_earnings_call_transcript`: simulated transcript text with fixed claims and `$2B` buyback.
  - `ManagementEvaluation._fetch_sec_def_14a`: simulated proxy text with fixed `60%` and `3-year` ROIC language.

- `src/investment_philosophy.py`
  - `UndervaluedMarginOfSafety.evaluate`: default `minimum_margin=0.25`.
  - `Compounding.evaluate`: defaults `years=10` and `benchmark="^GSPC"`.
  - `IntrinsicValue.evaluate`: default `years=10`.

### Evaluators With Hard-Coded Heuristic Thresholds

- `src/risk_behavior.py`
  - `ValueTraps`: `pe_ratio <= 12`, `debt_to_equity > 1.0`, `return_on_capital < 0.08`, `len(flags) >= 3`.
  - `TheImpactOfInflation`: inflation spike `> 3.0`, resilient margin change `>= -0.01`.
  - `DerivativesRisk`: exposure `> 3`, `> 1`; level 3 assets `> 0.15`, `> 0.05`.
  - `CommonBehavioralBiasesPsychologicalTrapsInInvesting`: holding period `< 1`, high risk if `len(flags) >= 2`.

- `src/business_moat.py`
  - `BusinessModelTypes`: `0.7`, `0.25`, `0.10`, `0.50`.
  - `GoodwillEconomicGoodwillVsAccountingGoodwill`: `<= 5`, `<= 8`, `>= 0.12`, `>= 0.08`.
  - `TheDurabilityOfCompetitiveAdvantage`: `return_on_capital >= 0.12`, score bands `3/2`.

- `src/industry_playbooks.py`
  - `UnderwritingDiscipline`: `< 95`, `< 100`.
  - `InsuranceFloat`: float growth `> 0`, combined ratio `< 100`.
  - `ConsumerBrandsRetail`: `gross_margin >= 0.35`, score bands `3/2`.
  - `MediaPublishing`: `subscription >= 0.5`, `ad <= 0.5`, `churn <= 0.10`.
  - `EnergyUtilities`: `regulated_asset_ratio >= 0.7`, `debt_to_ebitda <= 5`, `allowed_return_on_equity >= 0.09`.
  - `Railways`: `operating_ratio <= 0.65`, `maintenance_capex_ratio <= 0.6`.
  - `TechnologyInternet`: `recurring_revenue_ratio >= 0.6`, `net_revenue_retention >= 1.0`, `stock_comp_ratio <= 0.15`.
  - `IndustriesToAvoidCounterexamples`: `commodity_exposure >= 0.7`, `leverage_ratio >= 4`, `pricing_power <= 0.3`, avoid if `red_flags >= 2`.

- `src/thinking_frameworks.py`
  - `MrMarket`: fear at `-0.20` drawdown and `-0.10` vs 200d MA, greed at `-0.05` and `+0.10`, annualization constant `252`.
  - `LongtermOrientation`: `years >= 5`, `excess_return > 0.02`, `years >= 3`.
  - `MungersLatticeOfMentalModels`: `0.75`, `0.5`.
  - `IndependentThinking`: `evidence_strength >= 0.7`, `abs(valuation_gap) >= 0.15`.
  - `PatienceAsEdge`: holding period `>= 3`, turnover `<= 0.25`.

- `src/valuation_capital.py`
  - `TheRelationshipBetweenPurchasePriceAndIntrinsicValue`: `margin >= 0.25`, `>= 0.10`, `< 0`.
  - `CapitalAllocationAnalysis`: `fcf_to_debt >= 0.25`.
  - `SpecialInvestmentInstruments`: `coupon_rate >= 0.08`, `conversion_discount >= 0.10`, `collateral_coverage >= 1.0`.

- `src/management_governance.py`
  - `CorporateCulture`: `employee_turnover <= 0.15`, `insider_ownership >= 0.05`, `restructurings_per_5y <= 1`.
  - `AcquisitionLogicAcquisitionCriteria`: `purchase_multiple <= 12`, `return_on_invested_capital >= 0.12`.
  - `CorporateGovernanceAndShareholderOrientation`: `insider_ownership >= 0.05`, strong if score `>= 3`.

- `src/investment_philosophy.py`
  - `FocusInvesting`: top three weight `>= 0.50`.
  - `EfficientMarketTheory`: `abs(excess_return) <= 0.02`, `tracking_error <= 0.05`.
  - `MarketForecasting`: useful if `abs(forecast_error) <= 0.05`.

### Lower-Risk Evaluators

These rely mostly on direct input math and have fewer arbitrary embedded policies:

- `WhenToSellClearCriteria`
- `CorePrincipleSeeThroughAccountingToEconomicReality`
- `KeyFinancialMetrics`
- `LookthroughEarnings`
- `OpportunityCostAwareness`
- `MarginOfSafety`
- `DividendsRetainedEarningsAndTaxEfficiency`

### Cleanup Plan

1. Replace simulated evaluator inputs with explicit `NotImplementedError`, real fetchers, or caller-provided inputs.
2. Centralize shared operational defaults into one config module:
   - Ollama host
   - default model
   - temperature defaults
   - benchmark symbol
   - default lookback periods
3. Centralize evaluator thresholds into named constants grouped by domain:
   - risk behavior
   - moat
   - industry playbooks
   - management/governance
   - investment philosophy
4. Keep true heuristics, but make them explicit and reviewable from one file.
5. Add a short test per evaluator family to lock current threshold behavior before refactoring.

### Suggested Execution Order

1. `management_governance.py`
   - Highest risk because simulated transcript and proxy text can look like real evidence.
2. `financial_metrics.py`
   - Stubbed MD&A and fallback capex split can materially distort owner earnings.
3. `risk_behavior.py`
   - Stubbed footnotes and repeated Ollama defaults.
4. Shared Ollama config across all files.
5. Threshold centralization for remaining heuristic evaluators.

### Deliverables

- One shared config module for operational defaults.
- One thresholds module or per-domain threshold maps.
- Removal of simulated evaluator data paths.
- Tests proving evaluator outputs still match intended heuristics after centralization.
