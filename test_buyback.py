import json
from src.valuation_capital import ShareBuybackAnalysis

if __name__ == '__main__':
    print("\n--- Testing ShareBuybackAnalysis ---")
    try:
        sba = ShareBuybackAnalysis()
        # It fetches the MD&A section from the 10-K, which is typically ~15k chars.
        sba_result = sba.evaluate("AAPL")
        print(json.dumps(sba_result, indent=2))
    except Exception as e:
        print("Error:", e)

