import json
from src.management_governance import ManagementEvaluation, CorporateCulture
from src.valuation_capital import ShareBuybackAnalysis

if __name__ == '__main__':
    print("\n--- Testing ManagementEvaluation ---")
    try:
        me = ManagementEvaluation()
        me_result = me.evaluate("AAPL")
        print(me_result)
    except Exception as e:
        print("Error:", e)

    print("\n--- Testing ShareBuybackAnalysis ---")
    try:
        sba = ShareBuybackAnalysis()
        sba_result = sba.evaluate("AAPL")
        print(json.dumps(sba_result, indent=2))
    except Exception as e:
        print("Error:", e)

