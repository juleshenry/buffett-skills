import json
from src.management_governance import CorporateCulture

if __name__ == '__main__':
    print("\n--- Testing CorporateCulture ---")
    try:
        cc = CorporateCulture()
        cc_result = cc.evaluate(ticker="AAPL")
        print(json.dumps(cc_result, indent=2))
    except Exception as e:
        print("Error:", e)
