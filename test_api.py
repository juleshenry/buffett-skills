import os
import requests
import json

from src.evaluator_config import EARNINGSCALLS_API_BASE_URL, EARNINGSCALLS_API_KEY

if __name__ == '__main__':
    if not EARNINGSCALLS_API_KEY:
        print("No key!")
    else:
        url = f"{EARNINGSCALLS_API_BASE_URL}/speakers/27352"
        print(f"Requesting {url}")
        resp = requests.get(url, headers={"X-API-Key": EARNINGSCALLS_API_KEY})
        print(resp.status_code)
        try:
            print(json.dumps(resp.json(), indent=2)[:500])
        except Exception as e:
            print(e)
