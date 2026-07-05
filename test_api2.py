import os
import requests
import json
from src.evaluator_config import EARNINGSCALLS_API_BASE_URL, EARNINGSCALLS_API_KEY
if __name__ == '__main__':
    url = f"{EARNINGSCALLS_API_BASE_URL}/speakers/27352"
    resp = requests.get(url, headers={"X-API-Key": EARNINGSCALLS_API_KEY}).json()
    print("Keys in data:", resp["data"].keys())
