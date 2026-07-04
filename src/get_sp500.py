import io
import json
from pathlib import Path

import pandas as pd
import requests


def main():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    tickers = [ticker.replace(".", "-") for ticker in tables[0]["Symbol"].tolist()]

    root_dir = Path(__file__).resolve().parent.parent
    (root_dir / "sp500_tickers.json").write_text(json.dumps(tickers, indent=2))
    (root_dir / "sp500_tickers.txt").write_text("\n".join(tickers))
    print(f"Successfully pulled {len(tickers)} tickers and saved to sp500_tickers.json and sp500_tickers.txt")


if __name__ == "__main__":
    main()
