import html
import re
from typing import Dict, Iterable, Optional

import requests

from evaluator_config import DEFAULT_SEC_USER_AGENT


TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dash}/{primary_doc}"


def get_sec_headers() -> Dict[str, str]:
    return {"User-Agent": DEFAULT_SEC_USER_AGENT}


def get_cik_from_ticker(ticker: str) -> str:
    response = requests.get(TICKER_MAP_URL, headers=get_sec_headers(), timeout=60)
    response.raise_for_status()
    data = response.json()
    ticker_upper = ticker.upper()
    for value in data.values():
        if value["ticker"].upper() == ticker_upper:
            return str(value["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for ticker {ticker}")


def get_latest_filing_metadata(ticker: str, form: str = "10-K") -> Dict[str, str]:
    cik = get_cik_from_ticker(ticker)
    response = requests.get(
        SUBMISSIONS_URL_TEMPLATE.format(cik=cik),
        headers=get_sec_headers(),
        timeout=60,
    )
    response.raise_for_status()
    recent = response.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])

    for index, filing_form in enumerate(forms):
        if filing_form == form:
            accession_number = recent["accessionNumber"][index]
            primary_document = recent["primaryDocument"][index]
            accession_no_dash = accession_number.replace("-", "")
            filing_url = ARCHIVES_URL_TEMPLATE.format(
                cik_numeric=int(cik),
                accession_no_dash=accession_no_dash,
                primary_doc=primary_document,
            )
            return {
                "ticker": ticker.upper(),
                "cik": cik,
                "form": form,
                "accession_number": accession_number,
                "primary_document": primary_document,
                "filing_url": filing_url,
            }

    raise ValueError(f"No {form} filing found for ticker {ticker}")


def _strip_html(raw_html: str) -> str:
    without_scripts = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    without_styles = re.sub(r"<style.*?</style>", " ", without_scripts, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", without_styles)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_latest_filing_text(ticker: str, form: str = "10-K") -> str:
    metadata = get_latest_filing_metadata(ticker, form=form)
    response = requests.get(metadata["filing_url"], headers=get_sec_headers(), timeout=60)
    response.raise_for_status()
    return _strip_html(response.text)


def extract_section(
    text: str,
    start_markers: Iterable[str],
    end_markers: Optional[Iterable[str]] = None,
    max_chars: Optional[int] = None,
) -> str:
    if not text:
        return ""

    lower_text = text.lower()
    start_index = None
    for marker in start_markers:
        marker_index = lower_text.find(marker.lower())
        if marker_index != -1 and (start_index is None or marker_index < start_index):
            start_index = marker_index

    if start_index is None:
        start_index = 0

    end_index = len(text)
    if end_markers:
        for marker in end_markers:
            marker_index = lower_text.find(marker.lower(), start_index + 1)
            if marker_index != -1:
                end_index = min(end_index, marker_index)

    section = text[start_index:end_index].strip()
    if max_chars is not None:
        return section[:max_chars]
    return section


def fetch_filing_section(
    ticker: str,
    form: str,
    start_markers: Iterable[str],
    end_markers: Optional[Iterable[str]] = None,
    max_chars: Optional[int] = None,
) -> str:
    filing_text = fetch_latest_filing_text(ticker, form=form)
    return extract_section(
        filing_text,
        start_markers=start_markers,
        end_markers=end_markers,
        max_chars=max_chars,
    )
