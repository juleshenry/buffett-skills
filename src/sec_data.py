from cache_utils import disk_cache
import html
import re
import time
from typing import Dict, Iterable, Optional

import requests

from evaluator_config import DEFAULT_SEC_USER_AGENT


TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dash}/{primary_doc}"
INTERNAL_LINK_RE = re.compile(r"<a\b[^>]*href=[\"']#([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)

# SEC's fair-use policy caps automated access at 10 requests/second and will
# throttle or block User-Agents that exceed it. A 500-ticker batch run, with
# each ticker issuing 10+ SEC requests (10-K, 10-Q, 8-K, DEF 14A, plus
# multiple keyword-context searches per form), has no trouble tripping that
# limit if nothing paces the requests.
_MIN_SECONDS_BETWEEN_REQUESTS = 0.15  # ~6-7 req/sec, safely under the 10/sec cap
_last_request_time = 0.0

# Retried instead of treated as fatal: 429/403 are SEC's rate-limit/throttle
# signals, 5xx are transient server errors. Anything else (404, malformed
# request, etc.) is a real error and should raise immediately.
_RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4
_INITIAL_BACKOFF_SECONDS = 1.0


def get_sec_headers() -> Dict[str, str]:
    return {"User-Agent": DEFAULT_SEC_USER_AGENT}


def _throttle() -> None:
    """Enforces a minimum delay between outgoing SEC EDGAR requests."""
    global _last_request_time
    wait_time = _MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request_time)
    if wait_time > 0:
        time.sleep(wait_time)
    _last_request_time = time.monotonic()


def _sec_get(url: str, **kwargs) -> requests.Response:
    """
    Throttled, retrying wrapper around requests.get for every SEC EDGAR call
    in this module. Retries transient failures (rate limiting, momentary
    server errors) with exponential backoff instead of letting one flaky
    request take down an entire batch run across hundreds of tickers.
    """
    kwargs.setdefault("headers", get_sec_headers())
    kwargs.setdefault("timeout", 60)

    last_exception: Optional[Exception] = None
    for attempt in range(_MAX_ATTEMPTS):
        _throttle()
        try:
            response = requests.get(url, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_exception = exc
        else:
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response
            last_exception = requests.exceptions.HTTPError(
                f"Retryable status {response.status_code} from {url}"
            )

        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_INITIAL_BACKOFF_SECONDS * (2 ** attempt))

    raise last_exception


@disk_cache()
def _fetch_ticker_map() -> dict:
    """
    Downloads SEC's full ticker->CIK directory exactly once (subject to the
    disk cache TTL) instead of once per ticker. Before this, every one of the
    500 S&P tickers re-downloaded the same several-MB JSON file just to look
    itself up in it.
    """
    response = _sec_get(TICKER_MAP_URL)
    return response.json()


def get_cik_from_ticker(ticker: str) -> str:
    data = _fetch_ticker_map()
    ticker_upper = ticker.upper()
    for value in data.values():
        if value["ticker"].upper() == ticker_upper:
            return str(value["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for ticker {ticker}")


def get_latest_filing_metadata(ticker: str, form: str = "10-K") -> Dict[str, str]:
    cik = get_cik_from_ticker(ticker)
    response = _sec_get(SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
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


def _strip_tags(raw_html: str) -> str:
    return re.sub(r"<[^>]+>", " ", raw_html)


def _normalize_for_matching(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("\xa0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _cleanup_extracted_section(text: str) -> str:
    if not text:
        return ""

    cleaned = re.sub(r"^table of contents\s+", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btable of contents\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _marker_positions(text: str, markers: Iterable[str]) -> list[int]:
    positions: list[int] = []
    search_text = _normalize_for_matching(text)

    for marker in markers:
        search_marker = _normalize_for_matching(marker)
        start = 0
        while True:
            idx = search_text.find(search_marker, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1

    return sorted(set(positions))


def _choose_start_index(text: str, start_markers: Iterable[str]) -> int:
    positions = _marker_positions(text, start_markers)
    if not positions:
        return 0

    text_length = len(text)
    floor = int(text_length * 0.1)
    later_positions = [pos for pos in positions if pos >= floor]
    candidates = later_positions or positions

    def score(position: int) -> tuple[int, int]:
        window_start = max(0, position - 200)
        window_end = min(text_length, position + 200)
        window = _normalize_for_matching(text[window_start:window_end])
        item_heading_score = 1 if re.search(r"item\s+\d+[a-z]?\.", window) else 0
        body_text_score = 1 if re.search(r"(the following discussion|results of operations|liquidity and capital resources)", window) else 0
        return (item_heading_score + body_text_score, position)

    return max(candidates, key=score)


def _choose_notes_start_index(text: str, start_markers: Iterable[str]) -> int:
    positions = _marker_positions(text, start_markers)
    if not positions:
        return 0

    text_length = len(text)
    floor = int(text_length * 0.1)
    later_positions = [pos for pos in positions if pos >= floor]
    candidates = later_positions or positions

    def score(position: int) -> tuple[int, int]:
        window_start = max(0, position - 200)
        window_end = min(text_length, position + 500)
        window = _normalize_for_matching(text[window_start:window_end])
        title_score = 1 if "notes to consolidated" in window or "notes to financial statements" in window else 0
        note_body_score = 1 if re.search(r"note\s+1[\s\.-]", window) else 0
        accounting_score = 1 if "summary of significant accounting policies" in window or "description of business" in window else 0
        cross_ref_penalty = -1 if "included in item 8" in window or "can be found in" in window else 0
        return (title_score + note_body_score + accounting_score + cross_ref_penalty, position)

    return max(candidates, key=score)


def _choose_end_index(text: str, start_index: int, end_markers: Optional[Iterable[str]]) -> int:
    if not end_markers:
        return len(text)

    positions = _marker_positions(text, end_markers)
    later_positions = [pos for pos in positions if pos > start_index]
    if later_positions:
        return later_positions[0]

    return len(text)


@disk_cache()
def fetch_latest_filing_text(ticker: str, form: str = "10-K") -> str:
    metadata = get_latest_filing_metadata(ticker, form=form)
    response = _sec_get(metadata["filing_url"])
    return _strip_html(response.text)


@disk_cache()
def fetch_latest_filing_html(ticker: str, form: str = "10-K") -> str:
    metadata = get_latest_filing_metadata(ticker, form=form)
    response = _sec_get(metadata["filing_url"])
    return response.text


def _collect_internal_anchor_links(raw_html: str) -> dict[str, set[str]]:
    anchors: dict[str, set[str]] = {}
    for target, link_html in INTERNAL_LINK_RE.findall(raw_html):
        link_text = _normalize_for_matching(html.unescape(_strip_tags(link_html)))
        if not link_text:
            continue
        anchors.setdefault(target, set()).add(link_text)
    return anchors


def _score_anchor_link_texts(link_texts: set[str], markers: Iterable[str]) -> int:
    score = 0
    normalized_markers = [_normalize_for_matching(marker) for marker in markers]
    for link_text in link_texts:
        for marker in normalized_markers:
            if link_text == marker:
                score += 3
            elif marker in link_text:
                score += 1
    return score


def _find_anchor_target(raw_html: str, markers: Iterable[str]) -> Optional[str]:
    anchors = _collect_internal_anchor_links(raw_html)
    best_target = None
    best_score = 0

    for target, link_texts in anchors.items():
        score = _score_anchor_link_texts(link_texts, markers)
        if score > best_score:
            best_target = target
            best_score = score

    return best_target


def _find_anchor_position(raw_html: str, target: Optional[str]) -> Optional[int]:
    if not target:
        return None

    escaped_target = re.escape(target)
    patterns = [
        rf'id=[\"\']{escaped_target}[\"\']',
        rf'name=[\"\']{escaped_target}[\"\']',
        rf'<a\b[^>]*(?:id|name)=[\"\']{escaped_target}[\"\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_html, flags=re.IGNORECASE)
        if match:
            tag_start = raw_html.rfind("<", 0, match.start())
            return tag_start if tag_start != -1 else match.start()
    return None


def _extract_section_from_html(
    raw_html: str,
    start_markers: Iterable[str],
    end_markers: Optional[Iterable[str]] = None,
    max_chars: Optional[int] = None,
) -> str:
    start_target = _find_anchor_target(raw_html, start_markers)
    start_index = _find_anchor_position(raw_html, start_target)
    if start_index is None:
        return ""

    end_index = len(raw_html)
    if end_markers:
        end_target = _find_anchor_target(raw_html, end_markers)
        candidate_end_index = _find_anchor_position(raw_html, end_target)
        if candidate_end_index is not None and candidate_end_index > start_index:
            end_index = candidate_end_index

    section = _cleanup_extracted_section(_strip_html(raw_html[start_index:end_index]).strip())
    if max_chars is not None:
        return section[:max_chars]
    return section


def _looks_like_xbrl_blob(text: str) -> bool:
    if not text:
        return False

    sample = text[:1200].lower()
    indicators = (
        "http://fasb.org/us-gaap",
        "dei/",
        "ix:nonfraction",
        "otherassetsnoncurrent",
        "accruedliabilitiescurrent",
    )
    return sum(indicator in sample for indicator in indicators) >= 2


def extract_section(
    text: str,
    start_markers: Iterable[str],
    end_markers: Optional[Iterable[str]] = None,
    max_chars: Optional[int] = None,
    prefer_notes_body: bool = False,
) -> str:
    if not text:
        return ""

    start_index = _choose_notes_start_index(text, start_markers) if prefer_notes_body else _choose_start_index(text, start_markers)
    end_index = _choose_end_index(text, start_index, end_markers)

    if end_index <= start_index:
        end_index = len(text)

    section = text[start_index:end_index].strip()
    if max_chars is not None:
        return section[:max_chars]
    return section


def extract_keyword_context(
    text: str,
    keywords: Iterable[str],
    context_chars: int = 1200,
    max_matches: int = 3,
    max_chars: Optional[int] = None,
) -> str:
    if not text:
        return ""

    matches: list[tuple[int, int]] = []
    for keyword in keywords:
        if not keyword:
            continue
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end()))

    if not matches:
        return ""

    snippets: list[str] = []
    seen_windows: set[tuple[int, int]] = set()
    half_window = max(context_chars // 2, 1)

    for start, end in sorted(matches)[:max_matches]:
        window_start = max(0, start - half_window)
        window_end = min(len(text), end + half_window)
        window = (window_start, window_end)
        if window in seen_windows:
            continue
        seen_windows.add(window)
        snippet = _cleanup_extracted_section(text[window_start:window_end])
        if snippet:
            snippets.append(snippet)

    combined = "\n\n".join(snippets)
    if max_chars is not None:
        return combined[:max_chars]
    return combined


@disk_cache()
def fetch_filing_keyword_context(
    ticker: str,
    form: str,
    keywords: Iterable[str],
    context_chars: int = 1200,
    max_matches: int = 3,
    max_chars: Optional[int] = None,
) -> str:
    filing_text = fetch_latest_filing_text(ticker, form=form)
    return extract_keyword_context(
        filing_text,
        keywords=keywords,
        context_chars=context_chars,
        max_matches=max_matches,
        max_chars=max_chars,
    )


@disk_cache()
def fetch_filing_section(
    ticker: str,
    form: str,
    start_markers: Iterable[str],
    end_markers: Optional[Iterable[str]] = None,
    max_chars: Optional[int] = None,
    prefer_notes_body: bool = False,
) -> str:
    raw_html = fetch_latest_filing_html(ticker, form=form)
    html_section = _extract_section_from_html(
        raw_html,
        start_markers=start_markers,
        end_markers=end_markers,
        max_chars=max_chars,
    )
    if html_section and not _looks_like_xbrl_blob(html_section):
        return html_section

    filing_text = _strip_html(raw_html)
    fallback_section = extract_section(
        filing_text,
        start_markers=start_markers,
        end_markers=end_markers,
        max_chars=max_chars,
        prefer_notes_body=prefer_notes_body,
    )
    return _cleanup_extracted_section(fallback_section)
