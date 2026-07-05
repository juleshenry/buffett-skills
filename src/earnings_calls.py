from cache_utils import disk_cache
import json
from pathlib import Path
from typing import Any

import requests

from evaluator_config import EARNINGSCALLS_API_BASE_URL, EARNINGSCALLS_API_KEY


ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "output" / "earnings_calls"
SP500_TICKERS_PATH = ROOT_DIR / "sp500_tickers.json"
DEFAULT_CALLS_PER_TICKER = 8  # ~2 years of quarterly calls

# Raw API responses (call history + speaker segments) cached separately from
# the generic `.cache/` dir and committed to git. The earningscalls.dev API
# quota is capped at 5,000 requests/month, and every response here represents
# an immutable historical transcript, so this cache never expires and must
# never be dropped from version control (see .gitignore carve-out).
RAW_RESPONSE_CACHE_DIR = ROOT_DIR / ".cache" / "earnings_calls"


def _cache_path_for_ticker(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.json"


def load_sp500_tickers() -> list[str]:
    if not SP500_TICKERS_PATH.exists():
        raise FileNotFoundError("sp500_tickers.json not found. Run src/get_sp500.py first.")
    return json.loads(SP500_TICKERS_PATH.read_text(encoding="utf-8"))


def load_cached_transcripts(ticker: str) -> list[dict[str, Any]]:
    cache_path = _cache_path_for_ticker(ticker)
    if not cache_path.exists():
        return []
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _extract_call_id(call: dict[str, Any]) -> Any:
    return call.get("earnings_call_id") or call.get("id")


def _extract_event_datetime(call: dict[str, Any]) -> str:
    return str(call.get("event_date_time") or call.get("date") or "")


def _normalize_speaker_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for segment in segments:
        normalized.append(
            {
                "speaker_name": segment.get("speaker_name") or segment.get("speaker") or "Unknown",
                "speaker_type": segment.get("speaker_type") or segment.get("role") or "unknown",
                "text_content": segment.get("text_content") or segment.get("text") or "",
                "component_order": segment.get("component_order"),
            }
        )
    return normalized


def _segments_to_text(segments: list[dict[str, Any]]) -> str:
    lines = []
    ordered_segments = sorted(
        segments,
        key=lambda segment: (
            segment.get("component_order") is None,
            segment.get("component_order") or 0,
        ),
    )
    for segment in ordered_segments:
        speaker = segment.get("speaker_name") or "Unknown"
        text = (segment.get("text_content") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n\n".join(lines)


def save_cached_transcripts(ticker: str, transcripts: list[dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path_for_ticker(ticker).write_text(json.dumps(transcripts, indent=2), encoding="utf-8")


def save_transcripts_to_cache(ticker: str, transcripts: list[dict[str, Any]]) -> None:
    save_cached_transcripts(ticker, transcripts)


@disk_cache(days=3650, cache_dir=RAW_RESPONSE_CACHE_DIR)
def _request_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if not EARNINGSCALLS_API_KEY:
        raise RuntimeError("EARNINGSCALLS_API_KEY is not set")

    response = requests.get(
        url,
        params=params,
        headers={"X-API-Key": EARNINGSCALLS_API_KEY},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _fetch_company_call_history(ticker: str) -> list[dict[str, Any]]:
    data = _request_json(f"{EARNINGSCALLS_API_BASE_URL}/companies/ticker/{ticker.upper()}")
    company_data = data.get("data", {}) if isinstance(data, dict) else {}
    call_history = company_data.get("earnings_calls", [])
    if not isinstance(call_history, list):
        return []
    return call_history


def _is_earnings_call(call: dict[str, Any]) -> bool:
    # The API mixes real quarterly earnings calls in with conference
    # presentations, special events, etc. under the same history endpoint.
    # Only "earnings" events count toward the requested N earnings calls.
    event_type = str(call.get("event_type") or "").strip().lower()
    if event_type:
        return event_type == "earnings"
    title = str(call.get("transcript_title") or call.get("title") or "").lower()
    return "earnings" in title


def _fetch_speaker_segments(call_id: Any) -> list[dict[str, Any]]:
    data = _request_json(f"{EARNINGSCALLS_API_BASE_URL}/speakers/{call_id}")
    company_data = data.get("data", {}) if isinstance(data, dict) else {}
    segments = company_data.get("segments", [])
    if not isinstance(segments, list):
        return []
    return _normalize_speaker_segments(segments)


def _build_cached_call_record(ticker: str, call: dict[str, Any]) -> dict[str, Any]:
    call_id = _extract_call_id(call)
    segments = _fetch_speaker_segments(call_id) if call_id is not None else []
    return {
        "ticker": ticker.upper(),
        "earnings_call_id": call_id,
        "event_date_time": _extract_event_datetime(call),
        "transcript_title": call.get("transcript_title") or call.get("title") or "",
        "company_name": call.get("company_name") or "",
        "speaker_segments": segments,
        "transcript_text": _segments_to_text(segments),
    }


def fetch_transcripts_for_ticker(ticker: str, limit: int = DEFAULT_CALLS_PER_TICKER, force_refresh: bool = False) -> list[dict[str, Any]]:
    cached = load_cached_transcripts(ticker)
    if cached and len(cached) >= limit and not force_refresh:
        return cached[:limit]

    call_history = [call for call in _fetch_company_call_history(ticker) if _is_earnings_call(call)]
    transcripts = [_build_cached_call_record(ticker, call) for call in call_history[:limit]]
    save_cached_transcripts(ticker, transcripts)
    return transcripts


def load_cached_transcript_text(ticker: str, limit: int = DEFAULT_CALLS_PER_TICKER) -> str:
    cached_calls = load_cached_transcripts(ticker)[:limit]
    if not cached_calls:
        return ""

    call_sections = []
    for call in cached_calls:
        event_date = call.get("event_date_time") or "unknown-date"
        title = call.get("transcript_title") or f"{ticker.upper()} earnings call"
        transcript_text = (call.get("transcript_text") or "").strip()
        if transcript_text:
            call_sections.append(f"{event_date} | {title}\n\n{transcript_text}")

    return "\n\n====\n\n".join(call_sections)


def fetch_recent_calls_for_sp500(force_refresh: bool = False) -> dict[str, Any]:
    tickers = load_sp500_tickers()
    fetched = 0
    cached = 0
    failures: dict[str, str] = {}

    for ticker in tickers:
        existing = load_cached_transcripts(ticker)
        if existing and len(existing) >= DEFAULT_CALLS_PER_TICKER and not force_refresh:
            cached += 1
            continue
        try:
            fetch_transcripts_for_ticker(ticker, limit=DEFAULT_CALLS_PER_TICKER, force_refresh=force_refresh)
            fetched += 1
        except Exception as exc:
            failures[ticker] = str(exc)

    return {
        "tickers_total": len(tickers),
        "tickers_fetched": fetched,
        "tickers_cached": cached,
        "tickers_failed": len(failures),
        "failures": failures,
        "calls_per_ticker": DEFAULT_CALLS_PER_TICKER,
        "cache_dir": str(CACHE_DIR),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)

    if args.ticker:
        transcripts = fetch_transcripts_for_ticker(args.ticker, force_refresh=args.force_refresh)
        print(json.dumps({"ticker": args.ticker.upper(), "count": len(transcripts), "cache_path": str(_cache_path_for_ticker(args.ticker))}, indent=2))
        return

    summary = fetch_recent_calls_for_sp500(force_refresh=args.force_refresh)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
