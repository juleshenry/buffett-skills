import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import earnings_calls


class TestEarningsCalls(unittest.TestCase):
    def test_load_sp500_tickers_requires_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            with patch.object(earnings_calls, "SP500_TICKERS_PATH", fake_root / "missing.json"):
                with self.assertRaises(FileNotFoundError):
                    earnings_calls.load_sp500_tickers()

    def test_fetch_transcripts_uses_cache_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_payload = [{"id": i} for i in range(8)]
            (cache_dir / "AAPL.json").write_text(json.dumps(cached_payload), encoding="utf-8")

            with patch.object(earnings_calls, "CACHE_DIR", cache_dir):
                with patch.object(earnings_calls, "_request_json") as mock_request:
                    result = earnings_calls.fetch_transcripts_for_ticker("AAPL")
                    self.assertEqual(len(result), 8)
                    mock_request.assert_not_called()

    def test_fetch_recent_calls_for_sp500_counts_cached_and_fetched(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "earnings_calls"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "AAPL.json").write_text(json.dumps([{"id": i} for i in range(8)]), encoding="utf-8")

            with patch.object(earnings_calls, "CACHE_DIR", cache_dir):
                with patch.object(earnings_calls, "load_sp500_tickers", return_value=["AAPL", "MSFT"]):
                    with patch.object(earnings_calls, "fetch_transcripts_for_ticker", return_value=[{"id": 1}]):
                        summary = earnings_calls.fetch_recent_calls_for_sp500()
                        self.assertEqual(summary["tickers_total"], 2)
                        self.assertEqual(summary["tickers_cached"], 1)
                        self.assertEqual(summary["tickers_fetched"], 1)
                        self.assertEqual(summary["tickers_failed"], 0)

    def test_fetch_recent_calls_for_sp500_records_failures_without_aborting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "earnings_calls"
            cache_dir.mkdir(parents=True, exist_ok=True)

            def fake_fetch(ticker, limit=None, force_refresh=False):
                if ticker == "BAD":
                    raise RuntimeError("boom")
                return [{"id": 1}]

            with patch.object(earnings_calls, "CACHE_DIR", cache_dir):
                with patch.object(earnings_calls, "load_sp500_tickers", return_value=["AAPL", "BAD"]):
                    with patch.object(earnings_calls, "fetch_transcripts_for_ticker", side_effect=fake_fetch):
                        summary = earnings_calls.fetch_recent_calls_for_sp500()
                        self.assertEqual(summary["tickers_fetched"], 1)
                        self.assertEqual(summary["tickers_failed"], 1)
                        self.assertIn("BAD", summary["failures"])

    def test_load_cached_transcript_text_combines_calls_from_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            payload = [
                {
                    "event_date_time": "2025-07-01T00:00:00Z",
                    "transcript_title": "Q3 Call",
                    "transcript_text": "CEO: Growth remained strong.",
                },
                {
                    "event_date_time": "2025-04-01T00:00:00Z",
                    "transcript_title": "Q2 Call",
                    "transcript_text": "CFO: Margins expanded.",
                },
            ]
            (cache_dir / "AAPL.json").write_text(json.dumps(payload), encoding="utf-8")

            with patch.object(earnings_calls, "CACHE_DIR", cache_dir):
                combined = earnings_calls.load_cached_transcript_text("AAPL")
                self.assertIn("Q3 Call", combined)
                self.assertIn("CEO: Growth remained strong.", combined)
                self.assertIn("Q2 Call", combined)

    def test_load_cached_transcript_keyword_context_extracts_relevant_snippet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            payload = [
                {
                    "event_date_time": "2025-07-01T00:00:00Z",
                    "transcript_title": "Q3 Call",
                    "transcript_text": "CEO: Pricing remained strong and retention improved despite competition.",
                },
            ]
            (cache_dir / "AAPL.json").write_text(json.dumps(payload), encoding="utf-8")

            with patch.object(earnings_calls, "CACHE_DIR", cache_dir):
                excerpt = earnings_calls.load_cached_transcript_keyword_context(
                    "AAPL",
                    keywords=("pricing", "retention"),
                    context_chars=80,
                )
                self.assertIn("Pricing remained strong", excerpt)
                self.assertIn("retention improved", excerpt)


if __name__ == "__main__":
    unittest.main()
