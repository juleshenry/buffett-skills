import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import requests

import sec_data


class TestSecGetRetryAndThrottle(unittest.TestCase):
    """
    Locks in the retry/backoff/throttle contract for _sec_get, the single
    choke point all SEC EDGAR requests in sec_data.py go through. Without
    this, a 500-ticker batch run has no protection against SEC's 10 req/sec
    fair-use throttling and no resilience against momentary server errors.
    """

    def setUp(self):
        # Reset the module-level throttle clock so timing from one test
        # can't bleed into another.
        sec_data._last_request_time = 0.0

    @patch("sec_data.time.sleep")
    @patch("sec_data.requests.get")
    def test_retries_on_rate_limit_then_succeeds(self, mock_get, mock_sleep):
        rate_limited_response = MagicMock(status_code=429)
        ok_response = MagicMock(status_code=200)
        ok_response.raise_for_status.return_value = None
        mock_get.side_effect = [rate_limited_response, ok_response]

        result = sec_data._sec_get("https://example.com/filing")

        self.assertIs(result, ok_response)
        self.assertEqual(mock_get.call_count, 2)

    @patch("sec_data.time.sleep")
    @patch("sec_data.requests.get")
    def test_retries_on_server_error_then_succeeds(self, mock_get, mock_sleep):
        server_error_response = MagicMock(status_code=503)
        ok_response = MagicMock(status_code=200)
        ok_response.raise_for_status.return_value = None
        mock_get.side_effect = [server_error_response, server_error_response, ok_response]

        result = sec_data._sec_get("https://example.com/filing")

        self.assertIs(result, ok_response)
        self.assertEqual(mock_get.call_count, 3)

    @patch("sec_data.time.sleep")
    @patch("sec_data.requests.get")
    def test_raises_after_exhausting_all_retry_attempts(self, mock_get, mock_sleep):
        mock_get.return_value = MagicMock(status_code=503)

        with self.assertRaises(requests.exceptions.HTTPError):
            sec_data._sec_get("https://example.com/filing")

        self.assertEqual(mock_get.call_count, sec_data._MAX_ATTEMPTS)

    @patch("sec_data.time.sleep")
    @patch("sec_data.requests.get")
    def test_does_not_retry_non_retryable_status(self, mock_get, mock_sleep):
        not_found = MagicMock(status_code=404)
        not_found.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = not_found

        with self.assertRaises(requests.exceptions.HTTPError):
            sec_data._sec_get("https://example.com/filing")

        self.assertEqual(mock_get.call_count, 1)

    @patch("sec_data.time.sleep")
    @patch("sec_data.requests.get")
    def test_retries_on_connection_error_then_succeeds(self, mock_get, mock_sleep):
        ok_response = MagicMock(status_code=200)
        ok_response.raise_for_status.return_value = None
        mock_get.side_effect = [requests.exceptions.ConnectionError("boom"), ok_response]

        result = sec_data._sec_get("https://example.com/filing")

        self.assertIs(result, ok_response)
        self.assertEqual(mock_get.call_count, 2)

    @patch("sec_data.time.monotonic")
    @patch("sec_data.time.sleep")
    @patch("sec_data.requests.get")
    def test_throttle_sleeps_when_requests_are_too_close_together(self, mock_get, mock_sleep, mock_monotonic):
        ok_response = MagicMock(status_code=200)
        ok_response.raise_for_status.return_value = None
        mock_get.return_value = ok_response

        # First call establishes _last_request_time at t=0; second call
        # happens "instantly" after (t=0 again), so the throttle should sleep
        # for the full minimum interval before issuing the second request.
        mock_monotonic.side_effect = [0.0, 0.0, 0.0, 0.0]

        sec_data._sec_get("https://example.com/filing")
        mock_sleep.reset_mock()
        sec_data._sec_get("https://example.com/filing")

        mock_sleep.assert_any_call(sec_data._MIN_SECONDS_BETWEEN_REQUESTS)


if __name__ == "__main__":
    unittest.main()
