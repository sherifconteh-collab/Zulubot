from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

import requests

from trading_bot.clients import request_with_retry
from trading_bot.config import HttpRetryConfig


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeSession:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def request(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class HttpRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_http_retry")
        self.retry_config = HttpRetryConfig(
            enabled=True,
            max_attempts=3,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
            backoff_multiplier=2.0,
        )

    @patch("trading_bot.clients.time.sleep", return_value=None)
    def test_retries_timeout_then_succeeds(self, _sleep) -> None:
        session = _FakeSession([
            requests.Timeout("timed out"),
            _FakeResponse(200),
        ])

        response = request_with_retry(
            session,  # type: ignore[arg-type]
            "GET",
            "https://example.com/data",
            timeout_seconds=5,
            logger=self.logger,
            retry_config=self.retry_config,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.calls, 2)

    @patch("trading_bot.clients.time.sleep", return_value=None)
    def test_retries_retryable_status_then_succeeds(self, _sleep) -> None:
        session = _FakeSession([
            _FakeResponse(503),
            _FakeResponse(200),
        ])

        response = request_with_retry(
            session,  # type: ignore[arg-type]
            "POST",
            "https://example.com/data",
            timeout_seconds=5,
            logger=self.logger,
            retry_config=self.retry_config,
            json={"ok": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.calls, 2)

    @patch("trading_bot.clients.time.sleep", return_value=None)
    def test_disabled_retry_propagates_first_failure(self, _sleep) -> None:
        session = _FakeSession([requests.Timeout("timed out")])

        with self.assertRaises(requests.Timeout):
            request_with_retry(
                session,  # type: ignore[arg-type]
                "GET",
                "https://example.com/data",
                timeout_seconds=5,
                logger=self.logger,
                retry_config=HttpRetryConfig(enabled=False),
            )

        self.assertEqual(session.calls, 1)


if __name__ == "__main__":
    unittest.main()