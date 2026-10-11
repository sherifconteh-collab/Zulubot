from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from trading_bot.integrations import nl_router, slip_reader
from trading_bot.integrations.anthropic_errors import explain_api_failure

_BILLING = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
)


class _ApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ExplainApiFailureTests(unittest.TestCase):
    def test_billing_key_rate_limit_and_overload_are_explained(self):
        self.assertIn("no credit", explain_api_failure(_ApiError(_BILLING, 400)))
        self.assertIn("rejected the API key", explain_api_failure(_ApiError("nope", 401)))
        self.assertIn("rate limiting", explain_api_failure(_ApiError("slow down", 429)))
        self.assertIn("overloaded", explain_api_failure(_ApiError("Overloaded", 529)))

    def test_other_errors_are_left_to_the_caller(self):
        self.assertIsNone(explain_api_failure(ConnectionError("down")))
        self.assertIsNone(explain_api_failure(ValueError("no tool call in the reply")))


class CallersUseItTests(unittest.TestCase):
    def _failing(self, exc):
        client = MagicMock()
        client.messages.create.side_effect = exc
        return client

    def test_router_reports_billing_instead_of_not_understanding(self):
        client = self._failing(_ApiError(_BILLING, 400))
        with self.assertRaises(RuntimeError) as ctx:
            nl_router.route_message("bet $10", pending=None, api_key="", model="m", client=client)
        self.assertIn("no credit", str(ctx.exception))
        self.assertNotIn("work out", str(ctx.exception))

    def test_slip_reader_reports_billing_without_dumping_the_raw_error(self):
        client = self._failing(_ApiError(_BILLING, 400))
        with self.assertRaises(RuntimeError) as ctx:
            slip_reader.read_slip(b"x", "image/png", api_key="", model="m", client=client)
        self.assertIn("no credit", str(ctx.exception))
        self.assertNotIn("request_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
