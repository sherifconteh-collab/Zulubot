"""Kalshi combo + request-for-quote client calls (HTTP mocked)."""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock

from trading_bot.clients_kalshi import KalshiClient


def _client() -> KalshiClient:
    c = KalshiClient(logger=logging.getLogger("t"))
    c._post = MagicMock(return_value={"id": "rfq-9", "market_ticker": "KXMVE-1", "event_ticker": "KXMVE-EV"})
    c._get = MagicMock(return_value={})
    c._put = MagicMock(return_value={})
    c._delete = MagicMock(return_value={})
    return c


class ComboClientTests(unittest.TestCase):
    def test_combo_market_is_created_in_the_collection(self):
        c = _client()
        legs = [{"market_ticker": "A-1", "event_ticker": "A", "side": "yes"},
                {"market_ticker": "B-1", "event_ticker": "B", "side": "no"}]
        c.create_combo_market("COL-R", legs)
        path, payload = c._post.call_args.args
        self.assertEqual(path, "/multivariate_event_collections/COL-R")
        self.assertEqual(payload, {"selected_markets": legs})

    def test_rfq_is_sized_in_whole_contracts_and_never_rests(self):
        c = _client()
        self.assertEqual(c.create_rfq("KXMVE-1", 12), "rfq-9")
        path, payload = c._post.call_args.args
        self.assertEqual(path, "/communications/rfqs")
        self.assertEqual(payload, {"market_ticker": "KXMVE-1", "contracts_fp": "12.00", "rest_remainder": False})

    def test_quotes_are_read_by_rfq_with_auth(self):
        c = _client()
        c._get.return_value = {"quotes": [{"id": "q"}]}
        self.assertEqual(c.get_rfq_quotes("rfq-9"), [{"id": "q"}])
        args, kwargs = c._get.call_args.args, c._get.call_args.kwargs
        self.assertEqual(args[0], "/communications/quotes")
        self.assertEqual(args[1], {"rfq_id": "rfq-9"})
        self.assertTrue(kwargs["auth"])

    def test_single_quote_unwraps(self):
        c = _client()
        c._get.return_value = {"quote": {"status": "executed"}}
        self.assertEqual(c.get_rfq_quote("r", "q")["status"], "executed")
        self.assertEqual(c._get.call_args.args[0], "/communications/rfqs/r/quotes/q")

    def test_accept_is_a_put_with_the_side(self):
        c = _client()
        c.accept_rfq_quote("r", "q", "no")
        self.assertEqual(c._put.call_args.args, ("/communications/rfqs/r/quotes/q/accept", {"accepted_side": "no"}))

    def test_rfq_can_be_withdrawn(self):
        c = _client()
        c.delete_rfq("r")
        c._delete.assert_called_once_with("/communications/rfqs/r")

    def test_collections_are_paged(self):
        c = _client()
        c._get.side_effect = [
            {"multivariate_contracts": [{"collection_ticker": "A"}], "cursor": "next"},
            {"multivariate_contracts": [{"collection_ticker": "B"}], "cursor": ""},
        ]
        self.assertEqual([x["collection_ticker"] for x in c.get_multivariate_collections()], ["A", "B"])

    def test_put_requires_credentials(self):
        c = KalshiClient(logger=logging.getLogger("t"))
        with self.assertRaises(RuntimeError):
            c._put("/x", {})


if __name__ == "__main__":
    unittest.main()
