"""Coinbase broker order results must reflect whether the order was really accepted."""

from __future__ import annotations

import logging
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from trading_bot.coinbase_broker import CoinbaseBroker
from trading_bot.models import PortfolioState


def _broker(paper_mode: bool, response=None, error=None) -> CoinbaseBroker:
    client = MagicMock()
    client.to_coinbase_symbol.return_value = "BTC-USD"
    broker = CoinbaseBroker(api_key="k", api_secret="s", coinbase_client=client,
                            paper_mode=paper_mode, logger=logging.getLogger("t"))
    broker._post = MagicMock(return_value=response, side_effect=error)
    return broker


def _buy(broker):
    return broker.place_market_buy(PortfolioState(cash=0.0), binance_symbol="BTCUSDT", notional_usd=10.0,
                                   strategy="s", now=datetime.now(timezone.utc))


class CoinbaseOrderResultTests(unittest.TestCase):
    def test_paper_buy_reports_ok_without_calling_the_api(self):
        b = _broker(True)
        self.assertTrue(_buy(b))
        b._post.assert_not_called()

    def test_live_buy_success(self):
        self.assertTrue(_buy(_broker(False, {"success": True, "order_id": "1"})))

    def test_live_buy_rejected_with_200_is_a_failure(self):
        self.assertFalse(_buy(_broker(False, {"success": False, "failure_reason": "INSUFFICIENT_FUND"})))

    def test_live_buy_exception_is_a_failure(self):
        self.assertFalse(_buy(_broker(False, error=RuntimeError("500"))))

    def test_by_product_buy_returns_none_when_rejected(self):
        b = _broker(False, {"success": False, "failure_reason": "X"})
        self.assertIsNone(b.place_market_buy_by_product(product_id="SOL-USD", notional_usd=5.0))

    def test_by_product_buy_returns_result_on_success(self):
        b = _broker(False, {"success": True, "order_id": "9"})
        self.assertEqual(b.place_market_buy_by_product(product_id="SOL-USD", notional_usd=5.0)["order_id"], "9")

    def test_live_sell_reflects_success_flag(self):
        self.assertTrue(_broker(False, {"success": True}).place_market_sell_by_product(product_id="SOL-USD", quantity=1.0))
        self.assertFalse(_broker(False, {"success": False}).place_market_sell_by_product(product_id="SOL-USD", quantity=1.0))


if __name__ == "__main__":
    unittest.main()
