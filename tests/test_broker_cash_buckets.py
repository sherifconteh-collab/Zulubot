from __future__ import annotations

import logging
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.models import PortfolioState


class BrokerCashBucketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_broker_cash_buckets")
        self.broker = PaperBroker(True, self.logger)
        self.now = datetime(2026, 3, 31, 12, 0, tzinfo=ZoneInfo("UTC"))

    def test_open_and_close_position_uses_named_cash_bucket(self) -> None:
        state = PortfolioState(cash=100.0, cash_reserves={"kalshi": 25.0})

        position = self.broker.open_position(
            state,
            symbol="KL:TEST:YES",
            asset_type="prediction",
            strategy="coinbase_predictions",
            price=0.50,
            notional=10.0,
            opened_at=self.now,
            cash_bucket="kalshi",
            meta={"venue": "kalshi"},
        )

        self.assertIsNotNone(position)
        self.assertEqual(state.cash, 100.0)
        self.assertAlmostEqual(state.cash_reserves["kalshi"], 15.0, places=4)

        self.broker.close_position(
            state,
            symbol="KL:TEST:YES",
            price=0.60,
            reason="take_profit",
        )

        self.assertEqual(state.cash, 100.0)
        self.assertAlmostEqual(state.cash_reserves["kalshi"], 27.0, places=4)