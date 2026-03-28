"""Tests for CoinbasePriceFeed — history trimming, is_ready(), get_momentum_pct()."""

from __future__ import annotations

import logging
import time
import unittest
from collections import deque
from unittest.mock import MagicMock

from trading_bot.coinbase_feed import CoinbasePriceFeed


class FakeCoinbaseClient:
    """Fake CoinbaseClient whose get_spot_prices_by_product_ids returns preset prices."""

    def __init__(self, prices: dict[str, float] | None = None) -> None:
        self._prices = prices or {}

    def get_spot_prices_by_product_ids(self, product_ids: list[str]) -> dict[str, float]:
        return {pid: self._prices[pid] for pid in product_ids if pid in self._prices}


class CoinbasePriceFeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_coinbase_feed")

    def _make_feed(
        self,
        symbols: list[str] | None = None,
        history_seconds: float = 120.0,
        prices: dict[str, float] | None = None,
    ) -> CoinbasePriceFeed:
        client = FakeCoinbaseClient(prices)
        return CoinbasePriceFeed(
            client=client,
            symbols=symbols or ["BTC-USD", "ETH-USD"],
            history_seconds=history_seconds,
            logger=self.logger,
        )

    # ------------------------------------------------------------------
    # record_current_prices
    # ------------------------------------------------------------------

    def test_record_prices_stores_entries(self) -> None:
        feed = self._make_feed(prices={"BTC-USD": 70_000.0, "ETH-USD": 3_500.0})
        result = feed.record_current_prices()
        self.assertEqual(result, {"BTC-USD": 70_000.0, "ETH-USD": 3_500.0})
        self.assertEqual(feed.get_latest_price("BTC-USD"), 70_000.0)
        self.assertEqual(feed.get_latest_price("ETH-USD"), 3_500.0)

    def test_record_prices_skips_zero_or_negative(self) -> None:
        feed = self._make_feed(prices={"BTC-USD": 0.0, "ETH-USD": -1.0})
        result = feed.record_current_prices()
        # Both prices are invalid; nothing should be stored
        self.assertEqual(feed.get_latest_price("BTC-USD"), 0.0)
        self.assertEqual(feed.get_latest_price("ETH-USD"), 0.0)

    def test_record_prices_on_client_failure_returns_empty(self) -> None:
        client = MagicMock()
        client.get_spot_prices_by_product_ids.side_effect = RuntimeError("network error")
        feed = CoinbasePriceFeed(
            client=client, symbols=["BTC-USD"], history_seconds=120.0, logger=self.logger
        )
        result = feed.record_current_prices()
        self.assertEqual(result, {})

    # ------------------------------------------------------------------
    # History trimming
    # ------------------------------------------------------------------

    def test_old_entries_are_trimmed(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"], history_seconds=10.0)
        now = time.time()

        # Manually inject an old entry (20 s ago) and a recent one
        feed._history["BTC-USD"].append((now - 20.0, 68_000.0))  # should be trimmed
        feed._history["BTC-USD"].append((now - 5.0, 69_000.0))   # should stay

        # Recording a new price triggers trimming
        feed.client._prices["BTC-USD"] = 70_000.0
        feed.record_current_prices()

        history = feed._history["BTC-USD"]
        # The 20s-old entry should be gone
        self.assertTrue(all(ts >= now - 10.0 for ts, _ in history))
        # At least 2 entries remain (the 5s-old one + the new one)
        self.assertGreaterEqual(len(history), 2)

    def test_unknown_symbol_returns_zero(self) -> None:
        feed = self._make_feed()
        self.assertEqual(feed.get_latest_price("UNKNOWN-USD"), 0.0)

    # ------------------------------------------------------------------
    # is_ready()
    # ------------------------------------------------------------------

    def test_is_ready_false_when_no_history(self) -> None:
        feed = self._make_feed()
        self.assertFalse(feed.is_ready("BTC-USD", min_history_seconds=30.0))

    def test_is_ready_false_with_single_entry(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"])
        feed._history["BTC-USD"].append((time.time(), 70_000.0))
        self.assertFalse(feed.is_ready("BTC-USD", min_history_seconds=1.0))

    def test_is_ready_false_when_span_too_short(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"])
        now = time.time()
        feed._history["BTC-USD"].append((now - 10.0, 70_000.0))
        feed._history["BTC-USD"].append((now, 70_100.0))
        self.assertFalse(feed.is_ready("BTC-USD", min_history_seconds=30.0))

    def test_is_ready_true_when_span_sufficient(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"])
        now = time.time()
        feed._history["BTC-USD"].append((now - 35.0, 70_000.0))
        feed._history["BTC-USD"].append((now, 70_100.0))
        self.assertTrue(feed.is_ready("BTC-USD", min_history_seconds=30.0))

    # ------------------------------------------------------------------
    # get_momentum_pct()
    # ------------------------------------------------------------------

    def test_momentum_zero_with_insufficient_history(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"])
        self.assertEqual(feed.get_momentum_pct("BTC-USD", window_seconds=30.0), 0.0)

    def test_momentum_positive_on_price_increase(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"])
        now = time.time()
        # Price went from 80_000 to 80_400 in 30 s → +0.5%
        feed._history["BTC-USD"].append((now - 30.0, 80_000.0))
        feed._history["BTC-USD"].append((now, 80_400.0))
        momentum = feed.get_momentum_pct("BTC-USD", window_seconds=60.0)
        self.assertAlmostEqual(momentum, 0.5, places=4)

    def test_momentum_negative_on_price_decrease(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"])
        now = time.time()
        # Price went from 80_000 to 79_200 → -1.0%
        feed._history["BTC-USD"].append((now - 30.0, 80_000.0))
        feed._history["BTC-USD"].append((now, 79_200.0))
        momentum = feed.get_momentum_pct("BTC-USD", window_seconds=60.0)
        self.assertAlmostEqual(momentum, -1.0, places=4)

    def test_momentum_uses_oldest_entry_in_window(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"])
        now = time.time()
        # Three entries; window is 50s so the 60s-old entry is outside the window
        feed._history["BTC-USD"].append((now - 60.0, 78_000.0))  # outside 50s window
        feed._history["BTC-USD"].append((now - 40.0, 79_000.0))  # inside
        feed._history["BTC-USD"].append((now, 80_000.0))           # current
        momentum = feed.get_momentum_pct("BTC-USD", window_seconds=50.0)
        # Should use 79_000 → 80_000: +1.266...%
        expected = (80_000 - 79_000) / 79_000 * 100.0
        self.assertAlmostEqual(momentum, expected, places=3)

    def test_momentum_falls_back_to_oldest_when_window_short(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"])
        now = time.time()
        feed._history["BTC-USD"].append((now - 10.0, 70_000.0))
        feed._history["BTC-USD"].append((now, 70_700.0))
        # Window is 30s but we only have 10s of history — falls back to oldest
        momentum = feed.get_momentum_pct("BTC-USD", window_seconds=30.0)
        expected = (70_700 - 70_000) / 70_000 * 100.0
        self.assertAlmostEqual(momentum, expected, places=3)

    def test_history_span_seconds(self) -> None:
        feed = self._make_feed(symbols=["BTC-USD"])
        now = time.time()
        feed._history["BTC-USD"].append((now - 45.0, 70_000.0))
        feed._history["BTC-USD"].append((now, 70_000.0))
        self.assertAlmostEqual(feed.history_span_seconds("BTC-USD"), 45.0, delta=0.1)

    def test_history_span_seconds_returns_zero_for_empty(self) -> None:
        feed = self._make_feed()
        self.assertEqual(feed.history_span_seconds("BTC-USD"), 0.0)


if __name__ == "__main__":
    unittest.main()
