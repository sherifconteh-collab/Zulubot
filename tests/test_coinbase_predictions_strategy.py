from __future__ import annotations

import logging
import unittest
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.config import PredictionConfig, RiskConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.strategies.coinbase_predictions import CoinbasePredictionStrategy


def _make_market(
    ticker: str = "KXBTC-25DEC3100",
    yes_ask: float = 0.25,
    yes_bid: float = 0.20,
    volume: float = 10_000.0,
    no_ask: float | None = None,
    no_bid: float | None = None,
) -> dict[str, Any]:
    """Build a minimal Kalshi market dict as returned by the strategy's _discover_markets."""
    resolved_no_ask = (1.0 - yes_bid) if no_ask is None else no_ask
    resolved_no_bid = (1.0 - yes_ask) if no_bid is None else no_bid
    return {
        "ticker": ticker,
        "title": f"Test market {ticker}",
        "yes_ask_dollars": str(yes_ask),
        "yes_bid_dollars": str(yes_bid),
        "no_ask_dollars": str(resolved_no_ask),
        "no_bid_dollars": str(resolved_no_bid),
        "last_price_dollars": str(yes_ask),
        "volume_fp": str(volume),
        "_series": "KXBTC",
        "_category": "crypto",
        "_yes_ask": yes_ask,
        "_yes_bid": yes_bid,
        "_no_ask": resolved_no_ask,
        "_no_bid": resolved_no_bid,
        "_last": yes_ask,
        "_volume": volume,
        "_mid": (yes_bid + yes_ask) / 2,
    }


class FakeKalshiClient:
    """Minimal Kalshi client stub for unit tests."""

    def __init__(self, markets: list[dict[str, Any]] | None = None) -> None:
        self._markets = markets or []
        self.can_trade = False

    def get_markets(self, **kwargs: Any) -> tuple[list[dict[str, Any]], None]:
        return self._markets, None


class CoinbasePredictionStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_predictions")
        self.now = datetime(2026, 3, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        self.config = PredictionConfig(
            enabled=True,
            market_categories=["crypto"],
        )
        self.paper_broker = PaperBroker(True, self.logger)
        self.risk_manager = RiskManager(RiskConfig())

    def _make_state(self) -> PortfolioState:
        return PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

    def _make_strategy(
        self, markets: list[dict[str, Any]] | None = None
    ) -> CoinbasePredictionStrategy:
        kalshi = FakeKalshiClient(markets)
        return CoinbasePredictionStrategy(
            config=self.config,
            kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
        )

    # ------------------------------------------------------------------
    # last_predictions_scan_date tests
    # ------------------------------------------------------------------

    def test_scan_date_updated_when_markets_found(self) -> None:
        """last_predictions_scan_date is recorded when markets are discovered."""
        market = _make_market(yes_ask=0.20, yes_bid=0.18, volume=5_000.0)
        strategy = self._make_strategy(markets=[market])
        state = self._make_state()

        self.assertIsNone(state.last_predictions_scan_date)

        strategy.run(
            state,
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            now=self.now,
        )

        self.assertEqual(state.last_predictions_scan_date, self.now.date().isoformat())

    def test_scan_date_updated_when_no_markets_found(self) -> None:
        """last_predictions_scan_date is recorded even when Kalshi returns no markets."""
        strategy = self._make_strategy(markets=[])
        state = self._make_state()

        self.assertIsNone(state.last_predictions_scan_date)

        result = strategy.run(
            state,
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            now=self.now,
        )

        self.assertEqual(result["markets_discovered"], 0)
        self.assertEqual(state.last_predictions_scan_date, self.now.date().isoformat())

    def test_disabled_strategy_does_not_set_scan_date(self) -> None:
        """A disabled strategy should return immediately without touching state."""
        config = PredictionConfig(enabled=False)
        kalshi = FakeKalshiClient([])
        strategy = CoinbasePredictionStrategy(
            config=config,
            kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
        )
        state = self._make_state()

        result = strategy.run(
            state,
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            now=self.now,
        )

        self.assertEqual(result["status"], "disabled")
        self.assertIsNone(state.last_predictions_scan_date)

    # ------------------------------------------------------------------
    # Volume filter tests
    # ------------------------------------------------------------------

    def test_discover_markets_drops_low_volume(self) -> None:
        """_discover_markets() should exclude markets whose volume is below min_volume_usd."""
        low_volume_market = {
            "ticker": "KXBTC-LOW",
            "title": "Low volume market",
            "yes_ask_dollars": "0.25",
            "yes_bid_dollars": "0.20",
            "no_ask_dollars": "0.80",
            "no_bid_dollars": "0.75",
            "last_price_dollars": "0.22",
            "volume_fp": "100.0",   # below default min_volume_usd of 5000
        }
        high_volume_market = {
            "ticker": "KXBTC-HIGH",
            "title": "High volume market",
            "yes_ask_dollars": "0.25",
            "yes_bid_dollars": "0.20",
            "no_ask_dollars": "0.80",
            "no_bid_dollars": "0.75",
            "last_price_dollars": "0.22",
            "volume_fp": "10000.0",  # above default min_volume_usd
        }
        strategy = self._make_strategy(markets=[low_volume_market, high_volume_market])
        discovered = strategy._discover_markets()

        tickers = [m["ticker"] for m in discovered]
        self.assertNotIn("KXBTC-LOW", tickers)
        self.assertIn("KXBTC-HIGH", tickers)

    def test_discover_markets_configurable_volume_threshold(self) -> None:
        """min_volume_usd from config is respected in _discover_markets()."""
        market = {
            "ticker": "KXBTC-MED",
            "title": "Medium volume market",
            "yes_ask_dollars": "0.25",
            "yes_bid_dollars": "0.20",
            "no_ask_dollars": "0.80",
            "no_bid_dollars": "0.75",
            "last_price_dollars": "0.22",
            "volume_fp": "3000.0",
        }
        # With default threshold (5000), market should be excluded
        strategy_default = self._make_strategy(markets=[market])
        self.assertEqual(len(strategy_default._discover_markets()), 0)

        # With lower threshold (1000), market should be included
        config_low = PredictionConfig(enabled=True, market_categories=["crypto"], min_volume_usd=1_000)
        kalshi = FakeKalshiClient([market])
        strategy_low = CoinbasePredictionStrategy(
            config=config_low,
            kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
        )
        # FakeKalshiClient returns the market for every series call, so we get >0 results
        self.assertGreater(len(strategy_low._discover_markets()), 0)

    # ------------------------------------------------------------------
    # Spread rejection tests
    # ------------------------------------------------------------------

    def test_evaluate_signal_rejects_wide_yes_spread(self) -> None:
        """_evaluate_signal() returns None when YES bid-ask spread > 15 cents."""
        market = _make_market(yes_ask=0.50, yes_bid=0.30)  # spread = 0.20 > 0.15
        strategy = self._make_strategy()
        result = strategy._evaluate_signal(market, mid=market["_mid"], fear_greed_value=None)
        self.assertIsNone(result)

    def test_evaluate_signal_rejects_wide_no_spread(self) -> None:
        """_evaluate_signal() returns None when NO bid-ask spread > 15 cents."""
        # Tight YES spread but wide NO spread
        market = _make_market(yes_ask=0.22, yes_bid=0.20)  # YES spread = 0.02 (tight)
        # Override NO side to have a wide spread
        market["_no_ask"] = 0.95
        market["_no_bid"] = 0.75  # NO spread = 0.20 > 0.15
        market["_mid"] = (market["_yes_bid"] + market["_yes_ask"]) / 2
        strategy = self._make_strategy()
        # Wide NO spread rejects the entire market regardless of signal direction
        result = strategy._evaluate_signal(market, mid=0.80, fear_greed_value=None)
        self.assertIsNone(result)

    def test_evaluate_signal_passes_tight_spreads(self) -> None:
        """_evaluate_signal() proceeds normally when both spreads are within threshold."""
        market = _make_market(yes_ask=0.22, yes_bid=0.20, volume=10_000)
        # yes_spread = 0.02, no_spread = 0.02 — both tight
        strategy = self._make_strategy()
        # mid below buy_yes_threshold → should return a YES signal
        result = strategy._evaluate_signal(market, mid=0.21, fear_greed_value=None)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "yes")

    def test_evaluate_signal_rejects_missing_yes_bid(self) -> None:
        """Markets missing one side of the YES book should fail liquidity checks."""
        market = _make_market(yes_ask=0.22, yes_bid=0.0, volume=10_000)
        strategy = self._make_strategy()
        result = strategy._evaluate_signal(market, mid=0.21, fear_greed_value=None)
        self.assertIsNone(result)

    def test_evaluate_signal_rejects_missing_no_ask(self) -> None:
        """Markets missing one side of the NO book should fail liquidity checks."""
        market = _make_market(yes_ask=0.22, yes_bid=0.20, volume=10_000, no_ask=0.0, no_bid=0.78)
        strategy = self._make_strategy()
        result = strategy._evaluate_signal(market, mid=0.80, fear_greed_value=None)
        self.assertIsNone(result)

    def test_evaluate_signal_respects_configured_spread_threshold(self) -> None:
        """max_spread_usd from config should control the liquidity guard."""
        market = _make_market(yes_ask=0.50, yes_bid=0.32, no_ask=0.68, no_bid=0.50)
        default_strategy = self._make_strategy()
        self.assertIsNone(default_strategy._evaluate_signal(market, mid=0.41, fear_greed_value=None))

        config_wider = PredictionConfig(
            enabled=True,
            market_categories=["crypto"],
            max_spread_usd=0.20,
        )
        strategy_wider = CoinbasePredictionStrategy(
            config=config_wider,
            kalshi_client=FakeKalshiClient([]),  # type: ignore[arg-type]
            logger=self.logger,
        )
        result = strategy_wider._evaluate_signal(market, mid=0.29, fear_greed_value=None)
        self.assertEqual(result, ("yes", 0.50))


if __name__ == "__main__":
    unittest.main()
