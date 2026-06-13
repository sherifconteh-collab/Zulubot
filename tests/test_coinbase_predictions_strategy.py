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

    def __init__(
        self,
        markets: list[dict[str, Any]] | None = None,
        *,
        can_trade: bool = False,
    ) -> None:
        self._markets = markets or []
        self.can_trade = can_trade
        self.orders: list[dict[str, Any]] = []

    def get_markets(self, **kwargs: Any) -> tuple[list[dict[str, Any]], None]:
        return self._markets, None

    def get_market(self, ticker: str) -> dict[str, Any]:
        for market in self._markets:
            if market.get("ticker") == ticker:
                return market
        raise KeyError(ticker)

    def place_order(self, **kwargs: Any) -> dict[str, Any]:
        self.orders.append(kwargs)
        return {"order": kwargs}


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
        self,
        markets: list[dict[str, Any]] | None = None,
        *,
        can_trade: bool = False,
        live_trading_enabled: bool = False,
    ) -> CoinbasePredictionStrategy:
        kalshi = FakeKalshiClient(markets, can_trade=can_trade)
        return CoinbasePredictionStrategy(
            config=self.config,
            kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
            live_trading_enabled=live_trading_enabled,
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
        # With a threshold above the market's volume, market should be excluded
        config_high = PredictionConfig(enabled=True, market_categories=["crypto"], min_volume_usd=5_000)
        kalshi_high = FakeKalshiClient([market])
        strategy_high = CoinbasePredictionStrategy(
            config=config_high,
            kalshi_client=kalshi_high,  # type: ignore[arg-type]
            logger=self.logger,
        )
        self.assertEqual(len(strategy_high._discover_markets()), 0)

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
        """_evaluate_signal() returns None when YES bid-ask spread exceeds threshold."""
        # YES spread = 0.21, above the default max_spread_usd of 0.20
        market = _make_market(yes_ask=0.50, yes_bid=0.29)
        strategy = self._make_strategy()
        result = strategy._evaluate_signal(market, mid=market["_mid"], fear_greed_value=None)
        self.assertIsNone(result)

    def test_evaluate_signal_rejects_wide_no_spread(self) -> None:
        """_evaluate_signal() returns None when NO bid-ask spread exceeds threshold."""
        # Use explicit config with a threshold that the spread will exceed
        config = PredictionConfig(enabled=True, market_categories=["crypto"], max_spread_usd=0.15)
        kalshi = FakeKalshiClient([])
        strategy = CoinbasePredictionStrategy(
            config=config, kalshi_client=kalshi, logger=self.logger,  # type: ignore[arg-type]
        )
        # Tight YES spread but wide NO spread
        market = _make_market(yes_ask=0.22, yes_bid=0.20)  # YES spread = 0.02 (tight)
        # Override NO side to have a wide spread
        market["_no_ask"] = 0.95
        market["_no_bid"] = 0.75  # NO spread = 0.20 > 0.15
        market["_mid"] = (market["_yes_bid"] + market["_yes_ask"]) / 2
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
        # Spread of 0.18 on each side
        market = _make_market(yes_ask=0.50, yes_bid=0.32, no_ask=0.68, no_bid=0.50)
        config_tight = PredictionConfig(
            enabled=True,
            market_categories=["crypto"],
            max_spread_usd=0.15,  # tighter than the 0.18 spread → reject
        )
        strategy_tight = CoinbasePredictionStrategy(
            config=config_tight,
            kalshi_client=FakeKalshiClient([]),  # type: ignore[arg-type]
            logger=self.logger,
        )
        self.assertIsNone(strategy_tight._evaluate_signal(market, mid=0.29, fear_greed_value=None))

        config_wider = PredictionConfig(
            enabled=True,
            market_categories=["crypto"],
            max_spread_usd=0.20,  # wider than the 0.18 spread → accept
        )
        strategy_wider = CoinbasePredictionStrategy(
            config=config_wider,
            kalshi_client=FakeKalshiClient([]),  # type: ignore[arg-type]
            logger=self.logger,
        )
        result = strategy_wider._evaluate_signal(market, mid=0.29, fear_greed_value=None)
        self.assertEqual(result, ("yes", 0.50))

    def test_crypto_signal_ignores_fear_greed_by_default(self) -> None:
        """By default, crypto-category markets trade on the price threshold alone.

        Previously a hidden extra gate required extreme Fear & Greed
        (<=30 YES / >=70 NO), which kept the Kalshi strategy idle in neutral
        sentiment regimes. The gate is now opt-in via
        PredictionConfig.require_crypto_fear_greed.
        """
        market = _make_market(yes_ask=0.30, yes_bid=0.28, volume=10_000)
        strategy = self._make_strategy()
        # Neutral Fear & Greed (50): default strategy should still fire YES.
        result = strategy._evaluate_signal(market, mid=0.29, fear_greed_value=50)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "yes")

    def test_crypto_signal_respects_opt_in_fear_greed_gate(self) -> None:
        """With require_crypto_fear_greed=True, neutral F&G should veto the signal."""
        market = _make_market(yes_ask=0.30, yes_bid=0.28, volume=10_000)
        config = PredictionConfig(
            enabled=True,
            market_categories=["crypto"],
            require_crypto_fear_greed=True,
        )
        strategy = CoinbasePredictionStrategy(
            config=config,
            kalshi_client=FakeKalshiClient([]),  # type: ignore[arg-type]
            logger=self.logger,
        )
        # Neutral F&G veto.
        self.assertIsNone(strategy._evaluate_signal(market, mid=0.29, fear_greed_value=50))
        # Extreme fear lets the YES signal through.
        self.assertEqual(
            strategy._evaluate_signal(market, mid=0.29, fear_greed_value=20),
            ("yes", 0.30),
        )

    def test_live_trading_uses_kalshi_cash_bucket(self) -> None:
        market = _make_market(yes_ask=0.20, yes_bid=0.18, volume=10_000.0)
        kalshi = FakeKalshiClient([market], can_trade=True)
        strategy = CoinbasePredictionStrategy(
            config=self.config,
            kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
            live_trading_enabled=True,
        )
        state = PortfolioState(
            cash=0.0,
            cash_reserves={"kalshi": 50.0},
            daily_baseline_equity=50.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=50.0,
        )

        result = strategy.run(
            state,
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            now=self.now,
        )

        self.assertEqual(len(result["buys"]), 1)
        symbol = result["buys"][0]
        self.assertEqual(state.cash, 0.0)
        self.assertLess(state.cash_reserves["kalshi"], 50.0)
        self.assertEqual(state.positions[symbol].meta.get("cash_bucket"), "kalshi")
        self.assertEqual(state.positions[symbol].meta.get("venue"), "kalshi")
        self.assertEqual(len(kalshi.orders), 1)

    def test_credentials_do_not_trigger_live_order_in_paper_mode(self) -> None:
        market = _make_market(yes_ask=0.20, yes_bid=0.18, volume=10_000.0)
        kalshi = FakeKalshiClient([market], can_trade=True)
        strategy = CoinbasePredictionStrategy(
            config=self.config,
            kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
            live_trading_enabled=False,
        )
        state = self._make_state()

        result = strategy.run(
            state,
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            now=self.now,
        )

        self.assertEqual(len(result["buys"]), 1)
        self.assertEqual(result["can_trade"], False)
        self.assertEqual(kalshi.orders, [])

    def test_manage_existing_positions_values_no_side_with_no_market_price(self) -> None:
        market = _make_market(
            ticker="KXBTC-NO",
            yes_ask=0.30,
            yes_bid=0.28,
            no_ask=0.72,
            no_bid=0.70,
            volume=10_000.0,
        )
        kalshi = FakeKalshiClient([market])
        strategy = CoinbasePredictionStrategy(
            config=self.config,
            kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
        )
        state = self._make_state()
        symbol = "KX:KXBTC-NO:no"
        self.paper_broker.open_position(
            state,
            symbol=symbol,
            asset_type="prediction",
            strategy="coinbase_predictions",
            price=0.72,
            notional=7.2,
            opened_at=self.now,
            meta={"ticker": "KXBTC-NO", "side": "no"},
        )

        result = {"sells": []}
        strategy._manage_existing_positions(state, self.paper_broker, result)

        self.assertIn(symbol, state.positions)
        self.assertAlmostEqual(state.positions[symbol].current_price, 0.71, places=2)


class PagingKalshiClient(FakeKalshiClient):
    """Kalshi stub that serves cursor-paginated markets for one series."""

    def __init__(self, pages_by_series: dict[str, list[tuple[list[dict[str, Any]], str | None]]]) -> None:
        super().__init__([])
        self.pages_by_series = pages_by_series
        self.calls: list[tuple[str | None, str | None]] = []

    def get_markets(self, *, series_ticker: str | None = None, cursor: str | None = None, **kwargs: Any):
        self.calls.append((series_ticker, cursor))
        pages = self.pages_by_series.get(series_ticker or "")
        if not pages:
            return [], None
        index = 0 if cursor is None else int(cursor)
        return pages[index]


class MarketCoverageTests(unittest.TestCase):
    """Pagination, category round-robin, and one-position-per-event rules."""

    def setUp(self) -> None:
        self.logger = logging.getLogger("test_predictions_coverage")
        self.now = datetime(2026, 3, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        self.paper_broker = PaperBroker(True, self.logger)
        self.risk_manager = RiskManager(RiskConfig())

    def _make_state(self) -> PortfolioState:
        return PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

    def test_discover_markets_follows_pagination_cursor(self) -> None:
        """All pages of a series are fetched, not just the first."""
        page1 = _make_market(ticker="KXBTC-PAGE1", volume=10_000.0)
        page2 = _make_market(ticker="KXBTC-PAGE2", volume=9_000.0)
        kalshi = PagingKalshiClient({"KXBTC": [([page1], "1"), ([page2], None)]})
        strategy = CoinbasePredictionStrategy(
            config=PredictionConfig(enabled=True, market_categories=["crypto"]),
            kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
        )

        from unittest.mock import patch
        with patch("time.sleep"):
            markets = strategy._discover_markets()

        tickers = {m["ticker"] for m in markets}
        self.assertIn("KXBTC-PAGE1", tickers)
        self.assertIn("KXBTC-PAGE2", tickers)
        self.assertIn(("KXBTC", "1"), kalshi.calls)

    def test_discover_markets_respects_max_pages_per_series(self) -> None:
        page1 = _make_market(ticker="KXBTC-PAGE1", volume=10_000.0)
        page2 = _make_market(ticker="KXBTC-PAGE2", volume=9_000.0)
        kalshi = PagingKalshiClient({"KXBTC": [([page1], "1"), ([page2], None)]})
        strategy = CoinbasePredictionStrategy(
            config=PredictionConfig(
                enabled=True, market_categories=["crypto"], max_pages_per_series=1,
            ),
            kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
        )

        from unittest.mock import patch
        with patch("time.sleep"):
            markets = strategy._discover_markets()

        tickers = {m["ticker"] for m in markets}
        self.assertIn("KXBTC-PAGE1", tickers)
        self.assertNotIn("KXBTC-PAGE2", tickers)
        self.assertNotIn(("KXBTC", "1"), kalshi.calls)

    def test_round_robin_interleaves_categories_by_volume(self) -> None:
        markets = [
            {"_category": "economics", "_volume": 900.0, "ticker": "E1"},
            {"_category": "economics", "_volume": 800.0, "ticker": "E2"},
            {"_category": "economics", "_volume": 700.0, "ticker": "E3"},
            {"_category": "crypto", "_volume": 100.0, "ticker": "C1"},
            {"_category": "crypto", "_volume": 50.0, "ticker": "C2"},
        ]

        ordered = CoinbasePredictionStrategy._round_robin_by_category(markets)

        tickers = [m["ticker"] for m in ordered]
        # Low-volume crypto still gets an early slot instead of being pushed
        # behind every economics market.
        self.assertEqual(tickers, ["C1", "E1", "C2", "E2", "E3"])

    def test_one_position_per_multi_outcome_event_takes_best_edge(self) -> None:
        """Two outcomes of the same event → only the better-edge one is bought."""
        better = _make_market(ticker="KXCPI-LOW", yes_ask=0.20, yes_bid=0.18)
        worse = _make_market(ticker="KXCPI-HIGH", yes_ask=0.32, yes_bid=0.30)
        better["event_ticker"] = "KXCPI-26JAN"
        worse["event_ticker"] = "KXCPI-26JAN"
        strategy = CoinbasePredictionStrategy(
            config=PredictionConfig(enabled=True, market_categories=["crypto"]),
            kalshi_client=FakeKalshiClient([better, worse]),  # type: ignore[arg-type]
            logger=self.logger,
        )
        state = self._make_state()

        result = strategy.run(
            state, broker=self.paper_broker, risk_manager=self.risk_manager, now=self.now,
        )

        self.assertEqual(result["buys"], ["KX:KXCPI-LOW:yes"])
        position = state.positions["KX:KXCPI-LOW:yes"]
        self.assertEqual(position.meta.get("event_ticker"), "KXCPI-26JAN")

    def test_event_already_held_blocks_sibling_outcomes(self) -> None:
        sibling = _make_market(ticker="KXCPI-HIGH", yes_ask=0.20, yes_bid=0.18)
        sibling["event_ticker"] = "KXCPI-26JAN"
        strategy = CoinbasePredictionStrategy(
            config=PredictionConfig(enabled=True, market_categories=["crypto"]),
            kalshi_client=FakeKalshiClient([sibling]),  # type: ignore[arg-type]
            logger=self.logger,
        )
        state = self._make_state()
        self.paper_broker.open_position(
            state,
            symbol="KX:KXCPI-LOW:yes",
            asset_type="prediction",
            strategy="coinbase_predictions",
            price=0.25,
            notional=25.0,
            opened_at=self.now,
            meta={"ticker": "KXCPI-LOW", "side": "yes", "event_ticker": "KXCPI-26JAN"},
        )

        result = strategy.run(
            state, broker=self.paper_broker, risk_manager=self.risk_manager, now=self.now,
        )

        self.assertEqual(result["buys"], [])
        self.assertNotIn("KX:KXCPI-HIGH:yes", state.positions)


if __name__ == "__main__":
    unittest.main()
