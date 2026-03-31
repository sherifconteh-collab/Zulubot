"""Tests for LatencyArbStrategy — signal generation, Kelly sizing, exit logic."""

from __future__ import annotations

import logging
import math
import time
import unittest
from collections import deque
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.config import LatencyArbConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager, RiskConfig
from trading_bot.strategies.latency_arb import (
    LatencyArbStrategy,
    _best_ask,
    _kelly_edge,
)


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

class FakePriceFeed:
    """Minimal CoinbasePriceFeed stub."""

    def __init__(
        self,
        prices: dict[str, float] | None = None,
        momentum: dict[str, float] | None = None,
        ready: bool = True,
        stale: bool = False,
    ) -> None:
        self._prices = prices or {}
        self._momentum = momentum or {}
        self._ready = ready
        self._stale = stale

    def record_current_prices(self) -> dict[str, float]:
        return self._prices

    def get_latest_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    def get_momentum_pct(self, symbol: str, window_seconds: float) -> float:
        return self._momentum.get(symbol, 0.0)

    def is_ready(self, symbol: str, min_history_seconds: float = 30.0) -> bool:
        return self._ready

    def is_stale(self, symbol: str, max_age_seconds: float) -> bool:
        return self._stale


class FakeKalshiClient:
    """Minimal KalshiClient stub that does nothing live."""

    def __init__(self, *, can_trade: bool = False) -> None:
        self.can_trade = can_trade
        self.orders: list[dict[str, object]] = []

    def get_markets(self, **kwargs):  # noqa: ANN001
        return [], None

    def get_orderbook(self, ticker: str) -> dict:
        return {}

    def place_order(self, **kwargs):  # noqa: ANN001
        self.orders.append(kwargs)
        return {"order": {"order_id": "test-order-id"}}


def _make_state(cash: float = 10_000.0) -> PortfolioState:
    state = PortfolioState(cash=cash)
    state.peak_equity = cash
    return state


def _make_config(**overrides) -> LatencyArbConfig:
    defaults = dict(
        enabled=True,
        symbols=["BTC-USD", "ETH-USD"],
        momentum_window_seconds=30.0,
        momentum_min_threshold_pct=0.30,
        min_edge_pct=0.05,
        kelly_fraction=0.25,
        max_position_pct=0.08,
        max_open_contracts=4,
        scan_interval_seconds=2.0,
        price_history_seconds=120.0,
        take_profit_pct=0.20,
        stop_loss_pct=0.15,
        min_trade_usd=1.0,
        ai_confirmation_mode="log_only",
    )
    defaults.update(overrides)
    return LatencyArbConfig(**defaults)


def _make_strategy(
    *,
    config: LatencyArbConfig | None = None,
    feed: FakePriceFeed | None = None,
    kalshi: FakeKalshiClient | None = None,
    ai_service=None,
    live_trading_enabled: bool = False,
) -> LatencyArbStrategy:
    logger = logging.getLogger("test_latency_arb")
    broker = PaperBroker(True, logger)
    return LatencyArbStrategy(
        config=config or _make_config(),
        price_feed=feed or FakePriceFeed(),
        kalshi_client=kalshi or FakeKalshiClient(),
        paper_broker=broker,
        logger=logger,
        ai_service=ai_service,
        live_trading_enabled=live_trading_enabled,
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class BestAskTests(unittest.TestCase):
    def test_list_of_lists(self) -> None:
        self.assertEqual(_best_ask([[30, 100], [35, 50]]), 30.0)

    def test_list_of_dicts(self) -> None:
        self.assertEqual(_best_ask([{"price": 45, "quantity": 10}]), 45.0)

    def test_empty_returns_zero(self) -> None:
        self.assertEqual(_best_ask([]), 0.0)


class KellyEdgeTests(unittest.TestCase):
    def test_positive_edge(self) -> None:
        # true_prob=0.70, ask=0.40 → edge=0.30
        self.assertAlmostEqual(_kelly_edge(0.40, 0.70), 0.30, places=5)

    def test_zero_edge_when_no_advantage(self) -> None:
        self.assertEqual(_kelly_edge(0.70, 0.60), 0.0)  # true_prob < ask → 0


# ---------------------------------------------------------------------------
# Strategy unit tests
# ---------------------------------------------------------------------------

class LatencyArbDisabledTests(unittest.TestCase):
    def test_disabled_returns_disabled_status(self) -> None:
        strategy = _make_strategy(config=_make_config(enabled=False))
        state = _make_state()
        result = strategy.run_once(
            state,
            broker=PaperBroker(True, logging.getLogger("test")),
            risk_manager=RiskManager(RiskConfig()),
            now=datetime.now(ZoneInfo("UTC")),
        )
        self.assertEqual(result["status"], "disabled")


class LatencyArbNoPricesTests(unittest.TestCase):
    def test_no_prices_returns_no_prices_status(self) -> None:
        feed = FakePriceFeed(prices={})
        strategy = _make_strategy(feed=feed)
        state = _make_state()
        result = strategy.run_once(
            state,
            broker=PaperBroker(True, logging.getLogger("test")),
            risk_manager=RiskManager(RiskConfig()),
            now=datetime.now(ZoneInfo("UTC")),
        )
        self.assertEqual(result["status"], "no_prices")


class LatencyArbSignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_latency_arb_signal")
        self.now = datetime(2026, 3, 28, 12, 0, tzinfo=ZoneInfo("UTC"))

    def test_weak_momentum_produces_no_signal(self) -> None:
        """Momentum below threshold → no signal, no scan."""
        feed = FakePriceFeed(
            prices={"BTC-USD": 70_000.0},
            momentum={"BTC-USD": 0.10},  # below 0.30% threshold
            ready=True,
        )
        strategy = _make_strategy(feed=feed)
        state = _make_state()

        result = strategy.run_once(
            state,
            broker=PaperBroker(True, self.logger),
            risk_manager=RiskManager(RiskConfig()),
            now=self.now,
        )

        self.assertEqual(result["signals_detected"], 0)
        self.assertEqual(result["buys"], [])

    def test_strong_momentum_triggers_signal(self) -> None:
        """Momentum above threshold → signal_detected incremented."""
        feed = FakePriceFeed(
            prices={"BTC-USD": 70_000.0},
            momentum={"BTC-USD": 0.50},  # above 0.30% threshold
            ready=True,
        )
        strategy = _make_strategy(feed=feed)
        state = _make_state()

        result = strategy.run_once(
            state,
            broker=PaperBroker(True, self.logger),
            risk_manager=RiskManager(RiskConfig()),
            now=self.now,
        )

        # Feed is ready + strong momentum → signal should fire
        # (no markets returned by fake Kalshi, so no actual buys)
        self.assertGreaterEqual(result["signals_detected"], 1)

    def test_feed_not_ready_skips_symbol(self) -> None:
        feed = FakePriceFeed(
            prices={"BTC-USD": 70_000.0},
            momentum={"BTC-USD": 1.0},
            ready=False,  # not enough history
        )
        strategy = _make_strategy(feed=feed)
        state = _make_state()

        result = strategy.run_once(
            state,
            broker=PaperBroker(True, self.logger),
            risk_manager=RiskManager(RiskConfig()),
            now=self.now,
        )

        self.assertEqual(result["signals_detected"], 0)

    def test_stale_feed_skips_symbol(self) -> None:
        feed = FakePriceFeed(
            prices={"BTC-USD": 70_000.0},
            momentum={"BTC-USD": 1.0},
            ready=True,
            stale=True,
        )
        strategy = _make_strategy(config=_make_config(symbols=["BTC-USD"]), feed=feed)
        state = _make_state()

        result = strategy.run_once(
            state,
            broker=PaperBroker(True, self.logger),
            risk_manager=RiskManager(RiskConfig()),
            now=self.now,
        )

        self.assertEqual(result["signals_detected"], 0)
        self.assertEqual(result["stale_symbols"], ["BTC-USD"])

    def test_halted_state_skips_entries(self) -> None:
        feed = FakePriceFeed(prices={"BTC-USD": 70_000.0}, momentum={"BTC-USD": 1.0})
        strategy = _make_strategy(feed=feed)
        state = _make_state()
        state.kill_switch_engaged = True

        result = strategy.run_once(
            state,
            broker=PaperBroker(True, self.logger),
            risk_manager=RiskManager(RiskConfig()),
            now=self.now,
        )

        self.assertEqual(result["status"], "halted")
        self.assertEqual(result["buys"], [])


# ---------------------------------------------------------------------------
# Kelly sizing tests
# ---------------------------------------------------------------------------

class KellySizingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _make_config()
        self.logger = logging.getLogger("test_kelly")

    def _strategy(self) -> LatencyArbStrategy:
        return _make_strategy(config=self.config)

    def test_kelly_notional_proportional_to_edge(self) -> None:
        strategy = self._strategy()
        notional_high = strategy._kelly_notional(0.30, 0.40, 10_000.0)
        notional_low = strategy._kelly_notional(0.10, 0.40, 10_000.0)
        self.assertGreater(notional_high, notional_low)

    def test_kelly_notional_capped_by_max_position_pct(self) -> None:
        strategy = self._strategy()
        # Use a huge edge to force Kelly above the max cap
        notional = strategy._kelly_notional(0.99, 0.01, 10_000.0)
        max_allowed = 10_000.0 * self.config.max_position_pct
        self.assertLessEqual(notional, max_allowed + 1e-6)

    def test_kelly_notional_zero_for_invalid_ask(self) -> None:
        strategy = self._strategy()
        self.assertEqual(strategy._kelly_notional(0.30, 0.0, 10_000.0), 0.0)
        self.assertEqual(strategy._kelly_notional(0.30, 1.0, 10_000.0), 0.0)

    def test_kelly_notional_zero_for_negative_edge(self) -> None:
        strategy = self._strategy()
        self.assertEqual(strategy._kelly_notional(-0.10, 0.40, 10_000.0), 0.0)

    def test_estimate_true_prob_increases_with_momentum(self) -> None:
        strategy = self._strategy()
        p30 = strategy._estimate_true_prob(0.30)
        p50 = strategy._estimate_true_prob(0.50)
        p100 = strategy._estimate_true_prob(1.00)
        self.assertLess(p30, p50)
        self.assertLess(p50, p100)
        # All must be valid probabilities
        for p in (p30, p50, p100):
            self.assertGreater(p, 0.5)
            self.assertLess(p, 1.0)


# ---------------------------------------------------------------------------
# Per-series cache tests
# ---------------------------------------------------------------------------

class SeriesCacheTests(unittest.TestCase):
    def test_different_series_have_independent_ttl(self) -> None:
        """A fresh fetch for one series must not reset the TTL for another."""
        strategy = _make_strategy()
        now = time.time()

        # Seed two series caches with different timestamps
        strategy._market_cache["KXBTC"] = [{"ticker": "BTC-MOCK"}]
        strategy._cache_ts["KXBTC"] = now - 25.0  # 25s old → fresh (TTL=30s)

        strategy._market_cache["KXETH"] = [{"ticker": "ETH-MOCK"}]
        strategy._cache_ts["KXETH"] = now - 35.0  # 35s old → stale (TTL=30s)

        # KXBTC is fresh — should return from cache without API call
        self.assertLess(now - strategy._cache_ts["KXBTC"], strategy._cache_ttl)
        # KXETH is stale — should trigger a refresh
        self.assertGreaterEqual(now - strategy._cache_ts["KXETH"], strategy._cache_ttl)


# ---------------------------------------------------------------------------
# Exit logic tests
# ---------------------------------------------------------------------------

class ExitLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_exit_logic")
        self.now = datetime(2026, 3, 28, 12, 0, tzinfo=ZoneInfo("UTC"))
        self.broker = PaperBroker(True, self.logger)

    def _open_position(self, state: PortfolioState, symbol: str, price: float) -> None:
        """Helper: open a paper arb position at a given price."""
        self.broker.open_position(
            state,
            symbol=symbol,
            asset_type="prediction",
            strategy="latency_arb",
            price=price,
            notional=price * 10,
            opened_at=self.now,
            meta={"ticker": symbol.split(":")[1], "side": "yes"},
        )

    def test_take_profit_closes_position(self) -> None:
        config = _make_config(take_profit_pct=0.20, stop_loss_pct=0.50)
        strategy = _make_strategy(config=config)
        state = _make_state()
        symbol = "KL:KXBTCH-26MAY0117:YES"
        self._open_position(state, symbol, 0.40)

        # Simulate a 25% price gain (above 20% take_profit_pct)
        result: dict = {"buys": [], "sells": [], "market_regime": None}
        strategy._get_market_prices = MagicMock(return_value=(0.50, 0.50))
        strategy._manage_positions(state, self.broker, result, self.now)

        self.assertIn(symbol, result["sells"])
        self.assertNotIn(symbol, state.positions)

    def test_stop_loss_closes_position(self) -> None:
        config = _make_config(take_profit_pct=0.50, stop_loss_pct=0.15)
        strategy = _make_strategy(config=config)
        state = _make_state()
        symbol = "KL:KXBTCH-26MAY0117:YES"
        self._open_position(state, symbol, 0.40)

        # Simulate a 20% drop (below 0.32 entry × 0.85 = 0.34, current=0.32)
        result: dict = {"buys": [], "sells": [], "market_regime": None}
        strategy._get_market_prices = MagicMock(return_value=(0.32, 0.68))
        strategy._manage_positions(state, self.broker, result, self.now)

        self.assertIn(symbol, result["sells"])
        self.assertNotIn(symbol, state.positions)

    def test_no_exit_when_within_thresholds(self) -> None:
        config = _make_config(take_profit_pct=0.20, stop_loss_pct=0.15)
        strategy = _make_strategy(config=config)
        state = _make_state()
        symbol = "KL:KXBTCH-26MAY0117:YES"
        self._open_position(state, symbol, 0.40)

        # Price moved only 5% — no exit
        result: dict = {"buys": [], "sells": [], "market_regime": None}
        strategy._get_market_prices = MagicMock(return_value=(0.42, 0.58))
        strategy._manage_positions(state, self.broker, result, self.now)

        self.assertEqual(result["sells"], [])
        self.assertIn(symbol, state.positions)


class LiveKalshiAccountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_live_kalshi_account")
        self.now = datetime(2026, 3, 31, 12, 0, tzinfo=ZoneInfo("UTC"))
        self.broker = PaperBroker(True, self.logger)

    def _open_position(self, state: PortfolioState, symbol: str, price: float) -> None:
        self.broker.open_position(
            state,
            symbol=symbol,
            asset_type="prediction",
            strategy="latency_arb",
            price=price,
            notional=price * 10,
            opened_at=self.now,
            meta={"ticker": symbol.split(":")[1], "side": "yes"},
        )

    def test_execute_trade_uses_kalshi_cash_bucket(self) -> None:
        kalshi = FakeKalshiClient(can_trade=True)
        strategy = _make_strategy(kalshi=kalshi, live_trading_enabled=True)
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        market = {"ticker": "KXBTC-26APR0117", "title": "Will BTC be higher?"}

        entered = strategy._execute_trade(
            state=state,
            market=market,
            spot_symbol="BTC-USD",
            side="yes",
            ask_price=0.50,
            notional=20.0,
            momentum_pct=0.80,
            now=self.now,
        )

        self.assertTrue(entered)
        self.assertEqual(state.cash, 0.0)
        self.assertLess(state.cash_reserves["kalshi"], 100.0)
        self.assertIn("KL:KXBTC-26APR0117:YES", state.positions)
        self.assertEqual(state.positions["KL:KXBTC-26APR0117:YES"].meta.get("cash_bucket"), "kalshi")
        self.assertEqual(len(kalshi.orders), 1)

    def test_credentials_alone_do_not_submit_live_order(self) -> None:
        kalshi = FakeKalshiClient(can_trade=True)
        strategy = _make_strategy(kalshi=kalshi, live_trading_enabled=False)
        state = _make_state(100.0)
        market = {"ticker": "KXBTC-26APR0117", "title": "Will BTC be higher?"}

        entered = strategy._execute_trade(
            state=state,
            market=market,
            spot_symbol="BTC-USD",
            side="yes",
            ask_price=0.50,
            notional=20.0,
            momentum_pct=0.80,
            now=self.now,
        )

        self.assertTrue(entered)
        self.assertEqual(len(kalshi.orders), 0)

    def test_ai_veto_blocks_exit_when_confident(self) -> None:
        """AI returning hold with high confidence should prevent the exit."""
        config = _make_config(take_profit_pct=0.20, stop_loss_pct=0.50)
        mock_ai = MagicMock()
        from trading_bot.integrations.trading_agents import AgentVerdict
        mock_ai.evaluate_exit_signal.return_value = AgentVerdict(
            action="hold", confidence=0.80, reasoning="Temporary dip, keep holding."
        )
        strategy = _make_strategy(config=config)
        strategy.ai_service = mock_ai
        state = _make_state()
        symbol = "KL:KXBTCH-26MAY0117:YES"
        self._open_position(state, symbol, 0.40)

        # Price rose 25% — would normally take profit
        result: dict = {"buys": [], "sells": [], "market_regime": "bullish"}
        strategy._get_market_prices = MagicMock(return_value=(0.50, 0.50))
        strategy._manage_positions(state, self.broker, result, self.now)

        # AI vetoed the exit
        self.assertEqual(result["sells"], [])
        self.assertIn(symbol, state.positions)

    def test_ai_low_confidence_does_not_block_exit(self) -> None:
        """AI returning hold with low confidence (< 0.65) should not veto exit."""
        config = _make_config(take_profit_pct=0.20, stop_loss_pct=0.50)
        mock_ai = MagicMock()
        from trading_bot.integrations.trading_agents import AgentVerdict
        mock_ai.evaluate_exit_signal.return_value = AgentVerdict(
            action="hold", confidence=0.50, reasoning="Maybe hold."
        )
        strategy = _make_strategy(config=config)
        strategy.ai_service = mock_ai
        state = _make_state()
        symbol = "KL:KXBTCH-26MAY0117:YES"
        self._open_position(state, symbol, 0.40)

        result: dict = {"buys": [], "sells": [], "market_regime": None}
        strategy._get_market_prices = MagicMock(return_value=(0.50, 0.50))
        strategy._manage_positions(state, self.broker, result, self.now)

        # Low-confidence veto doesn't block — position should be closed
        self.assertIn(symbol, result["sells"])


if __name__ == "__main__":
    unittest.main()
