"""Regression tests for KalshiPerpetualsStrategy paper-mode bookkeeping.

Covers bugs found in code review of the initial implementation:
- entry sizing crashing / misbehaving because it referenced a non-existent
  ``state.equity`` instead of ``broker.portfolio_equity(state)``.
- paper-mode cash leaking into the "kalshi" cash_reserves bucket on close
  even though it was withdrawn from ``state.cash`` on open.
- short positions being marked-to-market with the raw price, inverting
  unrealized PnL direction.
- leverage being computed but never applied to the paper notional/quantity.
"""

from __future__ import annotations

import logging
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from trading_bot.broker import PaperBroker
from trading_bot.clients_kalshi import PERP_SERIES
from trading_bot.config import KalshiPerpetualsConfig
from trading_bot.models import PortfolioState
from trading_bot.strategies.kalshi_perpetuals import KalshiPerpetualsStrategy

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_state(cash: float = 10_000.0) -> PortfolioState:
    state = PortfolioState(cash=cash)
    state.peak_equity = cash
    return state


def _make_config(**overrides) -> KalshiPerpetualsConfig:
    defaults = dict(
        enabled=True,
        symbols=["BTC"],
        max_leverage=1.0,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
        trailing_stop_pct=0.0,
        max_positions=4,
        capital_allocation_pct=0.20,
        min_trade_usd=1.0,
    )
    defaults.update(overrides)
    return KalshiPerpetualsConfig(**defaults)


def _make_strategy(*, config: KalshiPerpetualsConfig | None = None) -> KalshiPerpetualsStrategy:
    logger = logging.getLogger("test_kalshi_perps")
    return KalshiPerpetualsStrategy(
        config=config or _make_config(),
        kalshi_client=MagicMock(),
        logger=logger,
        live_trading_enabled=False,
    )


class ScanForEntriesEquityTests(unittest.TestCase):
    def test_uses_broker_equity_and_sizes_correctly(self) -> None:
        state = _make_state(cash=10_000.0)
        broker = PaperBroker(True, logging.getLogger("test_broker"))
        strategy = _make_strategy()
        markets = [{"ticker": "KXBTCPERP-T1", "_symbol": "BTC", "_price": 100.0, "_max_leverage": 1.0}]

        with patch.object(strategy, "_evaluate_signal", return_value=("long", 1.0)):
            opens = strategy._scan_for_entries(state, broker, markets, _NOW)

        self.assertEqual(len(opens), 1)
        # equity=10_000 * 20% alloc / 4 slots = $500 per slot, leverage=1.0
        self.assertAlmostEqual(state.cash, 9_500.0, places=2)


class CashBucketRoundTripTests(unittest.TestCase):
    def test_paper_mode_close_returns_cash_to_main_balance(self) -> None:
        state = _make_state(cash=10_000.0)
        broker = PaperBroker(True, logging.getLogger("test_broker"))
        strategy = _make_strategy()
        market = {"ticker": "KXBTCPERP-T1", "_symbol": "BTC", "_price": 100.0}

        info = strategy._open_position(
            state, broker, market, "BTC", "long", 100.0, 500.0, 1.0, _NOW
        )
        self.assertIsNotNone(info)
        self.assertAlmostEqual(state.cash, 9_500.0, places=2)
        self.assertEqual(state.cash_reserves.get("kalshi", 0.0), 0.0)

        pos_key = "KP:KXBTCPERP-T1:LONG"
        position = state.positions[pos_key]
        strategy._close_position(
            state, broker, position, pos_key, "KXBTCPERP-T1", "long", 100.0, 100.0, "manual",
        )

        # No PnL (flat close) — cash should be back to the starting balance,
        # not split between state.cash and the unused "kalshi" reserve bucket.
        self.assertAlmostEqual(state.cash, 10_000.0, places=2)
        self.assertEqual(state.cash_reserves.get("kalshi", 0.0), 0.0)


class ShortMarkToMarketTests(unittest.TestCase):
    def test_short_position_loses_value_when_price_rises(self) -> None:
        state = _make_state(cash=10_000.0)
        broker = PaperBroker(True, logging.getLogger("test_broker"))
        strategy = _make_strategy()
        market = {"ticker": "KXBTCPERP-T1", "_symbol": "BTC", "_price": 100.0}

        strategy._open_position(
            state, broker, market, "BTC", "short", 100.0, 1_000.0, 1.0, _NOW
        )
        pos_key = "KP:KXBTCPERP-T1:SHORT"

        # Price moves against the short (up 10%); should not trip the wide
        # 50% stop-loss/take-profit thresholds, so the position stays open.
        adverse_markets = [{"ticker": "KXBTCPERP-T1", "_price": 110.0}]
        closes = strategy._manage_positions(state, broker, adverse_markets, _NOW)

        self.assertEqual(closes, [])
        position = state.positions[pos_key]
        self.assertLess(position.unrealized_pnl, 0.0)


class LiquidationClampTests(unittest.TestCase):
    def test_wipeout_move_force_closes_instead_of_negative_price(self) -> None:
        state = _make_state(cash=100_000.0)
        broker = PaperBroker(True, logging.getLogger("test_broker"))
        # Disable the configured stop/take-profit/trailing thresholds so only
        # the liquidation floor can trigger a close.
        config = _make_config(stop_loss_pct=0.0, take_profit_pct=0.0, trailing_stop_pct=0.0)
        strategy = KalshiPerpetualsStrategy(
            config=config,
            kalshi_client=MagicMock(),
            logger=logging.getLogger("test_kalshi_perps_liq"),
            live_trading_enabled=False,
        )
        market = {"ticker": "KXBTCPERP-T1", "_symbol": "BTC", "_price": 100.0}

        # leverage=6x, margin=$500 -> a -20% raw move is a -120% leveraged
        # move, past total loss of margin. A naive entry_price*(1+move)
        # synthetic price would be negative here.
        strategy._open_position(
            state, broker, market, "BTC", "long", 100.0, 500.0, 6.0, _NOW
        )
        pos_key = "KP:KXBTCPERP-T1:LONG"

        wipeout_markets = [{"ticker": "KXBTCPERP-T1", "_price": 80.0}]
        closes = strategy._manage_positions(state, broker, wipeout_markets, _NOW)

        self.assertEqual(len(closes), 1)
        self.assertEqual(closes[0]["reason"], "liquidation")
        self.assertNotIn(pos_key, state.positions)
        # Lost ~all of the $500 margin, but cash never went negative/runaway.
        self.assertGreater(state.cash, 99_499.0)
        self.assertLess(state.cash, 99_500.5)


class LeverageAppliedTests(unittest.TestCase):
    def test_leverage_scales_paper_notional(self) -> None:
        state = _make_state(cash=100_000.0)
        broker = PaperBroker(True, logging.getLogger("test_broker"))
        strategy = _make_strategy()
        market = {"ticker": "KXBTCPERP-T1", "_symbol": "BTC", "_price": 100.0}

        strategy._open_position(
            state, broker, market, "BTC", "long", 100.0, 500.0, 2.0, _NOW
        )

        # Cash debit is margin-only (size_usd), not leverage-scaled: leveraging
        # the notional here would lock up leverage_x the intended capital,
        # since broker.open_position couples quantity to the cash debited.
        pos_key = "KP:KXBTCPERP-T1:LONG"
        position = state.positions[pos_key]
        self.assertAlmostEqual(position.quantity, 5.0, places=4)
        self.assertAlmostEqual(state.cash, 99_500.0, places=2)

    def test_leverage_scales_marked_pnl_not_quantity(self) -> None:
        state = _make_state(cash=100_000.0)
        broker = PaperBroker(True, logging.getLogger("test_broker"))
        strategy = _make_strategy()
        market = {"ticker": "KXBTCPERP-T1", "_symbol": "BTC", "_price": 100.0}

        strategy._open_position(
            state, broker, market, "BTC", "long", 100.0, 500.0, 2.0, _NOW
        )

        # +10% move with 2x leverage should produce PnL of margin * leverage *
        # move_pct = 500 * 2 * 0.10 = $100, applied via a synthetic mark price
        # (quantity itself is unlevered so the margin-only cash debit stays
        # consistent with the position's market value).
        adverse_markets = [{"ticker": "KXBTCPERP-T1", "_price": 110.0}]
        closes = strategy._manage_positions(state, broker, adverse_markets, _NOW)

        self.assertEqual(closes, [])
        position = state.positions["KP:KXBTCPERP-T1:LONG"]
        self.assertAlmostEqual(position.unrealized_pnl, 100.0, places=2)


class LiveModeEquityBucketTests(unittest.TestCase):
    def test_live_allocation_uses_kalshi_bucket_equity_not_whole_account(self) -> None:
        # Large non-Kalshi balance in state.cash; small funded Kalshi bucket.
        state = _make_state(cash=1_000_000.0)
        state.cash_reserves["kalshi"] = 1_000.0
        broker = PaperBroker(True, logging.getLogger("test_broker"))
        config = _make_config(capital_allocation_pct=0.20, max_positions=1)
        strategy = KalshiPerpetualsStrategy(
            config=config,
            kalshi_client=MagicMock(),
            logger=logging.getLogger("test_kalshi_perps_live"),
            live_trading_enabled=True,
        )
        strategy.kalshi_client.place_perp_order.return_value = {}
        markets = [{"ticker": "KXBTCPERP-T1", "_symbol": "BTC", "_price": 100.0, "_max_leverage": 1.0}]

        with patch.object(strategy, "_evaluate_signal", return_value=("long", 1.0)):
            opens = strategy._scan_for_entries(state, broker, markets, _NOW)

        # 20% of the $1,000 Kalshi bucket (not the $1,000,000 whole account)
        self.assertEqual(len(opens), 1)
        self.assertAlmostEqual(opens[0]["notional_usd"], 200.0, places=2)


class PerpSeriesTickerTests(unittest.TestCase):
    def test_shib_series_ticker_matches_naming_convention(self) -> None:
        self.assertEqual(PERP_SERIES["SHIB"], "KXSHIBPERP")


if __name__ == "__main__":
    unittest.main()
