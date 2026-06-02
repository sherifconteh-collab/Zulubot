"""Unit tests for the options trading strategy."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from trading_bot.broker import PaperBroker
from trading_bot.config import OptionsStrategyConfig
from trading_bot.models import PortfolioState, Position
from trading_bot.risk import RiskManager, RiskConfig
from trading_bot.strategies.options import OptionsStrategy, _safe_float


def _make_strategy(
    enabled: bool = True,
    buy_call_fear_threshold: int = 30,
    buy_put_fear_threshold: int = 70,
    min_delta: float = 0.70,
    take_profit_pct: float = 0.50,
    stop_loss_pct: float = 0.50,
    close_at_dte: int = 7,
    max_positions: int = 5,
    symbols: list[str] | None = None,
    live_trading: bool = False,
) -> tuple[OptionsStrategy, MagicMock, MagicMock]:
    config = OptionsStrategyConfig(
        enabled=enabled,
        symbols=symbols or ["AAPL", "SPY"],
        min_dte=21,
        max_dte=60,
        min_delta=min_delta,
        max_positions=max_positions,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        close_at_dte=close_at_dte,
        buy_call_fear_threshold=buy_call_fear_threshold,
        buy_put_fear_threshold=buy_put_fear_threshold,
        capital_allocation_pct=0.10,
        check_interval_minutes=60,
        min_volume=1,
        min_open_interest=1,
        max_iv=1.50,
    )
    robinhood = MagicMock()
    robinhood.can_trade = False
    alternative = MagicMock()
    import logging
    strategy = OptionsStrategy(
        config=config,
        robinhood_client=robinhood,
        alternative_client=alternative,
        logger=logging.getLogger("test"),
        live_trading_enabled=live_trading,
    )
    return strategy, robinhood, alternative


def _make_state(cash: float = 100_000.0) -> PortfolioState:
    return PortfolioState(cash=cash)


def _make_broker() -> PaperBroker:
    import logging
    return PaperBroker(paper_mode=True, logger=logging.getLogger("test"))


def _make_risk() -> RiskManager:
    return RiskManager(RiskConfig())


def _now() -> datetime:
    return datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


# ── _safe_float helper ─────────────────────────────────────────────────────────

class TestSafeFloat(unittest.TestCase):
    def test_converts_string(self) -> None:
        self.assertAlmostEqual(_safe_float("1.23"), 1.23)

    def test_converts_int(self) -> None:
        self.assertAlmostEqual(_safe_float(5), 5.0)

    def test_returns_default_on_none(self) -> None:
        self.assertEqual(_safe_float(None), 0.0)

    def test_returns_default_on_nan(self) -> None:
        self.assertEqual(_safe_float(float("nan")), 0.0)

    def test_returns_default_on_bad_string(self) -> None:
        self.assertEqual(_safe_float("abc"), 0.0)


# ── OptionsStrategy.run() ──────────────────────────────────────────────────────

class TestOptionsStrategyRun(unittest.TestCase):
    def setUp(self) -> None:
        self.broker = _make_broker()
        self.risk = _make_risk()
        self.now = _now()

    def test_disabled_returns_disabled(self) -> None:
        strategy, _, _ = _make_strategy(enabled=False)
        state = _make_state()
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["opens"], [])

    def test_kill_switch_halts(self) -> None:
        strategy, _, _ = _make_strategy()
        state = _make_state()
        state.kill_switch_engaged = True
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["status"], "halted")
        self.assertEqual(result["reason"], "kill_switch")

    def test_daily_halt_halts(self) -> None:
        strategy, _, _ = _make_strategy()
        state = _make_state()
        state.halt_new_trades_for_day = True
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["status"], "halted")
        self.assertEqual(result["reason"], "daily_halt")

    def test_no_signal_between_thresholds(self) -> None:
        strategy, _, alternative = _make_strategy()
        fg = MagicMock()
        fg.value = 50  # between 30 and 70
        alternative.get_fear_and_greed.return_value = fg
        state = _make_state()
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["reason"], "no_signal")
        self.assertEqual(result["opens"], [])

    def test_fear_and_greed_error_returns_error(self) -> None:
        strategy, _, alternative = _make_strategy()
        alternative.get_fear_and_greed.side_effect = RuntimeError("network error")
        state = _make_state()
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["status"], "error")
        self.assertIn("Fear & Greed unavailable", result["error"])

    @patch("trading_bot.strategies.options._YF_AVAILABLE", False)
    def test_extreme_fear_buys_calls_no_contract_on_missing_yf(self) -> None:
        strategy, _, alternative = _make_strategy()
        fg = MagicMock()
        fg.value = 20  # extreme fear
        alternative.get_fear_and_greed.return_value = fg
        state = _make_state()
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["option_type"], "call")
        # No contracts found because yfinance is disabled
        self.assertEqual(result["opens"], [])

    @patch("trading_bot.strategies.options._YF_AVAILABLE", False)
    def test_extreme_greed_targets_puts(self) -> None:
        strategy, _, alternative = _make_strategy()
        fg = MagicMock()
        fg.value = 80  # extreme greed
        alternative.get_fear_and_greed.return_value = fg
        state = _make_state()
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["option_type"], "put")

    def test_calls_bought_when_contract_found(self) -> None:
        strategy, _, alternative = _make_strategy(symbols=["AAPL"])
        fg = MagicMock()
        fg.value = 20  # extreme fear → buy calls
        alternative.get_fear_and_greed.return_value = fg

        contract = {
            "expiration_date": "2026-07-17",
            "strike": "145.0",
            "bid": 12.0,
            "ask": 12.5,
            "mark": 12.25,
            "delta": 0.75,
            "implied_volatility": 0.30,
            "volume": 500,
            "open_interest": 1000,
        }
        strategy._find_best_contract = MagicMock(return_value=contract)

        state = _make_state()
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["opens"]), 1)
        opened_sym = result["opens"][0]
        self.assertTrue(opened_sym.startswith("OPT:AAPL:"))
        self.assertIn("C", opened_sym)
        self.assertIn(opened_sym, state.positions)

    def test_puts_bought_when_contract_found(self) -> None:
        strategy, _, alternative = _make_strategy(symbols=["SPY"])
        fg = MagicMock()
        fg.value = 85  # extreme greed → buy puts
        alternative.get_fear_and_greed.return_value = fg

        contract = {
            "expiration_date": "2026-07-17",
            "strike": "610.0",
            "bid": 8.0,
            "ask": 8.4,
            "mark": 8.2,
            "delta": -0.72,
            "implied_volatility": 0.22,
            "volume": 300,
            "open_interest": 800,
        }
        strategy._find_best_contract = MagicMock(return_value=contract)

        state = _make_state()
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)

        self.assertEqual(len(result["opens"]), 1)
        self.assertIn("P", result["opens"][0])

    def test_skips_symbol_if_already_have_position(self) -> None:
        strategy, _, alternative = _make_strategy(symbols=["AAPL"])
        fg = MagicMock()
        fg.value = 20
        alternative.get_fear_and_greed.return_value = fg

        # Pre-seed an existing AAPL options position
        state = _make_state()
        state.positions["OPT:AAPL:20260717:145.0:C"] = Position(
            symbol="OPT:AAPL:20260717:145.0:C",
            asset_type="option",
            strategy="options_fg",
            quantity=1.0,
            entry_price=12.0,
            current_price=12.0,
            opened_at=self.now.isoformat(),
        )

        contract = {
            "expiration_date": "2026-07-17",
            "strike": "145.0",
            "bid": 12.0,
            "ask": 12.5,
            "mark": 12.25,
            "delta": 0.75,
            "implied_volatility": 0.30,
            "volume": 500,
            "open_interest": 1000,
        }
        strategy._find_best_contract = MagicMock(return_value=contract)
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["opens"], [])

    def test_max_positions_respected(self) -> None:
        strategy, _, alternative = _make_strategy(symbols=["AAPL", "SPY"], max_positions=1)
        fg = MagicMock()
        fg.value = 20
        alternative.get_fear_and_greed.return_value = fg

        contract = {
            "expiration_date": "2026-07-17",
            "strike": "145.0",
            "bid": 12.0,
            "ask": 12.5,
            "mark": 12.25,
            "delta": 0.75,
            "implied_volatility": 0.30,
            "volume": 500,
            "open_interest": 1000,
        }
        strategy._find_best_contract = MagicMock(return_value=contract)

        state = _make_state()
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        # With max_positions=1, only one position should be opened
        self.assertEqual(len(result["opens"]), 1)


# ── Position management (take-profit / stop-loss / DTE) ───────────────────────

class TestOptionsManagePositions(unittest.TestCase):
    def setUp(self) -> None:
        self.broker = _make_broker()
        self.risk = _make_risk()
        self.now = _now()

    def _add_position(
        self,
        state: PortfolioState,
        sym: str,
        entry: float,
        current: float,
        expiration: str = "2026-07-17",
    ) -> None:
        state.positions[sym] = Position(
            symbol=sym,
            asset_type="option",
            strategy="options_fg",
            quantity=1.0,
            entry_price=entry,
            current_price=current,
            opened_at=self.now.isoformat(),
            meta={
                "underlying": "AAPL",
                "expiration_date": expiration,
                "strike": "145.0",
                "option_type": "call",
                "contracts": 1,
                "venue": "robinhood",
            },
        )
        state.last_prices[sym] = current

    def test_take_profit_closes_position(self) -> None:
        strategy, _, _ = _make_strategy(take_profit_pct=0.50)
        state = _make_state()
        self._add_position(state, "OPT:AAPL:20260717:145.0:C", entry=10.0, current=15.5)

        strategy._get_current_option_price = MagicMock(return_value=None)
        closes = strategy._manage_positions(state, self.broker, self.now)

        self.assertIn("OPT:AAPL:20260717:145.0:C", closes)
        self.assertNotIn("OPT:AAPL:20260717:145.0:C", state.positions)

    def test_stop_loss_closes_position(self) -> None:
        strategy, _, _ = _make_strategy(stop_loss_pct=0.50)
        state = _make_state()
        self._add_position(state, "OPT:AAPL:20260717:145.0:C", entry=10.0, current=4.5)

        strategy._get_current_option_price = MagicMock(return_value=None)
        closes = strategy._manage_positions(state, self.broker, self.now)

        self.assertIn("OPT:AAPL:20260717:145.0:C", closes)

    def test_dte_expiry_closes_position(self) -> None:
        strategy, _, _ = _make_strategy(close_at_dte=7)
        state = _make_state()
        # Expiration 3 days from now → DTE = 3 < 7
        near_exp = "2026-06-18"
        self._add_position(state, "OPT:AAPL:20260618:145.0:C", entry=10.0, current=10.0, expiration=near_exp)

        strategy._get_current_option_price = MagicMock(return_value=None)
        closes = strategy._manage_positions(state, self.broker, self.now)

        self.assertIn("OPT:AAPL:20260618:145.0:C", closes)

    def test_profitable_position_not_closed_prematurely(self) -> None:
        strategy, _, _ = _make_strategy(take_profit_pct=0.50)
        state = _make_state()
        # Only +20% gain, below +50% threshold
        self._add_position(state, "OPT:AAPL:20260717:145.0:C", entry=10.0, current=12.0)

        strategy._get_current_option_price = MagicMock(return_value=None)
        closes = strategy._manage_positions(state, self.broker, self.now)

        self.assertEqual(closes, [])
        self.assertIn("OPT:AAPL:20260717:145.0:C", state.positions)

    def test_ignores_non_options_positions(self) -> None:
        strategy, _, _ = _make_strategy()
        state = _make_state()
        state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            asset_type="crypto",
            strategy="crypto_strategy",
            quantity=1.0,
            entry_price=50000.0,
            current_price=1.0,  # huge loss, but not an options position
            opened_at=self.now.isoformat(),
        )
        closes = strategy._manage_positions(state, self.broker, self.now)
        self.assertEqual(closes, [])


# ── Delta selection (_select_by_delta) ────────────────────────────────────────

class TestSelectByDelta(unittest.TestCase):
    def _strategy(self) -> OptionsStrategy:
        s, _, _ = _make_strategy(min_delta=0.70)
        return s

    def test_selects_contract_above_min_delta(self) -> None:
        contracts = [
            {"delta": "0.65", "bid_price": "5.0", "ask_price": "5.5",
             "adjusted_mark_price": "5.25", "implied_volatility": "0.30",
             "volume": "200", "open_interest": "600",
             "strike_price": "150.0", "expiration_date": "2026-07-17"},
            {"delta": "0.72", "bid_price": "8.0", "ask_price": "8.5",
             "adjusted_mark_price": "8.25", "implied_volatility": "0.28",
             "volume": "300", "open_interest": "900",
             "strike_price": "145.0", "expiration_date": "2026-07-17"},
        ]
        strategy = self._strategy()
        result = strategy._select_by_delta(contracts, "call")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result["delta"], 0.72)

    def test_returns_none_when_none_qualify(self) -> None:
        contracts = [
            {"delta": "0.50", "bid_price": "5.0", "ask_price": "5.5",
             "adjusted_mark_price": "5.25", "implied_volatility": "0.30",
             "volume": "200", "open_interest": "600",
             "strike_price": "150.0", "expiration_date": "2026-07-17"},
        ]
        strategy = self._strategy()
        result = strategy._select_by_delta(contracts, "call")
        self.assertIsNone(result)

    def test_skips_high_iv(self) -> None:
        contracts = [
            {"delta": "0.75", "bid_price": "5.0", "ask_price": "5.5",
             "adjusted_mark_price": "5.25", "implied_volatility": "2.00",  # > max_iv=1.50
             "volume": "200", "open_interest": "600",
             "strike_price": "145.0", "expiration_date": "2026-07-17"},
        ]
        strategy = self._strategy()
        result = strategy._select_by_delta(contracts, "call")
        self.assertIsNone(result)

    def test_uses_closest_to_min_delta(self) -> None:
        contracts = [
            {"delta": "0.72", "bid_price": "8.0", "ask_price": "8.5",
             "adjusted_mark_price": "8.25", "implied_volatility": "0.28",
             "volume": "300", "open_interest": "900",
             "strike_price": "145.0", "expiration_date": "2026-07-17"},
            {"delta": "0.90", "bid_price": "15.0", "ask_price": "15.5",
             "adjusted_mark_price": "15.25", "implied_volatility": "0.28",
             "volume": "300", "open_interest": "900",
             "strike_price": "130.0", "expiration_date": "2026-07-17"},
        ]
        strategy = self._strategy()
        result = strategy._select_by_delta(contracts, "call")
        # Should prefer delta=0.72 (closest to 0.70) over delta=0.90
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result["delta"], 0.72)


# ── Helpers ────────────────────────────────────────────────────────────────────

class TestHelpers(unittest.TestCase):
    def test_pos_symbol_call(self) -> None:
        sym = OptionsStrategy._pos_symbol("AAPL", "2026-07-17", "145.0", "call")
        self.assertEqual(sym, "OPT:AAPL:20260717:145.0:C")

    def test_pos_symbol_put(self) -> None:
        sym = OptionsStrategy._pos_symbol("SPY", "2026-07-17", "610.0", "put")
        self.assertEqual(sym, "OPT:SPY:20260717:610.0:P")

    def test_days_to_expiration_future(self) -> None:
        now = datetime(2026, 6, 15, tzinfo=timezone.utc)
        dte = OptionsStrategy._days_to_expiration("2026-07-17", now)
        self.assertEqual(dte, 32)

    def test_days_to_expiration_past(self) -> None:
        now = datetime(2026, 6, 15, tzinfo=timezone.utc)
        dte = OptionsStrategy._days_to_expiration("2026-06-10", now)
        self.assertEqual(dte, 0)

    def test_days_to_expiration_invalid(self) -> None:
        now = _now()
        dte = OptionsStrategy._days_to_expiration("not-a-date", now)
        self.assertEqual(dte, 0)


if __name__ == "__main__":
    unittest.main()
