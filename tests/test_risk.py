from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from trading_bot.config import RiskConfig
from trading_bot.models import PortfolioState, Position
from trading_bot.risk import RiskManager


class RiskManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timezone = ZoneInfo("America/New_York")
        self.now = datetime(2026, 3, 28, 12, 0, tzinfo=self.timezone)

    def test_daily_halt_triggers_at_five_percent_drawdown(self) -> None:
        manager = RiskManager(RiskConfig())
        state = PortfolioState(
            cash=95_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        snapshot = manager.evaluate(state, equity=95_000.0, now=self.now)

        self.assertTrue(snapshot.daily_halt_active)
        self.assertFalse(snapshot.kill_switch_active)
        self.assertAlmostEqual(snapshot.daily_drawdown, 0.05)

    def test_kill_switch_triggers_at_fifteen_percent_drawdown(self) -> None:
        manager = RiskManager(RiskConfig())
        state = PortfolioState(
            cash=85_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        snapshot = manager.evaluate(state, equity=85_000.0, now=self.now)

        self.assertTrue(snapshot.kill_switch_active)
        self.assertTrue(snapshot.kill_switch_triggered_now)
        self.assertAlmostEqual(snapshot.total_drawdown, 0.15)

    # ------------------------------------------------------------------ #
    # Trailing stop-loss tests                                             #
    # ------------------------------------------------------------------ #

    def _make_position(self, symbol: str, entry: float, current: float) -> Position:
        return Position(
            symbol=symbol,
            asset_type="crypto",
            strategy="test",
            quantity=1.0,
            entry_price=entry,
            current_price=current,
            opened_at=self.now.isoformat(),
        )

    def test_trailing_stop_disabled_by_default(self) -> None:
        manager = RiskManager(RiskConfig())
        state = PortfolioState(cash=90_000.0)
        state.positions["BTC"] = self._make_position("BTC", entry=100.0, current=80.0)
        state.position_high_watermarks["BTC"] = 120.0

        result = manager.check_trailing_stops(state)
        self.assertEqual(result, [])

    def test_trailing_stop_triggers_when_drop_exceeds_threshold(self) -> None:
        manager = RiskManager(RiskConfig(trailing_stop_pct=0.10))
        state = PortfolioState(cash=90_000.0)
        state.positions["BTC"] = self._make_position("BTC", entry=100.0, current=108.0)
        state.position_high_watermarks["BTC"] = 120.0  # dropped from 120 → 108 = 10%

        result = manager.check_trailing_stops(state)
        self.assertEqual(result, ["BTC"])

    def test_trailing_stop_does_not_trigger_below_threshold(self) -> None:
        manager = RiskManager(RiskConfig(trailing_stop_pct=0.10))
        state = PortfolioState(cash=90_000.0)
        state.positions["BTC"] = self._make_position("BTC", entry=100.0, current=115.0)
        state.position_high_watermarks["BTC"] = 120.0  # dropped from 120 → 115 = 4.2%

        result = manager.check_trailing_stops(state)
        self.assertEqual(result, [])

    def test_trailing_stop_uses_entry_price_when_no_watermark(self) -> None:
        manager = RiskManager(RiskConfig(trailing_stop_pct=0.10))
        state = PortfolioState(cash=90_000.0)
        state.positions["ETH"] = self._make_position("ETH", entry=100.0, current=89.0)
        # No watermark recorded yet — should use entry_price as fallback

        result = manager.check_trailing_stops(state)
        self.assertEqual(result, ["ETH"])

    def test_update_high_watermarks_tracks_peak(self) -> None:
        manager = RiskManager(RiskConfig())
        state = PortfolioState(cash=90_000.0)
        state.positions["BTC"] = self._make_position("BTC", entry=100.0, current=150.0)

        manager.update_high_watermarks(state)
        self.assertEqual(state.position_high_watermarks["BTC"], 150.0)

        # Price drops — watermark should NOT decrease
        state.positions["BTC"].current_price = 130.0
        manager.update_high_watermarks(state)
        self.assertEqual(state.position_high_watermarks["BTC"], 150.0)

    def test_update_high_watermarks_cleans_closed_positions(self) -> None:
        manager = RiskManager(RiskConfig())
        state = PortfolioState(cash=90_000.0)
        state.position_high_watermarks["OLD"] = 200.0  # stale entry

        manager.update_high_watermarks(state)
        self.assertNotIn("OLD", state.position_high_watermarks)

    # ------------------------------------------------------------------ #
    # Max open positions tests                                             #
    # ------------------------------------------------------------------ #

    def test_max_open_positions_unlimited_by_default(self) -> None:
        manager = RiskManager(RiskConfig())
        state = PortfolioState(cash=90_000.0)
        for i in range(10):
            state.positions[f"SYM{i}"] = self._make_position(f"SYM{i}", 100.0, 100.0)

        self.assertTrue(manager.can_open_new_position(state))

    def test_max_open_positions_blocks_when_at_limit(self) -> None:
        manager = RiskManager(RiskConfig(max_open_positions=3))
        state = PortfolioState(cash=90_000.0)
        for i in range(3):
            state.positions[f"SYM{i}"] = self._make_position(f"SYM{i}", 100.0, 100.0)

        self.assertFalse(manager.can_open_new_position(state))

    def test_max_open_positions_allows_below_limit(self) -> None:
        manager = RiskManager(RiskConfig(max_open_positions=3))
        state = PortfolioState(cash=90_000.0)
        state.positions["SYM0"] = self._make_position("SYM0", 100.0, 100.0)

        self.assertTrue(manager.can_open_new_position(state))

    # ------------------------------------------------------------------ #
    # Consecutive loss cooldown tests                                      #
    # ------------------------------------------------------------------ #

    def test_cooldown_disabled_by_default(self) -> None:
        manager = RiskManager(RiskConfig())
        state = PortfolioState(cash=90_000.0)

        for _ in range(10):
            manager.record_trade_result(state, is_loss=True)

        self.assertEqual(state.cooldown_remaining, 0)
        self.assertTrue(manager.can_open_new_position(state))

    def test_cooldown_triggers_after_consecutive_losses(self) -> None:
        manager = RiskManager(RiskConfig(cooldown_consecutive_losses=3, cooldown_periods=2))
        state = PortfolioState(cash=90_000.0)

        manager.record_trade_result(state, is_loss=True)
        self.assertEqual(state.consecutive_losses, 1)
        self.assertEqual(state.cooldown_remaining, 0)

        manager.record_trade_result(state, is_loss=True)
        self.assertEqual(state.consecutive_losses, 2)
        self.assertEqual(state.cooldown_remaining, 0)

        manager.record_trade_result(state, is_loss=True)
        self.assertEqual(state.consecutive_losses, 3)
        self.assertEqual(state.cooldown_remaining, 2)
        self.assertFalse(manager.can_open_new_position(state))

    def test_cooldown_resets_on_win(self) -> None:
        manager = RiskManager(RiskConfig(cooldown_consecutive_losses=3, cooldown_periods=2))
        state = PortfolioState(cash=90_000.0)

        manager.record_trade_result(state, is_loss=True)
        manager.record_trade_result(state, is_loss=True)
        self.assertEqual(state.consecutive_losses, 2)

        manager.record_trade_result(state, is_loss=False)
        self.assertEqual(state.consecutive_losses, 0)
        self.assertEqual(state.cooldown_remaining, 0)

    def test_tick_cooldown_decrements_and_expires(self) -> None:
        manager = RiskManager(RiskConfig(cooldown_consecutive_losses=2, cooldown_periods=3))
        state = PortfolioState(cash=90_000.0)

        # Trigger cooldown
        manager.record_trade_result(state, is_loss=True)
        manager.record_trade_result(state, is_loss=True)
        self.assertEqual(state.cooldown_remaining, 3)
        self.assertFalse(manager.can_open_new_position(state))

        manager.tick_cooldown(state)
        self.assertEqual(state.cooldown_remaining, 2)
        self.assertFalse(manager.can_open_new_position(state))

        manager.tick_cooldown(state)
        self.assertEqual(state.cooldown_remaining, 1)

        manager.tick_cooldown(state)
        self.assertEqual(state.cooldown_remaining, 0)
        self.assertEqual(state.consecutive_losses, 0)
        self.assertTrue(manager.can_open_new_position(state))

    def test_cooldown_active_in_risk_snapshot(self) -> None:
        manager = RiskManager(RiskConfig(cooldown_consecutive_losses=1, cooldown_periods=2))
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        manager.record_trade_result(state, is_loss=True)
        snapshot = manager.evaluate(state, equity=100_000.0, now=self.now)
        self.assertTrue(snapshot.cooldown_active)

    def test_cooldown_not_active_when_no_losses(self) -> None:
        manager = RiskManager(RiskConfig(cooldown_consecutive_losses=3, cooldown_periods=2))
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        snapshot = manager.evaluate(state, equity=100_000.0, now=self.now)
        self.assertFalse(snapshot.cooldown_active)

    # ------------------------------------------------------------------ #
    # State serialisation includes new fields                              #
    # ------------------------------------------------------------------ #

    def test_portfolio_state_serialises_new_fields(self) -> None:
        state = PortfolioState(
            cash=100_000.0,
            position_high_watermarks={"BTC": 150.0},
            consecutive_losses=2,
            cooldown_remaining=1,
        )
        d = state.to_dict()
        self.assertEqual(d["position_high_watermarks"], {"BTC": 150.0})
        self.assertEqual(d["consecutive_losses"], 2)
        self.assertEqual(d["cooldown_remaining"], 1)


if __name__ == "__main__":
    unittest.main()

