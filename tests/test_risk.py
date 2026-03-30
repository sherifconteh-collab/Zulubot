from __future__ import annotations

import logging
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

    def test_cooldown_does_not_extend_while_already_active(self) -> None:
        """Additional losses during an active cooldown must not reset the timer."""
        manager = RiskManager(RiskConfig(cooldown_consecutive_losses=2, cooldown_periods=5))
        state = PortfolioState(cash=90_000.0)

        # Trigger cooldown
        manager.record_trade_result(state, is_loss=True)
        manager.record_trade_result(state, is_loss=True)
        self.assertEqual(state.cooldown_remaining, 5)

        # Tick down a bit
        manager.tick_cooldown(state)
        manager.tick_cooldown(state)
        self.assertEqual(state.cooldown_remaining, 3)

        # Another loss recorded during cooldown should NOT reset the timer
        manager.record_trade_result(state, is_loss=True)
        self.assertEqual(state.cooldown_remaining, 3)

    # ------------------------------------------------------------------ #
    # Broker on_close callback tests                                       #
    # ------------------------------------------------------------------ #

    def test_broker_on_close_callback_fires(self) -> None:
        """Verify that PaperBroker invokes the on_close callback on every close."""
        from trading_bot.broker import PaperBroker

        broker = PaperBroker(paper_mode=True, logger=logging.getLogger("test"))
        results: list[tuple[float, float]] = []

        def _on_close(state: PortfolioState, entry: float, close: float) -> None:
            results.append((entry, close))

        broker.set_on_close(_on_close)

        state = PortfolioState(cash=100_000.0)
        broker.open_position(
            state, symbol="BTC", asset_type="crypto", strategy="test",
            price=100.0, notional=1000.0, opened_at=self.now,
        )
        broker.close_position(state, symbol="BTC", price=90.0, reason="test_close")

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][0], 100.0)
        self.assertAlmostEqual(results[0][1], 90.0)

    def test_broker_on_close_not_called_without_callback(self) -> None:
        """Verify close works normally when no callback is set."""
        from trading_bot.broker import PaperBroker

        broker = PaperBroker(paper_mode=True, logger=logging.getLogger("test"))
        state = PortfolioState(cash=100_000.0)
        broker.open_position(
            state, symbol="ETH", asset_type="crypto", strategy="test",
            price=50.0, notional=500.0, opened_at=self.now,
        )
        closed = broker.close_position(state, symbol="ETH", price=55.0, reason="test")
        self.assertIsNotNone(closed)

    # ------------------------------------------------------------------ #
    # State persistence round-trip                                         #
    # ------------------------------------------------------------------ #

    def test_state_store_round_trips_new_fields(self) -> None:
        """Verify StateStore.load() hydrates the new risk fields from disk."""
        import json
        import tempfile
        from pathlib import Path
        from trading_bot.state import StateStore

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            store = StateStore(path, initial_cash=100_000.0, timezone=self.timezone)

            state = PortfolioState(
                cash=90_000.0,
                position_high_watermarks={"BTC": 120.0, "ETH": 50.0},
                consecutive_losses=3,
                cooldown_remaining=2,
            )
            store.save(state)

            loaded = store.load()
            self.assertEqual(loaded.position_high_watermarks, {"BTC": 120.0, "ETH": 50.0})
            self.assertEqual(loaded.consecutive_losses, 3)
            self.assertEqual(loaded.cooldown_remaining, 2)

    def test_state_store_loads_defaults_for_old_state_files(self) -> None:
        """Verify StateStore.load() provides safe defaults for legacy state files."""
        import json
        import tempfile
        from pathlib import Path
        from trading_bot.state import StateStore

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            # Simulate an old state file without the new fields
            old_payload = {"cash": 80_000.0, "positions": {}, "last_prices": {}}
            path.write_text(json.dumps(old_payload), encoding="utf-8")

            store = StateStore(path, initial_cash=100_000.0, timezone=self.timezone)
            loaded = store.load()
            self.assertEqual(loaded.position_high_watermarks, {})
            self.assertEqual(loaded.consecutive_losses, 0)
            self.assertEqual(loaded.cooldown_remaining, 0)

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

