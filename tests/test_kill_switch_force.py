from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_bot.config import AppConfig
from trading_bot.service import TradingBotService


class _ConstantEquityBroker:
    """Stand-in for service.broker — only ``portfolio_equity`` is used by clear_kill_switch."""

    def __init__(self, equity: float, real_broker) -> None:
        self._equity = equity
        self._real = real_broker

    def __getattr__(self, name):  # pragma: no cover - delegation passthrough
        return getattr(self._real, name)

    def portfolio_equity(self, state) -> float:
        return self._equity


def _make_service_with_initial_state(state_payload: dict) -> tuple[TradingBotService, Path]:
    """Build a service backed by a tmp state.json containing ``state_payload``."""
    tmp = Path(tempfile.mkdtemp(prefix="zb_killswitch_"))
    os.environ["BOT_DATA_DIR"] = str(tmp / "data")
    os.environ["BOT_LOG_DIR"] = str(tmp / "logs")
    os.environ["PAPER_MODE"] = "true"
    os.environ["INITIAL_CASH"] = "100"
    os.environ["TRADING_AGENTS_ENABLED"] = "false"
    cfg = AppConfig.load()
    cfg.state_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.state_path.write_text(json.dumps(state_payload))
    svc = TradingBotService(cfg, logging.getLogger("zb_test"))
    return svc, cfg.state_path


class ClearKillSwitchForceTests(unittest.TestCase):
    """Reproduce the stuck-kill-switch scenario from the Telegram screenshot
    (peak_equity=$113, current equity=$23.55, drawdown=79.23%) and verify
    that /resume force resets the high-water mark.
    """

    def _stuck_state(self) -> dict:
        # Matches the user's stuck state: peak ~$113, cash $14, kill switch on.
        # The PaperBroker's portfolio_equity will be overridden in the test;
        # this payload only seeds the state file for the StateStore.
        return {
            "cash": 14.00,
            "cash_reserves": {"kalshi": 9.55},
            "positions": {},
            "last_prices": {},
            "last_price_timestamps": {},
            "daily_baseline_equity": 100.0,
            "daily_baseline_date": "2026-04-21",
            "peak_equity": 113.00,
            "halt_new_trades_for_day": False,
            "kill_switch_engaged": True,
            "realized_pnl": 0.0,
            "rotation_data": {},
            "price_history": {},
            "last_price_record_date": "",
            "position_high_watermarks": {},
            "consecutive_losses": 0,
            "cooldown_remaining": 0,
            "performance_memory": {},
        }

    def test_clear_kill_switch_raises_when_drawdown_exceeds_threshold(self) -> None:
        svc, _ = _make_service_with_initial_state(self._stuck_state())
        svc.broker = _ConstantEquityBroker(equity=23.55, real_broker=svc.broker)

        with self.assertRaises(RuntimeError) as ctx:
            svc.clear_kill_switch()

        self.assertIn("Kill switch cannot be cleared", str(ctx.exception))
        self.assertRegex(str(ctx.exception), r"79\.\d{2}%")
        self.assertIn("threshold is 15.00%", str(ctx.exception))
        # State must be untouched on the failure path.
        state = svc.state_store.load()
        self.assertTrue(state.kill_switch_engaged)
        self.assertEqual(state.peak_equity, 113.00)

    def test_force_clear_resets_peak_to_current_equity_and_clears_flag(self) -> None:
        # Seed with halt_new_trades_for_day=True and today's date so the
        # date-rollover path in _evaluate_risk() cannot clear the flag on its
        # own — only our explicit state.halt_new_trades_for_day = False does it.
        today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        payload = self._stuck_state()
        payload["halt_new_trades_for_day"] = True
        payload["daily_baseline_date"] = today
        svc, _ = _make_service_with_initial_state(payload)
        svc.broker = _ConstantEquityBroker(equity=23.55, real_broker=svc.broker)

        summary = svc.clear_kill_switch(force=True)

        state = svc.state_store.load()
        self.assertFalse(state.kill_switch_engaged)
        self.assertAlmostEqual(state.peak_equity, 23.55)
        self.assertAlmostEqual(state.daily_baseline_equity, 23.55)
        self.assertFalse(state.halt_new_trades_for_day)

        ks = (summary or {}).get("modules", {}).get("kill_switch", {})
        self.assertEqual(ks.get("status"), "cleared")
        self.assertEqual(ks.get("reason"), "manual_resume_forced")
        self.assertAlmostEqual(ks.get("peak_reset_from"), 113.00)
        self.assertAlmostEqual(ks.get("peak_reset_to"), 23.55)

    def test_force_clear_also_clears_daily_halt_flag(self) -> None:
        # Scenario: daily drawdown tripped halt_new_trades_for_day before the
        # kill switch fired.  A same-day /resume force must unblock both flags
        # so that new entries are permitted immediately.
        # Use today's actual date so the date-rollover path in _evaluate_risk()
        # cannot clear the flag on its own.
        today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        state_payload = self._stuck_state()
        state_payload["halt_new_trades_for_day"] = True
        state_payload["daily_baseline_date"] = today
        svc, _ = _make_service_with_initial_state(state_payload)
        svc.broker = _ConstantEquityBroker(equity=23.55, real_broker=svc.broker)

        svc.clear_kill_switch(force=True)

        state = svc.state_store.load()
        self.assertFalse(state.kill_switch_engaged)
        self.assertFalse(state.halt_new_trades_for_day)

    def test_clear_without_force_succeeds_when_drawdown_under_threshold(self) -> None:
        # Recovered scenario: peak $100, equity $95 → 5% drawdown, well below 15%.
        payload = self._stuck_state()
        payload["peak_equity"] = 100.0
        svc, _ = _make_service_with_initial_state(payload)
        svc.broker = _ConstantEquityBroker(equity=95.0, real_broker=svc.broker)

        summary = svc.clear_kill_switch()  # no force needed

        state = svc.state_store.load()
        self.assertFalse(state.kill_switch_engaged)
        self.assertEqual(state.peak_equity, 100.0)  # NOT reset
        ks = (summary or {}).get("modules", {}).get("kill_switch", {})
        self.assertEqual(ks.get("reason"), "manual_resume")
        self.assertNotIn("peak_reset_from", ks)


if __name__ == "__main__":
    unittest.main()
