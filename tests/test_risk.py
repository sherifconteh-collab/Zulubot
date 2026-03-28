from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from trading_bot.config import RiskConfig
from trading_bot.models import PortfolioState
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


if __name__ == "__main__":
    unittest.main()

