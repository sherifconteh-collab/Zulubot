"""Live-mode drawdown baseline and kill-switch safety."""

from __future__ import annotations

import logging
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from trading_bot.broker import PaperBroker
from trading_bot.config import RiskConfig
from trading_bot.models import PortfolioState, Position
from trading_bot.risk import RiskManager
from trading_bot.service import TradingBotService

_NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def _service(initial_cash=0.0, paper_mode=False):
    svc = object.__new__(TradingBotService)
    svc.logger = logging.getLogger("t")
    svc.config = SimpleNamespace(initial_cash=initial_cash, paper_mode=paper_mode)
    svc.broker = PaperBroker(paper_mode, logging.getLogger("b"))
    return svc


class SeedPeakTests(unittest.TestCase):
    def test_peak_is_seeded_from_the_same_equity_the_risk_check_measures(self):
        svc = _service()
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 13.6})
        svc._seed_live_peak(state)
        self.assertEqual(state.peak_equity, 13.6)
        self.assertEqual(state.daily_baseline_equity, 13.6)

    def test_impossibly_high_stale_peak_is_reset_but_not_during_a_kill_switch(self):
        svc = _service()
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 13.6}, peak_equity=115.0)
        svc._seed_live_peak(state)
        self.assertEqual(state.peak_equity, 13.6)
        state2 = PortfolioState(cash=0.0, cash_reserves={"kalshi": 13.6}, peak_equity=115.0)
        state2.kill_switch_engaged = True
        svc._seed_live_peak(state2)
        self.assertEqual(state2.peak_equity, 115.0)

    def test_a_real_moderate_drawdown_is_not_reset(self):
        svc = _service()
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 80.0}, peak_equity=100.0)
        svc._seed_live_peak(state)
        self.assertEqual(state.peak_equity, 100.0)


class KillSwitchDataSafetyTests(unittest.TestCase):
    def _risk(self):
        return RiskManager(RiskConfig(kill_switch_drawdown=0.15, daily_halt_drawdown=0.05,
                                      max_position_fraction=0.02, max_open_positions=0, trailing_stop_pct=0.0,
                                      cooldown_consecutive_losses=0, cooldown_periods=3))

    def test_bad_data_cannot_trigger_the_kill_switch(self):
        state = PortfolioState(cash=100.0, peak_equity=100.0, daily_baseline_equity=100.0)
        snap = self._risk().evaluate(state, 40.0, _NOW, allow_kill_switch=False)
        self.assertFalse(snap.kill_switch_active)
        self.assertAlmostEqual(snap.total_drawdown, 0.6)

    def test_real_drawdown_still_triggers_it(self):
        state = PortfolioState(cash=100.0, peak_equity=100.0, daily_baseline_equity=100.0)
        self.assertTrue(self._risk().evaluate(state, 40.0, _NOW).kill_switch_triggered_now)


if __name__ == "__main__":
    unittest.main()
