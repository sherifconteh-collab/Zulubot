"""Coinbase is data-only unless COINBASE_TRADING_ENABLED=true: no orders, no balance, no crypto cycles."""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from trading_bot.config import CoinbaseConfig
from trading_bot.models import PortfolioState
from trading_bot.scheduler import SchedulerService
from trading_bot.service import TradingBotService


def _service(paper_mode, trading_enabled):
    svc = object.__new__(TradingBotService)
    svc.logger = logging.getLogger("t")
    svc.config = SimpleNamespace(paper_mode=paper_mode, coinbase=CoinbaseConfig(trading_enabled=trading_enabled))
    svc.coinbase_broker = MagicMock()
    return svc


class CoinbaseDataOnlyTests(unittest.TestCase):
    def test_default_is_data_only(self):
        self.assertFalse(CoinbaseConfig().trading_enabled)

    def test_live_without_trading_flag_is_inactive(self):
        self.assertFalse(_service(False, False).coinbase_trading_active)

    def test_live_with_trading_flag_and_paper_mode_are_active(self):
        self.assertTrue(_service(False, True).coinbase_trading_active)
        self.assertTrue(_service(True, False).coinbase_trading_active)

    def test_balance_sync_skips_the_api_and_zeroes_stale_cash(self):
        svc = _service(False, False)
        state = PortfolioState(cash=500.0)
        svc._sync_coinbase_balance(state)
        self.assertEqual(state.cash, 0.0)
        svc.coinbase_broker.get_total_balance.assert_not_called()

    def test_balance_sync_runs_when_trading_enabled(self):
        svc = _service(False, True)
        svc.coinbase_broker.get_total_balance.return_value = (120.0, {"USD": 20.0, "BTC": 100.0})
        state = PortfolioState(cash=0.0)
        svc._sync_coinbase_balance(state)
        self.assertEqual(state.cash, 20.0)


class SchedulerGatingTests(unittest.TestCase):
    def _sched(self, paper_mode, trading_enabled):
        s = object.__new__(SchedulerService)
        s.config = SimpleNamespace(
            paper_mode=paper_mode, coinbase=CoinbaseConfig(trading_enabled=trading_enabled),
            altcoin_rotation=SimpleNamespace(enabled=True, check_interval_minutes=60),
        )
        s.scheduler_config = SimpleNamespace(crypto_interval_minutes=10)
        s._last_crypto_run = None
        s._last_rotation_run = None
        return s

    def test_crypto_and_rotation_do_not_run_live_without_coinbase_trading(self):
        s = self._sched(False, False)
        self.assertFalse(s._should_run_crypto(None))
        self.assertFalse(s._should_run_rotation(None))

    def test_they_run_when_coinbase_trading_is_enabled(self):
        s = self._sched(False, True)
        self.assertTrue(s._should_run_crypto(None))
        self.assertTrue(s._should_run_rotation(None))


if __name__ == "__main__":
    unittest.main()
