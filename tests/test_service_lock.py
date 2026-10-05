"""Service cycles must not interleave: two concurrent runs could place the same bet twice."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

from trading_bot.service import TradingBotService, _locked


class _Fake:
    def __init__(self):
        self._state_lock = threading.RLock()
        self.active = 0
        self.max_active = 0

    @_locked
    def run(self):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        self.active -= 1


class ServiceLockTests(unittest.TestCase):
    def test_concurrent_runs_are_serialised(self):
        fake = _Fake()
        threads = [threading.Thread(target=fake.run) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(fake.max_active, 1)

    def test_public_state_methods_are_locked(self):
        for name in ("run", "status", "engage_kill_switch", "clear_kill_switch"):
            self.assertTrue(hasattr(getattr(TradingBotService, name), "__wrapped__"), name)

    def test_reentrant_calls_do_not_deadlock(self):
        fake = _Fake()
        with fake._state_lock:
            fake.run()


if __name__ == "__main__":
    unittest.main()
