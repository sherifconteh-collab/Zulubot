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
        for name in ("run", "status", "engage_kill_switch", "clear_kill_switch",
                     "list_sports_games", "quote_sports_bet", "place_sports_bet",
                     "quote_sports_parlay", "place_sports_parlay"):
            self.assertTrue(hasattr(getattr(TradingBotService, name), "__wrapped__"), name)

    def test_reentrant_calls_do_not_deadlock(self):
        fake = _Fake()
        with fake._state_lock:
            fake.run()


class SportsBetServiceTests(unittest.TestCase):
    def _fake(self, **sports):
        from datetime import timezone
        from types import SimpleNamespace
        from trading_bot.models import PortfolioState
        cfg = dict(enabled=True, ignore_global_halt=True)
        cfg.update(sports)
        state = PortfolioState(cash=100.0)
        fake = SimpleNamespace(
            config=SimpleNamespace(kalshi_sports=SimpleNamespace(**cfg), paper_mode=True, timezone=timezone.utc),
            state_store=MagicMock(), broker=MagicMock(), kalshi_sports_strategy=MagicMock(),
            _hydrate_ai_budget=MagicMock(), refresh_open_positions=MagicMock(), _persist_ai_budget=MagicMock(),
        )
        fake.state_store.load.return_value = state
        fake._sports_bet_state = lambda now: TradingBotService._sports_bet_state(fake, now)
        fake._sports_bet_blocker = lambda st: TradingBotService._sports_bet_blocker(fake, st)
        return fake, state

    def test_successful_bet_is_saved(self):
        fake, state = self._fake()
        fake.kalshi_sports_strategy.place_user_bet.return_value = {"ok": True, "symbol": "KS:X:YES"}
        out = TradingBotService.place_sports_bet.__wrapped__(fake, "X", "yes", 5.0, 0.4)
        self.assertTrue(out["ok"])
        fake.state_store.save.assert_called_once_with(state)
        self.assertEqual(fake.kalshi_sports_strategy.place_user_bet.call_args.kwargs["max_ask"], 0.4)

    def test_refused_bet_is_not_saved(self):
        fake, _ = self._fake()
        fake.kalshi_sports_strategy.place_user_bet.return_value = {"ok": False, "reason": "no"}
        TradingBotService.place_sports_bet.__wrapped__(fake, "X", "yes", 5.0)
        fake.state_store.save.assert_not_called()

    def test_disabled_strategy_blocks_bets(self):
        fake, _ = self._fake(enabled=False)
        out = TradingBotService.place_sports_bet.__wrapped__(fake, "X", "yes", 5.0)
        self.assertFalse(out["ok"])
        fake.kalshi_sports_strategy.place_user_bet.assert_not_called()

    def test_successful_parlay_is_saved_and_refused_one_is_not(self):
        fake, state = self._fake()
        fake.kalshi_sports_strategy.place_user_parlay.return_value = {"ok": True}
        self.assertTrue(TradingBotService.place_sports_parlay.__wrapped__(fake, [{"ticker": "A", "side": "yes"}], 5.0, 0.3)["ok"])
        fake.state_store.save.assert_called_once_with(state)
        kwargs = fake.kalshi_sports_strategy.place_user_parlay.call_args.kwargs
        self.assertEqual((kwargs["amount"], kwargs["max_price"]), (5.0, 0.3))
        fake, _ = self._fake()
        fake.kalshi_sports_strategy.place_user_parlay.return_value = {"ok": False, "reason": "no quote"}
        TradingBotService.place_sports_parlay.__wrapped__(fake, [], None, None)
        fake.state_store.save.assert_not_called()

    def test_parlay_quote_respects_disabled_strategy(self):
        fake, _ = self._fake(enabled=False)
        out = TradingBotService.quote_sports_parlay.__wrapped__(fake, [], None, None)
        self.assertFalse(out["ok"])
        fake.kalshi_sports_strategy.quote_user_parlay.assert_not_called()

    def test_kill_switch_blocks_only_when_sports_honours_it(self):
        fake, state = self._fake(ignore_global_halt=False)
        state.kill_switch_engaged = True
        out = TradingBotService.quote_sports_bet.__wrapped__(fake, "X", "yes", 5.0)
        self.assertFalse(out["ok"])
        fake, state = self._fake(ignore_global_halt=True)
        state.kill_switch_engaged = True
        fake.kalshi_sports_strategy.quote_user_bet.return_value = {"ok": True}
        self.assertTrue(TradingBotService.quote_sports_bet.__wrapped__(fake, "X", "yes", 5.0)["ok"])


if __name__ == "__main__":
    unittest.main()
