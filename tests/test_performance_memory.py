from __future__ import annotations

import unittest

from trading_bot.config import LearningConfig
from trading_bot.models import PortfolioState
from trading_bot.performance_memory import (
    is_symbol_benched,
    record_close,
    summary,
)


def _fresh_state() -> PortfolioState:
    return PortfolioState(cash=10_000.0)


class PerformanceMemoryTests(unittest.TestCase):
    def test_win_increments_wins_and_resets_loss_streak(self) -> None:
        state = _fresh_state()
        cfg = LearningConfig()
        record_close(state, symbol="BTCUSDT", realized_pnl=-10.0, config=cfg)
        record_close(state, symbol="BTCUSDT", realized_pnl=25.0, config=cfg)

        rec = summary(state, "BTCUSDT")
        self.assertEqual(rec["wins"], 1)
        self.assertEqual(rec["losses"], 1)
        self.assertEqual(rec["consecutive_losses"], 0)
        self.assertEqual(rec["cooldown_remaining"], 0)
        self.assertAlmostEqual(rec["total_pnl"], 15.0)
        self.assertFalse(is_symbol_benched(state, "BTCUSDT", config=cfg))

    def test_break_even_close_does_not_reset_or_increment_streak(self) -> None:
        state = _fresh_state()
        cfg = LearningConfig()
        record_close(state, symbol="ETHUSDT", realized_pnl=-5.0, config=cfg)
        record_close(state, symbol="ETHUSDT", realized_pnl=0.0, config=cfg)

        rec = summary(state, "ETHUSDT")
        self.assertEqual(rec["wins"], 0)
        self.assertEqual(rec["losses"], 1)
        self.assertEqual(rec["consecutive_losses"], 1)

    def test_cooldown_triggers_after_n_consecutive_losses_and_decrements(self) -> None:
        state = _fresh_state()
        cfg = LearningConfig(
            enabled=True, loser_cooldown_losses=3, loser_cooldown_cycles=2,
        )
        for _ in range(3):
            record_close(state, symbol="SOLUSDT", realized_pnl=-5.0, config=cfg)

        # First two checks see the bench; each call decrements the counter.
        self.assertTrue(
            is_symbol_benched(state, "SOLUSDT", config=cfg, decrement=True)
        )
        self.assertTrue(
            is_symbol_benched(state, "SOLUSDT", config=cfg, decrement=True)
        )
        # Cooldown has expired on the third cycle.
        self.assertFalse(
            is_symbol_benched(state, "SOLUSDT", config=cfg, decrement=True)
        )

    def test_winning_close_clears_active_cooldown(self) -> None:
        state = _fresh_state()
        cfg = LearningConfig(
            enabled=True, loser_cooldown_losses=2, loser_cooldown_cycles=10,
        )
        record_close(state, symbol="LINKUSDT", realized_pnl=-5.0, config=cfg)
        record_close(state, symbol="LINKUSDT", realized_pnl=-5.0, config=cfg)
        self.assertTrue(is_symbol_benched(state, "LINKUSDT", config=cfg))

        record_close(state, symbol="LINKUSDT", realized_pnl=7.0, config=cfg)
        self.assertFalse(is_symbol_benched(state, "LINKUSDT", config=cfg))
        rec = summary(state, "LINKUSDT")
        self.assertEqual(rec["consecutive_losses"], 0)
        self.assertEqual(rec["cooldown_remaining"], 0)

    def test_disabled_config_is_a_no_op(self) -> None:
        state = _fresh_state()
        cfg = LearningConfig(enabled=False)
        record_close(state, symbol="AVAXUSDT", realized_pnl=-100.0, config=cfg)
        self.assertEqual(summary(state, "AVAXUSDT"), {})
        self.assertFalse(is_symbol_benched(state, "AVAXUSDT", config=cfg))

    def test_unseen_symbol_is_never_benched(self) -> None:
        state = _fresh_state()
        self.assertFalse(
            is_symbol_benched(state, "NEWUSDT", config=LearningConfig())
        )


if __name__ == "__main__":
    unittest.main()
