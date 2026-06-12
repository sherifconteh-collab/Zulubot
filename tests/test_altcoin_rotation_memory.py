"""Tests for the altcoin rotation "never go backwards" memory expiry."""

from __future__ import annotations

import logging
import unittest
from typing import Any

from trading_bot.config import AltcoinRotationConfig
from trading_bot.models import PortfolioState
from trading_bot.strategies.altcoin_rotation import AltcoinRotationStrategy


def _make_strategy(qty_memory_cycles: int = 10) -> AltcoinRotationStrategy:
    return AltcoinRotationStrategy(
        config=AltcoinRotationConfig(qty_memory_cycles=qty_memory_cycles),
        coinbase_client=None,  # type: ignore[arg-type]
        coinbase_broker=None,  # type: ignore[arg-type]
        logger=logging.getLogger("test_rotation_memory"),
    )


def _make_rd(**overrides: Any) -> dict[str, Any]:
    rd: dict[str, Any] = {
        "current_coin": None,
        "ratios": {},
        "coin_quantities": {},
        "cycle_count": 0,
        "qty_recorded_cycle": {},
    }
    rd.update(overrides)
    return rd


class QtyMemoryTests(unittest.TestCase):
    def test_remembered_qty_expires_after_limit(self) -> None:
        strategy = _make_strategy(qty_memory_cycles=10)
        rd = _make_rd()
        strategy._record_qty(rd, "XRP-USD", 100.0)

        rd["cycle_count"] = 10
        self.assertEqual(strategy._remembered_qty(rd, "XRP-USD"), 100.0)

        rd["cycle_count"] = 11
        self.assertEqual(strategy._remembered_qty(rd, "XRP-USD"), 0.0)

    def test_zero_limit_never_expires(self) -> None:
        strategy = _make_strategy(qty_memory_cycles=0)
        rd = _make_rd()
        strategy._record_qty(rd, "XRP-USD", 100.0)
        rd["cycle_count"] = 10_000

        self.assertEqual(strategy._remembered_qty(rd, "XRP-USD"), 100.0)

    def test_legacy_entry_without_stamp_is_treated_as_fresh(self) -> None:
        strategy = _make_strategy(qty_memory_cycles=10)
        rd = _make_rd(coin_quantities={"XRP-USD": 50.0}, cycle_count=100)

        self.assertEqual(strategy._remembered_qty(rd, "XRP-USD"), 50.0)

    def test_get_rotation_data_stamps_legacy_quantities(self) -> None:
        state = PortfolioState(
            cash=0.0,
            daily_baseline_equity=0.0,
            daily_baseline_date="2026-01-01",
            peak_equity=0.0,
        )
        state.rotation_data = {  # type: ignore[attr-defined]
            "coin_quantities": {"BTC-USD": 1.0},
        }

        rd = AltcoinRotationStrategy._get_rotation_data(state)

        self.assertEqual(rd["qty_recorded_cycle"]["BTC-USD"], rd["cycle_count"])

    def test_scout_swap_allows_reentry_after_memory_expiry(self) -> None:
        """A stale huge remembered quantity no longer blocks a favorable swap."""
        prices = {"BTC-USD": 100.0, "XRP-USD": 1.0}
        # Baseline ratio 95 vs current 100 → ~5.3% shift, above the
        # fee_pct(0.006) × scout_multiplier(3) = 1.8% threshold.
        base_rd = dict(
            current_coin="BTC-USD",
            ratios={"BTC-USD": {"XRP-USD": 95.0}},
            coin_quantities={"BTC-USD": 10.0, "XRP-USD": 999_999.0},
            qty_recorded_cycle={"BTC-USD": 0, "XRP-USD": 0},
        )

        # Legacy behavior (no expiry): the remembered XRP quantity blocks it.
        strategy = _make_strategy(qty_memory_cycles=0)
        rd = _make_rd(**base_rd, cycle_count=50)
        self.assertIsNone(strategy._scout_swap(rd, "BTC-USD", prices))

        # With expiry: the 50-cycle-old memory is ignored and the swap fires.
        strategy = _make_strategy(qty_memory_cycles=10)
        rd = _make_rd(**base_rd, cycle_count=50)
        result = strategy._scout_swap(rd, "BTC-USD", prices)
        self.assertIsNotNone(result)
        target, shift = result
        self.assertEqual(target, "XRP-USD")
        self.assertGreater(shift, 0.018)


if __name__ == "__main__":
    unittest.main()
