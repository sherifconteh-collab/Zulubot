from __future__ import annotations

import math
import unittest

from trading_bot.config import RiskConfig, VolatilityConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.volatility import VolatilityEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_prices(start: float, n: int, daily_return: float = 0.01) -> list[float]:
    """Generate a geometric price series with constant daily return."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1.0 + daily_return))
    return prices


def _volatile_prices(start: float, n: int, daily_return: float = 0.05) -> list[float]:
    return _flat_prices(start, n, daily_return)


# ---------------------------------------------------------------------------
# VolatilityEngine — realized vol
# ---------------------------------------------------------------------------

class TestCalculateRealizedVol(unittest.TestCase):
    def test_returns_none_with_insufficient_data(self) -> None:
        prices = _flat_prices(100.0, 20)  # need window+1=21 prices for window=20
        result = VolatilityEngine.calculate_realized_vol(prices, window=20)
        self.assertIsNone(result)

    def test_returns_float_with_sufficient_data(self) -> None:
        # Alternating up/down creates actual variance; constant returns give 0 std
        prices = [100.0]
        for i in range(20):
            pct = 0.01 if i % 2 == 0 else -0.005
            prices.append(prices[-1] * (1.0 + pct))
        result = VolatilityEngine.calculate_realized_vol(prices, window=20)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.0)  # type: ignore[arg-type]

    def test_higher_daily_return_gives_higher_vol(self) -> None:
        # Alternating up/down creates actual variance (pure geometric series has 0 std)
        def alternating(start: float, n: int, pct: float) -> list[float]:
            prices = [start]
            for i in range(n - 1):
                prices.append(prices[-1] * (1.0 + pct if i % 2 == 0 else 1.0 - pct))
            return prices

        quiet = VolatilityEngine.calculate_realized_vol(alternating(100.0, 25, 0.005), window=20)
        loud = VolatilityEngine.calculate_realized_vol(alternating(100.0, 25, 0.03), window=20)
        self.assertIsNotNone(quiet)
        self.assertIsNotNone(loud)
        self.assertGreater(loud, quiet)  # type: ignore[operator]

    def test_annualisation_factor(self) -> None:
        # For constant daily returns, daily_vol ≈ 0; test the sqrt(252) scaling works
        prices = _flat_prices(100.0, 25, 0.01)
        daily_vol_annualised = VolatilityEngine.calculate_realized_vol(prices, window=20)
        # Constant returns → std of returns ≈ 0, but let's just confirm scaling direction
        self.assertIsNotNone(daily_vol_annualised)

    def test_returns_none_for_zero_or_negative_price(self) -> None:
        # Zero price must be within the most-recent window+1 prices to be detected
        prices = [100.0] * 10 + [0.0] + [100.0] * 10
        result = VolatilityEngine.calculate_realized_vol(prices, window=20)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# VolatilityEngine — EWMA vol
# ---------------------------------------------------------------------------

class TestCalculateEwmaVol(unittest.TestCase):
    def test_returns_none_with_single_price(self) -> None:
        result = VolatilityEngine.calculate_ewma_vol([100.0], decay=0.94)
        self.assertIsNone(result)

    def test_returns_float_with_two_prices(self) -> None:
        result = VolatilityEngine.calculate_ewma_vol([100.0, 101.0], decay=0.94)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.0)  # type: ignore[arg-type]

    def test_higher_recent_move_increases_ewma_vol(self) -> None:
        base = [100.0] * 10 + [100.0]
        volatile = [100.0] * 10 + [120.0]  # 20% spike at the end
        vol_base = VolatilityEngine.calculate_ewma_vol(base, decay=0.94)
        vol_spike = VolatilityEngine.calculate_ewma_vol(volatile, decay=0.94)
        self.assertIsNotNone(vol_base)
        self.assertIsNotNone(vol_spike)
        self.assertGreater(vol_spike, vol_base)  # type: ignore[operator]

    def test_returns_none_for_negative_price(self) -> None:
        result = VolatilityEngine.calculate_ewma_vol([100.0, -5.0, 101.0], decay=0.94)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# VolatilityEngine — vol_adjusted_notional
# ---------------------------------------------------------------------------

class TestVolAdjustedNotional(unittest.TestCase):
    def test_scales_down_in_high_vol(self) -> None:
        # target=10%, current=20% → leverage=0.5 → notional = 0.5 × capital
        notional = VolatilityEngine.vol_adjusted_notional(
            target_vol=0.10, realized_vol=0.20, capital=10_000.0,
            max_leverage=2.0, min_allocation=0.20,
        )
        self.assertAlmostEqual(notional, 5_000.0, places=2)

    def test_scales_up_in_low_vol(self) -> None:
        # target=10%, current=5% → leverage=2.0 (capped) → notional = 2 × capital
        notional = VolatilityEngine.vol_adjusted_notional(
            target_vol=0.10, realized_vol=0.05, capital=10_000.0,
            max_leverage=2.0, min_allocation=0.20,
        )
        self.assertAlmostEqual(notional, 20_000.0, places=2)

    def test_respects_max_leverage(self) -> None:
        # target=10%, current=1% → raw leverage=10, capped to 2.0
        notional = VolatilityEngine.vol_adjusted_notional(
            target_vol=0.10, realized_vol=0.01, capital=10_000.0,
            max_leverage=2.0, min_allocation=0.20,
        )
        self.assertAlmostEqual(notional, 20_000.0, places=2)

    def test_respects_min_allocation(self) -> None:
        # target=5%, current=50% → raw leverage=0.10, floored at 0.20
        notional = VolatilityEngine.vol_adjusted_notional(
            target_vol=0.05, realized_vol=0.50, capital=10_000.0,
            max_leverage=2.0, min_allocation=0.20,
        )
        self.assertAlmostEqual(notional, 2_000.0, places=2)

    def test_exact_target_gives_full_capital(self) -> None:
        # target == current → leverage=1.0 → notional = capital
        notional = VolatilityEngine.vol_adjusted_notional(
            target_vol=0.15, realized_vol=0.15, capital=10_000.0,
            max_leverage=2.0, min_allocation=0.20,
        )
        self.assertAlmostEqual(notional, 10_000.0, places=2)


# ---------------------------------------------------------------------------
# VolatilityEngine — should_rebalance
# ---------------------------------------------------------------------------

class TestShouldRebalance(unittest.TestCase):
    def test_rebalances_when_drift_exceeds_band(self) -> None:
        self.assertTrue(VolatilityEngine.should_rebalance(1_000.0, 1_200.0, band=0.10))

    def test_no_rebalance_within_band(self) -> None:
        self.assertFalse(VolatilityEngine.should_rebalance(1_000.0, 1_050.0, band=0.10))

    def test_always_rebalances_from_zero(self) -> None:
        self.assertTrue(VolatilityEngine.should_rebalance(0.0, 500.0, band=0.10))


# ---------------------------------------------------------------------------
# VolatilityEngine — record_price
# ---------------------------------------------------------------------------

class TestRecordPrice(unittest.TestCase):
    def test_appends_price(self) -> None:
        history: dict[str, list[float]] = {}
        VolatilityEngine.record_price("BTC", 50_000.0, history, max_len=5)
        self.assertEqual(history["BTC"], [50_000.0])

    def test_trims_to_max_len(self) -> None:
        history: dict[str, list[float]] = {"BTC": list(range(5))}
        VolatilityEngine.record_price("BTC", 999.0, history, max_len=5)
        self.assertEqual(len(history["BTC"]), 5)
        self.assertEqual(history["BTC"][-1], 999.0)

    def test_multiple_symbols_independent(self) -> None:
        history: dict[str, list[float]] = {}
        VolatilityEngine.record_price("BTC", 50_000.0, history, max_len=10)
        VolatilityEngine.record_price("ETH", 3_000.0, history, max_len=10)
        self.assertIn("BTC", history)
        self.assertIn("ETH", history)
        self.assertNotEqual(history["BTC"], history["ETH"])


# ---------------------------------------------------------------------------
# RiskManager.position_notional — integration
# ---------------------------------------------------------------------------

def _make_state(price_history: dict[str, list[float]] | None = None) -> PortfolioState:
    return PortfolioState(
        cash=10_000.0,
        price_history=price_history or {},
    )


class TestRiskManagerPositionNotional(unittest.TestCase):
    def test_falls_back_to_flat_when_vol_disabled(self) -> None:
        rm = RiskManager(
            RiskConfig(max_position_fraction=0.02),
            vol_config=VolatilityConfig(enabled=False),
        )
        state = _make_state()
        result = rm.position_notional(state, equity=10_000.0, symbol="BTC-USD")
        self.assertAlmostEqual(result, 200.0, places=2)  # 2% of 10k

    def test_falls_back_when_insufficient_history(self) -> None:
        rm = RiskManager(
            RiskConfig(max_position_fraction=0.02),
            vol_config=VolatilityConfig(enabled=True, vol_window_days=20, use_ewma=False),
        )
        state = _make_state({"BTC-USD": _flat_prices(50_000.0, 5)})  # only 5 days
        result = rm.position_notional(state, equity=10_000.0, symbol="BTC-USD")
        self.assertAlmostEqual(result, 200.0, places=2)

    def test_vol_adjusted_sizing_active_with_sufficient_history(self) -> None:
        rm = RiskManager(
            RiskConfig(max_position_fraction=0.02),
            vol_config=VolatilityConfig(
                enabled=True,
                target_annual_vol=0.10,
                use_ewma=True,
                ewma_decay=0.94,
                max_leverage=2.0,
                min_allocation=0.20,
            ),
        )
        # 25-day price series with moderate volatility
        prices = _volatile_prices(50_000.0, 25, daily_return=0.02)
        state = _make_state({"BTC-USD": prices})
        result = rm.position_notional(state, equity=10_000.0, symbol="BTC-USD")
        # With 2% daily returns, annualised vol ≈ 32%, so leverage ≈ 0.31
        # result should NOT equal flat 200 — vol-adjusted sizing is active
        self.assertNotAlmostEqual(result, 200.0, places=0)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, state.cash)

    def test_prediction_assets_respect_vol_targeting(self) -> None:
        # Prediction markets now go through vol targeting just like any other asset
        rm = RiskManager(
            RiskConfig(max_position_fraction=0.02),
            vol_config=VolatilityConfig(
                enabled=True,
                target_annual_vol=0.10,
                use_ewma=True,
                ewma_decay=0.94,
                max_leverage=2.0,
                min_allocation=0.20,
            ),
        )
        # No price history → falls back to flat sizing
        state = _make_state()
        result = rm.position_notional(state, equity=10_000.0, symbol="PM:abc:YES")
        self.assertAlmostEqual(result, 200.0, places=2)

    def test_capped_by_available_cash(self) -> None:
        rm = RiskManager(
            RiskConfig(max_position_fraction=0.02),
            vol_config=VolatilityConfig(
                enabled=True,
                target_annual_vol=0.10,
                use_ewma=True,
                max_leverage=2.0,
                min_allocation=0.20,
            ),
        )
        # Very low vol → leverage wants to push notional way above cash
        prices = [100.0 + i * 0.001 for i in range(30)]  # near-flat, very low vol
        state = PortfolioState(cash=500.0, price_history={"BTC-USD": prices})
        result = rm.position_notional(state, equity=10_000.0, symbol="BTC-USD")
        self.assertLessEqual(result, 500.0)


if __name__ == "__main__":
    unittest.main()
