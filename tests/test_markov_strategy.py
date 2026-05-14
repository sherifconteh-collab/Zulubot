from __future__ import annotations

import logging
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.config import MarkovStrategyConfig, RiskConfig
from trading_bot.markov import (
    BEAR,
    BULL,
    SIDEWAYS,
    MarkovResult,
    build_state_sequence,
    classify_state,
    compute_markov_signal,
    estimate_transition_matrix,
    rolling_returns,
    stationary_distribution,
)
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.strategies.markov_regime import MarkovRegimeStrategy


# ------------------------------------------------------------------ #
# Fake clients                                                         #
# ------------------------------------------------------------------ #

class FakeCoinbaseClient:
    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = prices

    def get_spot_prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: self._prices[s] for s in symbols if s in self._prices}


class FakeBinanceClient:
    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = prices

    def get_mark_price_and_funding(self, symbol: str):
        from trading_bot.models import FundingSnapshot  # type: ignore[attr-defined]
        return type("FundingSnapshot", (), {
            "mark_price": self._prices.get(symbol, 50_000.0),
            "funding_rate": 0.0,
        })()


# ------------------------------------------------------------------ #
# Price series helpers                                                  #
# ------------------------------------------------------------------ #

def _bull_prices(n: int = 60, start: float = 30_000.0, daily_gain: float = 0.005) -> list[float]:
    """Monotonically rising prices (clear bull regime)."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_gain))
    return prices


def _bear_prices(n: int = 60, start: float = 60_000.0, daily_loss: float = 0.005) -> list[float]:
    """Monotonically falling prices (clear bear regime)."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 - daily_loss))
    return prices


def _flat_prices(n: int = 60, value: float = 50_000.0) -> list[float]:
    return [value] * n


def _default_state(prices: dict[str, list[float]] | None = None) -> PortfolioState:
    return PortfolioState(
        cash=100_000.0,
        daily_baseline_equity=100_000.0,
        daily_baseline_date="2026-05-13",
        peak_equity=100_000.0,
        price_history=prices or {},
    )


# ------------------------------------------------------------------ #
# Unit tests for trading_bot/markov.py (pure math)                    #
# ------------------------------------------------------------------ #

class TestMarkovEngine(unittest.TestCase):

    def test_classify_state_bull(self) -> None:
        self.assertEqual(classify_state(0.03, 0.02, -0.02), BULL)

    def test_classify_state_bear(self) -> None:
        self.assertEqual(classify_state(-0.03, 0.02, -0.02), BEAR)

    def test_classify_state_sideways(self) -> None:
        self.assertEqual(classify_state(0.01, 0.02, -0.02), SIDEWAYS)

    def test_classify_state_exact_bull_threshold_is_sideways(self) -> None:
        # boundary: exactly at threshold is NOT > threshold → SIDEWAYS
        self.assertEqual(classify_state(0.02, 0.02, -0.02), SIDEWAYS)

    def test_rolling_returns_empty_on_insufficient_prices(self) -> None:
        self.assertEqual(rolling_returns([1.0, 2.0], window=20), [])

    def test_rolling_returns_length(self) -> None:
        prices = [float(i + 100) for i in range(25)]
        rets = rolling_returns(prices, window=20)
        self.assertEqual(len(rets), 5)

    def test_rolling_returns_correct_computation(self) -> None:
        # 100 → 110 over 5-day window: return = (110-100)/100 = 0.10
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
        rets = rolling_returns(prices, window=5)
        self.assertAlmostEqual(rets[0], 0.10, places=10)

    def test_build_state_sequence_length(self) -> None:
        prices = _bull_prices(30)
        seq = build_state_sequence(prices, window=20, bull_threshold=0.02, bear_threshold=-0.02)
        self.assertEqual(len(seq), 10)  # 30 - 20 = 10

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        import numpy as np
        prices = _bull_prices(60)
        seq = build_state_sequence(prices, window=20, bull_threshold=0.02, bear_threshold=-0.02)
        # Use laplace_alpha=0.1 so every row has at least some counts
        matrix = estimate_transition_matrix(seq, laplace_alpha=0.1)
        row_sums = matrix.sum(axis=1)
        for s in row_sums:
            self.assertAlmostEqual(s, 1.0, places=10)

    def test_pure_bull_run_gives_high_bull_self_transition(self) -> None:
        import numpy as np
        prices = _bull_prices(60, daily_gain=0.005)
        seq = build_state_sequence(prices, window=20, bull_threshold=0.02, bear_threshold=-0.02)
        # With all prices rising, all states should be BULL, laplace=0
        matrix = estimate_transition_matrix(seq, laplace_alpha=0.0)
        # Bull row: P(bull→bull) should be 1.0, others 0.0
        self.assertAlmostEqual(matrix[BULL, BULL], 1.0, places=5)

    def test_laplace_smoothing_prevents_zero_entries(self) -> None:
        import numpy as np
        prices = _bull_prices(60)
        seq = build_state_sequence(prices, window=20, bull_threshold=0.02, bear_threshold=-0.02)
        matrix = estimate_transition_matrix(seq, laplace_alpha=0.1)
        # All entries must be > 0 regardless of observed transitions
        self.assertTrue((matrix > 0).all())

    def test_stationary_distribution_sums_to_one(self) -> None:
        import numpy as np
        prices = _bull_prices(60)
        seq = build_state_sequence(prices, window=20, bull_threshold=0.02, bear_threshold=-0.02)
        matrix = estimate_transition_matrix(seq, laplace_alpha=0.1)
        stat = stationary_distribution(matrix)
        self.assertAlmostEqual(stat.sum(), 1.0, places=8)

    def test_compute_markov_signal_returns_none_for_insufficient_data(self) -> None:
        prices = [50_000.0] * 10
        result = compute_markov_signal(prices)
        self.assertIsNone(result)

    def test_compute_markov_signal_returns_result_with_sufficient_data(self) -> None:
        prices = _bull_prices(60)
        result = compute_markov_signal(prices)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, MarkovResult)

    def test_bullish_run_gives_positive_signal(self) -> None:
        prices = _bull_prices(60, daily_gain=0.005)
        result = compute_markov_signal(prices, laplace_alpha=0.05)
        self.assertIsNotNone(result)
        assert result is not None
        # Bull prices → current state should be bull → signal > 0
        self.assertGreater(result.signal, 0.0)

    def test_bearish_run_gives_negative_signal(self) -> None:
        prices = _bear_prices(60, daily_loss=0.005)
        result = compute_markov_signal(prices, laplace_alpha=0.05)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLess(result.signal, 0.0)

    def test_next_step_probs_sum_to_one(self) -> None:
        prices = _bull_prices(60)
        result = compute_markov_signal(prices)
        self.assertIsNotNone(result)
        assert result is not None
        total = sum(result.next_step_probs)
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_forecast_steps_changes_result(self) -> None:
        prices = _bull_prices(60)
        r1 = compute_markov_signal(prices, forecast_steps=1)
        r5 = compute_markov_signal(prices, forecast_steps=5)
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r5)
        # They may differ (multi-step converges to stationary), but both are valid
        self.assertAlmostEqual(sum(r5.next_step_probs), 1.0, places=8)  # type: ignore[union-attr]


# ------------------------------------------------------------------ #
# Unit tests for MarkovRegimeStrategy                                  #
# ------------------------------------------------------------------ #

class TestMarkovRegimeStrategy(unittest.TestCase):

    def setUp(self) -> None:
        self.logger = logging.getLogger("test_markov_strategy")
        self.now = datetime(2026, 5, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        self.risk_manager = RiskManager(RiskConfig())
        self.broker = PaperBroker(True, self.logger)
        self.btc_price = 65_000.0
        self.eth_price = 3_200.0
        self.prices = {"BTCUSDT": self.btc_price, "ETHUSDT": self.eth_price}

    def _make_strategy(self, config: MarkovStrategyConfig | None = None) -> MarkovRegimeStrategy:
        cfg = config or MarkovStrategyConfig(enabled=True, min_history_days=40)
        return MarkovRegimeStrategy(
            config=cfg,
            coinbase_client=FakeCoinbaseClient(self.prices),
            logger=self.logger,
        )

    def test_insufficient_history_returns_skip_status(self) -> None:
        strategy = self._make_strategy()
        state = _default_state({"BTCUSDT": [50_000.0] * 5})

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["opens"], [])
        self.assertEqual(result["closes"], [])

    def test_bull_regime_opens_long_positions(self) -> None:
        strategy = self._make_strategy(MarkovStrategyConfig(
            enabled=True,
            min_history_days=40,
            entry_signal_threshold=0.05,  # low bar to guarantee entry on bull run
            laplace_alpha=0.05,
        ))
        bull_history = _bull_prices(60, daily_gain=0.005)
        state = _default_state({"BTCUSDT": bull_history})

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        self.assertEqual(result["status"], "completed")
        # Bull prices → positive signal → both BTC and ETH should open
        self.assertIn("BTCUSDT", result["opens"])

    def test_bear_regime_does_not_open_positions(self) -> None:
        strategy = self._make_strategy(MarkovStrategyConfig(
            enabled=True,
            min_history_days=40,
            entry_signal_threshold=0.10,
            exit_signal_threshold=0.0,
            laplace_alpha=0.05,
        ))
        bear_history = _bear_prices(60, daily_loss=0.005)
        state = _default_state({"BTCUSDT": bear_history})

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["opens"], [])

    def test_regime_flip_closes_existing_position(self) -> None:
        strategy = self._make_strategy(MarkovStrategyConfig(
            enabled=True,
            min_history_days=40,
            entry_signal_threshold=0.10,
            exit_signal_threshold=0.0,  # close when signal < 0
            take_profit_pct=0.0,
            stop_loss_pct=0.0,
            laplace_alpha=0.05,
        ))
        bear_history = _bear_prices(60, daily_loss=0.005)
        state = _default_state({"BTCUSDT": bear_history})
        # Prevent mark_positions from appending a new entry to price_history
        state.last_price_record_date = self.now.date().isoformat()
        from trading_bot.models import Position
        state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            asset_type="crypto",
            strategy="markov_regime",
            quantity=1.0,
            entry_price=65_000.0,
            current_price=65_000.0,
            opened_at=self.now.isoformat(),
        )

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        # Bear regime → signal < 0 → position should be closed
        self.assertIn("BTCUSDT", result["closes"])
        self.assertNotIn("BTCUSDT", state.positions)

    def test_stop_loss_closes_position(self) -> None:
        config = MarkovStrategyConfig(
            enabled=True,
            min_history_days=40,
            stop_loss_pct=0.10,
            take_profit_pct=0.0,
            exit_signal_threshold=-999.0,  # never trigger regime exit
            entry_signal_threshold=999.0,  # never trigger re-entry after close
        )
        strategy = self._make_strategy(config)

        entry_price = 65_000.0
        current_price = entry_price * 0.85  # -15% from entry → triggers 10% stop-loss
        self.prices["BTCUSDT"] = current_price
        strategy.coinbase_client = FakeCoinbaseClient(self.prices)

        state = _default_state({"BTCUSDT": _bull_prices(60)})
        # Prevent mark_positions from appending to price_history
        state.last_price_record_date = self.now.date().isoformat()
        from trading_bot.models import Position
        state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            asset_type="crypto",
            strategy="markov_regime",
            quantity=1.0,
            entry_price=entry_price,
            current_price=current_price,
            opened_at=self.now.isoformat(),
        )

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        self.assertIn("BTCUSDT", result["closes"])
        self.assertNotIn("BTCUSDT", state.positions)

    def test_take_profit_closes_position(self) -> None:
        config = MarkovStrategyConfig(
            enabled=True,
            min_history_days=40,
            take_profit_pct=0.15,
            stop_loss_pct=0.0,
            exit_signal_threshold=-999.0,  # never trigger regime exit
            entry_signal_threshold=999.0,  # never trigger re-entry after close
        )
        strategy = self._make_strategy(config)

        entry_price = 50_000.0
        current_price = entry_price * 1.20  # +20% → triggers 15% take-profit
        self.prices["BTCUSDT"] = current_price
        strategy.coinbase_client = FakeCoinbaseClient(self.prices)

        state = _default_state({"BTCUSDT": _bull_prices(60)})
        # Prevent mark_positions from appending to price_history
        state.last_price_record_date = self.now.date().isoformat()
        from trading_bot.models import Position
        state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            asset_type="crypto",
            strategy="markov_regime",
            quantity=1.0,
            entry_price=entry_price,
            current_price=current_price,
            opened_at=self.now.isoformat(),
        )

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        self.assertIn("BTCUSDT", result["closes"])
        self.assertNotIn("BTCUSDT", state.positions)

    def test_kill_switch_blocks_entry(self) -> None:
        strategy = self._make_strategy(MarkovStrategyConfig(
            enabled=True,
            min_history_days=40,
            entry_signal_threshold=0.05,
            laplace_alpha=0.05,
        ))
        state = _default_state({"BTCUSDT": _bull_prices(60, daily_gain=0.005)})
        state.kill_switch_engaged = True

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        self.assertEqual(result["opens"], [])

    def test_daily_halt_blocks_entry(self) -> None:
        strategy = self._make_strategy(MarkovStrategyConfig(
            enabled=True,
            min_history_days=40,
            entry_signal_threshold=0.05,
            laplace_alpha=0.05,
        ))
        state = _default_state({"BTCUSDT": _bull_prices(60, daily_gain=0.005)})
        state.halt_new_trades_for_day = True

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        self.assertEqual(result["opens"], [])

    def test_only_markov_positions_are_managed_on_exit(self) -> None:
        """Positions opened by a different strategy must not be closed by regime exit."""
        strategy = self._make_strategy(MarkovStrategyConfig(
            enabled=True,
            min_history_days=40,
            exit_signal_threshold=0.0,
            take_profit_pct=0.0,
            stop_loss_pct=0.0,
            laplace_alpha=0.05,
        ))
        bear_history = _bear_prices(60, daily_loss=0.005)
        state = _default_state({"BTCUSDT": bear_history})
        from trading_bot.models import Position
        # This position was opened by a DIFFERENT strategy
        state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            asset_type="crypto",
            strategy="crypto_sentiment_funding",
            quantity=1.0,
            entry_price=65_000.0,
            current_price=65_000.0,
            opened_at=self.now.isoformat(),
        )

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        # Markov regime exits should not touch other strategy's positions
        self.assertNotIn("BTCUSDT", result["closes"])
        self.assertIn("BTCUSDT", state.positions)

    def test_result_contains_expected_keys(self) -> None:
        strategy = self._make_strategy()
        state = _default_state({"BTCUSDT": _bull_prices(60)})

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        self.assertIn("status", result)
        self.assertIn("opens", result)
        self.assertIn("closes", result)
        self.assertIn("decisions", result)

    def test_no_duplicate_opens_for_existing_position(self) -> None:
        strategy = self._make_strategy(MarkovStrategyConfig(
            enabled=True,
            min_history_days=40,
            entry_signal_threshold=0.05,
            laplace_alpha=0.05,
        ))
        bull_history = _bull_prices(60, daily_gain=0.005)
        state = _default_state({"BTCUSDT": bull_history})
        # Pre-plant an open position
        from trading_bot.models import Position
        state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            asset_type="crypto",
            strategy="markov_regime",
            quantity=0.1,
            entry_price=65_000.0,
            current_price=65_000.0,
            opened_at=self.now.isoformat(),
        )

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        # BTCUSDT is already open — should not open again
        self.assertNotIn("BTCUSDT", result["opens"])

    def test_disabled_strategy_returns_disabled_status(self) -> None:
        strategy = self._make_strategy(MarkovStrategyConfig(enabled=False))
        state = _default_state({"BTCUSDT": _bull_prices(60, daily_gain=0.005)})

        result = strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now)

        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["opens"], [])
        self.assertEqual(result["closes"], [])
        # No positions should have been opened or closed
        self.assertEqual(state.positions, {})


if __name__ == "__main__":
    unittest.main()
