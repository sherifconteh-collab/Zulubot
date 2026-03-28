from __future__ import annotations

import logging
import unittest
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.config import PolymarketStrategyConfig, RiskConfig
from trading_bot.models import PolymarketMarket, PortfolioState
from trading_bot.polymarket_broker import PolymarketBroker
from trading_bot.risk import RiskManager
from trading_bot.strategies.polymarket import PolymarketStrategy


def _make_market(
    condition_id: str = "abc12345",
    question: str = "Will BTC reach $100k?",
    yes_price: float = 0.25,
    no_price: float = 0.75,
    tags: list[str] | None = None,
    volume_24h: float = 200_000.0,
    liquidity: float = 50_000.0,
    resolved: bool = False,
    outcome: str | None = None,
) -> PolymarketMarket:
    return PolymarketMarket(
        condition_id=condition_id,
        question=question,
        description="Test market",
        yes_token_id=f"yes_token_{condition_id}",
        no_token_id=f"no_token_{condition_id}",
        yes_price=yes_price,
        no_price=no_price,
        volume_24h=volume_24h,
        liquidity=liquidity,
        tags=tags or ["crypto"],
        end_date="2026-12-31",
        active=True,
        resolved=resolved,
        outcome=outcome,
    )


class FakeGammaClient:
    """Fake Gamma API client for testing."""

    def __init__(self, markets: list[PolymarketMarket] | None = None) -> None:
        self.markets = markets or []
        self._market_map: dict[str, PolymarketMarket] = {
            m.condition_id: m for m in self.markets
        }

    def discover_markets(self, **kwargs: Any) -> list[PolymarketMarket]:
        return self.markets

    def get_market(self, condition_id: str) -> PolymarketMarket | None:
        return self._market_map.get(condition_id)


class FakeClobClient:
    """Fake CLOB API client for testing."""

    def __init__(self, midpoints: dict[str, float] | None = None) -> None:
        self.midpoints = midpoints or {}

    def get_midpoint(self, token_id: str) -> float:
        return self.midpoints.get(token_id, 0.0)

    def get_midpoints(self, token_ids: list[str]) -> dict[str, float]:
        return {tid: self.midpoints.get(tid, 0.0) for tid in token_ids}


class PolymarketStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_polymarket")
        self.now = datetime(2026, 3, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        self.config = PolymarketStrategyConfig(
            enabled=True,
            buy_yes_threshold=0.30,
            buy_no_threshold=0.70,
            take_profit_threshold=0.50,
            stop_loss_threshold=0.15,
            max_positions=3,
        )
        self.paper_broker = PaperBroker(True, self.logger)
        self.risk_manager = RiskManager(RiskConfig())

    def _make_strategy(
        self,
        markets: list[PolymarketMarket] | None = None,
        midpoints: dict[str, float] | None = None,
    ) -> PolymarketStrategy:
        gamma = FakeGammaClient(markets or [])
        clob = FakeClobClient(midpoints or {})
        pm_broker = PolymarketBroker(self.paper_broker, True, self.logger)
        return PolymarketStrategy(
            config=self.config,
            gamma_client=gamma,
            clob_client=clob,
            polymarket_broker=pm_broker,
            logger=self.logger,
        )

    def test_buys_cheap_yes_on_non_crypto_market(self) -> None:
        """Should buy YES when price < 0.30 on non-crypto market (no F&G needed)."""
        market = _make_market(
            condition_id="fin_0001",
            question="Will the Fed cut rates?",
            yes_price=0.20,
            no_price=0.80,
            tags=["finance"],
        )
        strategy = self._make_strategy(
            markets=[market],
            midpoints={"yes_token_fin_0001": 0.20},
        )
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        result = strategy.run(
            state,
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            now=self.now,
            fear_greed_value=None,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["buys"]), 1)
        self.assertIn("PM:fin_0001:YES", result["buys"])
        self.assertIn("PM:fin_0001:YES", state.positions)

    def test_buys_no_on_expensive_non_crypto_market(self) -> None:
        """Should buy NO when YES price > 0.70 on non-crypto market."""
        market = _make_market(
            condition_id="fin_0002",
            question="Will inflation exceed 5%?",
            yes_price=0.85,
            no_price=0.15,
            tags=["finance"],
        )
        strategy = self._make_strategy(
            markets=[market],
            midpoints={"yes_token_fin_0002": 0.85},
        )
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        result = strategy.run(
            state,
            broker=self.paper_broker,
            risk_manager=self.risk_manager,
            now=self.now,
        )

        self.assertEqual(len(result["buys"]), 1)
        self.assertIn("PM:fin_0002:NO", result["buys"])

    def test_crypto_market_requires_fear_greed_alignment(self) -> None:
        """Crypto-tagged markets need F&G confirmation. High fear = buy YES."""
        market = _make_market(
            condition_id="cry_0001",
            question="Will BTC reach $100k?",
            yes_price=0.20,
            tags=["crypto"],
        )
        strategy = self._make_strategy(
            markets=[market],
            midpoints={"yes_token_cry_0001": 0.20},
        )

        # Without fear (F&G = 50): no buy
        state1 = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )
        result1 = strategy.run(
            state1, broker=self.paper_broker, risk_manager=self.risk_manager,
            now=self.now, fear_greed_value=50,
        )
        self.assertEqual(len(result1["buys"]), 0)

        # With high fear (F&G = 15): buy YES
        state2 = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )
        strategy2 = self._make_strategy(
            markets=[market],
            midpoints={"yes_token_cry_0001": 0.20},
        )
        result2 = strategy2.run(
            state2, broker=self.paper_broker, risk_manager=self.risk_manager,
            now=self.now, fear_greed_value=15,
        )
        self.assertEqual(len(result2["buys"]), 1)
        self.assertIn("PM:cry_0001:YES", result2["buys"])

    def test_respects_max_positions(self) -> None:
        """Should not open more positions than max_positions."""
        config = PolymarketStrategyConfig(enabled=True, max_positions=1)
        markets = [
            _make_market(condition_id="m1", yes_price=0.20, tags=["finance"]),
            _make_market(condition_id="m2", yes_price=0.15, tags=["finance"]),
        ]
        gamma = FakeGammaClient(markets)
        clob = FakeClobClient({"yes_token_m1": 0.20, "yes_token_m2": 0.15})
        pm_broker = PolymarketBroker(self.paper_broker, True, self.logger)
        strategy = PolymarketStrategy(
            config=config,
            gamma_client=gamma,
            clob_client=clob,
            polymarket_broker=pm_broker,
            logger=self.logger,
        )
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        result = strategy.run(
            state, broker=self.paper_broker, risk_manager=self.risk_manager, now=self.now,
        )

        self.assertEqual(len(result["buys"]), 1)
        self.assertEqual(len(state.positions), 1)

    def test_disabled_returns_immediately(self) -> None:
        """When disabled, strategy returns without scanning."""
        config = PolymarketStrategyConfig(enabled=False)
        gamma = FakeGammaClient([])
        clob = FakeClobClient({})
        pm_broker = PolymarketBroker(self.paper_broker, True, self.logger)
        strategy = PolymarketStrategy(
            config=config, gamma_client=gamma, clob_client=clob,
            polymarket_broker=pm_broker, logger=self.logger,
        )
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        result = strategy.run(
            state, broker=self.paper_broker, risk_manager=self.risk_manager, now=self.now,
        )

        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["markets_scanned"], 0)

    def test_stop_loss_closes_position(self) -> None:
        """Position should close when price drops beyond stop_loss threshold."""
        market = _make_market(condition_id="sl_test", yes_price=0.25, tags=["finance"])
        midpoints = {"yes_token_sl_test": 0.25}
        strategy = self._make_strategy(markets=[], midpoints={"yes_token_sl_test": 0.08})

        state = PortfolioState(
            cash=98_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )
        # Manually open a position at 0.25
        self.paper_broker.open_position(
            state,
            symbol="PM:sl_test:YES",
            asset_type="prediction",
            strategy="polymarket_threshold",
            price=0.25,
            notional=2_000.0,
            opened_at=self.now,
            meta={"condition_id": "sl_test", "token_id": "yes_token_sl_test", "side": "YES"},
        )
        self.assertIn("PM:sl_test:YES", state.positions)

        # Run strategy — CLOB returns 0.08 (drop of 0.17 > stop_loss 0.15)
        result = strategy.run(
            state, broker=self.paper_broker, risk_manager=self.risk_manager, now=self.now,
        )

        self.assertIn("PM:sl_test:YES", result["sells"])
        self.assertNotIn("PM:sl_test:YES", state.positions)


if __name__ == "__main__":
    unittest.main()
