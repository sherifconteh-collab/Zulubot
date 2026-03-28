"""Tests for TradingAgents wrapper service with mocked LLM calls."""

from __future__ import annotations

import logging
import time
import unittest
from unittest.mock import MagicMock, patch

from trading_bot.config import TradingAgentsConfig
from trading_bot.integrations.trading_agents import (
    AgentVerdict,
    TradingAgentsService,
    apply_ai_confirmation,
)


class FakeAnthropicMessage:
    """Fake Anthropic message response."""
    def __init__(self, text: str) -> None:
        self.content = [MagicMock(text=text)]


class FakeAnthropicClient:
    """Fake Anthropic client that returns predetermined responses."""
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ['{"action": "buy", "confidence": 0.8, "reasoning": "looks good"}']
        self._call_idx = 0
        self.messages = self

    def create(self, **kwargs) -> FakeAnthropicMessage:
        text = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return FakeAnthropicMessage(text)


class TradingAgentsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_trading_agents")
        self.config = TradingAgentsConfig(
            enabled=True,
            api_key="test-key",
            quick_model="claude-haiku-4-5-20251001",
            deep_model="claude-sonnet-4-6",
            cache_ttl_hours=4,
            max_cost_per_run=0.50,
            max_cost_per_month=20.00,
        )

    def _make_service(self, responses: list[str] | None = None) -> TradingAgentsService:
        """Create service with mocked Anthropic client."""
        service = TradingAgentsService(
            TradingAgentsConfig(enabled=False),  # Skip real init
            self.logger,
        )
        # Manually set up for testing
        service.config = self.config
        service._anthropic_client = FakeAnthropicClient(responses)
        return service

    def test_evaluate_stock_returns_verdict(self) -> None:
        service = self._make_service([
            "yes",  # Haiku screen
            '{"action": "buy", "confidence": 0.85, "reasoning": "Strong value metrics"}',
        ])

        verdict = service.evaluate_stock(
            symbol="AAPL",
            fundamentals={"pe_ratio": 15, "pb_ratio": 2.0, "debt_to_equity": 0.3},
            vix_value=28.0,
        )

        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.action, "buy")
        self.assertGreater(verdict.confidence, 0.5)
        self.assertIn("Strong value", verdict.reasoning)

    def test_haiku_screen_rejects_skips_sonnet(self) -> None:
        service = self._make_service([
            "no",  # Haiku screen says don't investigate
        ])

        verdict = service.evaluate_stock(
            symbol="XYZ",
            fundamentals={"pe_ratio": 18, "pb_ratio": 2.5},
            vix_value=22.0,
        )

        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.action, "skip")
        self.assertEqual(verdict.model_used, "claude-haiku-4-5-20251001")

    def test_cache_returns_cached_verdict(self) -> None:
        service = self._make_service([
            "yes",
            '{"action": "hold", "confidence": 0.6, "reasoning": "Wait and see"}',
        ])

        # First call — hits LLM
        v1 = service.evaluate_crypto("BTCUSDT", 20, -0.001, 65000.0)
        self.assertIsNotNone(v1)
        self.assertFalse(v1.cached)

        # Second call — should return cached
        v2 = service.evaluate_crypto("BTCUSDT", 20, -0.001, 65000.0)
        self.assertIsNotNone(v2)
        self.assertTrue(v2.cached)
        self.assertEqual(v2.action, v1.action)
        self.assertEqual(v2.cost_usd, 0.0)

    def test_monthly_budget_cap(self) -> None:
        config = TradingAgentsConfig(
            enabled=True,
            api_key="test-key",
            max_cost_per_month=0.001,  # Extremely low cap (below Haiku cost)
        )
        service = self._make_service()
        service.config = config
        service._month_cost = 0.001  # Already at cap

        verdict = service.evaluate_stock(
            symbol="TEST",
            fundamentals={"pe_ratio": 10},
            vix_value=30.0,
        )

        # Should return None when budget exceeded (even Haiku can't afford)
        self.assertIsNone(verdict)

    def test_disabled_returns_none(self) -> None:
        service = TradingAgentsService(
            TradingAgentsConfig(enabled=False),
            self.logger,
        )

        verdict = service.evaluate_stock("AAPL", {}, 25.0)
        self.assertIsNone(verdict)

    def test_polymarket_bypasses_haiku_screen(self) -> None:
        service = self._make_service([
            # Only one response — Sonnet directly (no Haiku pre-screen)
            '{"action": "buy", "confidence": 0.7, "reasoning": "Underpriced", "estimated_probability": 0.45}',
        ])

        verdict = service.evaluate_polymarket(
            market_question="Will BTC hit $100k by Dec?",
            market_description="Bitcoin price prediction",
            yes_price=0.25,
            tags=["crypto"],
            fear_greed_value=20,
        )

        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.action, "buy")
        self.assertIsNotNone(verdict.estimated_probability)
        self.assertAlmostEqual(verdict.estimated_probability, 0.45, places=2)


class ApplyAIConfirmationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_confirmation")

    def test_log_only_does_not_change_notional(self) -> None:
        verdict = AgentVerdict(action="sell", confidence=0.9, reasoning="Overvalued")
        notional, meta = apply_ai_confirmation(verdict, "log_only", 2000.0, self.logger)

        self.assertEqual(notional, 2000.0)
        self.assertIn("ai_verdict", meta)

    def test_boost_increases_on_buy(self) -> None:
        verdict = AgentVerdict(action="buy", confidence=0.8, reasoning="Good buy")
        notional, meta = apply_ai_confirmation(verdict, "boost", 2000.0, self.logger)

        self.assertEqual(notional, 3000.0)  # 1.5x

    def test_boost_decreases_on_sell(self) -> None:
        verdict = AgentVerdict(action="sell", confidence=0.9, reasoning="Bad signal")
        notional, meta = apply_ai_confirmation(verdict, "boost", 2000.0, self.logger)

        self.assertEqual(notional, 1000.0)  # 0.5x

    def test_require_blocks_on_sell(self) -> None:
        verdict = AgentVerdict(action="sell", confidence=0.9, reasoning="Don't buy")
        notional, meta = apply_ai_confirmation(verdict, "require", 2000.0, self.logger)

        self.assertEqual(notional, 0.0)  # Blocked

    def test_require_allows_on_buy(self) -> None:
        verdict = AgentVerdict(action="buy", confidence=0.7, reasoning="Good signal")
        notional, meta = apply_ai_confirmation(verdict, "require", 2000.0, self.logger)

        self.assertEqual(notional, 2000.0)  # Allowed

    def test_none_verdict_passes_through(self) -> None:
        notional, meta = apply_ai_confirmation(None, "require", 2000.0, self.logger)

        self.assertEqual(notional, 2000.0)
        self.assertEqual(meta, {})


if __name__ == "__main__":
    unittest.main()
