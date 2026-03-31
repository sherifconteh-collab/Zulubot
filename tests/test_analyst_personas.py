"""Tests for multi-analyst persona system and signal aggregation."""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock

from trading_bot.config import TradingAgentsConfig
from trading_bot.integrations.analyst_personas import (
    ANALYST_REGISTRY,
    AnalystPersona,
    AnalystSignal,
    aggregate_analyst_signals,
    get_analyst,
    get_available_analysts,
    get_default_analysts,
)
from trading_bot.integrations.trading_agents import (
    AgentVerdict,
    TradingAgentsService,
)


# ── Registry tests ─────────────────────────────────────────────────

class AnalystRegistryTests(unittest.TestCase):
    def test_registry_has_six_personas(self) -> None:
        self.assertEqual(len(ANALYST_REGISTRY), 6)

    def test_all_personas_have_required_fields(self) -> None:
        for key, persona in ANALYST_REGISTRY.items():
            self.assertEqual(persona.key, key)
            self.assertIsInstance(persona.display_name, str)
            self.assertTrue(len(persona.display_name) > 0)
            self.assertIsInstance(persona.description, str)
            self.assertIsInstance(persona.investing_style, str)
            self.assertIsInstance(persona.system_prompt, str)
            self.assertTrue(len(persona.system_prompt) > 20)
            self.assertIsInstance(persona.order, int)

    def test_get_analyst_returns_persona(self) -> None:
        persona = get_analyst("value_investor")
        self.assertIsNotNone(persona)
        self.assertEqual(persona.key, "value_investor")
        self.assertEqual(persona.display_name, "Value Investor")

    def test_get_analyst_returns_none_for_unknown(self) -> None:
        self.assertIsNone(get_analyst("nonexistent_analyst"))

    def test_get_available_analysts_returns_sorted_list(self) -> None:
        analysts = get_available_analysts()
        self.assertEqual(len(analysts), 6)
        orders = [a["order"] for a in analysts]
        self.assertEqual(orders, sorted(orders))

    def test_get_default_analysts_returns_three(self) -> None:
        defaults = get_default_analysts()
        self.assertEqual(len(defaults), 3)
        for key in defaults:
            self.assertIn(key, ANALYST_REGISTRY)

    def test_expected_personas_exist(self) -> None:
        expected = [
            "value_investor", "growth_investor", "contrarian",
            "technical_analyst", "sentiment_analyst", "risk_analyst",
        ]
        for key in expected:
            self.assertIn(key, ANALYST_REGISTRY, f"Missing persona: {key}")


# ── Signal model tests ─────────────────────────────────────────────

class AnalystSignalTests(unittest.TestCase):
    def test_signal_to_dict(self) -> None:
        signal = AnalystSignal(
            persona_key="value_investor",
            signal="bullish",
            confidence=0.85,
            reasoning="Low P/E with strong FCF",
        )
        d = signal.to_dict()
        self.assertEqual(d["persona_key"], "value_investor")
        self.assertEqual(d["signal"], "bullish")
        self.assertAlmostEqual(d["confidence"], 0.85)
        self.assertIn("Low P/E", d["reasoning"])

    def test_signal_is_frozen(self) -> None:
        signal = AnalystSignal("test", "neutral", 0.5, "reason")
        with self.assertRaises(AttributeError):
            signal.signal = "bullish"  # type: ignore[misc]


# ── Aggregation tests ──────────────────────────────────────────────

class AggregateSignalsTests(unittest.TestCase):
    def test_empty_signals_returns_hold(self) -> None:
        result = aggregate_analyst_signals([])
        self.assertEqual(result["action"], "hold")
        self.assertEqual(result["confidence"], 0.0)

    def test_unanimous_bullish(self) -> None:
        signals = [
            AnalystSignal("a", "bullish", 0.9, "Great value"),
            AnalystSignal("b", "bullish", 0.8, "Strong momentum"),
            AnalystSignal("c", "bullish", 0.7, "Sentiment positive"),
        ]
        result = aggregate_analyst_signals(signals)
        self.assertEqual(result["action"], "buy")
        self.assertGreater(result["confidence"], 0.5)

    def test_unanimous_bearish(self) -> None:
        signals = [
            AnalystSignal("a", "bearish", 0.9, "Overvalued"),
            AnalystSignal("b", "bearish", 0.8, "Downtrend"),
            AnalystSignal("c", "bearish", 0.7, "Fear high"),
        ]
        result = aggregate_analyst_signals(signals)
        self.assertEqual(result["action"], "sell")
        self.assertGreater(result["confidence"], 0.5)

    def test_mixed_signals_trend_bullish(self) -> None:
        signals = [
            AnalystSignal("a", "bullish", 0.8, "Buy signal"),
            AnalystSignal("b", "bearish", 0.4, "Mild concern"),
            AnalystSignal("c", "bullish", 0.7, "Momentum up"),
        ]
        result = aggregate_analyst_signals(signals)
        self.assertEqual(result["action"], "buy")

    def test_evenly_split_returns_hold(self) -> None:
        signals = [
            AnalystSignal("a", "bullish", 0.8, "Bull case"),
            AnalystSignal("b", "bearish", 0.8, "Bear case"),
        ]
        result = aggregate_analyst_signals(signals)
        self.assertEqual(result["action"], "hold")

    def test_all_neutral_returns_hold(self) -> None:
        signals = [
            AnalystSignal("a", "neutral", 0.5, "No opinion"),
            AnalystSignal("b", "neutral", 0.6, "Wait"),
        ]
        result = aggregate_analyst_signals(signals)
        self.assertEqual(result["action"], "hold")

    def test_confidence_weighting(self) -> None:
        """High-confidence bear outweighs low-confidence bull."""
        signals = [
            AnalystSignal("a", "bullish", 0.3, "Weak buy"),
            AnalystSignal("b", "bearish", 0.95, "Strong sell"),
        ]
        result = aggregate_analyst_signals(signals)
        self.assertEqual(result["action"], "sell")

    def test_reasoning_includes_all_analysts(self) -> None:
        signals = [
            AnalystSignal("value_investor", "bullish", 0.8, "r1"),
            AnalystSignal("contrarian", "bearish", 0.6, "r2"),
        ]
        result = aggregate_analyst_signals(signals)
        self.assertIn("value_investor", result["reasoning"])
        self.assertIn("contrarian", result["reasoning"])
        self.assertIn("1 bullish", result["reasoning"])
        self.assertIn("1 bearish", result["reasoning"])

    def test_analyst_signals_in_output(self) -> None:
        signals = [
            AnalystSignal("a", "bullish", 0.7, "Reason A"),
        ]
        result = aggregate_analyst_signals(signals)
        self.assertEqual(len(result["analyst_signals"]), 1)
        self.assertEqual(result["analyst_signals"][0]["persona_key"], "a")

    def test_confidence_clamped_to_one(self) -> None:
        signals = [
            AnalystSignal("a", "bullish", 1.0, "Max bull"),
            AnalystSignal("b", "bullish", 1.0, "Max bull"),
        ]
        result = aggregate_analyst_signals(signals)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_single_signal(self) -> None:
        signals = [AnalystSignal("solo", "bearish", 0.9, "Only analyst")]
        result = aggregate_analyst_signals(signals)
        self.assertEqual(result["action"], "sell")
        self.assertGreater(result["confidence"], 0.5)

    def test_thresholds_can_be_overridden(self) -> None:
        signals = [
            AnalystSignal("a", "bullish", 0.3, "Weak buy"),
            AnalystSignal("b", "neutral", 1.0, "Wait"),
        ]

        result = aggregate_analyst_signals(signals, buy_threshold=0.25, sell_threshold=-0.25)

        self.assertEqual(result["action"], "hold")


# ── Multi-analyst evaluation integration tests ─────────────────────

class FakeAnthropicMessage:
    """Fake Anthropic message response."""
    def __init__(self, text: str) -> None:
        self.content = [MagicMock(text=text)]


class FakeAnthropicClient:
    """Fake Anthropic client that returns persona-specific responses."""
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or [
            '{"signal": "bullish", "confidence": 0.8, "reasoning": "looks good"}',
        ]
        self._call_idx = 0
        self.messages = self

    def create(self, **kwargs) -> FakeAnthropicMessage:
        text = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return FakeAnthropicMessage(text)


class MultiAnalystEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_multi_analyst")
        self.config = TradingAgentsConfig(
            enabled=True,
            api_key="test-key",
            quick_model="claude-haiku-4-5-20251001",
            deep_model="claude-sonnet-4-6",
            cache_ttl_hours=4,
            max_cost_per_run=0.50,
            max_cost_per_month=20.00,
            multi_analyst_enabled=True,
            selected_analysts=("value_investor", "contrarian", "sentiment_analyst"),
        )

    def _make_service(self, responses: list[str] | None = None) -> TradingAgentsService:
        service = TradingAgentsService(
            TradingAgentsConfig(enabled=False), self.logger,
        )
        service.config = self.config
        service._anthropic_client = FakeAnthropicClient(responses)
        return service

    def test_multi_analyst_stock_returns_aggregated_verdict(self) -> None:
        service = self._make_service([
            '{"signal": "bullish", "confidence": 0.85, "reasoning": "Strong value"}',
            '{"signal": "bearish", "confidence": 0.5, "reasoning": "Overvalued"}',
            '{"signal": "bullish", "confidence": 0.7, "reasoning": "Fear extreme"}',
        ])

        verdict = service.evaluate_stock(
            symbol="AAPL",
            fundamentals={"pe_ratio": 15, "pb_ratio": 2.0},
            vix_value=28.0,
        )

        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.action, "buy")  # 2 bullish vs 1 bearish
        self.assertIn("multi-analyst", verdict.model_used)
        self.assertIn("value_investor", verdict.reasoning)
        self.assertIn("contrarian", verdict.reasoning)

    def test_multi_analyst_crypto_returns_aggregated_verdict(self) -> None:
        service = self._make_service([
            '{"signal": "bullish", "confidence": 0.9, "reasoning": "Deep fear"}',
            '{"signal": "bullish", "confidence": 0.7, "reasoning": "Contrarian buy"}',
            '{"signal": "neutral", "confidence": 0.5, "reasoning": "Mixed"}',
        ])

        verdict = service.evaluate_crypto("BTCUSDT", 18, -0.002, 65000.0)

        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.action, "buy")

    def test_multi_analyst_uses_cache(self) -> None:
        service = self._make_service([
            '{"signal": "bullish", "confidence": 0.8, "reasoning": "r1"}',
            '{"signal": "bullish", "confidence": 0.7, "reasoning": "r2"}',
            '{"signal": "bullish", "confidence": 0.6, "reasoning": "r3"}',
        ])

        v1 = service.evaluate_stock("TEST", {"pe_ratio": 10}, 30.0)
        v2 = service.evaluate_stock("TEST", {"pe_ratio": 10}, 30.0)

        self.assertIsNotNone(v1)
        self.assertFalse(v1.cached)
        self.assertIsNotNone(v2)
        self.assertTrue(v2.cached)
        self.assertEqual(v2.cost_usd, 0.0)

    def test_multi_analyst_budget_check(self) -> None:
        config = TradingAgentsConfig(
            enabled=True,
            api_key="test-key",
            max_cost_per_month=0.001,
            multi_analyst_enabled=True,
            selected_analysts=("value_investor",),
        )
        service = self._make_service()
        service.config = config
        service._month_cost = 0.001

        verdict = service.evaluate_stock("TEST", {"pe_ratio": 10}, 30.0)
        self.assertIsNone(verdict)

    def test_fallback_to_two_tier_when_multi_analyst_disabled(self) -> None:
        """When multi_analyst_enabled=False, use the standard pipeline."""
        config = TradingAgentsConfig(
            enabled=True,
            api_key="test-key",
            quick_model="claude-haiku-4-5-20251001",
            deep_model="claude-sonnet-4-6",
            cache_ttl_hours=4,
            max_cost_per_run=0.50,
            max_cost_per_month=20.00,
            multi_analyst_enabled=False,
        )
        service = TradingAgentsService(
            TradingAgentsConfig(enabled=False), self.logger,
        )
        service.config = config
        service._anthropic_client = FakeAnthropicClient([
            "yes",  # Haiku screen
            '{"action": "buy", "confidence": 0.85, "reasoning": "Standard"}',
        ])

        verdict = service.evaluate_stock("AAPL", {"pe_ratio": 15}, 28.0)

        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.action, "buy")
        self.assertNotIn("multi-analyst", verdict.model_used)

    def test_multi_analyst_with_custom_personas(self) -> None:
        service = self._make_service([
            '{"signal": "bearish", "confidence": 0.9, "reasoning": "Fragile"}',
            '{"signal": "bearish", "confidence": 0.8, "reasoning": "Chart break"}',
        ])
        service.config = TradingAgentsConfig(
            enabled=True,
            api_key="test-key",
            max_cost_per_run=0.50,
            max_cost_per_month=20.00,
            multi_analyst_enabled=True,
            selected_analysts=("risk_analyst", "technical_analyst"),
        )

        verdict = service.evaluate_stock("XYZ", {"pe_ratio": 40}, 15.0)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.action, "sell")

    def test_multi_analyst_handles_model_failure_gracefully(self) -> None:
        """If all persona calls fail, returns None."""
        service = self._make_service()
        service._anthropic_client = MagicMock()
        service._anthropic_client.messages.create.side_effect = Exception("API down")

        verdict = service._multi_analyst_evaluation(
            "test:fail", "Evaluate something.",
        )
        self.assertIsNone(verdict)

    def test_multi_analyst_partial_failure_still_aggregates(self) -> None:
        """If some personas fail, aggregate what we have."""
        call_count = 0

        class PartialFailureClient:
            def __init__(self):
                self.messages = self

            def create(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("Temporary failure")
                return FakeAnthropicMessage(
                    '{"signal": "bullish", "confidence": 0.7, "reasoning": "OK"}'
                )

        service = self._make_service()
        service._anthropic_client = PartialFailureClient()

        verdict = service._multi_analyst_evaluation(
            "test:partial", "Evaluate something.",
        )
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.action, "buy")


if __name__ == "__main__":
    unittest.main()
