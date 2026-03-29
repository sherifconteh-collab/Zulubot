from __future__ import annotations

import logging
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.config import RiskConfig, StockStrategyConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.strategies.stocks import StockValueStrategy


class FakeFmpClient:
    def __init__(self) -> None:
        self.ratios = {
            "AAA": {"symbol": "AAA", "peRatioTTM": 12, "pbRatioTTM": 1.5, "debtToEquityTTM": 0.2},
            "BBB": {"symbol": "BBB", "peRatioTTM": 10, "pbRatioTTM": 1.8, "debtToEquityTTM": 0.1},
            "CCC": {"symbol": "CCC", "peRatioTTM": 15, "pbRatioTTM": 2.1, "debtToEquityTTM": 0.3},
            "DDD": {"symbol": "DDD", "peRatioTTM": 30, "pbRatioTTM": 4.0, "debtToEquityTTM": 0.8},
        }
        self.key_metrics = {
            "AAA": {"symbol": "AAA", "pfcfRatioTTM": 8},
            "BBB": {"symbol": "BBB", "pfcfRatioTTM": 7},
            "CCC": {"symbol": "CCC", "pfcfRatioTTM": 9},
            "DDD": {"symbol": "DDD", "pfcfRatioTTM": 12},
        }
        self.vix_value = 30.0
        self.quotes = {"AAA": 100.0, "BBB": 50.0, "CCC": 80.0, "DDD": 25.0}

    def get_sp500_constituents(self) -> list[str]:
        return ["AAA", "BBB", "CCC", "DDD"]

    def get_ratios_ttm_bulk(self) -> dict[str, dict]:
        return self.ratios

    def get_key_metrics_ttm_bulk(self) -> dict[str, dict]:
        return self.key_metrics

    def get_vix_price(self, symbol: str) -> float:
        return self.vix_value

    def get_batch_quotes(self, symbols: list[str]) -> dict[str, float]:
        return {symbol: self.quotes[symbol] for symbol in symbols}


class StockStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_stock_strategy")
        self.now = datetime(2026, 3, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        self.client = FakeFmpClient()
        self.strategy = StockValueStrategy(
            config=StockStrategyConfig(),
            client=self.client,
            logger=self.logger,
        )
        self.broker = PaperBroker(True, self.logger)
        self.risk_manager = RiskManager(RiskConfig())

    def test_buys_top_three_qualified_stocks_when_vix_is_high(self) -> None:
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        result = self.strategy.run(
            state,
            broker=self.broker,
            risk_manager=self.risk_manager,
            now=self.now,
            force=True,
        )

        self.assertEqual(result["top_targets"], ["BBB", "AAA", "CCC"])
        self.assertEqual(sorted(result["buys"]), ["AAA", "BBB", "CCC"])
        self.assertEqual(len(state.positions), 3)

    def test_sells_when_fundamentals_deteriorate_beyond_threshold(self) -> None:
        state = PortfolioState(
            cash=98_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )
        self.broker.open_position(
            state,
            symbol="AAA",
            asset_type="stock",
            strategy="value_investing",
            price=100.0,
            notional=2_000.0,
            opened_at=self.now,
            meta={},
        )
        self.client.ratios["AAA"]["peRatioTTM"] = 25

        result = self.strategy.run(
            state,
            broker=self.broker,
            risk_manager=self.risk_manager,
            now=self.now,
            force=True,
        )

        self.assertIn("AAA", result["sells"])
        self.assertNotIn("AAA", state.positions)


class MarketIntelligenceIntegrationTests(unittest.TestCase):
    """Tests for StockValueStrategy free market intelligence client integration."""

    _SAMPLE_SIGNALS = {
        "options_flow": {
            "calls_volume": 7_500, "puts_volume": 2_000,
            "calls_open_interest": 50_000, "puts_open_interest": 15_000,
            "put_call_volume_ratio": 0.27,
            "net_sentiment": "bullish",
        },
        "insider": {"recent_buys": 2, "recent_sells": 0, "net_sentiment": "bullish"},
        "short_interest": {"short_percent_of_float": 3.1, "days_to_cover": 2.5, "net_sentiment": "bullish"},
    }

    def setUp(self) -> None:
        self.logger = logging.getLogger("test_market_intel")
        self.now = datetime(2026, 3, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        self.client = _make_fmp_client()
        self.broker = PaperBroker(True, self.logger)
        self.risk_manager = RiskManager(RiskConfig())

    def _make_strategy(
        self,
        market_intel_client=None,
        ai_service=None,
    ) -> StockValueStrategy:
        return StockValueStrategy(
            config=StockStrategyConfig(),
            client=self.client,
            logger=self.logger,
            market_intel_client=market_intel_client,
            ai_service=ai_service,
        )

    def test_market_signals_persisted_in_position_meta(self) -> None:
        """Market signals returned by the client must be stored in opened position meta."""
        intel_client = _FakeMarketIntelClient(self._SAMPLE_SIGNALS)
        strategy = self._make_strategy(market_intel_client=intel_client)

        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now, force=True)

        for symbol, position in state.positions.items():
            self.assertIn("market_signals", position.meta, f"Missing market_signals in meta for {symbol}")
            self.assertEqual(position.meta["market_signals"], self._SAMPLE_SIGNALS)

    def test_market_signals_passed_to_ai_evaluate_stock(self) -> None:
        """Market signals must be forwarded to ai_service.evaluate_stock."""
        intel_client = _FakeMarketIntelClient(self._SAMPLE_SIGNALS)
        ai_service = _RecordingAiService()
        strategy = self._make_strategy(market_intel_client=intel_client, ai_service=ai_service)

        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        strategy.run(state, broker=self.broker, risk_manager=self.risk_manager, now=self.now, force=True)

        self.assertTrue(ai_service.calls, "ai_service.evaluate_stock was never called")
        for call_kwargs in ai_service.calls:
            self.assertEqual(call_kwargs.get("market_signals"), self._SAMPLE_SIGNALS)

    def test_market_intel_client_exception_does_not_abort_strategy(self) -> None:
        """A raising market_intel_client must not crash the strategy loop; buys continue with None signals."""
        intel_client = _ErrorMarketIntelClient()
        strategy = self._make_strategy(market_intel_client=intel_client)

        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        result = strategy.run(
            state, broker=self.broker, risk_manager=self.risk_manager, now=self.now, force=True
        )

        # Strategy should still open positions (graceful degradation)
        self.assertTrue(result.get("buys"), "Expected buys even when market intel client raises")
        for symbol, position in state.positions.items():
            self.assertNotIn("market_signals", position.meta, f"market_signals must be absent when fetch failed")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_fmp_client() -> FakeFmpClient:
    """Return a fresh FakeFmpClient with qualifying stocks."""
    return FakeFmpClient()


class _FakeMarketIntelClient:
    """Returns a fixed signals dict for every ticker."""
    def __init__(self, signals: dict) -> None:
        self._signals = signals

    def get_stock_signals(self, ticker: str) -> dict:
        return self._signals


class _ErrorMarketIntelClient:
    """Always raises on get_stock_signals."""
    def get_stock_signals(self, ticker: str) -> dict:
        raise RuntimeError("Simulated market intel failure")


class _RecordingAiService:
    """Records all calls to evaluate_stock and returns a buy verdict."""
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def is_available(self) -> bool:
        return True

    def evaluate_stock(self, **kwargs) -> object:
        from trading_bot.integrations.trading_agents import AgentVerdict
        self.calls.append(kwargs)
        return AgentVerdict(action="buy", confidence=0.8, reasoning="fake verdict")


if __name__ == "__main__":
    unittest.main()
