from __future__ import annotations

import logging
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.config import CryptoStrategyConfig, RiskConfig
from trading_bot.models import FearGreedReading, FundingSnapshot, PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.strategies.crypto import CryptoSentimentFundingStrategy


class FakeAlternativeClient:
    def __init__(self, value: int, classification: str) -> None:
        self.reading = FearGreedReading(
            value=value,
            classification=classification,
            timestamp="1710000000",
            time_until_update=3600,
        )

    def get_fear_and_greed(self) -> FearGreedReading:
        return self.reading


class FakeBinanceClient:
    def __init__(self, funding_rates: dict[str, float]) -> None:
        self.funding_rates = funding_rates

    def get_mark_price_and_funding(self, symbol: str) -> FundingSnapshot:
        return FundingSnapshot(
            symbol=symbol,
            mark_price=50_000.0 if symbol == "BTCUSDT" else 3_000.0,
            funding_rate=self.funding_rates[symbol],
            next_funding_time=1710003600,
            observed_at=1710000000,
        )


class CryptoStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_crypto_strategy")
        self.now = datetime(2026, 3, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        self.risk_manager = RiskManager(RiskConfig())
        self.broker = PaperBroker(True, self.logger)

    def test_buy_requires_extreme_fear_and_negative_funding(self) -> None:
        strategy = CryptoSentimentFundingStrategy(
            config=CryptoStrategyConfig(require_negative_funding=True),
            alternative_client=FakeAlternativeClient(20, "Extreme Fear"),
            binance_client=FakeBinanceClient({"BTCUSDT": -0.0005, "ETHUSDT": 0.0002}),
            logger=self.logger,
        )
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        result = strategy.run(
            state,
            broker=self.broker,
            risk_manager=self.risk_manager,
            now=self.now,
        )

        self.assertEqual(result["buys"], ["BTCUSDT"])
        self.assertIn("BTCUSDT", state.positions)
        self.assertNotIn("ETHUSDT", state.positions)

    def test_default_buys_on_fear_regardless_of_funding_rate(self) -> None:
        """Default (aggressive) behaviour: F&G < buy_fear_threshold is enough to buy,
        even when funding rate is positive. The funding-rate gate is opt-in."""
        strategy = CryptoSentimentFundingStrategy(
            config=CryptoStrategyConfig(),  # require_negative_funding default = False
            alternative_client=FakeAlternativeClient(20, "Extreme Fear"),
            binance_client=FakeBinanceClient({"BTCUSDT": 0.0008, "ETHUSDT": 0.0002}),
            logger=self.logger,
        )
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )

        result = strategy.run(
            state,
            broker=self.broker,
            risk_manager=self.risk_manager,
            now=self.now,
        )

        self.assertIn("BTCUSDT", result["buys"])
        self.assertIn("ETHUSDT", result["buys"])

    def test_sell_triggers_when_fear_and_greed_exceeds_threshold(self) -> None:
        strategy = CryptoSentimentFundingStrategy(
            config=CryptoStrategyConfig(),
            alternative_client=FakeAlternativeClient(80, "Extreme Greed"),
            binance_client=FakeBinanceClient({"BTCUSDT": -0.0005, "ETHUSDT": -0.0001}),
            logger=self.logger,
        )
        state = PortfolioState(
            cash=95_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )
        self.broker.open_position(
            state,
            symbol="BTCUSDT",
            asset_type="crypto",
            strategy="crypto_sentiment_funding",
            price=50_000.0,
            notional=2_000.0,
            opened_at=self.now,
            meta={},
        )

        result = strategy.run(
            state,
            broker=self.broker,
            risk_manager=self.risk_manager,
            now=self.now,
        )

        self.assertIn("BTCUSDT", result["sells"])
        self.assertNotIn("BTCUSDT", state.positions)

    def test_take_profit_closes_position_independent_of_fear_and_greed(self) -> None:
        """Autonomous sell: when a position is +take_profit_pct from entry,
        close it regardless of the F&G value. This lets the bot "sell as it
        sees fit" on Coinbase rather than waiting for F&G > sell_threshold."""
        # Neutral F&G (won't trigger either F&G buy or F&G sell branch).
        # Entry at 50_000; exit price 58_000 → +16% move, above default
        # take_profit_pct of 15%.
        strategy = CryptoSentimentFundingStrategy(
            config=CryptoStrategyConfig(take_profit_pct=0.15, stop_loss_pct=0.0),
            alternative_client=FakeAlternativeClient(50, "Neutral"),
            binance_client=FakeBinanceClient({"BTCUSDT": 0.0, "ETHUSDT": 0.0}),
            logger=self.logger,
        )
        state = PortfolioState(
            cash=95_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )
        self.broker.open_position(
            state,
            symbol="BTCUSDT",
            asset_type="crypto",
            strategy="crypto_sentiment_funding",
            price=50_000.0,
            notional=5_000.0,
            opened_at=self.now,
            meta={},
        )
        # Next cycle: mark price climbs 16% so Binance mark_price goes to 58k.
        # FakeBinanceClient returns 50_000 for BTC, so override via last_prices.
        state.last_prices["BTCUSDT"] = 58_000.0
        # Simulate a Binance fake returning the new mark price by swapping it in.
        strategy.binance_client = FakeBinanceClient({"BTCUSDT": 0.0, "ETHUSDT": 0.0})
        # Patch the mark price via a one-off subclass behaviour:
        original = strategy.binance_client.get_mark_price_and_funding

        def patched(symbol):  # type: ignore[no-redef]
            snap = original(symbol)
            if symbol == "BTCUSDT":
                from trading_bot.models import FundingSnapshot
                return FundingSnapshot(
                    symbol=symbol, mark_price=58_000.0,
                    funding_rate=snap.funding_rate,
                    next_funding_time=snap.next_funding_time,
                    observed_at=snap.observed_at,
                )
            return snap

        strategy.binance_client.get_mark_price_and_funding = patched  # type: ignore[assignment]

        result = strategy.run(
            state, broker=self.broker, risk_manager=self.risk_manager, now=self.now,
        )

        self.assertIn("BTCUSDT", result["sells"])
        self.assertNotIn("BTCUSDT", state.positions)
        reasons = [d.get("reason") for d in result["decisions"] if d.get("action") == "sell"]
        self.assertIn("take_profit", reasons)

    def test_stop_loss_closes_position_independent_of_fear_and_greed(self) -> None:
        """Autonomous sell: when a position is −stop_loss_pct from entry,
        close it regardless of F&G."""
        strategy = CryptoSentimentFundingStrategy(
            config=CryptoStrategyConfig(take_profit_pct=0.0, stop_loss_pct=0.10),
            alternative_client=FakeAlternativeClient(50, "Neutral"),
            binance_client=FakeBinanceClient({"BTCUSDT": 0.0, "ETHUSDT": 0.0}),
            logger=self.logger,
        )
        state = PortfolioState(
            cash=95_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )
        self.broker.open_position(
            state,
            symbol="BTCUSDT",
            asset_type="crypto",
            strategy="crypto_sentiment_funding",
            price=50_000.0,
            notional=5_000.0,
            opened_at=self.now,
            meta={},
        )
        # Price drops 11% → trigger stop-loss (threshold 10%).
        original = strategy.binance_client.get_mark_price_and_funding

        def patched(symbol):  # type: ignore[no-redef]
            snap = original(symbol)
            if symbol == "BTCUSDT":
                from trading_bot.models import FundingSnapshot
                return FundingSnapshot(
                    symbol=symbol, mark_price=44_500.0,
                    funding_rate=snap.funding_rate,
                    next_funding_time=snap.next_funding_time,
                    observed_at=snap.observed_at,
                )
            return snap

        strategy.binance_client.get_mark_price_and_funding = patched  # type: ignore[assignment]

        result = strategy.run(
            state, broker=self.broker, risk_manager=self.risk_manager, now=self.now,
        )

        self.assertIn("BTCUSDT", result["sells"])
        reasons = [d.get("reason") for d in result["decisions"] if d.get("action") == "sell"]
        self.assertIn("stop_loss", reasons)

    def test_learning_memory_benches_symbol_after_consecutive_losses(self) -> None:
        """After LEARNING_LOSER_COOLDOWN_LOSSES consecutive losses on a symbol,
        the crypto strategy skips re-entering it even when F&G is in the buy
        zone. The cooldown expires after N cycles so the bot can try again."""
        from trading_bot.config import LearningConfig
        from trading_bot.performance_memory import record_close

        strategy = CryptoSentimentFundingStrategy(
            config=CryptoStrategyConfig(),  # aggressive defaults
            alternative_client=FakeAlternativeClient(20, "Extreme Fear"),
            binance_client=FakeBinanceClient({"BTCUSDT": 0.0, "ETHUSDT": 0.0}),
            logger=self.logger,
            learning_config=LearningConfig(
                enabled=True, loser_cooldown_losses=2, loser_cooldown_cycles=5,
            ),
        )
        state = PortfolioState(
            cash=100_000.0,
            daily_baseline_equity=100_000.0,
            daily_baseline_date=self.now.date().isoformat(),
            peak_equity=100_000.0,
        )
        # Seed two consecutive losses on BTCUSDT → cooldown engaged.
        record_close(
            state, symbol="BTCUSDT", realized_pnl=-50.0,
            config=strategy.learning_config,
        )
        record_close(
            state, symbol="BTCUSDT", realized_pnl=-50.0,
            config=strategy.learning_config,
        )

        result = strategy.run(
            state, broker=self.broker, risk_manager=self.risk_manager, now=self.now,
        )

        self.assertNotIn("BTCUSDT", result["buys"])
        self.assertIn("ETHUSDT", result["buys"])  # unaffected symbol still bought
        reasons = [
            d.get("reason") for d in result["decisions"]
            if d.get("symbol") == "BTCUSDT" and d.get("action") == "skip_buy"
        ]
        self.assertIn("performance_memory_benched", reasons)


if __name__ == "__main__":
    unittest.main()
