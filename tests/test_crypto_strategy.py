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
            config=CryptoStrategyConfig(),
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


if __name__ == "__main__":
    unittest.main()
