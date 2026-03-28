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


if __name__ == "__main__":
    unittest.main()

