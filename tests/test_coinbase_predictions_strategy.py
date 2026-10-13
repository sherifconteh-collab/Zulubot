from __future__ import annotations

import logging
import unittest
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.config import PredictionConfig, RiskConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.strategies.coinbase_predictions import CoinbasePredictionStrategy


def _market(ticker: str, yes_bid: float, yes_ask: float, *, status: str = "active", result: str = "") -> dict[str, Any]:
    return {
        "ticker": ticker, "status": status, "result": result,
        "yes_bid_dollars": str(yes_bid), "yes_ask_dollars": str(yes_ask),
        "no_bid_dollars": str(round(1 - yes_ask, 4)), "no_ask_dollars": str(round(1 - yes_bid, 4)),
        "volume_fp": "50000",
    }


class FakeKalshi:
    def __init__(self, markets: list[dict[str, Any]]) -> None:
        self._markets = {m["ticker"]: m for m in markets}
        self.can_trade = False
        self.orders: list[dict[str, Any]] = []
        self.listed = 0

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self._markets[ticker]

    def get_markets(self, **kwargs: Any):
        self.listed += 1
        return list(self._markets.values()), None

    def place_order(self, **kwargs: Any) -> dict[str, Any]:
        self.orders.append(kwargs)
        return {"order": kwargs}


class LegacyPredictionPositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_predictions")
        self.now = datetime(2026, 10, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        self.broker = PaperBroker(True, self.logger)
        self.risk = RiskManager(RiskConfig())

    def _state(self) -> PortfolioState:
        return PortfolioState(cash=100_000.0, daily_baseline_equity=100_000.0,
                              daily_baseline_date=self.now.date().isoformat(), peak_equity=100_000.0)

    def _strategy(self, markets: list[dict[str, Any]], enabled: bool = True):
        kalshi = FakeKalshi(markets)
        return CoinbasePredictionStrategy(
            config=PredictionConfig(enabled=enabled), kalshi_client=kalshi,  # type: ignore[arg-type]
            logger=self.logger,
        ), kalshi

    def _open(self, state: PortfolioState, ticker: str, side: str, price: float):
        symbol = f"KX:{ticker}:{side}"
        self.broker.open_position(
            state, symbol=symbol, asset_type="prediction", strategy="coinbase_predictions", price=price,
            notional=price * 10, opened_at=self.now, meta={"ticker": ticker, "side": side},
        )
        return symbol

    def test_never_opens_a_position_even_on_a_cheap_long_shot(self) -> None:
        strategy, kalshi = self._strategy([_market("KXCPI-T1", 0.03, 0.04), _market("KXBTC-T2", 0.96, 0.97)])
        state = self._state()
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["buys"], [])
        self.assertEqual(kalshi.orders, [])
        self.assertEqual(kalshi.listed, 0)           # it does not even scan for markets
        self.assertEqual(state.positions, {})

    def test_disabled_does_nothing(self) -> None:
        strategy, _ = self._strategy([], enabled=False)
        self.assertEqual(strategy.run(self._state(), broker=self.broker, risk_manager=self.risk, now=self.now)["status"],
                         "disabled")

    def test_resolved_contract_is_closed_at_its_settlement_value(self) -> None:
        strategy, _ = self._strategy([_market("KXCPI-T1", 0.0, 0.0, status="finalized", result="yes")])
        state = self._state()
        symbol = self._open(state, "KXCPI-T1", "yes", 0.20)
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertEqual(result["resolved_closed"], [symbol])
        self.assertNotIn(symbol, state.positions)

    def test_take_profit_and_stop_loss_still_apply_to_held_positions(self) -> None:
        strategy, _ = self._strategy([_market("UP", 0.55, 0.57), _market("DOWN", 0.20, 0.22)])
        state = self._state()
        up = self._open(state, "UP", "yes", 0.30)
        down = self._open(state, "DOWN", "yes", 0.45)
        result = strategy.run(state, broker=self.broker, risk_manager=self.risk, now=self.now)
        self.assertCountEqual(result["sells"], [up, down])

    def test_no_side_is_valued_at_the_no_price(self) -> None:
        strategy, _ = self._strategy([_market("KXBTC-NO", 0.28, 0.30)])
        state = self._state()
        symbol = self._open(state, "KXBTC-NO", "no", 0.72)
        strategy._manage_existing_positions(state, self.broker, {"sells": []})
        self.assertAlmostEqual(state.positions[symbol].current_price, 0.71, places=2)


if __name__ == "__main__":
    unittest.main()
