"""Kalshi order construction, fill handling and live booking safety."""

from __future__ import annotations

import logging
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from trading_bot.broker import PaperBroker
from trading_bot.clients_kalshi import KalshiClient, order_fill_count
from trading_bot.config import KalshiSportsConfig, PredictionConfig
from trading_bot.models import PortfolioState, Position
from trading_bot.strategies.coinbase_predictions import CoinbasePredictionStrategy
from trading_bot.strategies.kalshi_sports import KalshiSportsStrategy

_NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def _client() -> KalshiClient:
    c = KalshiClient(logger=logging.getLogger("t"))
    c._post = MagicMock(return_value={"order": {"order_id": "o1", "fill_count": 3}})
    return c


class PlaceOrderTests(unittest.TestCase):
    def test_no_side_price_uses_no_price(self):
        c = _client()
        c.place_order(ticker="T", side="no", count=2, price_cents=45, time_in_force="immediate_or_cancel")
        payload = c._post.call_args.args[1]
        self.assertEqual(payload["no_price"], 45)
        self.assertNotIn("yes_price", payload)
        self.assertEqual(payload["time_in_force"], "immediate_or_cancel")

    def test_yes_side_price_uses_yes_price(self):
        c = _client()
        c.place_order(ticker="T", side="yes", count=2, price_cents=57)
        self.assertEqual(c._post.call_args.args[1]["yes_price"], 57)

    def test_price_is_required_and_clamped(self):
        c = _client()
        with self.assertRaises(ValueError):
            c.place_order(ticker="T", side="yes", count=1)
        c.place_order(ticker="T", side="yes", count=1, price_cents=150)
        self.assertEqual(c._post.call_args.args[1]["yes_price"], 99)

    def test_sell_to_close_prices_just_below_the_bid(self):
        c = _client()
        c.get_orderbook = MagicMock(return_value={"orderbook_fp": {
            "yes_dollars": [["0.4300", "50"]], "no_dollars": [["0.5500", "50"]]}})
        c.sell_to_close(ticker="T", side="yes", count=4)
        p = c._post.call_args.args[1]
        self.assertEqual((p["action"], p["side"], p["yes_price"]), ("sell", "yes", 40))   # 43c bid - 3c
        self.assertEqual(p["time_in_force"], "immediate_or_cancel")

    def test_sell_to_close_without_a_bid_raises(self):
        c = _client()
        c.get_orderbook = MagicMock(return_value={"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})
        with self.assertRaises(RuntimeError):
            c.sell_to_close(ticker="T", side="yes", count=1)

    def test_fill_count_parsing(self):
        self.assertEqual(order_fill_count({"order": {"fill_count": 3}}), 3)
        self.assertEqual(order_fill_count({"order": {"fill_count_fp": "2.00"}}), 2)
        self.assertEqual(order_fill_count({"order": {"taker_fill_count": 1, "maker_fill_count": 2}}), 3)
        self.assertEqual(order_fill_count({"order": {"fill_count": 0}}), 0)
        self.assertIsNone(order_fill_count({"order": {"status": "resting"}}))


class BrokerResizeTests(unittest.TestCase):
    def test_resize_refunds_the_unfilled_part(self):
        broker = PaperBroker(True, logging.getLogger("t"))
        state = PortfolioState(cash=100.0)
        broker.open_position(state, symbol="X", asset_type="prediction", strategy="s", price=0.5,
                             notional=20.0, opened_at=_NOW)           # 40 contracts, cash 80
        broker.resize_position(state, "X", 10)
        self.assertEqual(state.positions["X"].quantity, 10)
        self.assertAlmostEqual(state.cash, 95.0)

    def test_resize_to_zero_removes_and_refunds(self):
        broker = PaperBroker(True, logging.getLogger("t"))
        state = PortfolioState(cash=100.0)
        broker.open_position(state, symbol="X", asset_type="prediction", strategy="s", price=0.5,
                             notional=20.0, opened_at=_NOW)
        self.assertIsNone(broker.resize_position(state, "X", 0))
        self.assertNotIn("X", state.positions)
        self.assertAlmostEqual(state.cash, 100.0)


def _sports(kalshi) -> KalshiSportsStrategy:
    return KalshiSportsStrategy(
        config=KalshiSportsConfig(enabled=True, leagues=["nfl"], min_trade_usd=1.0, max_position_pct=0.5),
        kalshi_client=kalshi, espn_client=MagicMock(), logger=logging.getLogger("t"), live_trading_enabled=True,
    )


def _cand(side="yes", ask=0.50):
    from trading_bot.sports_model import BetDecision, Posterior, Signal
    game = MagicMock(league="football/nfl", label="A @ B", odds_provider="DK",
                     start_time=_NOW + timedelta(hours=5))
    decision = BetDecision(side=side, price=ask, p=0.6, sd=0.02, fee=0.02, net_edge=0.08,
                           stake_fraction=0.1, confidence=0.99)
    post = Posterior(p=0.6, sd=0.02, used=[Signal("sportsbook", 0.6, 0.025)])
    return {"market": {"ticker": "KXNFLGAME-26OCT04ABAB-A", "event_ticker": "KXNFLGAME-26OCT04ABAB"},
            "game": game, "decision": decision, "posterior": post, "pick": "A", "depth": 0, "edge": 0.008}


class SportsLiveBookingTests(unittest.TestCase):
    def _run(self, order_response=None, order_error=None, side="yes"):
        kalshi = MagicMock()
        if order_error:
            kalshi.place_order.side_effect = order_error
        else:
            kalshi.place_order.return_value = order_response
        strat = _sports(kalshi)
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        broker = PaperBroker(False, logging.getLogger("t"))
        pos = strat._place_bet(state, broker, _cand(side), 20.0, _NOW, "kalshi")
        return kalshi, state, pos

    def test_full_fill_books_the_bet_with_correct_price_arguments(self):
        kalshi, state, pos = self._run({"order": {"order_id": "o9", "fill_count": 38}})
        kwargs = kalshi.place_order.call_args.kwargs
        self.assertEqual((kwargs["side"], kwargs["price_cents"], kwargs["time_in_force"]),
                         ("yes", 50, "immediate_or_cancel"))
        self.assertEqual(pos.quantity, 38)           # filled 38 of the 40 requested
        self.assertEqual(pos.meta["kalshi_order_id"], "o9")

    def test_no_side_order_uses_the_no_price(self):
        kalshi, _, pos = self._run({"order": {"fill_count": 38}}, side="no")
        self.assertEqual(kalshi.place_order.call_args.kwargs["side"], "no")
        self.assertEqual(kalshi.place_order.call_args.kwargs["price_cents"], 50)

    def test_partial_fill_books_only_what_filled(self):
        _, state, pos = self._run({"order": {"order_id": "o", "fill_count": 5}})
        self.assertEqual(pos.quantity, 5)
        self.assertGreater(state.cash_reserves["kalshi"], 90.0)   # only 5 contracts paid for

    def test_unfilled_order_books_nothing(self):
        _, state, pos = self._run({"order": {"order_id": "o", "fill_count": 0}})
        self.assertIsNone(pos)
        self.assertEqual(state.positions, {})
        self.assertEqual(state.cash_reserves["kalshi"], 100.0)

    def test_failed_order_books_nothing(self):
        _, state, pos = self._run(order_error=RuntimeError("400"))
        self.assertIsNone(pos)
        self.assertEqual(state.positions, {})

    def test_unknown_response_shape_trusts_the_request_for_the_sync_backstop(self):
        _, _, pos = self._run({"order": {"order_id": "o", "status": "executed"}})
        self.assertEqual(pos.quantity, 40)        # 20 / 0.50: live entry is the ask, Kalshi bills the fee


def _pred_strategy(kalshi) -> CoinbasePredictionStrategy:
    return CoinbasePredictionStrategy(
        config=PredictionConfig(enabled=True), kalshi_client=kalshi,
        logger=logging.getLogger("t"), live_trading_enabled=True,
    )


def _pred_position(qty=10):
    return Position(symbol="KX:T:yes", asset_type="prediction", strategy="coinbase_predictions",
                    quantity=qty, entry_price=0.4, current_price=0.4, opened_at="x",
                    meta={"ticker": "T", "side": "yes", "cash_bucket": "kalshi"})


class PredictionSellTests(unittest.TestCase):
    def _close(self, kalshi):
        strat = _pred_strategy(kalshi)
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 0.0})
        pos = _pred_position()
        state.positions[pos.symbol] = pos
        broker = PaperBroker(False, logging.getLogger("t"))
        result = {"sells": [], "decisions": []}
        strat._close_position(state, broker, pos.symbol, pos, 0.55, "take_profit", result)
        return state, result

    def test_failed_sell_keeps_the_position_open(self):
        kalshi = MagicMock()
        kalshi.sell_to_close.side_effect = RuntimeError("No yes bid")
        state, result = self._close(kalshi)
        self.assertIn("KX:T:yes", state.positions)
        self.assertEqual(result["sells"], [])

    def test_unfilled_sell_keeps_the_position_open(self):
        kalshi = MagicMock()
        kalshi.sell_to_close.return_value = {"order": {"fill_count": 0}}
        state, _ = self._close(kalshi)
        self.assertIn("KX:T:yes", state.positions)

    def test_partial_sell_shrinks_the_position(self):
        kalshi = MagicMock()
        kalshi.sell_to_close.return_value = {"order": {"fill_count": 4}}
        state, result = self._close(kalshi)
        self.assertEqual(state.positions["KX:T:yes"].quantity, 6)
        self.assertEqual(result["sells"], [])

    def test_full_sell_closes_the_position(self):
        kalshi = MagicMock()
        kalshi.sell_to_close.return_value = {"order": {"fill_count": 10}}
        state, result = self._close(kalshi)
        self.assertNotIn("KX:T:yes", state.positions)
        self.assertEqual(result["sells"], ["KX:T:yes"])


if __name__ == "__main__":
    unittest.main()
