"""Kalshi order construction, fill handling and live booking safety."""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import requests

from trading_bot.broker import PaperBroker
from trading_bot.clients_kalshi import KalshiClient, order_fill_count
from trading_bot.config import KalshiSportsConfig, PredictionConfig
from trading_bot.models import PortfolioState, Position
from trading_bot.strategies.coinbase_predictions import CoinbasePredictionStrategy
from trading_bot.strategies.kalshi_sports import KalshiSportsStrategy

_NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def _client() -> KalshiClient:
    c = KalshiClient(logger=logging.getLogger("t"))
    # The V2 create-order response (https://docs.kalshi.com/api-reference/orders/create-order-v2).
    c._post = MagicMock(return_value={
        "order_id": "o1", "client_order_id": "c1", "fill_count": "3.00", "remaining_count": "0.00", "ts_ms": 1})
    return c


def _http_error(status: int) -> requests.HTTPError:
    return requests.HTTPError(f"{status}", response=SimpleNamespace(status_code=status, text="nope"))


class PlaceOrderTests(unittest.TestCase):
    """Orders go to the V2 endpoint, which quotes everything from the YES side."""

    def _payload(self, c):
        path, payload = c._post.call_args.args
        self.assertEqual(path, "/portfolio/events/orders")
        return payload

    def test_buy_yes_is_a_bid_at_the_yes_price(self):
        c = _client()
        c.place_order(ticker="T", side="yes", count=2, price_cents=57, time_in_force="immediate_or_cancel")
        p = self._payload(c)
        self.assertEqual((p["side"], p["price"], p["count"]), ("bid", "0.5700", "2.00"))
        self.assertEqual(p["time_in_force"], "immediate_or_cancel")
        self.assertEqual(p["self_trade_prevention_type"], "taker_at_cross")
        self.assertEqual(p["ticker"], "T")
        self.assertTrue(p["client_order_id"])
        for legacy in ("action", "yes_price", "no_price", "type"):
            self.assertNotIn(legacy, p)

    def test_buy_no_is_an_ask_at_one_minus_the_no_price(self):
        c = _client()
        c.place_order(ticker="T", side="no", count=2, price_cents=45, time_in_force="immediate_or_cancel")
        p = self._payload(c)
        self.assertEqual((p["side"], p["price"]), ("ask", "0.5500"))     # buy NO @45 = sell YES @55

    def test_selling_is_the_mirror_image(self):
        c = _client()
        c.place_order(ticker="T", side="yes", action="sell", count=1, price_cents=40)
        self.assertEqual((self._payload(c)["side"], self._payload(c)["price"]), ("ask", "0.4000"))
        c.place_order(ticker="T", side="no", action="sell", count=1, price_cents=40)
        self.assertEqual((self._payload(c)["side"], self._payload(c)["price"]), ("bid", "0.6000"))  # sell NO @40 = buy YES @60

    def test_yes_price_argument_is_always_the_yes_side_price(self):
        c = _client()
        c.place_order(ticker="T", side="no", count=1, yes_price=57)
        p = self._payload(c)
        self.assertEqual((p["side"], p["price"]), ("ask", "0.5700"))

    def test_price_is_required_clamped_and_inputs_validated(self):
        c = _client()
        with self.assertRaises(ValueError):
            c.place_order(ticker="T", side="yes", count=1)
        with self.assertRaises(ValueError):
            c.place_order(ticker="T", side="maybe", count=1, price_cents=50)
        c.place_order(ticker="T", side="yes", count=1, price_cents=150)
        self.assertEqual(self._payload(c)["price"], "0.9900")
        c.place_order(ticker="T", side="yes", count=1, price_cents=0)
        self.assertEqual(self._payload(c)["price"], "0.0100")
        self.assertEqual(self._payload(c)["time_in_force"], "good_till_canceled")

    def test_response_keeps_the_shape_callers_read(self):
        c = _client()
        out = c.place_order(ticker="T", side="yes", count=3, price_cents=50)
        self.assertEqual(out["order"]["order_id"], "o1")
        self.assertEqual(order_fill_count(out), 3)
        self.assertEqual(order_fill_count({"order_id": "x", "fill_count": "0.00"}), 0)

    def test_a_rejected_order_raises_kalshi_s_own_error_and_is_never_retried_elsewhere(self):
        # The legacy endpoint answers 410 "Please switch to the V2 endpoints", so falling back to it
        # would only replace the real V2 error with a misleading one.
        for status in (400, 404, 410):
            c = _client()
            c._post.side_effect = _http_error(status)
            with self.assertRaises(requests.HTTPError) as ctx:
                c.place_order(ticker="T", side="yes", count=1, price_cents=50)
            self.assertEqual(ctx.exception.response.status_code, status)
            self.assertEqual(c._post.call_count, 1)
            self.assertEqual(c._post.call_args.args[0], "/portfolio/events/orders")

    def test_sell_to_close_prices_just_below_the_bid(self):
        c = _client()
        c.get_orderbook = MagicMock(return_value={"orderbook_fp": {
            "yes_dollars": [["0.4300", "50"]], "no_dollars": [["0.5500", "50"]]}})
        c.sell_to_close(ticker="T", side="yes", count=4)
        p = self._payload(c)
        self.assertEqual((p["side"], p["price"], p["count"]), ("ask", "0.4000", "4.00"))   # sell YES: 43c bid - 3c
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


class SportsAccountGuardTests(unittest.TestCase):
    """The account, not the bot's own records, decides whether a game already has a bet."""

    def _strat(self, held):
        kalshi = MagicMock()
        if isinstance(held, Exception):
            kalshi.get_held_tickers.side_effect = held
        else:
            kalshi.get_held_tickers.return_value = held
        return _sports(kalshi), kalshi

    def test_events_come_from_the_tickers_the_account_holds(self):
        strat, _ = self._strat({"KXNHLGAME-26OCT11VGKMIN-VGK", "KXNFLPASSYDS-26OCT12DENLAC-DENJOHN4-100"})
        self.assertEqual(strat._account_held_events(),
                         {"KXNHLGAME-26OCT11VGKMIN", "KXNFLPASSYDS-26OCT12DENLAC-DENJOHN4"})

    def test_unreadable_positions_are_none_and_paper_mode_asks_nobody(self):
        strat, _ = self._strat(RuntimeError("down"))
        self.assertIsNone(strat._account_held_events())
        strat.live_trading_enabled = False
        self.assertEqual(strat._account_held_events(), set())

    def test_a_cycle_places_nothing_when_the_account_cannot_be_read(self):
        strat, kalshi = self._strat(RuntimeError("down"))
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        broker = PaperBroker(False, logging.getLogger("t"))
        result = {"decisions": [], "buys": [], "bankroll": {}, "predictions": []}
        strat._execute_candidates(state, broker, [_cand()], _NOW, result)
        kalshi.place_order.assert_not_called()
        self.assertIn("positions_unavailable", [d["reason"] for d in result["decisions"]])

    def test_a_game_the_account_already_holds_is_not_bet_again(self):
        strat, kalshi = self._strat({"KXNFLGAME-26OCT04ABAB-A"})   # same event as the candidate, local state empty
        kalshi.place_order.return_value = {"order": {"order_id": "o", "fill_count": 5}}
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        broker = PaperBroker(False, logging.getLogger("t"))
        result = {"decisions": [], "buys": [], "bankroll": {}, "predictions": []}
        strat._execute_candidates(state, broker, [_cand()], _NOW, result)
        kalshi.place_order.assert_not_called()
        self.assertEqual(result["buys"], [])
        # ...and a different game is still bet
        other = _cand()
        other["market"] = {"ticker": "KXNFLGAME-26OCT04CDCD-C", "event_ticker": "KXNFLGAME-26OCT04CDCD"}
        strat._execute_candidates(state, broker, [other], _NOW, result)
        kalshi.place_order.assert_called_once()


class AccountExposureTests(unittest.TestCase):
    """How much is at risk comes from the account, so a redeploy cannot reset the exposure budget."""

    def test_client_sums_cost_of_open_positions_only(self):
        c = KalshiClient(logger=logging.getLogger("t"))
        c.get_positions = MagicMock(return_value=[
            {"ticker": "KXNHLGAME-26OCT11VGKMIN-VGK", "position_fp": "4.00", "market_exposure_dollars": "3.9800"},
            {"ticker": "KXGDP-26-T3.5", "position_fp": "-13.00", "market_exposure_dollars": "1.7300"},
            {"ticker": "KXOLD-1-A", "position_fp": "0.00", "market_exposure_dollars": "0.0000"},
            {"ticker": "KXBAD-1-A", "position_fp": "x", "market_exposure_dollars": "1.0"},
        ])
        self.assertEqual(c.get_exposure_by_ticker(), {"KXNHLGAME-26OCT11VGKMIN-VGK": 3.98, "KXGDP-26-T3.5": 1.73})

    def test_only_sports_series_count_toward_the_sports_budget(self):
        kalshi = MagicMock()
        kalshi.get_exposure_by_ticker.return_value = {
            "KXNHLGAME-26OCT11VGKMIN-VGK": 3.98, "KXUCLGAME-26OCT14LENSCP-LEN": 1.88,
            "KXNFLPASSYDS-26OCT12DENLAC-DENJOHN4-100": 2.0, "KXGDP-26-T3.5": 1.73, "KXINFL-26-T3.3": 3.2}
        self.assertAlmostEqual(_sports(kalshi)._account_sports_exposure(), 7.86)

    def test_unreadable_account_falls_back_to_zero_and_paper_asks_nobody(self):
        kalshi = MagicMock()
        kalshi.get_exposure_by_ticker.side_effect = RuntimeError("down")
        strat = _sports(kalshi)
        self.assertEqual(strat._account_sports_exposure(), 0.0)
        strat.live_trading_enabled = False
        kalshi.get_exposure_by_ticker.side_effect = None
        kalshi.get_exposure_by_ticker.return_value = {"KXNHLGAME-26OCT11VGKMIN-VGK": 3.98}
        self.assertEqual(strat._account_sports_exposure(), 0.0)

    def test_bankroll_shows_what_the_account_holds_when_local_records_are_empty(self):
        kalshi = MagicMock()
        kalshi.get_held_tickers.return_value = set()
        kalshi.get_exposure_by_ticker.return_value = {"KXNHLGAME-26OCT11VGKMIN-VGK": 15.0}
        strat = _sports(kalshi)
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        broker = PaperBroker(False, logging.getLogger("t"))
        result = {"decisions": [], "buys": [], "bankroll": {}, "predictions": []}
        strat._execute_candidates(state, broker, [], _NOW, result)
        self.assertAlmostEqual(result["bankroll"]["open_exposure"], 15.0)


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

    def test_rejected_and_unfilled_orders_are_recorded_with_kalshi_s_words(self):
        err = requests.HTTPError("400", response=SimpleNamespace(status_code=400, text='{"error":"insufficient_balance"}'))
        kalshi = MagicMock()
        kalshi.place_order.side_effect = err
        strat = _sports(kalshi)
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        broker = PaperBroker(False, logging.getLogger("t"))
        self.assertIsNone(strat._place_bet(state, broker, _cand(), 20.0, _NOW, "kalshi"))
        self.assertEqual(strat._order_issues[0]["reason"], "order_failed")
        self.assertIn("HTTP 400", strat._order_issues[0]["error"])
        self.assertIn("insufficient_balance", strat._order_issues[0]["error"])
        kalshi.place_order.side_effect = None
        kalshi.place_order.return_value = {"order": {"fill_count": 0}}
        other = _cand()
        other["market"] = {"ticker": "KXNHLGAME-26OCT04ABAB-A", "event_ticker": "KXNHLGAME-26OCT04ABAB"}
        strat._place_bet(state, broker, other, 20.0, _NOW, "kalshi")
        self.assertEqual(strat._order_issues[1]["reason"], "order_not_filled")

    def _cand_for(self, ticker):
        cand = _cand()
        cand["market"] = {"ticker": ticker, "event_ticker": ticker.rsplit("-", 1)[0]}
        return cand

    def test_other_failures_carry_the_orders_numbers_and_stop_after_three(self):
        err = requests.HTTPError("500", response=SimpleNamespace(status_code=500, text="boom"))
        kalshi = MagicMock()
        kalshi.place_order.side_effect = err
        strat = _sports(kalshi)
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        broker = PaperBroker(False, logging.getLogger("t"))
        for n in range(5):
            strat._place_bet(state, broker, self._cand_for(f"KXA{n}-26OCT04ABAB-X"), 20.0, _NOW, "kalshi")
        self.assertEqual(kalshi.place_order.call_count, 3)          # no more attempts once Kalshi keeps refusing
        first = strat._order_issues[0]["error"]
        self.assertTrue(first.startswith("KXA0-26OCT04ABAB-X yes x40 @ 0.50: HTTP 500"), first)

    def test_shard_errors_skip_that_series_but_keep_trying_others(self):
        err = requests.HTTPError("404", response=SimpleNamespace(
            status_code=404, text='{"error":{"code":"insufficient_shard_balance","details":"Exchange user not found"}}'))
        kalshi = MagicMock()
        kalshi.place_order.side_effect = [err, {"order": {"order_id": "ok", "fill_count": 38}}]
        strat = _sports(kalshi)
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        broker = PaperBroker(False, logging.getLogger("t"))
        results = [
            strat._place_bet(state, broker, self._cand_for("KXSOCCER-26OCT04AAA-X"), 20.0, _NOW, "kalshi"),   # refused
            strat._place_bet(state, broker, self._cand_for("KXSOCCER-26OCT04BBB-X"), 20.0, _NOW, "kalshi"),   # same series: skipped
            strat._place_bet(state, broker, self._cand_for("KXNFLGAME-26OCT04CCC-X"), 20.0, _NOW, "kalshi"),  # other series: placed
        ]
        self.assertIsNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNotNone(results[2])
        self.assertEqual(kalshi.place_order.call_count, 2)           # the skipped series was never asked again
        self.assertTrue(strat._order_issues[0]["shard"])
        self.assertIn("insufficient_shard_balance", strat._order_issues[0]["error"])
        # a hand-picked bet is never silently skipped
        kalshi.place_order.side_effect = None
        kalshi.place_order.return_value = {"order": {"order_id": "m", "fill_count": 38}}
        manual = self._cand_for("KXSOCCER-26OCT04DDD-X")
        manual["user_bet"] = True
        self.assertIsNotNone(strat._place_bet(state, broker, manual, 20.0, _NOW, "kalshi"))

    def test_shard_balances_are_read_from_the_balance_breakdown(self):
        from unittest.mock import PropertyMock, patch
        c = _client()
        with patch.object(KalshiClient, "can_trade", new_callable=PropertyMock, return_value=True):
            c._get = MagicMock(return_value={"balance": 5861, "balance_breakdown": [
                {"exchange_index": 1, "balance": "0.0000"}, {"exchange_index": 0, "balance": "58.6100"}, {"bad": 1}]})
            self.assertEqual(c.get_balance_breakdown(), [(0, 58.61), (1, 0.0)])
            c._get = MagicMock(return_value={"balance": 5861})
            self.assertEqual(c.get_balance_breakdown(), [])
        self.assertEqual(c.get_balance_breakdown(), [])          # no credentials: nothing to ask

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
