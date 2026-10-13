from __future__ import annotations

import logging
import time
import unittest
from datetime import datetime, timezone
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients_nowcast import GdpNowClient, NowcastReading, parse_fred_csv, quarter_label
from trading_bot.config import GdpNowcastConfig, RiskConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.strategies.gdp_nowcast import (
    GdpNowcastStrategy,
    evaluate_market,
    kelly_stake,
    lead_sd,
    market_quarter,
    prob_above,
)
from trading_bot.telegram_bot import _format_gdp

_NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)      # 10 days before the Oct 30 release
_LOG = logging.getLogger("test_gdp")


def _reading(value: float = 4.0, quarter: str = "2026Q3") -> NowcastReading:
    return NowcastReading(quarter=quarter, value=value, observed_for="2026-07-01",
                          fetched_at=time.time(), source="FRED GDPNOW")


def _market(strike: float = 3.5, bid: float = 0.36, ask: float = 0.37, *, event: str = "KXGDP-26OCT30",
            quarter: str = "Q3 2026", release: str = "2026-10-30T14:00:00Z", close: str = "2026-10-30T12:29:00Z",
            size: float = 200.0, strike_type: str = "greater") -> dict[str, Any]:
    return {
        "ticker": f"{event}-T{strike:g}", "event_ticker": event, "status": "active",
        "title": f"Will **real GDP** increase by more than {strike:g}% in {quarter}?",
        "strike_type": strike_type, "floor_strike": strike,
        "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}",
        "no_bid_dollars": f"{1 - ask:.4f}", "no_ask_dollars": f"{1 - bid:.4f}",
        "yes_ask_size_fp": f"{size:.2f}", "no_ask_size_fp": f"{size:.2f}",
        "expected_expiration_time": release, "close_time": close,
    }


class FakeNowcast:
    def __init__(self, reading: NowcastReading | None, by_quarter: dict[str, NowcastReading] | None = None) -> None:
        self.reading = reading
        self.by_quarter = by_quarter or {}
        self.last_error = "" if reading else "boom"

    def latest(self) -> NowcastReading | None:
        return self.reading

    def for_quarter(self, quarter: str) -> NowcastReading | None:
        return self.by_quarter.get(quarter)


class FakeKalshi:
    def __init__(self, markets: list[dict[str, Any]]) -> None:
        self.markets = markets
        self.orders: list[dict[str, Any]] = []
        self.sells: list[dict[str, Any]] = []
        self.held: set[str] | Exception = set()
        self.exposure: dict[str, float] = {}
        self.fill: int | None = None           # None -> fill whatever was asked
        self.order_error: Exception | None = None

    def get_markets(self, **_: Any) -> tuple[list[dict[str, Any]], None]:
        return list(self.markets), None

    def get_market(self, ticker: str) -> dict[str, Any]:
        return next(m for m in self.markets if m["ticker"] == ticker)

    def get_series(self, _: str) -> dict[str, Any]:
        return {"fee_type": "quadratic", "fee_multiplier": 1}

    def get_held_tickers(self) -> set[str]:
        if isinstance(self.held, Exception):
            raise self.held
        return set(self.held)

    def get_exposure_by_ticker(self) -> dict[str, float]:
        return dict(self.exposure)

    def place_order(self, **kwargs: Any) -> dict[str, Any]:
        if self.order_error:
            raise self.order_error
        self.orders.append(kwargs)
        count = kwargs["count"] if self.fill is None else self.fill
        return {"order": {"order_id": "o1", "fill_count": count}}

    def sell_to_close(self, **kwargs: Any) -> dict[str, Any]:
        self.sells.append(kwargs)
        return {"order": {"order_id": "s1", "fill_count": kwargs["count"]}}


def _strategy(markets, reading=None, *, live=False, **cfg):
    cfg.setdefault("enabled", True)
    kalshi = FakeKalshi(markets)
    nowcast = FakeNowcast(_reading() if reading is None else reading)
    strat = GdpNowcastStrategy(config=GdpNowcastConfig(**cfg), kalshi_client=kalshi,  # type: ignore[arg-type]
                               nowcast_client=nowcast, logger=_LOG, live_trading_enabled=live)  # type: ignore[arg-type]
    return strat, kalshi, nowcast


def _run(strat, state=None, *, live=False):
    state = state or (PortfolioState(cash=0.0, cash_reserves={"kalshi": 1000.0}) if live
                      else PortfolioState(cash=1000.0))
    broker = PaperBroker(not live, _LOG)
    result = strat.run(state, broker=broker, risk_manager=RiskManager(RiskConfig()), now=_NOW)
    return state, broker, result


class ParsingTests(unittest.TestCase):
    def test_last_usable_row_wins_and_bad_rows_are_skipped(self):
        text = ("observation_date,GDPNOW\n2026-04-01,2.1\n2026-07-01,3.7\n2026-10-01,.\nnot-a-date,9\n"
                "2027-01-01,99\n")
        self.assertEqual(parse_fred_csv(text), (__import__("datetime").date(2026, 7, 1), 3.7))

    def test_old_header_and_empty_input(self):
        self.assertEqual(parse_fred_csv("DATE,GDPNOW\n2026-07-01,1.5\n")[1], 1.5)
        self.assertIsNone(parse_fred_csv(""))
        self.assertIsNone(parse_fred_csv("DATE,GDPNOW\n2026-07-01,.\n"))

    def test_quarter_labels(self):
        import datetime
        self.assertEqual(quarter_label(datetime.date(2026, 7, 1)), "2026Q3")
        self.assertEqual(quarter_label(datetime.date(2027, 1, 1)), "2027Q1")

    def test_market_title_gives_the_quarter(self):
        self.assertEqual(market_quarter(_market(quarter="Q4 2026")), "2026Q4")
        self.assertEqual(market_quarter({"title": "something else"}), "")


class FakeResponse:
    def __init__(self, text: str, status: int = 200) -> None:
        self.text, self.status = text, status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class ClientTests(unittest.TestCase):
    def _client(self, responses, max_age=48.0):
        calls = iter(responses)

        def get(*_a, **_k):
            item = next(calls)
            if isinstance(item, Exception):
                raise item
            return item
        return GdpNowClient(url="http://x", timeout_seconds=5, max_age_hours=max_age, logger=_LOG, http_get=get)

    def test_reads_the_latest_quarter(self):
        client = self._client([FakeResponse("DATE,GDPNOW\n2026-07-01,3.7\n")])
        reading = client.latest(now=1000.0)
        self.assertEqual((reading.quarter, reading.value), ("2026Q3", 3.7))

    def test_falls_back_to_the_last_good_reading_then_gives_up(self):
        client = self._client([FakeResponse("DATE,GDPNOW\n2026-07-01,3.7\n"), RuntimeError("down"),
                               RuntimeError("down")], max_age=2.0)
        client.latest(now=0.0)
        self.assertEqual(client.latest(now=3600.0).value, 3.7)      # 1 hour old: still fine
        self.assertIsNone(client.latest(now=3 * 3600.0))            # 3 hours old: too stale to bet on
        self.assertIn("RuntimeError", client.last_error)

    def test_every_quarter_in_the_file_can_be_looked_up(self):
        client = self._client([FakeResponse("DATE,GDPNOW\n2026-04-01,2.1\n2026-07-01,3.7\n")])
        client.latest(now=0.0)
        self.assertEqual(client.for_quarter("2026Q2", now=60.0).value, 2.1)
        self.assertEqual(client.for_quarter("2026Q3", now=60.0).value, 3.7)
        self.assertIsNone(client.for_quarter("2026Q4", now=60.0))
        self.assertIsNone(client.for_quarter("2026Q3", now=10 * 3600.0 * 10))     # too old

    def test_http_error_or_garbage_with_no_history_means_no_reading(self):
        self.assertIsNone(self._client([FakeResponse("", 500)]).latest())
        self.assertIsNone(self._client([FakeResponse("<html>nope</html>")]).latest())


class MathTests(unittest.TestCase):
    def test_probability_is_half_at_the_nowcast_and_falls_with_the_strike(self):
        self.assertAlmostEqual(prob_above(3.0, 3.0, 1.2), 0.5)
        self.assertGreater(prob_above(2.0, 3.0, 1.2), prob_above(4.0, 3.0, 1.2))
        self.assertAlmostEqual(prob_above(3.5, 4.0, 1.2), 0.662, places=2)

    def test_wider_error_pulls_probabilities_toward_half_and_tails_are_clamped(self):
        self.assertLess(prob_above(5.0, 3.0, 3.0), prob_above(5.0, 3.0, 1.0) + 0.5)
        self.assertGreater(prob_above(5.0, 3.0, 3.0), prob_above(5.0, 3.0, 1.0))
        self.assertEqual(prob_above(20.0, 3.0, 0.5), 0.01)
        self.assertEqual(prob_above(-20.0, 3.0, 0.5), 0.99)

    def test_error_grows_with_lead_time_after_the_free_days(self):
        cfg = GdpNowcastConfig()
        self.assertEqual(lead_sd(cfg, 3), cfg.base_sd)
        self.assertEqual(lead_sd(cfg, 7), cfg.base_sd)
        self.assertAlmostEqual(lead_sd(cfg, 27), cfg.base_sd + 0.3)

    def test_kelly_needs_an_edge_and_is_fractional(self):
        self.assertEqual(kelly_stake(0.40, 0.40, 0.25), 0.0)
        self.assertEqual(kelly_stake(0.30, 0.40, 0.25), 0.0)
        self.assertAlmostEqual(kelly_stake(0.60, 0.40, 0.25), 0.25 * 0.20 / 0.60)


class EvaluateTests(unittest.TestCase):
    cfg = GdpNowcastConfig(enabled=True)

    def _eval(self, market, value=4.0):
        return evaluate_market(market, _reading(value), 10.0, self.cfg)

    def test_model_above_the_price_is_a_yes_bet(self):
        row = self._eval(_market(3.5, 0.36, 0.37), 4.0)
        self.assertEqual((row["side"], row["reason"]), ("yes", ""))
        self.assertGreater(row["edge"], self.cfg.min_edge)
        self.assertLess(row["p_blend"], row["p_model"])            # shrunk toward the market

    def test_model_below_the_price_is_a_no_bet(self):
        row = self._eval(_market(3.0, 0.63, 0.65), 2.6)
        self.assertEqual((row["side"], row["reason"]), ("no", ""))

    def test_a_small_gap_is_not_enough_after_the_fee_and_shrinkage(self):
        row = self._eval(_market(3.5, 0.50, 0.51), 3.6)
        self.assertEqual(row["reason"], "edge_too_small")

    def test_guards_that_fail_closed(self):
        self.assertEqual(self._eval(_market(3.5, 0.30, 0.40))["reason"], "wide_spread")
        self.assertEqual(self._eval(_market(3.5, 0.36, 0.37), 7.0)["reason"], "model_market_gap")
        self.assertEqual(self._eval(_market(3.5, 0.0, 0.37))["reason"], "no_two_sided_book")
        self.assertEqual(self._eval(_market(3.5, strike_type="between"))["reason"], "unsupported_market")
        # both sides outside the 5c-95c band: never bet an extreme tail on a normal model
        self.assertEqual(self._eval(_market(0.0, 0.98, 0.99), 4.0)["reason"], "price_out_of_range")


class StrategyTests(unittest.TestCase):
    def test_disabled_does_nothing(self):
        strat, kalshi, _ = _strategy([_market()], enabled=False)
        _, _, result = _run(strat)
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(kalshi.orders, [])

    def test_no_nowcast_means_no_bets(self):
        strat, kalshi, nowcast = _strategy([_market()])
        nowcast.reading = None
        state, _, result = _run(strat)
        self.assertEqual((result["status"], result["buys"], state.positions), ("no_nowcast", [], {}))

    def test_paper_mode_books_an_edge_bet_with_the_reasoning_attached(self):
        strat, kalshi, _ = _strategy([_market(3.5, 0.36, 0.37)])
        state, _, result = _run(strat)
        self.assertEqual(result["buys"], ["KX:KXGDP-26OCT30-T3.5:yes"])
        pos = state.positions["KX:KXGDP-26OCT30-T3.5:yes"]
        self.assertEqual((pos.strategy, pos.asset_type, pos.meta["venue"]), ("gdp_nowcast", "prediction", "kalshi"))
        self.assertEqual(pos.meta["nowcast"], 4.0)
        self.assertEqual(kalshi.orders, [])                         # paper: nothing sent to Kalshi
        self.assertLessEqual(pos.quantity * pos.entry_price, 1000.0 * 0.05 + 0.01)   # per-bet cap

    def test_a_nowcast_for_another_quarter_places_nothing(self):
        strat, _, _ = _strategy([_market()], _reading(quarter="2026Q4"))
        state, _, result = _run(strat)
        self.assertEqual((result["buys"], state.positions), ([], {}))
        self.assertEqual(result["events"][0]["reason"], "nowcast_is_for_another_quarter")

    def test_each_market_uses_its_own_quarters_row_even_if_a_later_quarter_is_listed_last(self):
        strat, _, nowcast = _strategy([_market(3.5, 0.36, 0.37)], _reading(2.0, "2026Q4"))
        nowcast.by_quarter = {"2026Q3": _reading(4.0, "2026Q3")}
        state, _, result = _run(strat)
        self.assertEqual(result["buys"], ["KX:KXGDP-26OCT30-T3.5:yes"])
        self.assertEqual(state.positions[result["buys"][0]].meta["nowcast"], 4.0)
        self.assertEqual(result["events"][0]["nowcast"], 4.0)

    def test_results_are_plain_json(self):
        import json
        strat, _, _ = _strategy([_market(3.5, 0.36, 0.37)])
        _, _, result = _run(strat)
        json.dumps(result)                                          # no private objects leak into the summary

    def test_too_far_from_release_places_nothing(self):
        far = _market(release="2026-12-30T14:00:00Z", close="2026-12-30T12:29:00Z")
        strat, _, _ = _strategy([far])
        _, _, result = _run(strat)
        self.assertEqual((result["buys"], result["events"][0]["reason"]), ([], "too_far_from_release"))

    def test_at_most_two_strikes_per_quarter(self):
        markets = [_market(3.0, 0.45, 0.46), _market(3.5, 0.36, 0.37), _market(4.0, 0.22, 0.23)]
        strat, _, _ = _strategy(markets, _reading(4.0), max_position_pct=0.2, capital_allocation_pct=0.9)
        state, _, result = _run(strat)
        self.assertLessEqual(len(result["buys"]), 2)
        self.assertEqual(len(state.positions), len(result["buys"]))

    def test_live_order_is_an_immediate_or_cancel_at_the_ask_and_books_only_the_fill(self):
        strat, kalshi, _ = _strategy([_market(3.5, 0.36, 0.37)], live=True)
        kalshi.fill = 2
        state, _, result = _run(strat, live=True)
        order = kalshi.orders[0]
        self.assertEqual((order["side"], order["action"], order["price_cents"], order["time_in_force"]),
                         ("yes", "buy", 37, "immediate_or_cancel"))
        self.assertEqual(state.positions[result["buys"][0]].quantity, 2)

    def test_failed_or_unfilled_orders_book_nothing(self):
        strat, kalshi, _ = _strategy([_market()], live=True)
        kalshi.order_error = RuntimeError("HTTP 404: insufficient_shard_balance")
        state, _, result = _run(strat, live=True)
        self.assertEqual(state.positions, {})
        self.assertIn("order_failed", [d["reason"] for d in result["decisions"]])
        strat2, kalshi2, _ = _strategy([_market()], live=True)
        kalshi2.fill = 0
        state2, _, result2 = _run(strat2, live=True)
        self.assertEqual(state2.positions, {})
        self.assertIn("order_not_filled", [d["reason"] for d in result2["decisions"]])

    def test_unreadable_account_places_nothing(self):
        strat, kalshi, _ = _strategy([_market()], live=True)
        kalshi.held = RuntimeError("down")
        state, _, result = _run(strat, live=True)
        self.assertEqual((kalshi.orders, state.positions), ([], {}))
        self.assertIn("positions_unavailable", [d["reason"] for d in result["decisions"]])

    def test_a_strike_the_account_already_holds_is_not_bought_again(self):
        strat, kalshi, _ = _strategy([_market(3.5, 0.36, 0.37)], live=True)
        kalshi.held = {"KXGDP-26OCT30-T3.5"}
        _, _, result = _run(strat, live=True)
        self.assertEqual((kalshi.orders, result["buys"]), ([], []))

    def test_gdp_exposure_already_on_the_account_uses_up_the_budget(self):
        strat, kalshi, _ = _strategy([_market()], live=True)
        kalshi.exposure = {"KXGDP-27JAN28-T3.5": 400.0, "KXCPI-26OCT-T3": 999.0}   # 400 of a 150 budget
        _, _, result = _run(strat, live=True)
        self.assertEqual((kalshi.orders, result["buys"]), ([], []))

    def test_kill_switch_stops_new_bets_but_still_reports(self):
        strat, kalshi, _ = _strategy([_market()])
        state = PortfolioState(cash=1000.0, kill_switch_engaged=True)
        _, _, result = _run(strat, state)
        self.assertEqual(result["buys"], [])
        self.assertIn("risk_guardrails", [d["reason"] for d in result["decisions"]])
        self.assertTrue(result["events"])

    def test_resolved_markets_close_at_one_or_zero(self):
        strat, kalshi, _ = _strategy([_market()])
        state, broker, result = _run(strat)
        symbol = result["buys"][0]
        kalshi.markets[0].update(status="finalized", result="yes")
        _, _, again = _run(strat, state)
        self.assertEqual(again["resolved_closed"], [symbol])
        self.assertNotIn(symbol, state.positions)

    def test_sells_early_only_when_the_bid_pays_more_than_our_value(self):
        strat, kalshi, _ = _strategy([_market(3.5, 0.36, 0.37)])
        state, broker, result = _run(strat)
        symbol = result["buys"][0]
        # market runs far above our value of about 0.5: bid 0.80 beats value + fee + margin
        kalshi.markets[0].update(yes_bid_dollars="0.8000", yes_ask_dollars="0.8100",
                                 no_bid_dollars="0.1900", no_ask_dollars="0.2000")
        _, _, after = _run(strat, state)
        self.assertEqual(after["sells"], [symbol])
        # but a bid near our value is held
        strat2, kalshi2, _ = _strategy([_market(3.5, 0.36, 0.37)])
        state2, _, r2 = _run(strat2)
        kalshi2.markets[0].update(yes_bid_dollars="0.5000", yes_ask_dollars="0.5100",
                                  no_bid_dollars="0.4900", no_ask_dollars="0.5000")
        _, _, held = _run(strat2, state2)
        self.assertEqual(held["sells"], [])

    def test_preview_reads_without_placing_orders(self):
        strat, kalshi, _ = _strategy([_market(3.5, 0.36, 0.37)], live=True, enabled=False)
        out = strat.preview(_NOW)
        self.assertEqual((out["status"], out["enabled"], kalshi.orders), ("ok", False, []))
        self.assertEqual(out["events"][0]["rows"][0]["side"], "yes")


class FormatTests(unittest.TestCase):
    def test_table_shows_the_bet_and_the_reasons(self):
        strat, _, _ = _strategy([_market(3.5, 0.36, 0.37), _market(4.0, 0.22, 0.40)])
        text = _format_gdp(strat.preview(_NOW))
        self.assertIn("GDPNow: 4.00% for 2026Q3", text)
        self.assertIn("BUY YES at 37c", text)
        self.assertIn("bid-ask spread too wide", text)
        self.assertIn("Trading: OFF", _format_gdp({**strat.preview(_NOW), "enabled": False}))

    def test_thin_edge_shows_the_edge_against_the_requirement(self):
        result = {"nowcast": {"value": 3.59, "quarter": "2026Q3", "age_hours": 0.0}, "min_edge": 0.07,
                  "events": [{"quarter": "2026Q3", "rows": [{
                      "strike": 3.5, "yes_bid": 0.35, "yes_ask": 0.36, "p_model": 0.53, "p_blend": 0.44,
                      "reason": "edge_too_small", "edge": 0.06}]}]}
        text = _format_gdp(result)
        self.assertIn("edge too small (+6.0c after fee, needs 7c)", text)

    def test_no_nowcast_says_so(self):
        text = _format_gdp({"nowcast": None, "error": "HTTP 500"})
        self.assertIn("No nowcast available (HTTP 500)", text)


class ConfigTests(unittest.TestCase):
    def test_off_by_default_with_conservative_settings(self):
        cfg = GdpNowcastConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual((cfg.model_weight, cfg.kelly_fraction, cfg.max_position_pct), (0.5, 0.25, 0.05))
        self.assertLessEqual(cfg.capital_allocation_pct, 0.15)


if __name__ == "__main__":
    unittest.main()
