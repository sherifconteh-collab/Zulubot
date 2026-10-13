from __future__ import annotations

import logging
import math
import random
import unittest
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.config import CryptoMarketsConfig, RiskConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.strategies.crypto_markets import (
    CryptoMarketsStrategy,
    fit_ladder,
    price_cdf,
    realized_sigma,
)
from trading_bot.telegram_bot import _format_crypto_markets

_LOG = logging.getLogger("test_crypto")
_NOW = datetime(2026, 10, 13, 17, 30, tzinfo=timezone.utc)      # 90 minutes before the 3pm EDT settlement
_CLOSE = "2026-10-13T19:00:00Z"
_SIGMA_5M = 0.0012                                              # true 5-minute spread of the fake candles
_STRIKES = [92000 + 100 * i for i in range(-12, 13)]


def _candles(sigma: float = _SIGMA_5M, last_age_s: int = 300, n: int = 288, price: float = 92_300.0):
    rng = random.Random(3)
    closes = [price]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0.0, sigma)))
    end = int(_NOW.timestamp()) - last_age_s
    rows = [{"start": str(end - 300 * (n - 1 - i)), "close": f"{c:.2f}"} for i, c in enumerate(closes)]
    return list(reversed(rows)), closes[-1]


class FakeCoinbase:
    def __init__(self, spot: float = 92_300.0, sigma: float = _SIGMA_5M, last_age_s: int = 300) -> None:
        self.spot = spot
        self.candles, _ = _candles(sigma, last_age_s)
        self.fail = False

    def get_spot_price_by_product_id(self, product: str) -> float:
        if self.fail:
            raise RuntimeError("down")
        return self.spot

    def get_candles(self, symbol: str, granularity: str = "", limit: int = 0):
        if self.fail:
            raise RuntimeError("down")
        return self.candles


def _ladder(center: float, sd: float, *, event: str = "KXBTCD-26OCT1315", close: str = _CLOSE,
            spread: float = 0.02, size: float = 500.0) -> list[dict[str, Any]]:
    """A greater-than ladder priced the way a market maker with spread ``sd`` around ``center`` would."""
    out = []
    for strike in _STRIKES:
        mid = 1.0 - NormalDist().cdf((math.log(strike / center) + 0.5 * sd * sd) / sd)
        bid = max(0.01, round(mid - spread / 2, 2))
        ask = min(0.99, round(mid + spread / 2, 2))
        out.append({
            "ticker": f"{event}-T{strike}", "event_ticker": event, "status": "active",
            "title": f"Bitcoin price on Oct 13, 2026? ${strike:,} or above", "strike_type": "greater",
            "floor_strike": float(strike), "cap_strike": None, "close_time": close,
            "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}",
            "no_bid_dollars": f"{1 - ask:.4f}", "no_ask_dollars": f"{1 - bid:.4f}",
            "yes_ask_size_fp": f"{size:.2f}", "no_ask_size_fp": f"{size:.2f}",
        })
    return out


class FakeKalshi:
    def __init__(self, markets: list[dict[str, Any]]) -> None:
        self.markets = markets
        self.orders: list[dict[str, Any]] = []
        self.held: set[str] = set()

    def get_markets(self, **kwargs: Any):
        series = kwargs.get("series_ticker")
        return [m for m in self.markets if m["ticker"].startswith(f"{series}-")], None

    def get_market(self, ticker: str) -> dict[str, Any]:
        return next(m for m in self.markets if m["ticker"] == ticker)

    def get_series(self, _: str) -> dict[str, Any]:
        return {"fee_type": "quadratic", "fee_multiplier": 1}

    def get_held_tickers(self) -> set[str]:
        return set(self.held)

    def get_exposure_by_ticker(self) -> dict[str, float]:
        return {}

    def place_order(self, **kwargs: Any) -> dict[str, Any]:
        self.orders.append(kwargs)
        return {"order": {"order_id": "o", "fill_count": kwargs["count"]}}

    def sell_to_close(self, **kwargs: Any) -> dict[str, Any]:
        return {"order": {"order_id": "s", "fill_count": kwargs["count"]}}


def _model_sd(minutes: float = 90.0, sigma: float = _SIGMA_5M, cfg: CryptoMarketsConfig | None = None) -> float:
    cfg = cfg or CryptoMarketsConfig()
    realized = max(sigma, cfg.min_sigma_5m) * cfg.vol_multiplier
    return math.sqrt((realized * math.sqrt(minutes / 5.0)) ** 2 + cfg.basis_sd ** 2)


def _strategy(markets, *, live=False, coinbase=None, **cfg):
    cfg.setdefault("enabled", True)
    kalshi = FakeKalshi(markets)
    coinbase = coinbase or FakeCoinbase()
    strat = CryptoMarketsStrategy(config=CryptoMarketsConfig(**cfg), kalshi_client=kalshi,  # type: ignore[arg-type]
                                  coinbase_client=coinbase, logger=_LOG, live_trading_enabled=live)
    return strat, kalshi, coinbase


def _run(strat, *, live=False):
    state = (PortfolioState(cash=0.0, cash_reserves={"kalshi": 1000.0}) if live else PortfolioState(cash=1000.0))
    broker = PaperBroker(not live, _LOG)
    return state, strat.run(state, broker=broker, risk_manager=RiskManager(RiskConfig()), now=_NOW)


class MathTests(unittest.TestCase):
    def test_realized_sigma_recovers_the_spread_and_needs_enough_data(self):
        rows, _ = _candles(0.002)
        closes = [float(r["close"]) for r in reversed(rows)]
        self.assertAlmostEqual(realized_sigma(closes), 0.002, delta=0.0003)
        self.assertIsNone(realized_sigma(closes[:5]))

    def test_cdf_is_monotone_centered_and_handles_edges(self):
        self.assertLess(price_cdf(90_000, 92_000, 0.01), price_cdf(94_000, 92_000, 0.01))
        self.assertAlmostEqual(price_cdf(92_000, 92_000, 0.01), 0.5, delta=0.01)
        self.assertEqual(price_cdf(0, 92_000, 0.01), 0.0)
        self.assertEqual(price_cdf(95_000, 92_000, 0.0), 1.0)

    def test_ladder_fit_recovers_the_center_and_spread_it_was_built_from(self):
        center, sd = fit_ladder(_ladder(92_300, 0.006))
        self.assertAlmostEqual(center, 92_300, delta=60)
        self.assertAlmostEqual(sd, 0.006, delta=0.0008)

    def test_ladder_fit_works_without_our_price_so_a_stale_ladder_is_measured_as_stale(self):
        center, sd = fit_ladder(_ladder(91_000, 0.006))
        self.assertAlmostEqual(center, 91_000, delta=60)

    def test_ladder_fit_needs_mid_range_prices(self):
        self.assertIsNone(fit_ladder(_ladder(92_300, 0.0001)))              # everything is 1c or 99c
        self.assertIsNone(fit_ladder([]))


class EventGuardTests(unittest.TestCase):
    def _reason(self, markets, **kwargs):
        strat, _, _ = _strategy(markets, **kwargs)
        out = strat.preview(_NOW)
        return out, [e["reason"] for e in out["events"]]

    def test_a_consistent_ladder_is_open_for_betting(self):
        _, reasons = self._reason(_ladder(92_300, _model_sd()))
        self.assertEqual(reasons, [""])

    def test_settling_too_soon_or_too_late_is_skipped(self):
        _, soon = self._reason(_ladder(92_300, 0.003, close="2026-10-13T17:40:00Z"))
        _, late = self._reason(_ladder(92_300, 0.01, close="2026-10-14T05:00:00Z"))
        self.assertEqual((soon, late), (["too_close_to_close"], ["too_far_from_close"]))

    def test_model_and_market_volatility_that_disagree_means_no_bet(self):
        _, reasons = self._reason(_ladder(92_300, _model_sd() * 4))
        self.assertEqual(reasons, ["volatility_mismatch"])

    def test_a_ladder_with_no_mid_range_prices_cannot_be_checked(self):
        _, reasons = self._reason(_ladder(92_300, 0.0002))
        self.assertEqual(reasons, ["cannot_check_volatility"])

    def test_a_price_far_outside_the_ladder_is_treated_as_a_bad_feed(self):
        strat, _, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=FakeCoinbase(spot=96_000))
        self.assertEqual([e["reason"] for e in strat.preview(_NOW)["events"]], ["price_far_from_market"])

    def test_no_price_history_or_stale_candles_means_no_bets_at_all(self):
        for coinbase in (FakeCoinbase(last_age_s=3600), ):
            strat, kalshi, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=coinbase)
            state, result = _run(strat)
            self.assertEqual((result["status"], result["buys"], state.positions), ("no_spot", [], {}))
        down = FakeCoinbase()
        down.fail = True
        strat, _, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=down)
        self.assertEqual(_run(strat)[1]["status"], "no_spot")


class BettingTests(unittest.TestCase):
    def test_a_market_priced_where_the_model_agrees_places_nothing(self):
        coinbase = FakeCoinbase(spot=92_300)
        strat, kalshi, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=coinbase)
        state, result = _run(strat)
        self.assertEqual((result["buys"], state.positions, kalshi.orders), ([], {}, []))

    def test_a_stale_ladder_after_a_price_move_gets_bet_the_right_way(self):
        # The market still centers on 92,300 while Coinbase has moved up to 92,900: YES on strikes
        # around the move is underpriced.
        coinbase = FakeCoinbase(spot=92_900)
        strat, kalshi, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=coinbase)
        state, result = _run(strat)
        self.assertTrue(result["buys"], result["decisions"])
        position = state.positions[result["buys"][0]]
        self.assertEqual((position.strategy, position.asset_type, position.meta["side"]), ("crypto_markets", "prediction", "yes"))
        self.assertEqual(position.meta["spot"], 92_900)
        self.assertLessEqual(len(result["buys"]), 2)                      # per-event cap

    def test_the_mirror_case_buys_no(self):
        coinbase = FakeCoinbase(spot=91_700)
        strat, _, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=coinbase)
        state, result = _run(strat)
        self.assertTrue(result["buys"])
        self.assertEqual(state.positions[result["buys"][0]].meta["side"], "no")

    def test_live_orders_are_immediate_or_cancel_at_the_ask_and_capped(self):
        coinbase = FakeCoinbase(spot=92_900)
        strat, kalshi, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=coinbase, live=True)
        state, result = _run(strat, live=True)
        self.assertTrue(kalshi.orders)
        order = kalshi.orders[0]
        self.assertEqual((order["action"], order["time_in_force"]), ("buy", "immediate_or_cancel"))
        spent = sum(p.quantity * p.entry_price for p in state.positions.values())
        self.assertLessEqual(spent, 1000.0 * 0.06 + 1.0)                  # total allocation cap
        for position in state.positions.values():
            self.assertLessEqual(position.quantity * position.entry_price, 1000.0 * 0.02 + 1.0)

    def test_nothing_is_bet_when_disabled_or_halted(self):
        strat, kalshi, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=FakeCoinbase(spot=92_900), enabled=False)
        self.assertEqual(_run(strat)[1]["status"], "disabled")
        strat, kalshi, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=FakeCoinbase(spot=92_900))
        state = PortfolioState(cash=1000.0, halt_new_trades_for_day=True)
        result = strat.run(state, broker=PaperBroker(True, _LOG), risk_manager=RiskManager(RiskConfig()), now=_NOW)
        self.assertEqual((result["buys"], state.positions), ([], {}))

    def test_a_held_strike_is_not_bought_again(self):
        coinbase = FakeCoinbase(spot=92_900)
        markets = _ladder(92_300, _model_sd())
        strat, kalshi, _ = _strategy(markets, coinbase=coinbase, live=True)
        _, first = _run(strat, live=True)
        kalshi.held = {t.split("KX:")[1].rsplit(":", 1)[0] for t in first["buys"]}
        kalshi.orders.clear()
        _, second = _run(strat, live=True)
        for order in kalshi.orders:
            self.assertNotIn(order["ticker"], kalshi.held)


class FormatTests(unittest.TestCase):
    def test_table_shows_prices_the_check_and_the_verdicts(self):
        strat, _, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=FakeCoinbase(spot=92_900))
        text = _format_crypto_markets(strat.preview(_NOW))
        self.assertIn("BTC-USD: $92,900.00", text)
        self.assertIn("market implies", text)
        self.assertIn("BUY", text)

    def test_no_prices_says_so(self):
        down = FakeCoinbase()
        down.fail = True
        strat, _, _ = _strategy(_ladder(92_300, _model_sd()), coinbase=down)
        self.assertIn("No Coinbase prices available", _format_crypto_markets(strat.preview(_NOW)))


if __name__ == "__main__":
    unittest.main()
