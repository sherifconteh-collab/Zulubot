"""GDP markets on Kalshi, traded only when a forecast-based probability beats the price.

The old predictions rule bought whatever was cheap. This strategy has an actual view: the Atlanta
Fed's GDPNow estimate for the quarter, turned into a probability for each Kalshi strike ("GDP above
3.5%") using the nowcast's known forecast error. It then asks whether that probability, averaged
with the market's own price, still beats the price you would pay after Kalshi's fee. Every guard
fails closed: no nowcast, an unreadable account, a wide book or a model that disagrees wildly with
the market all mean no bet.

Per Kalshi contract bought at price ``c`` (fee included) with win probability ``p``, the stake
follows fractional Kelly: ``f = kelly * (p - c) / (1 - c)``, capped per bet and in total.
"""

from __future__ import annotations

import logging
import math
import re
import time
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients_kalshi import KalshiClient, order_fill_count
from trading_bot.clients_nowcast import GdpNowClient, NowcastReading, iso_to_epoch
from trading_bot.config import GdpNowcastConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.sports_model import kalshi_fee

STRATEGY = "gdp_nowcast"
_CASH_BUCKET = "kalshi"
_RESOLVED = {"settled", "finalized", "determined"}
_QUARTER = re.compile(r"\bQ([1-4])\s+(\d{4})\b")
_FEE_RATE = 0.07
_MAX_ORDER_FAILURES = 3


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def market_quarter(market: dict[str, Any]) -> str:
    """``Will real GDP increase by more than 3.5% in Q3 2026?`` -> ``2026Q3`` ('' if absent)."""
    found = _QUARTER.search(str(market.get("title", "")))
    return f"{found.group(2)}Q{found.group(1)}" if found else ""


def lead_sd(cfg: GdpNowcastConfig, days_to_release: float) -> float:
    """Forecast error (percentage points) of the nowcast at this lead time."""
    return cfg.base_sd + cfg.sd_per_day * max(0.0, days_to_release - cfg.sd_free_days)


def prob_above(strike: float, nowcast: float, sd: float) -> float:
    """P(advance GDP growth > strike) if growth is normal around the nowcast."""
    if sd <= 0:
        return 1.0 if nowcast > strike else 0.0
    return min(0.99, max(0.01, 0.5 * math.erfc((strike - nowcast) / (sd * math.sqrt(2.0)))))


def kelly_stake(p: float, cost: float, fraction: float) -> float:
    """Share of bankroll to stake on a $1 binary contract costing ``cost`` (fee included)."""
    if cost <= 0 or cost >= 1 or p <= cost:
        return 0.0
    return max(0.0, fraction * (p - cost) / (1.0 - cost))


def evaluate_market(
    market: dict[str, Any],
    reading: NowcastReading,
    days_to_release: float,
    cfg: GdpNowcastConfig,
    fee_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Score one strike. ``reason`` is empty when it is a bet, otherwise why it is not."""
    row: dict[str, Any] = {
        "ticker": str(market.get("ticker", "")), "strike": None, "yes_bid": 0.0, "yes_ask": 0.0,
        "p_market": None, "p_model": None, "p_blend": None, "side": "", "price": 0.0, "fee": 0.0,
        "edge": None, "ask_size": 0.0, "reason": "",
    }
    if str(market.get("strike_type", "")).lower() != "greater" or market.get("floor_strike") in (None, ""):
        row["reason"] = "unsupported_market"
        return row
    strike = _num(market.get("floor_strike"))
    yes_bid, yes_ask = _num(market.get("yes_bid_dollars")), _num(market.get("yes_ask_dollars"))
    no_ask = _num(market.get("no_ask_dollars"))
    row.update(strike=strike, yes_bid=yes_bid, yes_ask=yes_ask)
    if yes_bid <= 0 or yes_ask <= 0 or no_ask <= 0 or yes_ask < yes_bid:
        row["reason"] = "no_two_sided_book"
        return row
    p_market = (yes_bid + yes_ask) / 2.0
    p_model = prob_above(strike, reading.value, lead_sd(cfg, days_to_release))
    weight = min(1.0, max(0.0, cfg.model_weight))
    p_blend = min(0.98, max(0.02, weight * p_model + (1.0 - weight) * p_market))
    row.update(p_market=p_market, p_model=p_model, p_blend=p_blend)
    if yes_ask - yes_bid > cfg.max_spread:
        row["reason"] = "wide_spread"
        return row
    if abs(p_model - p_market) > cfg.max_model_market_gap:
        row["reason"] = "model_market_gap"
        return row

    best: tuple[float, str, float, float, float] | None = None   # edge, side, price, fee, p_side
    for side, price, p_side, depth_key in (
        ("yes", yes_ask, p_blend, "yes_ask_size_fp"),
        ("no", no_ask, 1.0 - p_blend, "no_ask_size_fp"),
    ):
        if not cfg.min_price <= price <= cfg.max_price:
            continue
        fee = kalshi_fee(price, _FEE_RATE, fee_multiplier)
        edge = p_side - price - fee
        if best is None or edge > best[0]:
            best = (edge, side, price, fee, p_side)
            row["ask_size"] = _num(market.get(depth_key))
    if best is None:
        row["reason"] = "price_out_of_range"
        return row
    edge, side, price, fee, _p_side = best
    row.update(side=side, price=price, fee=fee, edge=edge)
    if edge < cfg.min_edge:
        row["reason"] = "edge_too_small"
    return row


def _public(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events as plain data for results and Telegram: drops the private objects used while deciding."""
    return [
        {**{k: v for k, v in e.items() if not k.startswith("_")},
         "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in e["rows"]]}
        for e in events
    ]


class GdpNowcastStrategy:
    def __init__(
        self,
        *,
        config: GdpNowcastConfig,
        kalshi_client: KalshiClient,
        nowcast_client: GdpNowClient,
        logger: logging.Logger,
        live_trading_enabled: bool = False,
    ) -> None:
        self.config = config
        self.kalshi = kalshi_client
        self.nowcast = nowcast_client
        self.logger = logger.getChild("gdp_nowcast")
        self.live_trading_enabled = live_trading_enabled
        self._fee_multiplier: float | None = None

    # ------------------------------------------------------------------ run

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "ok", "buys": [], "sells": [], "resolved_closed": [], "decisions": [],
            "events": [], "nowcast": None, "can_trade": self.live_trading_enabled,
        }
        if not self.config.enabled:
            result["status"] = "disabled"
            return result
        try:
            reading = self.nowcast.latest()
            if reading is None:
                result["status"] = "no_nowcast"
                result["error"] = self.nowcast.last_error or "no GDP nowcast available"
                self._skip(result, "*", "no_nowcast", error=result["error"])
            else:
                result["nowcast"] = {
                    "quarter": reading.quarter, "value": reading.value, "source": reading.source,
                    "age_hours": round(reading.age_hours(), 1),
                }
            self._manage_positions(state, broker, reading, now, result)
            if reading is None:
                return result
            if state.kill_switch_engaged or state.halt_new_trades_for_day:
                self._skip(result, "*", "risk_guardrails",
                           kill_switch=state.kill_switch_engaged, daily_halt=state.halt_new_trades_for_day)
                result["events"] = _public(self._score_events(self._discover(), reading, now))
                return result
            self._enter(state, broker, reading, now, result)
        except Exception as exc:
            self.logger.error("GDP nowcast cycle failed: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)
        return result

    def preview(self, now: datetime) -> dict[str, Any]:
        """Read-only view of model versus market for every open strike. Places no orders."""
        out: dict[str, Any] = {
            "status": "ok", "enabled": self.config.enabled, "events": [], "nowcast": None,
            "buys": [], "sells": [], "decisions": [],
        }
        try:
            reading = self.nowcast.latest()
            if reading is None:
                out["status"] = "no_nowcast"
                out["error"] = self.nowcast.last_error or "no GDP nowcast available"
                return out
            out["nowcast"] = {
                "quarter": reading.quarter, "value": reading.value, "source": reading.source,
                "age_hours": round(reading.age_hours(), 1),
            }
            out["events"] = _public(self._score_events(self._discover(), reading, now))
        except Exception as exc:
            self.logger.error("GDP preview failed: %s", exc, exc_info=True)
            out["status"] = "error"
            out["error"] = str(exc)
        return out

    # ------------------------------------------------------------ discovery

    def _discover(self) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(3):
            page, cursor = self.kalshi.get_markets(
                series_ticker=self.config.series, status="open", limit=100, cursor=cursor,
            )
            markets.extend(page)
            if not cursor:
                break
        return [m for m in markets if str(m.get("status", "active")).lower() in ("active", "open")]

    def _fee_mult(self) -> float:
        if self._fee_multiplier is None:
            multiplier = 1.0
            try:
                info = self.kalshi.get_series(self.config.series)
                fee_type = str(info.get("fee_type", "")).lower()
                raw = info.get("fee_multiplier")
                if "quadratic" in fee_type and raw not in (None, ""):
                    multiplier = max(0.0, float(raw))
            except Exception as exc:
                self.logger.debug("Could not read fee info for %s (assuming standard fee): %s",
                                  self.config.series, exc)
            self._fee_multiplier = multiplier
        return self._fee_multiplier

    def _event_context(
        self, event_markets: list[dict[str, Any]], reading: NowcastReading, now: datetime,
    ) -> dict[str, Any]:
        """Timing and quarter facts for one event; ``reason`` says why the whole event is skipped."""
        first = event_markets[0]
        quarter = market_quarter(first)
        release = iso_to_epoch(str(first.get("expected_expiration_time") or first.get("close_time") or ""))
        close = iso_to_epoch(str(first.get("close_time") or ""))
        # Match the market to its own quarter's row, so the order in which FRED lists quarters never matters.
        own = reading if reading.quarter == quarter else (
            getattr(self.nowcast, "for_quarter", lambda _q: None)(quarter) if quarter else None)
        ctx: dict[str, Any] = {
            "event": str(first.get("event_ticker", "")), "quarter": quarter, "days": None, "sd": None, "reason": "",
            "nowcast": own.value if own is not None else None, "_reading": own,
        }
        if release is None or not quarter:
            ctx["reason"] = "unreadable_event"
            return ctx
        days = (release - now.timestamp()) / 86400.0
        ctx["days"], ctx["sd"] = days, lead_sd(self.config, max(0.0, days))
        if own is None:
            ctx["reason"] = "nowcast_is_for_another_quarter"
        elif days > self.config.max_days_to_release:
            ctx["reason"] = "too_far_from_release"
        elif close is not None and (close - now.timestamp()) / 3600.0 < self.config.min_hours_to_close:
            ctx["reason"] = "too_close_to_close"
        elif days <= 0:
            ctx["reason"] = "past_release"
        return ctx

    def _score_events(
        self, markets: list[dict[str, Any]], reading: NowcastReading, now: datetime,
    ) -> list[dict[str, Any]]:
        by_event: dict[str, list[dict[str, Any]]] = {}
        for market in markets:
            by_event.setdefault(str(market.get("event_ticker", "")), []).append(market)
        events: list[dict[str, Any]] = []
        for event_markets in by_event.values():
            ctx = self._event_context(event_markets, reading, now)
            rows: list[dict[str, Any]] = []
            for market in sorted(event_markets, key=lambda m: _num(m.get("floor_strike"))):
                if ctx["reason"]:
                    rows.append({"ticker": market.get("ticker", ""), "strike": _num(market.get("floor_strike")),
                                 "yes_bid": _num(market.get("yes_bid_dollars")),
                                 "yes_ask": _num(market.get("yes_ask_dollars")), "reason": ctx["reason"],
                                 "p_model": None, "p_blend": None, "side": "", "edge": None})
                else:
                    row = evaluate_market(market, ctx["_reading"], ctx["days"], self.config, self._fee_mult())
                    row["_market"] = market
                    rows.append(row)
            events.append({**ctx, "rows": rows})
        return sorted(events, key=lambda e: e["event"])

    # -------------------------------------------------------------- entries

    def _my_positions(self, state: PortfolioState) -> dict[str, Any]:
        return {s: p for s, p in state.positions.items()
                if p.asset_type == "prediction" and p.strategy == STRATEGY}

    def _enter(
        self, state: PortfolioState, broker: PaperBroker, reading: NowcastReading, now: datetime,
        result: dict[str, Any],
    ) -> None:
        cfg = self.config
        bucket = _CASH_BUCKET if self.live_trading_enabled else None
        events = self._score_events(self._discover(), reading, now)
        result["events"] = _public(events)

        account_tickers: set[str] = set()
        if self.live_trading_enabled:
            try:
                account_tickers = set(self.kalshi.get_held_tickers())
            except Exception as exc:
                self.logger.warning("Could not read the account's positions; no GDP bets: %s", exc)
                self._skip(result, "*", "positions_unavailable")
                return
        mine = self._my_positions(state)

        equity = broker.portfolio_equity_for_bucket(state, cash_bucket=bucket)
        cash = broker.get_cash_balance(state, cash_bucket=bucket)
        exposure = max(sum(p.market_value for p in mine.values()), self._account_exposure())
        budget = max(0.0, equity * cfg.capital_allocation_pct - exposure)

        candidates: list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for event in events:
            event_id = event["event"]
            held_here = {t for t in account_tickers if t.rsplit("-", 1)[0] == event_id}
            held_here |= {str(p.meta.get("ticker", "")) for p in mine.values()
                          if p.meta.get("event_ticker") == event_id}
            for row in event["rows"]:
                if row["reason"]:
                    continue
                if row["ticker"] in account_tickers:
                    row["reason"] = "already_held"
                    continue
                if len(held_here) >= cfg.max_positions_per_event:
                    row["reason"] = "event_position_cap"
                    continue
                candidates.append((row["edge"], row, event, row["_market"]))
        candidates.sort(key=lambda c: c[0], reverse=True)

        failures = 0
        for _edge, row, event, market in candidates:
            event_id = event["event"]
            already = len({t for t in account_tickers if t.rsplit("-", 1)[0] == event_id}
                          | {str(p.meta.get("ticker", "")) for p in self._my_positions(state).values()
                             if p.meta.get("event_ticker") == event_id})
            if already >= cfg.max_positions_per_event:
                continue
            cost = row["price"] + row["fee"]
            p_side = row["p_blend"] if row["side"] == "yes" else 1.0 - row["p_blend"]
            stake = equity * kelly_stake(p_side, cost, cfg.kelly_fraction)
            notional = min(stake, equity * cfg.max_position_pct, budget, cash * 0.95)
            if row["ask_size"] > 0:
                notional = min(notional, row["ask_size"] * row["price"])
            quantity = int(notional / cost) if cost > 0 else 0
            if quantity < 1 or quantity * cost < cfg.min_trade_usd:
                self._skip(result, row["ticker"], "stake_below_min", notional=round(notional, 2))
                continue
            if failures >= _MAX_ORDER_FAILURES:
                break
            booked = self._buy(state, broker, market, row, event, event["_reading"], quantity, bucket, now, result)
            if booked is None:
                failures += 1
                continue
            spent = booked * cost
            budget = max(0.0, budget - spent)
            cash = max(0.0, cash - spent)

    def _account_exposure(self) -> float:
        """Dollars at risk in this series on the account (covers bets other strategies placed)."""
        if not self.live_trading_enabled:
            return 0.0
        reader = getattr(self.kalshi, "get_exposure_by_ticker", None)
        if reader is None:
            return 0.0
        try:
            prefix = f"{self.config.series}-"
            return sum(cost for ticker, cost in reader().items() if ticker.startswith(prefix))
        except Exception as exc:
            self.logger.warning("Could not read the account's GDP exposure: %s", exc)
            return 0.0

    def _buy(
        self, state: PortfolioState, broker: PaperBroker, market: dict[str, Any], row: dict[str, Any],
        event: dict[str, Any], reading: NowcastReading, quantity: int, bucket: str | None,
        now: datetime, result: dict[str, Any],
    ) -> int | None:
        ticker, side, price = row["ticker"], row["side"], row["price"]
        if self.live_trading_enabled:
            try:
                order = self.kalshi.place_order(
                    ticker=ticker, side=side, action="buy", count=quantity,
                    price_cents=round(price * 100), time_in_force="immediate_or_cancel",
                )
            except Exception as exc:
                self.logger.warning("Kalshi buy failed for %s: %s", ticker, exc)
                self._skip(result, ticker, "order_failed", error=str(exc)[:300], side=side, count=quantity)
                return None
            filled = order_fill_count(order)
            if filled is not None:
                if filled < 1:
                    self._skip(result, ticker, "order_not_filled", side=side, count=quantity)
                    return None
                quantity = min(quantity, filled)
        symbol = f"KX:{ticker}:{side}"
        position = broker.open_position(
            state, symbol=symbol, asset_type="prediction", strategy=STRATEGY, price=price,
            notional=quantity * price, opened_at=now, cash_bucket=bucket,
            meta={
                "ticker": ticker, "side": side, "entry_price": price, "venue": "kalshi",
                "event_ticker": event["event"], "title": market.get("title", ""),
                "strike": row["strike"], "nowcast": reading.value, "p_model": round(row["p_model"], 4),
                "p_blend": round(row["p_blend"], 4), "edge": round(row["edge"], 4),
            },
        )
        if position is None:
            return None
        result["buys"].append(symbol)
        result["decisions"].append({
            "strategy": STRATEGY, "symbol": symbol, "action": "buy", "reason": "model_edge",
            "signals": {
                "title": str(market.get("title", ""))[:70], "side": side, "price": round(price, 4),
                "count": quantity, "nowcast": reading.value, "p_model": round(row["p_model"], 3),
                "p_blend": round(row["p_blend"], 3), "net_edge": round(row["edge"], 3),
                "days_to_release": round(event["days"], 1),
            },
        })
        self.logger.info("GDP bet: %s x%d @ %.2f (nowcast %.2f, blend %.2f, edge %.3f)",
                         symbol, quantity, price, reading.value, row["p_blend"], row["edge"])
        return quantity

    # ----------------------------------------------------------- management

    def _manage_positions(
        self, state: PortfolioState, broker: PaperBroker, reading: NowcastReading | None, now: datetime,
        result: dict[str, Any],
    ) -> None:
        bucket = _CASH_BUCKET if self.live_trading_enabled else None
        for symbol, position in list(self._my_positions(state).items()):
            ticker = str(position.meta.get("ticker", ""))
            side = str(position.meta.get("side", "yes"))
            if not ticker:
                continue
            try:
                market = self.kalshi.get_market(ticker)
            except Exception as exc:
                self.logger.warning("Could not read %s: %s", ticker, exc)
                continue
            status = str(market.get("status", "")).lower()
            outcome = str(market.get("result", "")).lower()
            if status in _RESOLVED and outcome:
                closed = broker.close_position(
                    state, symbol=symbol, price=1.0 if outcome == side else 0.0,
                    reason=f"resolved:{outcome}", cash_bucket=bucket,
                )
                if closed is not None:
                    result["resolved_closed"].append(symbol)
                continue
            if reading is not None and status in ("active", "open"):
                self._maybe_exit(state, broker, symbol, position, market, reading, now, bucket, result)

    def _maybe_exit(
        self, state: PortfolioState, broker: PaperBroker, symbol: str, position: Any,
        market: dict[str, Any], reading: NowcastReading, now: datetime, bucket: str | None,
        result: dict[str, Any],
    ) -> None:
        """Sell early only when the bid pays more than we think the contract is worth."""
        ctx = self._event_context([market], reading, now)
        if ctx["reason"] in ("unreadable_event", "nowcast_is_for_another_quarter") or ctx["days"] is None:
            return
        row = evaluate_market(market, ctx["_reading"], max(0.0, ctx["days"]), self.config, self._fee_mult())
        if row["p_blend"] is None or row["reason"] in ("model_market_gap", "unsupported_market", "no_two_sided_book"):
            return          # no trustworthy model value: hold rather than act on it
        side = str(position.meta.get("side", "yes"))
        value = row["p_blend"] if side == "yes" else 1.0 - row["p_blend"]
        bid = _num(market.get(f"{side}_bid_dollars"))
        if bid <= 0:
            return
        proceeds = bid - kalshi_fee(bid, _FEE_RATE, self._fee_mult())
        if proceeds < value + self.config.exit_margin:
            return
        held = int(position.quantity)
        if self.live_trading_enabled:
            try:
                order = self.kalshi.sell_to_close(ticker=str(position.meta.get("ticker", "")), side=side, count=held)
            except Exception as exc:
                self.logger.warning("Kalshi sell failed for %s: %s; keeping the position.", symbol, exc)
                return
            sold = order_fill_count(order)
            if sold is not None and sold < 1:
                return
            if sold is not None and sold < held:
                broker.resize_position(state, symbol, held - sold, refund_price=bid)
                return
        closed = broker.close_position(state, symbol=symbol, price=bid, reason="edge_gone", cash_bucket=bucket)
        if closed is not None:
            result["sells"].append(symbol)
            result["decisions"].append({
                "strategy": STRATEGY, "symbol": symbol, "action": "sell", "reason": "edge_gone",
                "signals": {"bid": round(bid, 3), "our_value": round(value, 3)},
            })

    # --------------------------------------------------------------- helpers

    def _skip(self, result: dict[str, Any], symbol: str, reason: str, **signals: Any) -> None:
        result["decisions"].append({
            "strategy": STRATEGY, "symbol": symbol, "action": "skip_buy", "reason": reason, "signals": signals,
        })
