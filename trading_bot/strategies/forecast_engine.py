"""Shared engine for Kalshi strategies that bet only where a forecast beats the price.

A strategy supplies a ``ForecastModel``: where its forecast comes from, how to read an event's timing
and how to turn the forecast into P(YES) for one market. This engine does everything that must be
identical and careful for every such strategy:

* it averages the model with the market's own price, because the market also sees public forecasts,
* it buys YES or NO only if that probability beats the ask plus Kalshi's fee by ``min_edge``,
* it sizes by fractional Kelly under per-bet, total and per-event caps,
* every guard fails closed: no forecast, a wide book, a model that disagrees wildly with the market,
  an unreadable account, a daily halt or the kill switch all mean no bet,
* it books only contracts that really filled, and sells early only when the bid beats its own value.

Per Kalshi contract bought at price ``c`` (fee included) with win probability ``p``, the stake follows
fractional Kelly: ``f = kelly * (p - c) / (1 - c)``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Protocol

from trading_bot.broker import PaperBroker
from trading_bot.clients_kalshi import KalshiClient, order_fill_count
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.sports_model import kalshi_fee

CASH_BUCKET = "kalshi"
RESOLVED = {"settled", "finalized", "determined"}
FEE_RATE = 0.07
MAX_ORDER_FAILURES = 3


def num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def kelly_stake(p: float, cost: float, fraction: float) -> float:
    """Share of bankroll to stake on a $1 binary contract costing ``cost`` (fee included)."""
    if cost <= 0 or cost >= 1 or p <= cost:
        return 0.0
    return max(0.0, fraction * (p - cost) / (1.0 - cost))


def series_of(market: dict[str, Any]) -> str:
    return str(market.get("series_ticker") or str(market.get("ticker", "")).split("-")[0])


def public_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events as plain data for results and Telegram: drops the private objects used while deciding."""
    return [
        {**{k: v for k, v in e.items() if not k.startswith("_")},
         "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in e["rows"]]}
        for e in events
    ]


def score_market(
    market: dict[str, Any], p_model: float, cfg: Any, fee_multiplier: float = 1.0,
    *, strike: float | None = None, label: str = "",
) -> dict[str, Any]:
    """Score one market given the model's P(YES). ``reason`` is empty when it is a bet, else why not."""
    row: dict[str, Any] = {
        "ticker": str(market.get("ticker", "")), "strike": strike, "label": label,
        "yes_bid": 0.0, "yes_ask": 0.0, "p_market": None, "p_model": None, "p_blend": None,
        "side": "", "price": 0.0, "fee": 0.0, "edge": None, "ask_size": 0.0, "reason": "",
    }
    yes_bid, yes_ask = num(market.get("yes_bid_dollars")), num(market.get("yes_ask_dollars"))
    no_ask = num(market.get("no_ask_dollars"))
    row.update(yes_bid=yes_bid, yes_ask=yes_ask)
    if yes_bid <= 0 or yes_ask <= 0 or no_ask <= 0 or yes_ask < yes_bid:
        row["reason"] = "no_two_sided_book"
        return row
    p_market = (yes_bid + yes_ask) / 2.0
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
        fee = kalshi_fee(price, FEE_RATE, fee_multiplier)
        edge = p_side - price - fee
        if best is None or edge > best[0]:
            best = (edge, side, price, fee, p_side)
            row["ask_size"] = num(market.get(depth_key))
    if best is None:
        row["reason"] = "price_out_of_range"
        return row
    edge, side, price, fee, _p_side = best
    row.update(side=side, price=price, fee=fee, edge=edge)
    if edge < cfg.min_edge:
        row["reason"] = "edge_too_small"
    return row


class ForecastModel(Protocol):
    """What a strategy supplies to the engine."""

    strategy: str                   # position strategy name, e.g. "gdp_nowcast"
    summary_key: str                # key under which the forecast summary appears in results
    series: tuple[str, ...]         # Kalshi series to scan
    label: str                      # short name used in logs, e.g. "GDP"

    def refresh(self, now: datetime) -> dict[str, Any] | None:
        """Load the forecast. Return a display summary, or None when it is unavailable."""

    @property
    def error(self) -> str:
        """Why the last refresh returned None."""

    def event_context(self, event_markets: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
        """Timing and forecast facts for one event. ``reason`` non-empty means do not bet on it;
        ``can_value`` False means the model cannot value its markets at all (no early exit either)."""

    def probability(self, market: dict[str, Any], ctx: dict[str, Any]) -> tuple[float, float | None, str] | None:
        """(P(YES), strike for display, label for display), or None for an unsupported market."""

    def position_meta(self, row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]: ...

    def decision_signals(self, row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]: ...


class ForecastEngine:
    def __init__(
        self,
        *,
        config: Any,
        model: ForecastModel,
        kalshi_client: KalshiClient,
        logger: logging.Logger,
        live_trading_enabled: bool = False,
    ) -> None:
        self.config = config
        self.model = model
        self.kalshi = kalshi_client
        self.logger = logger.getChild(model.strategy)
        self.live_trading_enabled = live_trading_enabled
        self._fee_multipliers: dict[str, float] = {}

    # ------------------------------------------------------------------ run

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
    ) -> dict[str, Any]:
        key = self.model.summary_key
        result: dict[str, Any] = {
            "status": "ok", "buys": [], "sells": [], "resolved_closed": [], "decisions": [],
            "events": [], key: None, "can_trade": self.live_trading_enabled,
        }
        if not self.config.enabled:
            result["status"] = "disabled"
            return result
        try:
            summary = self.model.refresh(now)
            if summary is None:
                result["status"] = f"no_{key}"
                result["error"] = self.model.error or f"no {self.model.label} forecast available"
                self._skip(result, "*", f"no_{key}", error=result["error"])
            else:
                result[key] = summary
            self._manage_positions(state, broker, summary is not None, now, result)
            if summary is None:
                return result
            if state.kill_switch_engaged or state.halt_new_trades_for_day:
                self._skip(result, "*", "risk_guardrails",
                           kill_switch=state.kill_switch_engaged, daily_halt=state.halt_new_trades_for_day)
                result["events"] = public_events(self._score_events(self._discover(), now))
                return result
            self._enter(state, broker, now, result)
        except Exception as exc:
            self.logger.error("%s cycle failed: %s", self.model.label, exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)
        return result

    def preview(self, now: datetime) -> dict[str, Any]:
        """Read-only view of model versus market for every open market. Places no orders."""
        key = self.model.summary_key
        out: dict[str, Any] = {
            "status": "ok", "enabled": self.config.enabled, "min_edge": self.config.min_edge,
            "events": [], key: None, "buys": [], "sells": [], "decisions": [],
        }
        try:
            summary = self.model.refresh(now)
            if summary is None:
                out["status"] = f"no_{key}"
                out["error"] = self.model.error or f"no {self.model.label} forecast available"
                return out
            out[key] = summary
            out["events"] = public_events(self._score_events(self._discover(), now))
        except Exception as exc:
            self.logger.error("%s preview failed: %s", self.model.label, exc, exc_info=True)
            out["status"] = "error"
            out["error"] = str(exc)
        return out

    # ------------------------------------------------------------ discovery

    def _discover(self) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        for series in self.model.series:
            cursor: str | None = None
            for _ in range(3):
                page, cursor = self.kalshi.get_markets(series_ticker=series, status="open", limit=100, cursor=cursor)
                markets.extend(page)
                if not cursor:
                    break
        return [m for m in markets if str(m.get("status", "active")).lower() in ("active", "open")]

    def _fee_mult(self, series: str) -> float:
        if series not in self._fee_multipliers:
            multiplier = 1.0
            try:
                info = self.kalshi.get_series(series)
                fee_type = str(info.get("fee_type", "")).lower()
                raw = info.get("fee_multiplier")
                if "quadratic" in fee_type and raw not in (None, ""):
                    multiplier = max(0.0, float(raw))
            except Exception as exc:
                self.logger.debug("Could not read fee info for %s (assuming standard fee): %s", series, exc)
            self._fee_multipliers[series] = multiplier
        return self._fee_multipliers[series]

    def _score_events(self, markets: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
        by_event: dict[str, list[dict[str, Any]]] = {}
        for market in markets:
            by_event.setdefault(str(market.get("event_ticker", "")), []).append(market)
        events: list[dict[str, Any]] = []
        for event_markets in by_event.values():
            ctx = self.model.event_context(event_markets, now)
            rows: list[dict[str, Any]] = []
            for market in sorted(event_markets, key=lambda m: num(m.get("floor_strike")) or num(m.get("cap_strike"))):
                if ctx["reason"]:
                    rows.append(self._blocked_row(market, ctx["reason"]))
                    continue
                valued = self.model.probability(market, ctx)
                if valued is None:
                    rows.append(self._blocked_row(market, "unsupported_market"))
                    continue
                p_yes, strike, label = valued
                row = score_market(market, p_yes, self.config, self._fee_mult(series_of(market)),
                                   strike=strike, label=label)
                row["_market"] = market
                rows.append(row)
            events.append({**ctx, "rows": rows})
        return sorted(events, key=lambda e: e["event"])

    @staticmethod
    def _blocked_row(market: dict[str, Any], reason: str) -> dict[str, Any]:
        strike = market.get("floor_strike") if market.get("floor_strike") is not None else market.get("cap_strike")
        return {"ticker": market.get("ticker", ""), "strike": num(strike), "label": "",
                "yes_bid": num(market.get("yes_bid_dollars")), "yes_ask": num(market.get("yes_ask_dollars")),
                "reason": reason, "p_model": None, "p_blend": None, "side": "", "edge": None}

    # -------------------------------------------------------------- entries

    def _my_positions(self, state: PortfolioState) -> dict[str, Any]:
        return {s: p for s, p in state.positions.items()
                if p.asset_type == "prediction" and p.strategy == self.model.strategy}

    def _enter(self, state: PortfolioState, broker: PaperBroker, now: datetime, result: dict[str, Any]) -> None:
        cfg = self.config
        bucket = CASH_BUCKET if self.live_trading_enabled else None
        events = self._score_events(self._discover(), now)
        result["events"] = public_events(events)

        account_tickers: set[str] = set()
        if self.live_trading_enabled:
            try:
                account_tickers = set(self.kalshi.get_held_tickers())
            except Exception as exc:
                self.logger.warning("Could not read the account's positions; no %s bets: %s", self.model.label, exc)
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
            if failures >= MAX_ORDER_FAILURES:
                break
            booked = self._buy(state, broker, market, row, event, quantity, bucket, now, result)
            if booked is None:
                failures += 1
                continue
            spent = booked * cost
            budget = max(0.0, budget - spent)
            cash = max(0.0, cash - spent)

    def _account_exposure(self) -> float:
        """Dollars at risk in this strategy's series on the account (covers bets placed elsewhere)."""
        if not self.live_trading_enabled:
            return 0.0
        reader = getattr(self.kalshi, "get_exposure_by_ticker", None)
        if reader is None:
            return 0.0
        try:
            prefixes = tuple(f"{s}-" for s in self.model.series)
            return sum(cost for ticker, cost in reader().items() if ticker.startswith(prefixes))
        except Exception as exc:
            self.logger.warning("Could not read the account's %s exposure: %s", self.model.label, exc)
            return 0.0

    def _buy(
        self, state: PortfolioState, broker: PaperBroker, market: dict[str, Any], row: dict[str, Any],
        event: dict[str, Any], quantity: int, bucket: str | None, now: datetime, result: dict[str, Any],
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
            state, symbol=symbol, asset_type="prediction", strategy=self.model.strategy, price=price,
            notional=quantity * price, opened_at=now, cash_bucket=bucket,
            meta={
                "ticker": ticker, "side": side, "entry_price": price, "venue": "kalshi",
                "event_ticker": event["event"], "title": market.get("title", ""),
                "strike": row["strike"], "p_model": round(row["p_model"], 4),
                "p_blend": round(row["p_blend"], 4), "edge": round(row["edge"], 4),
                **self.model.position_meta(row, event),
            },
        )
        if position is None:
            return None
        result["buys"].append(symbol)
        result["decisions"].append({
            "strategy": self.model.strategy, "symbol": symbol, "action": "buy", "reason": "model_edge",
            "signals": {
                "title": str(market.get("title", ""))[:70], "side": side, "price": round(price, 4),
                "count": quantity, "p_model": round(row["p_model"], 3),
                "p_blend": round(row["p_blend"], 3), "net_edge": round(row["edge"], 3),
                **self.model.decision_signals(row, event),
            },
        })
        self.logger.info("%s bet: %s x%d @ %.2f (blend %.2f, edge %.3f)",
                         self.model.label, symbol, quantity, price, row["p_blend"], row["edge"])
        return quantity

    # ----------------------------------------------------------- management

    def _manage_positions(
        self, state: PortfolioState, broker: PaperBroker, have_forecast: bool, now: datetime,
        result: dict[str, Any],
    ) -> None:
        bucket = CASH_BUCKET if self.live_trading_enabled else None
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
            if status in RESOLVED and outcome:
                closed = broker.close_position(
                    state, symbol=symbol, price=1.0 if outcome == side else 0.0,
                    reason=f"resolved:{outcome}", cash_bucket=bucket,
                )
                if closed is not None:
                    result["resolved_closed"].append(symbol)
                continue
            if have_forecast and status in ("active", "open"):
                self._maybe_exit(state, broker, symbol, position, market, now, bucket, result)

    def _maybe_exit(
        self, state: PortfolioState, broker: PaperBroker, symbol: str, position: Any,
        market: dict[str, Any], now: datetime, bucket: str | None, result: dict[str, Any],
    ) -> None:
        """Sell early only when the bid pays more than we think the contract is worth."""
        ctx = self.model.event_context([market], now)
        if not ctx.get("can_value", True):
            return
        valued = self.model.probability(market, ctx)
        if valued is None:
            return
        p_yes, strike, label = valued
        row = score_market(market, p_yes, self.config, self._fee_mult(series_of(market)), strike=strike, label=label)
        if row["p_blend"] is None or row["reason"] in ("model_market_gap", "unsupported_market", "no_two_sided_book"):
            return          # no trustworthy model value: hold rather than act on it
        side = str(position.meta.get("side", "yes"))
        value = row["p_blend"] if side == "yes" else 1.0 - row["p_blend"]
        bid = num(market.get(f"{side}_bid_dollars"))
        if bid <= 0:
            return
        proceeds = bid - kalshi_fee(bid, FEE_RATE, self._fee_mult(series_of(market)))
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
                "strategy": self.model.strategy, "symbol": symbol, "action": "sell", "reason": "edge_gone",
                "signals": {"bid": round(bid, 3), "our_value": round(value, 3)},
            })

    # --------------------------------------------------------------- helpers

    def _skip(self, result: dict[str, Any], symbol: str, reason: str, **signals: Any) -> None:
        result["decisions"].append({
            "strategy": self.model.strategy, "symbol": symbol, "action": "skip_buy", "reason": reason,
            "signals": signals,
        })
