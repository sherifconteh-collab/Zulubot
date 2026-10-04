"""Kalshi sports (game-winner) strategy driven by ESPN data.

Strategy overview
-----------------
1. Pull upcoming games from ESPN's scoreboard for each configured league.
2. Pull open Kalshi game-winner markets for the matching series
   (e.g. ``KXNFLGAME``) and match each market to an ESPN game + team.
3. For each matched game build a win-probability estimate by blending:
     * the vig-free sportsbook moneyline ESPN shows for the game,
     * ESPN's own matchup predictor,
     * a log5 team-record model with a home-field prior.
4. Compare the estimate to Kalshi's ask on both YES and NO. Bet only when the
   edge clears ``min_edge_pct`` and every sanity filter passes.
5. Size with fractional Kelly, capped per bet and in total sports exposure.
6. Hold to settlement (default) and close at $1/$0 when Kalshi resolves.

Position symbols
----------------
  KS:{ticker}:{YES|NO}   asset_type="prediction", strategy="kalshi_sports"

Notes
-----
* Two-way model only: leagues where a draw is possible (soccer) are not
  supported by the default series map.
* Sportsbook lines are the sharpest public input; records alone are not
  enough to beat Kalshi's price, so ``require_market_signal`` defaults on.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from datetime import datetime, timedelta
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients_espn import (
    DEFAULT_KALSHI_SERIES,
    EspnClient,
    EspnGame,
    EspnTeam,
    devig_two_way,
)
from trading_bot.clients_kalshi import KalshiClient
from trading_bot.config import KalshiSportsConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager

_CASH_BUCKET = "kalshi"
_SYMBOL_PREFIX = "KS"
# An "edge" this large almost always means a team mismatch or bad quote, not alpha.
_MAX_PLAUSIBLE_EDGE = 0.20
# Shrink small-sample records toward .500 (equivalent to 5 phantom wins + 5 losses).
_RECORD_PRIOR_GAMES = 10.0
_RESOLVED_STATUSES = {"settled", "finalized", "determined"}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _word_in(alias: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None


def _team_in_text(team: EspnTeam, text: str) -> bool:
    return any(_word_in(alias, text) for alias in team.aliases())


def _event_code(event_ticker: str) -> str:
    """"KXNFLGAME-25SEP28BUFBAL" -> "BUFBAL" (also handles MLB's HHMM stamp)."""
    tail = event_ticker.upper().split("-", 1)[-1]
    m = re.match(r"^\d{2}[A-Z]{3}\d{2}(?:\d{4})?([A-Z0-9]+)$", tail)
    return m.group(1) if m else ""


def match_market_to_game(
    market: dict[str, Any],
    games: list[EspnGame],
) -> tuple[EspnGame, str] | None:
    """Return (game, "home"|"away") for the team a market's YES side backs.

    Returns None when the market cannot be matched to exactly one game and one team.
    """
    ticker = str(market.get("ticker", "")).upper()
    event_ticker = str(market.get("event_ticker", "")).upper()
    yes_sub = str(market.get("yes_sub_title", "") or "").lower()
    blob = " ".join(
        str(market.get(k, "") or "") for k in ("title", "subtitle", "yes_sub_title", "no_sub_title")
    ).lower()
    code = _event_code(event_ticker)

    hits: list[EspnGame] = []
    for game in games:
        by_name = _team_in_text(game.home, blob) and _team_in_text(game.away, blob)
        h, a = game.home.abbreviation, game.away.abbreviation
        by_code = bool(code) and h and a and code in (a + h, h + a)
        if by_name or by_code:
            hits.append(game)
    if len(hits) != 1:
        return None
    game = hits[0]

    home_hit = _team_in_text(game.home, yes_sub) if yes_sub else False
    away_hit = _team_in_text(game.away, yes_sub) if yes_sub else False
    if home_hit != away_hit:
        return game, "home" if home_hit else "away"

    suffix = ticker.rsplit("-", 1)[-1] if "-" in ticker else ""
    if suffix and suffix == game.home.abbreviation:
        return game, "home"
    if suffix and suffix == game.away.abbreviation:
        return game, "away"
    return None


def _kalshi_price(market: dict[str, Any], name: str) -> float:
    """Price in [0, 1] for e.g. ``yes_ask``; 0.0 when absent."""
    dollars = market.get(f"{name}_dollars")
    if dollars not in (None, ""):
        try:
            return max(0.0, float(dollars))
        except (TypeError, ValueError):
            pass
    cents = market.get(name)
    if cents not in (None, ""):
        try:
            return max(0.0, float(cents) / 100.0)
        except (TypeError, ValueError):
            pass
    return 0.0


def _shrunk_win_pct(team: EspnTeam) -> float:
    if team.win_pct is None or team.games_played <= 0:
        return 0.5
    wins = team.win_pct * team.games_played
    return (wins + _RECORD_PRIOR_GAMES / 2) / (team.games_played + _RECORD_PRIOR_GAMES)


def record_home_prob(game: EspnGame, home_advantage: float) -> float | None:
    """log5 head-to-head from records, adjusted for home field. None if no records."""
    if game.home.win_pct is None and game.away.win_pct is None:
        return None
    a, b = _shrunk_win_pct(game.home), _shrunk_win_pct(game.away)
    denom = a + b - 2 * a * b
    neutral = 0.5 if denom <= 0 else (a - a * b) / denom
    hfa = min(max(home_advantage, 0.01), 0.99)
    odds = (neutral / (1 - neutral)) * (hfa / (1 - hfa))
    return odds / (1 + odds)


def estimate_home_win_prob(
    game: EspnGame,
    config: KalshiSportsConfig,
) -> tuple[float, dict[str, float]] | None:
    """Blend the available signals into P(home wins).

    Returns (probability, components) or None when the data is too thin to bet.
    """
    components: dict[str, float] = {}
    weights: dict[str, float] = {}

    book = devig_two_way(game.home_moneyline, game.away_moneyline)
    if book is not None:
        components["sportsbook"] = book
        weights["sportsbook"] = config.weight_sportsbook
    if game.predictor_home_prob is not None:
        components["espn_predictor"] = game.predictor_home_prob
        weights["espn_predictor"] = config.weight_espn_predictor
    rec = record_home_prob(game, config.home_advantage)
    if rec is not None:
        components["record"] = rec
        weights["record"] = config.weight_record

    if config.require_market_signal and "sportsbook" not in components and "espn_predictor" not in components:
        return None
    total_weight = sum(weights.values())
    if not components or total_weight <= 0:
        return None
    prob = sum(components[k] * weights[k] for k in components) / total_weight
    return min(max(prob, 0.01), 0.99), components


def kelly_notional(edge: float, ask: float, equity: float, config: KalshiSportsConfig) -> float:
    """Fractional-Kelly stake in USD for a binary contract, capped per bet."""
    if ask <= 0 or ask >= 1 or edge <= 0:
        return 0.0
    full = edge / (1.0 - ask)
    return min(equity * full * config.kelly_fraction, equity * config.max_position_pct)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class KalshiSportsStrategy:
    """Bets on Kalshi game-winner markets using ESPN schedule, odds and projections."""

    def __init__(
        self,
        *,
        config: KalshiSportsConfig,
        kalshi_client: KalshiClient,
        espn_client: EspnClient,
        logger: logging.Logger,
        live_trading_enabled: bool = False,
    ) -> None:
        self.config = config
        self.kalshi = kalshi_client
        self.espn = espn_client
        self.logger = logger.getChild("strategy.kalshi_sports")
        self.live_trading_enabled = live_trading_enabled

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "ok",
            "buys": [],
            "sells": [],
            "resolved_closed": [],
            "games_scanned": 0,
            "markets_scanned": 0,
            "markets_matched": 0,
            "predictions": [],
            "decisions": [],
            "can_trade_live": self.live_trading_enabled,
        }
        if not self.config.enabled:
            result["status"] = "disabled"
            return result

        try:
            self._close_resolved_positions(state, broker, result)
            self._manage_open_positions(state, broker, result)

            if state.kill_switch_engaged or state.halt_new_trades_for_day:
                result["status"] = "halted"
                result["decisions"].append({
                    "strategy": "kalshi_sports",
                    "symbol": "*",
                    "action": "skip_buy",
                    "reason": "risk_guardrails",
                })
                return result

            candidates = self._build_candidates(now, result)
            if result["games_scanned"] == 0 and result["markets_scanned"] == 0:
                result["status"] = "no_data"
            self._execute_candidates(state, broker, candidates, now, result)
        except Exception as exc:
            self.logger.error("Kalshi sports cycle failed: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)
        return result

    # ------------------------------------------------------------------
    # Discovery + prediction
    # ------------------------------------------------------------------

    def _series_map(self) -> dict[str, str]:
        mapping = {lg: DEFAULT_KALSHI_SERIES[lg] for lg in self.config.leagues if lg in DEFAULT_KALSHI_SERIES}
        for entry in self.config.series_overrides:
            league, _, series = entry.partition("=")
            if league.strip() and series.strip():
                mapping[league.strip()] = series.strip().upper()
        return mapping

    def _fetch_games(self, league: str, now: datetime) -> list[EspnGame]:
        end = now + timedelta(hours=self.config.max_hours_ahead)
        # Query a day either side of UTC so late-evening US games aren't clipped.
        dates = f"{(now - timedelta(days=1)):%Y%m%d}-{(end + timedelta(days=1)):%Y%m%d}"
        games = self.espn.get_scoreboard(league, date=dates)
        unique = {g.game_id: g for g in games}
        return list(unique.values())

    def _fetch_markets(self, series: str) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        cursor = None
        for _ in range(3):
            page, cursor = self.kalshi.get_markets(
                status="open", series_ticker=series, limit=200, cursor=cursor,
            )
            markets.extend(page)
            if not cursor:
                break
        return markets

    def _build_candidates(self, now: datetime, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Score every matched market; return tradeable candidates (best edge per game)."""
        best_per_game: dict[str, dict[str, Any]] = {}

        for league, series in self._series_map().items():
            try:
                games = self._fetch_games(league, now)
                markets = self._fetch_markets(series)
            except Exception as exc:
                self.logger.warning("Sports data fetch failed for %s: %s", league, exc)
                result["decisions"].append({
                    "strategy": "kalshi_sports", "symbol": league,
                    "action": "skip", "reason": "data_fetch_failed",
                    "signals": {"error": str(exc)[:120]},
                })
                continue
            result["games_scanned"] += len(games)
            result["markets_scanned"] += len(markets)

            upcoming = [
                g for g in games
                if g.state == "pre"
                and self.config.min_minutes_to_start * 60
                <= (g.start_time - now).total_seconds()
                <= self.config.max_hours_ahead * 3600
            ]
            matched: list[tuple[dict[str, Any], EspnGame, str]] = []
            for market in markets:
                found = match_market_to_game(market, upcoming)
                if found is not None:
                    matched.append((market, found[0], found[1]))
            result["markets_matched"] += len(matched)

            enriched: set[str] = set()
            for market, game, team_side in matched:
                if game.game_id not in enriched:
                    self.espn.enrich_game(game)
                    enriched.add(game.game_id)

                evaluation = self._evaluate_market(market, game, team_side, result)
                if evaluation is None:
                    continue
                key = f"{game.league}:{game.game_id}"
                current = best_per_game.get(key)
                if current is None or evaluation["edge"] > current["edge"]:
                    best_per_game[key] = evaluation

        return sorted(best_per_game.values(), key=lambda c: c["edge"], reverse=True)

    def _evaluate_market(
        self,
        market: dict[str, Any],
        game: EspnGame,
        team_side: str,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        cfg = self.config
        ticker = str(market.get("ticker", ""))
        estimate = estimate_home_win_prob(game, cfg)
        if estimate is None:
            self._log_skip(result, ticker, "insufficient_data", game=game.label)
            return None
        p_home, components = estimate
        p_team = p_home if team_side == "home" else 1.0 - p_home

        yes_ask = _kalshi_price(market, "yes_ask")
        yes_bid = _kalshi_price(market, "yes_bid")
        no_ask = _kalshi_price(market, "no_ask")
        no_bid = _kalshi_price(market, "no_bid")
        backed = game.home if team_side == "home" else game.away
        other = game.away if team_side == "home" else game.home

        prediction = {
            "game": game.label,
            "league": game.league,
            "start": game.start_time.isoformat(),
            "ticker": ticker,
            "yes_team": backed.display_name,
            "p_yes": round(p_team, 4),
            "components": {k: round(v, 4) for k, v in components.items()},
            "yes_ask": round(yes_ask, 4),
            "no_ask": round(no_ask, 4),
        }

        options = []
        if yes_ask > 0:
            options.append(("yes", yes_ask, yes_bid, p_team, backed))
        if no_ask > 0:
            options.append(("no", no_ask, no_bid, 1.0 - p_team, other))
        if not options:
            self._log_skip(result, ticker, "no_quotes", game=game.label)
            return None

        side, ask, bid, prob, pick_team = max(options, key=lambda o: o[3] - o[1])
        edge = prob - ask
        prediction.update({"pick": pick_team.display_name, "side": side, "edge": round(edge, 4)})
        result["predictions"].append(prediction)

        volume = int(float(market.get("volume", 0) or 0))
        reason = None
        if edge < cfg.min_edge_pct:
            reason = "edge_below_min"
        elif edge > _MAX_PLAUSIBLE_EDGE:
            reason = "edge_implausible"
        elif prob < cfg.min_model_prob:
            reason = "model_prob_too_low"
        elif ask > cfg.max_entry_price:
            reason = "price_too_high"
        elif volume < cfg.min_kalshi_volume:
            reason = "illiquid"
        elif bid > 0 and ask - bid > cfg.max_spread:
            reason = "spread_too_wide"
        if reason:
            self._log_skip(result, ticker, reason, game=game.label, edge=round(edge, 4), ask=round(ask, 4))
            return None

        return {
            "market": market,
            "game": game,
            "side": side,
            "ask": ask,
            "prob": prob,
            "edge": edge,
            "pick": pick_team.display_name,
            "components": components,
            "volume": volume,
        }

    def _log_skip(self, result: dict[str, Any], ticker: str, reason: str, **signals: Any) -> None:
        result["decisions"].append({
            "strategy": "kalshi_sports",
            "symbol": ticker,
            "action": "skip_buy",
            "reason": reason,
            "signals": signals,
        })

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _open_positions(self, state: PortfolioState) -> list[Any]:
        return [
            p for p in state.positions.values()
            if p.strategy == "kalshi_sports" and p.asset_type == "prediction"
        ]

    def _execute_candidates(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        candidates: list[dict[str, Any]],
        now: datetime,
        result: dict[str, Any],
    ) -> None:
        cfg = self.config
        bucket = _CASH_BUCKET if self.live_trading_enabled else None
        equity = broker.portfolio_equity_for_bucket(state, cash_bucket=bucket)
        open_positions = self._open_positions(state)
        exposure = sum(p.quantity * p.entry_price for p in open_positions)
        exposure_cap = equity * cfg.capital_allocation_pct
        held_events = {str(p.meta.get("event_ticker", "")) for p in open_positions}

        for cand in candidates:
            market = cand["market"]
            ticker = str(market.get("ticker", ""))
            event_ticker = str(market.get("event_ticker", ""))

            if len(open_positions) >= cfg.max_open_bets:
                self._log_skip(result, ticker, "max_open_bets")
                break
            if event_ticker and event_ticker in held_events:
                self._log_skip(result, ticker, "already_bet_on_game")
                continue

            notional = kelly_notional(cand["edge"], cand["ask"], equity, cfg)
            notional = min(notional, max(0.0, exposure_cap - exposure))
            if notional < cfg.min_trade_usd:
                self._log_skip(result, ticker, "stake_below_min", notional=round(notional, 2))
                continue

            position = self._place_bet(state, broker, cand, notional, now, bucket)
            if position is None:
                continue
            symbol = position.symbol
            exposure += position.quantity * position.entry_price
            open_positions.append(position)
            held_events.add(event_ticker)
            result["buys"].append(symbol)
            result["decisions"].append({
                "strategy": "kalshi_sports",
                "symbol": symbol,
                "action": "buy",
                "reason": "model_edge",
                "signals": {
                    "game": cand["game"].label,
                    "pick": cand["pick"],
                    "side": cand["side"],
                    "model_prob": round(cand["prob"], 4),
                    "ask": round(cand["ask"], 4),
                    "edge": round(cand["edge"], 4),
                    "notional": round(position.quantity * position.entry_price, 2),
                    "components": {k: round(v, 4) for k, v in cand["components"].items()},
                },
            })

    def _place_bet(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        cand: dict[str, Any],
        notional: float,
        now: datetime,
        bucket: str | None,
    ) -> Any | None:
        market, game, side, ask = cand["market"], cand["game"], cand["side"], cand["ask"]
        ticker = str(market.get("ticker", ""))
        symbol = f"{_SYMBOL_PREFIX}:{ticker}:{side.upper()}"
        if symbol in state.positions:
            return None

        if self.live_trading_enabled:
            notional = min(notional, broker.get_cash_balance(state, cash_bucket=_CASH_BUCKET) * 0.95)
        quantity = math.floor(notional / ask)
        if quantity < 1:
            return None

        meta = {
            "ticker": ticker,
            "event_ticker": str(market.get("event_ticker", "")),
            "side": side,
            "venue": "kalshi",
            "strategy": "kalshi_sports",
            "league": game.league,
            "game": game.label,
            "pick": cand["pick"],
            "game_start": game.start_time.isoformat(),
            "model_prob": round(cand["prob"], 4),
            "edge": round(cand["edge"], 4),
            "components": {k: round(v, 4) for k, v in cand["components"].items()},
            "odds_provider": game.odds_provider,
        }
        position = broker.open_position(
            state,
            symbol=symbol,
            asset_type="prediction",
            strategy="kalshi_sports",
            price=ask,
            notional=quantity * ask,
            opened_at=now,
            meta=meta,
            cash_bucket=bucket,
        )
        if position is None:
            return None

        filled = int(round(position.quantity))
        self.logger.info(
            "KALSHI SPORTS %s %s %s on %s @ $%.2f x%d | model=%.1f%% edge=%.1f%%",
            "LIVE" if self.live_trading_enabled else "PAPER",
            side.upper(), cand["pick"], game.label, ask, filled,
            cand["prob"] * 100, cand["edge"] * 100,
        )

        if self.live_trading_enabled and filled >= 1:
            try:
                ask_cents = min(99, max(1, round(ask * 100)))
                order = self.kalshi.place_order(
                    ticker=ticker,
                    side=side,
                    action="buy",
                    count=filled,
                    yes_price=ask_cents if side == "yes" else 100 - ask_cents,
                    client_order_id=str(uuid.uuid4()),
                )
                state.positions[symbol].meta["kalshi_order_id"] = order.get("order", {}).get("order_id", "")
            except Exception as exc:
                self.logger.error("Live Kalshi sports order failed for %s: %s", ticker, exc)
                # Roll the paper position back so we don't track a bet that never placed.
                broker.close_position(state, symbol=symbol, price=ask, reason="order_failed", cash_bucket=bucket)
                return None
        return position

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _close_resolved_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        result: dict[str, Any],
    ) -> None:
        for symbol, position in list(state.positions.items()):
            if position.strategy != "kalshi_sports":
                continue
            ticker = str(position.meta.get("ticker", ""))
            if not ticker:
                continue
            try:
                market = self.kalshi.get_market(ticker)
                status = str(market.get("status", "")).lower()
                outcome = str(market.get("result", "")).lower()
                if outcome not in ("yes", "no"):
                    if status in _RESOLVED_STATUSES:
                        self.logger.warning("%s resolved without a result; leaving for manual review.", ticker)
                    continue
                side = str(position.meta.get("side", "yes")).lower()
                close_price = 1.0 if outcome == side else 0.0
                broker.close_position(
                    state, symbol=symbol, price=close_price, reason=f"resolved:{outcome}",
                )
                result["resolved_closed"].append(symbol)
                result["decisions"].append({
                    "strategy": "kalshi_sports",
                    "symbol": symbol,
                    "action": "sell",
                    "reason": f"resolved:{outcome}",
                    "signals": {"won": close_price == 1.0, "game": position.meta.get("game", "")},
                })
            except Exception as exc:
                self.logger.warning("Resolution check failed for %s: %s", ticker, exc)

    def _manage_open_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        result: dict[str, Any],
    ) -> None:
        """Optional paper-only take-profit / stop-loss when hold_to_settlement is off.

        Live mode always holds to settlement: closing early would need a Kalshi sell
        order, which this strategy deliberately does not place.
        """
        cfg = self.config
        if cfg.hold_to_settlement or self.live_trading_enabled:
            return
        for symbol, position in list(state.positions.items()):
            if position.strategy != "kalshi_sports" or position.entry_price <= 0:
                continue
            price = state.last_prices.get(symbol, position.current_price)
            if price <= 0:
                continue
            change = (price - position.entry_price) / position.entry_price
            reason = None
            if cfg.take_profit_pct > 0 and change >= cfg.take_profit_pct:
                reason = "take_profit"
            elif cfg.stop_loss_pct > 0 and -change >= cfg.stop_loss_pct:
                reason = "stop_loss"
            if reason and broker.close_position(state, symbol=symbol, price=price, reason=reason):
                result["sells"].append(symbol)
