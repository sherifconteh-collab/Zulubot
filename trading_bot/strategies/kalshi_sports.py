"""Kalshi sports (game-winner) strategy driven by ESPN data and an adaptive model.

Strategy overview
-----------------
1. Pull upcoming games from ESPN's scoreboard for each configured league.
2. Pull open Kalshi game-winner markets for the matching series
   (e.g. ``KXNFLGAME``) and match each market to an ESPN game + team.
3. Treat each input as a noisy measurement of the true win probability and fuse
   them (``sports_model.combine``): the sportsbook closing line, Kalshi's own
   size-weighted order-book price, ESPN's predictor and (only when there is no
   line) a records model. Each source's weight comes from how accurate it has
   actually been on settled games (``SourceCalibration``), not fixed numbers.
4. Value YES and NO net of Kalshi's fee. Stake by Kelly shrunk by
   edge-versus-uncertainty, so there is no arbitrary minimum-edge cutoff: weak
   or noisy edges get tiny stakes, strong well-supported ones get larger ones.
5. Cap each bet at the liquidity resting at the ask, a per-bet % of equity and
   a total exposure budget. One bet per game.
6. Hold to settlement and close at Kalshi's settlement value ($1, $0, or $0.50
   on a tie).

Position symbols
----------------
  KS:{ticker}:{YES|NO}   asset_type="prediction", strategy="kalshi_sports"

Notes
-----
* Two-way model only: leagues where a draw is possible (soccer) are not
  supported by the default series map.
* A sportsbook line is mandatory by default: ESPN's predictor and team records
  alone are not sharper than Kalshi's own price.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from datetime import datetime, timedelta
from typing import Any

from trading_bot.broker import PaperBroker
from pathlib import Path

from trading_bot.clients_espn import (
    LEAGUES,
    EspnClient,
    EspnGame,
    EspnTeam,
    LeagueSpec,
    devig_three_way,
    devig_two_way,
    line_movement,
)
from trading_bot.clients_kalshi import KalshiClient, order_fill_count
from trading_bot.clients_polymarket_sports import PolyGame, PolyQuote
from trading_bot.config import KalshiSportsConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager
from trading_bot.sports_model import (
    Signal,
    DailySession,
    SourceCalibration,
    combine,
    drawdown_scale,
    evaluate_side,
    microprice,
    parlay_value,
)

_CASH_BUCKET = "kalshi"
_SYMBOL_PREFIX = "KS"
# Settle-check budget per cycle (get_market calls) when scoring past forecasts.
_MAX_OUTCOME_CHECKS = 25
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


def _parse_iso(raw: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _game_time_fits(market: dict[str, Any], game: EspnGame) -> bool:
    """True when the market's expected expiry is 0-10h after the game's start.

    False when the market carries no expiry: without it we can't tell which day's
    game it belongs to, so it is not traded.
    """
    expiry = _parse_iso(market.get("expected_expiration_time") or market.get("occurrence_datetime"))
    if expiry is None:
        return False
    return 0 <= (expiry - game.start_time).total_seconds() <= 10 * 3600


def is_tie_market(market: dict[str, Any]) -> bool:
    ticker = str(market.get("ticker", "")).upper()
    return (
        str(market.get("yes_sub_title", "") or "").strip().lower() == "tie"
        or ticker.endswith("-TIE")
        or "tie is the result" in str(market.get("title", "")).lower()
    )


def match_market_to_game(
    market: dict[str, Any],
    games: list[EspnGame],
    event_games: dict[str, EspnGame] | None = None,
) -> tuple[EspnGame, str] | None:
    """Return (game, "home"|"away"|"draw") for the outcome a market's YES side backs.

    Kalshi titles are just "<Team> wins", so games are matched by the team
    abbreviations in the event ticker ("...-26OCT12BUFLAR") or, failing that,
    by the YES team's name plus kickoff/expiry timing. "Tie" markets carry no
    team name at all, so they are resolved through the event their sibling
    team markets already matched (``event_games``) or the ticker code.
    Returns None when the market cannot be matched to exactly one game and outcome.
    """
    ticker = str(market.get("ticker", "")).upper()
    event_ticker = str(market.get("event_ticker", "")).upper()
    yes_sub = str(market.get("yes_sub_title", "") or "").lower()
    blob = " ".join(
        str(market.get(k, "") or "") for k in ("title", "subtitle", "yes_sub_title", "no_sub_title")
    ).lower()
    code = _event_code(event_ticker)

    if is_tie_market(market):
        if event_games and event_ticker in event_games:
            return event_games[event_ticker], "draw"
        pool = [
            g for g in games
            if code and g.home.abbreviation and g.away.abbreviation
            and code in (g.away.abbreviation + g.home.abbreviation, g.home.abbreviation + g.away.abbreviation)
            and _game_time_fits(market, g)
        ]
        return (pool[0], "draw") if len(pool) == 1 else None

    # Every match must also agree on *when* the game is: playoff series and
    # home-and-home pairs reuse the same teams on different days, and pairing a
    # market with another day's game would price it off the wrong odds.
    def hits(predicate) -> list[EspnGame]:
        return [g for g in games if _game_time_fits(market, g) and predicate(g)]

    def by_code(g: EspnGame) -> bool:
        h, a = g.home.abbreviation, g.away.abbreviation
        return bool(code and h and a and code in (a + h, h + a))

    def by_both_names(g: EspnGame) -> bool:
        return _team_in_text(g.home, blob) and _team_in_text(g.away, blob)

    def by_yes_team_and_time(g: EspnGame) -> bool:
        return bool(yes_sub) and (
            _team_in_text(g.home, yes_sub) or _team_in_text(g.away, yes_sub)
        ) and _game_time_fits(market, g)

    found: list[EspnGame] = []
    for predicate in (by_code, by_both_names, by_yes_team_and_time):
        found = hits(predicate)
        if found:
            break
    if len(found) != 1:
        return None
    game = found[0]

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


def _settlement_yes_value(market: dict[str, Any]) -> float | None:
    """YES payout in dollars once a market has settled (1, 0, or 0.5 on a tie); else None."""
    raw = market.get("settlement_value_dollars")
    settled = str(market.get("status", "")).lower() in _RESOLVED_STATUSES
    if raw not in (None, "") and (settled or str(market.get("result", "")).lower() in ("yes", "no")):
        try:
            return min(max(float(raw), 0.0), 1.0)
        except (TypeError, ValueError):
            pass
    result = str(market.get("result", "")).lower()
    if result == "yes":
        return 1.0
    if result == "no":
        return 0.0
    return None


def _kalshi_size(market: dict[str, Any], name: str) -> float:
    for key in (f"{name}_fp", name):
        raw = market.get(key)
        if raw not in (None, ""):
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    return 0.0


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


def link_polymarket(game: EspnGame, poly_games: list[PolyGame]) -> dict[str, PolyQuote] | None:
    """Polymarket quotes for ``game`` keyed home/away/draw, or None if it can't be matched.

    Needs both teams named in the Polymarket labels and a kickoff within 4 hours.
    Quotes are normalised so the outcomes sum to 1 (removing Polymarket's own spread bias).
    """
    best: dict[str, PolyQuote] | None = None
    for pg in poly_games:
        if abs((pg.start - game.start_time).total_seconds()) > 4 * 3600:
            continue
        home = [q for label, q in pg.quotes.items() if _team_in_text(game.home, label)]
        away = [q for label, q in pg.quotes.items() if _team_in_text(game.away, label)]
        if len(home) != 1 or len(away) != 1 or home[0] is away[0]:
            continue
        found = {"home": home[0], "away": away[0]}
        if "draw" in pg.quotes:
            found["draw"] = pg.quotes["draw"]
        if game.three_way and "draw" not in found:
            continue
        if best is not None:  # ambiguous: two Polymarket games fit
            return None
        best = found
    if best is None:
        return None
    total = sum(q.mid for q in best.values())
    if total <= 0:
        return None
    return {
        k: PolyQuote(mid=q.mid / total, bid=q.bid, ask=q.ask, liquidity=q.liquidity)
        for k, q in best.items()
    }


def sportsbook_outcome_probs(game: EspnGame) -> dict[str, float] | None:
    """Vig-free {home, away[, draw]} probabilities from the game's moneylines."""
    if game.three_way:
        probs = devig_three_way(game.home_moneyline, game.draw_moneyline, game.away_moneyline)
        return None if probs is None else dict(zip(("home", "draw", "away"), probs))
    book = devig_two_way(game.home_moneyline, game.away_moneyline)
    return None if book is None else {"home": book, "away": 1.0 - book}


def build_signals(
    game: EspnGame,
    outcome: str,
    market: dict[str, Any],
    config: KalshiSportsConfig,
    calibration: SourceCalibration,
) -> list[Signal]:
    """Independent estimates of P(YES) for a market (YES = ``outcome``: home, away or draw)."""
    signals: list[Signal] = []

    book = sportsbook_outcome_probs(game)
    if book is not None and outcome in book:
        signals.append(Signal("sportsbook", book[outcome], calibration.sd("sportsbook")))

    yes_bid, yes_ask = _kalshi_price(market, "yes_bid"), _kalshi_price(market, "yes_ask")
    micro = microprice(
        yes_bid, _kalshi_size(market, "yes_bid_size"), yes_ask, _kalshi_size(market, "yes_ask_size"),
    )
    if micro is not None:
        # A wide spread means the market is unsure of its own price.
        sd = max(calibration.sd("kalshi_book"), (yes_ask - yes_bid) / 2.0)
        signals.append(Signal("kalshi_book", micro, sd))

    poly = (game.extra or {}).get("polymarket")
    if poly and outcome in poly:
        quote = poly[outcome]
        signals.append(Signal("polymarket", quote.mid, max(calibration.sd("polymarket"), quote.spread / 2.0)))

    # ESPN's predictor and the records model are two-way only (no draw probability).
    if not game.three_way and outcome in ("home", "away"):
        if game.predictor_home_prob is not None:
            p = game.predictor_home_prob if outcome == "home" else 1.0 - game.predictor_home_prob
            signals.append(Signal("espn_predictor", p, calibration.sd("espn_predictor")))
        # Records are already priced into a sportsbook line; blending them in would
        # only shrink a sharp number toward .500. They are a fallback for no-line games.
        if book is None:
            rec = record_home_prob(game, config.home_advantage)
            if rec is not None:
                signals.append(Signal("record", rec if outcome == "home" else 1.0 - rec,
                                      calibration.sd("record")))
    return signals


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class _Label:
    """Stand-in for a team when a contract side has no single team (draw / not-a-team)."""

    def __init__(self, name: str) -> None:
        self.display_name = name


class KalshiSportsStrategy:
    """Bets on Kalshi game-winner markets using ESPN schedule, odds and projections."""

    def __init__(
        self,
        *,
        config: KalshiSportsConfig,
        kalshi_client: KalshiClient,
        espn_client: EspnClient,
        logger: logging.Logger,
        poly_client: Any | None = None,
        live_trading_enabled: bool = False,
        calibration_path: Path | None = None,
    ) -> None:
        self.config = config
        self.kalshi = kalshi_client
        self.espn = espn_client
        self.poly = poly_client
        self.calibration = SourceCalibration(calibration_path, logger)
        self.session = DailySession(
            calibration_path.with_name("sports_session.json") if calibration_path else None, logger,
        )
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
            "parlays": [],
            "bankroll": {},
            "decisions": [],
            "can_trade_live": self.live_trading_enabled,
        }
        if not self.config.enabled:
            result["status"] = "disabled"
            return result

        try:
            self._close_resolved_positions(state, broker, result)
            self._manage_open_positions(state, broker, result)
            self._score_past_forecasts()

            if not self.config.ignore_global_halt and (
                state.kill_switch_engaged or state.halt_new_trades_for_day
            ):
                result["status"] = "halted"
                result["decisions"].append({
                    "strategy": "kalshi_sports",
                    "symbol": "*",
                    "action": "skip_buy",
                    "reason": "risk_guardrails",
                })
                return result

            candidates = self._build_candidates(now, result)
            result["parlays"] = self._suggest_parlays(candidates)
            if result["games_scanned"] == 0 and result["markets_scanned"] == 0:
                result["status"] = "no_data"
            self._execute_candidates(state, broker, candidates, now, result)
        except Exception as exc:
            self.logger.error("Kalshi sports cycle failed: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)
        finally:
            result["source_accuracy"] = self.calibration.summary()
            self.calibration.save()
        return result

    def _score_past_forecasts(self) -> None:
        """Teach the model which sources were right: score forecasts for settled games."""
        for ticker in list(self.calibration.pending)[:_MAX_OUTCOME_CHECKS]:
            try:
                value = _settlement_yes_value(self.kalshi.get_market(ticker))
            except Exception as exc:
                self.logger.debug("Outcome check failed for %s: %s", ticker, exc)
                continue
            if value is not None:
                self.calibration.record_outcome(ticker, value)

    # ------------------------------------------------------------------
    # Discovery + prediction
    # ------------------------------------------------------------------

    def _league_specs(self) -> list[tuple[LeagueSpec, list[str]]]:
        """Configured leagues with their Kalshi series (built-ins plus overrides)."""
        extra: dict[str, list[str]] = {}
        for entry in self.config.series_overrides:
            key, _, series = entry.partition("=")
            if key.strip() and series.strip():
                extra.setdefault(key.strip().lower(), []).append(series.strip().upper())
        specs: list[tuple[LeagueSpec, list[str]]] = []
        for key in self.config.leagues:
            spec = LEAGUES.get(key.lower())
            if spec is None:
                self.logger.warning("Unknown sports league '%s' (known: %s)", key, ", ".join(sorted(LEAGUES)))
                continue
            specs.append((spec, list(spec.kalshi_series) + extra.get(spec.key, [])))
        return specs

    def _fetch_games(self, spec: LeagueSpec, now: datetime) -> list[EspnGame]:
        """Scoreboard for each day in the window (ESPN rejects multi-day ranges)."""
        days = math.ceil(self.config.max_hours_ahead / 24.0) + 1
        games: dict[str, EspnGame] = {}
        errors: list[Exception] = []
        for offset in range(days + 1):
            day = f"{now + timedelta(days=offset):%Y%m%d}"
            try:
                for game in self.espn.get_scoreboard(
                    spec.espn_path, date=day, groups=spec.groups, three_way=spec.three_way,
                ):
                    games[game.game_id] = game
            except Exception as exc:
                errors.append(exc)
        if errors and not games:
            raise errors[0]
        return list(games.values())

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

        for spec, series_list in self._league_specs():
            try:
                markets: list[dict[str, Any]] = []
                for series in series_list:
                    markets.extend(self._fetch_markets(series))
                if not markets:  # out of season: don't spend ESPN calls on it
                    continue
                games = self._fetch_games(spec, now)
            except Exception as exc:
                self.logger.warning("Sports data fetch failed for %s: %s", spec.key, exc)
                result["decisions"].append({
                    "strategy": "kalshi_sports", "symbol": spec.key,
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
            # Team markets first; "Tie" markets then ride on their sibling's event match.
            matched: list[tuple[dict[str, Any], EspnGame, str]] = []
            event_games: dict[str, EspnGame] = {}
            for market in (m for m in markets if not is_tie_market(m)):
                found = match_market_to_game(market, upcoming)
                if found is not None:
                    matched.append((market, found[0], found[1]))
                    event_games[str(market.get("event_ticker", "")).upper()] = found[0]
            for market in (m for m in markets if is_tie_market(m)):
                found = match_market_to_game(market, upcoming, event_games)
                if found is not None:
                    matched.append((market, found[0], found[1]))
            result["markets_matched"] += len(matched)

            if self.poly is not None and spec.poly_sport and matched:
                try:
                    poly_games = self.poly.get_upcoming_games(spec.poly_sport, now=now)
                except Exception as exc:
                    self.logger.debug("Polymarket fetch failed for %s: %s", spec.key, exc)
                    poly_games = []
                for game in {g.game_id: g for _, g, _ in matched}.values():
                    quotes = link_polymarket(game, poly_games)
                    if quotes:
                        game.extra["polymarket"] = quotes
                result["polymarket_linked"] = result.get("polymarket_linked", 0) + len(
                    {g.game_id for _, g, _ in matched if g.extra.get("polymarket")})

            enriched: set[str] = set()
            for market, game, outcome in matched:
                if game.game_id not in enriched:
                    self.espn.enrich_game(game)
                    enriched.add(game.game_id)

                evaluation = self._evaluate_market(market, game, outcome, result)
                if evaluation is None:
                    continue
                key = f"{game.league}:{game.game_id}"
                current = best_per_game.get(key)
                if current is None or evaluation["edge"] > current["edge"]:
                    best_per_game[key] = evaluation

        # Both team markets of a game yield the same pick; keep the better one for display.
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for pred in result["predictions"]:
            key = (pred["league"], pred["game"], pred["start"])
            if key not in unique or pred["edge"] > unique[key]["edge"]:
                unique[key] = pred
        result["predictions"] = sorted(unique.values(), key=lambda p: p["edge"], reverse=True)
        return sorted(best_per_game.values(), key=lambda c: c["edge"], reverse=True)

    def _evaluate_market(
        self,
        market: dict[str, Any],
        game: EspnGame,
        team_side: str,  # "home" | "away" | "draw": the outcome YES backs
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        cfg = self.config
        ticker = str(market.get("ticker", ""))
        signals = build_signals(game, team_side, market, cfg, self.calibration)
        if cfg.require_market_signal and not any(s.name == "sportsbook" for s in signals):
            self._log_skip(result, ticker, "no_sportsbook_line", game=game.label)
            return None
        post = combine(signals)
        if post is None:
            self._log_skip(result, ticker, "insufficient_data", game=game.label)
            return None
        # Remember every source's raw forecast so the model can learn who was right.
        self.calibration.record_forecast(ticker, {s.name: s.p for s in signals})
        if post.conflict:
            self._log_skip(result, ticker, "sources_conflict", game=game.label,
                           sources={s.name: round(s.p, 3) for s in post.used})
            return None

        # Display names for the two sides of the contract. With draws possible, NO on
        # "X wins" means "X doesn't win", not "the other team wins".
        if team_side == "draw":
            backed, other = _Label("Draw"), _Label("No draw")
        else:
            backed = game.home if team_side == "home" else game.away
            other = _Label(f"Not {backed.display_name}") if game.three_way else (
                game.away if team_side == "home" else game.home)
        yes_ask = _kalshi_price(market, "yes_ask")
        no_ask = _kalshi_price(market, "no_ask")
        move = line_movement(game) if team_side in ("home", "away") else None
        if move is not None and team_side == "away":
            move = -move  # express movement toward the YES team

        prediction: dict[str, Any] = {
            "game": game.label,
            "league": game.league,
            "start": game.start_time.isoformat(),
            "ticker": ticker,
            "yes_team": backed.display_name,
            "p_yes": round(post.p, 4),
            "sd": round(post.sd, 4),
            "sources": post.to_dict()["sources"],
            "dropped": post.dropped,
            "line_move_to_yes": None if move is None else round(move, 4),
            "yes_ask": round(yes_ask, 4),
            "no_ask": round(no_ask, 4),
        }

        options = []
        yes_decision = evaluate_side(post, "yes", post.p, yes_ask, fee_rate=cfg.fee_rate,
                                     kelly_fraction=cfg.kelly_fraction) if yes_ask > 0 else None
        no_decision = evaluate_side(post, "no", 1.0 - post.p, no_ask, fee_rate=cfg.fee_rate,
                                    kelly_fraction=cfg.kelly_fraction) if no_ask > 0 else None
        if yes_decision:
            options.append((yes_decision, backed, _kalshi_size(market, "yes_ask_size")))
        if no_decision:
            options.append((no_decision, other, _kalshi_size(market, "no_ask_size")))

        if not options:
            # Report the better-valued side even though it isn't a bet, for the picks list.
            best_p, best_ask, best_side, best_team = max(
                [(post.p, yes_ask, "yes", backed), (1.0 - post.p, no_ask, "no", other)],
                key=lambda o: o[0] - (o[1] or 1.0),
            )
            prediction.update({"pick": best_team.display_name, "side": best_side,
                               "edge": round(best_p - best_ask, 4) if best_ask else 0.0,
                               "net_edge": None})
            result["predictions"].append(prediction)
            self._log_skip(result, ticker, "no_edge_after_fees", game=game.label,
                           p=round(post.p, 4), sd=round(post.sd, 4))
            return None

        decision, pick_team, depth = max(options, key=lambda o: o[0].stake_fraction * o[0].net_edge)
        prediction.update({
            "pick": pick_team.display_name,
            "side": decision.side,
            "edge": round(decision.p - decision.price, 4),
            "net_edge": round(decision.net_edge, 4),
            "confidence": round(decision.confidence, 3),
            "stake_fraction": round(decision.stake_fraction, 5),
        })
        result["predictions"].append(prediction)
        return {
            "market": market,
            "game": game,
            "decision": decision,
            "posterior": post,
            "pick": pick_team.display_name,
            "depth": depth,
            "edge": decision.stake_fraction * decision.net_edge,  # expected growth: ranking key
        }

    def _suggest_parlays(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parlay ideas built only from legs that already have a positive net edge.

        Candidates are one-per-game, so legs are independent. A parlay of
        negative-edge legs can't be profitable, so none are suggested then. Betting
        the legs separately with Kelly grows the bankroll faster than the same legs
        in a parlay unless Kalshi's quoted parlay price is below ``max_price``.
        """
        cfg = self.config
        legs = sorted(candidates, key=lambda c: c["edge"], reverse=True)
        suggestions: list[dict[str, Any]] = []
        for size in range(2, min(cfg.parlay_max_legs, len(legs)) + 1):
            chosen = legs[:size]
            value = parlay_value(
                [(c["decision"].p, c["decision"].sd) for c in chosen], fee_rate=cfg.fee_rate,
            )
            suggestions.append({
                "legs": [
                    {"ticker": c["market"].get("ticker", ""), "game": c["game"].label,
                     "pick": c["pick"], "side": c["decision"].side,
                     "win_prob": round(c["decision"].p, 4), "kalshi_ask": round(c["decision"].price, 4)}
                    for c in chosen
                ],
                "fair_prob": round(value["fair"], 4),
                "uncertainty_sd": round(value["sd"], 4),
                "max_price": round(value["max_price"], 4),
                "fair_decimal_odds": round(value["fair_decimal_odds"], 2),
            })
        return suggestions

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
        # Bankroll = current wallet + open bets, re-read every cycle: wins grow it
        # (bigger stakes, more bets) and losses shrink it, with no fixed dollar amounts.
        equity = broker.portfolio_equity_for_bucket(state, cash_bucket=bucket)
        day = self.session.update(now.date().isoformat(), equity)
        scale = drawdown_scale(day["drawdown"], cfg.drawdown_throttle)
        open_positions = self._open_positions(state)
        exposure = sum(p.quantity * p.entry_price for p in open_positions)
        exposure_cap = equity * cfg.capital_allocation_pct
        held_events = {str(p.meta.get("event_ticker", "")) for p in open_positions}
        result["bankroll"] = {
            "equity": round(equity, 2),
            "day_start": round(day["day_start"], 2),
            "day_pnl": round(day["day_pnl"], 2),
            "day_drawdown": round(day["drawdown"], 4),
            "stake_scale": round(scale, 3),
            "open_exposure": round(exposure, 2),
            "exposure_budget": round(exposure_cap, 2),
        }

        # Size every candidate first, then fit them into the exposure budget together
        # (simultaneous Kelly): when opportunities pile up, each stake is trimmed
        # proportionally instead of the first ones eating the whole budget.
        slots = max(0, cfg.max_open_bets - len(open_positions))
        plan: list[tuple[dict[str, Any], float]] = []
        for cand in candidates:
            market = cand["market"]
            ticker = str(market.get("ticker", ""))
            event_ticker = str(market.get("event_ticker", ""))
            if event_ticker and event_ticker in held_events:
                self._log_skip(result, ticker, "already_bet_on_game")
                continue
            if len(plan) >= slots:
                self._log_skip(result, ticker, "max_open_bets")
                continue
            decision = cand["decision"]
            raw = min(equity * scale * decision.stake_fraction, equity * cfg.max_position_pct)
            if cand["depth"] > 0:  # can't buy more than what's offered at our price
                raw = min(raw, cand["depth"] * decision.price)
            plan.append((cand, raw))
        wanted = sum(raw for _, raw in plan)
        budget = max(0.0, exposure_cap - exposure)
        fit = min(1.0, budget / wanted) if wanted > 0 else 1.0

        for cand, raw in plan:
            market = cand["market"]
            ticker = str(market.get("ticker", ""))
            event_ticker = str(market.get("event_ticker", ""))
            decision = cand["decision"]
            notional = raw * fit
            if notional < cfg.min_trade_usd:
                self._log_skip(result, ticker, "stake_below_min", notional=round(notional, 2),
                               net_edge=round(decision.net_edge, 4), sd=round(decision.sd, 4))
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
                    "side": decision.side,
                    "model_prob": round(decision.p, 4),
                    "uncertainty_sd": round(decision.sd, 4),
                    "ask": round(decision.price, 4),
                    "fee": round(decision.fee, 4),
                    "net_edge": round(decision.net_edge, 4),
                    "confidence": round(decision.confidence, 3),
                    "notional": round(position.quantity * position.entry_price, 2),
                    "sources": cand["posterior"].to_dict()["sources"],
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
        market, game, decision = cand["market"], cand["game"], cand["decision"]
        side, ask = decision.side, decision.price
        ticker = str(market.get("ticker", ""))
        symbol = f"{_SYMBOL_PREFIX}:{ticker}:{side.upper()}"
        if symbol in state.positions:
            return None

        # Paper accounting books the fee into the entry price so simulated PnL is
        # net of costs; live, Kalshi debits the fee from the balance itself.
        entry_price = ask if self.live_trading_enabled else ask + decision.fee
        if self.live_trading_enabled:
            notional = min(notional, broker.get_cash_balance(state, cash_bucket=_CASH_BUCKET) * 0.95)
        quantity = math.floor(notional / entry_price)
        if quantity < 1:
            return None

        # Live: place the order first and book only what actually filled, so an
        # unfilled or partly filled order never leaves a phantom bet in state.
        order_id = ""
        if self.live_trading_enabled:
            try:
                order = self.kalshi.place_order(
                    ticker=ticker,
                    side=side,
                    action="buy",
                    count=quantity,
                    price_cents=round(ask * 100),
                    time_in_force="immediate_or_cancel",
                    client_order_id=str(uuid.uuid4()),
                )
            except Exception as exc:
                self.logger.error("Live Kalshi sports order failed for %s: %s", ticker, exc)
                return None
            order_id = str(order.get("order", {}).get("order_id", ""))
            filled = order_fill_count(order)
            if filled is not None:
                if filled < 1:
                    self.logger.info("Kalshi sports order for %s did not fill; no bet booked.", ticker)
                    return None
                quantity = min(quantity, filled)

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
            "model_prob": round(decision.p, 4),
            "uncertainty_sd": round(decision.sd, 4),
            "net_edge": round(decision.net_edge, 4),
            "ask": round(ask, 4),
            "fee": round(decision.fee, 4),
            "sources": cand["posterior"].to_dict()["sources"],
            "odds_provider": game.odds_provider,
        }
        if order_id:
            meta["kalshi_order_id"] = order_id
        position = broker.open_position(
            state,
            symbol=symbol,
            asset_type="prediction",
            strategy="kalshi_sports",
            price=entry_price,
            notional=quantity * entry_price,
            opened_at=now,
            meta=meta,
            cash_bucket=bucket,
        )
        if position is None:
            return None

        filled = int(round(position.quantity))
        self.logger.info(
            "KALSHI SPORTS %s %s %s on %s @ $%.2f (+$%.2f fee) x%d | p=%.1f%%±%.1f net edge=%.1f%%",
            "LIVE" if self.live_trading_enabled else "PAPER",
            side.upper(), cand["pick"], game.label, ask, decision.fee, filled,
            decision.p * 100, decision.sd * 100, decision.net_edge * 100,
        )

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
                yes_value = _settlement_yes_value(market)
                if yes_value is None:
                    if str(market.get("status", "")).lower() in _RESOLVED_STATUSES:
                        self.logger.warning("%s resolved without a result; leaving for manual review.", ticker)
                    continue
                side = str(position.meta.get("side", "yes")).lower()
                close_price = yes_value if side == "yes" else 1.0 - yes_value
                outcome = "tie" if yes_value == 0.5 else ("yes" if yes_value >= 1.0 else "no")
                broker.close_position(
                    state, symbol=symbol, price=close_price, reason=f"resolved:{outcome}",
                )
                result["resolved_closed"].append(symbol)
                result["decisions"].append({
                    "strategy": "kalshi_sports",
                    "symbol": symbol,
                    "action": "sell",
                    "reason": f"resolved:{outcome}",
                    "signals": {"won": close_price > position.entry_price, "payout": close_price,
                                "game": position.meta.get("game", "")},
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
