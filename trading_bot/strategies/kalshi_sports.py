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
import time
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
    BetDecision,
    Signal,
    DailySession,
    SourceCalibration,
    combine,
    drawdown_scale,
    evaluate_side,
    kalshi_fee,
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
# Kalshi player-stat series for football (one market per player and threshold, e.g. "Hurts: 250+ passing yards").
FOOTBALL_PROP_SERIES = (
    "KXNFLPASSYDS", "KXNFLPASSTDS", "KXNFLRSHYDS", "KXNFLRECYDS", "KXNFLREC", "KXNFLTD",
    "KXNFLSPREAD",  # alternate spreads: one per game in a combo, both sides offered
)
_SPREAD_SERIES = "KXNFLSPREAD"


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_TITLE_THRESHOLD = re.compile(r"^(\d+(?:\.\d+)?)\+\s*(.*)$")
_SPREAD_TITLE = re.compile(r"^(.*?) wins by over ([\d.]+)", re.IGNORECASE)


def _name_tokens(name: str) -> list[str]:
    """Lowercase ASCII words of a name without punctuation or Jr./III suffixes."""
    import unicodedata
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    return [t for t in re.findall(r"[a-z]+", plain) if t not in _NAME_SUFFIXES]


def _nickname(team: str) -> str:
    words = _name_tokens(team)
    return words[-1] if words else ""

def _prop_key(ticker: str, side: str) -> str:
    return f"{ticker}#{side}"


def sport_of(league: str) -> str:
    """'football/nfl' -> 'football': parlays may not mix sports."""
    return str(league).split("/")[0].lower()


def _ticker_player(ticker: str) -> str:
    """Player token of a prop ticker (KXNFLPASSYDS-26OCT11MINNO-NOTSHOUGH6-375 -> NOTSHOUGH6)."""
    parts = ticker.split("-")
    return parts[2] if len(parts) >= 4 else ""


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


class _BoardGame:
    """The few game fields ``_place_bet`` records, for bets chosen from the board."""

    odds_provider = ""

    def __init__(self, league: str, label: str, start_time: datetime) -> None:
        self.league = league
        self.label = label
        self.start_time = start_time


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
        self._series_fee: dict[str, float] = {}
        self.calibration = SourceCalibration(calibration_path, logger)
        self.session = DailySession(
            calibration_path.with_name("sports_session.json") if calibration_path else None, logger,
        )
        self.logger = logger.getChild("strategy.kalshi_sports")
        self.live_trading_enabled = live_trading_enabled
        # ticker -> every matched market from the last scan (for hand-picked bets).
        self.board: dict[str, dict[str, Any]] = {}
        # ticker -> football player-stat markets from the last /props search.
        self.props: dict[str, dict[str, Any]] = {}
        self._props_cache: tuple[float, dict[str, dict[str, Any]]] | None = None
        self._combo_collections: tuple[float, list[tuple[str, frozenset[str]]]] | None = None
        self._sleep = time.sleep
        # Live orders that failed or didn't fill this cycle, so /sports can say why nothing was bought.
        self._order_issues: list[dict[str, Any]] = []
        # Market series Kalshi refused for a shard/funding reason -> when it happened (monotonic seconds).
        self._unroutable: dict[str, float] = {}

    _UNROUTABLE_SECONDS = 6 * 3600.0

    @staticmethod
    def _is_shard_problem(exc: BaseException) -> bool:
        """The order failed because of where the market lives or which shard holds the funds, not the order itself."""
        response = getattr(exc, "response", None)
        text = (str(getattr(response, "text", "") or "") or str(exc)).lower()
        status = getattr(response, "status_code", None)
        return status == 404 or "shard" in text or "exchange user not found" in text or "insufficient_balance" in text

    @staticmethod
    def _describe_order_error(exc: BaseException) -> str:
        """Kalshi's own words for a rejected order (HTTP status and body), clipped."""
        response = getattr(exc, "response", None)
        if response is not None:
            body = str(getattr(response, "text", "") or "")[:500]
            return f"HTTP {getattr(response, 'status_code', '?')}: {body}".strip()
        return str(exc)[:220]

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
            "auto_parlays": [],
            "bankroll": {},
            "decisions": [],
            "can_trade_live": self.live_trading_enabled,
        }
        if not self.config.enabled:
            result["status"] = "disabled"
            return result
        self._order_issues = []

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
            self._auto_parlays(state, broker, now, result)
            for issue in self._order_issues[:6]:
                self._log_skip(result, issue["ticker"], issue["reason"], error=issue["error"])
        except Exception as exc:
            self.logger.error("Kalshi sports cycle failed: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)
        finally:
            result["source_accuracy"] = self.calibration.summary()
            self.calibration.save()
        return result

    def _account_held_events(self) -> set[str] | None:
        """Events the Kalshi account holds any position in (live only); None if it can't be read.

        Local records can be lost or late, and a second bet on a game already held is exactly what the
        one-bet-per-game rule forbids, so the account itself is the source of truth.
        """
        if not self.live_trading_enabled:
            return set()
        try:
            return {t.rsplit("-", 1)[0] if "-" in t else t for t in self.kalshi.get_held_tickers()}
        except Exception as exc:
            self.logger.warning("Could not read the account's positions: %s", exc)
            return None

    def _sports_series(self) -> set[str]:
        """Kalshi series this strategy trades (game winners, football props, configured extras)."""
        series: set[str] = set(FOOTBALL_PROP_SERIES)
        for spec in LEAGUES.values():
            series.update(spec.kalshi_series)
        for entry in self.config.series_overrides:
            if "=" in entry:
                series.add(entry.split("=", 1)[1].strip().upper())
        return series

    def _account_sports_exposure(self) -> float:
        """Dollars the account has at risk in sports markets right now, from Kalshi (live only).

        The bot's own position records vanish on a redeploy without a persistent volume, which would
        reset the exposure budget to zero; the account itself is the source of truth. Combo (parlay)
        positions have their own tickers and are only known from the bot's records. Best effort: if
        Kalshi can't be read this returns 0 and the bot's own records apply (the held-market guard
        still fails safe on its own).
        """
        if not self.live_trading_enabled:
            return 0.0
        reader = getattr(self.kalshi, "get_exposure_by_ticker", None)
        if reader is None:
            return 0.0
        try:
            series = self._sports_series()
            return sum(cost for ticker, cost in reader().items() if ticker.split("-", 1)[0].upper() in series)
        except Exception as exc:
            self.logger.warning("Could not read the account's exposure: %s", exc)
            return 0.0

    def _shard_balances(self) -> list[dict[str, Any]]:
        """Kalshi balance per exchange shard, for /sports (best effort)."""
        try:
            return [{"index": index, "balance": round(balance, 2)} for index, balance in self.kalshi.get_balance_breakdown()]
        except Exception as exc:
            self.logger.debug("Could not read the per-shard balance: %s", exc)
            return []

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

    def _fee_multiplier(self, series: str) -> float:
        """Kalshi's fee multiplier for a series (cached); 1.0 if it can't be read."""
        if series not in self._series_fee:
            multiplier = 1.0
            try:
                info = self.kalshi.get_series(series)
                fee_type = str(info.get("fee_type", "")).lower()
                raw = info.get("fee_multiplier")
                if "quadratic" in fee_type and raw not in (None, ""):
                    multiplier = max(0.0, float(raw))
                elif fee_type and "quadratic" not in fee_type:
                    self.logger.warning("Series %s has fee_type '%s'; assuming the standard fee.", series, fee_type)
            except Exception as exc:
                self.logger.debug("Could not read fee info for %s (assuming standard fee): %s", series, exc)
            self._series_fee[series] = multiplier
        return self._series_fee[series]

    def _fetch_markets(self, series: str, pages: int = 3, pause: float = 0.0) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        cursor = None
        for _ in range(pages):
            page, cursor = self.kalshi.get_markets(
                status="open", series_ticker=series, limit=200, cursor=cursor,
            )
            markets.extend(page)
            if not cursor:
                break
            if pause:
                self._sleep(pause)  # long prop listings trip Kalshi's rate limit when fetched back to back
        return markets

    def _build_candidates(self, now: datetime, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Score every matched market; return tradeable candidates (best edge per game)."""
        best_per_game: dict[str, dict[str, Any]] = {}
        self.board = {}

        for spec, series_list in self._league_specs():
            try:
                markets: list[dict[str, Any]] = []
                for series in series_list:
                    fee_multiplier = self._fee_multiplier(series)
                    for m in self._fetch_markets(series):
                        m["_fee_multiplier"] = fee_multiplier
                        markets.append(m)
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
                rank = (not evaluation.get("action"), evaluation["edge"])   # a real edge beats an action pick
                if current is None or rank > (not current.get("action"), current["edge"]):
                    best_per_game[key] = evaluation

        # Both team markets of a game yield the same pick; keep the better one for display.
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for pred in result["predictions"]:
            key = (pred["league"], pred["game"], pred["start"])
            if key not in unique or pred["edge"] > unique[key]["edge"]:
                unique[key] = pred
        result["predictions"] = sorted(unique.values(), key=lambda p: p["edge"], reverse=True)
        return sorted(best_per_game.values(), key=lambda c: (not c.get("action"), c["edge"]), reverse=True)

    def _evaluate_market(
        self,
        market: dict[str, Any],
        game: EspnGame,
        team_side: str,  # "home" | "away" | "draw": the outcome YES backs
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        cfg = self.config
        ticker = str(market.get("ticker", ""))

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
        mult = float(market.get("_fee_multiplier", 1.0))
        # Every matched market goes on the board for hand-picked bets, with or without a model view.
        board_entry: dict[str, Any] = {
            "ticker": ticker,
            "event_ticker": str(market.get("event_ticker", "")),
            "league": game.league,
            "game": game.label,
            "start": game.start_time.isoformat(),
            "three_way": bool(game.three_way),
            "yes_team": backed.display_name,
            "no_team": other.display_name,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "fee_multiplier": mult,
            "p_yes": None,
            "sd": None,
        }
        self.board[ticker] = board_entry

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
        board_entry["p_yes"] = post.p
        board_entry["sd"] = post.sd

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
                                     kelly_fraction=cfg.kelly_fraction, fee_multiplier=mult) if yes_ask > 0 else None
        no_decision = evaluate_side(post, "no", 1.0 - post.p, no_ask, fee_rate=cfg.fee_rate,
                                    kelly_fraction=cfg.kelly_fraction, fee_multiplier=mult) if no_ask > 0 else None
        if yes_decision:
            options.append((yes_decision, backed, _kalshi_size(market, "yes_ask_size")))
        if no_decision:
            options.append((no_decision, other, _kalshi_size(market, "no_ask_size")))

        if not options:
            # Nothing is mispriced. Pick the best-value side anyway (highest win probability
            # relative to its all-in cost); the strategy may place it as a small "action pick".
            sides = [(post.p, yes_ask, "yes", backed, "yes_ask_size"),
                     (1.0 - post.p, no_ask, "no", other, "no_ask_size")]
            priced = [o for o in sides if o[1] > 0]
            best_p, best_ask, best_side, best_team, size_key = max(
                priced or sides,
                key=lambda o: o[0] - o[1] - kalshi_fee(o[1], cfg.fee_rate, mult) if o[1] > 0 else -1.0,
            )
            fee = kalshi_fee(best_ask, cfg.fee_rate, mult)
            net = best_p - best_ask - fee if best_ask > 0 else 0.0
            prediction.update({"pick": best_team.display_name, "side": best_side,
                               "edge": round(best_p - best_ask, 4) if best_ask else 0.0,
                               "net_edge": None, "action_value": round(net, 4)})
            result["predictions"].append(prediction)
            self._log_skip(result, ticker, "no_edge_after_fees", game=game.label,
                           p=round(post.p, 4), sd=round(post.sd, 4))
            if not cfg.action_bets or best_ask <= 0 or best_ask >= 1 or best_p < cfg.action_min_win_prob:
                return None
            decision = BetDecision(
                side=best_side, price=best_ask, p=best_p, sd=max(post.sd, 0.012), fee=fee, net_edge=net,
                stake_fraction=0.0, confidence=0.0,
            )
            return {
                "market": market, "game": game, "decision": decision, "posterior": post,
                "pick": best_team.display_name, "depth": _kalshi_size(market, size_key),
                "edge": net, "action": True, "prediction": prediction,
            }

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
            "action": False,
            "prediction": prediction,
        }

    def _suggest_parlays(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parlay ideas built only from legs that already have a positive net edge.

        Candidates are one-per-game, so legs are independent. A parlay of
        negative-edge legs can't be profitable, so none are suggested then. Betting
        the legs separately with Kelly grows the bankroll faster than the same legs
        in a parlay unless Kalshi's quoted parlay price is below ``max_price``.
        """
        cfg = self.config
        legs = sorted((c for c in candidates if not c.get("action")), key=lambda c: c["edge"], reverse=True)
        suggestions: list[dict[str, Any]] = []
        for size in range(2, min(cfg.parlay_max_legs, len(legs)) + 1):
            chosen = legs[:size]
            value = parlay_value(
                [(c["decision"].p, c["decision"].sd) for c in chosen], fee_rate=cfg.fee_rate,
                fee_multiplier=max(float(c["market"].get("_fee_multiplier", 1.0)) for c in chosen),
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

    @staticmethod
    def _held_events(positions: list[Any]) -> set[str]:
        """Games already bet on, including every game inside a parlay."""
        held: set[str] = set()
        for p in positions:
            held.add(str(p.meta.get("event_ticker", "")))
            held.update(str(e) for e in p.meta.get("event_tickers", []))
        return held

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
        result["action_picks"] = {
            "enabled": bool(cfg.action_bets),
            "today": self.session.actions,
            "per_day": cfg.action_bets_per_day,
            "per_scan": cfg.action_max_per_scan,
        }
        open_positions = self._open_positions(state)
        exposure = max(sum(p.quantity * p.entry_price for p in open_positions), self._account_sports_exposure())
        exposure_cap = equity * cfg.capital_allocation_pct
        held_events = self._held_events(open_positions)
        result["bankroll"] = {
            "equity": round(equity, 2),
            "day_start": round(day["day_start"], 2),
            "day_pnl": round(day["day_pnl"], 2),
            "day_drawdown": round(day["drawdown"], 4),
            "stake_scale": round(scale, 3),
            "open_exposure": round(exposure, 2),
            "exposure_budget": round(exposure_cap, 2),
            # What the bot believes Kalshi has available to bet right now (live only), to compare with the app.
            "cash": round(broker.get_cash_balance(state, cash_bucket=bucket), 2) if bucket else None,
            "shards": self._shard_balances() if bucket else [],
        }

        account_events = self._account_held_events()
        if account_events is None:
            self._log_skip(result, "account", "positions_unavailable")
            return  # can't tell what the account already holds: place nothing rather than risk a duplicate
        held_events |= account_events

        # Size every candidate first, then fit them into the exposure budget together
        # (simultaneous Kelly): when opportunities pile up, each stake is trimmed
        # proportionally instead of the first ones eating the whole budget.
        slots = max(0, cfg.max_open_bets - len(open_positions))
        plan: list[tuple[dict[str, Any], float]] = []
        for cand in (c for c in candidates if not c.get("action")):
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

        if cfg.action_bets:
            self._place_action_picks(
                state, broker, [c for c in candidates if c.get("action")], now, result,
                equity=equity, scale=scale, bucket=bucket, exposure=exposure, exposure_cap=exposure_cap,
                held_events=held_events, open_positions=open_positions,
            )
            result["action_picks"]["today"] = self.session.actions

    def _place_action_picks(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        picks: list[dict[str, Any]],
        now: datetime,
        result: dict[str, Any],
        *,
        equity: float,
        scale: float,
        bucket: str | None,
        exposure: float,
        exposure_cap: float,
        held_events: set[str],
        open_positions: list[Any],
    ) -> None:
        """Small bets on the best-value side of the best games when nothing is mispriced.

        These have no edge after Kalshi's fee, so each costs a little in expectation. They
        are limited per scan and per day, sized as a small share of the bankroll, and ranked
        by value (win probability against all-in cost), so favourites and underdogs are chosen
        by the numbers, not by a rule.
        """
        cfg = self.config
        allowance = min(cfg.action_max_per_scan, max(0, cfg.action_bets_per_day - self.session.actions))
        if allowance <= 0 or not picks:
            if picks:
                self._log_skip(result, "action_picks", "action_picks_used_up_today",
                               placed=self.session.actions, per_day=cfg.action_bets_per_day)
            return
        placed = 0
        ranked = sorted(picks, key=lambda c: c["edge"], reverse=True)
        # Don't let the ranking turn into "always the underdog": at most half of the day's action
        # picks may back an underdog (win prob < 50%), as long as a favourite is available.
        favourites_available = any(c["decision"].p >= 0.5 for c in ranked)
        dog_cap = math.ceil(cfg.action_bets_per_day / 2)
        for cand in ranked:
            if placed >= allowance or len(open_positions) >= cfg.max_open_bets:
                break
            market = cand["market"]
            event_ticker = str(market.get("event_ticker", ""))
            if event_ticker and event_ticker in held_events:
                self._log_skip(result, str(market.get("ticker", "")), "already_bet_on_game")
                continue
            decision = cand["decision"]
            is_dog = decision.p < 0.5
            if is_dog and favourites_available and self.session.dogs >= dog_cap:
                self._log_skip(result, str(market.get("ticker", "")), "underdog_cap_for_today")
                continue
            notional = min(
                equity * scale * cfg.action_stake_pct,
                equity * cfg.max_position_pct,
                max(0.0, exposure_cap - exposure),
            )
            if cand["depth"] > 0:
                notional = min(notional, cand["depth"] * decision.price)
            if notional < cfg.min_trade_usd:
                self._log_skip(result, str(market.get("ticker", "")), "action_stake_below_min",
                               notional=round(notional, 2))
                continue
            position = self._place_bet(state, broker, cand, notional, now, bucket)
            if position is None:
                continue
            position.meta["action_bet"] = True
            exposure += position.quantity * position.entry_price
            open_positions.append(position)
            held_events.add(event_ticker)
            self.session.record_action(underdog=is_dog)
            placed += 1
            self._mark_placed(result, cand, decision)
            result["buys"].append(position.symbol)
            result["decisions"].append({
                "strategy": "kalshi_sports",
                "symbol": position.symbol,
                "action": "buy",
                "reason": "action_pick",
                "signals": {
                    "game": cand["game"].label,
                    "pick": cand["pick"],
                    "side": decision.side,
                    "model_prob": round(decision.p, 4),
                    "ask": round(decision.price, 4),
                    "fee": round(decision.fee, 4),
                    "expected_edge_after_fees": round(decision.net_edge, 4),
                    "notional": round(position.quantity * position.entry_price, 2),
                },
            })

    @staticmethod
    def _mark_placed(result: dict[str, Any], cand: dict[str, Any], decision: BetDecision) -> None:
        """Show an action pick that was actually placed as a bet in the picks list."""
        pred = cand["prediction"]
        pred.update({"net_edge": round(decision.net_edge, 4), "action": True,
                     "pick": cand["pick"], "side": decision.side})
        for i, existing in enumerate(result["predictions"]):
            if (existing["game"], existing["start"], existing["league"]) == (
                pred["game"], pred["start"], pred["league"]
            ):
                result["predictions"][i] = pred

    def _place_bet(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        cand: dict[str, Any],
        notional: float,
        now: datetime,
        bucket: str | None,
        quantity: int | None = None,
    ) -> Any | None:
        market, game, decision = cand["market"], cand["game"], cand["decision"]
        posterior = cand.get("posterior")
        cfg_fee_rate = self.config.fee_rate
        side, ask = decision.side, decision.price
        ticker = str(market.get("ticker", ""))
        symbol = f"{_SYMBOL_PREFIX}:{ticker}:{side.upper()}"
        if symbol in state.positions:
            return None
        series = ticker.split("-", 1)[0]
        if self.live_trading_enabled and not cand.get("user_bet"):
            since = self._unroutable.get(series)
            if since is not None and time.monotonic() - since < self._UNROUTABLE_SECONDS:
                return None  # Kalshi refused this series for a shard/funding reason: don't keep asking
            if sum(i["reason"] == "order_failed" and not i.get("shard") for i in self._order_issues) >= 3:
                return None  # Kalshi keeps rejecting orders for other reasons: stop instead of trying every game

        # Paper accounting books the fee into the entry price so simulated PnL is
        # net of costs; live, Kalshi debits the fee from the balance itself.
        entry_price = ask if self.live_trading_enabled else ask + decision.fee
        if self.live_trading_enabled:
            notional = min(notional, broker.get_cash_balance(state, cash_bucket=_CASH_BUCKET) * 0.95)
        if quantity is None:
            quantity = math.floor(notional / entry_price)
        if quantity < 1:
            return None
        if not self.live_trading_enabled:
            # Exact fee for this order size (rounding is per order, so bigger orders pay less per contract).
            per_contract = kalshi_fee(ask, cfg_fee_rate, float(market.get("_fee_multiplier", 1.0)), quantity)
            entry_price = ask + per_contract

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
                shard = self._is_shard_problem(exc)
                if shard:
                    self._unroutable[series] = time.monotonic()
                self._order_issues.append({
                    "ticker": ticker, "reason": "order_failed", "shard": shard,
                    "error": f"{ticker} {side} x{quantity} @ {ask:.2f}: {self._describe_order_error(exc)}"})
                return None
            order_id = str(order.get("order", {}).get("order_id", ""))
            filled = order_fill_count(order)
            if filled is not None:
                if filled < 1:
                    self.logger.info("Kalshi sports order for %s did not fill; no bet booked.", ticker)
                    self._order_issues.append({"ticker": ticker, "reason": "order_not_filled",
                                               "error": f"no contracts filled at {ask:.2f}"})
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
            "fee": round(entry_price - ask, 4) if not self.live_trading_enabled else round(decision.fee, 4),
            "sources": posterior.to_dict()["sources"] if posterior is not None else {},
            "odds_provider": game.odds_provider,
        }
        if cand.get("user_bet"):
            meta["user_bet"] = True
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
    # Hand-picked bets (Telegram /games, /bet, /confirm)
    # ------------------------------------------------------------------

    def refresh_board(self, now: datetime) -> list[dict[str, Any]]:
        """Re-scan ESPN + Kalshi and return the bettable games (no bets are placed)."""
        result: dict[str, Any] = {
            "decisions": [], "predictions": [], "games_scanned": 0,
            "markets_scanned": 0, "markets_matched": 0,
        }
        try:
            self._build_candidates(now, result)
        finally:
            self.calibration.save()
        return self.board_games(now)

    def board_games(self, now: datetime) -> list[dict[str, Any]]:
        """Group the last scan's markets into games with one option per team (or draw).

        Each option is the cheapest way to back that outcome: YES on its own market or, in
        two-way sports, NO on the opponent's market.
        """
        cfg = self.config
        games: dict[tuple[str, str, str], dict[str, Any]] = {}
        for entry in self.board.values():
            start = _parse_iso(entry["start"])
            if start is None or start <= now + timedelta(minutes=cfg.min_minutes_to_start):
                continue
            sides = [(entry["yes_team"], "yes", entry["yes_ask"], entry["p_yes"])]
            if not entry["three_way"]:
                sides.append((entry["no_team"], "no", entry["no_ask"],
                              None if entry["p_yes"] is None else 1.0 - entry["p_yes"]))
            game = games.setdefault((entry["league"], entry["game"], entry["start"]), {
                "league": entry["league"], "game": entry["game"], "start": entry["start"], "options": {},
            })
            for label, side, ask, p in sides:
                if not 0.0 < ask < 1.0:
                    continue
                current = game["options"].get(label)
                if current is not None and current["ask"] <= ask:
                    continue
                game["options"][label] = {
                    "label": label, "side": side, "ticker": entry["ticker"],
                    "event_ticker": entry["event_ticker"], "ask": ask,
                    "fee": kalshi_fee(ask, cfg.fee_rate, entry["fee_multiplier"]),
                    "p": p, "sd": entry["sd"],
                }
        out = []
        for game in games.values():
            game["options"] = sorted(game["options"].values(), key=lambda o: o["label"])
            if game["options"]:
                out.append(game)
        return sorted(out, key=lambda g: (g["start"], g["game"]))

    def quote_user_bet(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        *,
        ticker: str,
        side: str,
        amount: float | None,
        now: datetime,
    ) -> dict[str, Any]:
        """Price a hand-picked bet against the live order book and the safety limits.

        Returns ``{"ok": False, "reason": ...}`` or ``{"ok": True, ...}`` with the contract
        count, all-in cost and, when the model has a view, its probability for this side.
        """
        cfg = self.config
        side = side.lower()
        entry = self.board.get(ticker)
        if entry is None or side not in ("yes", "no"):
            return {"ok": False, "reason": "That game isn't on the board any more. Send /games to refresh."}
        start = _parse_iso(entry["start"])
        if start is None or start <= now:
            return {"ok": False, "reason": "That game has already started."}
        try:
            market = self.kalshi.get_market(ticker)
        except Exception as exc:
            return {"ok": False, "reason": f"Couldn't read the Kalshi market: {exc}"}
        if str(market.get("status", "")).lower() not in ("open", "active"):
            return {"ok": False, "reason": "Kalshi has closed this market."}
        ask = _kalshi_price(market, f"{side}_ask")
        if not 0.0 < ask < 1.0:
            return {"ok": False, "reason": "Nobody is selling that side on Kalshi right now."}
        mult = float(entry["fee_multiplier"])

        bucket = _CASH_BUCKET if self.live_trading_enabled else None
        held = self._held_events(self._open_positions(state)) | (self._account_held_events() or set())
        if entry["event_ticker"] and entry["event_ticker"] in held:
            return {"ok": False, "reason": "You already have a bet on this game."}
        equity = broker.portfolio_equity_for_bucket(state, cash_bucket=bucket)
        cash = broker.get_cash_balance(state, cash_bucket=bucket)
        stake = equity * cfg.user_bet_default_pct if amount is None else float(amount)
        limit = equity * cfg.user_bet_max_pct
        if stake < cfg.min_trade_usd:
            return {"ok": False, "reason": f"Minimum bet is ${cfg.min_trade_usd:.2f}."}
        if stake > limit + 1e-9:
            return {"ok": False, "reason": (
                f"${stake:.2f} is over the per-bet limit of ${limit:.2f} "
                f"({cfg.user_bet_max_pct:.0%} of your ${equity:.2f} bankroll).")}
        if stake > cash * 0.95 + 1e-9:
            return {"ok": False, "reason": f"Only ${cash:.2f} is available to bet."}

        quantity = math.floor(stake / (ask + kalshi_fee(ask, cfg.fee_rate, mult)))
        while quantity > 0 and quantity * ask + kalshi_fee(ask, cfg.fee_rate, mult, quantity) * quantity > stake + 1e-9:
            quantity -= 1
        depth = _kalshi_size(market, f"{side}_ask_size")
        if depth > 0:
            quantity = min(quantity, int(depth))
        if quantity < 1:
            return {"ok": False, "reason": f"${stake:.2f} doesn't buy a contract at ${ask:.2f} plus fee."}
        fee_each = kalshi_fee(ask, cfg.fee_rate, mult, quantity)
        p_yes = entry["p_yes"]
        p_side = None if p_yes is None else (p_yes if side == "yes" else 1.0 - p_yes)
        label = entry["yes_team"] if side == "yes" else entry["no_team"]
        return {
            "ok": True, "ticker": ticker, "event_ticker": entry["event_ticker"], "side": side,
            "pick": label, "game": entry["game"], "league": entry["league"], "start": entry["start"],
            "ask": ask, "fee_each": fee_each, "quantity": quantity,
            "cost": quantity * (ask + fee_each), "payout": float(quantity),
            "model_prob": p_side, "sd": entry["sd"], "equity": equity, "stake": stake, "live": self.live_trading_enabled, "market": market,
            "fee_multiplier": mult,
        }

    def place_user_bet(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        *,
        ticker: str,
        side: str,
        amount: float | None,
        now: datetime,
        max_ask: float | None = None,
    ) -> dict[str, Any]:
        """Re-quote a hand-picked bet and place it (live order on Kalshi, or paper booking).

        ``max_ask`` is the price the user agreed to; if the book has moved above it the
        bet is refused instead of placed.
        """
        quote = self.quote_user_bet(state, broker, ticker=ticker, side=side, amount=amount, now=now)
        if not quote["ok"]:
            return quote
        if max_ask is not None and quote["ask"] > max_ask + 1e-9:
            quote.pop("market", None)
            return {"ok": False, "reason": (
                f"The price moved from {max_ask * 100:.0f}¢ to {quote['ask'] * 100:.0f}¢. "
                "Nothing was bought; send /bet again to re-check."), "moved": True, "quote": quote}
        entry = self.board[ticker]
        ask, p_side = quote["ask"], quote["model_prob"]
        sd = float(quote["sd"] or 0.0)
        decision = BetDecision(
            side=quote["side"], price=ask, p=ask if p_side is None else p_side, sd=sd,
            fee=quote["fee_each"], net_edge=0.0 if p_side is None else p_side - ask - quote["fee_each"],
            stake_fraction=0.0, confidence=0.0,
        )
        start = _parse_iso(entry["start"]) or now
        cand = {
            "market": quote["market"],
            "game": _BoardGame(entry["league"], entry["game"], start),
            "decision": decision,
            "pick": quote["pick"],
            "user_bet": True,
        }
        bucket = _CASH_BUCKET if self.live_trading_enabled else None
        position = self._place_bet(
            state, broker, cand, quote["cost"], now, bucket, quantity=quote["quantity"],
        )
        if position is None:
            return {"ok": False, "reason": (
                "Kalshi didn't fill the order (the price moved or it was rejected). Nothing was bought.")}
        quote["filled"] = int(round(position.quantity))
        quote["symbol"] = position.symbol
        quote["spent"] = position.quantity * position.entry_price
        quote.pop("market", None)
        return quote

    _PROPS_CACHE_SECONDS = 600.0

    def _load_props(self, now: datetime) -> dict[str, dict[str, Any]]:
        """Every football player-stat market (YES side) for upcoming board games; cached for a few minutes."""
        cached = self._props_cache
        if cached is not None and time.monotonic() - cached[0] < self._PROPS_CACHE_SECONDS:
            return cached[1]
        if not self.board:
            self.refresh_board(now)
        games: dict[str, dict[str, Any]] = {}
        for entry in self.board.values():
            start = _parse_iso(entry["start"])
            if sport_of(entry["league"]) != "football" or start is None or start <= now + timedelta(
                    minutes=self.config.min_minutes_to_start):
                continue
            games[entry["event_ticker"].split("-", 1)[-1]] = entry
        found: dict[str, dict[str, Any]] = {}
        failed = False
        for series in FOOTBALL_PROP_SERIES:
            try:
                markets = self._fetch_markets(series, pages=10, pause=0.3)
            except Exception as exc:
                failed = True
                self.logger.warning("Prop fetch failed for %s: %s", series, exc)
                continue
            for market in markets:
                game = games.get(str(market.get("event_ticker", "")).split("-", 1)[-1])
                if game is None:
                    continue
                title = str(market.get("title") or market.get("yes_sub_title") or "")
                ticker = str(market.get("ticker", ""))
                spread = series == _SPREAD_SERIES
                if spread:
                    team, _, rest = title.partition(" wins by over ")
                    margin = rest.split(" ")[0]
                    labels = {"yes": (team.strip(), f"{team.strip()} win by over {margin}"),
                              "no": (team.strip(), f"{team.strip()} don't win by over {margin}")}
                else:
                    player, _, stat = title.partition(":")
                    labels = {"yes": (player.strip(), stat.strip() or title)}
                for side, (player, stat) in labels.items():
                    ask = _kalshi_price(market, f"{side}_ask")
                    if not 0.0 < ask < 1.0:
                        continue
                    bid = _kalshi_price(market, f"{side}_bid")
                    try:
                        volume = float(market.get("volume_fp") or market.get("volume") or 0.0)
                    except (TypeError, ValueError):
                        volume = 0.0
                    found[_prop_key(ticker, side)] = {
                        "ticker": ticker, "side": side, "event_ticker": str(market.get("event_ticker", "")),
                        "league": game["league"], "game": game["game"], "start": game["start"],
                        "player": "Spread" if spread else player, "stat": stat, "title": title,
                        "kind": "spread" if spread else "stat",
                        "yes_ask": ask, "yes_bid": bid, "mid": (bid + ask) / 2.0 if bid > 0 else ask,
                        "volume": volume,
                    }
        if not failed:  # a partial listing would hide players for the whole cache window
            self._props_cache = (time.monotonic(), found)
        return found

    def refresh_props(self, now: datetime, query: str) -> list[dict[str, Any]]:
        """Football player-stat markets matching ``query`` (words in the title or game; numbers match a threshold)."""
        universe = self._load_props(now)
        words = query.lower().split()
        found: dict[str, dict[str, Any]] = {}
        for ticker, prop in universe.items():
            hay = f"{prop['title']} {prop['stat']} {prop['game']}".lower()
            if all(re.search(rf"(?<![\d.]){re.escape(w)}\+", hay) if w.replace(".", "", 1).isdigit() else (w in hay)
                   for w in words):
                found[ticker] = prop
        self.props.update(found)
        return sorted(found.values(), key=lambda p: (p["start"], p["game"], p["player"], p["title"], p["side"]))

    def build_stack(self, now: datetime, query: str, legs: int, leg_prob: float) -> dict[str, Any]:
        """Assemble a same-game stack of high-probability legs, one per player (plus one spread).

        For each player the market whose Kalshi mid price is closest to ``leg_prob`` is chosen
        (so 9 legs near 85% multiply to roughly a +400 parlay), skipping thin markets with a wide
        bid/ask. Players are ranked by closeness to the target, then by trading volume.
        """
        universe = self._load_props(now)
        words = query.lower().split()
        games: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for prop in universe.values():
            if words and all(w in prop["game"].lower() for w in words):
                games.setdefault((prop["game"], prop["start"]), []).append(prop)
        if not games:
            return {"ok": False, "reason": f"No upcoming football game matches '{query}'. Check /games."}
        if len(games) > 1:
            names = ", ".join(sorted({g for g, _ in games})[:6])
            return {"ok": False, "reason": f"'{query}' matches several games ({names}). Name both teams."}
        (game, _), props = next(iter(games.items()))
        return self._stack_from_props(game, props, legs, leg_prob)

    def _stack_from_props(self, game: str, props: list[dict[str, Any]], legs: int, leg_prob: float) -> dict[str, Any]:
        best: dict[str, dict[str, Any]] = {}
        for prop in props:
            if prop["yes_bid"] <= 0 or prop["yes_ask"] - prop["yes_bid"] > 0.10 or not 0.03 < prop["mid"] < 0.97:
                continue  # one-sided or thin market: the mid price isn't a real price
            key = "spread" if prop["kind"] == "spread" else prop["player"]
            rank = (round(abs(prop["mid"] - leg_prob), 2), -prop["volume"])
            if key not in best or rank < best[key][0]:
                best[key] = (rank, prop)
        chosen = [prop for _, prop in sorted(best.values(), key=lambda row: row[0])[:legs]]
        if len(chosen) < 2:
            return {"ok": False, "reason": f"Not enough liquid player markets for {game} yet."}
        reference = 1.0
        for prop in chosen:
            reference *= prop["mid"]
        for prop in chosen:
            self.props[_prop_key(prop["ticker"], prop["side"])] = prop
        return {
            "ok": True, "game": game, "legs": [{"ticker": p["ticker"], "side": p["side"]} for p in chosen],
            "rows": chosen, "reference": reference, "leg_prob": leg_prob,
        }

    def match_slip_legs(self, now: datetime, slip: dict[str, Any]) -> dict[str, Any]:
        """Map legs read from a bet-slip picture onto Kalshi markets, exactly or not at all.

        A leg is only matched when Kalshi lists the same player, stat and threshold (or the same
        team and spread line). Near misses are reported with what Kalshi does offer rather than
        silently swapped for a different bet.
        """
        universe = self._load_props(now)
        board = list(self.board_games(now))
        # Narrow to the slip's game when both of its teams can be found on the board.
        slip_words = {w for w in _name_tokens(str(slip.get("game") or "")) if len(w) > 3}
        game_label = None
        best_hits = 1
        for game in board:
            hits = sum(1 for w in slip_words if w in game["game"].lower())
            if hits > best_hits:
                best_hits, game_label = hits, game["game"]
        props = [p for p in universe.values() if game_label is None or p["game"] == game_label]
        # No game on the slip (a typed list): use the game most legs point at only to settle
        # bare surnames, never to drop legs from other games.
        hint_game = game_label or self._infer_slip_game(slip, props)

        matched: list[dict[str, Any]] = []
        unmatched: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add(leg: dict[str, Any], prop: dict[str, Any]) -> None:
            key = (prop["ticker"], prop["side"])
            if key in seen:
                return
            seen.add(key)
            if prop["kind"] != "moneyline":  # game winners are priced from the board
                self.props[_prop_key(prop["ticker"], prop["side"])] = prop
            matched.append({
                "text": leg["text"], "ticker": prop["ticker"], "side": prop["side"],
                "pick": prop["stat"] if prop["kind"] in ("spread", "moneyline") else f"{prop['player']} {prop['stat']}",
                "ask": prop["yes_ask"],
            })

        for leg in slip.get("legs", []):
            text = leg.get("text") or leg.get("player") or leg.get("team") or "(unreadable leg)"
            kind = leg.get("type")
            if kind == "player_stat":
                reason = self._match_stat_leg(leg, props, add, hint_game)
            elif kind == "spread":
                reason = self._match_spread_leg(leg, props, add)
            elif kind == "moneyline":
                reason = self._match_moneyline_leg(leg, board, game_label, add)
            else:
                reason = "this kind of leg (totals and others) isn't offered here yet"
            if reason:
                unmatched.append({"text": text, "reason": reason})
        return {"matched": matched, "unmatched": unmatched, "game": game_label}

    @staticmethod
    def _infer_slip_game(slip: dict[str, Any], props: list[dict[str, Any]]) -> str | None:
        """The one game that the most legs point to (at least 2, strictly ahead of the rest), else None."""
        hits: dict[str, int] = {}
        games = {p["game"] for p in props}
        for leg in slip.get("legs", []):
            kind = leg.get("type")
            if kind == "player_stat" and leg.get("player"):
                wanted = set(_name_tokens(str(leg["player"])))
                for game in {p["game"] for p in props if p["kind"] == "stat" and wanted
                             and wanted <= set(_name_tokens(p["player"]))}:
                    hits[game] = hits.get(game, 0) + 1
            elif kind in ("spread", "moneyline") and leg.get("team"):
                nick = _nickname(str(leg["team"]))
                for game in games:
                    if nick and nick in _name_tokens(game):
                        hits[game] = hits.get(game, 0) + 1
        ranked = sorted(hits.items(), key=lambda row: -row[1])
        if ranked and ranked[0][1] >= 2 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            return ranked[0][0]
        return None

    def _match_stat_leg(
        self, leg: dict[str, Any], props: list[dict[str, Any]], add: Any, hint_game: str | None = None,
    ) -> str:
        player, family, threshold = leg.get("player"), leg.get("stat"), leg.get("threshold")
        if not player or not family or threshold is None:
            return "couldn't tell the player, stat and number"
        if leg.get("direction") == "under":
            return "unders aren't offered here (Kalshi's player markets are 'N or more')"
        wanted = _name_tokens(player)
        pool = [p for p in props if p["kind"] == "stat" and wanted and set(wanted) <= set(_name_tokens(p["player"]))]
        if not pool:
            return "no Kalshi market for that player in this game (props appear closer to kickoff)"
        same_stat: list[tuple[float, dict[str, Any]]] = []
        for prop in pool:
            m = _TITLE_THRESHOLD.match(prop["stat"])
            if m and m.group(2).strip().lower() == family:
                same_stat.append((float(m.group(1)), prop))
        players = {prop["player"] for _, prop in same_stat}
        if len(players) > 1 and hint_game:
            same_stat = [(v, p) for v, p in same_stat if p["game"] == hint_game] or same_stat
            players = {prop["player"] for _, prop in same_stat}
        if len(players) > 1:
            return f"more than one player matches ({', '.join(sorted(players)[:4])}); use the full name"
        if not same_stat:
            return f"Kalshi has no {family} market for them"
        for value, prop in same_stat:
            if abs(value - float(threshold)) < 1e-9:
                add(leg, prop)
                return ""
        have = ", ".join(f"{v:g}+" for v, _ in sorted(same_stat))
        return f"Kalshi doesn't list {threshold:g}+ {family} for them (it has {have})"

    def _match_spread_leg(self, leg: dict[str, Any], props: list[dict[str, Any]], add: Any) -> str:
        team, line = leg.get("team"), leg.get("line")
        if not team or line is None:
            return "couldn't tell the team and spread"
        nick = _nickname(team)
        margin = abs(float(line))
        underdog = float(line) > 0
        offered: list[float] = []
        for prop in props:
            if prop["kind"] != "spread":
                continue
            m = _SPREAD_TITLE.match(prop["title"])
            if not m or nick not in _name_tokens(prop["game"]):
                continue
            is_team = nick in _name_tokens(m.group(1))
            # Team +N: the opponent must not win by over N (NO on the opponent's market).
            # Team -N: the team must win by over N (YES on its own market).
            if underdog and not is_team and prop["side"] == "no" or not underdog and is_team and prop["side"] == "yes":
                offered.append(float(m.group(2)))
                if abs(float(m.group(2)) - margin) < 1e-9:
                    add(leg, prop)
                    return ""
        have = ", ".join(f"{v:g}" for v in sorted(set(offered))[:12])
        return f"Kalshi doesn't list a {margin:g}-point spread for {team} (lines: {have or 'none'})"

    def _match_moneyline_leg(self, leg: dict[str, Any], board: list[dict[str, Any]], game_label: str | None, add: Any) -> str:
        nick = _nickname(str(leg.get("team") or ""))
        for game in board:
            if game_label is not None and game["game"] != game_label:
                continue
            for option in game["options"]:
                if nick and nick in _name_tokens(option["label"]):
                    add(leg, {
                        "ticker": option["ticker"], "side": option["side"], "kind": "moneyline", "player": option["label"],
                        "stat": f"{option['label']} to win", "yes_ask": option["ask"], "title": option["label"],
                        "event_ticker": option["event_ticker"], "game": game["game"], "start": game["start"],
                    })
                    return ""
        return "no Kalshi winner market found for that team"

    # ------------------------------------------------------------------
    # Hand-built parlays (Telegram /parlay): Kalshi combo + request for quote
    # ------------------------------------------------------------------

    _COMBO_CACHE_SECONDS = 1800.0
    _QUOTE_POLL_SECONDS = 1.0
    _EXECUTION_WAIT_SECONDS = 15.0

    def _combo_collection_for(self, events: set[str]) -> str | None:
        """First open Kalshi combo collection whose allowed events include every leg's game."""
        cached = self._combo_collections
        if cached is None or time.monotonic() - cached[0] > self._COMBO_CACHE_SECONDS:
            collections = self.kalshi.get_multivariate_collections()
            cached = (time.monotonic(), [
                (str(c.get("collection_ticker", "")), frozenset(c.get("associated_event_tickers", [])))
                for c in collections
            ])
            self._combo_collections = cached
        for ticker, allowed in cached[1]:
            if ticker and events <= allowed:
                return ticker
        return None

    def _combo_fee_multiplier(self, collection_ticker: str) -> float:
        series = collection_ticker.rsplit("-R", 1)[0] if collection_ticker.endswith("-R") else collection_ticker
        return self._fee_multiplier(series)

    def quote_user_parlay(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        *,
        legs: list[dict[str, str]],
        amount: float | None,
        max_price: float | None,
        now: datetime,
        anchor_price: float | None = None,
    ) -> dict[str, Any]:
        """Check a hand-built parlay and work out the most we'd pay; places nothing.

        ``legs`` are ``{"ticker", "side"}`` rows taken from the board. The combo pays $1 only
        if every leg wins. ``max_price`` is the all-in ceiling per contract (dollars); by
        default the model's fair probability, so the bot never pays more than the parlay is worth.
        """
        cfg = self.config
        if not self.live_trading_enabled:
            return {"ok": False, "reason": "Parlays use live Kalshi quotes, so they need live trading (PAPER_MODE=false)."}
        if cfg.user_parlay_max_legs < 2:
            return {"ok": False, "reason": "Parlays are turned off (KALSHI_SPORTS_USER_PARLAY_MAX_LEGS)."}
        if not 2 <= len(legs) <= cfg.user_parlay_max_legs:
            return {"ok": False, "reason": f"A parlay needs 2 to {cfg.user_parlay_max_legs} legs."}

        held = self._held_events(self._open_positions(state)) | (self._account_held_events() or set())
        rows: list[dict[str, Any]] = []
        for leg in legs:
            ticker, side = leg["ticker"], str(leg["side"]).lower()
            entry = self.board.get(ticker)
            prop = self.props.get(_prop_key(ticker, side)) if entry is None else None
            if (entry is None and prop is None) or side not in ("yes", "no"):
                return {"ok": False, "reason": "A leg isn't on the board any more. Send /games (or /props) to refresh."}
            source = entry or prop
            start = _parse_iso(source["start"])
            if start is None or start <= now:
                return {"ok": False, "reason": f"{source['game']} has already started."}
            try:
                market = self.kalshi.get_market(ticker)
            except Exception as exc:
                return {"ok": False, "reason": f"Couldn't read the Kalshi market for {source['game']}: {exc}"}
            if str(market.get("status", "")).lower() not in ("open", "active"):
                return {"ok": False, "reason": f"Kalshi has closed betting on {source['game']}."}
            if source["event_ticker"] in held:
                return {"ok": False, "reason": f"You already have a bet on {source['game']}."}
            bid = _kalshi_price(market, f"{side}_bid")
            ask_now = _kalshi_price(market, f"{side}_ask")
            mid = (bid + ask_now) / 2.0 if bid > 0 and ask_now > 0 else ask_now
            if prop is not None:
                p, pick = None, (prop["stat"] if prop["kind"] == "spread" else f"{prop['player']} {prop['stat']}")
            else:
                p_yes = entry["p_yes"]
                p = None if p_yes is None else (p_yes if side == "yes" else 1.0 - p_yes)
                pick = entry["yes_team"] if side == "yes" else entry["no_team"]
            rows.append({
                "market_ticker": ticker, "event_ticker": source["event_ticker"], "side": side,
                "game": source["game"], "league": source["league"], "start": source["start"],
                "sport": sport_of(source["league"]), "kind": "prop" if prop is not None else "game",
                "pick": pick, "p": p, "ask": ask_now,
                # What the leg is worth: the model's probability, else Kalshi's own mid price.
                "ref": p if p is not None else mid,
            })
        sports = sorted({r["sport"] for r in rows})
        if len(sports) > 1:
            return {"ok": False, "reason": f"Parlays have to stay in one sport. You mixed {' and '.join(sports)}."}
        if len({(r["event_ticker"], _ticker_player(r["market_ticker"])) for r in rows}) != len(rows):
            return {"ok": False, "reason": (
                "Each leg needs a different game or a different player (two legs on the same game "
                "winner, or the same player's stat, aren't independent).")}

        fair = None
        if all(r["p"] is not None for r in rows):
            fair = 1.0
            for r in rows:
                fair *= r["p"]
        # Reference price: the model's fair value when it has a view on every leg, otherwise the
        # legs' own Kalshi mid prices multiplied (what the market says all of them hitting is worth).
        reference = None
        if all(r["ref"] > 0 for r in rows):
            reference = 1.0
            for r in rows:
                reference *= r["ref"]
        ceiling = max_price if max_price is not None else reference
        if max_price is None and anchor_price is not None and 0.0 < anchor_price < 1.0:
            # The price the user was shown elsewhere (a sportsbook slip) is one they'd already accept.
            ceiling = max(ceiling or 0.0, anchor_price)
        if ceiling is None:
            return {"ok": False, "reason": (
                "One of those markets has no price right now, so there's no reference value. "
                "Set your own ceiling, e.g. /parlay a + b 5 max 30 (cents).")}
        if not 0.0 < ceiling < 1.0:
            return {"ok": False, "reason": "The maximum price has to be between 0 and 100 cents."}

        collection = self._combo_collection_for({r["event_ticker"] for r in rows})
        if collection is None:
            return {"ok": False, "reason": "Kalshi doesn't offer a combo for those games together."}

        bucket = _CASH_BUCKET
        equity = broker.portfolio_equity_for_bucket(state, cash_bucket=bucket)
        cash = broker.get_cash_balance(state, cash_bucket=bucket)
        stake = equity * cfg.user_parlay_default_pct if amount is None else float(amount)
        limit = equity * cfg.user_parlay_max_pct
        if stake < cfg.min_trade_usd:
            return {"ok": False, "reason": f"Minimum bet is ${cfg.min_trade_usd:.2f}."}
        if stake > limit + 1e-9:
            return {"ok": False, "reason": (
                f"${stake:.2f} is over the parlay limit of ${limit:.2f} "
                f"({cfg.user_parlay_max_pct:.0%} of your ${equity:.2f} bankroll). Parlays rarely hit.")}
        if stake > cash * 0.95 + 1e-9:
            return {"ok": False, "reason": f"Only ${cash:.2f} is available to bet."}
        contracts = math.floor(stake / ceiling)
        if contracts < 1:
            return {"ok": False, "reason": f"${stake:.2f} doesn't buy a contract at up to ${ceiling:.2f}."}
        return {
            "ok": True, "kind": "parlay", "legs": rows, "collection": collection, "fair": fair, "reference": reference,
            "same_game": len({r["event_ticker"].split("-", 1)[-1] for r in rows}) < len(rows),
            "max_price": ceiling, "contracts": contracts, "stake": stake,
            "max_cost": contracts * ceiling, "payout": float(contracts), "equity": equity, "live": True,
        }

    def place_user_parlay(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        *,
        legs: list[dict[str, str]],
        amount: float | None,
        max_price: float | None,
        now: datetime,
        anchor_price: float | None = None,
        auto: bool = False,
    ) -> dict[str, Any]:
        """Ask Kalshi market makers for a price on the parlay and buy it if one is at or under the ceiling."""
        plan = self.quote_user_parlay(
            state, broker, legs=legs, amount=amount, max_price=max_price, now=now, anchor_price=anchor_price,
        )
        if not plan["ok"]:
            return plan
        cfg = self.config
        rows, contracts, ceiling = plan["legs"], plan["contracts"], plan["max_price"]
        multiplier = self._combo_fee_multiplier(plan["collection"])
        try:
            combo = self.kalshi.create_combo_market(plan["collection"], [
                {"market_ticker": r["market_ticker"], "event_ticker": r["event_ticker"], "side": r["side"]}
                for r in rows
            ])
            combo_ticker = str(combo["market_ticker"])
            rfq_id = self.kalshi.create_rfq(combo_ticker, contracts)
        except Exception as exc:
            self.logger.error("Parlay setup failed: %s", exc)
            return {"ok": False, "reason": f"Kalshi wouldn't set up that combo: {exc}"}
        if not rfq_id:
            return {"ok": False, "reason": "Kalshi didn't return a quote request id."}

        def all_in(quote: dict[str, Any]) -> float | None:
            """What a YES contract costs us from this quote (maker bids NO at no_bid), fee included."""
            try:
                no_bid = float(quote.get("no_bid_dollars") or 0.0)
                size = float(quote.get("contracts_fp") or contracts)
            except (TypeError, ValueError):
                return None
            if not 0.0 < no_bid < 1.0 or size + 1e-9 < contracts or str(quote.get("status", "open")) != "open":
                return None
            price = 1.0 - no_bid
            return price + kalshi_fee(price, cfg.fee_rate, multiplier, contracts)

        deadline = time.monotonic() + cfg.parlay_quote_wait_seconds
        best_seen: float | None = None
        accepted: dict[str, Any] | None = None
        tried: set[str] = set()
        while accepted is None:
            try:
                quotes = self.kalshi.get_rfq_quotes(rfq_id)
            except Exception as exc:
                self.logger.warning("Reading parlay quotes failed: %s", exc)
                quotes = []
            priced = sorted(
                ((all_in(q), q) for q in quotes if str(q.get("id", "")) not in tried),
                key=lambda row: (row[0] is None, row[0] or 0.0),
            )
            priced = [(c, q) for c, q in priced if c is not None]
            if priced:
                best_seen = priced[0][0] if best_seen is None else min(best_seen, priced[0][0])
            for cost, quote in priced:
                if cost > ceiling + 1e-9:
                    break
                tried.add(str(quote["id"]))
                try:
                    self.kalshi.accept_rfq_quote(rfq_id, str(quote["id"]), "no")
                except Exception as exc:
                    self.logger.warning("Accepting parlay quote failed: %s", exc)
                    continue
                accepted = {"quote": quote, "cost": cost}
                break
            if accepted is not None or time.monotonic() >= deadline:
                break
            self._sleep(self._QUOTE_POLL_SECONDS)

        if accepted is None:
            try:
                self.kalshi.delete_rfq(rfq_id)
            except Exception as exc:
                self.logger.debug("Could not delete RFQ %s: %s", rfq_id, exc)
            seen = (f" The best price offered was {best_seen * 100:.0f}¢ all-in." if best_seen is not None
                    else " No market maker quoted it.")
            return {"ok": False, "reason": (
                f"Nothing bought: no quote at or under {ceiling * 100:.0f}¢.{seen}"), "best_seen": best_seen}

        quote = accepted["quote"]
        status = self._await_execution(rfq_id, str(quote["id"]))
        if status != "executed":
            reason = ("The market maker didn't confirm in time, so nothing was bought."
                      if status in ("cancelled", "open", "accepted") else
                      "I couldn't confirm whether the parlay went through. Check /positions and your Kalshi app.")
            return {"ok": False, "reason": reason, "uncertain": status not in ("cancelled",), "status": status}

        price = 1.0 - float(quote["no_bid_dollars"])
        fee_each = kalshi_fee(price, cfg.fee_rate, multiplier, contracts)
        held_side, held_count, warning = self._verify_combo_position(combo_ticker, contracts)
        meta = {
            "ticker": combo_ticker, "event_ticker": str(combo.get("event_ticker", "")),
            "event_tickers": [r["event_ticker"] for r in rows], "side": held_side, "venue": "kalshi",
            "strategy": "kalshi_sports", "league": "parlay",
            "game": " + ".join(r["game"] for r in rows), "pick": " + ".join(r["pick"] for r in rows),
            "game_start": min(r["start"] for r in rows), "model_prob": round(plan["fair"] or 0.0, 4),
            "ask": round(price, 4), "fee": round(fee_each, 4), "parlay": True,
            **({"auto_parlay": True} if auto else {"user_bet": True}),
            "legs": [{"ticker": r["market_ticker"], "side": r["side"], "pick": r["pick"]} for r in rows],
            "kalshi_rfq_id": rfq_id, "kalshi_quote_id": str(quote["id"]),
        }
        entry_price = price if held_side == "yes" else 1.0 - price
        position = broker.open_position(
            state, symbol=f"{_SYMBOL_PREFIX}:{combo_ticker}:{held_side.upper()}", asset_type="prediction",
            strategy="kalshi_sports", price=entry_price, notional=held_count * entry_price,
            opened_at=now, meta=meta, cash_bucket=_CASH_BUCKET,
        )
        self.logger.info(
            "KALSHI SPORTS LIVE PARLAY %s x%d @ $%.3f (+$%.3f fee) fair=%s ceiling=%.3f",
            combo_ticker, held_count, price, fee_each,
            "n/a" if plan["fair"] is None else f"{plan['fair']:.3f}", ceiling,
        )
        return {
            "ok": True, "kind": "parlay", "combo_ticker": combo_ticker, "legs": rows,
            "filled": held_count, "price": price, "fee_each": fee_each, "fair": plan["fair"],
            "spent": held_count * (price + fee_each), "payout": float(held_count),
            "symbol": position.symbol if position is not None else "", "warning": warning,
        }

    def _await_execution(self, rfq_id: str, quote_id: str) -> str:
        """Poll an accepted quote until the maker confirms and it executes (or it is cancelled)."""
        deadline = time.monotonic() + self._EXECUTION_WAIT_SECONDS
        status = "accepted"
        while True:
            try:
                status = str(self.kalshi.get_rfq_quote(rfq_id, quote_id).get("status", status)).lower()
            except Exception as exc:
                self.logger.debug("Quote status check failed: %s", exc)
                status = "unknown"
            if status in ("executed", "cancelled") or time.monotonic() >= deadline:
                return status
            self._sleep(self._QUOTE_POLL_SECONDS)

    def _verify_combo_position(self, ticker: str, expected: int) -> tuple[str, int, str]:
        """What the account holds in the combo after execution: (side, contracts, warning).

        Accepting the wrong side of a quote would leave a NO position, so check it. If the
        positions can't be read the expected YES holding is assumed.
        """
        try:
            for pos in self.kalshi.get_positions():
                if pos.get("ticker") != ticker:
                    continue
                raw = pos.get("position_fp", pos.get("position"))
                signed = int(round(float(raw))) if raw not in (None, "") else 0
                if signed > 0:
                    return "yes", signed, ""
                if signed < 0:
                    return "no", -signed, (
                        "Kalshi shows you holding NO on this parlay, the opposite of what was intended. "
                        "Check it in the Kalshi app.")
        except Exception as exc:
            self.logger.warning("Could not verify the parlay position: %s", exc)
        return "yes", expected, "Couldn't confirm the position with Kalshi yet; /status will show it once it syncs."

    # ------------------------------------------------------------------
    # Automatic parlays
    # ------------------------------------------------------------------

    def _auto_parlays(self, state: PortfolioState, broker: PaperBroker, now: datetime, result: dict[str, Any]) -> None:
        """Buy same-game stacks of player-stat legs for upcoming football games, within hard caps.

        Reuses the hand-built parlay path (combo + request for quote), so a parlay is only bought at or
        under its price ceiling, the position is verified afterwards, and an uncertain outcome stops
        everything for the cycle. Caps: parlays per day, stake per parlay, open parlay cost, quote tries
        per game. Football only (that is where Kalshi lists player-stat markets).
        """
        cfg = self.config
        if not cfg.auto_parlays or not self.live_trading_enabled:
            return

        def skip(reason: str, **signals: Any) -> None:
            self._log_skip(result, "auto_parlay", reason, **signals)

        if cfg.auto_parlays_per_day - self.session.parlays <= 0:
            skip("parlays_per_day_reached", today=self.session.parlays)
            return
        open_positions = self._open_positions(state)
        equity = broker.portfolio_equity_for_bucket(state, cash_bucket=_CASH_BUCKET)
        open_cost = sum(p.quantity * p.entry_price for p in open_positions if p.meta.get("parlay"))
        stake = max(equity * cfg.auto_parlay_stake_pct, cfg.auto_parlay_min_stake_usd)
        cash = broker.get_cash_balance(state, cash_bucket=_CASH_BUCKET)
        if stake < cfg.min_trade_usd or stake > cash * 0.95:
            skip("parlay_stake_unaffordable", stake=round(stake, 2), cash=round(cash, 2))
            return
        budget = equity * cfg.auto_parlay_exposure_pct - open_cost
        if budget < stake:
            skip("parlay_exposure_budget", open_cost=round(open_cost, 2), budget=round(budget, 2))
            return
        try:
            universe = self._load_props(now)
        except Exception as exc:
            self.logger.warning("Auto parlays: could not load player props: %s", exc)
            skip("props_unavailable", error=str(exc)[:120])
            return

        def code_of(event_ticker: str) -> str:
            return event_ticker.split("-", 1)[-1]

        parlay_games = {code_of(str(e)) for p in open_positions if p.meta.get("parlay")
                        for e in p.meta.get("event_tickers", [])}
        games: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for prop in universe.values():
            games.setdefault((prop["game"], prop["start"]), []).append(prop)
        placed_any = False
        for (game, start_raw), props in sorted(games.items(), key=lambda row: row[0][1]):
            if cfg.auto_parlays_per_day - self.session.parlays <= 0 or budget < stake:
                break
            start = _parse_iso(start_raw)
            if start is None:
                continue
            minutes = (start - now).total_seconds() / 60.0
            if not cfg.auto_parlay_min_minutes <= minutes <= cfg.auto_parlay_hours_ahead * 60.0:
                continue
            code = code_of(props[0]["event_ticker"])
            if code in parlay_games:
                continue
            if self.session.parlay_tries.get(code, 0) >= cfg.auto_parlay_tries_per_game:
                skip("parlay_tries_used", game=game)
                continue
            waited = (now.timestamp() - self.session.parlay_last_try.get(code, 0.0)) / 60.0
            if waited < cfg.auto_parlay_retry_minutes:
                continue  # space the tries out: quotes are likelier closer to kickoff
            built = self._stack_from_props(game, props, cfg.auto_parlay_legs, cfg.auto_parlay_leg_prob)
            if not built.get("ok"):
                skip("no_liquid_stack", game=game, why=str(built.get("reason", ""))[:100])
                continue
            reference = float(built["reference"])
            ceiling = min(0.99, reference * (1.0 + cfg.auto_parlay_max_premium))
            self.session.record_parlay_try(code, now.timestamp())
            outcome = self.place_user_parlay(
                state, broker, legs=built["legs"], amount=stake, max_price=ceiling, now=now, auto=True,
            )
            if not outcome.get("ok"):
                skip("parlay_not_filled", game=game, why=str(outcome.get("reason", ""))[:160])
                if outcome.get("uncertain"):
                    # It may have gone through: count it and stop, rather than stack more on unknown state.
                    self.session.record_parlay()
                    self.logger.error("Auto parlay for %s has an uncertain outcome; stopping for this cycle.", game)
                    return
                continue
            self.session.record_parlay()
            budget -= float(outcome["spent"])
            parlay_games.add(code)
            placed_any = True
            symbol = str(outcome.get("symbol") or outcome["combo_ticker"])
            summary = {
                "game": game, "legs": [r["pick"] for r in outcome["legs"]], "contracts": outcome["filled"],
                "price": round(float(outcome["price"]), 4), "spent": round(float(outcome["spent"]), 2),
                "payout": round(float(outcome["payout"]), 2), "reference": round(reference, 4),
                "ceiling": round(ceiling, 4), "warning": outcome.get("warning", ""),
            }
            result["auto_parlays"].append(summary)
            result["buys"].append(symbol)
            result["decisions"].append({
                "strategy": "kalshi_sports", "symbol": symbol, "action": "buy", "reason": "auto_parlay",
                "signals": summary,
            })
        if not placed_any and not result["auto_parlays"]:
            skip("no_parlay_this_cycle", games_seen=len(games))

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
