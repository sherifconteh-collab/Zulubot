"""ESPN public sports data client.

Uses ESPN's unauthenticated ``site.api.espn.com`` JSON endpoints to pull:

* the scoreboard (schedule, teams, records, start time, game state),
* sportsbook odds embedded in the scoreboard (moneyline / spread),
* ESPN's matchup predictor (BPI-style win projection) from the game summary.

These endpoints are undocumented and can change shape; every parser here is
defensive and returns ``None`` for any field it cannot read.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from trading_bot.clients import request_with_retry
from trading_bot.config import HttpRetryConfig

@dataclass(frozen=True)
class LeagueSpec:
    """One bettable league: where ESPN lists it and which Kalshi series trade it."""

    key: str
    espn_path: str                    # "<sport>/<league>" under site.api.espn.com
    kalshi_series: tuple[str, ...]    # game-winner series tickers
    groups: str | None = None         # ESPN `groups` filter (college divisions)
    three_way: bool = False           # draws possible (soccer): three markets per game


# ⚠️ Series tickers verified against Kalshi's live /series listing (Oct 2026); override or
# extend with KALSHI_SPORTS_SERIES="key=SERIES,...". Leagues out of season simply return
# no markets and are skipped without touching ESPN.
LEAGUES: dict[str, LeagueSpec] = {s.key: s for s in (
    LeagueSpec("nfl", "football/nfl", ("KXNFLGAME",)),
    LeagueSpec("ncaaf", "football/college-football", ("KXNCAAFGAME", "KXNCAAFBOWLGAME"), groups="80"),
    LeagueSpec("ncaaf-fcs", "football/college-football", ("KXNCAAFCSGAME",), groups="81"),
    LeagueSpec("ncaamb", "basketball/mens-college-basketball", ("KXNCAAMBGAME",), groups="50"),
    LeagueSpec("ncaawb", "basketball/womens-college-basketball", ("KXNCAAWBGAME",), groups="50"),
    LeagueSpec("nba", "basketball/nba", ("KXNBAGAME",)),
    LeagueSpec("wnba", "basketball/wnba", ("KXWNBAGAME",)),
    LeagueSpec("mlb", "baseball/mlb", ("KXMLBGAME",)),
    LeagueSpec("nhl", "hockey/nhl", ("KXNHLGAME",)),
    LeagueSpec("epl", "soccer/eng.1", ("KXEPLGAME",), three_way=True),
    LeagueSpec("laliga", "soccer/esp.1", ("KXLALIGAGAME",), three_way=True),
    LeagueSpec("bundesliga", "soccer/ger.1", ("KXBUNDESLIGAGAME",), three_way=True),
    LeagueSpec("seriea", "soccer/ita.1", ("KXSERIEAGAME",), three_way=True),
    LeagueSpec("ligue1", "soccer/fra.1", ("KXLIGUE1GAME",), three_way=True),
    LeagueSpec("mls", "soccer/usa.1", ("KXMLSGAME",), three_way=True),
    LeagueSpec("ligamx", "soccer/mex.1", ("KXLIGAMXGAME",), three_way=True),
    LeagueSpec("ucl", "soccer/uefa.champions", ("KXUCLGAME",), three_way=True),
    LeagueSpec("uel", "soccer/uefa.europa", ("KXUELGAME",), three_way=True),
    LeagueSpec("worldcup", "soccer/fifa.world", ("KXWCGAME",), three_way=True),
)}
DEFAULT_LEAGUES: list[str] = [
    "nfl", "ncaaf", "ncaaf-fcs", "ncaamb", "ncaawb", "nba", "wnba", "mlb", "nhl",
    "epl", "laliga", "bundesliga", "seriea", "ligue1", "mls", "ligamx", "ucl", "uel",
]


@dataclass(frozen=True)
class EspnTeam:
    team_id: str
    name: str                 # "Bills"
    location: str             # "Buffalo"
    display_name: str         # "Buffalo Bills"
    short_name: str           # "Bills" / "Buffalo"
    abbreviation: str         # "BUF"
    record: str = ""          # "10-3" (overall)
    win_pct: float | None = None
    games_played: int = 0

    def aliases(self) -> list[str]:
        raw = [self.display_name, self.location, self.name, self.short_name]
        seen: list[str] = []
        for alias in raw:
            alias = alias.strip().lower()
            if len(alias) >= 3 and alias not in seen:
                seen.append(alias)
        return seen


@dataclass
class EspnGame:
    game_id: str
    league: str               # ESPN path, e.g. "football/nfl"
    start_time: datetime
    state: str                # "pre" | "in" | "post"
    home: EspnTeam
    away: EspnTeam
    # Sportsbook American moneylines (None when ESPN shows no odds for the game)
    home_moneyline: float | None = None
    away_moneyline: float | None = None
    odds_provider: str = ""
    # Opening moneylines (None if unavailable); close-vs-open shows where bettors pushed the line
    home_open_moneyline: float | None = None
    away_open_moneyline: float | None = None
    # Draw moneyline (three-way sports only)
    draw_moneyline: float | None = None
    three_way: bool = False
    # ESPN predictor: P(home wins), 0-1
    predictor_home_prob: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.away.display_name} @ {self.home.display_name}"


# ---------------------------------------------------------------------------
# Pure helpers (also used by the strategy and tests)
# ---------------------------------------------------------------------------

def american_to_prob(odds: float | None) -> float | None:
    """Convert American moneyline odds to an implied probability (with vig)."""
    if odds is None or odds == 0:
        return None
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def devig_two_way(home_odds: float | None, away_odds: float | None) -> float | None:
    """Return the vig-free P(home) from a pair of American moneylines."""
    p_home = american_to_prob(home_odds)
    p_away = american_to_prob(away_odds)
    if p_home is None or p_away is None:
        return None
    total = p_home + p_away
    if total <= 0:
        return None
    return p_home / total


def devig_three_way(
    home_odds: float | None, draw_odds: float | None, away_odds: float | None,
) -> tuple[float, float, float] | None:
    """Vig-free (P(home), P(draw), P(away)) from three American moneylines."""
    probs = [american_to_prob(o) for o in (home_odds, draw_odds, away_odds)]
    if any(p is None for p in probs):
        return None
    total = sum(probs)  # type: ignore[arg-type]
    if total <= 0:
        return None
    return tuple(p / total for p in probs)  # type: ignore[return-value]


def parse_record(summary: str) -> tuple[float | None, int]:
    """Parse "10-3", "10-3-1" or "30-20-5" into (win_pct, games_played).

    A third number is treated as a tie / overtime loss worth half a win.
    """
    nums = [int(n) for n in re.findall(r"\d+", summary or "")]
    if len(nums) < 2:
        return None, 0
    wins, losses = nums[0], nums[1]
    ties = nums[2] if len(nums) > 2 else 0
    games = wins + losses + ties
    if games <= 0:
        return None, 0
    return (wins + 0.5 * ties) / games, games


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        text = str(value).strip().replace("−", "-")
        if text.upper() in {"EVEN", "PK"}:
            return 100.0
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_time(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        # ESPN uses "2025-09-28T17:00Z" (no seconds)
        text = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_team(competitor: dict[str, Any]) -> EspnTeam:
    team = competitor.get("team", {}) or {}
    record = ""
    for rec in competitor.get("records", []) or []:
        if rec.get("type") in ("total", None) or str(rec.get("name", "")).lower() == "overall":
            record = str(rec.get("summary", ""))
            break
    win_pct, games = parse_record(record)
    return EspnTeam(
        team_id=str(team.get("id", "")),
        name=str(team.get("name", "") or ""),
        location=str(team.get("location", "") or ""),
        display_name=str(team.get("displayName", "") or ""),
        short_name=str(team.get("shortDisplayName", "") or ""),
        abbreviation=str(team.get("abbreviation", "") or "").upper(),
        record=record,
        win_pct=win_pct,
        games_played=games,
    )


def _moneylines_from_odds(odds_entry: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract (home_ml, away_ml) from either ESPN odds layout."""
    home = _to_float((odds_entry.get("homeTeamOdds") or {}).get("moneyLine"))
    away = _to_float((odds_entry.get("awayTeamOdds") or {}).get("moneyLine"))
    if home is not None and away is not None:
        return home, away

    ml = odds_entry.get("moneyline") or {}
    for key in ("close", "current", "open"):
        h = _to_float(((ml.get("home") or {}).get(key) or {}).get("odds"))
        a = _to_float(((ml.get("away") or {}).get(key) or {}).get("odds"))
        if h is not None and a is not None:
            return h, a
    return None, None


def _draw_moneyline(odds_entry: dict[str, Any]) -> float | None:
    return _to_float((odds_entry.get("drawOdds") or {}).get("moneyLine"))


def _open_moneylines(odds_entry: dict[str, Any]) -> tuple[float | None, float | None]:
    ml = odds_entry.get("moneyline") or {}
    return (
        _to_float(((ml.get("home") or {}).get("open") or {}).get("odds")),
        _to_float(((ml.get("away") or {}).get("open") or {}).get("odds")),
    )


def line_movement(game: "EspnGame") -> float | None:
    """Change in the vig-free P(home) from the opening to the current line (+ = toward home)."""
    now = devig_two_way(game.home_moneyline, game.away_moneyline)
    then = devig_two_way(game.home_open_moneyline, game.away_open_moneyline)
    if now is None or then is None:
        return None
    return now - then


def parse_scoreboard(payload: dict[str, Any], league: str, three_way: bool = False) -> list[EspnGame]:
    """Turn an ESPN scoreboard payload into ``EspnGame`` objects."""
    games: list[EspnGame] = []
    for event in payload.get("events", []) or []:
        try:
            comps = event.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            home_raw = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away_raw = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if home_raw is None or away_raw is None:
                continue
            start = _parse_time(str(event.get("date") or comp.get("date") or ""))
            if start is None:
                continue
            state = str(((event.get("status") or comp.get("status") or {}).get("type") or {}).get("state", "pre"))

            home_ml = away_ml = home_open = away_open = draw_ml = None
            provider = ""
            for odds_entry in comp.get("odds") or []:
                home_ml, away_ml = _moneylines_from_odds(odds_entry)
                if home_ml is not None and away_ml is not None:
                    provider = str((odds_entry.get("provider") or {}).get("name", ""))
                    home_open, away_open = _open_moneylines(odds_entry)
                    draw_ml = _draw_moneyline(odds_entry)
                    break

            games.append(EspnGame(
                game_id=str(event.get("id", "")),
                league=league,
                start_time=start,
                state=state,
                home=_parse_team(home_raw),
                away=_parse_team(away_raw),
                home_moneyline=home_ml,
                away_moneyline=away_ml,
                odds_provider=provider,
                home_open_moneyline=home_open,
                away_open_moneyline=away_open,
                draw_moneyline=draw_ml,
                three_way=three_way,
            ))
        except Exception:  # one malformed event must not drop the whole slate
            continue
    return games


def parse_predictor(payload: dict[str, Any]) -> float | None:
    """Extract ESPN's P(home wins) from a game summary payload (0-1)."""
    predictor = payload.get("predictor") or {}
    home = _to_float((predictor.get("homeTeam") or {}).get("gameProjection"))
    away = _to_float((predictor.get("awayTeam") or {}).get("gameProjection"))
    if home is not None and away is not None and home + away > 0:
        return (home / (home + away))
    if home is not None:
        return home / 100.0
    if away is not None:
        return 1.0 - away / 100.0
    return None


def parse_summary_moneylines(
    payload: dict[str, Any],
) -> tuple[float | None, float | None, str, float | None, float | None, float | None]:
    """Fallback odds source: the summary's ``pickcenter`` list.

    Returns (home_ml, away_ml, provider, home_open_ml, away_open_ml, draw_ml).
    """
    for entry in payload.get("pickcenter") or []:
        home, away = _moneylines_from_odds(entry)
        if home is not None and away is not None:
            h_open, a_open = _open_moneylines(entry)
            return home, away, str((entry.get("provider") or {}).get("name", "")), h_open, a_open, _draw_moneyline(entry)
    return None, None, "", None, None, None


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class EspnClient:
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

    def __init__(
        self,
        *,
        timeout_seconds: int = 15,
        logger: logging.Logger,
        retry_config: HttpRetryConfig | None = None,
    ) -> None:
        self.timeout = timeout_seconds
        self.logger = logger.getChild("espn")
        self.retry_config = retry_config or HttpRetryConfig()
        self.session = requests.Session()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = request_with_retry(
            self.session,
            "GET",
            f"{self.BASE_URL}/{path}",
            params=params,
            timeout_seconds=self.timeout,
            logger=self.logger,
            retry_config=self.retry_config,
        )
        resp.raise_for_status()
        return resp.json()

    def get_scoreboard(
        self, league: str, date: str | None = None, *, groups: str | None = None, three_way: bool = False,
    ) -> list[EspnGame]:
        """Games for a league. ``date`` is ``YYYYMMDD``; ``groups`` filters college divisions."""
        params: dict[str, Any] = {"limit": 300}
        if date:
            params["dates"] = date
        if groups:
            params["groups"] = groups
        return parse_scoreboard(self._get(f"{league}/scoreboard", params=params), league, three_way)

    def enrich_game(self, game: EspnGame) -> EspnGame:
        """Fill in ESPN's predictor (and fallback odds) via the game summary."""
        try:
            payload = self._get(f"{game.league}/summary", params={"event": game.game_id})
        except Exception as exc:
            self.logger.debug("ESPN summary failed for %s: %s", game.game_id, exc)
            return game
        game.predictor_home_prob = parse_predictor(payload)
        if game.home_moneyline is None or game.away_moneyline is None:
            home, away, provider, h_open, a_open, draw = parse_summary_moneylines(payload)
            if home is not None and away is not None:
                game.home_moneyline, game.away_moneyline, game.odds_provider = home, away, provider
                game.home_open_moneyline, game.away_open_moneyline = h_open, a_open
                game.draw_moneyline = draw
        return game
