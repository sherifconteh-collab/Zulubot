"""Polymarket sports prices (real-money crowd) via the public Gamma API.

Polymarket lists moneyline markets per game. Their prices come from a different
pool of bettors than Kalshi or the sportsbooks, which makes them a useful
independent check on the win probability.

Two layouts exist:
* two-way sports: one market, outcomes ``[TeamA, TeamB]`` (bid/ask are for outcome 0);
* soccer: separate Yes/No markets, "Will X win on <date>?" and "...end in a draw?".

Only markets with a live two-sided quote are kept.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from trading_bot.clients import BaseHttpClient


@dataclass(frozen=True)
class PolyQuote:
    mid: float
    bid: float
    ask: float
    liquidity: float = 0.0

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)


@dataclass
class PolyGame:
    title: str
    start: datetime
    # label (lowercase team name, or "draw") -> quote for "that label happens"
    quotes: dict[str, PolyQuote] = field(default_factory=dict)


def _f(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _list(value: Any) -> list[Any]:
    """Gamma returns some list fields as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []


def _parse_time(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _quote(bid: float | None, ask: float | None, liquidity: float | None) -> PolyQuote | None:
    if bid is None or ask is None or not (0.0 < bid <= ask < 1.0 + 1e-9) or ask >= 1.0:
        return None
    return PolyQuote(mid=(bid + ask) / 2.0, bid=bid, ask=ask, liquidity=liquidity or 0.0)


_WIN_RE = re.compile(r"^will (.+?) win\b", re.IGNORECASE)


def parse_poly_events(events: list[dict[str, Any]]) -> list[PolyGame]:
    """Turn Gamma ``/events`` payloads into ``PolyGame`` objects with moneyline quotes."""
    games: list[PolyGame] = []
    for event in events or []:
        markets = [m for m in event.get("markets") or [] if m.get("sportsMarketType") == "moneyline"]
        if not markets:
            continue
        start = _parse_time(markets[0].get("gameStartTime") or event.get("gameStartTime"))
        if start is None:
            continue
        game = PolyGame(title=str(event.get("title", "")), start=start)
        for m in markets:
            outcomes = [str(o) for o in _list(m.get("outcomes"))]
            bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
            liq = _f(m.get("liquidityNum") or m.get("liquidity"))
            if len(outcomes) == 2 and {o.lower() for o in outcomes} == {"yes", "no"}:
                question = str(m.get("question", ""))
                if "draw" in question.lower():
                    label = "draw"
                else:
                    match = _WIN_RE.match(question)
                    if not match:
                        continue
                    label = match.group(1).strip().lower()
                yes_first = outcomes[0].lower() == "yes"
                q = _quote(bid, ask, liq) if yes_first else (
                    _quote(1 - ask, 1 - bid, liq) if bid is not None and ask is not None else None)
                if q is not None:
                    game.quotes[label] = q
            elif len(outcomes) == 2:
                q0 = _quote(bid, ask, liq)
                if q0 is None:
                    continue
                game.quotes[outcomes[0].lower()] = q0
                game.quotes[outcomes[1].lower()] = PolyQuote(
                    mid=1.0 - q0.mid, bid=1.0 - q0.ask, ask=1.0 - q0.bid, liquidity=q0.liquidity,
                )
        if game.quotes:
            games.append(game)
    return games


class PolymarketSportsClient(BaseHttpClient):
    """Upcoming game moneylines from https://gamma-api.polymarket.com (public)."""

    def __init__(self, *, timeout_seconds: int, logger: logging.Logger) -> None:
        super().__init__(
            base_url="https://gamma-api.polymarket.com",
            timeout_seconds=timeout_seconds,
            logger=logger.getChild("polymarket_sports"),
        )
        self._series: dict[str, str] | None = None

    def series_id(self, sport_code: str) -> str | None:
        if self._series is None:
            payload = self._get_json("sports")
            self._series = {
                str(item.get("sport")): str(item.get("series"))
                for item in payload if isinstance(item, dict) and item.get("series")
            }
        return self._series.get(sport_code)

    def get_upcoming_games(self, sport_code: str, *, now: datetime, limit: int = 200) -> list[PolyGame]:
        series = self.series_id(sport_code)
        if series is None:
            return []
        payload = self._get_json("events", {
            "series_id": series,
            "active": "true",
            "closed": "false",
            "limit": limit,
            "order": "startDate",
            "ascending": "true",
            "end_date_min": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        return parse_poly_events(payload if isinstance(payload, list) else [])
