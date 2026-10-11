"""Read a typed list of bet legs into the same structure the screenshot reader produces.

No AI and no network: a plain line-by-line parser for lists such as::

    Make this bet at $10
    • Brian Thomas Jr. 2+ receptions
    • Lawrence 150+ passing yards
    • the Eagles +16.5 spread

Only lines that fit a known leg shape count as legs. Everything else is returned as the
``header`` (the stake and wording such as "bet $10"), so the text is never executed or
interpreted beyond picking the legs it names. The caller still matches every leg exactly
against Kalshi, so a leg that is ambiguous or unlisted is reported rather than guessed.
"""

from __future__ import annotations

import re
from typing import Any

MIN_LEGS = 2
_MAX_LEGS = 20
_BULLETS = "•·●▪‣*-–—"

_STAT_ALIASES: tuple[tuple[str, str], ...] = (
    ("passing touchdowns", "passing touchdowns"), ("passing touchdown", "passing touchdowns"),
    ("passing tds", "passing touchdowns"), ("passing td", "passing touchdowns"),
    ("pass tds", "passing touchdowns"), ("pass td", "passing touchdowns"),
    ("passing yards", "passing yards"), ("passing yds", "passing yards"), ("passing yd", "passing yards"),
    ("pass yards", "passing yards"), ("pass yds", "passing yards"),
    ("rushing yards", "rushing yards"), ("rushing yds", "rushing yards"), ("rushing yd", "rushing yards"),
    ("rush yards", "rushing yards"), ("rush yds", "rushing yards"),
    ("receiving yards", "receiving yards"), ("receiving yds", "receiving yards"), ("receiving yd", "receiving yards"),
    ("rec yards", "receiving yards"), ("rec yds", "receiving yards"),
    ("receptions", "receptions"), ("reception", "receptions"), ("catches", "receptions"),
    ("touchdowns", "touchdowns"), ("touchdown", "touchdowns"), ("tds", "touchdowns"), ("td", "touchdowns"),
)
_STATS = dict(_STAT_ALIASES)

_NAME = r"[A-Za-z][A-Za-z.'’\- ]*?"
_STAT_LINE = re.compile(rf"^(?P<name>{_NAME})\s+(?:over\s+)?(?P<thr>\d+(?:\.\d+)?)\s*\+\s*(?P<stat>[A-Za-z ]+)$", re.IGNORECASE)
_ALIAS_ALT = "|".join(re.escape(a) for a, _ in sorted(_STAT_ALIASES, key=lambda row: -len(row[0])))
_STAT_FIRST = re.compile(
    rf"^(?P<name>{_NAME})\s+(?:alt(?:ernate)?\s+)?(?P<stat>{_ALIAS_ALT})\s*:?\s*(?P<thr>\d+(?:\.\d+)?)\s*\+$",
    re.IGNORECASE,
)
_OVER_UNDER = re.compile(
    rf"^(?P<name>{_NAME})\s+(?P<dir>over|under|o|u)\s+(?P<line>\d+\.5)\s+(?P<stat>[A-Za-z ]+)$", re.IGNORECASE,
)
_ANYTIME_TD = re.compile(
    rf"^(?P<name>{_NAME})\s+(?:anytime\s+(?:td|touchdown)s?|to\s+score(?:\s+a)?\s+(?:td|touchdown))$", re.IGNORECASE,
)
_SPREAD_LINE = re.compile(
    r"^(?:the\s+)?(?P<team>[A-Za-z][A-Za-z.'’ ]*?)\s*\(?\s*(?P<line>[+\-−]\s*\d+(?:\.\d+)?)\s*\)?"
    r"(?:\s+(?:alt(?:ernate)?\s+)?(?:point\s+)?spread)?$",
    re.IGNORECASE,
)
_MONEYLINE_LINE = re.compile(
    r"^(?:the\s+)?(?P<team>[A-Za-z][A-Za-z.'’ ]*?)\s+(?:ml|moneyline|to\s+win|win)$", re.IGNORECASE,
)
_SEPARATORS = re.compile(r"[\u2022\u25cf;]|,\s+|\s+\+\s+|\s+and\s+|\s+&\s+", re.IGNORECASE)
# "bet $10 on Lawrence 150+ passing yards": the stake clause up front, then the first leg.
_BET_PREFIX = re.compile(
    r"^.*?\b(?:bet|place|put|wager|play|lock|take)\b[^\n]*?"
    r"(?:\$\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:dollars?|bucks))\s*(?:on|with|for|:|-)?\s+",
    re.IGNORECASE,
)
_LEADING_NUMBER = re.compile(r"^\d{1,2}[.)]\s+")
_TRAILING_ODDS = re.compile(r"\s*[(\[]?\s*[+\-−]\d{3,4}\s*[)\]]?\s*$")


_TRAILING_VERB = re.compile(r"\s+(?:to\s+)?(?:record|have|get|hit|make|catch|reach|score|go\s+for)\s*$", re.IGNORECASE)


def _clean_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw).strip(" .-:\u2013\u2014")
    name = _TRAILING_VERB.sub("", name).strip(" .-:\u2013\u2014")
    return re.sub(r"^(?:alt|alternate)\s+", "", name, flags=re.IGNORECASE)


def _stat_family(raw: str) -> str | None:
    words = [w for w in re.findall(r"[a-z]+", raw.lower()) if w not in ("alt", "alternate", "player")]
    return _STATS.get(" ".join(words))


def _parse_line(line: str) -> dict[str, Any] | None:
    text = _LEADING_NUMBER.sub("", line.strip())
    text = text.strip(_BULLETS + " \t")
    if not text or len(text) > 120:
        return None
    anytime = _ANYTIME_TD.match(text)
    if anytime:
        return _leg("player_stat", text, player=_clean_name(anytime.group("name")), stat="touchdowns", threshold=1.0)
    stat_line = _STAT_LINE.match(text)
    if stat_line:
        family = _stat_family(stat_line.group("stat"))
        if family is None:
            return None
        return _leg(
            "player_stat", text, player=_clean_name(stat_line.group("name")), stat=family,
            threshold=float(stat_line.group("thr")), direction="over",
        )
    over_under = _OVER_UNDER.match(text)
    if over_under:
        family = _stat_family(over_under.group("stat"))
        if family is None:
            return None
        line_value = float(over_under.group("line"))
        under = over_under.group("dir").lower() in ("under", "u")
        # Stats are whole numbers, so "over 149.5" is exactly "150+".
        return _leg(
            "player_stat", text, player=_clean_name(over_under.group("name")), stat=family,
            threshold=line_value - 0.5 if under else line_value + 0.5, direction="under" if under else "over",
        )
    stat_first = _STAT_FIRST.match(text)
    if stat_first:
        return _leg(
            "player_stat", text, player=_clean_name(stat_first.group("name")), stat=_STATS[stat_first.group("stat").lower()],
            threshold=float(stat_first.group("thr")), direction="over",
        )
    stripped = _TRAILING_ODDS.sub("", text) if re.search(r"spread|\(", text) else text
    spread = _SPREAD_LINE.match(stripped)
    if spread:
        line_value = float(re.sub(r"\s+", "", spread.group("line")).replace("−", "-"))
        return _leg("spread", text, team=_clean_name(spread.group("team")), line=line_value)
    moneyline = _MONEYLINE_LINE.match(text)
    if moneyline:
        return _leg("moneyline", text, team=_clean_name(moneyline.group("team")))
    return None


def _leg(kind: str, text: str, **fields: Any) -> dict[str, Any]:
    leg: dict[str, Any] = {
        "type": kind, "player": None, "team": None, "stat": None, "threshold": None,
        "line": None, "direction": None, "text": text[:120],
    }
    leg.update(fields)
    return leg


def parse_slip_text(text: str) -> dict[str, Any] | None:
    """Return ``{"game", "odds", "legs", "header"}`` for a typed leg list, or None if it isn't one.

    A message counts as a slip only when at least ``MIN_LEGS`` lines each fit a leg shape; one
    stray matching line in ordinary chat is not enough.
    """
    pieces: list[str] = []
    for raw in (text or "").splitlines():
        pieces.extend(part for part in _SEPARATORS.split(raw) if part.strip())
    legs: list[dict[str, Any]] = []
    header: list[str] = []
    for piece in pieces:
        leg = _parse_line(piece)
        if leg is None:
            prefix = _BET_PREFIX.match(piece)
            leg = _parse_line(piece[prefix.end():]) if prefix else None
            if leg is not None:
                header.append(piece[:prefix.end()].strip())
        if leg is not None and len(legs) < _MAX_LEGS:
            legs.append(leg)
        else:
            header.append(piece.strip())
    if len(legs) < MIN_LEGS:
        return None
    return {"game": None, "odds": None, "legs": legs, "header": " ".join(h for h in header if h)}
