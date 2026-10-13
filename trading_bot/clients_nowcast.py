"""Public GDP nowcast: the Atlanta Fed's GDPNow as published by FRED (a plain CSV, no key).

FRED's ``GDPNOW`` series has one row per quarter, dated the first day of the quarter. The last row is
the quarter currently being tracked and holds the latest nowcast (annualized percent change); earlier
rows keep each past quarter's final nowcast.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable

import requests

_MIN_VALUE, _MAX_VALUE = -15.0, 15.0     # a nowcast outside this is a parsing or unit mistake, not data


@dataclass(frozen=True)
class NowcastReading:
    quarter: str          # "2026Q3"
    value: float          # annualized percent change, e.g. 3.7
    observed_for: str     # the date FRED labels the row with (first day of the quarter)
    fetched_at: float     # epoch seconds when we read it
    source: str

    def age_hours(self, now: float | None = None) -> float:
        return max(0.0, ((time.time() if now is None else now) - self.fetched_at) / 3600.0)


def quarter_label(day: date) -> str:
    return f"{day.year}Q{(day.month - 1) // 3 + 1}"


def parse_fred_rows(text: str) -> list[tuple[date, float]]:
    """Every usable ``(date, value)`` row of a FRED CSV. Skips blanks, ``.`` and bad rows."""
    rows: list[tuple[date, float]] = []
    for line in text.splitlines()[1:]:                       # first line is the header
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[1] or parts[1] == ".":
            continue
        try:
            when = date.fromisoformat(parts[0])
            value = float(parts[1])
        except ValueError:
            continue
        if math.isfinite(value) and _MIN_VALUE <= value <= _MAX_VALUE:
            rows.append((when, value))
    return rows


def parse_fred_csv(text: str) -> tuple[date, float] | None:
    """The last usable row of a FRED CSV (the quarter being tracked), or None."""
    rows = parse_fred_rows(text)
    return rows[-1] if rows else None


class GdpNowClient:
    """Reads the latest GDPNow value; keeps the last good reading so a short outage is not fatal."""

    def __init__(
        self,
        *,
        url: str,
        timeout_seconds: int,
        max_age_hours: float,
        logger: logging.Logger,
        http_get: Callable[..., requests.Response] | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout_seconds
        self.max_age_hours = max_age_hours
        self.logger = logger.getChild("gdpnow")
        self._get = http_get or requests.get
        self._last: NowcastReading | None = None
        self._by_quarter: dict[str, NowcastReading] = {}
        self.last_error: str = ""

    def for_quarter(self, quarter: str, now: float | None = None) -> NowcastReading | None:
        """The nowcast row for one quarter ("2026Q3") from the last read, if it is recent enough."""
        reading = self._by_quarter.get(quarter)
        stamp = time.time() if now is None else now
        return reading if reading is not None and reading.age_hours(stamp) <= self.max_age_hours else None

    def latest(self, now: float | None = None) -> NowcastReading | None:
        """A fresh reading, else the last good one if it is recent enough, else None (place no bets)."""
        stamp = time.time() if now is None else now
        try:
            response = self._get(self.url, timeout=self.timeout, headers={"User-Agent": "zulubot-gdp-nowcast"})
            response.raise_for_status()
            rows = parse_fred_rows(response.text)
            if not rows:
                raise ValueError("no usable rows in the CSV")
            readings = {
                quarter_label(when): NowcastReading(
                    quarter=quarter_label(when), value=value, observed_for=when.isoformat(),
                    fetched_at=stamp, source="FRED GDPNOW",
                )
                for when, value in rows
            }
            self._by_quarter = readings
            self._last = readings[quarter_label(rows[-1][0])]
            self.last_error = ""
        except Exception as exc:  # network, HTTP or format problem: fall back, never guess
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            self.logger.warning("Could not read the GDP nowcast: %s", self.last_error)
        if self._last is not None and self._last.age_hours(stamp) <= self.max_age_hours:
            return self._last
        return None


def iso_to_epoch(value: str) -> float | None:
    """Kalshi timestamps like ``2026-10-30T14:00:00Z`` to epoch seconds."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None
