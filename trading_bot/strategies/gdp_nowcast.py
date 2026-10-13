"""GDP markets on Kalshi, traded only when a forecast-based probability beats the price.

The old predictions rule bought whatever was cheap. This strategy has an actual view: the Atlanta
Fed's GDPNow estimate for the quarter, turned into a probability for each Kalshi strike ("GDP above
3.5%") using the nowcast's known forecast error. The shared ``forecast_engine`` then asks whether that
probability, averaged with the market's own price, still beats the price you would pay after Kalshi's
fee, and fails closed on anything missing or odd.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Any

from trading_bot.clients_kalshi import KalshiClient
from trading_bot.clients_nowcast import GdpNowClient, NowcastReading, iso_to_epoch
from trading_bot.config import GdpNowcastConfig
from trading_bot.strategies.forecast_engine import ForecastEngine, kelly_stake, num, score_market  # noqa: F401

STRATEGY = "gdp_nowcast"
_QUARTER = re.compile(r"\bQ([1-4])\s+(\d{4})\b")


def _num(value: Any) -> float:
    return num(value)


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


def _supported(market: dict[str, Any]) -> bool:
    return str(market.get("strike_type", "")).lower() == "greater" and market.get("floor_strike") not in (None, "")


def evaluate_market(
    market: dict[str, Any],
    reading: NowcastReading,
    days_to_release: float,
    cfg: GdpNowcastConfig,
    fee_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Score one strike. ``reason`` is empty when it is a bet, otherwise why it is not."""
    if not _supported(market):
        row = ForecastEngine._blocked_row(market, "unsupported_market")
        row["strike"] = None
        return row
    strike = _num(market.get("floor_strike"))
    p_model = prob_above(strike, reading.value, lead_sd(cfg, days_to_release))
    return score_market(market, p_model, cfg, fee_multiplier, strike=strike, label=f">{strike:g}%")


class GdpModel:
    """GDPNow turned into P(real GDP growth above each strike)."""

    strategy = STRATEGY
    summary_key = "nowcast"
    label = "GDP"

    def __init__(self, config: GdpNowcastConfig, nowcast_client: GdpNowClient) -> None:
        self.config = config
        self.nowcast = nowcast_client
        self.series = (config.series,)
        self._reading: NowcastReading | None = None

    @property
    def error(self) -> str:
        return getattr(self.nowcast, "last_error", "") or ""

    def refresh(self, now: datetime) -> dict[str, Any] | None:
        self._reading = self.nowcast.latest()
        reading = self._reading
        if reading is None:
            return None
        return {"quarter": reading.quarter, "value": reading.value, "source": reading.source,
                "age_hours": round(reading.age_hours(), 1)}

    def event_context(self, event_markets: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
        """Timing and quarter facts for one event; ``reason`` says why the whole event is skipped."""
        reading = self._reading
        first = event_markets[0]
        quarter = market_quarter(first)
        release = iso_to_epoch(str(first.get("expected_expiration_time") or first.get("close_time") or ""))
        close = iso_to_epoch(str(first.get("close_time") or ""))
        # Match the market to its own quarter's row, so the order in which FRED lists quarters never matters.
        own = reading if reading is not None and reading.quarter == quarter else (
            getattr(self.nowcast, "for_quarter", lambda _q: None)(quarter) if quarter else None)
        ctx: dict[str, Any] = {
            "event": str(first.get("event_ticker", "")), "quarter": quarter, "days": None, "sd": None, "reason": "",
            "nowcast": own.value if own is not None else None, "_reading": own, "can_value": True,
        }
        if release is None or not quarter:
            ctx["reason"], ctx["can_value"] = "unreadable_event", False
            return ctx
        days = (release - now.timestamp()) / 86400.0
        ctx["days"], ctx["sd"] = days, lead_sd(self.config, max(0.0, days))
        if own is None:
            ctx["reason"], ctx["can_value"] = "nowcast_is_for_another_quarter", False
        elif days > self.config.max_days_to_release:
            ctx["reason"] = "too_far_from_release"
        elif close is not None and (close - now.timestamp()) / 3600.0 < self.config.min_hours_to_close:
            ctx["reason"] = "too_close_to_close"
        elif days <= 0:
            ctx["reason"] = "past_release"
        return ctx

    def probability(self, market: dict[str, Any], ctx: dict[str, Any]) -> tuple[float, float | None, str] | None:
        if not _supported(market) or ctx.get("_reading") is None or ctx.get("sd") is None:
            return None
        strike = _num(market.get("floor_strike"))
        return prob_above(strike, ctx["_reading"].value, ctx["sd"]), strike, f">{strike:g}%"

    def position_meta(self, row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        return {"nowcast": ctx["_reading"].value}

    def decision_signals(self, row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        return {"nowcast": ctx["_reading"].value, "days_to_release": round(ctx["days"], 1)}


class GdpNowcastStrategy(ForecastEngine):
    def __init__(
        self,
        *,
        config: GdpNowcastConfig,
        kalshi_client: KalshiClient,
        nowcast_client: GdpNowClient,
        logger: logging.Logger,
        live_trading_enabled: bool = False,
    ) -> None:
        super().__init__(
            config=config, model=GdpModel(config, nowcast_client), kalshi_client=kalshi_client,
            logger=logger, live_trading_enabled=live_trading_enabled,
        )
        self.nowcast = nowcast_client
