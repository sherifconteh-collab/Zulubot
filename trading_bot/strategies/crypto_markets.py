"""Kalshi crypto price markets ("Bitcoin above $92,300 at 3pm?"), bet only where a price model beats the quote.

The model is the plain one: the live Coinbase price is the center, and the price at the settlement time
is log-normal around it with a spread set by recent realized volatility. Coinbase is not the settlement
source (Kalshi settles on a CF Benchmarks index, a blend of exchanges), so a small basis error is added.

Two checks guard against the model being wrong in ways nobody can see from here:

* the realized-volatility spread must agree, within a factor of two, with the spread the market's own
  price ladder implies; if the two disagree the data or the model is wrong and nothing is bet,
* the ladder must hold enough mid-range prices to measure that implied spread at all, and our price
  may not sit more than a couple of model spreads away from the ladder's own center (a bigger gap is
  more likely a bad price feed than a real opportunity).

The shared ``forecast_engine`` does the rest: average with the market's price, require an edge after
Kalshi's fee, size by fractional Kelly and fail closed.
"""

from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime
from statistics import NormalDist
from typing import Any

from trading_bot.clients_kalshi import KalshiClient
from trading_bot.clients_nowcast import iso_to_epoch
from trading_bot.config import CryptoMarketsConfig
from trading_bot.strategies.forecast_engine import ForecastEngine, num

STRATEGY = "crypto_markets"
_NORMAL = NormalDist()
_PRODUCT_BY_SERIES = {
    "KXBTC": "BTC-USD", "KXBTCD": "BTC-USD", "KXETH": "ETH-USD", "KXETHD": "ETH-USD",
    "KXSOL": "SOL-USD", "KXSOLD": "SOL-USD", "KXXRP": "XRP-USD", "KXXRPD": "XRP-USD",
}
_BINANCE_BY_PRODUCT = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT", "XRP-USD": "XRPUSDT"}
_ABOVE = {"greater", "greater_or_equal"}


def _series_of_event(event_ticker: str) -> str:
    return event_ticker.split("-")[0]


def realized_sigma(closes: list[float]) -> float | None:
    """Standard deviation of log returns between consecutive closes (None if too little data)."""
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    if len(returns) < 12:
        return None
    return statistics.pstdev(returns)


def price_cdf(level: float, spot: float, sd: float) -> float:
    """P(price at settlement < level) for a log-normal price centered on ``spot`` (zero expected move)."""
    if level <= 0:
        return 0.0
    if sd <= 0:
        return 1.0 if level > spot else 0.0
    return _NORMAL.cdf((math.log(level / spot) + 0.5 * sd * sd) / sd)


def fit_ladder(markets: list[dict[str, Any]]) -> tuple[float, float] | None:
    """(center price, spread) the market's own 'above' ladder implies, from strikes priced 10-90 cents.

    If P(S > K) = mid then ln K = ln(center) + spread * z with z = inv_cdf(1 - mid), so a straight-line fit
    of ln K on z gives both. It does not use our spot price, so a stale ladder is measured as stale
    instead of being mistaken for a different volatility.
    """
    points: list[tuple[float, float]] = []
    for market in markets:
        if str(market.get("strike_type", "")).lower() not in _ABOVE or market.get("floor_strike") in (None, ""):
            continue
        bid, ask = num(market.get("yes_bid_dollars")), num(market.get("yes_ask_dollars"))
        strike = num(market.get("floor_strike"))
        if bid <= 0 or ask <= 0 or ask < bid or strike <= 0:
            continue
        mid = (bid + ask) / 2.0
        if 0.10 <= mid <= 0.90:
            points.append((_NORMAL.inv_cdf(1.0 - mid), math.log(strike)))
    if len(points) < 4:
        return None
    mean_z = sum(z for z, _ in points) / len(points)
    mean_k = sum(k for _, k in points) / len(points)
    var_z = sum((z - mean_z) ** 2 for z, _ in points)
    if var_z <= 1e-9:
        return None
    slope = sum((z - mean_z) * (k - mean_k) for z, k in points) / var_z
    if slope <= 0:
        return None
    return math.exp(mean_k - slope * mean_z), slope


def _money(value: float) -> str:
    return f"${value:,.2f}".rstrip("0").rstrip(".") if value < 100 else f"${value:,.0f}"


class CryptoPriceModel:
    strategy = STRATEGY
    summary_key = "spot"
    label = "crypto"

    def __init__(self, config: CryptoMarketsConfig, coinbase_client: Any, logger: logging.Logger) -> None:
        self.config = config
        self.coinbase = coinbase_client
        self.logger = logger.getChild("crypto_model")
        self.series = tuple(config.series)
        self._spot: dict[str, float] = {}
        self._sigma: dict[str, float] = {}      # per 5-minute log-return spread
        self.error = ""

    # ------------------------------------------------------------- forecast

    def _products(self) -> list[str]:
        return sorted({_PRODUCT_BY_SERIES[s] for s in self.series if s in _PRODUCT_BY_SERIES})

    def refresh(self, now: datetime) -> dict[str, Any] | None:
        self._spot, self._sigma, self.error = {}, {}, ""
        problems: list[str] = []
        for product in self._products():
            try:
                price = float(self.coinbase.get_spot_price_by_product_id(product))
                sigma = self._sigma_for(product, now)
            except Exception as exc:                       # network, HTTP or format problem: no bets on this coin
                problems.append(f"{product}: {type(exc).__name__}: {exc}"[:120])
                continue
            if price > 0 and sigma is not None:
                self._spot[product], self._sigma[product] = price, sigma
            else:
                problems.append(f"{product}: no usable price history")
        if not self._spot:
            self.error = "; ".join(problems) or "no crypto prices available"
            return None
        return {
            "coins": {p: {"price": self._spot[p], "sigma_5m_pct": round(self._sigma[p] * 100, 3)} for p in self._spot},
            "source": "Coinbase", "age_hours": 0.0, "problems": problems,
        }

    def _sigma_for(self, product: str, now: datetime) -> float | None:
        """Conservative 5-minute volatility: the larger of the last 3 hours and the last day, scaled up."""
        symbol = _BINANCE_BY_PRODUCT.get(product)
        if symbol is None:
            return None
        candles = self.coinbase.get_candles(symbol, granularity="FIVE_MINUTE", limit=288)
        rows = sorted(
            ((int(float(c["start"])), float(c["close"])) for c in candles
             if isinstance(c, dict) and c.get("start") is not None and c.get("close") is not None),
            key=lambda r: r[0],
        )
        if len(rows) < 60:
            return None
        if now.timestamp() - rows[-1][0] > self.config.max_candle_age_minutes * 60:
            return None                                     # stale history: do not trust it
        closes = [close for _, close in rows]
        day, recent = realized_sigma(closes), realized_sigma(closes[-37:])
        if day is None or recent is None:
            return None
        return max(day, recent, self.config.min_sigma_5m) * self.config.vol_multiplier

    # ---------------------------------------------------------------- events

    def event_context(self, event_markets: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
        first = event_markets[0]
        event = str(first.get("event_ticker", ""))
        product = _PRODUCT_BY_SERIES.get(_series_of_event(event), "")
        close = iso_to_epoch(str(first.get("close_time") or ""))
        ctx: dict[str, Any] = {
            "event": event, "quarter": "", "product": product, "hours": None, "sd": None, "spot": None,
            "reason": "", "can_value": True, "implied_sd": None,
        }
        if not product or close is None:
            ctx["reason"], ctx["can_value"] = "unreadable_event", False
            return ctx
        minutes = (close - now.timestamp()) / 60.0
        ctx["hours"] = minutes / 60.0
        spot, sigma = self._spot.get(product), self._sigma.get(product)
        if spot is None or sigma is None:
            ctx["reason"], ctx["can_value"] = "no_price", False
            return ctx
        sd = math.sqrt((sigma * math.sqrt(max(minutes, 1.0) / 5.0)) ** 2 + self.config.basis_sd ** 2)
        ctx.update(spot=spot, sd=sd)
        if minutes < self.config.min_minutes_to_close:
            ctx["reason"] = "too_close_to_close"
        elif minutes / 60.0 > self.config.max_hours_to_close:
            ctx["reason"] = "too_far_from_close"
        else:
            fit = fit_ladder(event_markets)
            if fit is None:
                ctx["reason"] = "cannot_check_volatility"
            else:
                center, implied = fit
                ctx["implied_sd"], ctx["ladder_center"] = implied, center
                if not 0.5 <= sd / implied <= 2.0:
                    ctx["reason"] = "volatility_mismatch"
                elif abs(math.log(spot / center)) > self.config.max_offset_sds * sd:
                    ctx["reason"] = "price_far_from_market"
        return ctx

    def probability(self, market: dict[str, Any], ctx: dict[str, Any]) -> tuple[float, float | None, str] | None:
        spot, sd = ctx.get("spot"), ctx.get("sd")
        if not spot or not sd:
            return None
        kind = str(market.get("strike_type", "")).lower()
        floor, cap = market.get("floor_strike"), market.get("cap_strike")
        if kind in _ABOVE and floor not in (None, ""):
            strike = num(floor)
            return 1.0 - price_cdf(strike, spot, sd), strike, f"above {_money(strike)}"
        if kind == "less" and cap not in (None, ""):
            strike = num(cap)
            return price_cdf(strike, spot, sd), strike, f"below {_money(strike)}"
        if kind == "between" and floor not in (None, "") and cap not in (None, ""):
            low, high = num(floor), num(cap)
            return max(0.0, price_cdf(high, spot, sd) - price_cdf(low, spot, sd)), low, f"{_money(low)}-{_money(high)}"
        return None

    def position_meta(self, row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        return {"spot": ctx["spot"], "product": ctx["product"]}

    def decision_signals(self, row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        return {"spot": ctx["spot"], "hours_to_close": round(ctx["hours"], 2), "sd_pct": round(ctx["sd"] * 100, 3)}


class CryptoMarketsStrategy(ForecastEngine):
    def __init__(
        self,
        *,
        config: CryptoMarketsConfig,
        kalshi_client: KalshiClient,
        coinbase_client: Any,
        logger: logging.Logger,
        live_trading_enabled: bool = False,
    ) -> None:
        super().__init__(
            config=config, model=CryptoPriceModel(config, coinbase_client, logger), kalshi_client=kalshi_client,
            logger=logger, live_trading_enabled=live_trading_enabled,
        )


__all__ = ["STRATEGY", "CryptoMarketsStrategy", "CryptoPriceModel", "fit_ladder", "price_cdf", "realized_sigma"]
