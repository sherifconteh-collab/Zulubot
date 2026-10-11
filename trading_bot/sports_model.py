"""Adaptive probability model + bet sizing for sports markets.

Why this exists
---------------
Fixed rules like "bet when edge >= 6 points" are arbitrary: 3 points of edge
on a deep, tight market backed by a sharp sportsbook line can be a better bet
than 8 points off a noisy model. Instead this module:

1. Treats every input (sportsbook line, Kalshi's own order-book price, ESPN's
   predictor, team records) as a *noisy measurement* of the true win
   probability, each with its own uncertainty (``sd``).
2. Combines them by inverse-variance weighting in log-odds space, so reliable
   sources dominate and a missing source costs nothing.
3. Learns each source's real reliability from settled games
   (``SourceCalibration``), starting from sensible priors.
4. Sizes bets by *net* edge (after Kalshi's fee) shrunk by how large that edge
   is relative to the posterior uncertainty. Tiny or shaky edges get tiny
   stakes automatically; there is no hard edge cutoff.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Prior standard deviation of each source's error, in probability points.
# These only matter until SourceCalibration has seen real outcomes.
PRIOR_SD: dict[str, float] = {
    "sportsbook": 0.025,       # closing line from a major book: sharpest public number
    "kalshi_book": 0.030,      # Kalshi's own size-weighted order-book price
    "polymarket": 0.030,       # Polymarket mid: a separate real-money crowd
    "espn_predictor": 0.060,   # ESPN matchup predictor: ignores much market info
    "record": 0.090,           # records + home field: weakest
}
_MIN_SD = 0.012                # never trust any source tighter than this
_PRIOR_STRENGTH = 40.0         # pseudo-observations behind the prior sd
_OUTLIER_Z = 3.5               # source this many sds from the others is treated as a data fault


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def kalshi_fee(price: float, fee_rate: float = 0.07, multiplier: float = 1.0, count: int = 1) -> float:
    """Taker fee per contract, in dollars.

    Kalshi's schedule: ``fee = ceil(multiplier * 0.07 * C * P * (1 - P))`` for an order of ``C``
    contracts at price ``P``. ``multiplier`` is set per series (e.g. 0.5 for MLB games) and is
    read from Kalshi's API. A retail balance settles to whole cents, so the order total is
    rounded up to the next cent and spread across the contracts. With ``count=1`` this is a
    conservative upper bound: larger orders round less per contract. Maker (resting) orders
    pay a lower fee, but the bot only places immediate-or-cancel taker orders.
    """
    if price <= 0 or price >= 1 or fee_rate <= 0 or multiplier <= 0 or count < 1:
        return 0.0
    total = multiplier * fee_rate * count * price * (1 - price)
    return math.ceil(total * 100 - 1e-9) / 100.0 / count


def microprice(bid: float, bid_size: float, ask: float, ask_size: float) -> float | None:
    """Size-weighted fair price: leans toward the side with *less* resting size.

    A thin ask with a heavy bid means buyers are pressing, so fair value sits
    nearer the ask. Returns None when either side of the book is empty.
    """
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    total = bid_size + ask_size
    if total <= 0:
        return (bid + ask) / 2.0
    return (bid * ask_size + ask * bid_size) / total


@dataclass
class Signal:
    name: str
    p: float                  # P(the backed outcome happens)
    sd: float                 # std. error in probability points


# 99th-percentile chi-square critical values by degrees of freedom.
_CHI2_99 = {1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28, 5: 15.09, 6: 16.81}


@dataclass
class Posterior:
    p: float
    sd: float
    used: list[Signal] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    # True when the remaining sources disagree more than chance allows (p < 0.01):
    # usually a mismatched team or stale quote, so the estimate shouldn't be traded.
    conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "p": round(self.p, 4),
            "sd": round(self.sd, 4),
            "sources": {s.name: {"p": round(s.p, 4), "sd": round(s.sd, 4)} for s in self.used},
            "dropped": self.dropped,
            "conflict": self.conflict,
        }


def combine(signals: list[Signal]) -> Posterior | None:
    """Inverse-variance combination in log-odds space with outlier rejection."""
    live = [s for s in signals if 0.0 < s.p < 1.0 and s.sd > 0]
    if not live:
        return None
    dropped: list[str] = []

    def fuse(items: list[Signal]) -> tuple[float, float]:
        # Convert each sd to log-odds space via the delta method: sd_logit = sd / (p(1-p)).
        w = [1.0 / ((s.sd / (s.p * (1 - s.p))) ** 2) for s in items]
        mean = sum(wi * _logit(s.p) for wi, s in zip(w, items)) / sum(w)
        return mean, math.sqrt(1.0 / sum(w))

    # Drop a source that contradicts the consensus of the others by far more
    # than the uncertainty allows (typically a mismatched team or a bad quote).
    while len(live) >= 3:
        worst, worst_z = None, 0.0
        for s in live:
            others = [o for o in live if o is not s]
            m, sd_o = fuse(others)
            sd_s = s.sd / (s.p * (1 - s.p))
            z = abs(_logit(s.p) - m) / math.sqrt(sd_o ** 2 + sd_s ** 2)
            if z > worst_z:
                worst, worst_z = s, z
        if worst is None or worst_z < _OUTLIER_Z:
            break
        live.remove(worst)
        dropped.append(worst.name)

    mean, sd_logit = fuse(live)
    # Heterogeneity: if sources disagree more than their stated errors allow, the
    # fused number is less certain than inverse-variance math claims (Birge ratio).
    conflict = False
    if len(live) >= 2:
        df = len(live) - 1
        chi2 = sum(
            ((_logit(s.p) - mean) / (s.sd / (s.p * (1 - s.p)))) ** 2 for s in live
        )
        sd_logit *= max(1.0, math.sqrt(chi2 / df))
        critical = _CHI2_99.get(df, df + 2.33 * math.sqrt(2 * df) + 2.0)
        conflict = chi2 > critical
    p = _sigmoid(mean)
    return Posterior(p=p, sd=sd_logit * p * (1 - p), used=live, dropped=dropped, conflict=conflict)


@dataclass
class BetDecision:
    side: str
    price: float
    p: float
    sd: float
    fee: float
    net_edge: float           # expected profit per $1 contract after the fee
    stake_fraction: float     # fraction of equity to stake (before caps)
    confidence: float         # P(true edge > 0) under the posterior


def evaluate_side(post: Posterior, side: str, p_side: float, ask: float, *, fee_rate: float,
                  kelly_fraction: float, fee_multiplier: float = 1.0) -> BetDecision | None:
    """Value of buying ``side`` at ``ask`` given the posterior for that side."""
    if ask <= 0 or ask >= 1:
        return None
    fee = kalshi_fee(ask, fee_rate, fee_multiplier)
    net_edge = p_side - ask - fee
    if net_edge <= 1e-9:
        return None
    sd = max(post.sd, _MIN_SD)
    # Kelly for a binary contract paying $1: f* = edge / (1 - price + fee).
    full_kelly = net_edge / (1.0 - ask + fee)
    # Shrink by signal-to-noise: edges that are small next to their uncertainty
    # are mostly noise, so stake falls off smoothly (-> 0) instead of at a cliff.
    reliability = net_edge ** 2 / (net_edge ** 2 + sd ** 2)
    return BetDecision(
        side=side,
        price=ask,
        p=p_side,
        sd=sd,
        fee=fee,
        net_edge=net_edge,
        stake_fraction=full_kelly * kelly_fraction * reliability,
        confidence=_norm_cdf(net_edge / sd),
    )


def parlay_value(legs: list[tuple[float, float]], *, fee_rate: float, fee_multiplier: float = 1.0) -> dict[str, float]:
    """Fair value of a parlay from independent legs given as (win prob, sd).

    ``fair`` is the product of leg probabilities; ``max_price`` is the most worth
    paying for the combined contract once Kalshi's fee is covered. Legs are
    assumed independent, which holds for different games but not same-game combos.
    """
    fair = 1.0
    rel_var = 0.0
    for p, sd in legs:
        fair *= p
        rel_var += (sd / p) ** 2
    sd_fair = fair * math.sqrt(rel_var)
    max_price = fair
    for _ in range(3):  # fee depends on the price: iterate to a fixed point
        max_price = max(fair - kalshi_fee(max_price, fee_rate, fee_multiplier), 0.0)
    return {"fair": fair, "sd": sd_fair, "max_price": max_price, "fair_decimal_odds": 1.0 / fair}


# ---------------------------------------------------------------------------
# Learning from results
# ---------------------------------------------------------------------------

class SourceCalibration:
    """Tracks how accurate each source has been and derives its sd.

    Every evaluated game logs each source's forecast (``record_forecast``). When
    Kalshi settles the market, ``record_outcome`` scores them. A source's error
    sd is its excess Brier score over the best source seen, blended with the
    prior by pseudo-count so a handful of games can't swing the model.
    """

    def __init__(self, path: Path | None = None, logger: logging.Logger | None = None) -> None:
        self.path = path
        self.logger = logger or logging.getLogger("sports_calibration")
        self.stats: dict[str, dict[str, float]] = {}      # name -> {"n", "brier"}
        self.pending: dict[str, dict[str, float]] = {}    # ticker -> {source: p_yes}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.stats = data.get("stats", {})
            self.pending = data.get("pending", {})
        except (OSError, ValueError) as exc:
            self.logger.warning("Could not load sports calibration (%s); starting fresh.", exc)

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"stats": self.stats, "pending": self.pending}))
            tmp.replace(self.path)
        except OSError as exc:
            self.logger.warning("Could not save sports calibration: %s", exc)

    # -- learning ----------------------------------------------------------
    def record_forecast(self, ticker: str, yes_probs: dict[str, float]) -> None:
        """Remember each source's P(YES) for a market (latest forecast wins)."""
        self.pending[ticker] = {k: float(v) for k, v in yes_probs.items()}
        if len(self.pending) > 2000:  # bound growth: drop oldest-inserted
            for key in list(self.pending)[: len(self.pending) - 2000]:
                del self.pending[key]

    def record_outcome(self, ticker: str, yes_value: float) -> bool:
        """Score a settled market. ``yes_value`` is YES's payout (1, 0, or 0.5 for a tie)."""
        forecasts = self.pending.pop(ticker, None)
        if not forecasts:
            return False
        for name, p in forecasts.items():
            st = self.stats.setdefault(name, {"n": 0.0, "brier": 0.0})
            n = st["n"]
            st["brier"] = (st["brier"] * n + (p - yes_value) ** 2) / (n + 1)
            st["n"] = n + 1
        return True

    def sd(self, name: str) -> float:
        prior = PRIOR_SD.get(name, 0.08)
        st = self.stats.get(name)
        if not st or st["n"] < 1:
            return prior
        best = min(v["brier"] for v in self.stats.values() if v["n"] >= 1)
        excess = max(st["brier"] - best, 0.0)
        learned = math.sqrt(excess) if excess > 0 else _MIN_SD
        n = st["n"]
        blended_var = (_PRIOR_STRENGTH * prior ** 2 + n * max(learned, _MIN_SD) ** 2) / (_PRIOR_STRENGTH + n)
        return max(math.sqrt(blended_var), _MIN_SD)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {"games": st["n"], "brier": round(st["brier"], 4), "sd": round(self.sd(name), 4)}
            for name, st in self.stats.items()
        }


# ---------------------------------------------------------------------------
# Bankroll tracking
# ---------------------------------------------------------------------------

def drawdown_scale(drawdown: float, throttle: float, floor: float = 0.25) -> float:
    """Stake multiplier that eases off as today's bankroll falls from its peak.

    Linear from 1.0 (no drawdown) down to ``floor`` and never below it, so betting
    gets smaller after a bad stretch but never switches off. ``throttle`` is the
    drawdown at which the scale reaches zero before the floor applies; 0 disables.
    """
    if throttle <= 0:
        return 1.0
    return max(floor, 1.0 - max(drawdown, 0.0) / throttle)


class DailySession:
    """Persists the day's starting and peak bankroll so sizing survives restarts."""

    def __init__(self, path: Path | None = None, logger: logging.Logger | None = None) -> None:
        self.path = path
        self.logger = logger or logging.getLogger("sports_session")
        self.date = ""
        self.start = 0.0
        self.peak = 0.0
        self.actions = 0          # "action pick" bets placed today
        self.dogs = 0             # ...of which backed an underdog (win prob < 50%)
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text())
                self.date = str(data.get("date", ""))
                self.start = float(data.get("start", 0.0))
                self.peak = float(data.get("peak", 0.0))
                self.actions = int(data.get("actions", 0))
                self.dogs = int(data.get("dogs", 0))
            except (OSError, ValueError) as exc:
                self.logger.warning("Could not load sports session (%s); starting fresh.", exc)

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "date": self.date, "start": self.start, "peak": self.peak,
                "actions": self.actions, "dogs": self.dogs,
            }))
        except OSError as exc:
            self.logger.warning("Could not save sports session: %s", exc)

    def record_action(self, underdog: bool = False) -> None:
        """Count one action pick (and whether it backed an underdog) against today's allowance."""
        self.actions += 1
        if underdog:
            self.dogs += 1
        self._save()

    def update(self, today: str, equity: float) -> dict[str, float]:
        """Record the current bankroll; returns day PnL and drawdown from the day's peak."""
        if self.date != today:
            self.date, self.start, self.peak, self.actions, self.dogs = today, equity, equity, 0, 0
        elif equity > self.peak:
            self.peak = equity
        self._save()
        drawdown = (self.peak - equity) / self.peak if self.peak > 0 and equity < self.peak else 0.0
        return {"day_start": self.start, "day_peak": self.peak, "day_pnl": equity - self.start,
                "drawdown": drawdown}
