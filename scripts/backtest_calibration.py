"""Measure what buying Kalshi contracts at the ask actually returned, by price, using settled markets.

Read-only: public market data only, no credentials, no orders. For each settled market it takes the
quotes from the last daily candle that ended at least ``--lead-days`` before the market closed, then
asks: had you bought YES at that ask, or NO at that ask, paying Kalshi's fee, what would you have
made? Results are grouped by price bucket and by category, with a 95% interval that resamples whole
events (the strikes of one event move together, so counting them as independent would overstate how
sure we are).

    python scripts/backtest_calibration.py --per-series 120 --lead-days 1 --out calibration.json

It is slow on purpose (Kalshi rate-limits) and caches every candle request next to ``--out`` so a
re-run only fetches what is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trading_bot.sports_model import kalshi_fee  # noqa: E402

BASE = "https://api.elections.kalshi.com/trade-api/v2"

CATEGORIES: dict[str, list[str]] = {
    "economics": ["KXCPI", "KXCPIYOY", "KXCPICORE", "KXFED", "KXPAYROLLS", "KXGDP", "KXRECSSNBER"],
    "finance": ["KXINX", "KXINXU", "KXNASDAQ100", "KXNASDAQ100U"],
    "crypto": ["KXBTC", "KXETH", "KXBTCD", "KXETHD"],
    "weather": ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHLAX", "KXHIGHDEN", "KXHIGHPHIL"],
}
BUCKETS = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 0.90), (0.90, 1.0)]

_session = requests.Session()


def _get(path: str, params: dict, pause: float) -> dict | None:
    for attempt in range(6):
        time.sleep(pause)
        try:
            response = _session.get(f"{BASE}{path}", params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if response.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        if response.status_code != 200:
            return None
        return response.json()
    return None


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def settled_markets(series: str, limit: int, pause: float) -> list[dict]:
    out: list[dict] = []
    cursor = None
    while len(out) < limit * 3:                      # oversample, then sample at random below
        params = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        page = _get("/markets", params, pause)
        if not page:
            break
        out.extend(m for m in page.get("markets", []) if m.get("result") in ("yes", "no"))
        cursor = page.get("cursor")
        if not cursor:
            break
    random.shuffle(out)
    return out[:limit]


def entry_quotes(series: str, market: dict, lead_days: float, pause: float, cache: dict) -> tuple[float, float] | None:
    """(yes_bid, yes_ask) from the last daily candle ending at least ``lead_days`` before the close."""
    ticker = market["ticker"]
    key = f"{ticker}|{lead_days}"
    if key in cache:
        value = cache[key]
        return tuple(value) if value else None
    try:
        close = datetime.fromisoformat(str(market["close_time"]).replace("Z", "+00:00")).timestamp()
    except (KeyError, ValueError):
        return None
    cutoff = close - lead_days * 86400
    data = _get(f"/series/{series}/markets/{ticker}/candlesticks",
                {"start_ts": int(cutoff - 6 * 86400), "end_ts": int(cutoff + 86400), "period_interval": 1440}, pause)
    quotes = None
    if data:
        usable = [c for c in data.get("candlesticks", []) if c.get("end_period_ts", 0) <= cutoff]
        if usable:
            last = usable[-1]
            bid = _num((last.get("yes_bid") or {}).get("close_dollars"))
            ask = _num((last.get("yes_ask") or {}).get("close_dollars"))
            if 0 < bid < ask < 1:
                quotes = (bid, ask)
    cache[key] = list(quotes) if quotes else None
    return quotes


def observations(market: dict, quotes: tuple[float, float]) -> list[tuple[float, float]]:
    """(cost incl. fee, payoff) for buying each side at its ask."""
    yes_bid, yes_ask = quotes
    won_yes = market["result"] == "yes"
    out = []
    for price, won in ((yes_ask, won_yes), (1.0 - yes_bid, not won_yes)):
        if 0.01 <= price <= 0.99:
            out.append((price + kalshi_fee(price), 1.0 if won else 0.0))
    return out


def roi(rows: list[tuple[float, float]]) -> float:
    spent = sum(c for c, _ in rows)
    return (sum(p for _, p in rows) - spent) / spent if spent else 0.0


def cluster_interval(by_event: dict[str, list[tuple[float, float]]], draws: int = 400) -> tuple[float, float]:
    events = list(by_event.values())
    if len(events) < 5:
        return (float("nan"), float("nan"))
    rng = random.Random(7)
    results = []
    for _ in range(draws):
        sample = [row for _ in events for row in rng.choice(events)]
        results.append(roi(sample))
    results.sort()
    return results[int(0.025 * draws)], results[int(0.975 * draws)]


def report(title: str, rows_by_bucket: dict[tuple[float, float], dict[str, list[tuple[float, float]]]]) -> list[dict]:
    print(f"\n{title}")
    print(f"{'ask price':>12} {'contracts':>9} {'events':>7} {'win rate':>9} {'avg cost':>9} {'return':>8}   95% range")
    summary = []
    for bucket in BUCKETS:
        by_event = rows_by_bucket.get(bucket, {})
        rows = [r for event_rows in by_event.values() for r in event_rows]
        if not rows:
            continue
        low, high = cluster_interval(by_event)
        wins = sum(1 for _, p in rows if p > 0) / len(rows)
        avg_cost = sum(c for c, _ in rows) / len(rows)
        print(f"{bucket[0]:>5.2f}-{bucket[1]:<5.2f} {len(rows):>9} {len(by_event):>7} {wins:>8.1%} {avg_cost:>9.3f} "
              f"{roi(rows):>+8.1%}   {low:+.1%} to {high:+.1%}")
        summary.append({"low": bucket[0], "high": bucket[1], "contracts": len(rows), "events": len(by_event),
                        "win_rate": round(wins, 4), "avg_cost": round(avg_cost, 4), "roi": round(roi(rows), 4),
                        "ci_low": None if low != low else round(low, 4), "ci_high": None if high != high else round(high, 4)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-series", type=int, default=100, help="settled markets to sample per series")
    parser.add_argument("--lead-days", type=float, default=1.0, help="how long before the close to buy")
    parser.add_argument("--pause", type=float, default=0.3, help="seconds between requests")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parser.add_argument("--out", default="calibration.json")
    args = parser.parse_args()
    random.seed(11)

    out_path = Path(args.out)
    cache_path = out_path.with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    per_category: dict[str, dict[tuple[float, float], dict[str, list]]] = {}
    for category in args.categories:
        bucketed: dict[tuple[float, float], dict[str, list]] = defaultdict(lambda: defaultdict(list))
        used = 0
        for series in CATEGORIES[category]:
            markets = settled_markets(series, args.per_series, args.pause)
            got = 0
            for market in markets:
                quotes = entry_quotes(series, market, args.lead_days, args.pause, cache)
                if not quotes:
                    continue
                got += 1
                for cost, payoff in observations(market, quotes):
                    price = cost  # bucket on the all-in price
                    for bucket in BUCKETS:
                        if bucket[0] <= price < bucket[1] or (bucket[1] == 1.0 and price >= bucket[0]):
                            bucketed[bucket][market["event_ticker"]].append((cost, payoff))
                            break
            used += got
            print(f"  {category}/{series}: {got} of {len(markets)} settled markets had usable quotes", flush=True)
            cache_path.write_text(json.dumps(cache))
        per_category[category] = bucketed

    combined: dict[tuple[float, float], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for bucketed in per_category.values():
        for bucket, events in bucketed.items():
            for event, rows in events.items():
                combined[bucket][event].extend(rows)

    result = {"lead_days": args.lead_days, "per_series": args.per_series, "all": report(
        f"ALL non-sports categories, buying at the ask {args.lead_days:g} day(s) before the close (fee included)", combined)}
    for category, bucketed in per_category.items():
        result[category] = report(f"{category}", bucketed)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
