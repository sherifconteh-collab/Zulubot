"""Quick test: fetch live prediction markets from Kalshi and evaluate signals."""

import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"

SERIES = [
    "KXBTC", "KXETH", "KXCPIYOY", "KXCPI", "KXFED", "KXGDP",
    "KXNFP", "KXNEWPOPE", "KXINX", "KXRECESSION", "KXSPY",
    "KXFOMC", "KXTARIFF",
]

CRYPTO_KW = {"btc", "eth", "bitcoin", "ethereum", "crypto", "sol"}
ECON_KW = {"cpi", "gdp", "fed", "fomc", "nfp", "inflation", "rate", "recession", "tariff"}


def categorize(ticker, title):
    text = (ticker + " " + title).lower()
    if any(k in text for k in CRYPTO_KW):
        return "crypto"
    if any(k in text for k in ECON_KW):
        return "economics"
    return "other"


def main():
    print("\n" + "=" * 90)
    print("  KALSHI PREDICTION MARKET - LIVE SIGNAL ANALYSIS")
    print("=" * 90)

    # Fetch active markets from known series
    all_markets = []
    for s in SERIES:
        try:
            r = requests.get(f"{BASE}/markets", params={
                "series_ticker": s, "status": "open", "limit": 20,
            }, timeout=15)
            for m in r.json().get("markets", []):
                yes_bid = float(m.get("yes_bid_dollars", "0") or "0")
                yes_ask = float(m.get("yes_ask_dollars", "0") or "0")
                vol = float(m.get("volume_fp", "0") or "0")
                last = float(m.get("last_price_dollars", "0") or "0")
                liq = float(m.get("liquidity_dollars", "0") or "0")
                # Only include markets with some activity
                if yes_ask > 0.01 or yes_bid > 0 or vol > 0:
                    m["_yes_bid"] = yes_bid
                    m["_yes_ask"] = yes_ask
                    m["_mid"] = (yes_bid + yes_ask) / 2 if (yes_bid > 0 and yes_ask > 0) else (yes_ask or yes_bid or last)
                    m["_vol"] = vol
                    m["_liq"] = liq
                    m["_last"] = last
                    m["_cat"] = categorize(m.get("ticker", ""), m.get("title", ""))
                    all_markets.append(m)
            time.sleep(0.35)
        except Exception as e:
            pass

    all_markets.sort(key=lambda m: m["_vol"], reverse=True)

    print(f"\n  Found {len(all_markets)} active markets across {len(SERIES)} series\n")

    # Show top markets by volume
    print(f"{'-' * 90}")
    print(f"  TOP 15 ACTIVE MARKETS")
    print(f"{'-' * 90}")
    print(f"  {'Ticker':<32} {'Cat':<8} {'Bid':>6} {'Ask':>6} {'Last':>6} {'Vol':>8}  Subtitle")
    print(f"  {'-' * 86}")
    for m in all_markets[:15]:
        t = m.get("ticker", "?")[:31]
        sub = m.get("subtitle", "")[:25]
        print(f"  {t:<32} {m['_cat']:<8} ${m['_yes_bid']:>4.2f} ${m['_yes_ask']:>4.2f} ${m['_last']:>4.2f} {m['_vol']:>8.0f}  {sub}")

    # Category breakdown
    cats = {}
    for m in all_markets:
        c = m["_cat"]
        cats.setdefault(c, []).append(m)

    for cat_name, markets in sorted(cats.items()):
        print(f"\n{'-' * 90}")
        print(f"  {cat_name.upper()} MARKETS ({len(markets)})")
        print(f"{'-' * 90}")
        print(f"  {'Ticker':<32} {'Bid':>6} {'Ask':>6} {'Last':>6} {'Vol':>8}  Subtitle")
        for m in markets[:8]:
            t = m.get("ticker", "?")[:31]
            sub = m.get("subtitle", "")[:28]
            print(f"  {t:<32} ${m['_yes_bid']:>4.2f} ${m['_yes_ask']:>4.2f} ${m['_last']:>4.2f} {m['_vol']:>8.0f}  {sub}")

    # --- SIGNAL EVALUATION ---
    buy_yes_thresh = 0.30  # From .env
    buy_no_thresh = 0.70
    min_volume = 100  # Require some liquidity

    print(f"\n{'=' * 90}")
    print(f"  STRATEGY SIGNAL EVALUATION")
    print(f"  Buy YES when price < ${buy_yes_thresh:.2f}  |  Buy NO when price > ${buy_no_thresh:.2f}")
    print(f"  Min volume: {min_volume}")
    print(f"{'=' * 90}")

    buy_yes_signals = []
    buy_no_signals = []

    for m in all_markets:
        if m["_vol"] < min_volume:
            continue
        mid = m["_mid"]
        if mid <= 0:
            continue

        ticker = m.get("ticker", "")
        subtitle = m.get("subtitle", "")
        title = m.get("title", "")

        if mid < buy_yes_thresh:
            buy_yes_signals.append({
                "ticker": ticker,
                "subtitle": subtitle,
                "title": title,
                "mid": mid,
                "bid": m["_yes_bid"],
                "ask": m["_yes_ask"],
                "vol": m["_vol"],
                "cat": m["_cat"],
                "cost": m["_yes_ask"],  # You'd buy at the ask
            })
        elif mid > buy_no_thresh:
            no_ask = float(m.get("no_ask_dollars", "0") or "0")
            buy_no_signals.append({
                "ticker": ticker,
                "subtitle": subtitle,
                "title": title,
                "mid": mid,
                "bid": m["_yes_bid"],
                "ask": m["_yes_ask"],
                "vol": m["_vol"],
                "cat": m["_cat"],
                "cost": no_ask,
            })

    buy_yes_signals.sort(key=lambda s: s["mid"])
    buy_no_signals.sort(key=lambda s: s["mid"], reverse=True)

    if buy_yes_signals:
        print(f"\n  >> BUY YES signals ({len(buy_yes_signals)} markets priced < ${buy_yes_thresh:.2f}):")
        print(f"  {'Ticker':<30} {'Cat':<8} {'Mid':>6} {'Ask':>6} {'Vol':>8}  What you're betting")
        print(f"  {'-' * 86}")
        for s in buy_yes_signals[:15]:
            t = s["ticker"][:29]
            sub = s["subtitle"][:30]
            print(f"  {t:<30} {s['cat']:<8} ${s['mid']:>4.2f} ${s['ask']:>4.2f} {s['vol']:>8.0f}  YES: {sub}")
    else:
        print(f"\n  >> No BUY YES signals (no markets priced below ${buy_yes_thresh:.2f})")

    if buy_no_signals:
        print(f"\n  >> BUY NO signals ({len(buy_no_signals)} markets priced > ${buy_no_thresh:.2f}):")
        print(f"  {'Ticker':<30} {'Cat':<8} {'Mid':>6} {'Bid':>6} {'Vol':>8}  What you're betting")
        print(f"  {'-' * 86}")
        for s in buy_no_signals[:15]:
            t = s["ticker"][:29]
            sub = s["subtitle"][:30]
            print(f"  {t:<30} {s['cat']:<8} ${s['mid']:>4.2f} ${s['bid']:>4.2f} {s['vol']:>8.0f}  NO: {sub}")
    else:
        print(f"\n  >> No BUY NO signals (no markets priced above ${buy_no_thresh:.2f})")

    total_signals = len(buy_yes_signals) + len(buy_no_signals)

    # Example trade
    if buy_yes_signals or buy_no_signals:
        best = buy_yes_signals[0] if buy_yes_signals else buy_no_signals[0]
        side = "YES" if buy_yes_signals and best == buy_yes_signals[0] else "NO"
        print(f"\n  {'=' * 86}")
        print(f"  BEST TRADE CANDIDATE:")
        print(f"    Market:  {best['ticker']}")
        print(f"    Title:   {best['title']}")
        print(f"    Bet:     {side} - {best['subtitle']}")
        print(f"    Cost:    ${best['cost']:.2f} per contract")
        print(f"    Payout:  $1.00 if correct (profit ${1.0 - best['cost']:.2f})")
        print(f"    Volume:  {best['vol']:.0f} contracts traded")
        print(f"  {'=' * 86}")

    # Summary
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Active markets:     {len(all_markets)}")
    print(f"  Crypto:             {len(cats.get('crypto', []))}")
    print(f"  Economics:          {len(cats.get('economics', []))}")
    print(f"  Other:              {len(cats.get('other', []))}")
    print(f"  BUY YES signals:    {len(buy_yes_signals)}")
    print(f"  BUY NO signals:     {len(buy_no_signals)}")
    print(f"  Total actionable:   {total_signals}")
    print()
    print(f"  To place real trades, add Kalshi credentials to .env:")
    print(f"    KALSHI_KEY_ID=your-key-id")
    print(f"    KALSHI_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----...")
    print(f"    Sign up: https://kalshi.com > Settings > API Keys")
    print()


if __name__ == "__main__":
    main()
