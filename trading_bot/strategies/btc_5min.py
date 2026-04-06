"""BTC 5-minute spread monitor strategy.

Watches BTC price across three venues every 5 minutes:
  1. Coinbase spot  — real-time BTC-USD price
  2. Kalshi KXBTC  — implied median BTC price from binary prediction markets
  3. Polymarket    — implied median BTC price from bitcoin-tagged prediction markets

When a venue's implied price diverges from spot by more than ``buy_spread_threshold``
the strategy trades the mispriced venue (buys NO when implied > spot, YES when implied
< spot) and optionally takes a spot BTC position in the same direction.
"""

from __future__ import annotations

import email.utils
import logging
import re
from datetime import datetime, timezone
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients import CoinbaseClient
from trading_bot.clients_kalshi import KalshiClient
from trading_bot.clients_polymarket import PolymarketClobClient, PolymarketGammaClient
from trading_bot.coinbase_broker import CoinbaseBroker
from trading_bot.config import BTC5MinConfig
from trading_bot.models import PortfolioState
from trading_bot.polymarket_broker import PolymarketBroker
from trading_bot.risk import RiskManager

# Matches dollar amounts like "$83,000", "$83k", "$83K", "$83.5k"
_STRIKE_RE = re.compile(r"\$([0-9]+(?:[,_][0-9]{3})*(?:\.[0-9]+)?)\s*([kKmM]?)")


def parse_strike(question: str) -> float | None:
    """Extract a USD strike price from a prediction market question string."""
    match = _STRIKE_RE.search(question)
    if not match:
        return None
    num_str = match.group(1).replace(",", "").replace("_", "")
    suffix = match.group(2).lower()
    try:
        value = float(num_str)
    except ValueError:
        return None
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return value


def compute_implied_median(
    strike_prob_pairs: list[tuple[float, float]],
) -> float | None:
    """Interpolate the implied median (P=0.50) BTC price from strike/probability pairs.

    ``strike_prob_pairs`` is a list of ``(strike_price, P(BTC > strike))`` tuples
    sorted ascending by strike.  Returns the strike where the YES price crosses 0.50
    via linear interpolation, or the nearest boundary if 0.50 is not bracketed.
    """
    if not strike_prob_pairs:
        return None

    pairs = sorted(strike_prob_pairs, key=lambda x: x[0])

    # If all probabilities are above 0.50, the implied price is above the highest strike
    if pairs[-1][1] >= 0.50:
        return pairs[-1][0]

    # If all probabilities are below 0.50, the implied price is below the lowest strike
    if pairs[0][1] <= 0.50:
        return pairs[0][0]

    # Linear interpolation between the two strikes that bracket 0.50
    for i in range(len(pairs) - 1):
        s_lo, p_lo = pairs[i]
        s_hi, p_hi = pairs[i + 1]
        if p_lo >= 0.50 >= p_hi:
            if p_lo == p_hi:
                return (s_lo + s_hi) / 2
            # Interpolate: where does probability cross 0.50?
            t = (p_lo - 0.50) / (p_lo - p_hi)
            return s_lo + t * (s_hi - s_lo)

    return pairs[len(pairs) // 2][0]  # fallback: median strike


class BTC5MinSpreadStrategy:
    """5-minute BTC spread monitor and arbitrage executor.

    Detects mispricing between Coinbase spot BTC and prediction-market implied
    prices on Kalshi and Polymarket, then trades the mispriced venue.
    """

    STRATEGY_NAME = "btc_5min"
    _KALSHI_CASH_BUCKET = "kalshi"

    def __init__(
        self,
        *,
        config: BTC5MinConfig,
        coinbase_client: CoinbaseClient,
        coinbase_broker: CoinbaseBroker,
        kalshi_client: KalshiClient,
        gamma_client: PolymarketGammaClient,
        clob_client: PolymarketClobClient,
        polymarket_broker: PolymarketBroker,
        logger: logging.Logger,
        live_kalshi_trading: bool = False,
    ) -> None:
        self.config = config
        self.coinbase_client = coinbase_client
        self.coinbase_broker = coinbase_broker
        self.kalshi_client = kalshi_client
        self.gamma_client = gamma_client
        self.clob_client = clob_client
        self.polymarket_broker = polymarket_broker
        self.logger = logger.getChild("btc5min")
        self.live_kalshi_trading = live_kalshi_trading

    # ── Market data helpers ───────────────────────────────────────────

    def _get_spot_price(self) -> float | None:
        """Fetch current BTC-USD spot price from Coinbase."""
        try:
            prices = self.coinbase_client.get_spot_prices([self.config.btc_symbol])
            return prices.get(self.config.btc_symbol)
        except Exception as exc:
            self.logger.warning("Coinbase spot price fetch failed: %s", exc)
            return None

    def _get_kalshi_implied(self) -> tuple[float | None, list[dict[str, Any]]]:
        """Fetch KXBTC markets and compute the implied median BTC price.

        Returns (implied_price_or_None, list_of_parsed_market_dicts).
        """
        try:
            markets, _ = self.kalshi_client.get_markets(
                series_ticker="KXBTC", status="open", limit=100
            )
        except Exception as exc:
            self.logger.warning("Kalshi KXBTC fetch failed: %s", exc)
            return None, []

        now_ts = datetime.now(timezone.utc).timestamp()
        max_age_secs = self.config.max_strike_age_hours * 3600
        strike_probs: list[tuple[float, float]] = []
        parsed: list[dict[str, Any]] = []

        for m in markets:
            title = m.get("title", "") or m.get("subtitle", "")
            strike = parse_strike(title)
            if strike is None:
                continue

            # Skip markets with insufficient volume
            volume = float(m.get("volume_fp") or m.get("volume") or 0)
            if volume < self.config.min_kalshi_volume_usd:
                continue

            # Skip markets expiring beyond our time window
            end_date = m.get("end_date") or m.get("expiration_time", "")
            if end_date:
                try:
                    # Kalshi may use ISO strings or epoch seconds
                    if isinstance(end_date, (int, float)):
                        expires_ts = float(end_date)
                    else:
                        # Try ISO first, then RFC 2822
                        try:
                            expires_ts = datetime.fromisoformat(
                                end_date.replace("Z", "+00:00")
                            ).timestamp()
                        except ValueError:
                            expires_ts = float(email.utils.parsedate_to_datetime(end_date).timestamp())
                    if expires_ts - now_ts > max_age_secs:
                        continue
                except Exception:
                    pass  # If we can't parse, include it

            yes_bid = float(m.get("yes_bid_dollars") or 0)
            yes_ask = float(m.get("yes_ask_dollars") or 0)
            last_price = float(m.get("last_price_dollars") or 0)
            if yes_bid or yes_ask:
                yes_mid = (yes_bid + yes_ask) / 2
            elif last_price:
                yes_mid = last_price
            else:
                # No active quotes and no trade history — skip to avoid noise
                continue

            strike_probs.append((strike, yes_mid))
            parsed.append({
                "ticker": m.get("ticker", ""),
                "title": title[:80],
                "strike": strike,
                "yes_mid": round(yes_mid, 4),
                "volume": volume,
            })

        implied = compute_implied_median(strike_probs)
        self.logger.info(
            "Kalshi KXBTC: %d markets parsed, implied=%.0f",
            len(strike_probs), implied or 0,
        )
        return implied, parsed

    def _get_polymarket_implied(self) -> tuple[float | None, list[dict[str, Any]]]:
        """Fetch Polymarket bitcoin markets and compute the implied median BTC price.

        Returns (implied_price_or_None, list_of_parsed_market_dicts).
        """
        try:
            markets = self.gamma_client.discover_markets(
                tags=["bitcoin"],
                min_volume=self.config.min_poly_volume_usd,
                limit=50,
            )
        except Exception as exc:
            self.logger.warning("Polymarket bitcoin market fetch failed: %s", exc)
            return None, []

        now_ts = datetime.now(timezone.utc).timestamp()
        max_age_secs = self.config.max_strike_age_hours * 3600
        strike_probs: list[tuple[float, float]] = []
        parsed: list[dict[str, Any]] = []

        for mkt in markets:
            strike = parse_strike(mkt.question)
            if strike is None:
                continue

            # Skip far-future markets
            if mkt.end_date:
                try:
                    expires_ts = datetime.fromisoformat(
                        mkt.end_date.replace("Z", "+00:00")
                    ).timestamp()
                    if expires_ts - now_ts > max_age_secs:
                        continue
                except Exception:
                    pass

            # Refresh YES price from CLOB for accuracy
            yes_mid = mkt.yes_price
            if mkt.yes_token_id:
                try:
                    live = self.clob_client.get_midpoint(mkt.yes_token_id)
                    if live is not None:
                        yes_mid = live
                except Exception:
                    pass

            strike_probs.append((strike, yes_mid))
            parsed.append({
                "condition_id": mkt.condition_id[:16],
                "question": mkt.question[:80],
                "strike": strike,
                "yes_mid": round(yes_mid, 4),
                "volume_24h": mkt.volume_24h,
            })

        implied = compute_implied_median(strike_probs)
        self.logger.info(
            "Polymarket BTC: %d markets parsed, implied=%.0f",
            len(strike_probs), implied or 0,
        )
        return implied, parsed

    # ── Execution helpers ─────────────────────────────────────────────

    def _open_spot_position(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        risk_manager: RiskManager,
        price: float,
        now: datetime,
        meta: dict[str, Any],
    ) -> bool:
        """Open a spot BTC long position if not already held and risk allows."""
        symbol = self.config.btc_symbol
        if symbol in state.positions:
            return False
        if not risk_manager.can_open_new_position(state):
            self.logger.info("Spot BTC buy blocked by risk guardrails.")
            return False
        equity = broker.portfolio_equity(state)
        notional = risk_manager.position_notional(state, equity, symbol=symbol)
        if notional <= 0:
            return False
        self.coinbase_broker.place_market_buy(
            state,
            binance_symbol=symbol,
            notional_usd=notional,
            strategy=self.STRATEGY_NAME,
            now=now,
            meta=meta,
        )
        broker.open_position(
            state,
            symbol=symbol,
            asset_type="crypto",
            strategy=self.STRATEGY_NAME,
            price=price,
            notional=notional,
            opened_at=now,
            meta=meta,
        )
        return True

    def _close_spot_position(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        price: float,
        reason: str,
    ) -> bool:
        """Close an open spot BTC position."""
        symbol = self.config.btc_symbol
        if symbol not in state.positions:
            return False
        pos = state.positions[symbol]
        self.coinbase_broker.place_market_sell(
            state,
            binance_symbol=symbol,
            quantity=pos.quantity,
            strategy=self.STRATEGY_NAME,
            reason=reason,
        )
        broker.close_position(state, symbol=symbol, price=price, reason=reason)
        return True

    def _trade_kalshi(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        risk_manager: RiskManager,
        kalshi_markets: list[dict[str, Any]],
        side: str,  # "yes" or "no"
        spot_price: float,
        spread: float,
        now: datetime,
    ) -> list[str]:
        """Place YES or NO on the KXBTC market whose strike is closest to spot price."""
        if not kalshi_markets:
            return []

        # Pick the market whose strike is closest to spot
        best = min(kalshi_markets, key=lambda m: abs(m["strike"] - spot_price))
        ticker = best["ticker"]
        if not ticker:
            return []

        if not risk_manager.can_open_new_position(state):
            self.logger.info("Kalshi %s blocked by risk guardrails.", side.upper())
            return []

        equity = broker.portfolio_equity_for_bucket(
            state, cash_bucket=self._KALSHI_CASH_BUCKET if self.live_kalshi_trading else None
        )
        notional = risk_manager.position_notional(state, equity, symbol=ticker)
        if notional <= 0:
            return []

        yes_mid = best["yes_mid"]
        # price_cents is the cost of the contract side we're buying
        price_cents = max(1, min(99, round((yes_mid if side == "yes" else 1 - yes_mid) * 100)))
        contracts = max(1, int(notional / (price_cents / 100)))

        # KalshiClient.place_order always expects `yes_price` in cents; for NO orders
        # the YES price is the complement (100 - no_price_cents).
        yes_price_cents = price_cents if side == "yes" else 100 - price_cents

        self.logger.info(
            "Kalshi %s %s: ticker=%s strike=%.0f yes_mid=%.3f price_cents=%d contracts=%d spread=%.2f%%",
            side.upper(), ticker, ticker, best["strike"], yes_mid, price_cents, contracts, spread * 100,
        )

        if self.live_kalshi_trading:
            try:
                self.kalshi_client.place_order(
                    ticker=ticker,
                    side=side,
                    action="buy",
                    count=contracts,
                    yes_price=yes_price_cents,
                )
            except Exception as exc:
                self.logger.error("Kalshi order failed: %s", exc)
                return []

        # Book it in the paper broker under a synthetic symbol.
        # In live mode, use the Kalshi cash bucket so equity/risk accounting
        # tracks against the dedicated Kalshi buying-power balance.
        kal_symbol = f"KAL:{ticker[:16]}:{side.upper()}"
        broker.open_position(
            state,
            symbol=kal_symbol,
            asset_type="prediction",
            strategy=self.STRATEGY_NAME,
            price=price_cents / 100,
            notional=notional,
            opened_at=now,
            cash_bucket=self._KALSHI_CASH_BUCKET if self.live_kalshi_trading else None,
            meta={
                "ticker": ticker,
                "side": side,
                "strike": best["strike"],
                "yes_mid": yes_mid,
                "spread_pct": round(spread * 100, 3),
                "contracts": contracts,
            },
        )
        return [kal_symbol]

    def _trade_polymarket(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        risk_manager: RiskManager,
        poly_markets: list[dict[str, Any]],
        side: str,  # "yes" or "no"
        spot_price: float,
        spread: float,
        now: datetime,
    ) -> list[str]:
        """Place YES or NO on the Polymarket market closest to spot price."""
        if not poly_markets:
            return []

        best = min(poly_markets, key=lambda m: abs(m["strike"] - spot_price))
        condition_id = best.get("condition_id", "")
        if not condition_id:
            return []

        if not risk_manager.can_open_new_position(state):
            self.logger.info("Polymarket %s blocked by risk guardrails.", side.upper())
            return []

        # Fetch the full PolymarketMarket object needed by the broker
        try:
            market = self.gamma_client.get_market(condition_id)
        except Exception as exc:
            self.logger.warning("Failed to fetch Polymarket market %s: %s", condition_id, exc)
            return []

        equity = broker.portfolio_equity(state)
        notional = risk_manager.position_notional(state, equity, symbol=condition_id)
        if notional <= 0:
            return []

        self.logger.info(
            "Polymarket %s: condition=%s strike=%.0f yes_mid=%.3f notional=%.2f spread=%.2f%%",
            side.upper(), condition_id[:16], best["strike"], best["yes_mid"], notional, spread * 100,
        )

        price = best["yes_mid"] if side.upper() == "YES" else 1.0 - best["yes_mid"]
        price = max(0.01, min(0.99, price))

        self.polymarket_broker.place_limit_buy(
            state,
            market=market,
            side=side.upper(),
            price=price,
            notional_usd=notional,
            strategy=self.STRATEGY_NAME,
            now=now,
            meta={"spread_pct": round(spread * 100, 3), "strike": best["strike"]},
        )

        pm_symbol = f"PM:{condition_id[:8]}:{side.upper()}"
        return [pm_symbol]

    def _close_prediction_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        prefix: str,
        reason: str,
    ) -> list[str]:
        """Close open prediction positions whose symbol starts with *prefix*.

        Positions are closed at their own current_price (or entry_price if no
        mark is available), so P&L is computed in probability units (0–1) — not
        in spot-BTC dollars, which would produce wildly incorrect proceeds.
        """
        closed: list[str] = []
        for sym in list(state.positions.keys()):
            if not sym.startswith(prefix):
                continue
            pos = state.positions.get(sym)
            if pos is None:
                continue
            if pos.strategy != self.STRATEGY_NAME:
                continue
            close_price = (
                pos.current_price
                if getattr(pos, "current_price", None) is not None
                else pos.entry_price
            )
            broker.close_position(state, symbol=sym, price=close_price, reason=reason)
            self.logger.info("Closed prediction position %s at %.4f (%s)", sym, close_price, reason)
            closed.append(sym)
        return closed

    # ── Main run ──────────────────────────────────────────────────────

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"status": "disabled"}

        # ── 1. Fetch all three prices ─────────────────────────────────
        spot = self._get_spot_price()
        kalshi_implied, kalshi_markets = self._get_kalshi_implied()
        poly_implied, poly_markets = self._get_polymarket_implied()

        # Mark existing spot position with current price
        if spot is not None:
            broker.mark_positions(state, {self.config.btc_symbol: spot})

        # ── 2. Compute spreads ────────────────────────────────────────
        kalshi_spread: float | None = None
        poly_spread: float | None = None

        if spot and spot > 0:
            if kalshi_implied:
                kalshi_spread = (kalshi_implied - spot) / spot
            if poly_implied:
                poly_spread = (poly_implied - spot) / spot

        threshold = self.config.buy_spread_threshold
        actions: list[dict[str, Any]] = []
        opens: list[str] = []
        closes: list[str] = []
        decisions: list[dict[str, Any]] = []

        # ── 3. Act on Kalshi spread ───────────────────────────────────
        if kalshi_spread is not None:
            if kalshi_spread > threshold:
                # Kalshi is too bullish → sell YES (buy NO) on Kalshi
                self.logger.info("Kalshi spread +%.2f%% → buy NO", kalshi_spread * 100)
                syms = self._trade_kalshi(
                    state, broker, risk_manager, kalshi_markets,
                    side="no", spot_price=spot, spread=kalshi_spread, now=now,
                )
                opens.extend(syms)
                # Also buy spot BTC (price will catch up to implied)
                if self.config.spot_trading_enabled:
                    bought = self._open_spot_position(
                        state, broker, risk_manager, spot, now,
                        meta={"signal": "kalshi_spread", "spread_pct": round(kalshi_spread * 100, 3)},
                    )
                    if bought:
                        opens.append(self.config.btc_symbol)
                actions.append({
                    "venue": "kalshi",
                    "spread_pct": round(kalshi_spread * 100, 3),
                    "action": "buy_no",
                    "symbols": syms,
                })
                for sym in syms:
                    decisions.append({
                        "strategy": self.STRATEGY_NAME,
                        "symbol": sym,
                        "action": "buy_no",
                        "reason": "kalshi_bullish_spread",
                        "signals": {"spread_pct": round(kalshi_spread * 100, 3), "threshold_pct": threshold * 100},
                    })

            elif kalshi_spread < -threshold:
                # Kalshi is too bearish → buy YES on Kalshi
                self.logger.info("Kalshi spread %.2f%% → buy YES", kalshi_spread * 100)
                syms = self._trade_kalshi(
                    state, broker, risk_manager, kalshi_markets,
                    side="yes", spot_price=spot, spread=kalshi_spread, now=now,
                )
                opens.extend(syms)
                # Sell any open spot BTC position
                if self.config.spot_trading_enabled:
                    closed = self._close_spot_position(
                        state, broker, spot, reason="kalshi_bearish_spread"
                    )
                    if closed:
                        closes.append(self.config.btc_symbol)
                actions.append({
                    "venue": "kalshi",
                    "spread_pct": round(kalshi_spread * 100, 3),
                    "action": "buy_yes",
                    "symbols": syms,
                })
                for sym in syms:
                    decisions.append({
                        "strategy": self.STRATEGY_NAME,
                        "symbol": sym,
                        "action": "buy_yes",
                        "reason": "kalshi_bearish_spread",
                        "signals": {"spread_pct": round(kalshi_spread * 100, 3), "threshold_pct": threshold * 100},
                    })

            else:
                # Spread within threshold — flatten any open Kalshi prediction positions
                flat = self._close_prediction_positions(
                    state, broker, prefix="KAL:", reason="kalshi_spread_normalized"
                )
                closes.extend(flat)
                for sym in flat:
                    decisions.append({
                        "strategy": self.STRATEGY_NAME,
                        "symbol": sym,
                        "action": "close",
                        "reason": "kalshi_spread_normalized",
                        "signals": {"spread_pct": round(kalshi_spread * 100, 3), "threshold_pct": threshold * 100},
                    })

        # ── 4. Act on Polymarket spread ───────────────────────────────
        if poly_spread is not None:
            if poly_spread > threshold:
                self.logger.info("Polymarket spread +%.2f%% → buy NO", poly_spread * 100)
                syms = self._trade_polymarket(
                    state, broker, risk_manager, poly_markets,
                    side="no", spot_price=spot, spread=poly_spread, now=now,
                )
                opens.extend(syms)
                if self.config.spot_trading_enabled and self.config.btc_symbol not in state.positions:
                    bought = self._open_spot_position(
                        state, broker, risk_manager, spot, now,
                        meta={"signal": "poly_spread", "spread_pct": round(poly_spread * 100, 3)},
                    )
                    if bought:
                        opens.append(self.config.btc_symbol)
                actions.append({
                    "venue": "polymarket",
                    "spread_pct": round(poly_spread * 100, 3),
                    "action": "buy_no",
                    "symbols": syms,
                })
                for sym in syms:
                    decisions.append({
                        "strategy": self.STRATEGY_NAME,
                        "symbol": sym,
                        "action": "buy_no",
                        "reason": "poly_bullish_spread",
                        "signals": {"spread_pct": round(poly_spread * 100, 3), "threshold_pct": threshold * 100},
                    })

            elif poly_spread < -threshold:
                self.logger.info("Polymarket spread %.2f%% → buy YES", poly_spread * 100)
                syms = self._trade_polymarket(
                    state, broker, risk_manager, poly_markets,
                    side="yes", spot_price=spot, spread=poly_spread, now=now,
                )
                opens.extend(syms)
                if self.config.spot_trading_enabled:
                    closed = self._close_spot_position(
                        state, broker, spot, reason="poly_bearish_spread"
                    )
                    if closed:
                        closes.append(self.config.btc_symbol)
                actions.append({
                    "venue": "polymarket",
                    "spread_pct": round(poly_spread * 100, 3),
                    "action": "buy_yes",
                    "symbols": syms,
                })
                for sym in syms:
                    decisions.append({
                        "strategy": self.STRATEGY_NAME,
                        "symbol": sym,
                        "action": "buy_yes",
                        "reason": "poly_bearish_spread",
                        "signals": {"spread_pct": round(poly_spread * 100, 3), "threshold_pct": threshold * 100},
                    })

            else:
                # Spread within threshold — flatten any open Polymarket prediction positions
                flat = self._close_prediction_positions(
                    state, broker, prefix="PM:", reason="poly_spread_normalized"
                )
                closes.extend(flat)
                for sym in flat:
                    decisions.append({
                        "strategy": self.STRATEGY_NAME,
                        "symbol": sym,
                        "action": "close",
                        "reason": "poly_spread_normalized",
                        "signals": {"spread_pct": round(poly_spread * 100, 3), "threshold_pct": threshold * 100},
                    })

        if not actions:
            self.logger.info(
                "BTC5min HOLD — spot=%.0f kalshi_implied=%s (%.2f%%) poly_implied=%s (%.2f%%)",
                spot or 0,
                f"{kalshi_implied:.0f}" if kalshi_implied else "n/a",
                (kalshi_spread or 0) * 100,
                f"{poly_implied:.0f}" if poly_implied else "n/a",
                (poly_spread or 0) * 100,
            )
            decisions.append({
                "strategy": self.STRATEGY_NAME,
                "symbol": self.config.btc_symbol,
                "action": "hold",
                "reason": "spread_within_threshold",
                "signals": {
                    "kalshi_spread_pct": round(kalshi_spread * 100, 3) if kalshi_spread is not None else None,
                    "poly_spread_pct": round(poly_spread * 100, 3) if poly_spread is not None else None,
                    "threshold_pct": threshold * 100,
                },
            })

        return {
            "status": "completed",
            "spot_price": round(spot, 2) if spot else None,
            "kalshi_implied": round(kalshi_implied, 2) if kalshi_implied else None,
            "poly_implied": round(poly_implied, 2) if poly_implied else None,
            "kalshi_spread_pct": round(kalshi_spread * 100, 3) if kalshi_spread is not None else None,
            "poly_spread_pct": round(poly_spread * 100, 3) if poly_spread is not None else None,
            "kalshi_markets_used": len(kalshi_markets),
            "poly_markets_used": len(poly_markets),
            "actions": actions,
            "opens": opens,
            "closes": closes,
            "threshold_pct": threshold * 100,
            "decisions": decisions,
        }
