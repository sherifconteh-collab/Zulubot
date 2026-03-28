"""Latency arbitrage strategy on Kalshi prediction markets.

Strategy overview
-----------------
1. Monitor Coinbase BTC-USD / ETH-USD spot prices every few seconds.
2. When a sharp price move is detected (e.g. ≥ 0.3% in 30 s) but the
   corresponding Kalshi binary contract still shows stale 50/50 odds,
   buy the ``correct`` side before the market reprices.
3. Size each position using the fractional Kelly criterion.
4. Exit when the contract price moves in our favour by take_profit_pct,
   when stop_loss_pct is hit, or when the contract resolves.

Edge calculation
----------------
  true_prob = sigmoid(|momentum_pct| × SCALE)   # SCALE = 150
  edge      = true_prob − ask_price
  Only trade when edge > config.min_edge_pct.

Kelly position sizing (binary prediction market)
-------------------------------------------------
  f*       = edge / (1 − ask_price)
  notional = equity × f* × config.kelly_fraction
  notional = min(notional, equity × config.max_position_pct)

Position symbols
----------------
  KL:{ticker}:{YES|NO}   e.g. KL:KXBTCH-26MAY0117:YES
  asset_type = "prediction", strategy = "latency_arb"
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients_kalshi import KalshiClient
from trading_bot.coinbase_feed import CoinbasePriceFeed
from trading_bot.config import LatencyArbConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager

try:
    from trading_bot.integrations.trading_agents import (
        TradingAgentsService,
        apply_ai_confirmation,
    )
except ImportError:
    TradingAgentsService = None  # type: ignore[assignment,misc]
    apply_ai_confirmation = None  # type: ignore[assignment]

# Logistic scale: maps momentum % → probability
_SIGMOID_SCALE = 150.0

# Regime-based threshold adjustment multipliers
_VOLATILE_THRESHOLD_MULTIPLIER = 1.5   # raise bar in volatile markets
_BULLISH_THRESHOLD_MULTIPLIER = 0.85   # lower bar in calm bull markets

# Minimum AI confidence required to veto an exit decision
_AI_EXIT_VETO_MIN_CONFIDENCE = 0.65

# Keyword sets for inferring contract direction from title/subtitle
_HIGHER_KEYWORDS = frozenset({"higher", "above", "over", "up", "rise", "rises", "gain"})
_LOWER_KEYWORDS = frozenset({"lower", "below", "under", "down", "fall", "falls", "drop"})

# Kalshi series tickers for crypto assets (mapped from Coinbase symbol)
_ASSET_TO_SERIES: dict[str, list[str]] = {
    "BTC-USD": ["KXBTC"],
    "ETH-USD": ["KXETH"],
}


class LatencyArbStrategy:
    """Latency arbitrage on Kalshi prediction markets using Coinbase price feed."""

    def __init__(
        self,
        *,
        config: LatencyArbConfig,
        price_feed: CoinbasePriceFeed,
        kalshi_client: KalshiClient,
        paper_broker: PaperBroker,
        logger: logging.Logger,
        ai_service: Any | None = None,
    ) -> None:
        self.config = config
        self.feed = price_feed
        self.kalshi = kalshi_client
        self.broker = paper_broker
        self.logger = logger.getChild("strategy.latency_arb")
        self.ai_service = ai_service  # Haiku-only signal validation (log_only, never blocks)

        # Cache discovered Kalshi markets to avoid hammering the API
        # Maps series_ticker → list of markets
        self._market_cache: dict[str, list[dict[str, Any]]] = {}
        # Per-series cache timestamps so a fresh fetch for one series doesn't
        # keep stale data for another series beyond the TTL.
        self._cache_ts: dict[str, float] = {}
        self._cache_ttl: float = 30.0  # refresh every 30 s

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_once(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
    ) -> dict[str, Any]:
        """Execute one latency arb scan cycle.

        Args:
            state: Current portfolio state (mutated in place).
            broker: PaperBroker for equity calculations.
            risk_manager: For kill-switch / halt checks.
            now: Current wall-clock time.

        Returns:
            Result dict with keys: status, buys, sells, signals_detected,
            markets_scanned, momentum.
        """
        result: dict[str, Any] = {
            "status": "ok",
            "buys": [],
            "sells": [],
            "signals_detected": 0,
            "markets_scanned": 0,
            "momentum": {},
            "can_trade_live": self.kalshi.can_trade,
        }

        if not self.config.enabled:
            result["status"] = "disabled"
            return result

        try:
            # Step 1: Update price history
            prices = self.feed.record_current_prices()
            if not prices:
                result["status"] = "no_prices"
                return result

            # Step 2: Manage existing positions (take-profit / stop-loss)
            self._manage_positions(state, broker, result, now)

            # Step 3: Check risk gate
            if state.kill_switch_engaged or state.halt_new_trades_for_day:
                result["status"] = "halted"
                return result

            # Step 4: Count open arb positions
            open_arb = self._count_open_positions(state)
            if open_arb >= self.config.max_open_contracts:
                self.logger.debug(
                    "At max open arb contracts (%d). Skipping entry scan.", open_arb
                )
                return result

            equity = broker.portfolio_equity(state)

            # Step 5: Detect market regime (Haiku-only, 1h cache) to adjust thresholds
            effective_threshold = self.config.momentum_min_threshold_pct
            current_regime: str | None = None
            if self.ai_service is not None:
                try:
                    btc_price = self.feed.get_latest_price("BTC-USD")
                    eth_price = self.feed.get_latest_price("ETH-USD")
                    if btc_price > 0 and eth_price > 0:
                        # Pass a longer-window BTC momentum to improve regime accuracy
                        btc_momentum_60s = self.feed.get_momentum_pct("BTC-USD", 60.0)
                        regime = self.ai_service.detect_market_regime(
                            btc_price=btc_price,
                            eth_price=eth_price,
                            fear_greed_value=None,
                            btc_momentum_24h_pct=btc_momentum_60s if btc_momentum_60s != 0.0 else None,
                        )
                        if regime is not None:
                            current_regime = regime.action
                            result["market_regime"] = regime.action
                            if regime.action == "volatile" and regime.confidence >= 0.6:
                                # Volatile: raise threshold to reduce false signals
                                effective_threshold = self.config.momentum_min_threshold_pct * _VOLATILE_THRESHOLD_MULTIPLIER
                                self.logger.info(
                                    "Volatile regime (conf=%.2f). Raising threshold: "
                                    "%.2f%% → %.2f%%",
                                    regime.confidence,
                                    self.config.momentum_min_threshold_pct,
                                    effective_threshold,
                                )
                            elif regime.action == "bullish" and regime.confidence >= 0.7:
                                # Calm bull: slightly lower threshold for more opportunities
                                effective_threshold = self.config.momentum_min_threshold_pct * _BULLISH_THRESHOLD_MULTIPLIER
                                self.logger.info(
                                    "Bullish regime (conf=%.2f). Lowering threshold: "
                                    "%.2f%% → %.2f%%",
                                    regime.confidence,
                                    self.config.momentum_min_threshold_pct,
                                    effective_threshold,
                                )
                except Exception as exc:
                    self.logger.debug("Market regime detection failed (non-blocking): %s", exc)

            # Step 6: For each tracked symbol, check momentum → scan markets
            for symbol in self.config.symbols:
                if not self.feed.is_ready(symbol, self.config.momentum_window_seconds):
                    self.logger.debug(
                        "%s feed not ready (need %gs history).", symbol, self.config.momentum_window_seconds
                    )
                    continue

                momentum_pct = self.feed.get_momentum_pct(symbol, self.config.momentum_window_seconds)
                result["momentum"][symbol] = round(momentum_pct, 4)

                if abs(momentum_pct) < effective_threshold:
                    continue  # Signal too weak

                result["signals_detected"] += 1
                self.logger.info(
                    "%s momentum = %.3f%% (threshold %.2f%%)",
                    symbol, momentum_pct, self.config.momentum_min_threshold_pct,
                )

                # Fetch relevant Kalshi markets
                series_list = _ASSET_TO_SERIES.get(symbol, [])
                for series in series_list:
                    markets = self._discover_kalshi_contracts(series)
                    result["markets_scanned"] += len(markets)

                    for market in markets:
                        if open_arb >= self.config.max_open_contracts:
                            break

                        signal = self._get_signal(market, momentum_pct)
                        if signal is None:
                            continue

                        side, ask_price = signal
                        true_prob = self._estimate_true_prob(abs(momentum_pct))
                        edge = true_prob - ask_price

                        notional = self._kelly_notional(edge, ask_price, equity)
                        if notional < self.config.min_trade_usd:
                            self.logger.debug(
                                "Kelly notional too small ($%.2f < $%.2f). Skipping.",
                                notional, self.config.min_trade_usd,
                            )
                            continue

                        entered = self._execute_trade(
                            state=state,
                            market=market,
                            side=side,
                            ask_price=ask_price,
                            notional=notional,
                            momentum_pct=momentum_pct,
                            now=now,
                            market_regime=current_regime,
                        )
                        if entered:
                            result["buys"].append(f"KL:{market.get('ticker', '')}:{side}")
                            open_arb += 1

        except Exception as exc:
            self.logger.error("Latency arb cycle failed: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    def _discover_kalshi_contracts(self, series_ticker: str) -> list[dict[str, Any]]:
        """Fetch open Kalshi markets for a series, with 30s per-series cache."""
        now = time.time()
        series_ts = self._cache_ts.get(series_ticker, 0.0)

        if series_ticker in self._market_cache and (now - series_ts) < self._cache_ttl:
            return self._market_cache[series_ticker]

        try:
            markets, _ = self.kalshi.get_markets(
                status="open",
                series_ticker=series_ticker,
                limit=50,
            )
            self._market_cache[series_ticker] = markets
            self._cache_ts[series_ticker] = now
            self.logger.debug("Fetched %d Kalshi markets for series %s", len(markets), series_ticker)
            return markets
        except Exception as exc:
            self.logger.warning("Failed to fetch Kalshi markets for %s: %s", series_ticker, exc)
            return self._market_cache.get(series_ticker, [])

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _get_signal(
        self,
        market: dict[str, Any],
        momentum_pct: float,
    ) -> tuple[str, float] | None:
        """Return (side, ask_price) if there is a tradeable edge, else None.

        side is "yes" or "no" (Kalshi API format).
        """
        ticker = market.get("ticker", "")
        title = (market.get("title", "") + " " + market.get("subtitle", "")).lower()

        direction = self._infer_direction(title)
        if direction is None:
            self.logger.debug("Cannot infer direction for %s: '%s'", ticker, title[:80])
            return None

        # Retrieve prices from Kalshi orderbook (best ask for each side)
        yes_ask, no_ask = self._get_market_prices(ticker)
        if yes_ask <= 0 or no_ask <= 0:
            return None

        true_prob = self._estimate_true_prob(abs(momentum_pct))

        if direction == "HIGHER":
            if momentum_pct > 0:
                # Asset trending up → YES ("will be higher") likely correct
                edge = true_prob - yes_ask
                if edge >= self.config.min_edge_pct:
                    self.logger.info(
                        "Edge on YES for %s: true_prob=%.2f yes_ask=%.2f edge=%.2f",
                        ticker, true_prob, yes_ask, edge,
                    )
                    return ("yes", yes_ask)
            else:
                # Asset trending down → NO ("will NOT be higher") likely correct
                edge = true_prob - no_ask
                if edge >= self.config.min_edge_pct:
                    self.logger.info(
                        "Edge on NO for %s: true_prob=%.2f no_ask=%.2f edge=%.2f",
                        ticker, true_prob, no_ask, edge,
                    )
                    return ("no", no_ask)

        elif direction == "LOWER":
            if momentum_pct < 0:
                # Asset trending down → YES ("will be lower") likely correct
                edge = true_prob - yes_ask
                if edge >= self.config.min_edge_pct:
                    self.logger.info(
                        "Edge on YES for %s (lower): true_prob=%.2f yes_ask=%.2f edge=%.2f",
                        ticker, true_prob, yes_ask, edge,
                    )
                    return ("yes", yes_ask)
            else:
                # Asset trending up → NO ("will NOT be lower") likely correct
                edge = true_prob - no_ask
                if edge >= self.config.min_edge_pct:
                    self.logger.info(
                        "Edge on NO for %s (lower): true_prob=%.2f no_ask=%.2f edge=%.2f",
                        ticker, true_prob, no_ask, edge,
                    )
                    return ("no", no_ask)

        return None

    def _infer_direction(self, text: str) -> str | None:
        """Infer "HIGHER" or "LOWER" from contract title text."""
        words = set(text.lower().split())
        has_higher = bool(words & _HIGHER_KEYWORDS)
        has_lower = bool(words & _LOWER_KEYWORDS)
        if has_higher and not has_lower:
            return "HIGHER"
        if has_lower and not has_higher:
            return "LOWER"
        return None  # Ambiguous or neither

    def _get_market_prices(self, ticker: str) -> tuple[float, float]:
        """Return (yes_ask, no_ask) in [0, 1] range from Kalshi orderbook.

        Falls back to mid-price from market data if orderbook unavailable.
        """
        try:
            book = self.kalshi.get_orderbook(ticker)
            orderbook = book.get("orderbook", book)

            yes_ask = _best_ask(orderbook.get("yes", []))
            no_ask = _best_ask(orderbook.get("no", []))

            # Kalshi prices are in cents (1–99), convert to [0,1]
            if yes_ask > 1:
                yes_ask = yes_ask / 100.0
            if no_ask > 1:
                no_ask = no_ask / 100.0

            return yes_ask, no_ask

        except Exception as exc:
            self.logger.debug("Orderbook fetch failed for %s: %s", ticker, exc)
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Probability + sizing math
    # ------------------------------------------------------------------

    def _estimate_true_prob(self, abs_momentum_pct: float) -> float:
        """Convert |momentum %| to a true win probability via logistic sigmoid.

        Calibration:
          0.30% → 0.61,  0.50% → 0.68,  1.0% → 0.82,  2.0% → 0.95
        """
        x = (abs_momentum_pct / 100.0) * _SIGMOID_SCALE
        return 1.0 / (1.0 + math.exp(-x))

    def _kelly_notional(self, edge: float, ask_price: float, equity: float) -> float:
        """Return position notional in USD using fractional Kelly criterion.

        For a binary prediction market:
          f* = edge / (1 − ask_price)
          notional = equity × f* × kelly_fraction
          notional = min(notional, equity × max_position_pct)
        """
        if ask_price >= 1.0 or ask_price <= 0:
            return 0.0
        kelly_full = edge / (1.0 - ask_price)
        kelly_full = max(0.0, kelly_full)
        notional = equity * kelly_full * self.config.kelly_fraction
        max_allowed = equity * self.config.max_position_pct
        return min(notional, max_allowed)

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _execute_trade(
        self,
        *,
        state: PortfolioState,
        market: dict[str, Any],
        side: str,
        ask_price: float,
        notional: float,
        momentum_pct: float,
        now: datetime,
        market_regime: str | None = None,
    ) -> bool:
        """Execute a latency-arb trade (paper or live)."""
        ticker = market.get("ticker", "")
        symbol = f"KL:{ticker}:{side.upper()}"

        if symbol in state.positions:
            self.logger.debug("Already holding %s. Skipping.", symbol)
            return False

        # Optional Haiku-only AI signal validation.
        # The verdict is used to adjust Kelly notional via apply_ai_confirmation
        # (boost=1.5x on confident buy, 0.5x on skip, blocked in require mode).
        ai_meta: dict[str, Any] = {}
        ai_verdict = None
        if self.ai_service is not None:
            try:
                ai_verdict = self.ai_service.evaluate_arb_signal(
                    asset=ticker.split("-")[0] if "-" in ticker else ticker[:6],
                    momentum_pct=momentum_pct,
                    contract_title=market.get("title", ""),
                    contract_price=ask_price,
                    edge=_kelly_edge(ask_price, self._estimate_true_prob(abs(momentum_pct))),
                )
                if ai_verdict is not None:
                    ai_meta = {
                        "ai_action": ai_verdict.action,
                        "ai_confidence": round(ai_verdict.confidence, 3),
                        "ai_reasoning": ai_verdict.reasoning,
                        "ai_cost_usd": ai_verdict.cost_usd,
                        "ai_cached": ai_verdict.cached,
                    }
                    self.logger.info(
                        "AI arb check: action=%s confidence=%.2f (%s) | %s",
                        ai_verdict.action, ai_verdict.confidence,
                        "cached" if ai_verdict.cached else "live",
                        ai_verdict.reasoning[:80],
                    )
            except Exception as exc:
                self.logger.debug("AI arb validation failed (non-blocking): %s", exc)

        # Apply AI confirmation to Kelly notional (boost/require/log_only modes)
        if apply_ai_confirmation is not None:
            confirmation_mode = getattr(
                getattr(self, "config", None),
                "ai_confirmation_mode",
                "boost",
            )
            notional, conf_meta = apply_ai_confirmation(
                ai_verdict, confirmation_mode, notional, self.logger
            )
            ai_meta.update({k: v for k, v in conf_meta.items() if k not in ai_meta})

        if notional < self.config.min_trade_usd:
            self.logger.debug(
                "Notional after AI adjustment ($%.2f) below min $%.2f. Skipping.",
                notional, self.config.min_trade_usd,
            )
            return False

        # Contracts are $1 each on Kalshi; quantity = notional / ask_price
        quantity = math.floor(notional / ask_price)
        if quantity < 1:
            return False

        actual_notional = quantity * ask_price

        meta = {
            "ticker": ticker,
            "side": side,
            "series": market.get("event_ticker", ""),
            "title": market.get("title", "")[:120],
            "momentum_pct": round(momentum_pct, 4),
            "strategy": "latency_arb",
            "market_regime": market_regime,
            **ai_meta,
        }

        # Paper tracking (always)
        position = self.broker.open_position(
            state,
            symbol=symbol,
            asset_type="prediction",
            strategy="latency_arb",
            price=ask_price,
            notional=actual_notional,
            opened_at=now,
            meta=meta,
        )
        if position is None:
            return False

        # Use the broker-tracked quantity (may be reduced if cash was insufficient)
        filled_quantity = int(position.quantity)

        self.logger.info(
            "LATENCY ARB %s %s %s @ $%.2f × %d contracts = $%.2f | momentum=%.3f%%",
            "PAPER" if not self.kalshi.can_trade else "LIVE",
            side.upper(), ticker, ask_price, filled_quantity,
            filled_quantity * ask_price, momentum_pct,
        )

        # Live execution
        if self.kalshi.can_trade:
            try:
                yes_price_cents = int(ask_price * 100)
                order_result = self.kalshi.place_order(
                    ticker=ticker,
                    side=side,
                    action="buy",
                    count=filled_quantity,
                    yes_price=yes_price_cents if side == "yes" else (100 - yes_price_cents),
                    client_order_id=str(uuid.uuid4()),
                )
                order_id = order_result.get("order", {}).get("order_id", "")
                state.positions[symbol].meta["kalshi_order_id"] = order_id
                self.logger.info("Live Kalshi order placed: %s", order_id)
            except Exception as exc:
                self.logger.error("Live Kalshi order failed for %s: %s", ticker, exc)
                # Keep the paper position for tracking; live failed

        return True

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        result: dict[str, Any],
        now: datetime,
    ) -> None:
        """Apply take-profit and stop-loss to open latency arb positions.

        When the AI service is available, exit decisions for take-profit and
        stop-loss are validated by ``evaluate_exit_signal`` (Haiku-only).
        The AI can veto a premature exit (returns ``"hold"``); the quantitative
        rules are always authoritative if AI is unavailable.
        """
        arb_positions = {
            sym: pos
            for sym, pos in state.positions.items()
            if pos.strategy == "latency_arb" and pos.asset_type == "prediction"
        }
        # Use the cached regime label from the last run_once() if available
        current_regime: str | None = result.get("market_regime")

        for symbol, position in list(arb_positions.items()):
            ticker = position.meta.get("ticker", "")
            side = position.meta.get("side", "yes")
            if not ticker:
                continue

            try:
                yes_ask, no_ask = self._get_market_prices(ticker)
                current_price = yes_ask if side == "yes" else no_ask
                if current_price <= 0:
                    continue

                position.current_price = current_price
                state.last_prices[symbol] = current_price

                entry = position.entry_price
                gain_pct = (current_price - entry) / entry if entry > 0 else 0.0
                loss_pct = (entry - current_price) / entry if entry > 0 else 0.0

                reason = None
                if gain_pct >= self.config.take_profit_pct:
                    reason = "take_profit"
                elif loss_pct >= self.config.stop_loss_pct:
                    reason = "stop_loss"

                if reason:
                    # AI exit validation: Haiku may veto a borderline exit
                    if self.ai_service is not None:
                        try:
                            exit_verdict = self.ai_service.evaluate_exit_signal(
                                asset=ticker,
                                side=side,
                                entry_price=entry,
                                current_price=current_price,
                                gain_pct=gain_pct * 100.0 if reason == "take_profit" else -loss_pct * 100.0,
                                market_regime=current_regime,
                                reason=reason,
                            )
                            if exit_verdict is not None and exit_verdict.action == "hold" and exit_verdict.confidence >= _AI_EXIT_VETO_MIN_CONFIDENCE:
                                self.logger.info(
                                    "AI vetoed %s exit for %s (conf=%.2f): %s",
                                    reason, symbol, exit_verdict.confidence,
                                    exit_verdict.reasoning[:80],
                                )
                                continue  # Skip exit this cycle
                        except Exception as exc:
                            self.logger.debug("AI exit validation failed (non-blocking): %s", exc)

                    closed = broker.close_position(
                        state, symbol=symbol, price=current_price, reason=reason
                    )
                    if closed:
                        result["sells"].append(symbol)
                        self.logger.info(
                            "CLOSE %s %s @ $%.2f (entry=$%.2f, %s)",
                            symbol, reason, current_price, entry, reason,
                        )

            except Exception as exc:
                self.logger.warning("Position management failed for %s: %s", symbol, exc)

    def _count_open_positions(self, state: PortfolioState) -> int:
        """Count open latency arb positions."""
        return sum(
            1 for pos in state.positions.values()
            if pos.strategy == "latency_arb" and pos.asset_type == "prediction"
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _kelly_edge(ask_price: float, true_prob: float) -> float:
    """Compute edge = true_prob - ask_price for a binary contract."""
    return max(0.0, true_prob - ask_price)


def _best_ask(price_levels: list[Any]) -> float:
    """Extract the best (lowest) ask price from a Kalshi price-level list.

    Kalshi orderbook sides are lists of [price, quantity] pairs,
    sorted ascending for asks.
    """
    if not price_levels:
        return 0.0
    try:
        # Each level may be a list [price, qty] or a dict {"price": ..., "quantity": ...}
        for level in price_levels:
            if isinstance(level, (list, tuple)) and len(level) >= 1:
                return float(level[0])
            if isinstance(level, dict):
                return float(level.get("price", 0))
    except (TypeError, ValueError, IndexError):
        pass
    return 0.0
