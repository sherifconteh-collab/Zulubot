"""Kalshi Crypto Perpetuals strategy.

Trades leveraged, non-expiring perpetual contracts (BTC PERP, ETH PERP, etc.)
on Kalshi using a momentum signal derived from in-memory price history.

Signal logic:
- If the perp market price has risen ≥ momentum_threshold_pct over the past
  momentum_window_minutes → open a LONG position.
- If it has fallen ≥ momentum_threshold_pct → open a SHORT position
  (blocked for protected symbols).

Position management:
- Take profit when the move in our favour ≥ take_profit_pct.
- Stop loss when the move against us ≥ stop_loss_pct.
- Optional trailing stop: close when price retraces trailing_stop_pct from peak.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients_kalshi import KalshiClient, PERP_SERIES
from trading_bot.config import KalshiPerpetualsConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager

_STRATEGY_NAME = "kalshi_perps"
_ASSET_TYPE = "perp"
_CASH_BUCKET = "kalshi"
_LIQUIDATION_MOVE_PCT = -1.0  # 100% of margin lost = wiped out


class KalshiPerpetualsStrategy:
    """Momentum-based Kalshi Crypto Perpetuals strategy."""

    def __init__(
        self,
        *,
        config: KalshiPerpetualsConfig,
        kalshi_client: KalshiClient,
        protected_symbols: frozenset[str] = frozenset(),
        logger: logging.Logger,
        live_trading_enabled: bool = False,
    ) -> None:
        self.config = config
        self.kalshi_client = kalshi_client
        self.protected_symbols = protected_symbols
        self.logger = logger.getChild("kalshi_perps")
        self.live_trading_enabled = live_trading_enabled
        # symbol → deque of (timestamp, price) observations
        self._price_history: dict[str, deque[tuple[datetime, float]]] = {}

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"status": "disabled", "opens": [], "closes": [], "decisions": []}

        opens: list[dict[str, Any]] = []
        closes: list[dict[str, Any]] = []

        markets = self._discover_markets()
        if not markets:
            self.logger.info("Kalshi perps: no markets returned.")
            return {"status": "ok", "opens": [], "closes": [], "decisions": []}

        # Record prices for momentum tracking
        for m in markets:
            sym = m.get("_symbol", "")
            price = m.get("_price", 0.0)
            if sym and price > 0:
                self._record_price(sym, price, now)

        # Manage open positions first
        closes = self._manage_positions(state, broker, markets, now)

        # Scan for new entries
        if not risk_manager.can_open_new_position(state):
            self.logger.info("Kalshi perps: new entries blocked by risk guardrails.")
        else:
            opens = self._scan_for_entries(state, broker, markets, now)

        decisions = (
            [{"action": "close", **c} for c in closes]
            + [{"action": "open", **o} for o in opens]
        )
        return {
            "status": "ok",
            "opens": [o.get("symbol", "") for o in opens],
            "closes": [c.get("symbol", "") for c in closes],
            "decisions": decisions,
        }

    # ------------------------------------------------------------------ #
    # Market discovery                                                     #
    # ------------------------------------------------------------------ #

    def _discover_markets(self) -> list[dict[str, Any]]:
        """Fetch open perp markets for each configured symbol."""
        markets: list[dict[str, Any]] = []
        for raw_symbol in self.config.symbols:
            symbol = raw_symbol.upper()
            series = PERP_SERIES.get(symbol)
            if not series:
                self.logger.debug("No PERP_SERIES entry for %s; skipping.", symbol)
                continue
            try:
                page, _ = self.kalshi_client.get_perp_markets(
                    series_ticker=series, limit=20
                )
                for m in page:
                    m["_symbol"] = symbol
                    m["_price"] = self._extract_price(m)
                    m["_series"] = series
                    m["_max_leverage"] = self._extract_leverage(m)
                markets.extend(page)
            except Exception as exc:
                self.logger.warning("Failed to fetch %s perp markets: %s", symbol, exc)
        return markets

    def _extract_price(self, market: dict[str, Any]) -> float:
        """Extract the current mid-price from a perp market dict.

        Tries common field names; falls back to 0.0 if none present.
        """
        for field in ("mark_price", "last_price", "price", "close_price"):
            raw = market.get(field)
            if raw is not None:
                try:
                    return float(raw)
                except (ValueError, TypeError):
                    continue
        # Binary market fallback: mid of yes bid/ask (price in cents → dollars)
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        if yes_bid is not None and yes_ask is not None:
            try:
                return (float(yes_bid) + float(yes_ask)) / 2.0 / 100.0
            except (ValueError, TypeError):
                pass
        return 0.0

    def _extract_leverage(self, market: dict[str, Any]) -> float:
        """Extract max available leverage from market metadata."""
        raw = market.get("max_leverage") or market.get("leverage")
        if raw is not None:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass
        return self.config.max_leverage

    # ------------------------------------------------------------------ #
    # Leveraged/short PnL representation                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _synthetic_price(entry_price: float, leveraged_move_pct: float) -> float:
        """Encode leveraged/short PnL as a price broker.Position can consume.

        broker.Position has no leverage or short semantics — unrealized_pnl
        and market_value are plain (price - entry_price) * quantity — so we
        fabricate a price whose plain-long PnL matches our actual leveraged
        PnL. Clamped so a position can never be marked/closed past a total
        loss of the margin committed: a raw ``entry_price * (1 + move)``
        would hit zero or go negative for a big enough adverse move (move
        <= -1/leverage), which PaperBroker.mark_positions silently ignores
        (leaving stale PnL) and which close_position would otherwise turn
        into a negative-price, unbounded-loss close.
        """
        floor = -1.0 + 1e-6
        return entry_price * (1 + max(leveraged_move_pct, floor))

    # ------------------------------------------------------------------ #
    # Price history + momentum                                             #
    # ------------------------------------------------------------------ #

    def _record_price(self, symbol: str, price: float, now: datetime) -> None:
        if symbol not in self._price_history:
            self._price_history[symbol] = deque()
        self._price_history[symbol].append((now, price))
        cutoff = now - timedelta(minutes=self.config.momentum_window_minutes * 2)
        hist = self._price_history[symbol]
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def _momentum(self, symbol: str, now: datetime) -> float | None:
        """Return % price change over momentum_window_minutes. None if no data."""
        hist = self._price_history.get(symbol)
        if not hist or len(hist) < 2:
            return None
        cutoff = now - timedelta(minutes=self.config.momentum_window_minutes)
        old = [(t, p) for t, p in hist if t <= cutoff]
        if not old:
            return None
        oldest_price = old[-1][1]  # price closest to the window boundary, not the oldest in 2x window
        latest_price = hist[-1][1]
        if oldest_price <= 0:
            return None
        return (latest_price - oldest_price) / oldest_price

    # ------------------------------------------------------------------ #
    # Position management                                                  #
    # ------------------------------------------------------------------ #

    def _manage_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        markets: list[dict[str, Any]],
        now: datetime,
    ) -> list[dict[str, Any]]:
        closes: list[dict[str, Any]] = []
        market_by_ticker: dict[str, dict[str, Any]] = {
            m.get("ticker", ""): m for m in markets
        }

        for pos_key, position in list(state.positions.items()):
            if position.strategy != _STRATEGY_NAME:
                continue
            meta = position.meta or {}
            ticker = meta.get("ticker", "")
            side = meta.get("side", "long")
            entry_price = position.entry_price
            if entry_price <= 0:
                continue

            market = market_by_ticker.get(ticker)
            if market is None:
                try:
                    market = self.kalshi_client.get_perp_market(ticker)
                    market["_price"] = self._extract_price(market)
                except Exception:
                    continue
            current_price = market.get("_price") or self._extract_price(market)
            if current_price <= 0:
                continue

            # PnL from our perspective
            if side == "long":
                move_pct = (current_price - entry_price) / entry_price
            else:
                move_pct = (entry_price - current_price) / entry_price

            # broker.Position has no short or leverage semantics (unrealized_pnl
            # and market_value are plain quantity * price), so mark with a
            # synthetic price that encodes our actual leveraged PnL direction —
            # the same trick used in _close_position for the real close. Cash
            # was only debited the margin amount on open, so quantity reflects
            # margin/price, not leveraged exposure; the synthetic price makes
            # up the difference.
            leverage = meta.get("leverage", 1.0)
            leveraged_move = leverage * move_pct
            mark_price = self._synthetic_price(entry_price, leveraged_move)
            broker.mark_positions(state, {pos_key: mark_price}, now=now)

            # Update high-water-mark for trailing stop
            hwm = meta.get("hwm", 0.0)
            if move_pct > hwm:
                meta["hwm"] = move_pct
                hwm = move_pct

            reason: str | None = None
            tp = self.config.take_profit_pct
            sl = self.config.stop_loss_pct
            ts = self.config.trailing_stop_pct
            if leveraged_move <= _LIQUIDATION_MOVE_PCT:
                # Adverse move wiped out the margin (>= 100% leveraged loss) —
                # force-close regardless of configured tp/sl so we never carry
                # a position whose synthetic price has been clamped to the
                # near-zero floor (stale/pinned PnL) or feed a non-positive
                # price into close_position.
                reason = "liquidation"
            elif tp > 0 and move_pct >= tp:
                reason = "take_profit"
            elif sl > 0 and move_pct <= -sl:
                reason = "stop_loss"
            elif ts > 0 and hwm >= ts and (hwm - move_pct) >= ts:
                reason = "trailing_stop"

            if reason:
                info = self._close_position(
                    state, broker, position, pos_key, ticker, side,
                    entry_price, current_price, reason,
                )
                if info:
                    closes.append(info)
        return closes

    def _close_position(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        position: Any,
        pos_key: str,
        ticker: str,
        side: str,
        entry_price: float,
        current_price: float,
        reason: str,
    ) -> dict[str, Any] | None:
        symbol = (position.meta or {}).get("symbol", pos_key)

        if self.live_trading_enabled:
            try:
                close_side = "short" if side == "long" else "long"
                self.kalshi_client.close_perp_position(ticker, close_side)
            except Exception as exc:
                self.logger.error("Live close failed for perp %s: %s", ticker, exc)
                return None

        # For paper broker PnL: map short/leveraged gain to a synthetic close
        # price so broker.close_position computes (close_price - entry_price)
        # * qty correctly (see the matching comment in _manage_positions).
        leverage = (position.meta or {}).get("leverage", 1.0)
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        broker_close_price = self._synthetic_price(entry_price, leverage * pnl_pct)

        broker.close_position(state, symbol=pos_key, price=broker_close_price, reason=reason)
        self.logger.info(
            "Closed perp %s %s @ %.4f (%s). entry=%.4f",
            side, ticker, current_price, reason, entry_price,
        )
        return {
            "symbol": symbol,
            "ticker": ticker,
            "side": side,
            "entry_price": entry_price,
            "exit_price": current_price,
            "reason": reason,
        }

    # ------------------------------------------------------------------ #
    # Entry scanning                                                       #
    # ------------------------------------------------------------------ #

    def _scan_for_entries(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        markets: list[dict[str, Any]],
        now: datetime,
    ) -> list[dict[str, Any]]:
        opens: list[dict[str, Any]] = []
        open_symbols: set[str] = {
            (pos.meta or {}).get("symbol", "").upper()
            for pos in state.positions.values()
            if pos.strategy == _STRATEGY_NAME
        }
        existing_count = sum(
            1 for pos in state.positions.values() if pos.strategy == _STRATEGY_NAME
        )
        slots = self.config.max_positions - existing_count
        if slots <= 0:
            return opens

        # In live mode, size relative to the Kalshi-bucket's own equity (cash +
        # its open positions) — using whole-account equity here would let a
        # large non-Kalshi balance make capital_allocation_pct exceed the
        # entire Kalshi bucket via the min() below, silently defeating the cap.
        # In paper mode there's no real per-venue bucket; kalshi_cash falls
        # back to the shared state.cash, so whole-account equity is consistent.
        if self.live_trading_enabled:
            kalshi_cash = state.cash_reserves.get(_CASH_BUCKET, 0.0)
            equity_basis = broker.portfolio_equity_for_bucket(state, cash_bucket=_CASH_BUCKET)
        else:
            kalshi_cash = state.cash
            equity_basis = broker.portfolio_equity(state)
        total_budget = min(
            equity_basis * self.config.capital_allocation_pct,
            kalshi_cash,
        )
        per_slot = total_budget / slots if slots > 0 else 0.0
        if per_slot < self.config.min_trade_usd:
            return opens

        for market in markets:
            if len(opens) >= slots:
                break
            symbol = market.get("_symbol", "")
            if not symbol or symbol in open_symbols:
                continue

            signal, _confidence = self._evaluate_signal(symbol, now)
            if signal is None:
                continue

            # Never short a protected symbol
            if signal == "short" and self._is_protected(symbol):
                self.logger.info("Perps: skipping short on protected symbol %s.", symbol)
                continue

            price = market.get("_price", 0.0)
            if price <= 0:
                continue

            leverage = min(
                self.config.max_leverage,
                market.get("_max_leverage", self.config.max_leverage),
            )
            info = self._open_position(
                state, broker, market, symbol, signal, price, per_slot, leverage, now
            )
            if info:
                opens.append(info)
                open_symbols.add(symbol)

        return opens

    def _evaluate_signal(
        self, symbol: str, now: datetime
    ) -> tuple[str | None, float]:
        """Return (signal, confidence) where signal ∈ {'long', 'short', None}."""
        mom = self._momentum(symbol, now)
        if mom is None:
            return None, 0.0
        threshold = self.config.momentum_threshold_pct
        if mom >= threshold:
            return "long", min(1.0, mom / (threshold * 2))
        if mom <= -threshold:
            return "short", min(1.0, abs(mom) / (threshold * 2))
        return None, 0.0

    def _is_protected(self, symbol: str) -> bool:
        """Check if a bare symbol (e.g. 'XRP') matches any protected_symbols entry."""
        return (
            symbol in self.protected_symbols
            or f"{symbol}USDT" in self.protected_symbols
            or f"{symbol}USD" in self.protected_symbols
            or f"{symbol}-USD" in self.protected_symbols
        )

    def _open_position(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        market: dict[str, Any],
        symbol: str,
        side: str,
        price: float,
        size_usd: float,
        leverage: float,
        now: datetime,
    ) -> dict[str, Any] | None:
        ticker = market.get("ticker", "")
        pos_key = f"KP:{ticker}:{side.upper()}"
        cash_bucket = _CASH_BUCKET if self.live_trading_enabled else None

        if self.live_trading_enabled:
            try:
                self.kalshi_client.place_perp_order(
                    ticker=ticker,
                    side=side,
                    size_usd=size_usd,
                    leverage=leverage,
                    client_order_id=str(uuid.uuid4()),
                )
            except Exception as exc:
                self.logger.error("Live open failed for perp %s %s: %s", side, ticker, exc)
                return None

        # Debit only the margin (size_usd) from cash — broker.open_position
        # couples quantity to the cash debited, so leveraging the notional
        # here would lock up leverage_x the intended capital. Leveraged
        # exposure/PnL is instead applied via a synthetic mark/close price
        # (see _manage_positions / _close_position).
        broker.open_position(
            state,
            symbol=pos_key,
            asset_type=_ASSET_TYPE,
            strategy=_STRATEGY_NAME,
            price=price,
            notional=size_usd,
            opened_at=now,
            cash_bucket=cash_bucket,
            meta={
                "ticker": ticker,
                "symbol": symbol,
                "side": side,
                "leverage": leverage,
                "entry_price": price,
                "hwm": 0.0,
                "venue": "kalshi",
                "cash_bucket": cash_bucket,
            },
        )
        self.logger.info(
            "Opened perp %s %s @ %.4f  size=$%.2f  leverage=%.1fx",
            side, ticker, price, size_usd, leverage,
        )
        return {
            "symbol": symbol,
            "ticker": ticker,
            "side": side,
            "entry_price": price,
            "notional_usd": size_usd,
            "leverage": leverage,
        }
