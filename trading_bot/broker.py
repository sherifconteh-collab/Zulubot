from __future__ import annotations

import json
import logging
import math
import time
from datetime import date, datetime, timezone
from typing import Any, Callable

from trading_bot.models import PortfolioState, Position
from trading_bot.volatility import VolatilityEngine


class PaperBroker:
    def __init__(
        self,
        paper_mode: bool,
        logger: logging.Logger,
        max_price_history_days: int = 60,
    ) -> None:
        self.paper_mode = paper_mode
        self.logger = logger.getChild("broker")
        self._max_price_history_days = max_price_history_days
        self._on_close: Callable[[PortfolioState, float, float], None] | None = None
        self._on_position_closed: (
            Callable[[PortfolioState, str, float, float, float, str], None] | None
        ) = None
        self._on_trade: Callable[[dict[str, Any]], None] | None = None

    def set_on_close(
        self,
        callback: Callable[[PortfolioState, float, float], None],
    ) -> None:
        """Register a callback invoked after every ``close_position()``.

        The callback receives ``(state, entry_price, close_price)`` so it can
        track trade results (e.g. consecutive-loss cooldown) regardless of
        which code path closed the position.
        """
        self._on_close = callback

    def set_on_position_closed(
        self,
        callback: Callable[[PortfolioState, str, float, float, float, str], None],
    ) -> None:
        """Register a detailed close callback.

        Signature: ``(state, symbol, entry_price, close_price, realized_pnl, reason)``.
        Used by the per-symbol performance memory / learning layer.
        """
        self._on_position_closed = callback

    def set_on_trade(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked after each open/close bookkeeping event."""
        self._on_trade = callback

    def mark_positions(
        self,
        state: PortfolioState,
        prices: dict[str, float],
        *,
        now: datetime | None = None,
    ) -> None:
        today = now.date().isoformat() if now is not None else date.today().isoformat()
        record_today = state.last_price_record_date != today
        price_ts = now.timestamp() if now is not None else time.time()

        for symbol, price in prices.items():
            if price <= 0:
                continue
            self._record_price(state, symbol, price, timestamp=price_ts)
            if record_today:
                VolatilityEngine.record_price(
                    symbol, price, state.price_history, self._max_price_history_days
                )

        if record_today and prices:
            state.last_price_record_date = today

    def portfolio_equity(self, state: PortfolioState) -> float:
        return self.portfolio_equity_for_bucket(state)

    def portfolio_equity_for_bucket(
        self,
        state: PortfolioState,
        *,
        cash_bucket: str | None = None,
    ) -> float:
        if cash_bucket is None:
            total_cash = state.cash + sum(float(value) for value in state.cash_reserves.values())
        else:
            total_cash = self.get_cash_balance(state, cash_bucket=cash_bucket)
        market_value = 0.0
        for position in self._iter_bucket_positions(state, cash_bucket=cash_bucket):
            last_price = state.last_prices.get(position.symbol, position.current_price or position.entry_price)
            market_value += position.quantity * last_price
        return total_cash + market_value

    def get_cash_balance(self, state: PortfolioState, *, cash_bucket: str | None = None) -> float:
        if cash_bucket is None or cash_bucket == "":
            return state.cash
        return float(state.cash_reserves.get(cash_bucket, 0.0))

    def open_position(
        self,
        state: PortfolioState,
        *,
        symbol: str,
        asset_type: str,
        strategy: str,
        price: float,
        notional: float,
        opened_at: datetime,
        meta: dict[str, Any] | None = None,
        cash_bucket: str | None = None,
    ) -> Position | None:
        self._ensure_paper_mode()
        if symbol in state.positions:
            self.logger.info("Skipping buy for %s because a position already exists.", symbol)
            return None
        if price <= 0 or notional <= 0:
            self.logger.warning("Skipping buy for %s because price or notional is invalid.", symbol)
            return None

        spendable = min(notional, self.get_cash_balance(state, cash_bucket=cash_bucket))
        precision = 6 if asset_type == "crypto" else 4
        quantity = self._round_down(spendable / price, precision)
        if quantity <= 0:
            self.logger.warning("Skipping buy for %s because position size rounded to zero.", symbol)
            return None

        actual_notional = quantity * price
        self._adjust_cash_balance(state, -actual_notional, cash_bucket=cash_bucket)
        position_meta = dict(meta or {})
        if cash_bucket:
            position_meta.setdefault("cash_bucket", cash_bucket)
        position = Position(
            symbol=symbol,
            asset_type=asset_type,
            strategy=strategy,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            opened_at=opened_at.isoformat(),
            meta=position_meta,
        )
        state.positions[symbol] = position
        self._record_price(state, symbol, price, timestamp=opened_at.timestamp())
        self.logger.info(
            "OPEN %s asset=%s strategy=%s qty=%.6f price=%.4f notional=%.2f meta=%s",
            symbol,
            asset_type,
            strategy,
            quantity,
            price,
            actual_notional,
            json.dumps(meta or {}, sort_keys=True),
        )
        self._emit_trade_event(
            {
                "timestamp": opened_at.isoformat(),
                "event": "open",
                "symbol": symbol,
                "asset_type": asset_type,
                "strategy": strategy,
                "quantity": quantity,
                "price": price,
                "notional": actual_notional,
                "cash_bucket": cash_bucket,
                "meta": position_meta,
            }
        )
        return position

    def close_position(
        self,
        state: PortfolioState,
        *,
        symbol: str,
        price: float | None = None,
        reason: str,
        cash_bucket: str | None = None,
    ) -> Position | None:
        self._ensure_paper_mode()
        position = state.positions.get(symbol)
        if position is None:
            self.logger.info("Skipping close for %s because no position exists.", symbol)
            return None

        # `price=0.0` is meaningful (a prediction contract that settled worthless),
        # so only fall back when no price was given at all.
        if price is not None:
            close_price = price
        else:
            close_price = state.last_prices.get(symbol, position.current_price or position.entry_price)
        entry_price = position.entry_price
        proceeds = position.quantity * close_price
        realized_pnl = (close_price - entry_price) * position.quantity
        resolved_cash_bucket = cash_bucket if cash_bucket is not None else self._position_cash_bucket(position)
        self._adjust_cash_balance(state, proceeds, cash_bucket=resolved_cash_bucket)
        state.realized_pnl += realized_pnl
        self._record_price(state, symbol, close_price)
        del state.positions[symbol]
        self.logger.info(
            "CLOSE %s qty=%.6f price=%.4f proceeds=%.2f realized_pnl=%.2f reason=%s",
            symbol,
            position.quantity,
            close_price,
            proceeds,
            realized_pnl,
            reason,
        )
        self._emit_trade_event(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "close",
                "symbol": symbol,
                "asset_type": position.asset_type,
                "strategy": position.strategy,
                "quantity": position.quantity,
                "price": close_price,
                "notional": proceeds,
                "cash_bucket": resolved_cash_bucket,
                "realized_pnl": realized_pnl,
                "reason": reason,
                "entry_price": entry_price,
                "meta": dict(position.meta),
            }
        )
        if self._on_close is not None:
            self._on_close(state, entry_price, close_price)
        if self._on_position_closed is not None:
            self._on_position_closed(
                state, symbol, entry_price, close_price, realized_pnl, reason,
            )
        return position

    def resize_position(
        self,
        state: PortfolioState,
        symbol: str,
        new_quantity: float,
        *,
        refund_price: float | None = None,
        refund: bool = True,
    ) -> Position | None:
        """Shrink a position (e.g. after a partial fill) and return the cash for the rest.

        ``refund_price`` defaults to the entry price (use it to undo an over-booked buy;
        pass the sale price when contracts were actually sold). ``refund=False`` only fixes
        the bookkeeping, for when cash is synced from the real account. Returns the
        position, or None if it was removed because ``new_quantity`` is zero.
        """
        position = state.positions.get(symbol)
        if position is None or new_quantity >= position.quantity:
            return position
        price = position.entry_price if refund_price is None else refund_price
        refund_amount = (position.quantity - max(new_quantity, 0.0)) * price
        refund_cash = refund
        bucket = self._position_cash_bucket(position)
        if refund_cash:
            self._adjust_cash_balance(state, refund_amount, cash_bucket=bucket)
        if new_quantity <= 0:
            del state.positions[symbol]
            return None
        position.quantity = new_quantity
        return position

    def close_all(self, state: PortfolioState, *, reason: str) -> None:
        for symbol in list(state.positions):
            self.close_position(
                state,
                symbol=symbol,
                price=state.last_prices.get(symbol, state.positions[symbol].current_price),
                reason=reason,
            )

    def _ensure_paper_mode(self) -> None:
        # PaperBroker always tracks positions (bookkeeping).
        # Real crypto execution is handled by CoinbaseBroker separately.
        pass

    @staticmethod
    def _round_down(value: float, precision: int) -> float:
        factor = 10**precision
        return math.floor(value * factor) / factor

    @staticmethod
    def _position_cash_bucket(position: Position) -> str | None:
        raw_value = position.meta.get("cash_bucket")
        if isinstance(raw_value, str) and raw_value:
            return raw_value
        return None

    def _adjust_cash_balance(
        self,
        state: PortfolioState,
        amount: float,
        *,
        cash_bucket: str | None = None,
    ) -> None:
        if cash_bucket is None or cash_bucket == "":
            state.cash += amount
            return
        current = float(state.cash_reserves.get(cash_bucket, 0.0))
        state.cash_reserves[cash_bucket] = current + amount

    def _iter_bucket_positions(
        self,
        state: PortfolioState,
        *,
        cash_bucket: str | None = None,
    ) -> list[Position]:
        if cash_bucket is None:
            return list(state.positions.values())
        return [
            position
            for position in state.positions.values()
            if self._position_cash_bucket(position) == cash_bucket
        ]

    @staticmethod
    def _record_price(
        state: PortfolioState,
        symbol: str,
        price: float,
        *,
        timestamp: float | None = None,
    ) -> None:
        state.last_prices[symbol] = price
        state.last_price_timestamps[symbol] = time.time() if timestamp is None else timestamp
        if symbol in state.positions:
            state.positions[symbol].current_price = price

    def _emit_trade_event(self, payload: dict[str, Any]) -> None:
        if self._on_trade is None:
            return
        self._on_trade(payload)

