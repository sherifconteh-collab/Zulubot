from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from typing import Any

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

    def mark_positions(self, state: PortfolioState, prices: dict[str, float]) -> None:
        today = date.today().isoformat()
        record_today = state.last_price_record_date != today

        for symbol, price in prices.items():
            if price <= 0:
                continue
            state.last_prices[symbol] = price
            if symbol in state.positions:
                state.positions[symbol].current_price = price
            if record_today:
                VolatilityEngine.record_price(
                    symbol, price, state.price_history, self._max_price_history_days
                )

        if record_today and prices:
            state.last_price_record_date = today

    def portfolio_equity(self, state: PortfolioState) -> float:
        market_value = 0.0
        for position in state.positions.values():
            last_price = state.last_prices.get(position.symbol, position.current_price or position.entry_price)
            market_value += position.quantity * last_price
        return state.cash + market_value

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
    ) -> Position | None:
        self._ensure_paper_mode()
        if symbol in state.positions:
            self.logger.info("Skipping buy for %s because a position already exists.", symbol)
            return None
        if price <= 0 or notional <= 0:
            self.logger.warning("Skipping buy for %s because price or notional is invalid.", symbol)
            return None

        spendable = min(notional, state.cash)
        precision = 6 if asset_type == "crypto" else 4
        quantity = self._round_down(spendable / price, precision)
        if quantity <= 0:
            self.logger.warning("Skipping buy for %s because position size rounded to zero.", symbol)
            return None

        actual_notional = quantity * price
        state.cash -= actual_notional
        position = Position(
            symbol=symbol,
            asset_type=asset_type,
            strategy=strategy,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            opened_at=opened_at.isoformat(),
            meta=meta or {},
        )
        state.positions[symbol] = position
        state.last_prices[symbol] = price
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
        return position

    def close_position(
        self,
        state: PortfolioState,
        *,
        symbol: str,
        price: float | None = None,
        reason: str,
    ) -> Position | None:
        self._ensure_paper_mode()
        position = state.positions.get(symbol)
        if position is None:
            self.logger.info("Skipping close for %s because no position exists.", symbol)
            return None

        close_price = price or state.last_prices.get(symbol, position.current_price or position.entry_price)
        proceeds = position.quantity * close_price
        realized_pnl = (close_price - position.entry_price) * position.quantity
        state.cash += proceeds
        state.realized_pnl += realized_pnl
        state.last_prices[symbol] = close_price
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

