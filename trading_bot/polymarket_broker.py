"""Polymarket broker for paper and live execution of prediction market trades.

Paper mode: Uses the existing PaperBroker to track positions.
Live mode: Would use py-clob-client for limit orders (requires Polygon wallet).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.models import PolymarketMarket, PortfolioState, Position


class PolymarketBroker:
    """Broker for Polymarket prediction market trades.

    In paper mode, delegates to PaperBroker with asset_type="prediction".
    Symbol format: PM:{condition_id[:8]}:{YES|NO}
    Full condition_id and token_id stored in Position.meta.
    """

    def __init__(
        self,
        paper_broker: PaperBroker,
        paper_mode: bool,
        logger: logging.Logger,
    ) -> None:
        self.paper_broker = paper_broker
        self.paper_mode = paper_mode
        self.logger = logger.getChild("polymarket_broker")

    def place_limit_buy(
        self,
        state: PortfolioState,
        *,
        market: PolymarketMarket,
        side: str,  # "YES" or "NO"
        price: float,
        notional_usd: float,
        strategy: str,
        now: datetime,
        meta: dict[str, Any] | None = None,
    ) -> Position | None:
        """Buy a prediction market position.

        Args:
            market: The Polymarket market to trade.
            side: "YES" or "NO" — which outcome to buy.
            price: The price (probability) to buy at (0.0 to 1.0).
            notional_usd: Dollar amount to spend.
            strategy: Strategy name for tracking.
            now: Current timestamp.
            meta: Additional metadata.
        """
        side = side.upper()
        if side not in ("YES", "NO"):
            self.logger.error("Invalid side '%s'. Must be YES or NO.", side)
            return None

        token_id = market.yes_token_id if side == "YES" else market.no_token_id
        symbol = f"PM:{market.condition_id[:8]}:{side}"

        if symbol in state.positions:
            self.logger.info("Already holding %s, skipping buy.", symbol)
            return None

        if price <= 0 or price >= 1.0:
            self.logger.warning("Invalid price %.4f for %s. Must be between 0 and 1.", price, symbol)
            return None

        position_meta = {
            "condition_id": market.condition_id,
            "token_id": token_id,
            "question": market.question[:200],
            "side": side,
            "market_tags": market.tags[:5],
            **(meta or {}),
        }

        if self.paper_mode:
            # Use PaperBroker for position tracking
            # Quantity = notional / price (number of outcome shares)
            position = self.paper_broker.open_position(
                state,
                symbol=symbol,
                asset_type="prediction",
                strategy=strategy,
                price=price,
                notional=notional_usd,
                opened_at=now,
                meta=position_meta,
            )
            if position:
                self.logger.info(
                    "PAPER BUY %s @ %.4f notional=$%.2f question='%s'",
                    symbol, price, notional_usd, market.question[:80],
                )
            return position
        else:
            # Live mode: would use py-clob-client
            self.logger.error(
                "Live Polymarket execution not yet implemented. "
                "Set PAPER_MODE=true or implement py-clob-client integration."
            )
            return None

    def place_limit_sell(
        self,
        state: PortfolioState,
        *,
        symbol: str,
        price: float,
        reason: str,
    ) -> Position | None:
        """Sell (close) a prediction market position.

        Args:
            symbol: The position symbol (PM:{id}:{YES|NO}).
            price: The current price to close at.
            reason: Why closing (take_profit, stop_loss, resolved, etc.).
        """
        if symbol not in state.positions:
            self.logger.info("No position for %s to sell.", symbol)
            return None

        if self.paper_mode:
            closed = self.paper_broker.close_position(
                state,
                symbol=symbol,
                price=price,
                reason=reason,
            )
            if closed:
                self.logger.info("PAPER SELL %s @ %.4f reason=%s", symbol, price, reason)
            return closed
        else:
            self.logger.error("Live Polymarket sell not yet implemented.")
            return None

    def get_prediction_positions(self, state: PortfolioState) -> dict[str, Position]:
        """Get all open prediction market positions."""
        return {
            symbol: pos
            for symbol, pos in state.positions.items()
            if pos.asset_type == "prediction"
        }
