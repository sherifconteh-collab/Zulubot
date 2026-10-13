"""Manager for Kalshi prediction-market positions opened by earlier versions.

The price-threshold rule that used to open these positions (buy whatever side was cheap) was
retired: it had no view of the true probability, so it paid fees on long shots that lose far more
often than their price implies. This class no longer opens anything. It only keeps the books honest
for positions that rule already opened: it closes resolved contracts and applies the old take-profit
and stop-loss exits. New bets on non-sports Kalshi markets come only from strategies that carry a
probability model (see ``gdp_nowcast``).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients_kalshi import KalshiClient, order_fill_count
from trading_bot.config import PredictionConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager


class CoinbasePredictionStrategy:
    """Closes and settles legacy prediction positions. Never opens a new one."""

    _KALSHI_CASH_BUCKET = "kalshi"

    def __init__(
        self,
        *,
        config: PredictionConfig,
        kalshi_client: KalshiClient,
        logger: logging.Logger,
        live_trading_enabled: bool = False,
    ) -> None:
        self.config = config
        self.kalshi = kalshi_client
        self.logger = logger.getChild("strategy.predictions")
        self.live_trading_enabled = live_trading_enabled

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
        fear_greed_value: int | None = None,
    ) -> dict[str, Any]:
        """Settle resolved contracts and apply the exits to positions already held."""
        if not self.config.enabled:
            return {"status": "disabled", "buys": [], "sells": [], "markets_scanned": 0}
        result: dict[str, Any] = {
            "status": "ok", "buys": [], "sells": [], "markets_scanned": 0, "markets_discovered": 0,
            "resolved_closed": [], "fear_greed": fear_greed_value, "can_trade": self.live_trading_enabled,
            "decisions": [], "note": "No new bets: the long-shot rule was retired. See /gdp.",
        }
        try:
            self._close_resolved_positions(state, broker, result)
            self._manage_existing_positions(state, broker, result)
        except Exception as exc:
            self.logger.error("Prediction position management failed: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)
        return result

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _get_prediction_positions(self, state: PortfolioState) -> dict[str, Any]:
        return {
            symbol: pos
            for symbol, pos in state.positions.items()
            if pos.asset_type == "prediction" and pos.strategy == "coinbase_predictions"
        }

    def _close_resolved_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        result: dict[str, Any],
    ) -> None:
        """Detect resolved/expired contracts and close them."""
        positions = self._get_prediction_positions(state)
        if not positions:
            return

        for symbol, position in list(positions.items()):
            ticker = position.meta.get("ticker", "")
            if not ticker:
                continue

            try:
                market = self.kalshi.get_market(ticker)
                status = market.get("status", "").lower()

                if status in ("settled", "finalized", "closed"):
                    result_val = market.get("result", "").lower()
                    side = position.meta.get("side", "yes")
                    if result_val == side:
                        close_price = 1.0
                    elif result_val:
                        close_price = 0.0
                    else:
                        close_price = position.current_price

                    self.logger.info(
                        "Market %s resolved: result=%s side=%s close_price=%.2f",
                        ticker, result_val, side, close_price,
                    )

                    broker.close_position(
                        state, symbol=symbol, price=close_price,
                        reason=f"resolved:{result_val or 'expired'}",
                    )
                    result["resolved_closed"].append(symbol)

            except Exception as exc:
                self.logger.warning("Failed to check resolution for %s: %s", ticker, exc)

    def _manage_existing_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        result: dict[str, Any],
    ) -> None:
        """Apply take-profit and stop-loss rules to existing positions."""
        positions = self._get_prediction_positions(state)
        if not positions:
            return

        for symbol, position in list(positions.items()):
            ticker = position.meta.get("ticker", "")
            if not ticker:
                continue

            try:
                market = self.kalshi.get_market(ticker)
                yes_bid = _parse_dollars(market.get("yes_bid_dollars"))
                yes_ask = _parse_dollars(market.get("yes_ask_dollars"))
                no_bid = _parse_dollars(market.get("no_bid_dollars"))
                no_ask = _parse_dollars(market.get("no_ask_dollars"))
                yes_mid = (yes_bid + yes_ask) / 2 if yes_bid > 0 and yes_ask > 0 else (yes_bid or yes_ask)
                no_mid = (no_bid + no_ask) / 2 if no_bid > 0 and no_ask > 0 else (no_bid or no_ask)
                if no_mid <= 0 and yes_mid > 0:
                    no_mid = 1.0 - yes_mid
                if yes_mid <= 0 and no_mid > 0:
                    yes_mid = 1.0 - no_mid

                side = position.meta.get("side", "yes")
                current_price = yes_mid if side == "yes" else no_mid

                if current_price <= 0:
                    continue

                broker.mark_positions(state, {symbol: current_price})
                entry_price = position.entry_price

                # Take profit
                if self._should_take_profit(entry_price, current_price):
                    self._close_position(state, broker, symbol, position, current_price, "take_profit", result)
                    continue

                # Stop loss
                if self._should_stop_loss(entry_price, current_price):
                    self._close_position(state, broker, symbol, position, current_price, "stop_loss", result)

            except Exception as exc:
                self.logger.warning("Failed to manage position %s: %s", symbol, exc)

    def _close_position(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        symbol: str,
        position: Any,
        price: float,
        reason: str,
        result: dict[str, Any],
    ) -> None:
        """Close a prediction position (live on Kalshi + paper bookkeeping)."""
        ticker = position.meta.get("ticker", "")
        side = position.meta.get("side", "yes")

        # Sell on Kalshi if we can trade. Only close the books when the contracts really
        # left: a failed or unfilled sell leaves the position open to retry next cycle.
        if self.live_trading_enabled and ticker:
            try:
                order = self.kalshi.sell_to_close(ticker=ticker, side=side, count=int(position.quantity))
            except Exception as exc:
                self.logger.warning("Kalshi sell failed for %s: %s; keeping position open.", ticker, exc)
                return
            sold = order_fill_count(order)
            if sold is not None and sold < 1:
                self.logger.warning("Kalshi sell for %s did not fill; keeping position open.", ticker)
                return
            held = int(position.quantity)
            if sold is not None and sold < held:
                # Partial fill: shrink the position to what is still held and retry later.
                broker.resize_position(state, symbol, held - sold, refund_price=price)
                self.logger.warning("Kalshi sell for %s filled %d of %d; remainder stays open.",
                                    ticker, sold, held)
                return

        closed = broker.close_position(
            state,
            symbol=symbol,
            price=price,
            reason=reason,
            cash_bucket=self._KALSHI_CASH_BUCKET if self.live_trading_enabled else None,
        )
        if closed is not None:
            result["sells"].append(symbol)
            result["decisions"].append({
                "strategy": "coinbase_predictions",
                "symbol": symbol,
                "action": "sell",
                "reason": reason,
                "signals": {
                    "side": side,
                    "entry_price": round(position.entry_price, 4),
                    "exit_price": round(price, 4),
                },
            })
            self.logger.info(
                "%s %s: side=%s entry=%.4f exit=%.4f",
                reason, symbol, side, position.entry_price, price,
            )

    def _should_take_profit(self, entry: float, current: float) -> bool:
        threshold = self.config.take_profit_threshold
        return entry < threshold and current >= threshold

    def _should_stop_loss(self, entry: float, current: float) -> bool:
        max_loss = self.config.stop_loss_pct
        return (entry - current) >= max_loss


def _parse_dollars(value: Any) -> float:
    """Parse a dollar-string field like '0.0100' to float."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
