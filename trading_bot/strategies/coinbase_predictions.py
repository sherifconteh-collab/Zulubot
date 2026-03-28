"""Coinbase prediction market strategy.

Trades Coinbase event contracts (powered by Kalshi) using the Advanced Trade API.
These use KX-prefixed product IDs and YES/NO sides with prices from 0.00 to 1.00.

Signal logic:
  BUY YES when price < buy_yes_threshold (default 0.30)
  BUY NO  when price > buy_no_threshold  (default 0.70)
  SELL    when price crosses take_profit_threshold (0.50) toward our side
  SELL    when price moves against by stop_loss_pct (0.15)

Cross-reference:
  For crypto-related predictions, Fear & Greed is used as an optional signal boost.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients import CoinbaseClient
from trading_bot.coinbase_broker import CoinbaseBroker
from trading_bot.config import PredictionConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager


class CoinbasePredictionStrategy:
    """Prediction market strategy for Coinbase event contracts.

    Discovers KX-prefixed products from the Coinbase API, evaluates
    price-based entry signals, and manages positions with stop-loss
    and take-profit logic.  All bookkeeping flows through PaperBroker;
    live execution (when enabled) goes through CoinbaseBroker.
    """

    SYMBOL_PREFIX = "KX"

    def __init__(
        self,
        *,
        config: PredictionConfig,
        coinbase_client: CoinbaseClient,
        coinbase_broker: CoinbaseBroker | None = None,
        logger: logging.Logger,
        ai_service: Any | None = None,
    ) -> None:
        self.config = config
        self.coinbase_client = coinbase_client
        self.coinbase_broker = coinbase_broker
        self.logger = logger.getChild("strategy.coinbase_predictions")
        self.ai_service = ai_service

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
        fear_greed_value: int | None = None,
    ) -> dict[str, Any]:
        """Execute one Coinbase prediction-market strategy cycle.

        Args:
            state: Current portfolio state.
            broker: PaperBroker for equity/risk calculations and bookkeeping.
            risk_manager: Risk manager for position sizing.
            now: Current timestamp.
            fear_greed_value: Optional Fear & Greed index (0-100) for
                crypto-tagged market filtering.

        Returns:
            Strategy result dict with status, buys, sells, markets_scanned, etc.
        """
        if not self.config.enabled:
            self.logger.debug("Coinbase predictions strategy is disabled.")
            return {"status": "disabled", "buys": [], "sells": [], "markets_scanned": 0}

        result: dict[str, Any] = {
            "status": "ok",
            "buys": [],
            "sells": [],
            "markets_scanned": 0,
            "markets_discovered": 0,
            "resolved_closed": [],
            "fear_greed": fear_greed_value,
        }

        try:
            # Step 1: Handle resolved/expired positions
            self._close_resolved_positions(state, broker, result)

            # Step 2: Manage existing positions (stop-loss / take-profit)
            self._manage_existing_positions(state, broker, result)

            # Step 3: Discover available prediction markets
            markets = self._discover_markets()
            result["markets_discovered"] = len(markets)

            if not markets:
                self.logger.info(
                    "No Coinbase prediction markets available (API may not expose KX products)."
                )
                return result

            result["markets_scanned"] = len(markets)

            # Step 4: Scan for new entry signals
            if not state.kill_switch_engaged and not state.halt_new_trades_for_day:
                self._scan_for_entries(
                    state, markets, broker, risk_manager, now, fear_greed_value, result,
                )

        except Exception as exc:
            self.logger.error("Coinbase predictions strategy failed: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)

        self.logger.info(
            "Coinbase predictions cycle complete. scanned=%d discovered=%d buys=%s sells=%s "
            "resolved=%s fear_greed=%s",
            result["markets_scanned"],
            result["markets_discovered"],
            result["buys"],
            result["sells"],
            result["resolved_closed"],
            fear_greed_value,
        )
        return result

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    def _discover_markets(self) -> list[dict[str, Any]]:
        """Fetch prediction-market products from Coinbase and filter them.

        Returns only active markets whose product_id starts with ``KX``
        and whose category matches ``config.market_categories``.
        """
        try:
            raw_markets = self.coinbase_client.get_prediction_markets()
        except Exception as exc:
            self.logger.warning("Failed to fetch prediction markets: %s", exc)
            return []

        if not raw_markets:
            return []

        filtered: list[dict[str, Any]] = []
        for market in raw_markets:
            product_id: str = market.get("product_id", "")
            if not product_id.startswith("KX"):
                continue
            status = market.get("status", "").lower()
            if status not in ("online", "active", ""):
                self.logger.debug("Skipping %s with status=%s", product_id, status)
                continue

            # Category filtering (best-effort; field may not be present)
            category = self._infer_category(market)
            if self.config.market_categories and category not in self.config.market_categories:
                self.logger.debug(
                    "Skipping %s: category '%s' not in %s",
                    product_id, category, self.config.market_categories,
                )
                continue

            filtered.append(market)

        self.logger.info(
            "Discovered %d prediction markets (%d after filtering).",
            len(raw_markets), len(filtered),
        )
        return filtered

    @staticmethod
    def _infer_category(market: dict[str, Any]) -> str:
        """Best-effort category inference from product metadata."""
        product_id: str = market.get("product_id", "").upper()
        base: str = market.get("base_currency_id", "").upper()

        if any(tok in product_id or tok in base for tok in ("BTC", "ETH", "SOL", "CRYPTO")):
            return "crypto"
        if any(tok in product_id or tok in base for tok in ("CPI", "GDP", "FOMC", "NFP", "ECON")):
            return "economics"
        if any(tok in product_id or tok in base for tok in ("ELEC", "PRES", "GOV", "VOTE")):
            return "politics"

        return "other"

    # ------------------------------------------------------------------
    # Position management helpers
    # ------------------------------------------------------------------

    def _get_prediction_positions(self, state: PortfolioState) -> dict[str, Any]:
        """Return all positions with asset_type='prediction' and our strategy tag."""
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
            product_id = position.meta.get("product_id", "")
            if not product_id:
                continue

            try:
                # Re-fetch market data to check status
                markets = self.coinbase_client.get_prediction_markets()
                market_data = next(
                    (m for m in markets if m.get("product_id") == product_id), None,
                )

                if market_data is None:
                    # Product disappeared -- treat as expired/delisted
                    self.logger.warning(
                        "Product %s no longer found; closing position %s as expired.",
                        product_id, symbol,
                    )
                    close_price = 0.0
                elif market_data.get("status", "").lower() in ("resolved", "settled", "expired"):
                    # Determine settlement price
                    outcome = market_data.get("outcome", "").upper()
                    side = position.meta.get("side", "YES")
                    if outcome == side:
                        close_price = 1.0
                    elif outcome:
                        close_price = 0.0
                    else:
                        close_price = position.current_price
                    self.logger.info(
                        "Product %s resolved: outcome=%s side=%s close_price=%.2f",
                        product_id, outcome, side, close_price,
                    )
                else:
                    continue  # Still active

                # Execute close via CoinbaseBroker if available
                if self.coinbase_broker is not None:
                    side = position.meta.get("side", "YES")
                    self.coinbase_broker.place_prediction_sell(
                        product_id=product_id,
                        side=side,
                        size=int(position.quantity),
                    )

                broker.close_position(
                    state,
                    symbol=symbol,
                    price=close_price,
                    reason=f"resolved:{market_data.get('outcome', 'expired') if market_data else 'delisted'}",
                )
                result["resolved_closed"].append(symbol)

            except Exception as exc:
                self.logger.warning(
                    "Failed to check resolution for %s (%s): %s", symbol, product_id, exc,
                )

    def _manage_existing_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        result: dict[str, Any],
    ) -> None:
        """Apply take-profit and stop-loss rules to existing prediction positions."""
        positions = self._get_prediction_positions(state)
        if not positions:
            return

        for symbol, position in list(positions.items()):
            product_id = position.meta.get("product_id", "")
            if not product_id:
                continue

            try:
                current_price = self._get_current_price(product_id)
                if current_price is None or current_price <= 0:
                    self.logger.debug("No price available for %s; skipping management.", product_id)
                    continue

                # Update tracked price
                position.current_price = current_price
                state.last_prices[symbol] = current_price

                side = position.meta.get("side", "YES")
                entry_price = position.entry_price

                # Take profit
                if self._should_take_profit(side, entry_price, current_price):
                    self._close_prediction_position(
                        state, broker, symbol, position, current_price, "take_profit", result,
                    )
                    self.logger.info(
                        "Take profit %s: side=%s entry=%.4f current=%.4f",
                        symbol, side, entry_price, current_price,
                    )
                    continue

                # Stop loss
                if self._should_stop_loss(side, entry_price, current_price):
                    self._close_prediction_position(
                        state, broker, symbol, position, current_price, "stop_loss", result,
                    )
                    self.logger.info(
                        "Stop loss %s: side=%s entry=%.4f current=%.4f",
                        symbol, side, entry_price, current_price,
                    )

            except Exception as exc:
                self.logger.warning("Failed to manage position %s: %s", symbol, exc)

    def _close_prediction_position(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        symbol: str,
        position: Any,
        price: float,
        reason: str,
        result: dict[str, Any],
    ) -> None:
        """Close a prediction position through both live and paper brokers."""
        product_id = position.meta.get("product_id", "")
        side = position.meta.get("side", "YES")

        if self.coinbase_broker is not None and product_id:
            self.coinbase_broker.place_prediction_sell(
                product_id=product_id,
                side=side,
                size=int(position.quantity),
            )

        closed = broker.close_position(state, symbol=symbol, price=price, reason=reason)
        if closed is not None:
            result["sells"].append(symbol)

    def _should_take_profit(self, side: str, entry: float, current: float) -> bool:
        """Check whether the position should be closed for profit.

        For YES positions bought cheaply: take profit when price rises above
        the take_profit_threshold.
        For NO positions (entry is the NO price): take profit when the
        equivalent YES price drops below the take_profit_threshold (i.e.
        the NO price rises above it).
        """
        threshold = self.config.take_profit_threshold
        if side == "YES":
            return entry < threshold and current >= threshold
        else:
            # NO side: entry is the NO price (1 - YES price at open).
            # Current is the YES price.  NO value = 1 - current.
            no_current = 1.0 - current
            return entry < threshold and no_current >= threshold

    def _should_stop_loss(self, side: str, entry: float, current: float) -> bool:
        """Check whether the position should be stopped out.

        A move of ``stop_loss_pct`` against entry triggers the stop.
        """
        max_loss = self.config.stop_loss_pct
        if side == "YES":
            # YES loses value when current drops below entry
            return (entry - current) >= max_loss
        else:
            # NO loses value when YES rises (NO value = 1 - current)
            no_current = 1.0 - current
            return (entry - no_current) >= max_loss

    # ------------------------------------------------------------------
    # Entry scanning
    # ------------------------------------------------------------------

    def _scan_for_entries(
        self,
        state: PortfolioState,
        markets: list[dict[str, Any]],
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
        fear_greed_value: int | None,
        result: dict[str, Any],
    ) -> None:
        """Scan discovered markets for new entry signals."""
        existing_count = len(self._get_prediction_positions(state))
        if existing_count >= self.config.max_positions:
            self.logger.info(
                "At max prediction positions (%d/%d). Skipping new entries.",
                existing_count, self.config.max_positions,
            )
            return

        slots_available = self.config.max_positions - existing_count
        equity = broker.portfolio_equity(state)

        for market in markets:
            if slots_available <= 0:
                break

            product_id: str = market.get("product_id", "")
            if not product_id:
                continue

            # Skip if already holding this product (either side)
            yes_symbol = f"KX:{product_id}:YES"
            no_symbol = f"KX:{product_id}:NO"
            if yes_symbol in state.positions or no_symbol in state.positions:
                continue

            # Get current YES price
            yes_price = self._extract_price(market)
            if yes_price is None or yes_price <= 0 or yes_price >= 1.0:
                self.logger.debug("Skipping %s: invalid price %.4f", product_id, yes_price or 0)
                continue

            # Evaluate signal
            signal = self._evaluate_signal(market, yes_price, fear_greed_value)
            if signal is None:
                continue

            side, price = signal

            # Position sizing
            notional = equity * risk_manager.config.max_position_fraction
            notional = min(notional, state.cash * 0.95)  # 5% cash buffer
            notional = max(notional, 0.0)

            if notional < self.config.min_trade_usd:
                self.logger.info(
                    "Insufficient funds for prediction entry on %s (notional=$%.2f, min=$%.2f).",
                    product_id, notional, self.config.min_trade_usd,
                )
                break

            # Determine quantity (prediction contracts are integer-quantity)
            quantity = int(notional / price) if price > 0 else 0
            if quantity <= 0:
                self.logger.info("Position size rounds to zero for %s at price %.4f.", product_id, price)
                continue

            actual_notional = quantity * price
            symbol = f"KX:{product_id}:{side}"

            # Execute on Coinbase (live)
            if self.coinbase_broker is not None:
                order = self.coinbase_broker.place_prediction_buy(
                    product_id=product_id,
                    side=side,
                    price=price,
                    size=quantity,
                )
                if order is None:
                    self.logger.warning(
                        "Live prediction buy failed for %s side=%s; skipping.", product_id, side,
                    )
                    continue

            # Bookkeep in PaperBroker
            position = broker.open_position(
                state,
                symbol=symbol,
                asset_type="prediction",
                strategy="coinbase_predictions",
                price=price,
                notional=actual_notional,
                opened_at=now,
                meta={
                    "product_id": product_id,
                    "side": side,
                    "entry_price": price,
                    "category": self._infer_category(market),
                    "fear_greed": fear_greed_value,
                },
            )

            if position is not None:
                result["buys"].append(symbol)
                slots_available -= 1
                self.logger.info(
                    "Prediction entry: %s side=%s price=%.4f qty=%d notional=$%.2f",
                    symbol, side, price, quantity, actual_notional,
                )

    def _evaluate_signal(
        self,
        market: dict[str, Any],
        yes_price: float,
        fear_greed_value: int | None,
    ) -> tuple[str, float] | None:
        """Evaluate whether a market has a buy signal.

        Returns:
            ``(side, price)`` tuple if signal found, ``None`` otherwise.
        """
        no_price = max(0.0, 1.0 - yes_price)
        category = self._infer_category(market)

        # Crypto-tagged markets: require Fear & Greed confirmation
        if category == "crypto" and fear_greed_value is not None:
            if yes_price < self.config.buy_yes_threshold:
                # Cheap YES -- contrarian buy when fear is high
                if fear_greed_value <= 30:
                    self.logger.debug(
                        "Crypto prediction signal: BUY YES %s (price=%.4f, F&G=%d)",
                        market.get("product_id"), yes_price, fear_greed_value,
                    )
                    return ("YES", yes_price)
            elif yes_price > self.config.buy_no_threshold:
                # Expensive YES -- contrarian NO when greed is high
                if fear_greed_value >= 70:
                    self.logger.debug(
                        "Crypto prediction signal: BUY NO %s (price=%.4f, F&G=%d)",
                        market.get("product_id"), yes_price, fear_greed_value,
                    )
                    return ("NO", no_price)
            return None

        # Non-crypto / no F&G data: pure threshold signals
        if yes_price < self.config.buy_yes_threshold:
            return ("YES", yes_price)
        elif yes_price > self.config.buy_no_threshold:
            return ("NO", no_price)

        return None

    # ------------------------------------------------------------------
    # Pricing helpers
    # ------------------------------------------------------------------

    def _extract_price(self, market: dict[str, Any]) -> float | None:
        """Extract the YES price from a market product dict.

        Tries ``price``, ``yes_price``, and ``mid_market_price`` keys.
        """
        for key in ("price", "yes_price", "mid_market_price"):
            raw = market.get(key)
            if raw is not None:
                try:
                    return float(raw)
                except (ValueError, TypeError):
                    continue
        return None

    def _get_current_price(self, product_id: str) -> float | None:
        """Fetch the current YES price for a single product.

        Falls back to scanning the full market list if no single-product
        endpoint is available.
        """
        try:
            markets = self.coinbase_client.get_prediction_markets()
            for market in markets:
                if market.get("product_id") == product_id:
                    return self._extract_price(market)
        except Exception as exc:
            self.logger.warning("Failed to get price for %s: %s", product_id, exc)
        return None
