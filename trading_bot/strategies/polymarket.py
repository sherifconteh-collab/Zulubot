"""Polymarket prediction market strategy.

Trades binary prediction markets based on:
1. Price thresholds (buy cheap YES, buy expensive NO)
2. Cross-module sentiment signals (Fear & Greed for crypto-tagged markets)
3. Stop-loss and take-profit exits
4. Resolved market cleanup
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trading_bot.clients_polymarket import PolymarketClobClient, PolymarketGammaClient
from trading_bot.config import PolymarketStrategyConfig
from trading_bot.models import PolymarketMarket, PortfolioState
from trading_bot.polymarket_broker import PolymarketBroker
from trading_bot.risk import RiskManager, markov_regime_blocks_entry

try:
    from trading_bot.integrations.trading_agents import TradingAgentsService
except ImportError:
    TradingAgentsService = None  # type: ignore[assignment,misc]


class PolymarketStrategy:
    """Prediction market trading strategy.

    Signal logic:
      BUY YES when price < buy_yes_threshold (default 0.30)
      BUY NO  when price > buy_no_threshold  (default 0.70)
      SELL    when price crosses take_profit_threshold (0.50) — take profit
      SELL    when price moves against by stop_loss (0.15) — cut losses

    Sentiment cross-reference:
      For crypto-tagged markets, requires Fear & Greed alignment.
    """

    def __init__(
        self,
        *,
        config: PolymarketStrategyConfig,
        gamma_client: PolymarketGammaClient,
        clob_client: PolymarketClobClient,
        polymarket_broker: PolymarketBroker,
        logger: logging.Logger,
        ai_service: Any | None = None,
    ) -> None:
        self.config = config
        self.gamma = gamma_client
        self.clob = clob_client
        self.broker = polymarket_broker
        self.logger = logger.getChild("strategy.polymarket")
        self.ai_service = ai_service

    def run(
        self,
        state: PortfolioState,
        *,
        broker: Any,  # PaperBroker — for equity calculations
        risk_manager: RiskManager,
        now: datetime,
        fear_greed_value: int | None = None,
    ) -> dict[str, Any]:
        """Execute one Polymarket strategy cycle.

        Args:
            state: Current portfolio state.
            broker: PaperBroker for equity/risk calculations.
            risk_manager: Risk manager for position sizing.
            now: Current timestamp.
            fear_greed_value: Optional Fear & Greed index for crypto-tagged market filtering.

        Returns:
            Strategy result dict with status, buys, sells, markets_scanned, etc.
        """
        if not self.config.enabled:
            return {"status": "disabled", "buys": [], "sells": [], "markets_scanned": 0}

        result: dict[str, Any] = {
            "status": "ok",
            "buys": [],
            "sells": [],
            "markets_scanned": 0,
            "markets_discovered": 0,
            "resolved_closed": [],
            "fear_greed": fear_greed_value,
            "decisions": [],
        }

        try:
            # Step 1: Close resolved markets
            self._close_resolved_markets(state, result)

            # Step 2: Manage existing positions (stop-loss, take-profit)
            self._manage_existing_positions(state, result)

            # Step 3: Discover new markets
            markets = self.gamma.discover_markets(
                tags=self.config.market_tags,
                min_volume=self.config.min_volume,
                min_liquidity=self.config.min_liquidity,
                limit=30,
            )
            result["markets_discovered"] = len(markets)

            # Step 4: Refresh prices from CLOB
            markets = self._refresh_prices(markets)
            result["markets_scanned"] = len(markets)

            # Step 5: Generate buy signals
            if state.kill_switch_engaged or state.halt_new_trades_for_day:
                result["decisions"].append({
                    "strategy": "polymarket_threshold",
                    "symbol": "*",
                    "action": "skip_buy",
                    "reason": "risk_guardrails",
                    "signals": {
                        "kill_switch": state.kill_switch_engaged,
                        "daily_halt": state.halt_new_trades_for_day,
                    },
                })
            elif markov_block := markov_regime_blocks_entry(
                state, self.config.use_markov_regime_gate, "polymarket_threshold"
            ):
                result["decisions"].append(markov_block)
            else:
                self._scan_for_entries(
                    state, markets, broker, risk_manager, now, fear_greed_value, result,
                )

            # Update scan date
            state.last_polymarket_scan_date = now.date().isoformat()

        except Exception as exc:
            self.logger.error("Polymarket strategy failed: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)

        return result

    def _close_resolved_markets(self, state: PortfolioState, result: dict[str, Any]) -> None:
        """Check if any held positions have resolved and close them."""
        prediction_positions = self.broker.get_prediction_positions(state)
        if not prediction_positions:
            return

        for symbol, position in list(prediction_positions.items()):
            condition_id = position.meta.get("condition_id", "")
            if not condition_id:
                continue

            try:
                market = self.gamma.get_market(condition_id)
                if market is None:
                    continue

                if market.resolved:
                    side = position.meta.get("side", "YES")
                    # If resolved in our favor: price = 1.0, otherwise 0.0
                    if market.outcome and market.outcome.upper() == side:
                        close_price = 1.0
                    elif market.outcome:
                        close_price = 0.0
                    else:
                        close_price = position.current_price

                    closed = self.broker.place_limit_sell(
                        state,
                        symbol=symbol,
                        price=close_price,
                        reason=f"resolved:{market.outcome or 'unknown'}",
                    )
                    if closed:
                        result["resolved_closed"].append(symbol)
                        self.logger.info(
                            "Closed resolved position %s outcome=%s price=%.2f",
                            symbol, market.outcome, close_price,
                        )
            except Exception as exc:
                self.logger.warning("Failed to check resolution for %s: %s", symbol, exc)

    def _manage_existing_positions(self, state: PortfolioState, result: dict[str, Any]) -> None:
        """Apply stop-loss and take-profit to existing prediction positions."""
        prediction_positions = self.broker.get_prediction_positions(state)
        if not prediction_positions:
            return

        for symbol, position in list(prediction_positions.items()):
            token_id = position.meta.get("token_id", "")
            if not token_id:
                continue

            try:
                current_price = self.clob.get_midpoint(token_id)
                if current_price <= 0:
                    continue

                # Update position price
                position.current_price = current_price
                state.last_prices[symbol] = current_price

                side = position.meta.get("side", "YES")

                # Take profit: price crossed to profitable side
                if self._should_take_profit(side, position.entry_price, current_price):
                    closed = self.broker.place_limit_sell(
                        state, symbol=symbol, price=current_price, reason="take_profit",
                    )
                    if closed:
                        result["sells"].append(symbol)
                        result["decisions"].append({
                            "strategy": "polymarket_threshold",
                            "symbol": symbol,
                            "action": "sell",
                            "reason": "take_profit",
                            "signals": {
                                "side": side,
                                "entry_price": round(position.entry_price, 4),
                                "current_price": round(current_price, 4),
                            },
                        })
                        self.logger.info(
                            "Take profit %s: entry=%.4f current=%.4f",
                            symbol, position.entry_price, current_price,
                        )
                    continue

                # Stop loss: price moved against us
                if self._should_stop_loss(side, position.entry_price, current_price):
                    closed = self.broker.place_limit_sell(
                        state, symbol=symbol, price=current_price, reason="stop_loss",
                    )
                    if closed:
                        result["sells"].append(symbol)
                        result["decisions"].append({
                            "strategy": "polymarket_threshold",
                            "symbol": symbol,
                            "action": "sell",
                            "reason": "stop_loss",
                            "signals": {
                                "side": side,
                                "entry_price": round(position.entry_price, 4),
                                "current_price": round(current_price, 4),
                            },
                        })
                        self.logger.info(
                            "Stop loss %s: entry=%.4f current=%.4f",
                            symbol, position.entry_price, current_price,
                        )

            except Exception as exc:
                self.logger.warning("Failed to manage position %s: %s", symbol, exc)

    def _should_take_profit(self, side: str, entry: float, current: float) -> bool:
        """Check if position should be closed for profit."""
        threshold = self.config.take_profit_threshold
        if side == "YES":
            # Bought YES cheap, take profit when price rises above threshold
            return entry < threshold and current >= threshold
        else:
            # Bought NO (because YES was expensive), take profit when YES drops below threshold
            return entry < threshold and current >= threshold

    def _should_stop_loss(self, side: str, entry: float, current: float) -> bool:
        """Check if position should be closed for loss."""
        max_loss = self.config.stop_loss_threshold
        if side == "YES":
            # Bought YES: lose when price drops
            return (entry - current) >= max_loss
        else:
            # Bought NO: lose when NO price drops (YES price rises)
            return (entry - current) >= max_loss

    def _refresh_prices(self, markets: list[PolymarketMarket]) -> list[PolymarketMarket]:
        """Refresh market prices from CLOB API for more accurate midpoints."""
        refreshed: list[PolymarketMarket] = []
        for market in markets:
            if not market.yes_token_id:
                refreshed.append(market)
                continue

            try:
                yes_mid = self.clob.get_midpoint(market.yes_token_id)
                if yes_mid > 0:
                    # Create updated market with fresh prices
                    # Since PolymarketMarket is frozen, we create a new instance
                    refreshed.append(PolymarketMarket(
                        condition_id=market.condition_id,
                        question=market.question,
                        description=market.description,
                        yes_token_id=market.yes_token_id,
                        no_token_id=market.no_token_id,
                        yes_price=yes_mid,
                        no_price=max(0, 1.0 - yes_mid),
                        volume_24h=market.volume_24h,
                        liquidity=market.liquidity,
                        tags=market.tags,
                        end_date=market.end_date,
                        active=market.active,
                        resolved=market.resolved,
                        outcome=market.outcome,
                    ))
                else:
                    refreshed.append(market)
            except Exception:
                refreshed.append(market)

        return refreshed

    def _scan_for_entries(
        self,
        state: PortfolioState,
        markets: list[PolymarketMarket],
        broker: Any,
        risk_manager: RiskManager,
        now: datetime,
        fear_greed_value: int | None,
        result: dict[str, Any],
    ) -> None:
        """Scan markets for buy signals."""
        # Count existing prediction positions
        existing_count = len(self.broker.get_prediction_positions(state))
        if existing_count >= self.config.max_positions:
            self.logger.info(
                "At max prediction positions (%d/%d). Skipping new entries.",
                existing_count, self.config.max_positions,
            )
            result["decisions"].append({
                "strategy": "polymarket_threshold",
                "symbol": "*",
                "action": "skip_buy",
                "reason": "max_positions_reached",
                "signals": {
                    "open_positions": existing_count,
                    "max_positions": self.config.max_positions,
                },
            })
            return

        equity = broker.portfolio_equity(state)
        slots_available = self.config.max_positions - existing_count

        for market in markets:
            if slots_available <= 0:
                break

            if market.resolved or not market.active:
                continue

            # Skip if we already hold this market (either side)
            yes_symbol = f"PM:{market.condition_id[:8]}:YES"
            no_symbol = f"PM:{market.condition_id[:8]}:NO"
            if yes_symbol in state.positions or no_symbol in state.positions:
                continue

            # Try AI signal first (if enabled), then fall back to threshold signal
            signal = self._get_ai_signal(market, fear_greed_value)
            if signal is None:
                signal = self._evaluate_signal(market, fear_greed_value)
            if signal is None:
                continue

            side, price = signal

            # Position sizing via risk manager (vol-adjusted when enabled)
            pm_symbol = f"PM:{market.condition_id[:8]}:{side}"
            notional = risk_manager.position_notional(state, equity, symbol=pm_symbol)
            notional = min(notional, state.cash * 0.95)  # Leave 5% cash buffer

            if notional < 1.0:
                self.logger.info("Insufficient funds for Polymarket entry (notional=$%.2f).", notional)
                result["decisions"].append({
                    "strategy": "polymarket_threshold",
                    "symbol": pm_symbol,
                    "action": "skip_buy",
                    "reason": "insufficient_funds",
                    "signals": {"notional": round(notional, 2), "side": side, "price": round(price, 4)},
                })
                break

            position = self.broker.place_limit_buy(
                state,
                market=market,
                side=side,
                price=price,
                notional_usd=notional,
                strategy="polymarket_threshold",
                now=now,
                meta={
                    "signal": f"threshold_{side.lower()}",
                    "fear_greed": fear_greed_value,
                    "volume_24h": market.volume_24h,
                    "liquidity": market.liquidity,
                },
            )

            if position:
                symbol = f"PM:{market.condition_id[:8]}:{side}"
                result["buys"].append(symbol)
                result["decisions"].append({
                    "strategy": "polymarket_threshold",
                    "symbol": symbol,
                    "action": "buy",
                    "reason": f"threshold_{side.lower()}",
                    "signals": {
                        "side": side,
                        "price": round(price, 4),
                        "notional": round(notional, 2),
                        "fear_greed": fear_greed_value,
                        "question": market.question[:60],
                    },
                })
                slots_available -= 1
                self.logger.info(
                    "Entry signal: %s side=%s price=%.4f question='%s'",
                    symbol, side, price, market.question[:60],
                )

    def _get_ai_signal(
        self,
        market: PolymarketMarket,
        fear_greed_value: int | None,
    ) -> tuple[str, float] | None:
        """Use AI to estimate fair probability and detect edge.

        Returns a signal only if AI estimates significant mispricing.
        """
        if self.ai_service is None or not self.config.use_trading_agents:
            return None

        verdict = self.ai_service.evaluate_polymarket(
            market_question=market.question,
            market_description=market.description,
            yes_price=market.yes_price,
            tags=market.tags,
            fear_greed_value=fear_greed_value,
        )
        if verdict is None or verdict.estimated_probability is None:
            return None

        fair_prob = verdict.estimated_probability
        market_prob = market.yes_price
        edge = fair_prob - market_prob

        self.logger.info(
            "AI probability estimate for '%s': fair=%.3f market=%.3f edge=%.3f confidence=%.2f",
            market.question[:60], fair_prob, market_prob, edge, verdict.confidence,
        )

        # Require minimum 10% edge and 0.5+ confidence
        min_edge = 0.10
        if verdict.confidence < 0.5:
            return None

        if edge >= min_edge:
            # Market underprices YES → buy YES
            return ("YES", market.yes_price)
        elif edge <= -min_edge:
            # Market overprices YES → buy NO
            return ("NO", market.no_price)

        return None

    def _evaluate_signal(
        self,
        market: PolymarketMarket,
        fear_greed_value: int | None,
    ) -> tuple[str, float] | None:
        """Evaluate whether a market has a buy signal.

        Returns:
            (side, price) tuple if signal found, None otherwise.
            side is "YES" or "NO".
        """
        yes_price = market.yes_price
        no_price = market.no_price

        # Validate prices
        if yes_price <= 0 or yes_price >= 1.0:
            return None

        # Check if crypto-tagged market needs F&G confirmation
        is_crypto_market = any(tag in ("crypto", "bitcoin", "ethereum") for tag in market.tags)
        if is_crypto_market and fear_greed_value is not None:
            # For crypto markets: only buy YES (bullish) when fear is high (contrarian)
            # Only buy NO (bearish) when greed is high (contrarian)
            if yes_price < self.config.buy_yes_threshold:
                # Cheap YES = market says unlikely. Buy if we're contrarian + fear is high
                if fear_greed_value <= 30:  # Fear supports contrarian YES buy
                    return ("YES", yes_price)
            elif yes_price > self.config.buy_no_threshold:
                # Expensive YES = market says likely. Buy NO if greed is high (contrarian)
                if fear_greed_value >= 70:  # Greed supports contrarian NO buy
                    return ("NO", no_price)
            return None

        # Non-crypto markets: pure threshold signals
        if yes_price < self.config.buy_yes_threshold:
            return ("YES", yes_price)
        elif yes_price > self.config.buy_no_threshold:
            return ("NO", no_price)

        return None
