from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients import AlternativeMeClient, BinanceFuturesClient, CoinbaseClient
from trading_bot.coinbase_broker import CoinbaseBroker
from trading_bot.config import CryptoStrategyConfig, LearningConfig
from trading_bot.models import PortfolioState
from trading_bot import performance_memory
from trading_bot.risk import RiskManager

try:
    from trading_bot.integrations.trading_agents import (
        TradingAgentsService,
        apply_ai_confirmation,
    )
except ImportError:
    TradingAgentsService = None  # type: ignore[assignment,misc]
    apply_ai_confirmation = None  # type: ignore[assignment]


class CryptoSentimentFundingStrategy:
    """Crypto strategy using Fear & Greed + Binance funding rates as signals.

    Pricing can come from either Coinbase (preferred for execution) or
    Binance mark price (fallback). Set ``config.use_coinbase_pricing`` to
    control which source is used.
    """

    def __init__(
        self,
        *,
        config: CryptoStrategyConfig,
        alternative_client: AlternativeMeClient,
        binance_client: BinanceFuturesClient,
        coinbase_client: CoinbaseClient | None = None,
        coinbase_broker: CoinbaseBroker | None = None,
        logger: logging.Logger,
        ai_service: Any | None = None,
        confirmation_mode: str = "log_only",
        learning_config: LearningConfig | None = None,
    ) -> None:
        self.config = config
        self.alternative_client = alternative_client
        self.binance_client = binance_client
        self.coinbase_client = coinbase_client
        self.coinbase_broker = coinbase_broker
        self.logger = logger.getChild("crypto_strategy")
        self.ai_service = ai_service
        self.confirmation_mode = confirmation_mode
        # Default to a disabled learning layer when not supplied, so direct
        # instantiation in tests doesn't require the full AppConfig.
        self.learning_config = learning_config or LearningConfig(enabled=False)

    def _get_execution_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get prices from Coinbase when available, fall back to Binance mark price."""
        if self.config.use_coinbase_pricing and self.coinbase_client is not None:
            prices = self.coinbase_client.get_spot_prices(symbols)
            if prices:
                self.logger.info("Using Coinbase spot prices for %d/%d symbols", len(prices), len(symbols))
                return prices
            self.logger.warning("Coinbase returned no prices, falling back to Binance mark prices")
        return {}

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
    ) -> dict[str, Any]:
        fear_greed = self.alternative_client.get_fear_and_greed()

        # Always fetch Binance funding rates (signal source)
        snapshots = {
            symbol: self.binance_client.get_mark_price_and_funding(symbol)
            for symbol in self.config.symbols
        }

        # Get execution prices (Coinbase preferred, Binance fallback)
        coinbase_prices = self._get_execution_prices(self.config.symbols)
        execution_prices: dict[str, float] = {}
        price_source: dict[str, str] = {}
        for symbol in self.config.symbols:
            if symbol in coinbase_prices:
                execution_prices[symbol] = coinbase_prices[symbol]
                price_source[symbol] = "coinbase"
            else:
                execution_prices[symbol] = snapshots[symbol].mark_price
                price_source[symbol] = "binance"

        broker.mark_positions(state, execution_prices)

        buys: list[str] = []
        sells: list[str] = []
        decisions: list[dict[str, Any]] = []

        # --- Autonomous bi-directional exits (run every cycle, independent
        # of the F&G macro gate). This lets the bot sell as it sees fit:
        # take profits at +take_profit_pct, cut losses at −stop_loss_pct.
        self._handle_autonomous_exits(
            state,
            broker=broker,
            execution_prices=execution_prices,
            sells=sells,
            decisions=decisions,
        )

        if fear_greed.value > self.config.sell_fear_threshold:
            for symbol in self.config.symbols:
                if symbol not in state.positions:
                    continue
                position = state.positions[symbol]
                # Place sell on Coinbase (paper or live)
                if self.coinbase_broker is not None:
                    self.coinbase_broker.place_market_sell(
                        state,
                        binance_symbol=symbol,
                        quantity=position.quantity,
                        strategy="crypto_sentiment_funding",
                        reason="fear_and_greed_sell_threshold",
                    )
                broker.close_position(
                    state,
                    symbol=symbol,
                    price=execution_prices.get(symbol),
                    reason="fear_and_greed_sell_threshold",
                )
                sells.append(symbol)
                decisions.append({
                    "strategy": "crypto_sentiment_funding",
                    "symbol": symbol,
                    "action": "sell",
                    "reason": "fear_and_greed_sell_threshold",
                    "signals": {
                        "fear_greed": fear_greed.value,
                        "classification": fear_greed.classification,
                        "threshold": self.config.sell_fear_threshold,
                    },
                })

        elif fear_greed.value < self.config.buy_fear_threshold:
            if not risk_manager.can_open_new_position(state):
                self.logger.info(
                    "Crypto buys blocked by risk guardrails. daily_halt=%s kill_switch=%s",
                    state.halt_new_trades_for_day,
                    state.kill_switch_engaged,
                )
                decisions.append({
                    "strategy": "crypto_sentiment_funding",
                    "symbol": "*",
                    "action": "skip_buy",
                    "reason": "risk_guardrails",
                    "signals": {
                        "fear_greed": fear_greed.value,
                        "classification": fear_greed.classification,
                        "daily_halt": state.halt_new_trades_for_day,
                        "kill_switch": state.kill_switch_engaged,
                    },
                })
            else:
                for symbol in self.config.symbols:
                    if symbol in state.positions:
                        continue
                    # "Learn as it goes": if the bot has been losing on this
                    # symbol repeatedly, bench it for a few cycles. A winning
                    # close resets the counter.
                    if performance_memory.is_symbol_benched(
                        state, symbol, config=self.learning_config, decrement=True,
                    ):
                        mem = performance_memory.summary(state, symbol)
                        self.logger.info(
                            "Skipping %s: benched by performance memory "
                            "(consecutive_losses=%s, cooldown_remaining=%s)",
                            symbol,
                            mem.get("consecutive_losses"),
                            mem.get("cooldown_remaining"),
                        )
                        decisions.append({
                            "strategy": "crypto_sentiment_funding",
                            "symbol": symbol,
                            "action": "skip_buy",
                            "reason": "performance_memory_benched",
                            "signals": {
                                "fear_greed": fear_greed.value,
                                "consecutive_losses": mem.get("consecutive_losses"),
                                "cooldown_remaining": mem.get("cooldown_remaining"),
                            },
                        })
                        continue
                    snapshot = snapshots[symbol]
                    # Funding-rate filter is opt-in (default OFF = more aggressive).
                    # Previously this was a hard gate, which in combination with the
                    # Fear & Greed threshold caused the strategy to almost never trade.
                    if self.config.require_negative_funding and snapshot.funding_rate >= 0:
                        self.logger.info(
                            "Skipping %s: funding_rate=%.6f (non-negative)", symbol, snapshot.funding_rate
                        )
                        decisions.append({
                            "strategy": "crypto_sentiment_funding",
                            "symbol": symbol,
                            "action": "skip_buy",
                            "reason": "non_negative_funding_rate",
                            "signals": {
                                "fear_greed": fear_greed.value,
                                "classification": fear_greed.classification,
                                "funding_rate": round(snapshot.funding_rate, 6),
                            },
                        })
                        continue
                    equity = broker.portfolio_equity(state)
                    notional = risk_manager.position_notional(state, equity, symbol=symbol)
                    if notional <= 0:
                        self.logger.warning("Skipping crypto buy for %s because position size is zero.", symbol)
                        decisions.append({
                            "strategy": "crypto_sentiment_funding",
                            "symbol": symbol,
                            "action": "skip_buy",
                            "reason": "position_size_zero",
                            "signals": {
                                "fear_greed": fear_greed.value,
                                "funding_rate": round(snapshot.funding_rate, 6),
                            },
                        })
                        continue

                    # AI confirmation (optional)
                    ai_meta: dict[str, Any] = {}
                    if self.ai_service is not None and apply_ai_confirmation is not None:
                        verdict = self.ai_service.evaluate_crypto(
                            symbol=symbol,
                            fear_greed_value=fear_greed.value,
                            funding_rate=snapshot.funding_rate,
                            price=execution_prices[symbol],
                        )
                        notional, ai_meta = apply_ai_confirmation(
                            verdict,
                            self.confirmation_mode,
                            notional,
                            self.logger,
                            config=getattr(self.ai_service, "config", None),
                        )
                        if notional <= 0:
                            self.logger.info("AI rejected crypto buy for %s.", symbol)
                            decisions.append({
                                "strategy": "crypto_sentiment_funding",
                                "symbol": symbol,
                                "action": "skip_buy",
                                "reason": "ai_rejected",
                                "signals": {
                                    "fear_greed": fear_greed.value,
                                    "funding_rate": round(snapshot.funding_rate, 6),
                                    "ai_verdict": ai_meta.get("ai_verdict", "unknown"),
                                },
                            })
                            continue

                    price = execution_prices[symbol]
                    # Place buy on Coinbase (paper or live)
                    if self.coinbase_broker is not None:
                        self.coinbase_broker.place_market_buy(
                            state,
                            binance_symbol=symbol,
                            notional_usd=notional,
                            strategy="crypto_sentiment_funding",
                            now=now,
                            meta={
                                "fear_and_greed": fear_greed.value,
                                "funding_rate": snapshot.funding_rate,
                            },
                        )
                    position = broker.open_position(
                        state,
                        symbol=symbol,
                        asset_type="crypto",
                        strategy="crypto_sentiment_funding",
                        price=price,
                        notional=notional,
                        opened_at=now,
                        meta={
                            "fear_and_greed": fear_greed.value,
                            "funding_rate": snapshot.funding_rate,
                            "classification": fear_greed.classification,
                            "price_source": price_source[symbol],
                            "coinbase_product": (
                                self.coinbase_client.to_coinbase_symbol(symbol)
                                if self.coinbase_client else "N/A"
                            ),
                            **ai_meta,
                        },
                    )
                    if position is not None:
                        buys.append(symbol)
                        decisions.append({
                            "strategy": "crypto_sentiment_funding",
                            "symbol": symbol,
                            "action": "buy",
                            "reason": "fear_and_greed_buy_threshold",
                            "signals": {
                                "fear_greed": fear_greed.value,
                                "classification": fear_greed.classification,
                                "funding_rate": round(snapshot.funding_rate, 6),
                                "price": round(price, 4),
                                "notional": round(notional, 2),
                            },
                        })

        else:
            decisions.append({
                "strategy": "crypto_sentiment_funding",
                "symbol": "*",
                "action": "hold",
                "reason": "neutral_fear_greed",
                "signals": {
                    "fear_greed": fear_greed.value,
                    "classification": fear_greed.classification,
                    "buy_threshold": self.config.buy_fear_threshold,
                    "sell_threshold": self.config.sell_fear_threshold,
                },
            })

        self.logger.info(
            "Crypto cycle complete. fear_and_greed=%d classification=%s buys=%s sells=%s "
            "funding=%s price_sources=%s",
            fear_greed.value,
            fear_greed.classification,
            buys,
            sells,
            {symbol: snapshot.funding_rate for symbol, snapshot in snapshots.items()},
            price_source,
        )
        return {
            "status": "completed",
            "fear_and_greed": fear_greed.value,
            "classification": fear_greed.classification,
            "funding_rates": {symbol: snapshot.funding_rate for symbol, snapshot in snapshots.items()},
            "execution_prices": execution_prices,
            "price_sources": price_source,
            "buys": buys,
            "sells": sells,
            "decisions": decisions,
        }

    def _handle_autonomous_exits(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        execution_prices: dict[str, float],
        sells: list[str],
        decisions: list[dict[str, Any]],
    ) -> None:
        """Close any open crypto positions that hit take-profit or stop-loss.

        Runs every cycle regardless of Fear & Greed — the point is to let the
        bot exit on its own terms. Both thresholds are optional; 0.0 disables
        the respective rule. Only inspects positions opened by this strategy.
        """
        tp = self.config.take_profit_pct
        sl = self.config.stop_loss_pct
        if tp <= 0 and sl <= 0:
            return

        for symbol in list(self.config.symbols):
            position = state.positions.get(symbol)
            if position is None or position.strategy != "crypto_sentiment_funding":
                continue
            price = execution_prices.get(symbol)
            if price is None or position.entry_price <= 0:
                continue
            move_pct = (price - position.entry_price) / position.entry_price

            hit_reason: str | None = None
            if tp > 0 and move_pct >= tp:
                hit_reason = "take_profit"
            elif sl > 0 and move_pct <= -sl:
                hit_reason = "stop_loss"
            if hit_reason is None:
                continue

            if self.coinbase_broker is not None:
                self.coinbase_broker.place_market_sell(
                    state,
                    binance_symbol=symbol,
                    quantity=position.quantity,
                    strategy="crypto_sentiment_funding",
                    reason=hit_reason,
                )
            broker.close_position(
                state,
                symbol=symbol,
                price=price,
                reason=hit_reason,
            )
            sells.append(symbol)
            decisions.append({
                "strategy": "crypto_sentiment_funding",
                "symbol": symbol,
                "action": "sell",
                "reason": hit_reason,
                "signals": {
                    "entry_price": round(position.entry_price, 4),
                    "exit_price": round(price, 4),
                    "move_pct": round(move_pct, 4),
                    "take_profit_pct": tp,
                    "stop_loss_pct": sl,
                },
            })
