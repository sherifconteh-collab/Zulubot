from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients import AlternativeMeClient, BinanceFuturesClient, CoinbaseClient
from trading_bot.coinbase_broker import CoinbaseBroker
from trading_bot.config import CryptoStrategyConfig
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
    ) -> None:
        self.config = config
        self.alternative_client = alternative_client
        self.binance_client = binance_client
        self.coinbase_client = coinbase_client
        self.coinbase_broker = coinbase_broker
        self.logger = logger.getChild("crypto_strategy")
        self.ai_service = ai_service
        self.confirmation_mode = confirmation_mode

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
