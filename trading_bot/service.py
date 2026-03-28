from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients import (
    AlternativeMeClient,
    BinanceFuturesClient,
    CoinbaseClient,
    FinancialModelingPrepClient,
)
from trading_bot.clients_polymarket import PolymarketClobClient, PolymarketGammaClient
from trading_bot.coinbase_broker import CoinbaseBroker
from trading_bot.config import AppConfig
from trading_bot.models import PortfolioState
from trading_bot.polymarket_broker import PolymarketBroker
from trading_bot.risk import RiskManager, RiskSnapshot
from trading_bot.state import StateStore
from trading_bot.strategies.altcoin_rotation import AltcoinRotationStrategy
from trading_bot.strategies.coinbase_predictions import CoinbasePredictionStrategy
from trading_bot.strategies.crypto import CryptoSentimentFundingStrategy
from trading_bot.strategies.polymarket import PolymarketStrategy
from trading_bot.strategies.stocks import StockValueStrategy

try:
    from trading_bot.integrations.trading_agents import TradingAgentsService
except ImportError:
    TradingAgentsService = None  # type: ignore[assignment,misc]


class TradingBotService:
    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger.getChild("service")
        self.state_store = StateStore(config.state_path, config.initial_cash, config.timezone)
        self.broker = PaperBroker(config.paper_mode, logger)
        self.risk_manager = RiskManager(config.risk)

        self.fmp_client = FinancialModelingPrepClient(
            api_key=config.fmp_api_key,
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
        )
        self.alternative_client = AlternativeMeClient(
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
        )
        self.binance_client = BinanceFuturesClient(
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
        )
        self.coinbase_client = CoinbaseClient(
            api_key=config.coinbase.api_key,
            api_secret=config.coinbase.api_secret,
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
        )

        # TradingAgents AI service (optional)
        self.ai_service = None
        if TradingAgentsService is not None and config.trading_agents.enabled:
            self.ai_service = TradingAgentsService(config.trading_agents, logger)
            if self.ai_service.is_available:
                self.logger.info("TradingAgents AI enabled. Mode: %s", config.trading_agents.confirmation_mode)
            else:
                self.logger.warning("TradingAgents enabled but not available (missing API key or SDK).")
                self.ai_service = None

        self.coinbase_broker = CoinbaseBroker(
            api_key=config.coinbase.api_key,
            api_secret=config.coinbase.api_secret,
            coinbase_client=self.coinbase_client,
            paper_mode=config.paper_mode,
            logger=logger,
        )

        self.stock_strategy = StockValueStrategy(
            config=config.stock,
            client=self.fmp_client,
            logger=logger,
            ai_service=self.ai_service,
            confirmation_mode=config.trading_agents.confirmation_mode,
            # Coinbase API is crypto-only — stock positions tracked internally
            coinbase_broker=None,
        )

        self.crypto_strategy = CryptoSentimentFundingStrategy(
            config=config.crypto,
            alternative_client=self.alternative_client,
            binance_client=self.binance_client,
            coinbase_client=self.coinbase_client,
            coinbase_broker=self.coinbase_broker,
            logger=logger,
            ai_service=self.ai_service,
            confirmation_mode=config.trading_agents.confirmation_mode,
        )

        # Polymarket clients and strategy (conditional on enabled)
        self.gamma_client = PolymarketGammaClient(
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
        )
        self.clob_client = PolymarketClobClient(
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
        )
        self.polymarket_broker = PolymarketBroker(
            paper_broker=self.broker,
            paper_mode=config.paper_mode,
            logger=logger,
        )
        self.polymarket_strategy = PolymarketStrategy(
            config=config.polymarket_strategy,
            gamma_client=self.gamma_client,
            clob_client=self.clob_client,
            polymarket_broker=self.polymarket_broker,
            logger=logger,
            ai_service=self.ai_service if config.polymarket_strategy.use_trading_agents else None,
        )

        # Altcoin rotation strategy
        self.rotation_strategy = AltcoinRotationStrategy(
            config=config.altcoin_rotation,
            coinbase_client=self.coinbase_client,
            coinbase_broker=self.coinbase_broker,
            logger=logger,
        )

        # Coinbase prediction markets strategy
        self.prediction_strategy = CoinbasePredictionStrategy(
            config=config.predictions,
            coinbase_client=self.coinbase_client,
            coinbase_broker=self.coinbase_broker,
            logger=logger,
        )

    def run(self, command: str, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(self.config.timezone)
        state = self.state_store.load()

        # Sync real Coinbase balance when in live mode
        if not self.config.paper_mode:
            self._sync_coinbase_balance(state)

        self.refresh_open_positions(state)
        pre_risk = self._evaluate_risk(state, now)
        module_results: dict[str, Any] = {}

        if command in {"stocks", "all"}:
            module_results["stocks"] = self._run_module(
                "stocks",
                lambda: self.stock_strategy.run(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                    force=force,
                ),
            )
        if command in {"crypto", "all"}:
            module_results["crypto"] = self._run_module(
                "crypto",
                lambda: self.crypto_strategy.run(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                ),
            )
        if command in {"rotation", "all"}:
            module_results["rotation"] = self._run_module(
                "rotation",
                lambda: self.rotation_strategy.run(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                ),
            )
        if command in {"predictions", "all"}:
            module_results["predictions"] = self._run_module(
                "predictions",
                lambda: self.prediction_strategy.run(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                ),
            )
        if command in {"polymarket", "all"}:
            # Get Fear & Greed for cross-module sentiment signal
            fg_value = None
            if command == "all" and "crypto" in module_results:
                fg_value = module_results["crypto"].get("fear_and_greed")
            elif command == "polymarket":
                try:
                    fg_reading = self.alternative_client.get_fear_and_greed()
                    fg_value = fg_reading.value
                except Exception:
                    pass
            module_results["polymarket"] = self._run_module(
                "polymarket",
                lambda: self.polymarket_strategy.run(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                    fear_greed_value=fg_value,
                ),
            )

        self.refresh_open_positions(state)
        post_risk = self._evaluate_risk(state, now)
        self.state_store.save(state)

        summary = self._build_summary(state, pre_risk=pre_risk, post_risk=post_risk, modules=module_results)
        self.logger.info("Run summary: %s", summary)
        return summary

    def status(self) -> dict[str, Any]:
        now = datetime.now(self.config.timezone)
        state = self.state_store.load()
        if not self.config.paper_mode:
            self._sync_coinbase_balance(state)
        self.refresh_open_positions(state)
        risk = self._evaluate_risk(state, now)
        self.state_store.save(state)
        return self._build_summary(state, pre_risk=risk, post_risk=risk, modules={})

    def refresh_open_positions(self, state: PortfolioState) -> None:
        stock_symbols = sorted(
            symbol for symbol, position in state.positions.items() if position.asset_type == "stock"
        )
        if stock_symbols:
            try:
                quotes = self.fmp_client.get_batch_quotes(stock_symbols)
                self.broker.mark_positions(state, quotes)
            except Exception as exc:
                self.logger.warning("Unable to refresh stock prices: %s", exc)

        crypto_symbols = sorted(
            symbol for symbol, position in state.positions.items() if position.asset_type == "crypto"
        )
        if crypto_symbols:
            try:
                # Prefer Coinbase spot prices, fall back to Binance mark prices
                prices: dict[str, float] = {}
                if self.config.crypto.use_coinbase_pricing:
                    prices = self.coinbase_client.get_spot_prices(crypto_symbols)
                # Fill any missing symbols from Binance
                for symbol in crypto_symbols:
                    if symbol not in prices:
                        prices[symbol] = self.binance_client.get_mark_price_and_funding(symbol).mark_price
                self.broker.mark_positions(state, prices)
            except Exception as exc:
                self.logger.warning("Unable to refresh crypto prices: %s", exc)

        # Refresh prediction market prices from CLOB
        prediction_symbols = sorted(
            symbol for symbol, position in state.positions.items() if position.asset_type == "prediction"
        )
        if prediction_symbols:
            try:
                prices: dict[str, float] = {}
                for symbol in prediction_symbols:
                    pos = state.positions[symbol]
                    token_id = pos.meta.get("token_id", "")
                    if token_id:
                        mid = self.clob_client.get_midpoint(token_id)
                        if mid > 0:
                            prices[symbol] = mid
                if prices:
                    self.broker.mark_positions(state, prices)
            except Exception as exc:
                self.logger.warning("Unable to refresh prediction prices: %s", exc)

    def _sync_coinbase_balance(self, state: PortfolioState) -> None:
        """Sync portfolio cash from real Coinbase USD balance."""
        try:
            total_usd, breakdown = self.coinbase_broker.get_total_balance()
            usd_cash = breakdown.get("USD", 0.0) + breakdown.get("USDC", 0.0)

            # Set cash to real Coinbase USD balance
            state.cash = usd_cash

            # Calculate real equity from Coinbase
            equity = usd_cash + sum(
                pos.quantity * pos.current_price for pos in state.positions.values()
            )
            # Reset peak/baseline if they look stale (from old paper-mode state
            # or different initial_cash) to prevent false kill-switch triggers
            if equity > 0 and (
                state.peak_equity <= 0
                or state.peak_equity == self.config.initial_cash
                or state.peak_equity > equity * 5  # peak is impossibly high vs reality
            ):
                self.logger.info(
                    "Resetting peak equity from $%.2f to real equity $%.2f",
                    state.peak_equity, equity,
                )
                state.peak_equity = equity
                state.daily_baseline_equity = equity

            self.logger.info(
                "Coinbase balance synced: USD=$%.2f total=$%.2f assets=%s",
                usd_cash, total_usd, {k: f"${v:.2f}" for k, v in breakdown.items()},
            )
        except Exception as exc:
            self.logger.warning("Failed to sync Coinbase balance: %s. Using cached state.", exc)
            # In live mode, never show fake money — if sync fails and cash
            # looks like the default initial_cash, zero it out.
            if state.cash >= self.config.initial_cash and self.config.initial_cash > 0:
                self.logger.warning("Zeroing cash to prevent phantom balance in live mode.")
                state.cash = 0.0

    def _evaluate_risk(self, state: PortfolioState, now: datetime) -> RiskSnapshot:
        equity = self.broker.portfolio_equity(state)
        snapshot = self.risk_manager.evaluate(state, equity, now)
        if snapshot.kill_switch_triggered_now:
            self.logger.error(
                "Kill switch triggered at equity=%.2f total_drawdown=%.2f%%. Closing all positions.",
                snapshot.equity,
                snapshot.total_drawdown * 100,
            )
            self.broker.close_all(state, reason="kill_switch")
        elif snapshot.daily_halt_active:
            self.logger.warning(
                "Daily halt active at equity=%.2f daily_drawdown=%.2f%%.",
                snapshot.equity,
                snapshot.daily_drawdown * 100,
            )
        return snapshot

    def _build_summary(
        self,
        state: PortfolioState,
        *,
        pre_risk: RiskSnapshot,
        post_risk: RiskSnapshot,
        modules: dict[str, Any],
    ) -> dict[str, Any]:
        equity = self.broker.portfolio_equity(state)
        positions = [
            {
                "symbol": position.symbol,
                "asset_type": position.asset_type,
                "strategy": position.strategy,
                "quantity": round(position.quantity, 6),
                "entry_price": round(position.entry_price, 6),
                "current_price": round(position.current_price, 6),
                "market_value": round(position.market_value, 2),
                "unrealized_pnl": round(position.unrealized_pnl, 2),
            }
            for position in state.positions.values()
        ]
        return {
            "timestamp": datetime.now(self.config.timezone).isoformat(),
            "paper_mode": self.config.paper_mode,
            "cash": round(state.cash, 2),
            "equity": round(equity, 2),
            "realized_pnl": round(state.realized_pnl, 2),
            "open_positions": positions,
            "risk": {
                "daily_halt_active": post_risk.daily_halt_active,
                "kill_switch_active": post_risk.kill_switch_active,
                "daily_drawdown_pct": round(post_risk.daily_drawdown * 100, 2),
                "total_drawdown_pct": round(post_risk.total_drawdown * 100, 2),
                "peak_equity": round(state.peak_equity, 2),
                "daily_baseline_equity": round(state.daily_baseline_equity, 2),
            },
            "last_stock_screen_date": state.last_stock_screen_date,
            "last_polymarket_scan_date": state.last_polymarket_scan_date,
            "modules": modules,
            "pre_risk_equity": round(pre_risk.equity, 2),
            "post_risk_equity": round(post_risk.equity, 2),
        }

    def _run_module(self, name: str, callback: Any) -> dict[str, Any]:
        try:
            return callback()
        except Exception as exc:
            self.logger.error("%s module failed: %s", name, exc)
            return {"status": "error", "error": str(exc)}
