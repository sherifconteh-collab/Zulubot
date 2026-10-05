from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from trading_bot.audit import TradeAuditLog
from trading_bot.broker import PaperBroker
from trading_bot.clients import (
    AlternativeMeClient,
    BinanceFuturesClient,
    CoinbaseClient,
    FinancialModelingPrepClient,
    FreeMarketIntelligenceClient,
)
from trading_bot.clients_kalshi import KalshiClient, order_fill_count
from trading_bot.clients_polymarket import PolymarketClobClient, PolymarketGammaClient
from trading_bot.clients_robinhood import RobinhoodClient
from trading_bot.coinbase_broker import CoinbaseBroker
from trading_bot.config import AppConfig
from trading_bot.models import PortfolioState, Position
from trading_bot.performance_memory import record_close as _perf_record_close
from trading_bot.observability import DecisionLog, ModuleLatencyTracker
from trading_bot.polymarket_broker import PolymarketBroker
from trading_bot.risk import RiskManager, RiskSnapshot
from trading_bot.state import StateStore
from trading_bot.coinbase_feed import CoinbasePriceFeed
from trading_bot.strategies.altcoin_rotation import AltcoinRotationStrategy
from trading_bot.strategies.btc_5min import BTC5MinSpreadStrategy
from trading_bot.strategies.coinbase_predictions import CoinbasePredictionStrategy
from trading_bot.strategies.crypto import CryptoSentimentFundingStrategy
from trading_bot.strategies.latency_arb import LatencyArbStrategy
from trading_bot.strategies.markov_regime import MarkovRegimeStrategy
from trading_bot.strategies.options import OptionsStrategy
from trading_bot.strategies.polymarket import PolymarketStrategy
from trading_bot.strategies.kalshi_perpetuals import KalshiPerpetualsStrategy
from trading_bot.strategies.kalshi_sports import KalshiSportsStrategy
from trading_bot.clients_espn import EspnClient
from trading_bot.clients_polymarket_sports import PolymarketSportsClient
from trading_bot.strategies.stocks import StockValueStrategy

try:
    from trading_bot.integrations.trading_agents import TradingAgentsService
except ImportError:
    TradingAgentsService = None  # type: ignore[assignment,misc]


class TradingBotService:
    _KALSHI_CASH_BUCKET = "kalshi"

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger.getChild("service")
        self.state_store = StateStore(config.state_path, config.initial_cash, config.timezone)
        self.broker = PaperBroker(config.paper_mode, logger, max_price_history_days=config.volatility.max_history_days)
        self.risk_manager = RiskManager(config.risk, vol_config=config.volatility)
        self.latency_tracker = ModuleLatencyTracker(config.observability.module_latency_window)
        self.trade_audit = TradeAuditLog(
            config.trade_audit_path,
            logger,
            enabled=config.observability.trade_audit_enabled,
        )
        self.broker.set_on_trade(self.trade_audit.record)
        self.decision_log = DecisionLog(config.data_dir / "decision_log.jsonl", logger)

        # Wire broker close callback → risk manager so all close paths update
        # the consecutive-loss counter (strategies, trailing stops, kill switch).
        def _on_position_closed(state: PortfolioState, entry_price: float, close_price: float) -> None:
            self.risk_manager.record_trade_result(state, is_loss=close_price < entry_price)

        self.broker.set_on_close(_on_position_closed)

        # In-memory cache of the most recent leads per strategy. Updated by
        # the scheduler after each module run; read by the /leads command.
        self._last_leads: dict[str, list[dict[str, Any]]] = {}

        # Per-symbol performance memory ("learn as it goes"). On every close
        # we record the realized P&L against the symbol, which strategies
        # then consult to bench losers and prefer winners.
        def _on_close_learn(
            state: PortfolioState,
            symbol: str,
            entry_price: float,
            close_price: float,
            realized_pnl: float,
            reason: str,
        ) -> None:
            _perf_record_close(
                state,
                symbol=symbol,
                realized_pnl=realized_pnl,
                config=self.config.learning,
            )

        self.broker.set_on_position_closed(_on_close_learn)

        self.fmp_client = FinancialModelingPrepClient(
            api_key=config.fmp_api_key,
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
        )
        self.alternative_client = AlternativeMeClient(
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
            retry_config=config.http_retry,
        )
        self.binance_client = BinanceFuturesClient(
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
            retry_config=config.http_retry,
        )
        self.coinbase_client = CoinbaseClient(
            api_key=config.coinbase.api_key,
            api_secret=config.coinbase.api_secret,
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
            retry_config=config.http_retry,
        )

        # Free market intelligence (options sentiment, insider activity, short interest)
        # Uses yfinance — no API key required.
        self.market_intel_client = FreeMarketIntelligenceClient(logger=logger)
        self.logger.info("Free market intelligence client enabled (yfinance).")

        # TradingAgents AI service (optional)
        self.ai_service = None
        if TradingAgentsService is not None and config.trading_agents.enabled:
            self.ai_service = TradingAgentsService(config.trading_agents, logger)
            if self.ai_service.is_available:
                self._register_ai_tools()
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
            retry_config=config.http_retry,
        )

        self.stock_strategy = StockValueStrategy(
            config=config.stock,
            client=self.fmp_client,
            logger=logger,
            ai_service=self.ai_service,
            confirmation_mode=config.trading_agents.confirmation_mode,
            # Coinbase API is crypto-only — stock positions tracked internally
            coinbase_broker=None,
            market_intel_client=self.market_intel_client,
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
            learning_config=config.learning,
            protected_symbols=config.protected_symbols,
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
            ai_service=self.ai_service,
            confirmation_mode=config.trading_agents.confirmation_mode,
            protected_symbols=config.protected_symbols,
        )

        # Kalshi prediction markets strategy
        self.kalshi_client = KalshiClient(
            key_id=config.predictions.kalshi_key_id,
            private_key_pem=config.predictions.kalshi_private_key,
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
            retry_config=config.http_retry,
        )
        self.prediction_strategy = CoinbasePredictionStrategy(
            config=config.predictions,
            kalshi_client=self.kalshi_client,
            logger=logger,
            ai_service=self.ai_service,
            live_trading_enabled=not config.paper_mode and self.kalshi_client.can_trade,
            learning_config=config.learning,
        )

        # Kalshi Crypto Perpetuals strategy
        self.kalshi_perps_strategy = KalshiPerpetualsStrategy(
            config=config.kalshi_perps,
            kalshi_client=self.kalshi_client,
            protected_symbols=config.protected_symbols,
            logger=logger,
            live_trading_enabled=not config.paper_mode and self.kalshi_client.can_trade,
        )

        # Kalshi sports strategy (ESPN schedule/odds/projections -> Kalshi game-winner markets)
        self.espn_client = EspnClient(
            timeout_seconds=config.request_timeout_seconds,
            logger=logger,
            retry_config=config.http_retry,
        )
        self.kalshi_sports_strategy = KalshiSportsStrategy(
            config=config.kalshi_sports,
            kalshi_client=self.kalshi_client,
            espn_client=self.espn_client,
            poly_client=PolymarketSportsClient(timeout_seconds=config.request_timeout_seconds, logger=logger)
            if config.kalshi_sports.use_polymarket else None,
            logger=logger,
            live_trading_enabled=not config.paper_mode and self.kalshi_client.can_trade,
            calibration_path=config.data_dir / "sports_calibration.json",
        )

        # Latency arbitrage strategy (Coinbase price feed → Kalshi execution)
        self.price_feed = CoinbasePriceFeed(
            client=self.coinbase_client,
            symbols=list(config.latency_arb.symbols),
            history_seconds=config.latency_arb.price_history_seconds,
            logger=logger,
        )
        self.latency_arb_strategy = LatencyArbStrategy(
            config=config.latency_arb,
            price_feed=self.price_feed,
            kalshi_client=self.kalshi_client,
            paper_broker=self.broker,
            logger=logger,
            ai_service=self.ai_service,
            live_trading_enabled=not config.paper_mode and self.kalshi_client.can_trade,
        )

        # Markov Chain regime-switching strategy
        self.markov_strategy = MarkovRegimeStrategy(
            config=config.markov,
            coinbase_client=self.coinbase_client,
            binance_client=self.binance_client,
            coinbase_broker=self.coinbase_broker,
            logger=logger,
            protected_symbols=config.protected_symbols,
        )

        # Robinhood client + options strategy
        self.robinhood_client = RobinhoodClient(
            username=config.robinhood.username,
            password=config.robinhood.password,
            mfa_secret=config.robinhood.mfa_secret,
            logger=logger,
        )
        self.options_strategy = OptionsStrategy(
            config=config.options,
            robinhood_client=self.robinhood_client,
            alternative_client=self.alternative_client,
            logger=logger,
            live_trading_enabled=not config.paper_mode and self.robinhood_client.can_trade,
        )

        # BTC 5-min spread strategy (spot vs Kalshi vs Polymarket)
        self.btc5min_strategy = BTC5MinSpreadStrategy(
            config=config.btc5min,
            coinbase_client=self.coinbase_client,
            coinbase_broker=self.coinbase_broker,
            kalshi_client=self.kalshi_client,
            gamma_client=self.gamma_client,
            clob_client=self.clob_client,
            polymarket_broker=self.polymarket_broker,
            logger=logger,
            live_kalshi_trading=not config.paper_mode and self.kalshi_client.can_trade,
        )

        # Kalshi balance cache (avoids network call on every summary)
        self._kalshi_balance_cache: float | None = None
        self._kalshi_balance_cache_ts: float = 0.0
        self._kalshi_balance_cache_ttl: float = 120.0  # 2-minute TTL

    def _skipped_module(self, reason: str) -> dict[str, Any]:
        return {
            "status": "skipped",
            "reason": reason,
            "opens": [],
            "closes": [],
        }

    def run(self, command: str, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(self.config.timezone)
        state = self.state_store.load()
        self._hydrate_ai_budget(state)

        # Tick down cooldown counter each trade cycle
        self.risk_manager.tick_cooldown(state)

        # Sync real balances when in live mode
        if not self.config.paper_mode:
            self._sync_coinbase_balance(state)
            self._sync_kalshi_balance(state)
            self._sync_kalshi_positions(state)

        self.refresh_open_positions(state, now=now)

        # If Markov is disabled, clear any stale regime state so downstream gates
        # (Polymarket, Kalshi) aren't blocked by a bear signal from a prior run.
        if not self.config.markov.enabled:
            state.markov_regime_state = ""
            state.markov_regime_signal = 0.0

        # Update trailing-stop high watermarks and close breached positions
        self.risk_manager.update_high_watermarks(state)
        trailing_closed = self._enforce_trailing_stops(state)

        pre_risk = self._evaluate_risk(state, now)
        module_results: dict[str, Any] = {}

        if trailing_closed:
            module_results["trailing_stops"] = {
                "closed": trailing_closed,
                "count": len(trailing_closed),
            }

        if command in {"stocks", "all"}:
            if self.config.paper_mode:
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
            else:
                self.logger.warning("Stocks module skipped in live mode: live stock execution is not implemented.")
                module_results["stocks"] = self._skipped_module("live_stock_execution_unavailable")
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
        # Markov runs before gated modules so state.markov_regime_signal is fresh.
        if command == "markov" or (command == "all" and self.config.markov.enabled):
            module_results["markov"] = self._run_module(
                "markov",
                lambda: self.markov_strategy.run(
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
        if command == "kalshi_perps" or (command == "all" and self.config.kalshi_perps.enabled):
            module_results["kalshi_perps"] = self._run_module(
                "kalshi_perps",
                lambda: self.kalshi_perps_strategy.run(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                ),
            )
        if command == "kalshi_sports" or (command == "all" and self.config.kalshi_sports.enabled):
            module_results["kalshi_sports"] = self._run_module(
                "kalshi_sports",
                lambda: self.kalshi_sports_strategy.run(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                ),
            )
        if command in {"polymarket", "all"}:
            if self.config.paper_mode:
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
            else:
                self.logger.warning("Polymarket module skipped in live mode: live Polymarket execution is not implemented.")
                module_results["polymarket"] = self._skipped_module("polymarket_live_execution_unavailable")
        if command in {"latency_arb", "all"}:
            module_results["latency_arb"] = self._run_module(
                "latency_arb",
                lambda: self.latency_arb_strategy.run_once(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                ),
            )
        if command == "btc5min":
            module_results["btc5min"] = self._run_module(
                "btc5min",
                lambda: self.btc5min_strategy.run(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                ),
            )
        if command in {"options", "all"}:
            module_results["options"] = self._run_module(
                "options",
                lambda: self.options_strategy.run(
                    state,
                    broker=self.broker,
                    risk_manager=self.risk_manager,
                    now=now,
                ),
            )

        self.refresh_open_positions(state, now=now)
        post_risk = self._evaluate_risk(state, now)
        self._persist_ai_budget(state)
        self.state_store.save(state)

        summary = self._build_summary(state, pre_risk=pre_risk, post_risk=post_risk, modules=module_results)
        self.logger.info("Run summary: %s", summary)
        return summary

    def status(self) -> dict[str, Any]:
        now = datetime.now(self.config.timezone)
        state = self.state_store.load()
        self._hydrate_ai_budget(state)
        if not self.config.paper_mode:
            self._sync_coinbase_balance(state)
            self._sync_kalshi_balance(state)
            self._sync_kalshi_positions(state)
        self.refresh_open_positions(state, now=now)
        risk = self._evaluate_risk(state, now)
        self._persist_ai_budget(state)
        self.state_store.save(state)
        return self._build_summary(state, pre_risk=risk, post_risk=risk, modules={})

    def engage_kill_switch(self, *, reason: str = "manual_kill_switch") -> dict[str, Any]:
        now = datetime.now(self.config.timezone)
        state = self.state_store.load()
        self._hydrate_ai_budget(state)
        if not self.config.paper_mode:
            self._sync_coinbase_balance(state)
            self._sync_kalshi_balance(state)
            self._sync_kalshi_positions(state)
        self.refresh_open_positions(state, now=now)

        pre_risk = self._evaluate_risk(state, now)
        state.kill_switch_engaged = True
        liquidation = self._liquidate_all_positions(state, reason=reason)
        post_risk = self._evaluate_risk(state, now)
        self._persist_ai_budget(state)
        self.state_store.save(state)

        summary = self._build_summary(
            state,
            pre_risk=pre_risk,
            post_risk=post_risk,
            modules={
                "kill_switch": {
                    "status": "engaged",
                    "closed": liquidation["closed"],
                    "failed": liquidation["failed"],
                    "reason": reason,
                }
            },
        )
        self.logger.warning("Kill switch engaged manually. closed=%s failed=%s", liquidation["closed"], liquidation["failed"])
        return summary

    def update_last_leads(self, strategy: str, leads: list[dict[str, Any]]) -> None:
        """Replace the cached leads for *strategy* (called after each scheduler run)."""
        self._last_leads[strategy] = list(leads)

    def get_last_leads(self) -> dict[str, list[dict[str, Any]]]:
        """Return the most recent per-strategy leads (for /leads command)."""
        return {strategy: list(leads) for strategy, leads in self._last_leads.items()}

    def clear_kill_switch(self, force: bool = False) -> dict[str, Any]:
        now = datetime.now(self.config.timezone)
        state = self.state_store.load()
        self._hydrate_ai_budget(state)
        if not self.config.paper_mode:
            self._sync_coinbase_balance(state)
            self._sync_kalshi_balance(state)
            self._sync_kalshi_positions(state)
        self.refresh_open_positions(state, now=now)

        equity = self.broker.portfolio_equity(state)
        total_drawdown = 0.0
        if state.peak_equity > 0 and equity < state.peak_equity:
            total_drawdown = (state.peak_equity - equity) / state.peak_equity

        peak_reset_from: float | None = None
        peak_reset_to: float | None = None
        if total_drawdown >= self.config.risk.kill_switch_drawdown:
            if not force:
                raise RuntimeError(
                    f"Kill switch cannot be cleared while total drawdown is {total_drawdown * 100:.2f}% "
                    f"and threshold is {self.config.risk.kill_switch_drawdown * 100:.2f}%"
                )
            # Forced clear: rebase the high-water mark to current equity so the
            # operator can resume after a stale peak (e.g. after major losses
            # or a strategy reset). daily_baseline_equity is also rebased so
            # the daily-halt check doesn't immediately re-trip.
            peak_reset_from = state.peak_equity
            peak_reset_to = equity
            state.peak_equity = equity
            state.daily_baseline_equity = equity
            state.halt_new_trades_for_day = False
            total_drawdown = 0.0
            self.logger.warning(
                "Kill switch force-cleared; peak_equity reset $%.2f -> $%.2f",
                peak_reset_from, peak_reset_to,
            )

        pre_risk = self._evaluate_risk(state, now)
        state.kill_switch_engaged = False
        state.kill_switch_cleared_at = now.isoformat()
        post_risk = self._evaluate_risk(state, now)
        self._persist_ai_budget(state)
        self.state_store.save(state)

        kill_switch_summary: dict[str, Any] = {
            "status": "cleared",
            "reason": "manual_resume_forced" if force and peak_reset_from is not None else "manual_resume",
        }
        if peak_reset_from is not None:
            kill_switch_summary["peak_reset_from"] = peak_reset_from
            kill_switch_summary["peak_reset_to"] = peak_reset_to

        summary = self._build_summary(
            state,
            pre_risk=pre_risk,
            post_risk=post_risk,
            modules={"kill_switch": kill_switch_summary},
        )
        self.logger.info("Kill switch cleared manually at drawdown=%.2f%%", total_drawdown * 100)
        return summary

    def _hydrate_ai_budget(self, state: PortfolioState) -> None:
        if self.ai_service is None:
            return
        self.ai_service.load_budget_state(state.ai_monthly_usage)

    def _persist_ai_budget(self, state: PortfolioState) -> None:
        if self.ai_service is None:
            return
        state.ai_monthly_usage = self.ai_service.export_budget_state()

    def _register_ai_tools(self) -> None:
        if self.ai_service is None:
            return

        self.ai_service.clear_tools()
        self.ai_service.register_tool(
            "get_current_price",
            "Get the latest spot price for a Coinbase product, Binance-style crypto symbol, or stock ticker.",
            {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Coinbase product ID like BTC-USD, Binance symbol like BTCUSDT, or equity ticker like AAPL.",
                    }
                },
                "required": ["symbol"],
            },
            self._ai_tool_get_current_price,
        )
        self.ai_service.register_tool(
            "get_fear_greed_index",
            "Get the latest Alternative.me crypto fear and greed reading.",
            {
                "type": "object",
                "properties": {},
            },
            self._ai_tool_get_fear_greed_index,
        )
        self.ai_service.register_tool(
            "get_funding_rate",
            "Get the current funding rate and mark price for a crypto perpetual symbol.",
            {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Binance-style symbol like BTCUSDT or a Coinbase-style symbol like BTC-USD.",
                    }
                },
                "required": ["symbol"],
            },
            self._ai_tool_get_funding_rate,
        )

    def _ai_tool_get_current_price(self, payload: dict[str, Any]) -> dict[str, Any]:
        symbol = str(payload.get("symbol", "")).strip().upper()
        if not symbol:
            raise ValueError("symbol is required")

        product_id = symbol
        try:
            if symbol.endswith("USDT"):
                product_id = self.coinbase_client.to_coinbase_symbol(symbol)
            elif "-" not in symbol:
                quotes = self.fmp_client.get_batch_quotes([symbol])
                price = quotes.get(symbol)
                if price is None:
                    raise ValueError(f"No price available for {symbol}")
                return {
                    "symbol": symbol,
                    "resolved_symbol": symbol,
                    "price": round(float(price), 8),
                    "source": "yfinance",
                }
        except ValueError:
            quotes = self.fmp_client.get_batch_quotes([symbol])
            price = quotes.get(symbol)
            if price is None:
                raise
            return {
                "symbol": symbol,
                "resolved_symbol": symbol,
                "price": round(float(price), 8),
                "source": "yfinance",
            }

        price = self.coinbase_client.get_spot_price_by_product_id(product_id)
        return {
            "symbol": symbol,
            "resolved_symbol": product_id,
            "price": round(float(price), 8),
            "source": "coinbase",
        }

    def _ai_tool_get_fear_greed_index(self, _payload: dict[str, Any]) -> dict[str, Any]:
        reading = self.alternative_client.get_fear_and_greed()
        return {
            "value": reading.value,
            "classification": reading.classification,
            "timestamp": reading.timestamp,
            "time_until_update": reading.time_until_update,
            "source": "alternative_me",
        }

    def _ai_tool_get_funding_rate(self, payload: dict[str, Any]) -> dict[str, Any]:
        symbol = str(payload.get("symbol", "")).strip().upper()
        if not symbol:
            raise ValueError("symbol is required")
        if symbol.endswith("-USD"):
            try:
                symbol = self.coinbase_client.to_binance_symbol(symbol)
            except ValueError:
                symbol = f"{symbol.split('-', 1)[0]}USDT"
        elif not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"

        snapshot = self.binance_client.get_mark_price_and_funding(symbol)
        return {
            "symbol": payload.get("symbol", symbol),
            "resolved_symbol": symbol,
            "mark_price": round(snapshot.mark_price, 8),
            "funding_rate": round(snapshot.funding_rate, 8),
            "source": "hyperliquid+coinbase",
        }

    def get_latency_snapshot(self) -> dict[str, dict[str, float]]:
        return self.latency_tracker.snapshot()

    def get_recent_audit_entries(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.trade_audit.tail(limit)

    def get_recent_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.decision_log.tail(limit)

    @staticmethod
    def _normalise_kalshi_probability(raw_value: Any) -> float:
        if raw_value in (None, ""):
            return 0.0
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 0.0
        if value <= 0:
            return 0.0
        return value / 100.0 if value > 1.0 else value

    def _kalshi_position_price(self, market: dict[str, Any], side: str) -> float:
        yes_bid = self._normalise_kalshi_probability(market.get("yes_bid_dollars") or market.get("yes_bid"))
        yes_ask = self._normalise_kalshi_probability(market.get("yes_ask_dollars") or market.get("yes_ask"))
        no_bid = self._normalise_kalshi_probability(market.get("no_bid_dollars") or market.get("no_bid"))
        no_ask = self._normalise_kalshi_probability(market.get("no_ask_dollars") or market.get("no_ask"))
        last_price = self._normalise_kalshi_probability(market.get("last_price_dollars") or market.get("last_price"))

        yes_mid = (yes_bid + yes_ask) / 2 if yes_bid > 0 and yes_ask > 0 else (yes_bid or yes_ask or last_price)
        no_mid = (no_bid + no_ask) / 2 if no_bid > 0 and no_ask > 0 else (no_bid or no_ask)
        if no_mid <= 0 and yes_mid > 0:
            no_mid = max(0.0, min(1.0, 1.0 - yes_mid))
        if yes_mid <= 0 and no_mid > 0:
            yes_mid = max(0.0, min(1.0, 1.0 - no_mid))

        side_price = no_mid if side == "no" else yes_mid
        return max(0.0, min(1.0, side_price))

    def _position_price_max_age_seconds(self, position: Position) -> float:
        if not self.config.price_staleness.enabled:
            return 0.0
        if position.asset_type == "stock":
            return self.config.price_staleness.stock_max_age_seconds
        if position.asset_type == "crypto":
            return self.config.price_staleness.crypto_max_age_seconds
        if position.asset_type == "prediction":
            return self.config.price_staleness.prediction_max_age_seconds
        return 0.0

    def _collect_stale_prices(
        self,
        state: PortfolioState,
        *,
        as_of: datetime,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self.config.price_staleness.enabled:
            return [], 0

        as_of_ts = as_of.timestamp()
        stale_entries: list[dict[str, Any]] = []
        for symbol, position in state.positions.items():
            max_age = self._position_price_max_age_seconds(position)
            if max_age <= 0:
                continue

            last_ts = float(state.last_price_timestamps.get(symbol, 0.0) or 0.0)
            is_missing = last_ts <= 0.0
            age_seconds = max(0.0, as_of_ts - last_ts) if not is_missing else None
            if is_missing or (age_seconds is not None and age_seconds > max_age):
                stale_entries.append(
                    {
                        "symbol": symbol,
                        "asset_type": position.asset_type,
                        "strategy": position.strategy,
                        "age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
                        "max_age_seconds": round(max_age, 1),
                        "last_price_at": (
                            datetime.fromtimestamp(last_ts, self.config.timezone).isoformat()
                            if not is_missing else None
                        ),
                        "stale": True,
                        "_sort_age": age_seconds if age_seconds is not None else float("inf"),
                    }
                )

        stale_entries.sort(key=lambda item: item["_sort_age"], reverse=True)
        total_count = len(stale_entries)
        max_items = max(1, self.config.observability.max_stale_symbols_in_summary)
        return [
            {key: value for key, value in entry.items() if key != "_sort_age"}
            for entry in stale_entries[:max_items]
        ], total_count

    def refresh_open_positions(self, state: PortfolioState, *, now: datetime | None = None) -> None:
        # Collect all stock + crypto prices in one batch so a single mark_positions
        # call fires the daily price-history gate. This ensures the Markov regime
        # symbol (a crypto ticker) is recorded to price_history even on days when
        # stock positions trigger the gate before the crypto refresh would run.
        batch_prices: dict[str, float] = {}

        stock_symbols = sorted(
            symbol for symbol, position in state.positions.items() if position.asset_type == "stock"
        )
        if stock_symbols:
            try:
                batch_prices.update(self.fmp_client.get_batch_quotes(stock_symbols))
            except Exception as exc:
                self.logger.warning("Unable to refresh stock prices: %s", exc)

        crypto_symbols = sorted({
            symbol
            for symbol, position in state.positions.items()
            if position.asset_type == "crypto"
        })
        # Always include the Markov regime symbol so its history accumulates
        # even when there are no open crypto positions.
        if self.config.markov.enabled:
            crypto_symbols = sorted(set(crypto_symbols) | {self.config.markov.regime_symbol})
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
                batch_prices.update(prices)
            except Exception as exc:
                self.logger.warning("Unable to refresh crypto prices: %s", exc)

        if batch_prices:
            self.broker.mark_positions(state, batch_prices, now=now)

        kalshi_prediction_prices: dict[str, float] = {}
        kalshi_prediction_symbols = {
            symbol: position
            for symbol, position in state.positions.items()
            if position.asset_type == "prediction" and position.meta.get("venue") == "kalshi"
        }
        if kalshi_prediction_symbols:
            try:
                for symbol, position in kalshi_prediction_symbols.items():
                    ticker = str(position.meta.get("ticker", "")).strip()
                    side = str(position.meta.get("side", "yes")).strip().lower()
                    if not ticker:
                        continue
                    market = self.kalshi_client.get_market(ticker)
                    price = self._kalshi_position_price(market, side)
                    if price > 0:
                        kalshi_prediction_prices[symbol] = price
                if kalshi_prediction_prices:
                    self.broker.mark_positions(state, kalshi_prediction_prices, now=now)
            except Exception as exc:
                self.logger.warning("Unable to refresh Kalshi prediction prices: %s", exc)

        # Refresh options positions via yfinance
        options_positions = {
            sym: pos
            for sym, pos in state.positions.items()
            if pos.asset_type == "option"
        }
        if options_positions:
            try:
                options_prices: dict[str, float] = {}
                for sym, pos in options_positions.items():
                    price = self.options_strategy.get_current_option_price(pos)
                    if price is not None and price > 0:
                        options_prices[sym] = price
                if options_prices:
                    self.broker.mark_positions(state, options_prices, now=now)
            except Exception as exc:
                self.logger.warning("Unable to refresh options prices: %s", exc)

        # Refresh prediction market prices from CLOB
        prediction_symbols = sorted(
            symbol
            for symbol, position in state.positions.items()
            if position.asset_type == "prediction" and position.meta.get("venue") != "kalshi"
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
                    self.broker.mark_positions(state, prices, now=now)
            except Exception as exc:
                self.logger.warning("Unable to refresh prediction prices: %s", exc)

    def _sync_coinbase_balance(self, state: PortfolioState) -> None:
        """Sync portfolio cash and equity from real Coinbase balances."""
        try:
            total_usd, breakdown = self.coinbase_broker.get_total_balance()
            usd_cash = breakdown.get("USD", 0.0) + breakdown.get("USDC", 0.0)

            # Set cash to real Coinbase USD/USDC balance
            state.cash = usd_cash

            # Use Coinbase total (includes all crypto) as the real equity
            # This is more accurate than summing positions which may lag
            real_equity = total_usd if total_usd > 0 else usd_cash

            # Reset peak/baseline if they look stale (from old paper-mode state
            # or different initial_cash) to prevent false kill-switch triggers
            if real_equity > 0 and (
                state.peak_equity <= 0
                or state.peak_equity == self.config.initial_cash
                or state.peak_equity > real_equity * 5  # peak is impossibly high vs reality
            ):
                self.logger.info(
                    "Resetting peak equity from $%.2f to real equity $%.2f",
                    state.peak_equity, real_equity,
                )
                state.peak_equity = real_equity
                state.daily_baseline_equity = real_equity

            self.logger.info(
                "Coinbase balance synced: cash=$%.2f total=$%.2f assets=%s",
                usd_cash, total_usd, {k: f"${v:.2f}" for k, v in breakdown.items()},
            )
        except Exception as exc:
            self.logger.warning("Failed to sync Coinbase balance: %s. Using cached state.", exc)
            # In live mode, never show fake money — if sync fails and cash
            # looks like the default initial_cash, zero it out.
            if state.cash >= self.config.initial_cash and self.config.initial_cash > 0:
                self.logger.warning("Zeroing cash to prevent phantom balance in live mode.")
                state.cash = 0.0

    def _sync_kalshi_balance(self, state: PortfolioState) -> None:
        """Sync live Kalshi buying power into a dedicated cash bucket."""
        if not self.kalshi_client.can_trade:
            return
        try:
            balance = self.kalshi_client.get_balance()
            state.cash_reserves[self._KALSHI_CASH_BUCKET] = balance
            self._kalshi_balance_cache = balance
            self._kalshi_balance_cache_ts = time.monotonic()
            self.logger.info("Kalshi balance synced: cash=$%.2f", balance)
        except Exception as exc:
            self.logger.warning("Failed to sync Kalshi balance: %s. Using cached balance.", exc)
            state.cash_reserves.setdefault(self._KALSHI_CASH_BUCKET, 0.0)

    _RECONCILE_GRACE_SECONDS = 600
    _KALSHI_SETTLED = {"settled", "finalized", "determined"}

    def _reconcile_bot_kalshi_positions(self, state: PortfolioState, held: dict[tuple[str, str], int]) -> None:
        """Make the bot's own live Kalshi bets match what the account actually holds.

        A bet that never filled (or was closed elsewhere) would otherwise sit in state as
        phantom value until the market settles. Bets whose market has settled are left for
        their strategy to close and record. Recent bets get a grace period.
        """
        now = datetime.now(self.config.timezone)
        for symbol, pos in list(state.positions.items()):
            if (
                pos.asset_type != "prediction"
                or pos.meta.get("venue") != "kalshi"
                or pos.meta.get("source") == "kalshi_sync"
            ):
                continue
            ticker = str(pos.meta.get("ticker", ""))
            key = (ticker, str(pos.meta.get("side", "")).upper())
            if not ticker or not key[1]:
                continue
            actual = held.get(key, 0)
            if actual >= pos.quantity:
                continue
            try:
                opened = datetime.fromisoformat(pos.opened_at)
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=now.tzinfo)
                if (now - opened).total_seconds() < self._RECONCILE_GRACE_SECONDS:
                    continue
                if str(self.kalshi_client.get_market(ticker).get("status", "")).lower() in self._KALSHI_SETTLED:
                    continue  # settled: the owning strategy closes and records it
            except Exception as exc:
                self.logger.debug("Reconcile check skipped for %s: %s", symbol, exc)
                continue
            if actual > 0:
                self.logger.warning("Kalshi holds %d of %.0f recorded for %s; shrinking.", actual, pos.quantity, symbol)
                self.broker.resize_position(state, symbol, float(actual), refund=False)
            else:
                self.logger.warning("Kalshi holds none of %s (unfilled or closed elsewhere); removing.", symbol)
                self.broker.resize_position(state, symbol, 0.0, refund=False)

    @staticmethod
    def _kalshi_position_counts(pos: dict[str, Any]) -> tuple[int, int]:
        """(yes_contracts, no_contracts) from a Kalshi market_position.

        Kalshi reports one signed ``position``: positive = YES, negative = NO.
        The count may arrive as ``position_fp`` (a string) in newer payloads; the
        legacy ``yes_count``/``no_count`` fields are honoured when present.
        """
        raw = pos.get("position_fp", pos.get("position"))
        try:
            signed = int(round(float(raw))) if raw not in (None, "") else 0
        except (TypeError, ValueError):
            signed = 0
        yes_count, no_count = max(signed, 0), max(-signed, 0)
        try:
            yes_count = max(yes_count, int(float(pos.get("yes_count") or 0)))
            no_count = max(no_count, int(float(pos.get("no_count") or 0)))
        except (TypeError, ValueError):
            pass
        return yes_count, no_count

    def _sync_kalshi_positions(self, state: PortfolioState) -> None:  # noqa: C901
        """Fetch live Kalshi positions and upsert them into PortfolioState.

        This surfaces any positions placed outside the bot (e.g. via Kalshi.com)
        so they appear in status reports and are tracked alongside bot-placed positions.
        Only runs when Kalshi credentials are configured.
        """
        if not self.kalshi_client.can_trade:
            return
        try:
            positions = self.kalshi_client.get_positions()  # raises on API failure: never prune then

            # What the account really holds right now: {(ticker, "YES"|"NO"): contracts}.
            held: dict[tuple[str, str], int] = {}
            for pos in positions or []:
                ticker = pos.get("ticker", "")
                if not ticker:
                    continue
                yes_count, no_count = self._kalshi_position_counts(pos)
                if yes_count > 0:
                    held[(ticker, "YES")] = yes_count
                if no_count > 0:
                    held[(ticker, "NO")] = no_count

            # Drop imported positions that are no longer held (settled, sold, expired).
            # Without this they linger forever at their last mark, inflating equity on
            # top of the cash already credited to the wallet and distorting drawdown.
            for symbol, existing in list(state.positions.items()):
                if existing.meta.get("source") != "kalshi_sync":
                    continue
                key = (str(existing.meta.get("ticker", "")), str(existing.meta.get("side", "")).upper())
                if key not in held:
                    self.logger.info("Kalshi position %s no longer held; removing from state.", symbol)
                    del state.positions[symbol]
                    state.last_prices.pop(symbol, None)
                    state.last_price_timestamps.pop(symbol, None)

            self._reconcile_bot_kalshi_positions(state, held)

            if not held:
                return

            # Contracts the bot placed itself (KX:/KS:/...) are already tracked;
            # importing them again as "KL:" manual positions would double-count.
            tracked = {
                (str(p.meta.get("ticker", "")), str(p.meta.get("side", "")).upper())
                for p in state.positions.values()
                if p.asset_type == "prediction" and p.meta.get("source") != "kalshi_sync"
            }

            synced = 0
            for (ticker, side), count in held.items():
                if count <= 0:
                    continue

                symbol = f"KL:{ticker}:{side}"
                if (ticker, side) in tracked and symbol not in state.positions:
                    continue
                if symbol in state.positions:
                    # Update quantity for existing tracked position
                    state.positions[symbol].quantity = float(count)
                    continue

                # New position not tracked by the bot — try to get a real price
                entry_price = 0.50
                try:
                    market = self.kalshi_client.get_market(ticker)
                    # Handles both ``*_dollars`` and legacy cent fields.
                    price = self._kalshi_position_price(market, side.lower())
                    if 0 < price < 1:
                        entry_price = price
                except Exception:
                    pass  # Fall back to 0.50 placeholder

                synced_at = datetime.now(self.config.timezone).isoformat()
                state.positions[symbol] = Position(
                    symbol=symbol,
                    asset_type="prediction",
                    strategy="kalshi_manual",
                    quantity=float(count),
                    entry_price=entry_price,
                    current_price=entry_price,
                    opened_at=synced_at,
                    meta={
                        "ticker": ticker,
                        "side": side.lower(),
                        "source": "kalshi_sync",
                        "venue": "kalshi",
                        "cash_bucket": self._KALSHI_CASH_BUCKET,
                    },
                )
                synced += 1

            if synced:
                self.logger.info("Synced %d new Kalshi positions from account.", synced)

        except Exception as exc:
            self.logger.warning("Failed to sync Kalshi positions: %s", exc)

    def _enforce_trailing_stops(self, state: PortfolioState) -> list[str]:
        """Close positions that breach their trailing stop-loss.

        Trade-result tracking (consecutive-loss cooldown) is handled
        automatically by the broker's ``on_close`` callback.
        """
        symbols = self.risk_manager.check_trailing_stops(state)
        closed: list[str] = []
        for symbol in symbols:
            position = state.positions.get(symbol)
            if position is None or self._exempt_from_global_halt(position):
                continue
            self.broker.close_position(
                state, symbol=symbol, reason="trailing_stop", price=position.current_price,
            )
            closed.append(symbol)
        return closed

    def _evaluate_risk(self, state: PortfolioState, now: datetime) -> RiskSnapshot:
        equity = self.broker.portfolio_equity(state)
        snapshot = self.risk_manager.evaluate(state, equity, now)
        if snapshot.kill_switch_triggered_now:
            self.logger.error(
                "Kill switch triggered at equity=%.2f total_drawdown=%.2f%%. Closing all positions.",
                snapshot.equity,
                snapshot.total_drawdown * 100,
            )
            liquidation = self._liquidate_all_positions(state, reason="kill_switch")
            if liquidation["failed"]:
                self.logger.error("Kill switch close failures: %s", liquidation["failed"])
        elif snapshot.daily_halt_active:
            self.logger.warning(
                "Daily halt active at equity=%.2f daily_drawdown=%.2f%%.",
                snapshot.equity,
                snapshot.daily_drawdown * 100,
            )
        return snapshot

    def _exempt_from_global_halt(self, position: Position) -> bool:
        """Sports bets settle on their own and are not closed by the global kill switch."""
        return (
            position.strategy == "kalshi_sports"
            and self.config.kalshi_sports.ignore_global_halt
        )

    def _liquidate_all_positions(self, state: PortfolioState, *, reason: str) -> dict[str, list[str]]:
        closed: list[str] = []
        failed: list[str] = []

        for symbol in list(state.positions.keys()):
            position = state.positions.get(symbol)
            if position is None:
                continue
            if self._exempt_from_global_halt(position):
                self.logger.info("Keeping %s through %s: sports bets hold to settlement.", symbol, reason)
                continue

            close_price = state.last_prices.get(symbol, position.current_price or position.entry_price)
            live_close_ok = True

            if not self.config.paper_mode:
                if position.asset_type == "crypto":
                    live_close_ok = self.coinbase_broker.place_market_sell_by_product(
                        product_id=position.symbol,
                        quantity=position.quantity,
                        reason=reason,
                    )
                elif position.asset_type == "prediction" and symbol.startswith(("KX:", "KS:", "KAL:")):
                    ticker = str(position.meta.get("ticker", ""))
                    side = str(position.meta.get("side", "yes")).lower()
                    contracts = max(1, int(position.quantity))
                    if self.kalshi_client.can_trade and ticker:
                        try:
                            order = self.kalshi_client.sell_to_close(ticker=ticker, side=side, count=contracts)
                            sold = order_fill_count(order)
                            if sold is not None and sold < contracts:
                                self.logger.error("Kill-switch sell for %s filled %d of %d.", symbol, sold, contracts)
                                live_close_ok = False
                        except Exception as exc:
                            self.logger.error("Kalshi kill-switch sell failed for %s: %s", symbol, exc)
                            live_close_ok = False
                    else:
                        live_close_ok = False
                elif position.asset_type == "option":
                    if self.robinhood_client.can_trade:
                        meta = position.meta or {}
                        close_px = position.current_price or position.entry_price
                        order = self.robinhood_client.place_option_sell_order(
                            symbol=str(meta.get("underlying", "")),
                            expiration_date=str(meta.get("expiration_date", "")),
                            strike=str(meta.get("strike", "")),
                            option_type=str(meta.get("option_type", "call")),
                            quantity=max(1, int(meta.get("contracts", 1))),
                            limit_price=max(0.01, round(close_px * 0.98, 2)),
                        )
                        live_close_ok = order is not None
                    else:
                        self.logger.warning("Skipping live option close for %s — Robinhood not configured.", symbol)
                        live_close_ok = False
                elif position.asset_type == "prediction":
                    self.logger.warning("Skipping unsupported live prediction close for %s during %s", symbol, reason)
                    live_close_ok = False

            if not live_close_ok:
                failed.append(symbol)
                continue

            self.broker.close_position(state, symbol=symbol, price=close_price, reason=reason)
            closed.append(symbol)

        return {"closed": closed, "failed": failed}

    def _build_summary(
        self,
        state: PortfolioState,
        *,
        pre_risk: RiskSnapshot,
        post_risk: RiskSnapshot,
        modules: dict[str, Any],
    ) -> dict[str, Any]:
        summary_time = datetime.now(self.config.timezone)
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
        stale_prices, stale_price_count = self._collect_stale_prices(state, as_of=summary_time)

        # Fetch Kalshi account balance with TTL cache to avoid network
        # calls on every run() / status() invocation.
        kalshi_balance: float | None = None
        if self.kalshi_client.can_trade:
            now_ts = time.monotonic()
            if (now_ts - self._kalshi_balance_cache_ts) < self._kalshi_balance_cache_ttl:
                kalshi_balance = self._kalshi_balance_cache
            else:
                try:
                    kalshi_balance = round(self.kalshi_client.get_balance(), 2)
                    self._kalshi_balance_cache = kalshi_balance
                    self._kalshi_balance_cache_ts = now_ts
                except Exception as exc:
                    self.logger.debug("Could not fetch Kalshi balance: %s", exc)
                    if self._kalshi_balance_cache is not None:
                        kalshi_balance = self._kalshi_balance_cache  # stale is better than None

        # AI/multi-analyst metadata
        ai_info: dict[str, Any] | None = None
        if self.ai_service:
            ai_info = {
                "budget_remaining_usd": round(self.ai_service.get_budget_remaining(), 4),
                "monthly_cost_usd": round(self.ai_service.get_monthly_cost(), 4),
                "multi_analyst_enabled": self.config.trading_agents.multi_analyst_enabled,
                "selected_analysts": list(self.config.trading_agents.selected_analysts),
                "confirmation_mode": self.config.trading_agents.confirmation_mode,
                "tool_calling_enabled": self.config.trading_agents.tool_calling_enabled,
                "streaming_enabled": self.config.trading_agents.streaming_enabled,
                "extended_thinking_enabled": self.config.trading_agents.extended_thinking_enabled,
                "prompt_guard_enabled": self.config.trading_agents.prompt_injection_protection_enabled,
                "vision_enabled": self.config.trading_agents.vision_enabled,
                "registered_tools": self.ai_service.get_registered_tools(),
                "circuit_breaker": self.ai_service.get_circuit_breaker_status(),
            }

        return {
            "timestamp": summary_time.isoformat(),
            "paper_mode": self.config.paper_mode,
            "cash": round(state.cash, 2),
            "cash_reserves": {
                bucket: round(balance, 2)
                for bucket, balance in state.cash_reserves.items()
            },
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
                "cooldown_active": post_risk.cooldown_active,
                "consecutive_losses": state.consecutive_losses,
                "cooldown_remaining": state.cooldown_remaining,
                "post_kill_cooldown_active": post_risk.post_kill_cooldown_active,
                "kill_switch_cleared_at": state.kill_switch_cleared_at or None,
            },
            "last_stock_screen_date": state.last_stock_screen_date,
            "last_polymarket_scan_date": state.last_polymarket_scan_date,
            "last_predictions_scan_date": state.last_predictions_scan_date,
            "modules": modules,
            "latency_ms": self.latency_tracker.snapshot(),
            "stale_prices": stale_prices,
            "stale_price_count": stale_price_count,
            "pre_risk_equity": round(pre_risk.equity, 2),
            "post_risk_equity": round(post_risk.equity, 2),
            "kalshi_balance": kalshi_balance,
            "ai_info": ai_info,
            # Keep for backwards compatibility
            "ai_budget_remaining_usd": round(self.ai_service.get_budget_remaining(), 4) if self.ai_service else None,
        }

    def _run_module(self, name: str, callback: Any) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = callback()
        except Exception as exc:
            self.logger.error("%s module failed: %s", name, exc)
            result = {"status": "error", "error": str(exc)}

        duration_ms = (time.perf_counter() - started) * 1000.0
        self.latency_tracker.record(name, duration_ms)
        if isinstance(result, dict):
            result.setdefault("latency_ms", round(duration_ms, 2))
            decisions = result.get("decisions")
            if decisions:
                self.decision_log.record_many(decisions)
        return result
