from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients import FinancialModelingPrepClient
from trading_bot.config import StockStrategyConfig
from trading_bot.models import PortfolioState, StockFundamentals
from trading_bot.risk import RiskManager

try:
    from trading_bot.integrations.trading_agents import (
        TradingAgentsService,
        apply_ai_confirmation,
    )
except ImportError:
    TradingAgentsService = None  # type: ignore[assignment,misc]
    apply_ai_confirmation = None  # type: ignore[assignment]


def _as_float(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


class StockValueStrategy:
    PE_KEYS = ["peRatioTTM", "priceEarningsRatioTTM", "peRatio", "priceEarningsRatio"]
    PB_KEYS = ["pbRatioTTM", "priceToBookRatioTTM", "pbRatio", "priceToBookRatio"]
    DEBT_TO_EQUITY_KEYS = ["debtToEquityTTM", "debtEquityRatioTTM", "debtToEquity", "debtEquityRatio"]
    PFCF_KEYS = ["pfcfRatioTTM", "pfcfRatio", "priceToFreeCashFlowsRatioTTM", "priceToFreeCashFlowsRatio"]
    POSITIVE_FCF_KEYS = ["freeCashFlowYieldTTM", "freeCashFlowPerShareTTM", "freeCashFlow", "freeCashFlowTTM"]

    def __init__(
        self,
        *,
        config: StockStrategyConfig,
        client: FinancialModelingPrepClient,
        logger: logging.Logger,
        ai_service: Any | None = None,
        confirmation_mode: str = "log_only",
        coinbase_broker: Any | None = None,
        uw_client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.logger = logger.getChild("stock_strategy")
        self.ai_service = ai_service
        self.confirmation_mode = confirmation_mode
        self.coinbase_broker = coinbase_broker
        self.uw_client = uw_client

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
        force: bool = False,
    ) -> dict[str, Any]:
        if not force and self._should_skip(state, now):
            next_run = datetime.fromisoformat(state.last_stock_screen_date) + timedelta(days=self.config.run_interval_days)
            self.logger.info("Skipping stock screen because the weekly interval has not elapsed.")
            return {
                "status": "skipped",
                "reason": "weekly_interval_not_elapsed",
                "next_eligible_date": next_run.date().isoformat(),
            }

        constituents = self.client.get_sp500_constituents()
        ratios = self.client.get_ratios_ttm_bulk()
        key_metrics = self.client.get_key_metrics_ttm_bulk()
        vix_value = self.client.get_vix_price(self.config.vix_symbol)

        fundamentals_by_symbol: dict[str, StockFundamentals] = {}
        missing_count = 0
        for symbol in constituents:
            merged = {**key_metrics.get(symbol, {}), **ratios.get(symbol, {})}
            fundamentals = self._build_fundamentals(symbol, merged)
            if fundamentals is None:
                missing_count += 1
                continue
            fundamentals_by_symbol[symbol] = fundamentals

        qualified = [item for item in fundamentals_by_symbol.values() if self._qualifies(item)]
        ranked = self._rank_candidates(qualified)
        top_targets = ranked[: self.config.max_positions]
        current_stock_positions = [
            symbol for symbol, position in state.positions.items() if position.asset_type == "stock"
        ]
        quote_symbols = sorted({*current_stock_positions, *[item.symbol for item in top_targets]})
        quotes = self.client.get_batch_quotes(quote_symbols) if quote_symbols else {}
        broker.mark_positions(state, quotes)

        sells: list[str] = []
        for symbol in list(current_stock_positions):
            fundamentals = fundamentals_by_symbol.get(symbol)
            if fundamentals is None or not self._qualifies(fundamentals):
                close_price = quotes.get(symbol, state.last_prices.get(symbol, state.positions[symbol].current_price))
                # Execute sell on Coinbase
                if self.coinbase_broker is not None:
                    qty = state.positions[symbol].quantity
                    self.coinbase_broker.place_stock_sell(
                        symbol=symbol, quantity=qty, reason="fundamental_threshold_breach",
                    )
                broker.close_position(
                    state,
                    symbol=symbol,
                    price=close_price,
                    reason="fundamental_threshold_breach",
                )
                sells.append(symbol)

        buys: list[str] = []
        stock_positions_after_sells = {
            symbol for symbol, position in state.positions.items() if position.asset_type == "stock"
        }
        if vix_value > self.config.vix_buy_threshold and risk_manager.can_open_new_position(state):
            for fundamentals in top_targets:
                if len(stock_positions_after_sells) >= self.config.max_positions:
                    break
                if fundamentals.symbol in stock_positions_after_sells:
                    continue
                price = quotes.get(fundamentals.symbol)
                if price is None:
                    self.logger.warning("Skipping stock buy for %s because no quote was returned.", fundamentals.symbol)
                    continue
                equity = broker.portfolio_equity(state)
                notional = risk_manager.position_notional(state, equity)
                if notional <= 0:
                    self.logger.warning("Skipping stock buy for %s because position size is zero.", fundamentals.symbol)
                    continue

                # Unusual Whales market intelligence (optional)
                uw_signals: dict[str, Any] | None = None
                if self.uw_client is not None:
                    uw_signals = self.uw_client.get_stock_signals(fundamentals.symbol)
                    self.logger.info(
                        "UW signals for %s: flow=%d options, dp=%d blocks, "
                        "insider=%s, congress=%s",
                        fundamentals.symbol,
                        uw_signals["options_flow"]["flow_count"],
                        uw_signals["darkpool"]["recent_trades"],
                        uw_signals["insider"]["net_sentiment"],
                        uw_signals["congress"]["net_sentiment"],
                    )

                # AI confirmation (optional)
                ai_meta: dict[str, Any] = {}
                if self.ai_service is not None and apply_ai_confirmation is not None:
                    verdict = self.ai_service.evaluate_stock(
                        symbol=fundamentals.symbol,
                        fundamentals=fundamentals.to_dict(),
                        vix_value=vix_value,
                        uw_signals=uw_signals,
                    )
                    notional, ai_meta = apply_ai_confirmation(
                        verdict, self.confirmation_mode, notional, self.logger,
                    )
                    if notional <= 0:
                        self.logger.info("AI rejected stock buy for %s.", fundamentals.symbol)
                        continue

                # Execute buy on Coinbase
                if self.coinbase_broker is not None:
                    self.coinbase_broker.place_stock_buy(
                        symbol=fundamentals.symbol, notional_usd=notional,
                    )

                position = broker.open_position(
                    state,
                    symbol=fundamentals.symbol,
                    asset_type="stock",
                    strategy="value_investing",
                    price=price,
                    notional=notional,
                    opened_at=now,
                    meta={
                        "fundamentals": fundamentals.to_dict(),
                        "vix_value": vix_value,
                        **({"uw_signals": uw_signals} if uw_signals else {}),
                        **ai_meta,
                    },
                )
                if position is not None:
                    stock_positions_after_sells.add(fundamentals.symbol)
                    buys.append(fundamentals.symbol)
        else:
            self.logger.info(
                "Stock buys blocked. vix=%.2f threshold=%.2f daily_halt=%s kill_switch=%s",
                vix_value,
                self.config.vix_buy_threshold,
                state.halt_new_trades_for_day,
                state.kill_switch_engaged,
            )

        state.last_stock_screen_date = now.date().isoformat()
        preview = [item.to_dict() for item in top_targets[:5]]
        self.logger.info(
            "Stock screen complete. vix=%.2f qualified=%d missing=%d buys=%s sells=%s top_targets=%s",
            vix_value,
            len(qualified),
            missing_count,
            buys,
            sells,
            preview,
        )
        return {
            "status": "completed",
            "vix_value": vix_value,
            "qualified_count": len(qualified),
            "missing_metric_count": missing_count,
            "top_targets": [item.symbol for item in top_targets],
            "buys": buys,
            "sells": sells,
        }

    def _should_skip(self, state: PortfolioState, now: datetime) -> bool:
        if not state.last_stock_screen_date:
            return False
        last_run = datetime.fromisoformat(state.last_stock_screen_date)
        return now.date() < (last_run + timedelta(days=self.config.run_interval_days)).date()

    def _build_fundamentals(self, symbol: str, payload: dict[str, Any]) -> StockFundamentals | None:
        pe_ratio = _as_float(payload, self.PE_KEYS)
        pb_ratio = _as_float(payload, self.PB_KEYS)
        debt_to_equity = _as_float(payload, self.DEBT_TO_EQUITY_KEYS)
        pfcf_ratio = _as_float(payload, self.PFCF_KEYS)
        direct_positive_fcf = _as_float(payload, self.POSITIVE_FCF_KEYS)

        if pe_ratio is None or pb_ratio is None or debt_to_equity is None:
            return None

        fcf_positive_signal = 0.0
        if direct_positive_fcf is not None:
            fcf_positive_signal = direct_positive_fcf
        elif pfcf_ratio is not None:
            fcf_positive_signal = 1.0 if pfcf_ratio > 0 else -1.0

        return StockFundamentals(
            symbol=symbol,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            debt_to_equity=debt_to_equity,
            fcf_positive_signal=fcf_positive_signal,
            pfcf_ratio=pfcf_ratio,
        )

    def _qualifies(self, fundamentals: StockFundamentals) -> bool:
        return (
            fundamentals.pe_ratio < self.config.max_pe_ratio
            and fundamentals.pb_ratio < self.config.max_pb_ratio
            and fundamentals.debt_to_equity < self.config.max_debt_to_equity
            and fundamentals.fcf_positive_signal > self.config.min_positive_fcf_signal
        )

    def _rank_candidates(self, candidates: list[StockFundamentals]) -> list[StockFundamentals]:
        if not candidates:
            return []
        scores = {candidate.symbol: 0 for candidate in candidates}
        metric_sorts = [
            sorted(candidates, key=lambda item: item.pe_ratio),
            sorted(candidates, key=lambda item: item.pb_ratio),
            sorted(candidates, key=lambda item: item.debt_to_equity),
            sorted(candidates, key=lambda item: item.pfcf_ratio if item.pfcf_ratio is not None else float("inf")),
        ]
        for metric_ranked_list in metric_sorts:
            for index, item in enumerate(metric_ranked_list, start=1):
                scores[item.symbol] += index
        return sorted(
            candidates,
            key=lambda item: (
                scores[item.symbol],
                item.pe_ratio,
                item.pb_ratio,
                item.debt_to_equity,
                item.pfcf_ratio if item.pfcf_ratio is not None else float("inf"),
            ),
        )

