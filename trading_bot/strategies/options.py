"""Directional options strategy using Fear & Greed as the entry signal.

Entry: extreme fear (F&G < threshold) → buy deep-ITM calls;
       extreme greed (F&G > threshold) → buy deep-ITM puts.
Contract filter: |delta| >= min_delta (default 0.70) so we trade
high-conviction, stock-like contracts with low theta decay.

Price discovery uses yfinance (paper mode) or the Robinhood API (live mode).
Order execution uses Robinhood only when live_trading_enabled=True.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients import AlternativeMeClient
from trading_bot.clients_robinhood import RobinhoodClient
from trading_bot.config import OptionsStrategyConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return default if f != f else f  # f != f is True only for IEEE-754 NaN
    except (TypeError, ValueError):
        return default


class OptionsStrategy:
    _ASSET_TYPE = "option"
    _STRATEGY_NAME = "options_fg"
    _POS_PREFIX = "OPT"

    def __init__(
        self,
        *,
        config: OptionsStrategyConfig,
        robinhood_client: RobinhoodClient,
        alternative_client: AlternativeMeClient,
        logger: logging.Logger,
        live_trading_enabled: bool = False,
    ) -> None:
        self.config = config
        self.robinhood = robinhood_client
        self.alternative_client = alternative_client
        self.logger = logger.getChild("options_strategy")
        self.live_trading_enabled = live_trading_enabled

    # ── Public entry point ─────────────────────────────────────────────

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"status": "disabled", "opens": [], "closes": []}

        if state.kill_switch_engaged:
            return {"status": "halted", "reason": "kill_switch", "opens": [], "closes": []}
        if state.halt_new_trades_for_day:
            return {"status": "halted", "reason": "daily_halt", "opens": [], "closes": []}

        # Manage existing positions before entering new ones
        closes = self._manage_positions(state, broker, now)

        # Fear & Greed drives directional bias
        try:
            fg_reading = self.alternative_client.get_fear_and_greed()
            fg_value: int = fg_reading.value
        except Exception as exc:
            self.logger.warning("Fear & Greed unavailable: %s", exc)
            return {
                "status": "error",
                "error": f"Fear & Greed unavailable: {exc}",
                "opens": [],
                "closes": closes,
            }

        self.logger.info("Options strategy: Fear & Greed = %d", fg_value)

        if fg_value < self.config.buy_call_fear_threshold:
            option_type = "call"
            signal_reason = "extreme_fear_buy_calls"
        elif fg_value > self.config.buy_put_fear_threshold:
            option_type = "put"
            signal_reason = "extreme_greed_buy_puts"
        else:
            return {
                "status": "completed",
                "fear_and_greed": fg_value,
                "opens": [],
                "closes": closes,
                "reason": "no_signal",
            }

        opens: list[str] = []
        decisions: list[dict[str, Any]] = []

        existing_options = [
            sym for sym, pos in state.positions.items()
            if pos.asset_type == self._ASSET_TYPE and pos.strategy == self._STRATEGY_NAME
        ]

        for symbol in self.config.symbols:
            if len(existing_options) + len(opens) >= self.config.max_positions:
                break
            if not risk_manager.can_open_new_position(state):
                self.logger.info("Options: risk manager blocked new position for %s", symbol)
                break

            # Only one position per underlying at a time
            if any(
                pos_sym.startswith(f"{self._POS_PREFIX}:{symbol}:")
                for pos_sym in state.positions
            ):
                decisions.append({
                    "strategy": self._STRATEGY_NAME,
                    "symbol": symbol,
                    "action": "skip_buy",
                    "reason": "already_have_position",
                    "signals": {"fear_greed": fg_value},
                })
                continue

            contract = self._find_best_contract(symbol, option_type, now)
            if contract is None:
                decisions.append({
                    "strategy": self._STRATEGY_NAME,
                    "symbol": symbol,
                    "action": "skip_buy",
                    "reason": "no_qualifying_contract",
                    "signals": {
                        "fear_greed": fg_value,
                        "option_type": option_type,
                        "min_delta": self.config.min_delta,
                    },
                })
                continue

            equity = broker.portfolio_equity(state)
            per_position_capital = (
                equity * self.config.capital_allocation_pct / max(1, self.config.max_positions)
            )

            mark = contract["mark"]
            if mark <= 0:
                continue

            # Each standard option contract controls 100 shares
            contracts_count = int(per_position_capital / (mark * 100))
            if contracts_count <= 0:
                self.logger.info(
                    "Options: skipping %s — contract cost ($%.2f) exceeds per-position capital ($%.2f)",
                    symbol, mark * 100, per_position_capital,
                )
                continue
            notional = contracts_count * mark * 100
            available_cash = (
                self.robinhood.get_account_buying_power()
                if self.live_trading_enabled and self.robinhood.can_trade
                else state.cash
            )
            if notional > available_cash:
                self.logger.info(
                    "Options: insufficient cash for %s (need $%.2f, have $%.2f)",
                    symbol, notional, available_cash,
                )
                continue
            limit_price = contract["ask"] if contract["ask"] > 0 else mark * 1.02

            order_placed = False
            if self.live_trading_enabled:
                order = self.robinhood.place_option_buy_order(
                    symbol=symbol,
                    expiration_date=contract["expiration_date"],
                    strike=contract["strike"],
                    option_type=option_type,
                    quantity=contracts_count,
                    limit_price=round(limit_price, 2),
                )
                order_placed = order is not None
                if not order_placed:
                    # Live order rejected — do not record a position we don't own
                    self.logger.warning("Options: live BUY order failed for %s; skipping bookkeeping.", symbol)
                    continue

            pos_symbol = self._pos_symbol(
                symbol, contract["expiration_date"], contract["strike"], option_type
            )
            pos = broker.open_position(
                state,
                symbol=pos_symbol,
                asset_type=self._ASSET_TYPE,
                strategy=self._STRATEGY_NAME,
                price=limit_price,
                notional=notional,
                opened_at=now,
                meta={
                    "underlying": symbol,
                    "expiration_date": contract["expiration_date"],
                    "strike": contract["strike"],
                    "option_type": option_type,
                    "contracts": contracts_count,
                    "entry_delta": contract.get("delta"),
                    "entry_iv": contract.get("implied_volatility"),
                    "signal": signal_reason,
                    "fear_greed": fg_value,
                    "venue": "robinhood",
                    "order_placed": order_placed,
                },
            )
            if pos is not None:
                opens.append(pos_symbol)
                decisions.append({
                    "strategy": self._STRATEGY_NAME,
                    "symbol": pos_symbol,
                    "action": "buy",
                    "reason": signal_reason,
                    "signals": {
                        "fear_greed": fg_value,
                        "option_type": option_type,
                        "expiration_date": contract["expiration_date"],
                        "strike": contract["strike"],
                        "delta": contract.get("delta"),
                        "mark_price": round(mark, 4),
                        "limit_price": round(limit_price, 4),
                        "contracts": contracts_count,
                        "notional": round(notional, 2),
                        "live_order": order_placed,
                    },
                })

        self.logger.info(
            "Options: F&G=%d %s opens=%s closes=%s",
            fg_value, option_type, opens, closes,
        )
        return {
            "status": "completed",
            "fear_and_greed": fg_value,
            "option_type": option_type,
            "signal_reason": signal_reason,
            "opens": opens,
            "closes": closes,
            "decisions": decisions,
        }

    # ── Position management ────────────────────────────────────────────

    def _manage_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        now: datetime,
    ) -> list[str]:
        closes: list[str] = []
        for sym, pos in list(state.positions.items()):
            if pos.asset_type != self._ASSET_TYPE or pos.strategy != self._STRATEGY_NAME:
                continue

            # Refresh mark price from yfinance
            refreshed = self._get_current_option_price(pos)
            if refreshed is not None and refreshed > 0:
                state.last_prices[sym] = refreshed
                pos.current_price = refreshed

            entry = pos.entry_price
            current = pos.current_price
            if entry <= 0 or current is None or current <= 0:
                continue

            pnl_pct = (current - entry) / entry
            expiration_date = pos.meta.get("expiration_date", "")
            dte = self._days_to_expiration(expiration_date, now) if expiration_date else 999

            close_reason: str | None = None
            if pnl_pct >= self.config.take_profit_pct:
                close_reason = "take_profit"
            elif pnl_pct <= -self.config.stop_loss_pct:
                close_reason = "stop_loss"
            elif dte < self.config.close_at_dte:
                close_reason = "dte_expiry"

            if close_reason is None:
                continue

            if self.live_trading_enabled:
                meta = pos.meta or {}
                sell_order = self.robinhood.place_option_sell_order(
                    symbol=str(meta.get("underlying", "")),
                    expiration_date=str(meta.get("expiration_date", "")),
                    strike=str(meta.get("strike", "")),
                    option_type=str(meta.get("option_type", "call")),
                    quantity=max(1, int(meta.get("contracts", 1))),
                    limit_price=max(0.01, round(current * 0.98, 2)),
                )
                if sell_order is None:
                    # Live sell failed — keep the paper position to retry next cycle
                    self.logger.warning(
                        "Options: live SELL order failed for %s (%s); retaining paper position.",
                        sym, close_reason,
                    )
                    continue

            broker.close_position(state, symbol=sym, price=current, reason=close_reason)
            closes.append(sym)
            self.logger.info(
                "Closed options position %s: reason=%s pnl=%.2f%% dte=%d",
                sym, close_reason, pnl_pct * 100, dte,
            )

        return closes

    # ── Contract discovery ─────────────────────────────────────────────

    def _find_best_contract(
        self,
        symbol: str,
        option_type: str,
        now: datetime,
    ) -> dict[str, Any] | None:
        """Find the best deep-ITM contract (|delta| >= min_delta).

        Prefers Robinhood (exact delta) when live; falls back to yfinance
        with a moneyness proxy when in paper mode or when credentials are absent.
        """
        if self.live_trading_enabled and self.robinhood.can_trade:
            return self._find_contract_robinhood(symbol, option_type, now)
        return self._find_contract_yfinance(symbol, option_type, now)

    def _find_contract_robinhood(
        self,
        symbol: str,
        option_type: str,
        now: datetime,
    ) -> dict[str, Any] | None:
        """Use the Robinhood API for exact delta filtering."""
        exp_dates = self.robinhood.get_expiration_dates(symbol)
        today = now.date()
        qualifying = [
            d for d in exp_dates
            if self.config.min_dte <= (date.fromisoformat(d) - today).days <= self.config.max_dte
        ]
        if not qualifying:
            self.logger.info("No %d-%d DTE expirations for %s on Robinhood", self.config.min_dte, self.config.max_dte, symbol)
            return None

        for exp_date in qualifying:
            contracts = self.robinhood.find_options(symbol, exp_date, option_type)
            best = self._select_by_delta(contracts, option_type)
            if best is not None:
                return best

        return None

    def _select_by_delta(
        self,
        contracts: list[dict[str, Any]],
        option_type: str,
    ) -> dict[str, Any] | None:
        """From a Robinhood options list, pick the contract closest to min_delta
        that still satisfies |delta| >= min_delta."""
        candidates: list[tuple[float, dict[str, Any]]] = []
        for c in contracts:
            delta = _safe_float(c.get("delta"))
            abs_delta = abs(delta)
            if abs_delta < self.config.min_delta:
                continue
            iv = _safe_float(c.get("implied_volatility"))
            if iv > self.config.max_iv:
                continue
            volume = _safe_float(c.get("volume"))
            oi = _safe_float(c.get("open_interest"))
            if volume < self.config.min_volume or oi < self.config.min_open_interest:
                continue
            candidates.append((abs_delta, c))

        if not candidates:
            return None

        # Prefer the contract closest to min_delta (least expensive deep-ITM)
        candidates.sort(key=lambda t: t[0])
        _, contract = candidates[0]

        bid = _safe_float(contract.get("bid_price"))
        ask = _safe_float(contract.get("ask_price"))
        mark = _safe_float(contract.get("adjusted_mark_price"))
        if mark <= 0:
            mark = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        if mark <= 0:
            return None

        return {
            "expiration_date": contract.get("expiration_date", ""),
            "strike": str(_safe_float(contract.get("strike_price"))),
            "bid": bid,
            "ask": ask,
            "mark": mark,
            "delta": _safe_float(contract.get("delta")),
            "implied_volatility": _safe_float(contract.get("implied_volatility")),
            "volume": _safe_float(contract.get("volume")),
            "open_interest": _safe_float(contract.get("open_interest")),
        }

    def _find_contract_yfinance(
        self,
        symbol: str,
        option_type: str,
        now: datetime,
    ) -> dict[str, Any] | None:
        """Use yfinance for paper-mode contract discovery.

        Delta is approximated by moneyness: call with strike <= spot*0.90
        or put with strike >= spot*1.10 corresponds to |delta| >= ~0.70.
        """
        if not _YF_AVAILABLE:
            self.logger.warning("yfinance not available; cannot discover option contracts.")
            return None
        try:
            ticker = yf.Ticker(symbol)
            exp_dates = ticker.options
            if not exp_dates:
                self.logger.info("No options available for %s on yfinance.", symbol)
                return None

            today = now.date()
            qualifying = [
                d for d in exp_dates
                if self.config.min_dte <= (date.fromisoformat(d) - today).days <= self.config.max_dte
            ]
            if not qualifying:
                return None

            target_date = qualifying[0]
            chain = ticker.option_chain(target_date)
            df = chain.calls if option_type == "call" else chain.puts
            if df.empty:
                return None

            fast_info = ticker.fast_info
            spot = _safe_float(getattr(fast_info, "last_price", None))
            if spot <= 0:
                return None

            # Approximate |delta| >= min_delta using moneyness.
            # A call/put delta of X corresponds to ~(X-0.50)/2 fraction ITM.
            # e.g. min_delta=0.70 → itm_pct=0.10 → calls ≤ 90% spot, puts ≥ 110% spot.
            itm_pct = (self.config.min_delta - 0.50) / 2.0
            if option_type == "call":
                itm_mask = df["strike"] <= spot * (1.0 - itm_pct)
            else:
                itm_mask = df["strike"] >= spot * (1.0 + itm_pct)

            df_itm = df[itm_mask].copy()

            # Apply volume / OI / IV filters
            df_itm = df_itm[df_itm["volume"].fillna(0) >= self.config.min_volume]
            df_itm = df_itm[df_itm["openInterest"].fillna(0) >= self.config.min_open_interest]
            df_itm = df_itm[df_itm["impliedVolatility"].fillna(99) <= self.config.max_iv]

            if df_itm.empty:
                self.logger.info(
                    "No deep-ITM %s contracts for %s on %s after filters.",
                    option_type, symbol, target_date,
                )
                return None

            # Pick the strike closest to the min_delta threshold (least expensive ITM)
            if option_type == "call":
                best_row = df_itm.sort_values("strike", ascending=False).iloc[0]
            else:
                best_row = df_itm.sort_values("strike", ascending=True).iloc[0]

            bid = _safe_float(best_row.get("bid"))
            ask = _safe_float(best_row.get("ask"))
            mark = (bid + ask) / 2.0 if bid > 0 and ask > 0 else _safe_float(best_row.get("lastPrice"))
            if mark <= 0:
                return None

            return {
                "expiration_date": target_date,
                "strike": str(float(best_row["strike"])),
                "bid": bid,
                "ask": ask,
                "mark": mark,
                "delta": None,  # not provided by yfinance
                "implied_volatility": _safe_float(best_row.get("impliedVolatility")),
                "volume": _safe_float(best_row.get("volume")),
                "open_interest": _safe_float(best_row.get("openInterest")),
            }
        except Exception as exc:
            self.logger.warning("yfinance contract discovery failed for %s %s: %s", symbol, option_type, exc)
            return None

    # ── Price refresh for open positions ──────────────────────────────

    def get_current_option_price(self, position: Any) -> float | None:
        """Public wrapper so service.py can call this during refresh cycles."""
        return self._get_current_option_price(position)

    def _get_current_option_price(self, position: Any) -> float | None:
        """Fetch current mark price. Uses Robinhood (live, real-time) or yfinance (paper, delayed)."""
        meta = position.meta or {}
        underlying = meta.get("underlying", "")
        expiration_date = meta.get("expiration_date", "")
        strike_str = meta.get("strike", "")
        option_type = meta.get("option_type", "call")
        if not (underlying and expiration_date and strike_str):
            return None

        # Live mode: use Robinhood for real-time pricing (avoids 15-min yfinance delay)
        if self.live_trading_enabled and self.robinhood.can_trade:
            try:
                data = self.robinhood.get_option_market_data(
                    symbol=underlying,
                    expiration_date=expiration_date,
                    strike=strike_str,
                    option_type=option_type,
                )
                if data:
                    mark = _safe_float(data.get("adjusted_mark_price"))
                    if mark > 0:
                        return mark
                    bid = _safe_float(data.get("bid_price"))
                    ask = _safe_float(data.get("ask_price"))
                    if bid > 0 and ask > 0:
                        return (bid + ask) / 2.0
            except Exception as exc:
                self.logger.warning("Robinhood price fetch failed for %s; falling back to yfinance: %s", underlying, exc)

        # Paper mode (or Robinhood fallback): use yfinance
        if not _YF_AVAILABLE:
            return None
        try:
            target_strike = float(strike_str)
            ticker = yf.Ticker(underlying)
            chain = ticker.option_chain(expiration_date)
            df = chain.calls if option_type == "call" else chain.puts
            if df.empty:
                return None

            row = df[df["strike"] == target_strike]
            if row.empty:
                df_c = df.copy()
                df_c["_dist"] = abs(df_c["strike"] - target_strike)
                row = df_c.sort_values("_dist").head(1)
            if row.empty:
                return None

            bid = _safe_float(row.iloc[0].get("bid"))
            ask = _safe_float(row.iloc[0].get("ask"))
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            last = _safe_float(row.iloc[0].get("lastPrice"))
            return last if last > 0 else None
        except Exception:
            return None

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _pos_symbol(symbol: str, expiration_date: str, strike: str, option_type: str) -> str:
        exp_compact = expiration_date.replace("-", "")
        type_char = option_type[0].upper()  # C or P
        return f"OPT:{symbol}:{exp_compact}:{strike}:{type_char}"

    @staticmethod
    def _days_to_expiration(expiration_date: str, now: datetime) -> int:
        try:
            return max(0, (date.fromisoformat(expiration_date) - now.date()).days)
        except (ValueError, TypeError):
            return 0
