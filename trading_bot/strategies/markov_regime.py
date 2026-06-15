from __future__ import annotations

"""Markov Chain regime-switching trading strategy.

Uses a 3-state (Bull / Bear / Sideways) discrete-time Markov chain fit to
daily price history to forecast the probability of being in a bull regime
at the next time step.

    signal = P(bull | current_state) − P(bear | current_state)

* signal > entry_signal_threshold  → open long positions
* signal < exit_signal_threshold   → close existing positions
* otherwise                        → hold (no new entries)

The regime is estimated from `config.regime_symbol` price history (stored
in PortfolioState.price_history) and the signal is applied to all symbols
in `config.symbols`.
"""

import logging
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients import BinanceFuturesClient, CoinbaseClient
from trading_bot.coinbase_broker import CoinbaseBroker
from trading_bot.config import MarkovStrategyConfig
from trading_bot.markov import MarkovResult, STATE_NAMES, compute_markov_signal
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager

_STRATEGY_NAME = "markov_regime"


class MarkovRegimeStrategy:
    """Regime-switching strategy driven by Markov chain transition probabilities."""

    def __init__(
        self,
        *,
        config: MarkovStrategyConfig,
        coinbase_client: CoinbaseClient | None = None,
        binance_client: BinanceFuturesClient | None = None,
        coinbase_broker: CoinbaseBroker | None = None,
        logger: logging.Logger,
        protected_symbols: frozenset[str] = frozenset(),
    ) -> None:
        self.config = config
        self.coinbase_client = coinbase_client
        self.binance_client = binance_client
        self.coinbase_broker = coinbase_broker
        self.protected_symbols = protected_symbols
        self.logger = logger.getChild("markov_regime")

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
    ) -> dict[str, Any]:
        """Single strategy cycle. Returns a summary dict."""

        if not self.config.enabled:
            return {"status": "disabled", "opens": [], "closes": [], "decisions": []}

        opens: list[str] = []
        closes: list[str] = []
        decisions: list[dict[str, Any]] = []

        # 1. Fetch and mark current prices
        prices = self._fetch_prices()
        if prices:
            broker.mark_positions(state, prices, now=now)

        # 2. Retrieve regime price history
        regime_prices = state.price_history.get(self.config.regime_symbol, [])

        # 3. Guard: insufficient history
        if len(regime_prices) < self.config.min_history_days:
            self.logger.info(
                "Markov: insufficient history for %s (%d/%d days). Skipping.",
                self.config.regime_symbol,
                len(regime_prices),
                self.config.min_history_days,
            )
            # Clear stale regime state so downstream gates don't block on a
            # previously-computed bear signal when history is unavailable.
            state.markov_regime_state = ""
            state.markov_regime_signal = 0.0
            return {
                "status": "insufficient_data",
                "regime_symbol": self.config.regime_symbol,
                "history_days": len(regime_prices),
                "min_history_days": self.config.min_history_days,
                "opens": opens,
                "closes": closes,
                "decisions": decisions,
            }

        # 4. Compute Markov signal
        markov = compute_markov_signal(
            regime_prices,
            return_window=self.config.return_window,
            bull_threshold=self.config.bull_threshold,
            bear_threshold=self.config.bear_threshold,
            forecast_steps=self.config.forecast_steps,
            laplace_alpha=self.config.laplace_alpha,
        )

        if markov is None:
            self.logger.warning("Markov: compute_markov_signal returned None despite history check.")
            state.markov_regime_state = ""
            state.markov_regime_signal = 0.0
            return {
                "status": "computation_failed",
                "opens": opens,
                "closes": closes,
                "decisions": decisions,
            }

        self.logger.info(
            "Markov: state=%s signal=%.4f bull=%.3f bear=%.3f sideways=%.3f transitions=%d",
            markov.current_state_name,
            markov.signal,
            markov.bull_prob,
            markov.bear_prob,
            markov.sideways_prob,
            markov.n_transitions,
        )

        # Publish regime signal to state so other strategies (Polymarket, Kalshi)
        # can use it as a market-wide regime gate.
        state.markov_regime_state = markov.current_state_name
        state.markov_regime_signal = markov.signal

        # 5. Autonomous exits: stop-loss / take-profit
        # Track which symbols were just closed so we don't immediately re-enter
        just_closed: set[str] = self._handle_autonomous_exits(
            state,
            broker=broker,
            prices=prices,
            markov=markov,
            closes=closes,
            decisions=decisions,
        )

        # 6. Regime-based exit: signal turned bearish
        if markov.signal < self.config.exit_signal_threshold:
            self._close_all_positions(
                state,
                broker=broker,
                prices=prices,
                markov=markov,
                reason="regime_exit",
                closes=closes,
                decisions=decisions,
            )

        # 7. Entry: signal is sufficiently bullish
        elif markov.signal > self.config.entry_signal_threshold:
            if not risk_manager.can_open_new_position(state):
                self.logger.info("Markov: buys blocked by risk guardrails.")
                decisions.append({
                    "strategy": _STRATEGY_NAME,
                    "symbol": "*",
                    "action": "skip_buy",
                    "reason": "risk_guardrails",
                    "signals": _signal_dict(markov),
                })
            else:
                equity = broker.portfolio_equity(state)
                for symbol in self.config.symbols:
                    if symbol in state.positions:
                        holder = state.positions[symbol].strategy
                        if holder != _STRATEGY_NAME:
                            self.logger.info(
                                "Markov: skipping %s buy — position already held by strategy %s",
                                symbol,
                                holder,
                            )
                            decisions.append({
                                "strategy": _STRATEGY_NAME,
                                "symbol": symbol,
                                "action": "skip_buy",
                                "reason": "held_by_other_strategy",
                                "signals": {**_signal_dict(markov), "holder": holder},
                            })
                        else:
                            decisions.append({
                                "strategy": _STRATEGY_NAME,
                                "symbol": symbol,
                                "action": "skip_buy",
                                "reason": "already_open",
                                "signals": _signal_dict(markov),
                            })
                        continue
                    if symbol in just_closed:
                        decisions.append({
                            "strategy": _STRATEGY_NAME,
                            "symbol": symbol,
                            "action": "skip_buy",
                            "reason": "just_closed_this_cycle",
                            "signals": _signal_dict(markov),
                        })
                        continue
                    notional = risk_manager.position_notional(state, equity, symbol=symbol)
                    if notional <= 0:
                        decisions.append({
                            "strategy": _STRATEGY_NAME,
                            "symbol": symbol,
                            "action": "skip_buy",
                            "reason": "position_size_zero",
                            "signals": _signal_dict(markov),
                        })
                        continue
                    price = prices.get(symbol)
                    if not price or price <= 0:
                        self.logger.warning("Markov: no price for %s, skipping.", symbol)
                        continue
                    position = broker.open_position(
                        state,
                        symbol=symbol,
                        asset_type="crypto",
                        strategy=_STRATEGY_NAME,
                        price=price,
                        notional=notional,
                        opened_at=now,
                        meta={
                            "markov_signal": round(markov.signal, 6),
                            "markov_state": markov.current_state_name,
                            "bull_prob": round(markov.bull_prob, 4),
                            "bear_prob": round(markov.bear_prob, 4),
                            "n_transitions": markov.n_transitions,
                        },
                    )
                    if position is not None:
                        opens.append(symbol)
                        decisions.append({
                            "strategy": _STRATEGY_NAME,
                            "symbol": symbol,
                            "action": "buy",
                            "reason": "markov_bull_signal",
                            "signals": {
                                **_signal_dict(markov),
                                "price": round(price, 4),
                                "notional": round(notional, 2),
                            },
                        })

        # 8. Neutral: hold, no new entries
        else:
            decisions.append({
                "strategy": _STRATEGY_NAME,
                "symbol": "*",
                "action": "hold",
                "reason": "neutral_signal",
                "signals": _signal_dict(markov),
            })

        return {
            "status": "completed",
            "regime_symbol": self.config.regime_symbol,
            "current_state": markov.current_state_name,
            "signal": round(markov.signal, 6),
            "bull_prob": round(markov.bull_prob, 4),
            "bear_prob": round(markov.bear_prob, 4),
            "sideways_prob": round(markov.sideways_prob, 4),
            "stationary_distribution": {
                STATE_NAMES[i]: round(v, 4)
                for i, v in enumerate(markov.stationary_distribution)
            },
            "n_transitions": markov.n_transitions,
            "history_days": markov.history_days,
            "opens": opens,
            "closes": closes,
            "buys": opens,
            "sells": closes,
            "decisions": decisions,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _fetch_prices(self) -> dict[str, float]:
        all_symbols = list({self.config.regime_symbol} | set(self.config.symbols))
        prices: dict[str, float] = {}

        if self.config.use_coinbase_pricing and self.coinbase_client is not None:
            try:
                prices = self.coinbase_client.get_spot_prices(all_symbols)
            except Exception as exc:
                self.logger.warning("Markov: Coinbase price fetch failed: %s", exc)

        # Fill any missing symbols from Binance (covers partial Coinbase responses)
        if self.binance_client is not None:
            for symbol in all_symbols:
                if symbol not in prices:
                    try:
                        snap = self.binance_client.get_mark_price_and_funding(symbol)
                        prices[symbol] = snap.mark_price
                    except Exception as exc:
                        self.logger.warning("Markov: Binance price fetch failed for %s: %s", symbol, exc)

        return prices

    def _handle_autonomous_exits(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        prices: dict[str, float],
        markov: MarkovResult,
        closes: list[str],
        decisions: list[dict[str, Any]],
    ) -> set[str]:
        """Close positions that hit stop-loss or take-profit. Returns the set of closed symbols."""
        just_closed: set[str] = set()
        tp = self.config.take_profit_pct
        sl = self.config.stop_loss_pct
        if tp <= 0 and sl <= 0:
            return just_closed

        for symbol in list(self.config.symbols):
            bare = symbol.replace("USDT", "").replace("USD", "").replace("-", "")
            if symbol in self.protected_symbols or bare in self.protected_symbols:
                continue
            position = state.positions.get(symbol)
            if position is None or position.strategy != _STRATEGY_NAME:
                continue
            price = prices.get(symbol)
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

            broker.close_position(state, symbol=symbol, price=price, reason=hit_reason)
            closes.append(symbol)
            just_closed.add(symbol)
            decisions.append({
                "strategy": _STRATEGY_NAME,
                "symbol": symbol,
                "action": "sell",
                "reason": hit_reason,
                "signals": {
                    **_signal_dict(markov),
                    "entry_price": round(position.entry_price, 4),
                    "exit_price": round(price, 4),
                    "move_pct": round(move_pct, 4),
                },
            })

        return just_closed

    def _close_all_positions(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        prices: dict[str, float],
        markov: MarkovResult,
        reason: str,
        closes: list[str],
        decisions: list[dict[str, Any]],
    ) -> None:
        for symbol in list(self.config.symbols):
            bare = symbol.replace("USDT", "").replace("USD", "").replace("-", "")
            if symbol in self.protected_symbols or bare in self.protected_symbols:
                self.logger.info("Protected: skipping %s close (%s).", symbol, reason)
                continue
            position = state.positions.get(symbol)
            if position is None or position.strategy != _STRATEGY_NAME:
                continue
            price = prices.get(symbol, position.current_price)
            broker.close_position(state, symbol=symbol, price=price, reason=reason)
            closes.append(symbol)
            decisions.append({
                "strategy": _STRATEGY_NAME,
                "symbol": symbol,
                "action": "sell",
                "reason": reason,
                "signals": _signal_dict(markov),
            })


def _signal_dict(markov: MarkovResult) -> dict[str, Any]:
    return {
        "current_state": markov.current_state_name,
        "signal": round(markov.signal, 6),
        "bull_prob": round(markov.bull_prob, 4),
        "bear_prob": round(markov.bear_prob, 4),
        "n_transitions": markov.n_transitions,
    }
