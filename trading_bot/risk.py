from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from typing import Any

from trading_bot.config import RiskConfig, VolatilityConfig
from trading_bot.models import PortfolioState
from trading_bot.volatility import VolatilityEngine

logger = logging.getLogger(__name__)


def markov_regime_blocks_entry(
    state: PortfolioState,
    gate_enabled: bool,
    strategy_name: str,
) -> dict[str, Any] | None:
    """Return a skip_buy decision if the Markov bear gate should block new entries, else None.

    Returns None when the gate is disabled, when no regime has been computed yet
    (markov_regime_state == ""), or when the signal is non-negative (neutral/bull).
    """
    if not gate_enabled:
        return None
    if not state.markov_regime_state:
        return None
    if state.markov_regime_signal >= 0:
        return None
    return {
        "strategy": strategy_name,
        "symbol": "*",
        "action": "skip_buy",
        "reason": "markov_bear_regime",
        "signals": {
            "markov_state": state.markov_regime_state,
            "markov_signal": state.markov_regime_signal,
        },
    }


@dataclass(frozen=True)
class RiskSnapshot:
    equity: float
    daily_drawdown: float
    total_drawdown: float
    daily_halt_active: bool
    kill_switch_active: bool
    kill_switch_triggered_now: bool
    cooldown_active: bool = False
    post_kill_cooldown_active: bool = False


class RiskManager:
    def __init__(
        self,
        config: RiskConfig,
        vol_config: VolatilityConfig | None = None,
    ) -> None:
        self.config = config
        self._vol_config = vol_config or VolatilityConfig()

    def evaluate(
        self,
        state: PortfolioState,
        equity: float,
        now: datetime,
        *,
        allow_kill_switch: bool = True,
    ) -> RiskSnapshot:
        """Update drawdown state. ``allow_kill_switch=False`` is for cycles where the equity
        inputs are known to be incomplete (a failed balance sync), so bad data can't liquidate."""
        session_date = now.date().isoformat()
        if state.daily_baseline_date != session_date:
            state.daily_baseline_date = session_date
            state.daily_baseline_equity = equity
            state.halt_new_trades_for_day = False

        if equity > state.peak_equity:
            state.peak_equity = equity

        daily_drawdown = 0.0
        if state.daily_baseline_equity > 0 and equity < state.daily_baseline_equity:
            daily_drawdown = (state.daily_baseline_equity - equity) / state.daily_baseline_equity

        total_drawdown = 0.0
        if state.peak_equity > 0 and equity < state.peak_equity:
            total_drawdown = (state.peak_equity - equity) / state.peak_equity

        if daily_drawdown >= self.config.daily_halt_drawdown:
            state.halt_new_trades_for_day = True

        kill_switch_triggered_now = False
        if allow_kill_switch and total_drawdown >= self.config.kill_switch_drawdown and not state.kill_switch_engaged:
            state.kill_switch_engaged = True
            kill_switch_triggered_now = True

        post_kill_cooldown_active = self._post_kill_cooldown_active(state, now)
        return RiskSnapshot(
            equity=equity,
            daily_drawdown=daily_drawdown,
            total_drawdown=total_drawdown,
            daily_halt_active=state.halt_new_trades_for_day,
            kill_switch_active=state.kill_switch_engaged,
            kill_switch_triggered_now=kill_switch_triggered_now,
            cooldown_active=state.cooldown_remaining > 0,
            post_kill_cooldown_active=post_kill_cooldown_active,
        )

    # ------------------------------------------------------------------ #
    # Gate for new entries                                                 #
    # ------------------------------------------------------------------ #

    def can_open_new_position(self, state: PortfolioState) -> bool:
        if state.kill_switch_engaged or state.halt_new_trades_for_day:
            return False
        if self.config.max_open_positions > 0 and len(state.positions) >= self.config.max_open_positions:
            return False
        if state.cooldown_remaining > 0:
            return False
        if self._post_kill_cooldown_active(state, datetime.now(timezone.utc)):
            return False
        return True

    def _post_kill_cooldown_active(self, state: PortfolioState, now: datetime) -> bool:
        """True when entries should be blocked following a recent kill switch clear."""
        hours = self.config.post_kill_cooldown_hours
        if hours <= 0 or not state.kill_switch_cleared_at:
            return False
        try:
            cleared = datetime.fromisoformat(state.kill_switch_cleared_at)
            if cleared.tzinfo is None:
                cleared = cleared.replace(tzinfo=timezone.utc)
            now_utc = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
            return (now_utc - cleared) < timedelta(hours=hours)
        except (ValueError, OSError):
            return False

    # ------------------------------------------------------------------ #
    # Trailing stop-loss                                                   #
    # ------------------------------------------------------------------ #

    def update_high_watermarks(self, state: PortfolioState) -> None:
        """Update per-position high watermarks from current prices."""
        for symbol, position in state.positions.items():
            price = position.current_price
            prev_high = state.position_high_watermarks.get(symbol, 0.0)
            if price > prev_high:
                state.position_high_watermarks[symbol] = price

        # Clean up watermarks for closed positions
        closed = [s for s in state.position_high_watermarks if s not in state.positions]
        for s in closed:
            del state.position_high_watermarks[s]

    def check_trailing_stops(self, state: PortfolioState) -> list[str]:
        """Return symbols that should be closed due to trailing stop-loss.

        A trailing stop triggers when the current price has fallen
        ``trailing_stop_pct`` (e.g. 10%) below the recorded high watermark.
        Returns an empty list when the feature is disabled (pct == 0).
        """
        if self.config.trailing_stop_pct <= 0:
            return []

        to_close: list[str] = []
        for symbol, position in state.positions.items():
            high = state.position_high_watermarks.get(symbol, position.entry_price)
            if high <= 0:
                continue
            drop_pct = (high - position.current_price) / high
            if drop_pct >= self.config.trailing_stop_pct:
                logger.info(
                    "Trailing stop triggered for %s: high=%.4f current=%.4f drop=%.2f%%",
                    symbol, high, position.current_price, drop_pct * 100,
                )
                to_close.append(symbol)
        return to_close

    # ------------------------------------------------------------------ #
    # Consecutive loss cooldown                                            #
    # ------------------------------------------------------------------ #

    def record_trade_result(self, state: PortfolioState, *, is_loss: bool) -> None:
        """Update consecutive-loss counter after a position is closed.

        When the loss streak reaches ``cooldown_consecutive_losses`` (and the
        feature is enabled), ``cooldown_remaining`` is set to
        ``cooldown_periods`` which blocks new entries until it is decremented
        back to zero.
        """
        if is_loss:
            state.consecutive_losses += 1
        else:
            state.consecutive_losses = 0

        threshold = self.config.cooldown_consecutive_losses
        if threshold > 0 and state.consecutive_losses >= threshold and state.cooldown_remaining == 0:
            state.cooldown_remaining = self.config.cooldown_periods
            logger.warning(
                "Cooldown activated: %d consecutive losses (threshold=%d). "
                "Pausing new entries for %d trade cycles.",
                state.consecutive_losses, threshold, self.config.cooldown_periods,
            )

    def tick_cooldown(self, state: PortfolioState) -> None:
        """Decrement cooldown counter once per trade cycle.

        Call this at the start of each ``service.run()`` invocation.
        """
        if state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1
            if state.cooldown_remaining == 0:
                logger.info("Cooldown expired — resuming normal trading.")
                state.consecutive_losses = 0

    # ------------------------------------------------------------------ #
    # Position sizing                                                      #
    # ------------------------------------------------------------------ #

    def position_notional(
        self,
        state: PortfolioState,
        equity: float,
        symbol: str = "",
    ) -> float:
        """Return the USD notional to allocate to a new position.

        When volatility targeting is enabled (``VOL_TARGETING_ENABLED=true``):
        - Positions are sized via ``target_vol / realized_vol × base_notional``
          where ``base_notional = equity × max_position_fraction``.
        - The leverage multiplier scales the per-trade budget up or down while
          keeping the base risk budget as the reference point.
        - Falls back to flat ``max_position_fraction`` sizing when there is
          insufficient price history for the symbol.
        """
        base_notional = min(state.cash, equity * self.config.max_position_fraction)

        if not self._vol_config.enabled:
            return base_notional

        prices = state.price_history.get(symbol, [])
        vol: float | None
        if self._vol_config.use_ewma:
            vol = VolatilityEngine.calculate_ewma_vol(prices, self._vol_config.ewma_decay)
        else:
            vol = VolatilityEngine.calculate_realized_vol(prices, self._vol_config.vol_window_days)

        if vol is None or vol <= 0:
            logger.debug(
                "vol_targeting fallback for %s: insufficient history (%d prices)",
                symbol,
                len(prices),
            )
            return base_notional

        vol_notional = VolatilityEngine.vol_adjusted_notional(
            target_vol=self._vol_config.target_annual_vol,
            realized_vol=vol,
            capital=base_notional,
            max_leverage=self._vol_config.max_leverage,
            min_allocation=self._vol_config.min_allocation,
        )
        result = min(vol_notional, state.cash)
        logger.debug(
            "vol_targeting %s: realized_vol=%.4f target=%.4f notional=%.2f",
            symbol,
            vol,
            self._vol_config.target_annual_vol,
            result,
        )
        return result
