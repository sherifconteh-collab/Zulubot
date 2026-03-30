from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from trading_bot.config import RiskConfig, VolatilityConfig
from trading_bot.models import PortfolioState
from trading_bot.volatility import VolatilityEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskSnapshot:
    equity: float
    daily_drawdown: float
    total_drawdown: float
    daily_halt_active: bool
    kill_switch_active: bool
    kill_switch_triggered_now: bool


class RiskManager:
    def __init__(
        self,
        config: RiskConfig,
        vol_config: VolatilityConfig | None = None,
    ) -> None:
        self.config = config
        self._vol_config = vol_config or VolatilityConfig()

    def evaluate(self, state: PortfolioState, equity: float, now: datetime) -> RiskSnapshot:
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
        if total_drawdown >= self.config.kill_switch_drawdown and not state.kill_switch_engaged:
            state.kill_switch_engaged = True
            kill_switch_triggered_now = True

        return RiskSnapshot(
            equity=equity,
            daily_drawdown=daily_drawdown,
            total_drawdown=total_drawdown,
            daily_halt_active=state.halt_new_trades_for_day,
            kill_switch_active=state.kill_switch_engaged,
            kill_switch_triggered_now=kill_switch_triggered_now,
        )

    def can_open_new_position(self, state: PortfolioState) -> bool:
        return not state.kill_switch_engaged and not state.halt_new_trades_for_day

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
