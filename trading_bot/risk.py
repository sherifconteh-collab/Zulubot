from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_bot.config import RiskConfig
from trading_bot.models import PortfolioState


@dataclass(frozen=True)
class RiskSnapshot:
    equity: float
    daily_drawdown: float
    total_drawdown: float
    daily_halt_active: bool
    kill_switch_active: bool
    kill_switch_triggered_now: bool


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

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

    def position_notional(self, state: PortfolioState, equity: float) -> float:
        return min(state.cash, equity * self.config.max_position_fraction)

