"""Per-symbol performance memory ("learn as it goes").

This is a deliberately simple, explainable learning layer — not a black-box
ML model. For every closed position the bot records the realized P&L, and
strategies consult the running record before opening new entries. The effect
is a form of online learning: symbols / markets that have accumulated losses
get benched, winners get preferred.

The record lives on ``PortfolioState.performance_memory`` so it is persisted
by ``StateStore`` between daemon restarts.

Schema (per symbol)::

    {
        "wins": int,                 # count of closes with realized_pnl > 0
        "losses": int,               # count of closes with realized_pnl < 0
        "total_pnl": float,          # cumulative realized P&L (not annualized)
        "consecutive_losses": int,   # streak of consecutive losing closes
        "cooldown_remaining": int,   # strategy cycles left before this symbol
                                     # may be re-entered; decremented lazily
                                     # by strategies on each scan.
        "last_result_at": str | None, # ISO timestamp of the last close
    }

The config is duck-typed: any object exposing ``enabled`` (bool),
``loser_cooldown_losses`` (int) and ``loser_cooldown_cycles`` (int) works
(in production this is ``config.LearningConfig``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from trading_bot.models import PortfolioState


class LearningConfigLike(Protocol):
    enabled: bool
    loser_cooldown_losses: int
    loser_cooldown_cycles: int


def _get_record(state: PortfolioState, symbol: str) -> dict[str, Any]:
    memory = state.performance_memory
    rec = memory.get(symbol)
    if rec is None:
        rec = {
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "consecutive_losses": 0,
            "cooldown_remaining": 0,
            "last_result_at": None,
        }
        memory[symbol] = rec
    return rec


def record_close(
    state: PortfolioState,
    *,
    symbol: str,
    realized_pnl: float,
    config: LearningConfigLike,
    now: datetime | None = None,
) -> None:
    """Update the performance record for ``symbol`` after a position closed.

    A break-even close (realized_pnl == 0) is treated as neither a win nor a
    loss — the consecutive-loss streak is unaffected.
    """
    if not config.enabled:
        return
    rec = _get_record(state, symbol)
    rec["total_pnl"] = float(rec.get("total_pnl", 0.0)) + float(realized_pnl)
    ts = (now or datetime.now(timezone.utc)).isoformat()
    rec["last_result_at"] = ts

    if realized_pnl > 0:
        rec["wins"] = int(rec.get("wins", 0)) + 1
        rec["consecutive_losses"] = 0
        rec["cooldown_remaining"] = 0
    elif realized_pnl < 0:
        rec["losses"] = int(rec.get("losses", 0)) + 1
        rec["consecutive_losses"] = int(rec.get("consecutive_losses", 0)) + 1
        if rec["consecutive_losses"] >= config.loser_cooldown_losses:
            rec["cooldown_remaining"] = config.loser_cooldown_cycles


def is_symbol_benched(
    state: PortfolioState,
    symbol: str,
    *,
    config: LearningConfigLike,
    decrement: bool = False,
) -> bool:
    """Return True if ``symbol`` is currently in cooldown.

    When ``decrement=True`` the cooldown counter is decremented by one after
    the check — strategies should call with ``decrement=True`` exactly once
    per evaluation cycle per symbol so the bench expires over time.
    """
    if not config.enabled:
        return False
    memory = state.performance_memory
    rec = memory.get(symbol)
    if rec is None:
        return False
    remaining = int(rec.get("cooldown_remaining", 0) or 0)
    if remaining <= 0:
        return False
    if decrement:
        rec["cooldown_remaining"] = remaining - 1
    return True


def summary(state: PortfolioState, symbol: str) -> dict[str, Any]:
    """Snapshot of a symbol's learning record. Empty dict if unseen."""
    return dict(state.performance_memory.get(symbol, {}))
