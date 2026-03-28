from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_bot.models import PortfolioState, Position


class StateStore:
    def __init__(self, state_path: Path, initial_cash: float, timezone: ZoneInfo) -> None:
        self.state_path = state_path
        self.initial_cash = initial_cash
        self.timezone = timezone

    def load(self) -> PortfolioState:
        if not self.state_path.exists():
            now = datetime.now(self.timezone)
            return PortfolioState(
                cash=self.initial_cash,
                positions={},
                last_prices={},
                daily_baseline_equity=self.initial_cash,
                daily_baseline_date=now.date().isoformat(),
                peak_equity=self.initial_cash,
                halt_new_trades_for_day=False,
                kill_switch_engaged=False,
                last_stock_screen_date=None,
                last_polymarket_scan_date=None,
                realized_pnl=0.0,
            )

        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        positions = {
            symbol: Position(**position_payload)
            for symbol, position_payload in payload.get("positions", {}).items()
        }
        return PortfolioState(
            cash=float(payload["cash"]),
            positions=positions,
            last_prices={symbol: float(price) for symbol, price in payload.get("last_prices", {}).items()},
            daily_baseline_equity=float(payload.get("daily_baseline_equity", self.initial_cash)),
            daily_baseline_date=payload.get("daily_baseline_date", ""),
            peak_equity=float(payload.get("peak_equity", self.initial_cash)),
            halt_new_trades_for_day=bool(payload.get("halt_new_trades_for_day", False)),
            kill_switch_engaged=bool(payload.get("kill_switch_engaged", False)),
            last_stock_screen_date=payload.get("last_stock_screen_date"),
            last_polymarket_scan_date=payload.get("last_polymarket_scan_date"),
            realized_pnl=float(payload.get("realized_pnl", 0.0)),
            rotation_data=payload.get("rotation_data", {}),
        )

    def save(self, state: PortfolioState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

