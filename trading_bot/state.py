from __future__ import annotations

import json
import os
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
                cash_reserves={},
                ai_monthly_usage={},
                positions={},
                last_prices={},
                last_price_timestamps={},
                daily_baseline_equity=self.initial_cash,
                daily_baseline_date=now.date().isoformat(),
                peak_equity=self.initial_cash,
                halt_new_trades_for_day=False,
                kill_switch_engaged=False,
                last_stock_screen_date=None,
                last_polymarket_scan_date=None,
                last_predictions_scan_date=None,
                realized_pnl=0.0,
            )

        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        positions = {
            symbol: Position(**position_payload)
            for symbol, position_payload in payload.get("positions", {}).items()
        }
        price_history = {
            sym: [float(p) for p in prices]
            for sym, prices in payload.get("price_history", {}).items()
        }
        return PortfolioState(
            cash=float(payload["cash"]),
            cash_reserves={
                bucket: float(value)
                for bucket, value in payload.get("cash_reserves", {}).items()
            },
            ai_monthly_usage=payload.get("ai_monthly_usage", {}),
            positions=positions,
            last_prices={symbol: float(price) for symbol, price in payload.get("last_prices", {}).items()},
            last_price_timestamps={
                symbol: float(ts)
                for symbol, ts in payload.get("last_price_timestamps", {}).items()
            },
            daily_baseline_equity=float(payload.get("daily_baseline_equity", self.initial_cash)),
            daily_baseline_date=payload.get("daily_baseline_date", ""),
            peak_equity=float(payload.get("peak_equity", self.initial_cash)),
            halt_new_trades_for_day=bool(payload.get("halt_new_trades_for_day", False)),
            kill_switch_engaged=bool(payload.get("kill_switch_engaged", False)),
            last_stock_screen_date=payload.get("last_stock_screen_date"),
            last_polymarket_scan_date=payload.get("last_polymarket_scan_date"),
            last_predictions_scan_date=payload.get("last_predictions_scan_date"),
            realized_pnl=float(payload.get("realized_pnl", 0.0)),
            rotation_data=payload.get("rotation_data", {}),
            price_history=price_history,
            last_price_record_date=payload.get("last_price_record_date", ""),
            position_high_watermarks={
                sym: float(val)
                for sym, val in payload.get("position_high_watermarks", {}).items()
            },
            consecutive_losses=int(payload.get("consecutive_losses", 0)),
            cooldown_remaining=int(payload.get("cooldown_remaining", 0)),
            performance_memory={
                str(sym): dict(rec)
                for sym, rec in (payload.get("performance_memory") or {}).items()
                if isinstance(rec, dict)
            },
        )

    def save(self, state: PortfolioState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_name(f"{self.state_path.name}.tmp")
        payload = json.dumps(state.to_dict(), indent=2, sort_keys=True)
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.state_path)

