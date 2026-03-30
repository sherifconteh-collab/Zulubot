from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Position:
    symbol: str
    asset_type: str
    strategy: str
    quantity: float
    entry_price: float
    current_price: float
    opened_at: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    daily_baseline_equity: float = 0.0
    daily_baseline_date: str = ""
    peak_equity: float = 0.0
    halt_new_trades_for_day: bool = False
    kill_switch_engaged: bool = False
    last_stock_screen_date: str | None = None
    last_polymarket_scan_date: str | None = None
    last_predictions_scan_date: str | None = None
    realized_pnl: float = 0.0
    rotation_data: dict[str, Any] = field(default_factory=dict)
    price_history: dict[str, list[float]] = field(default_factory=dict)
    # price_history: symbol → ordered list of one price snapshot per calendar day
    # (first mark recorded that day, oldest first; not guaranteed to be the market close).
    # Populated once per calendar day by PaperBroker.mark_positions().
    # Used by VolatilityEngine to compute realized / EWMA vol for position sizing.
    last_price_record_date: str = ""
    # rotation_data stores: {
    #   "current_coin": "SOL-USD",
    #   "ratios": {"BTC-USD": {"SOL-USD": 0.00234, ...}, ...},
    #   "coin_quantities": {"SOL-USD": 1.5, "BTC-USD": 0.0003, ...},
    #   "last_rotation_date": "2026-03-28",
    # }

    # Trailing stop-loss: per-position high watermark tracking
    position_high_watermarks: dict[str, float] = field(default_factory=dict)

    # Consecutive loss cooldown tracking
    consecutive_losses: int = 0
    cooldown_remaining: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": {symbol: position.to_dict() for symbol, position in self.positions.items()},
            "last_prices": self.last_prices,
            "daily_baseline_equity": self.daily_baseline_equity,
            "daily_baseline_date": self.daily_baseline_date,
            "peak_equity": self.peak_equity,
            "halt_new_trades_for_day": self.halt_new_trades_for_day,
            "kill_switch_engaged": self.kill_switch_engaged,
            "last_stock_screen_date": self.last_stock_screen_date,
            "last_polymarket_scan_date": self.last_polymarket_scan_date,
            "last_predictions_scan_date": self.last_predictions_scan_date,
            "realized_pnl": self.realized_pnl,
            "rotation_data": self.rotation_data,
            "price_history": self.price_history,
            "last_price_record_date": self.last_price_record_date,
            "position_high_watermarks": self.position_high_watermarks,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_remaining": self.cooldown_remaining,
        }


@dataclass(frozen=True)
class StockFundamentals:
    symbol: str
    pe_ratio: float
    pb_ratio: float
    debt_to_equity: float
    fcf_positive_signal: float
    pfcf_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "pe_ratio": self.pe_ratio,
            "pb_ratio": self.pb_ratio,
            "debt_to_equity": self.debt_to_equity,
            "fcf_positive_signal": self.fcf_positive_signal,
            "pfcf_ratio": self.pfcf_ratio,
        }


@dataclass(frozen=True)
class FearGreedReading:
    value: int
    classification: str
    timestamp: str
    time_until_update: int | None = None


@dataclass(frozen=True)
class PolymarketMarket:
    condition_id: str
    question: str
    description: str
    yes_token_id: str
    no_token_id: str
    yes_price: float
    no_price: float
    volume_24h: float
    liquidity: float
    tags: list[str]
    end_date: str
    active: bool
    resolved: bool
    outcome: str | None = None  # "Yes", "No", or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "description": self.description,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "tags": self.tags,
            "end_date": self.end_date,
            "active": self.active,
            "resolved": self.resolved,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class FundingSnapshot:
    symbol: str
    mark_price: float
    funding_rate: float
    next_funding_time: int | None
    observed_at: int | None

