from __future__ import annotations

import math


class VolatilityEngine:
    """Pure-math volatility calculations for position sizing.

    All methods are static so the class acts as a namespace — no state required.
    Uses only the Python stdlib (math module) to avoid adding numpy as a dependency.
    """

    # ------------------------------------------------------------------ #
    # Vol estimation                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_realized_vol(prices: list[float], window: int = 20) -> float | None:
        """Rolling standard-deviation annualised volatility.

        Requires at least ``window + 1`` prices (to produce ``window`` returns).
        Returns *None* when there is insufficient history — callers should fall
        back to a fixed notional in that case.

        Annualisation uses 252 trading days (uniform across all asset types).
        """
        if len(prices) < window + 1:
            return None

        # Use only the most recent window+1 prices
        recent = prices[-(window + 1):]
        returns: list[float] = []
        for i in range(1, len(recent)):
            prev, curr = recent[i - 1], recent[i]
            if prev <= 0 or curr <= 0:
                return None
            returns.append(math.log(curr / prev))

        if len(returns) < 2:
            return None

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = math.sqrt(variance)
        return daily_vol * math.sqrt(252)

    @staticmethod
    def calculate_ewma_vol(prices: list[float], decay: float = 0.94) -> float | None:
        """Exponentially-weighted moving-average annualised volatility.

        Decay = 0.94 is the RiskMetrics / professional-fund standard.
        alpha = 1 - decay (weight given to the most recent observation).

        Requires at least 2 prices.  Returns *None* on insufficient data.
        """
        if len(prices) < 2:
            return None

        alpha = 1.0 - decay
        log_returns: list[float] = []
        for i in range(1, len(prices)):
            prev, curr = prices[i - 1], prices[i]
            if prev <= 0 or curr <= 0:
                return None
            log_returns.append(math.log(curr / prev))

        # Bootstrap variance with the first squared return
        ewma_var = log_returns[0] ** 2
        for r in log_returns[1:]:
            ewma_var = alpha * r ** 2 + (1.0 - alpha) * ewma_var

        return math.sqrt(ewma_var * 252)

    # ------------------------------------------------------------------ #
    # Position sizing                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def vol_adjusted_notional(
        target_vol: float,
        realized_vol: float,
        capital: float,
        max_leverage: float = 2.0,
        min_allocation: float = 0.20,
    ) -> float:
        """Core volatility-targeting formula.

        leverage = target_vol / realized_vol
        notional  = clamp(leverage, min_allocation, max_leverage) × capital

        High vol  → leverage < 1 → smaller notional (scale down).
        Low vol   → leverage > 1 → larger notional  (scale up).
        """
        if realized_vol <= 0:
            return capital * min_allocation

        raw_leverage = target_vol / realized_vol
        leverage = max(min_allocation, min(raw_leverage, max_leverage))
        return capital * leverage

    # ------------------------------------------------------------------ #
    # No-trade band                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def should_rebalance(
        current_alloc: float,
        target_alloc: float,
        band: float = 0.10,
    ) -> bool:
        """Return True when the drift between current and target exceeds ``band``.

        Prevents excessive rebalancing and keeps transaction costs low.
        A band of 0.10 means: only rebalance when allocation has drifted > 10%
        relative to the target allocation.
        """
        if target_alloc <= 0:
            # If target is zero, rebalance only when current is non-zero (need to close out).
            # When both are zero there is nothing to do.
            return current_alloc != 0
        drift = abs(current_alloc - target_alloc) / target_alloc
        return drift > band

    # ------------------------------------------------------------------ #
    # Price history management                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def record_price(
        symbol: str,
        price: float,
        history: dict[str, list[float]],
        max_len: int = 60,
    ) -> None:
        """Append ``price`` to ``history[symbol]`` and trim to ``max_len`` entries."""
        if symbol not in history:
            history[symbol] = []
        history[symbol].append(price)
        if len(history[symbol]) > max_len:
            history[symbol] = history[symbol][-max_len:]
