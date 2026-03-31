"""Coinbase spot price feed using REST polling.

Maintains a rolling price history for tracked symbols (BTC-USD, ETH-USD, etc.)
by polling Coinbase's public market data endpoint on each strategy tick.

No WebSocket required — designed to work within the existing polling architecture.
History is kept in memory across ticks, so momentum calculations work as long
as the process stays alive.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_bot.clients import CoinbaseClient


class CoinbasePriceFeed:
    """Rolling price history from Coinbase REST polling.

    Call ``record_current_prices()`` on every strategy tick to accumulate
    history, then use ``get_momentum_pct()`` to detect price moves.

    Thread-safety: Not thread-safe. Call from a single async loop / thread.
    """

    def __init__(
        self,
        client: CoinbaseClient,
        symbols: list[str],
        *,
        history_seconds: float = 120.0,
        logger: logging.Logger,
    ) -> None:
        self.client = client
        self.symbols = list(symbols)
        self.history_seconds = history_seconds
        self.logger = logger.getChild("coinbase_feed")

        # symbol → deque of (unix_timestamp, price)
        self._history: dict[str, deque[tuple[float, float]]] = {
            s: deque() for s in symbols
        }

    def record_current_prices(self) -> dict[str, float]:
        """Fetch current prices from Coinbase and store in rolling history.

        Returns a dict of {symbol: price} for the prices that were
        successfully fetched.
        """
        try:
            prices = self.client.get_spot_prices_by_product_ids(self.symbols)
        except Exception as exc:
            self.logger.warning("Price fetch failed: %s", exc)
            return {}

        now = time.time()
        cutoff = now - self.history_seconds

        for symbol, price in prices.items():
            if price <= 0:
                continue
            history = self._history.setdefault(symbol, deque())
            history.append((now, price))

            # Trim entries older than history_seconds
            while history and history[0][0] < cutoff:
                history.popleft()

        return prices

    def get_latest_price(self, symbol: str) -> float:
        """Return the most recently recorded price for a symbol, or 0.0."""
        history = self._history.get(symbol)
        if not history:
            return 0.0
        return history[-1][1]

    def get_momentum_pct(self, symbol: str, window_seconds: float) -> float:
        """Return % price change over the last ``window_seconds``.

        Positive = price went up, negative = price went down.
        Returns 0.0 if there is insufficient history.

        Example: if BTC went from 80000 → 80480 in 30s, returns +0.6.
        """
        history = self._history.get(symbol)
        if not history or len(history) < 2:
            return 0.0

        now = time.time()
        cutoff = now - window_seconds

        # Find the oldest price within the window
        oldest_price: float | None = None
        for ts, price in history:
            if ts >= cutoff:
                if oldest_price is None:
                    oldest_price = price
                break

        # Fall back to oldest available if window is shorter than history
        if oldest_price is None:
            oldest_price = history[0][1]

        current_price = history[-1][1]
        if oldest_price <= 0:
            return 0.0

        return ((current_price - oldest_price) / oldest_price) * 100.0

    def is_ready(self, symbol: str, min_history_seconds: float = 30.0) -> bool:
        """Return True if we have at least ``min_history_seconds`` of price history."""
        history = self._history.get(symbol)
        if not history or len(history) < 2:
            return False
        span = history[-1][0] - history[0][0]
        return span >= min_history_seconds

    def latest_price_age_seconds(self, symbol: str) -> float:
        """Return the age in seconds of the most recent recorded tick."""
        history = self._history.get(symbol)
        if not history:
            return float("inf")
        return max(0.0, time.time() - history[-1][0])

    def is_stale(self, symbol: str, max_age_seconds: float) -> bool:
        """Return True when the latest price is older than ``max_age_seconds``."""
        return self.latest_price_age_seconds(symbol) > max_age_seconds

    def get_history(
        self,
        symbol: str,
        *,
        lookback_seconds: float | None = None,
    ) -> list[tuple[float, float]]:
        history = self._history.get(symbol)
        if not history:
            return []
        if lookback_seconds is None or lookback_seconds <= 0:
            return list(history)

        cutoff = time.time() - lookback_seconds
        return [(ts, price) for ts, price in history if ts >= cutoff]

    def history_span_seconds(self, symbol: str) -> float:
        """Return the time span (in seconds) of available history for a symbol."""
        history = self._history.get(symbol)
        if not history or len(history) < 2:
            return 0.0
        return history[-1][0] - history[0][0]
