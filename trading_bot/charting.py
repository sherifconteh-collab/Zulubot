from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def render_candlestick_chart_data_uri(
    symbol: str,
    samples: Sequence[tuple[float, float]],
    *,
    title: str | None = None,
    target_candles: int = 24,
) -> str | None:
    if len(samples) < 4:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return None

    candles = _build_candles(samples, target_candles=target_candles)
    if len(candles) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    _plot_candles(ax, candles, Rectangle)
    ax.set_title(title or f"{symbol} price action")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2, linestyle=":")
    fig.autofmt_xdate()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_candles(
    samples: Sequence[tuple[float, float]],
    *,
    target_candles: int,
) -> list[dict[str, float | datetime]]:
    ordered = sorted((float(ts), float(price)) for ts, price in samples if price > 0)
    if len(ordered) < 2:
        return []

    start = ordered[0][0]
    end = ordered[-1][0]
    span = max(end - start, 1.0)
    bucket_count = max(2, min(int(target_candles), len(ordered)))
    bucket_size = span / bucket_count

    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bucket_count)]
    for ts, price in ordered:
        index = int((ts - start) / bucket_size) if bucket_size > 0 else 0
        if index >= bucket_count:
            index = bucket_count - 1
        buckets[index].append((ts, price))

    candles: list[dict[str, float | datetime]] = []
    last_close = ordered[0][1]
    for index, bucket in enumerate(buckets):
        if not bucket:
            candle_time = datetime.fromtimestamp(start + (index * bucket_size), tz=timezone.utc)
            candles.append(
                {
                    "time": candle_time,
                    "open": last_close,
                    "high": last_close,
                    "low": last_close,
                    "close": last_close,
                }
            )
            continue

        prices = [price for _, price in bucket]
        last_close = prices[-1]
        candles.append(
            {
                "time": datetime.fromtimestamp(bucket[-1][0], tz=timezone.utc),
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
            }
        )

    return candles


def _plot_candles(ax: Axes, candles: Sequence[dict[str, float | datetime]], rectangle_cls: type) -> None:
    import matplotlib.dates as mdates

    width = 0.6 / 24.0
    for candle in candles:
        candle_time = candle["time"]
        open_price = float(candle["open"])
        close_price = float(candle["close"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        x = mdates.date2num(candle_time)
        color = "#198754" if close_price >= open_price else "#c1121f"

        ax.vlines(x, low_price, high_price, color=color, linewidth=1.2)
        body_bottom = min(open_price, close_price)
        body_height = max(abs(close_price - open_price), 1e-6)
        ax.add_patch(
            rectangle_cls(
                (x - (width / 2), body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
            )
        )