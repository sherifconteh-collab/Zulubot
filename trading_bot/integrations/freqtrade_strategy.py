"""Freqtrade IStrategy subclass that mirrors our crypto strategy signals.

Translates our Fear & Greed + funding rate logic into Freqtrade's
indicator/signal framework for backtesting and hyperparameter optimization.

This file is only used when freqtrade is installed. It defines a strategy
class that Freqtrade's backtesting engine can load directly.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import freqtrade.strategy as ft_strategy
    from freqtrade.strategy import IStrategy, merge_informative_pair
    import pandas as pd
    import talib.abstract as ta
    FREQTRADE_AVAILABLE = True
except ImportError:
    FREQTRADE_AVAILABLE = False
    # Stub for when freqtrade is not installed
    class IStrategy:  # type: ignore[no-redef]
        pass


if FREQTRADE_AVAILABLE:

    class CryptoSentimentStrategy(IStrategy):
        """Freqtrade strategy mirroring our crypto sentiment + funding logic.

        Uses RSI as a proxy for Fear & Greed (since F&G API data isn't
        available in historical candles), combined with volume analysis.

        Hyperoptable parameters:
          - buy_rsi: RSI threshold for buy signals (proxy for fear)
          - sell_rsi: RSI threshold for sell signals (proxy for greed)
          - buy_volume_factor: Volume spike multiplier for confirmation
          - rsi_period: RSI calculation period
        """

        # Strategy metadata
        INTERFACE_VERSION = 3
        timeframe = "1h"
        can_short = False
        stoploss = -0.15  # Maps to our kill_switch_drawdown
        trailing_stop = True
        trailing_stop_positive = 0.05
        trailing_stop_positive_offset = 0.10
        trailing_only_offset_is_reached = True

        # Hyperoptable parameters
        buy_rsi = 30  # Fear proxy: buy when RSI < this (like F&G < 25)
        sell_rsi = 70  # Greed proxy: sell when RSI > this (like F&G > 75)
        buy_volume_factor = 1.5  # Volume must be 1.5x average
        rsi_period = 14

        # ROI table — progressive take-profit
        minimal_roi = {
            "0": 0.15,    # 15% at any time
            "60": 0.10,   # 10% after 1 hour
            "120": 0.05,  # 5% after 2 hours
            "360": 0.02,  # 2% after 6 hours
        }

        def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            """Calculate technical indicators."""
            # RSI as Fear & Greed proxy
            dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)

            # Volume analysis
            dataframe["volume_mean"] = dataframe["volume"].rolling(window=20).mean()
            dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_mean"]

            # Bollinger Bands for volatility context
            bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
            dataframe["bb_upper"] = bollinger["upperband"]
            dataframe["bb_lower"] = bollinger["lowerband"]
            dataframe["bb_mid"] = bollinger["middleband"]
            dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]

            # MACD for trend confirmation
            macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
            dataframe["macd"] = macd["macd"]
            dataframe["macd_signal"] = macd["macdsignal"]
            dataframe["macd_hist"] = macd["macdhist"]

            # EMA for trend direction
            dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
            dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

            return dataframe

        def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            """Generate buy signals.

            Mirrors our crypto strategy:
              - RSI < buy_rsi (fear proxy, like F&G < 25)
              - Volume spike confirms capitulation
              - Price near lower Bollinger Band (oversold)
            """
            dataframe.loc[
                (
                    (dataframe["rsi"] < self.buy_rsi)
                    & (dataframe["volume_ratio"] > self.buy_volume_factor)
                    & (dataframe["close"] < dataframe["bb_lower"] * 1.02)
                    & (dataframe["volume"] > 0)
                ),
                "enter_long",
            ] = 1

            return dataframe

        def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            """Generate sell signals.

            Mirrors our crypto strategy:
              - RSI > sell_rsi (greed proxy, like F&G > 75)
              - Price near upper Bollinger Band (overbought)
            """
            dataframe.loc[
                (
                    (dataframe["rsi"] > self.sell_rsi)
                    & (dataframe["close"] > dataframe["bb_upper"] * 0.98)
                    & (dataframe["volume"] > 0)
                ),
                "exit_long",
            ] = 1

            return dataframe


class ValueInvestingStrategy:
    """Stub for stock value strategy backtesting.

    Freqtrade doesn't natively support stock backtesting, but this
    class provides the interface for future expansion with custom
    data feeds.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        logger.info("ValueInvestingStrategy stub initialized (stocks not backtestable via Freqtrade)")
