from __future__ import annotations

import importlib.util
import unittest

from trading_bot.charting import render_candlestick_chart_data_uri


_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


@unittest.skipUnless(_HAS_MATPLOTLIB, "matplotlib not installed")
class ChartingTests(unittest.TestCase):
    def test_render_candlestick_chart_returns_data_uri(self) -> None:
        base_ts = 1_711_886_400.0
        samples = [
            (base_ts + (idx * 60.0), 100.0 + ((idx % 5) - 2) * 1.5 + idx * 0.25)
            for idx in range(18)
        ]

        image_uri = render_candlestick_chart_data_uri("BTC-USD", samples)

        self.assertIsNotNone(image_uri)
        self.assertTrue(str(image_uri).startswith("data:image/png;base64,"))

    def test_render_candlestick_chart_requires_enough_samples(self) -> None:
        image_uri = render_candlestick_chart_data_uri("BTC-USD", [(1.0, 100.0), (2.0, 101.0)])

        self.assertIsNone(image_uri)