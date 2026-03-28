"""Tests for Freqtrade adapter and config generator.

Tests the config generation and adapter logic without requiring
Freqtrade to be installed.
"""

from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_bot.config import AppConfig
from trading_bot.integrations.freqtrade_config import (
    SYMBOL_TO_PAIR,
    generate_freqtrade_config,
    generate_hyperopt_config,
)
from trading_bot.integrations.freqtrade_adapter import FreqtradeAdapter


class FreqtradeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig.load()

    def test_generates_valid_config(self) -> None:
        ft_config = generate_freqtrade_config(self.config, dry_run=True)

        self.assertIn("exchange", ft_config)
        self.assertIn("pair_whitelist", ft_config["exchange"])
        self.assertTrue(ft_config["dry_run"])
        self.assertEqual(ft_config["stake_currency"], "USDT")

    def test_pairs_match_crypto_symbols(self) -> None:
        ft_config = generate_freqtrade_config(self.config)
        pairs = ft_config["exchange"]["pair_whitelist"]

        # Our default symbols are BTCUSDT, ETHUSDT
        self.assertIn("BTC/USDT", pairs)
        self.assertIn("ETH/USDT", pairs)

    def test_stoploss_maps_to_kill_switch(self) -> None:
        ft_config = generate_freqtrade_config(self.config)

        # Kill switch is 0.15, stoploss should be -0.15
        self.assertAlmostEqual(ft_config["stoploss"], -0.15, places=2)

    def test_dry_run_wallet_matches_initial_cash(self) -> None:
        ft_config = generate_freqtrade_config(self.config)

        self.assertEqual(ft_config["dry_run_wallet"], self.config.initial_cash)

    def test_api_server_uses_config_port(self) -> None:
        ft_config = generate_freqtrade_config(self.config)

        self.assertEqual(
            ft_config["api_server"]["listen_port"],
            self.config.freqtrade.api_server_port,
        )

    def test_hyperopt_config_has_parameters(self) -> None:
        hyperopt = generate_hyperopt_config(self.config, epochs=50)

        self.assertEqual(hyperopt["epochs"], 50)
        self.assertIn("buy_rsi", hyperopt["hyperopt_parameters"])
        self.assertIn("sell_rsi", hyperopt["hyperopt_parameters"])
        self.assertEqual(hyperopt["loss_function"], "SharpeHyperOptLoss")


class SymbolMappingTests(unittest.TestCase):
    def test_common_symbols_mapped(self) -> None:
        self.assertEqual(SYMBOL_TO_PAIR["BTCUSDT"], "BTC/USDT")
        self.assertEqual(SYMBOL_TO_PAIR["ETHUSDT"], "ETH/USDT")
        self.assertEqual(SYMBOL_TO_PAIR["SOLUSDT"], "SOL/USDT")

    def test_all_mappings_have_slash_format(self) -> None:
        for binance_sym, ft_pair in SYMBOL_TO_PAIR.items():
            self.assertIn("/", ft_pair, f"Pair {ft_pair} missing / separator")
            self.assertTrue(ft_pair.endswith("/USDT"), f"Pair {ft_pair} should end with /USDT")


class FreqtradeAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig.load()

    def test_adapter_reports_unavailable_when_disabled(self) -> None:
        adapter = FreqtradeAdapter(self.config)
        # Freqtrade is disabled by default in .env
        if not self.config.freqtrade.enabled:
            self.assertFalse(adapter.is_available)

    def test_adapter_raises_on_backtest_when_disabled(self) -> None:
        adapter = FreqtradeAdapter(self.config)
        if not adapter.is_available:
            with self.assertRaises(RuntimeError):
                adapter.run_backtest(timerange="20250101-20250301")


if __name__ == "__main__":
    unittest.main()
