from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from trading_bot.telegram_bot import TradingTelegramBot


class TradingTelegramBotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_telegram_bot")
        config = SimpleNamespace(bot_token="token", chat_id="123", is_configured=True)
        self.bot = TradingTelegramBot(config, self.logger)
        self.bot.send_message = AsyncMock()  # type: ignore[method-assign]

    async def test_trade_alert_minimal_format_when_trades_present(self) -> None:
        """Trade alert should be concise: header, one line per trade,
        one-line portfolio footer. No emoji clutter, no AI/latency/scan blocks."""
        result = {
            "paper_mode": False,
            "force_send": False,
            "timestamp": "2026-03-31T12:00:00+00:00",
            "modules": {
                "crypto": {"status": "ok", "buys": ["BTCUSDT"], "sells": []},
            },
            "equity": 101_000.0,
            "cash": 1_500.0,
            "realized_pnl": 1_000.0,
            "open_positions": [],
            "risk": {},
        }

        await self.bot.send_trade_alert(result)

        self.bot.send_message.assert_awaited_once()
        message = self.bot.send_message.await_args.args[0]
        # Core content present
        self.assertIn("Trade cycle", message)
        self.assertIn("LIVE", message)
        self.assertIn("BUY  BTCUSDT", message)
        self.assertIn("[crypto]", message)
        self.assertIn("Equity", message)
        # Minimal format: no decorative dividers or heavy emoji blocks
        self.assertNotIn("\u2500" * 10, message)  # no ─────── dividers
        self.assertNotIn("AI ANALYSTS", message)
        self.assertNotIn("LATENCY", message)
        self.assertNotIn("PORTFOLIO", message)

    async def test_trade_alert_silent_when_nothing_happened(self) -> None:
        """No trades, no errors, no risk event → no Telegram message at all.
        This is the core anti-spam behaviour for scheduled cycles."""
        result = {
            "paper_mode": True,
            "force_send": False,
            "modules": {
                "crypto": {"status": "ok", "buys": [], "sells": []},
                "predictions": {"status": "completed", "buys": [], "sells": []},
            },
            "equity": 100_000.0,
            "cash": 100_000.0,
            "realized_pnl": 0.0,
            "open_positions": [],
            "risk": {},
        }

        await self.bot.send_trade_alert(result)

        self.bot.send_message.assert_not_awaited()

    async def test_trade_alert_force_send_emits_even_without_trades(self) -> None:
        """force_send=True (used by /crypto, /predict etc. commands) must
        always produce a message so the user sees the command was handled."""
        result = {
            "paper_mode": True,
            "force_send": True,
            "modules": {
                "crypto": {"status": "ok", "buys": [], "sells": []},
            },
            "equity": 100_000.0,
            "cash": 100_000.0,
            "realized_pnl": 0.0,
            "open_positions": [],
            "risk": {},
        }

        await self.bot.send_trade_alert(result)

        self.bot.send_message.assert_awaited_once()
        message = self.bot.send_message.await_args.args[0]
        self.assertIn("No trades this cycle.", message)

    async def test_trade_alert_surfaces_module_errors(self) -> None:
        result = {
            "paper_mode": True,
            "force_send": False,
            "modules": {
                "predictions": {"status": "error", "buys": [], "sells": [], "error": "API timeout"},
            },
            "equity": 100_000.0,
            "cash": 100_000.0,
            "realized_pnl": 0.0,
            "open_positions": [],
            "risk": {},
        }

        await self.bot.send_trade_alert(result)

        self.bot.send_message.assert_awaited_once()
        message = self.bot.send_message.await_args.args[0]
        self.assertIn("Errors", message)
        self.assertIn("predictions", message)
        self.assertIn("API timeout", message)

    async def test_trade_alert_surfaces_kill_switch(self) -> None:
        result = {
            "paper_mode": False,
            "force_send": False,
            "modules": {"crypto": {"status": "halted", "buys": [], "sells": []}},
            "equity": 95_000.0,
            "cash": 50_000.0,
            "realized_pnl": -5_000.0,
            "open_positions": [],
            "risk": {"kill_switch_active": True, "daily_halt_active": False},
        }

        await self.bot.send_trade_alert(result)

        self.bot.send_message.assert_awaited_once()
        message = self.bot.send_message.await_args.args[0]
        self.assertIn("KILL SWITCH ACTIVE", message)

    async def test_heartbeat_compact_format(self) -> None:
        status = {
            "equity": 99_500.0,
            "cash": 2_000.0,
            "realized_pnl": -500.0,
            "open_positions": [
                {
                    "symbol": "BTCUSDT",
                    "asset_type": "crypto",
                    "entry_price": 100.0,
                    "current_price": 110.0,
                    "unrealized_pnl": 10.0,
                    "market_value": 110.0,
                    "quantity": 1.0,
                },
            ],
            "risk": {
                "daily_drawdown_pct": -1.2,
                "total_drawdown_pct": -2.5,
                "peak_equity": 102_000.0,
                "daily_halt_active": False,
                "kill_switch_active": False,
            },
            "kalshi_balance": 125.0,
        }

        await self.bot.send_heartbeat(status)

        self.bot.send_message.assert_awaited_once()
        message = self.bot.send_message.await_args.args[0]
        # Core content present
        self.assertIn("Status", message)
        self.assertIn("Equity", message)
        self.assertIn("Kalshi", message)
        self.assertIn("BTCUSDT", message)
        # No dropped blocks
        self.assertNotIn("AI ANALYSTS", message)
        self.assertNotIn("Last Scans", message)
        self.assertNotIn("Latency", message)
        # No decorative dividers
        self.assertNotIn("\u2500" * 10, message)

    async def test_risk_alert_minimal_single_line(self) -> None:
        await self.bot.send_risk_alert({
            "kill_switch_active": True,
            "daily_halt_active": False,
            "total_drawdown_pct": -15.0,
        })
        self.bot.send_message.assert_awaited_once()
        message = self.bot.send_message.await_args.args[0]
        self.assertIn("Kill switch", message)
        self.assertIn("All positions closed", message)
        # Old format had a big emoji banner — make sure it's gone.
        self.assertNotIn("RISK ALERT", message)

    async def test_error_minimal_single_line(self) -> None:
        await self.bot.send_error("connection refused")
        self.bot.send_message.assert_awaited_once()
        message = self.bot.send_message.await_args.args[0]
        self.assertIn("Error", message)
        self.assertIn("connection refused", message)
        # No dividers.
        self.assertNotIn("\u2500" * 10, message)


if __name__ == "__main__":
    unittest.main()
