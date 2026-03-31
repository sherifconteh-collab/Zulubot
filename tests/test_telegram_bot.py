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

    async def test_trade_alert_includes_smart_module_context(self) -> None:
        result = {
            "paper_mode": False,
            "force_send": True,
            "timestamp": "2026-03-31T12:00:00+00:00",
            "modules": {
                "predictions": {
                    "status": "ok",
                    "buys": [],
                    "sells": [],
                    "markets_scanned": 18,
                    "markets_discovered": 24,
                    "can_trade": True,
                },
                "rotation": {
                    "status": "completed",
                    "buys": [],
                    "sells": [],
                    "action": "swap_ai_blocked",
                    "current_coin": "BTC",
                    "from": "BTC",
                    "to": "ETH",
                    "coins_tracked": 12,
                    "holdings_synced": 1,
                    "ai_action": "skip",
                    "ai_confidence": 0.82,
                    "ai_reasoning": "Momentum is stretched and fees are not justified yet.",
                },
                "latency_arb": {
                    "status": "ok",
                    "buys": [],
                    "sells": [],
                    "signals_detected": 2,
                    "markets_scanned": 11,
                    "market_regime": "volatile",
                    "stale_symbols": ["BTC-USD"],
                    "can_trade_live": True,
                },
            },
            "equity": 101_000.0,
            "cash": 1_500.0,
            "realized_pnl": 1_000.0,
            "open_positions": [],
            "kalshi_balance": 250.0,
            "ai_info": {
                "confirmation_mode": "boost",
                "multi_analyst_enabled": True,
                "selected_analysts": ["macro", "risk"],
                "budget_remaining_usd": 24.12,
                "monthly_cost_usd": 0.88,
                "streaming_enabled": True,
                "tool_calling_enabled": True,
                "vision_enabled": True,
                "extended_thinking_enabled": True,
                "prompt_guard_enabled": True,
                "registered_tools": ["get_current_price", "get_funding_rate"],
                "circuit_breaker": {
                    "state": "open",
                    "failure_count": 3,
                    "remaining_open_seconds": 42.0,
                },
            },
            "risk": {},
        }

        await self.bot.send_trade_alert(result)

        self.bot.send_message.assert_awaited_once()
        message = self.bot.send_message.await_args.args[0]
        self.assertIn("Markets: 18/24 scanned", message)
        self.assertIn("Execution: live-enabled", message)
        self.assertIn("AI vetoed swap", message)
        self.assertIn("Signals: 2", message)
        self.assertIn("Stale feeds: BTC-USD", message)
        self.assertIn("Features: streaming, tools, vision, thinking, prompt-guard", message)
        self.assertIn("Circuit: OPEN (42s cooldown)", message)

    async def test_heartbeat_includes_feature_and_circuit_summary(self) -> None:
        status = {
            "equity": 99_500.0,
            "cash": 2_000.0,
            "realized_pnl": -500.0,
            "open_positions": [],
            "risk": {
                "daily_drawdown_pct": -1.2,
                "total_drawdown_pct": -2.5,
                "peak_equity": 102_000.0,
                "daily_halt_active": False,
                "kill_switch_active": False,
            },
            "kalshi_balance": 125.0,
            "ai_info": {
                "confirmation_mode": "boost",
                "multi_analyst_enabled": False,
                "selected_analysts": [],
                "budget_remaining_usd": 21.50,
                "monthly_cost_usd": 3.50,
                "streaming_enabled": True,
                "tool_calling_enabled": True,
                "vision_enabled": False,
                "extended_thinking_enabled": True,
                "prompt_guard_enabled": True,
                "registered_tools": ["get_current_price"],
                "circuit_breaker": {
                    "state": "half_open",
                    "failure_count": 2,
                    "remaining_open_seconds": 9.0,
                },
            },
            "stale_price_count": 1,
            "stale_prices": [{"symbol": "ETH-USD", "age_seconds": 91.0}],
            "latency_ms": {"crypto": {"last_ms": 12.5}},
            "last_stock_screen_date": "2026-03-30",
            "last_predictions_scan_date": "2026-03-31",
            "last_polymarket_scan_date": "2026-03-31",
        }

        await self.bot.send_heartbeat(status)

        self.bot.send_message.assert_awaited_once()
        message = self.bot.send_message.await_args.args[0]
        self.assertIn("Features: streaming, tools, thinking, prompt-guard", message)
        self.assertIn("Tools: get_current_price", message)
        self.assertIn("Circuit: HALF_OPEN (9s cooldown)", message)
        self.assertIn("Stale Prices", message)
        self.assertIn("ETH-USD", message)


if __name__ == "__main__":
    unittest.main()
