from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from trading_bot.telegram_bot import (
    TradingTelegramBot,
    _format_lead_line,
    _format_sports_pick,
    _format_sports_picks,
)


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


def _make_update() -> SimpleNamespace:
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=123),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )


class LeadLineTests(unittest.TestCase):
    def test_holder_signal_is_human_readable(self) -> None:
        line = _format_lead_line({
            "symbol": "XRPUSDT",
            "reason": "held_by_other_strategy",
            "signals": {"fear_greed": 20, "holder": "markov_regime"},
        })
        self.assertIn("held by markov_regime", line)
        self.assertIn("XRPUSDT", line)

    def test_trend_filter_signal_shows_gap_vs_sma(self) -> None:
        line = _format_lead_line({
            "symbol": "XRPUSDT",
            "reason": "trend_filter_downtrend",
            "signals": {"price": 1.8, "sma": 2.0, "sma_days": 20},
        })
        self.assertIn("-10.0% vs SMA20", line)


class ModuleContextTests(unittest.TestCase):
    def setUp(self) -> None:
        config = SimpleNamespace(bot_token="token", chat_id="123", is_configured=True)
        self.bot = TradingTelegramBot(config, logging.getLogger("test_telegram_bot"))

    def test_crypto_context_shows_weights_and_trend_gate(self) -> None:
        self.bot.set_service(SimpleNamespace(config=SimpleNamespace(
            crypto=SimpleNamespace(
                symbol_weights={"XRPUSDT": 2.0},
                trend_filter_enabled=True,
                trend_sma_days=20,
                trend_tolerance_pct=0.05,
            ),
        )))
        lines: list[str] = []
        self.bot._append_module_context(
            lines, "crypto",
            {"fear_and_greed": 30, "funding_rates": {}, "price_sources": {}},
        )
        joined = "\n".join(lines)
        self.assertIn("XRPUSDT ×2", joined)
        self.assertIn("20d SMA", joined)

    def test_predictions_context_lists_categories(self) -> None:
        lines: list[str] = []
        self.bot._append_module_context(
            lines, "predictions",
            {
                "markets_discovered": 40,
                "markets_scanned": 40,
                "can_trade": False,
                "categories": ["crypto", "economics", "weather"],
            },
        )
        self.assertIn("crypto, economics, weather", "\n".join(lines))


class CommandSurfacesNewSettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        config = SimpleNamespace(bot_token="token", chat_id="123", is_configured=True)
        self.bot = TradingTelegramBot(config, logging.getLogger("test_telegram_bot"))

    async def test_cmd_config_shows_xrp_and_kalshi_enhancements(self) -> None:
        cfg = SimpleNamespace(
            paper_mode=True,
            initial_cash=100_000.0,
            risk=SimpleNamespace(
                max_position_fraction=0.10,
                daily_halt_drawdown=0.05,
                kill_switch_drawdown=0.15,
            ),
            crypto=SimpleNamespace(
                symbols=["BTCUSDT", "XRPUSDT"],
                buy_fear_threshold=35,
                sell_fear_threshold=75,
                require_negative_funding=False,
                take_profit_pct=0.15,
                stop_loss_pct=0.10,
                symbol_weights={"XRPUSDT": 2.0},
                take_profit_overrides={"XRPUSDT": 0.25},
                stop_loss_overrides={},
                trend_filter_enabled=True,
                trend_sma_days=20,
                trend_tolerance_pct=0.05,
            ),
            stock=SimpleNamespace(
                max_pe_ratio=15, max_pb_ratio=1.5,
                max_debt_to_equity=0.5, vix_buy_threshold=25,
            ),
            altcoin_rotation=SimpleNamespace(
                enabled=True, fee_pct=0.006, scout_multiplier=3,
                check_interval_minutes=30, qty_memory_cycles=336,
            ),
            predictions=SimpleNamespace(
                enabled=True, buy_yes_threshold=0.45, buy_no_threshold=0.55,
                require_crypto_fear_greed=False,
                max_pages_per_series=3, discover_series=False,
            ),
            polymarket_strategy=SimpleNamespace(enabled=False),
            scheduler=SimpleNamespace(
                crypto_interval_minutes=30, stock_check_hour=15,
                heartbeat_enabled=False, heartbeat_interval_minutes=60,
            ),
            trading_agents=SimpleNamespace(
                enabled=True, confirmation_mode="boost",
                circuit_breaker_enabled=True,
            ),
        )
        self.bot.set_service(SimpleNamespace(config=cfg))
        update = _make_update()

        await self.bot._cmd_config(update, SimpleNamespace(args=[]))

        update.message.reply_text.assert_awaited_once()
        message = update.message.reply_text.await_args.args[0]
        self.assertIn("XRPUSDT ×2", message)
        self.assertIn("XRPUSDT 25%", message)
        self.assertIn("sma20-gate", message)
        self.assertIn("mem 336cy", message)
        self.assertIn("pages≤3", message)
        self.assertIn("breaker on", message)

    async def test_cmd_ai_shows_circuit_breaker_state(self) -> None:
        cfg = SimpleNamespace(
            trading_agents=SimpleNamespace(
                enabled=True,
                confirmation_mode="boost",
                quick_model="claude-haiku-4-5-20251001",
                deep_model="claude-sonnet-4-6",
                max_cost_per_month=5.0,
                multi_analyst_enabled=False,
                selected_analysts=[],
            ),
        )
        ai = SimpleNamespace(
            get_monthly_cost=lambda: 0.50,
            get_budget_remaining=lambda: 4.50,
            get_circuit_breaker_status=lambda: {
                "state": "open",
                "failure_count": 3,
                "failure_threshold": 3,
                "cooldown_seconds": 300,
                "remaining_open_seconds": 120.0,
            },
        )
        self.bot.set_service(SimpleNamespace(config=cfg, ai_service=ai))
        update = _make_update()

        await self.bot._cmd_ai(update, SimpleNamespace(args=[]))

        update.message.reply_text.assert_awaited_once()
        message = update.message.reply_text.await_args.args[0]
        self.assertIn("Breaker    open (3/3 failures), retry in 120s", message)


class SportsPickFormatTests(unittest.TestCase):
    NO_BET = {"game": "A @ B", "yes_team": "B Team", "p_yes": 0.768, "sd": 0.016, "side": "no",
              "pick": "A Team", "edge": 0.002, "net_edge": None, "yes_ask": 0.78, "no_ask": 0.22,
              "line_move_to_yes": 0.031}
    BET = {"game": "C @ D", "yes_team": "D Team", "p_yes": 0.40, "sd": 0.02, "side": "yes",
           "pick": "D Team", "edge": 0.07, "net_edge": 0.05, "yes_ask": 0.33, "no_ask": 0.69}

    def test_no_edge_game_reads_no_bet_and_names_no_pick(self):
        line = _format_sports_pick(self.NO_BET)
        self.assertIn("no bet", line)
        self.assertNotIn("A Team", line)          # no misleading "pick"
        self.assertNotIn("ask", line)
        self.assertIn("B Team win prob 76.8%", line)

    def test_a_bet_is_labelled_with_side_ask_and_edge(self):
        line = _format_sports_pick(self.BET)
        self.assertIn("BET: D Team", line)
        self.assertIn("YES ask 0.33", line)
        self.assertIn("net edge +5.0%", line)

    def test_bets_sort_ahead_of_no_bet_games(self):
        lines = _format_sports_picks([self.NO_BET, self.BET])
        self.assertIn("BET:", lines[0])
        self.assertIn("no bet", lines[1])

    def test_team_names_are_html_escaped(self):
        line = _format_sports_pick({**self.NO_BET, "game": "A & B <x>"})
        self.assertIn("A &amp; B &lt;x&gt;", line)


if __name__ == "__main__":
    unittest.main()
