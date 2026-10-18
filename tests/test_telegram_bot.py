from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from trading_bot.telegram_bot import (
    TradingTelegramBot,
    _format_auto_parlays,
    _build_line,
    _storage_line,
    _action_picks_line,
    _skip_examples,
    _skip_reason_summary,
    _sports_mode_line,
    _format_lead_line,
    _format_sports_pick,
    _find_bet_options,
    _format_bet_preview,
    _format_parlay_preview,
    _format_parlay_result,
    _parse_bet_args,
    _parse_parlay_args,
    _parse_slip_caption,
    _caption_wants_bet,
    _route_to_command,
    _slip_price,
    _parse_stack_args,
    _format_slip_summary,
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


class ParlayLogFormatTests(unittest.TestCase):
    def test_lists_what_was_tried_and_why_it_did_not_fill(self):
        from trading_bot.telegram_bot import _format_parlay_log
        text = "\n".join(_format_parlay_log([
            {"game": "Bills @ Ravens", "ok": False, "why": "Nothing bought: no quote at or under 31\u00a2. The best price offered was 38\u00a2 all-in."},
            {"game": "Jets @ Patriots", "ok": True, "why": ""},
        ]))
        self.assertIn("2 tried, 1 bought", text)
        self.assertIn("Bills @ Ravens: Nothing bought: no quote at or under 31", text)
        self.assertIn("Jets @ Patriots: bought", text)
        self.assertEqual(_format_parlay_log([]), [])
        self.assertEqual(_format_parlay_log(None), [])


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
            gdp_nowcast=SimpleNamespace(enabled=True, min_edge=0.07),
            crypto_markets=SimpleNamespace(enabled=True, min_edge=0.08),
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
        self.assertIn("GDP         on", message)
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

    def test_action_pick_is_labelled_honestly(self):
        pick = {**self.NO_BET, "net_edge": -0.0029, "action": True, "side": "no", "pick": "A Team"}
        line = _format_sports_pick(pick)
        self.assertIn("PICK: A Team", line)
        self.assertIn("small action bet", line)
        self.assertIn("-0.3% after fees", line)
        self.assertNotIn("BET:", line)

    def test_bets_sort_ahead_of_no_bet_games(self):
        lines = _format_sports_picks([self.NO_BET, self.BET])
        self.assertIn("BET:", lines[0])
        self.assertIn("no bet", lines[1])

    def test_team_names_are_html_escaped(self):
        line = _format_sports_pick({**self.NO_BET, "game": "A & B <x>"})
        self.assertIn("A &amp; B &lt;x&gt;", line)


_GAMES = [
    {"league": "mlb", "game": "Guardians at Tigers", "start": "2026-10-12T18:00:00+00:00", "start_label": "Sun Oct 12 1:00 PM",
     "options": [
         {"label": "Cleveland Guardians", "side": "yes", "ticker": "KXMLB-CLE", "ask": 0.45, "fee": 0.02, "p": 0.47},
         {"label": "Detroit Tigers", "side": "no", "ticker": "KXMLB-CLE", "ask": 0.57, "fee": 0.02, "p": 0.53}]},
    {"league": "nfl", "game": "Browns at Steelers", "start": "2026-10-12T20:00:00+00:00", "start_label": "Sun Oct 12 3:00 PM",
     "options": [
         {"label": "Cleveland Browns", "side": "yes", "ticker": "KXNFL-CLE", "ask": 0.38, "fee": 0.02, "p": 0.41},
         {"label": "Pittsburgh Steelers", "side": "yes", "ticker": "KXNFL-PIT", "ask": 0.64, "fee": 0.02, "p": 0.59}]},
]


class BetParsingTests(unittest.TestCase):
    def test_parse_variants(self):
        self.assertEqual(_parse_bet_args(["cleveland", "5"]),
                         {"game_no": None, "query": "cleveland", "amount": 5.0, "now": False})
        self.assertEqual(_parse_bet_args(["2", "browns", "$7.50", "now"]),
                         {"game_no": 2, "query": "browns", "amount": 7.5, "now": True})
        self.assertEqual(_parse_bet_args(["cleveland", "browns"])["amount"], None)
        self.assertEqual(_parse_bet_args(["3"])["query"], "3")

    def test_find_matches_team_names_across_games(self):
        rows = _find_bet_options(_GAMES, None, "cleveland")
        self.assertEqual([r[2]["label"] for r in rows], ["Cleveland Guardians", "Cleveland Browns"])
        self.assertEqual(len(_find_bet_options(_GAMES, None, "cleveland browns")), 1)
        self.assertEqual(len(_find_bet_options(_GAMES, 2, "cleveland")), 1)
        self.assertEqual(_find_bet_options(_GAMES, None, "lakers"), [])

    def test_preview_is_honest_when_model_disagrees(self):
        quote = {"ask": 0.38, "fee_each": 0.02, "quantity": 10, "cost": 4.0, "model_prob": 0.30,
                 "pick": "Cleveland Browns", "game": "Browns at Steelers", "league": "nfl", "live": True}
        text = _format_bet_preview(quote)
        self.assertIn("LIVE money", text)
        self.assertIn("Heads up", text)
        self.assertIn("/confirm", text)
        quote["model_prob"] = 0.50
        self.assertIn("Model agrees", _format_bet_preview(quote))
        quote["model_prob"] = None
        self.assertIn("purely your call", _format_bet_preview(quote))


class ManualBetFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        config = SimpleNamespace(bot_token="token", chat_id="123", is_configured=True)
        self.bot = TradingTelegramBot(config, logging.getLogger("test_manual_bets"))
        self.service = SimpleNamespace(
            list_sports_games=MagicMock(return_value=_GAMES),
            quote_sports_bet=MagicMock(return_value={
                "ok": True, "ask": 0.38, "fee_each": 0.02, "quantity": 12, "cost": 4.8, "stake": 5.0,
                "model_prob": 0.41, "pick": "Cleveland Browns", "game": "Browns at Steelers",
                "league": "nfl", "live": True}),
            place_sports_bet=MagicMock(return_value={
                "ok": True, "live": True, "filled": 12, "quantity": 12, "ask": 0.38, "spent": 4.8,
                "pick": "Cleveland Browns", "game": "Browns at Steelers"}),
        )
        self.bot.set_service(self.service)

    def _update(self, chat_id="123"):
        return SimpleNamespace(effective_chat=SimpleNamespace(id=int(chat_id)),
                               message=SimpleNamespace(reply_text=AsyncMock()))

    def _texts(self, update):
        return [c.args[0] for c in update.message.reply_text.await_args_list]

    async def test_games_lists_numbered_games_and_filters(self):
        update = self._update()
        await self.bot._cmd_games(update, SimpleNamespace(args=[]))
        text = "\n".join(self._texts(update))
        self.assertIn("1. <b>MLB</b>", text)
        self.assertIn("Cleveland Browns 38\u00a2 (model 41%)", text)
        update = self._update()
        await self.bot._cmd_games(update, SimpleNamespace(args=["nfl"]))
        self.assertEqual(len(self.bot._games), 1)

    async def test_ambiguous_team_asks_which_one(self):
        update = self._update()
        await self.bot._cmd_bet(update, SimpleNamespace(args=["cleveland", "5"]))
        self.assertIn("2 matches", self._texts(update)[-1])
        self.service.quote_sports_bet.assert_not_called()
        self.assertIsNone(self.bot._pending_bet)

    async def test_preview_then_confirm_places_the_bet(self):
        update = self._update()
        await self.bot._cmd_bet(update, SimpleNamespace(args=["browns", "5"]))
        self.service.quote_sports_bet.assert_called_once_with("KXNFL-CLE", "yes", 5.0)
        self.service.place_sports_bet.assert_not_called()  # preview alone never trades
        self.assertIn("Bet preview", self._texts(update)[-1])
        update = self._update()
        await self.bot._cmd_confirm(update, SimpleNamespace(args=[]))
        args = self.service.place_sports_bet.call_args.args
        self.assertEqual(args[:3], ("KXNFL-CLE", "yes", 5.0))
        self.assertAlmostEqual(args[3], 0.40)  # agreed ask + slip tolerance
        self.assertIn("Bet placed", self._texts(update)[-1])
        self.assertIsNone(self.bot._pending_bet)
        # a second /confirm has nothing to place
        update = self._update()
        await self.bot._cmd_confirm(update, SimpleNamespace(args=[]))
        self.assertEqual(self.service.place_sports_bet.call_count, 1)

    async def test_cancel_drops_the_pending_bet(self):
        await self.bot._cmd_bet(self._update(), SimpleNamespace(args=["browns"]))
        update = self._update()
        await self.bot._cmd_cancel(update, SimpleNamespace(args=[]))
        await self.bot._cmd_confirm(self._update(), SimpleNamespace(args=[]))
        self.service.place_sports_bet.assert_not_called()

    async def test_expired_preview_is_not_placed(self):
        await self.bot._cmd_bet(self._update(), SimpleNamespace(args=["browns"]))
        self.bot._pending_bet["expires"] -= 1000
        update = self._update()
        await self.bot._cmd_confirm(update, SimpleNamespace(args=[]))
        self.service.place_sports_bet.assert_not_called()
        self.assertIn("Nothing to confirm", self._texts(update)[-1])

    async def test_now_flag_places_immediately(self):
        update = self._update()
        await self.bot._cmd_bet(update, SimpleNamespace(args=["browns", "5", "now"]))
        self.service.place_sports_bet.assert_called_once()

    async def test_refused_quote_creates_no_pending_bet(self):
        self.service.quote_sports_bet.return_value = {"ok": False, "reason": "You already have a bet on this game."}
        update = self._update()
        await self.bot._cmd_bet(update, SimpleNamespace(args=["browns"]))
        self.assertIsNone(self.bot._pending_bet)
        self.assertIn("already have a bet", self._texts(update)[-1])

    async def test_failed_placement_is_reported(self):
        self.service.place_sports_bet.return_value = {"ok": False, "reason": "price moved"}
        await self.bot._cmd_bet(self._update(), SimpleNamespace(args=["browns"]))
        update = self._update()
        await self.bot._cmd_confirm(update, SimpleNamespace(args=[]))
        self.assertIn("Not placed: price moved", self._texts(update)[-1])

    async def test_other_chats_cannot_bet(self):
        for handler in (self.bot._cmd_games, self.bot._cmd_bet, self.bot._cmd_confirm, self.bot._cmd_cancel):
            await handler(self._update(chat_id="999"), SimpleNamespace(args=["browns"]))
        self.service.list_sports_games.assert_not_called()
        self.service.quote_sports_bet.assert_not_called()
        self.service.place_sports_bet.assert_not_called()


class ParlayCommandTests(unittest.IsolatedAsyncioTestCase):
    PLAN = {"ok": True, "kind": "parlay", "stake": 5.0, "max_price": 0.25, "fair": 0.25, "contracts": 20,
            "max_cost": 5.0, "payout": 20.0,
            "legs": [{"pick": "Cleveland Browns", "game": "Browns at Steelers", "p": 0.5},
                     {"pick": "Cleveland Guardians", "game": "Guardians at Tigers", "p": 0.5}]}

    def setUp(self) -> None:
        config = SimpleNamespace(bot_token="token", chat_id="123", is_configured=True)
        self.bot = TradingTelegramBot(config, logging.getLogger("test_parlay"))
        self.service = SimpleNamespace(
            list_sports_games=MagicMock(return_value=_GAMES),
            quote_sports_parlay=MagicMock(return_value=dict(self.PLAN)),
            place_sports_parlay=MagicMock(return_value={
                "ok": True, "filled": 20, "price": 0.2, "fee_each": 0.01, "spent": 4.2, "payout": 20.0,
                "legs": self.PLAN["legs"], "warning": ""}),
            place_sports_bet=MagicMock(), quote_sports_bet=MagicMock(),
        )
        self.bot.set_service(self.service)
        self.bot._games = _GAMES

    def _update(self):
        return SimpleNamespace(effective_chat=SimpleNamespace(id=123),
                               message=SimpleNamespace(reply_text=AsyncMock()))

    def test_parse_parlay_args(self):
        parsed = _parse_parlay_args("browns + guardians 5 max 30".split())
        self.assertEqual([l["query"] for l in parsed["legs"]], ["browns", "guardians"])
        self.assertEqual(parsed["amount"], 5.0)
        self.assertAlmostEqual(parsed["max_price"], 0.30)
        parsed = _parse_parlay_args("2 browns, 1 guardians $2.50".split())
        self.assertEqual([(l["game_no"], l["query"]) for l in parsed["legs"]], [(2, "browns"), (1, "guardians")])
        self.assertEqual((parsed["amount"], parsed["max_price"]), (2.5, None))
        self.assertEqual(_parse_parlay_args(["browns"])["legs"][0]["query"], "browns")

    async def test_preview_then_confirm_requests_quotes(self):
        update = self._update()
        await self.bot._cmd_parlay(update, SimpleNamespace(args="2 browns + 1 guardians 5".split()))
        legs, amount, max_price = self.service.quote_sports_parlay.call_args.args
        self.assertEqual([l["ticker"] for l in legs], ["KXNFL-CLE", "KXMLB-CLE"])
        self.assertEqual((amount, max_price), (5.0, None))
        self.service.place_sports_parlay.assert_not_called()
        self.assertIn("Parlay preview", update.message.reply_text.await_args.args[0])
        update = self._update()
        await self.bot._cmd_confirm(update, SimpleNamespace(args=[]))
        self.assertEqual(self.service.place_sports_parlay.call_args.args, (legs, 5.0, 0.25))
        self.assertIn("Parlay placed", update.message.reply_text.await_args.args[0])
        self.service.place_sports_bet.assert_not_called()

    async def test_confirm_can_restate_the_stake(self):
        await self.bot._cmd_parlay(self._update(), SimpleNamespace(args="2 browns + 1 guardians 5".split()))
        update = self._update()
        await self.bot._confirm_pending(update, stake=10.0)
        args = self.service.place_sports_parlay.call_args.args
        self.assertEqual(args[1], 10.0)
        self.assertEqual(args[2], 0.25)  # the previewed price ceiling is kept

    async def test_needs_two_legs(self):
        update = self._update()
        await self.bot._cmd_parlay(update, SimpleNamespace(args=["browns", "5"]))
        self.service.quote_sports_parlay.assert_not_called()
        self.assertIn("2 or more", update.message.reply_text.await_args.args[0])

    async def test_ambiguous_leg_asks_which(self):
        update = self._update()
        await self.bot._cmd_parlay(update, SimpleNamespace(args="cleveland + steelers 5".split()))
        self.service.quote_sports_parlay.assert_not_called()
        self.assertIn("2 matches", update.message.reply_text.await_args.args[0])

    async def test_refused_plan_leaves_nothing_pending(self):
        self.service.quote_sports_parlay.return_value = {"ok": False, "reason": "no combo"}
        update = self._update()
        await self.bot._cmd_parlay(update, SimpleNamespace(args="browns + guardians".split()))
        self.assertIsNone(self.bot._pending_bet)
        self.assertIn("no combo", update.message.reply_text.await_args.args[0])

    async def test_props_lists_numbered_markets_and_parlay_uses_them(self):
        self.service.list_sports_props = MagicMock(return_value=[
            {"ticker": "KXNFLPASSYDS-X-A-250", "player": "Josh Allen", "stat": "250+ passing yards", "yes_ask": 0.4,
             "game": "Bills at Ravens", "start_label": "Sun Oct 4 8:00 AM"}])
        update = self._update()
        await self.bot._cmd_props(update, SimpleNamespace(args=["allen"]))
        self.service.list_sports_props.assert_called_once_with("allen")
        text = "\n".join(c.args[0] for c in update.message.reply_text.await_args_list)
        self.assertIn("<b>Josh Allen</b>", text)
        self.assertIn("passing yards: p1 250+ 40\u00a2", text)
        self.bot._games = []  # prop-only legs don't need the games board
        update = self._update()
        await self.bot._cmd_parlay(update, SimpleNamespace(args="p1 + p1 3 max 20".split()))
        legs, amount, max_price = self.service.quote_sports_parlay.call_args.args
        self.assertEqual(legs, [{"ticker": "KXNFLPASSYDS-X-A-250", "side": "yes"}] * 2)
        self.assertEqual((amount, max_price), (3.0, 0.2))
        self.service.list_sports_games.assert_not_called()

    def test_props_ladder_groups_thresholds_by_player_and_stat(self):
        from trading_bot.telegram_bot import _format_props_ladder
        props = [
            {"player": "Jalen Hurts", "stat": "175+ passing yards", "yes_ask": 0.58, "game": "PHI @ JAC", "start_label": "Sun 1:00 PM"},
            {"player": "Jalen Hurts", "stat": "200+ passing yards", "yes_ask": 0.43, "game": "PHI @ JAC", "start_label": "Sun 1:00 PM"},
            {"player": "Jalen Hurts", "stat": "1+ touchdowns", "yes_ask": 0.40, "game": "PHI @ JAC", "start_label": "Sun 1:00 PM"},
        ]
        text = "\n".join(_format_props_ladder(props))
        self.assertEqual(text.count("Jalen Hurts"), 1)
        self.assertIn("passing yards: p1 175+ 58\u00a2 \u00b7 p2 200+ 43\u00a2", text)
        self.assertIn("touchdowns: p3 1+ 40\u00a2", text)

    async def test_parlay_leg_can_be_a_player_named_in_plain_words(self):
        prop = {"ticker": "KXNFLPASSYDS-X-HURTS-200", "player": "Jalen Hurts", "stat": "200+ passing yards",
                "yes_ask": 0.43, "game": "PHI @ JAC", "start_label": "x"}
        rush = {"ticker": "KXNFLRSHYDS-X-BARK-80", "player": "Saquon Barkley", "stat": "80+ rushing yards",
                "yes_ask": 0.30, "game": "PHI @ JAC", "start_label": "x"}
        self.service.list_sports_props = MagicMock(side_effect=lambda q: [prop] if "hurts" in q else [rush])
        update = self._update()
        await self.bot._cmd_parlay(update, SimpleNamespace(args="hurts 200 passing + barkley 80 rushing 3".split()))
        legs = self.service.quote_sports_parlay.call_args.args[0]
        self.assertEqual([l["ticker"] for l in legs], [prop["ticker"], rush["ticker"]])
        self.assertEqual(self.service.quote_sports_parlay.call_args.args[1], 3.0)

    async def test_ambiguous_player_leg_lists_choices(self):
        many = [{"ticker": f"T{i}", "player": "Jalen Hurts", "stat": f"{100 + 25 * i}+ passing yards",
                 "yes_ask": 0.5, "game": "PHI @ JAC", "start_label": "x"} for i in range(3)]
        self.service.list_sports_props = MagicMock(return_value=many)
        update = self._update()
        await self.bot._cmd_parlay(update, SimpleNamespace(args="hurts + browns 3".split()))
        self.service.quote_sports_parlay.assert_not_called()
        text = "\n".join(c.args[0] for c in update.message.reply_text.await_args_list)
        self.assertIn("3 player markets match 'hurts'", text)
        self.assertEqual([p["ticker"] for p in self.bot._props], ["T0", "T1", "T2"])

    def test_parse_stack_args(self):
        self.assertEqual(_parse_stack_args("eagles jaguars legs 9 p 85 2 max 20".split()),
                         {"query": "eagles jaguars", "legs": 9, "leg_prob": 0.85, "amount": 2.0, "max_price": 0.2})
        self.assertEqual(_parse_stack_args(["eagles"]), {"query": "eagles", "legs": 6, "leg_prob": 0.85,
                                                         "amount": None, "max_price": None})
        self.assertEqual(_parse_stack_args("eagles jaguars p 90 $1.50".split())["leg_prob"], 0.9)
        self.assertEqual(_parse_stack_args("eagles jaguars p 90 $1.50".split())["amount"], 1.5)

    async def test_stack_builds_previews_and_confirm_buys_it(self):
        legs = [{"ticker": "A-1", "side": "yes"}, {"ticker": "S-1", "side": "no"}]
        self.service.build_sports_stack = MagicMock(return_value={
            "ok": True, "game": "Eagles @ Jaguars", "legs": legs, "leg_prob": 0.85, "reference": 0.7, "rows": []})
        update = self._update()
        await self.bot._cmd_stack(update, SimpleNamespace(args="eagles jaguars legs 2 p 85 3".split()))
        self.service.build_sports_stack.assert_called_once_with("eagles jaguars", 2, 0.85)
        self.assertEqual(self.service.quote_sports_parlay.call_args.args, (legs, 3.0, None))
        text = update.message.reply_text.await_args.args[0]
        self.assertIn("Stack for Eagles @ Jaguars: 2 legs", text)
        self.assertIn("Parlay preview", text)
        self.service.place_sports_parlay.assert_not_called()
        await self.bot._cmd_confirm(self._update(), SimpleNamespace(args=[]))
        self.assertEqual(self.service.place_sports_parlay.call_args.args[0], legs)

    async def test_stack_errors_are_reported_and_nothing_is_pending(self):
        self.service.build_sports_stack = MagicMock(return_value={"ok": False, "reason": "matches several games"})
        update = self._update()
        await self.bot._cmd_stack(update, SimpleNamespace(args=["a"]))
        self.assertIn("matches several games", update.message.reply_text.await_args.args[0])
        self.assertIsNone(self.bot._pending_bet)
        update = self._update()
        await self.bot._cmd_stack(update, SimpleNamespace(args=[]))
        self.assertIn("Name the game", update.message.reply_text.await_args.args[0])
        update = self._update()
        await self.bot._cmd_stack(update, SimpleNamespace(args="eagles p 99".split()))
        self.assertIn("between 5 and 97", update.message.reply_text.await_args.args[0])

    async def test_spread_no_leg_keeps_its_side_from_props(self):
        spread = {"ticker": "KXNFLSPREAD-X-BAL10", "side": "no", "player": "Spread", "stat": "BAL don't win by over 9.5",
                  "yes_ask": 0.77, "game": "BUF @ BAL", "start_label": "x"}
        self.bot._props = [spread, dict(spread, ticker="Y", side="yes")]
        update = self._update()
        await self.bot._cmd_parlay(update, SimpleNamespace(args="p1 + p2 2".split()))
        legs = self.service.quote_sports_parlay.call_args.args[0]
        self.assertEqual([l["side"] for l in legs], ["no", "yes"])

    def test_slip_caption_and_summary(self):
        self.assertEqual(_parse_slip_caption("$5 max 30"), {"amount": 5.0, "max_price": 0.3})
        self.assertEqual(_parse_slip_caption(""), {"amount": None, "max_price": None})
        text = _format_slip_summary({
            "slip": {"game": "PHI @ JAC", "odds": "+478"},
            "matched": [{"pick": "Jalen Hurts 125+ passing yards", "ask": 0.85}],
            "unmatched": [{"text": "Strange 25+ <Yards>", "reason": "Kalshi has 20+, 30+"}]})
        self.assertIn("2 legs", text)
        self.assertIn("(+478 at the book)", text)
        self.assertIn("Jalen Hurts 125+ passing yards \u2014 85\u00a2", text)
        self.assertIn("Strange 25+ &lt;Yards&gt;", text)

    def _photo_update(self, caption="2", chat_id=123):
        tg_file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"\x89PNGdata")))
        photo = SimpleNamespace(get_file=AsyncMock(return_value=tg_file))
        message = SimpleNamespace(photo=[photo, photo], document=None, caption=caption, reply_text=AsyncMock())
        return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id), message=message)

    async def test_photo_of_a_slip_previews_the_matched_parlay(self):
        self.service.read_sports_slip = MagicMock(return_value={
            "ok": True, "slip": {"game": "PHI @ JAC", "odds": "+478"},
            "matched": [{"pick": "A", "ask": 0.8, "ticker": "T1", "side": "yes"},
                        {"pick": "B", "ask": 0.7, "ticker": "T2", "side": "no"}],
            "unmatched": [{"text": "C 25+", "reason": "not listed"}], "game": "PHI @ JAC"})
        update = self._photo_update("2 max 30")
        await self.bot._on_photo(update, SimpleNamespace(args=[]))
        data, media_type = self.service.read_sports_slip.call_args.args
        self.assertEqual((data, media_type), (b"\x89PNGdata", "image/jpeg"))
        legs, amount, max_price, anchor = self.service.quote_sports_parlay.call_args.args
        self.assertEqual(legs, [{"ticker": "T1", "side": "yes"}, {"ticker": "T2", "side": "no"}])
        self.assertEqual((amount, max_price), (2.0, 0.3))
        self.assertIsNone(anchor)  # a shorter parlay isn't comparable to the slip's price
        texts = [c.args[0] for c in update.message.reply_text.await_args_list]
        self.assertTrue(any("Read your slip" in t for t in texts))
        self.assertIn("only the 2 matched legs", texts[-1])
        self.assertIn("Parlay preview", texts[-1])
        self.service.place_sports_parlay.assert_not_called()  # reading a picture never buys anything
        self.assertEqual(self.bot._pending_bet["kind"], "parlay")

    def test_caption_intent_and_slip_price(self):
        self.assertTrue(_caption_wants_bet("bet $10 with this parlay"))
        self.assertTrue(_caption_wants_bet("Put 10 on this"))
        self.assertFalse(_caption_wants_bet("$10"))
        self.assertFalse(_caption_wants_bet("what would a $10 bet pay"))
        self.assertFalse(_caption_wants_bet(""))
        self.assertAlmostEqual(_slip_price("+478"), 100 / 578)
        self.assertAlmostEqual(_slip_price("-150"), 0.6)
        self.assertIsNone(_slip_price(None))
        self.assertIsNone(_slip_price("n/a"))

    def _full_match(self, unmatched=()):
        return {"ok": True, "slip": {"game": "PHI @ JAC", "odds": "+478"},
                "matched": [{"pick": "A", "ask": 0.8, "ticker": "T1", "side": "yes"},
                            {"pick": "B", "ask": 0.7, "ticker": "T2", "side": "no"}],
                "unmatched": list(unmatched), "game": "PHI @ JAC"}

    async def test_bet_caption_with_every_leg_matched_places_the_parlay_without_confirm(self):
        self.service.read_sports_slip = MagicMock(return_value=self._full_match())
        update = self._photo_update("bet $10 with this parlay")
        await self.bot._on_photo(update, SimpleNamespace(args=[]))
        legs, amount, max_price, anchor = self.service.quote_sports_parlay.call_args.args
        self.assertEqual((amount, max_price), (10.0, None))
        self.assertAlmostEqual(anchor, 100 / 578)  # the slip's own price is an acceptable ceiling
        place_args = self.service.place_sports_parlay.call_args.args
        self.assertEqual((place_args[0], place_args[1], place_args[2]), (legs, 5.0, 0.25))  # plan's stake and ceiling
        texts = [c.args[0] for c in update.message.reply_text.await_args_list]
        self.assertTrue(any(t.startswith("Placing $5.00") for t in texts))
        self.assertIn("Parlay placed", texts[-1])
        self.assertIsNone(self.bot._pending_bet)

    async def test_bet_caption_with_missing_legs_previews_instead_of_placing(self):
        self.service.read_sports_slip = MagicMock(return_value=self._full_match(
            unmatched=[{"text": "C", "reason": "not listed"}]))
        update = self._photo_update("bet $10 on this")
        await self.bot._on_photo(update, SimpleNamespace(args=[]))
        self.service.place_sports_parlay.assert_not_called()
        self.assertIsNone(self.service.quote_sports_parlay.call_args.args[3])
        self.assertIn("didn't place it", update.message.reply_text.await_args.args[0])
        self.assertIsNotNone(self.bot._pending_bet)

    async def test_plain_caption_still_only_previews(self):
        self.service.read_sports_slip = MagicMock(return_value=self._full_match())
        update = self._photo_update("10")
        await self.bot._on_photo(update, SimpleNamespace(args=[]))
        self.service.place_sports_parlay.assert_not_called()
        self.assertIn("Parlay preview", update.message.reply_text.await_args.args[0])

    async def test_photo_with_too_few_matches_or_a_read_error_leaves_nothing_pending(self):
        self.service.read_sports_slip = MagicMock(return_value={
            "ok": True, "slip": {}, "matched": [{"pick": "A", "ask": 0.8, "ticker": "T1", "side": "yes"}], "unmatched": []})
        update = self._photo_update()
        await self.bot._on_photo(update, SimpleNamespace(args=[]))
        self.assertIn("Fewer than 2", update.message.reply_text.await_args.args[0])
        self.service.quote_sports_parlay.assert_not_called()
        self.service.read_sports_slip.return_value = {"ok": False, "reason": "needs ANTHROPIC_API_KEY"}
        update = self._photo_update()
        await self.bot._on_photo(update, SimpleNamespace(args=[]))
        self.assertIn("needs ANTHROPIC_API_KEY", update.message.reply_text.await_args.args[0])
        self.assertIsNone(self.bot._pending_bet)

    async def test_photo_from_another_chat_is_ignored(self):
        self.service.read_sports_slip = MagicMock()
        await self.bot._on_photo(self._photo_update(chat_id=999), SimpleNamespace(args=[]))
        self.service.read_sports_slip.assert_not_called()

    SLIP_TEXT = (
        "Make this bet at $10\n\n\u2022 Brian Thomas Jr. 2+ receptions\n\u2022 Lawrence 150+ passing yards\n"
        "\u2022 the Eagles +16.5 spread"
    )

    def _text_update(self, text, chat_id=123):
        return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id),
                               message=SimpleNamespace(text=text, reply_text=AsyncMock()))

    def _typed_match(self, unmatched=()):
        result = self._full_match(unmatched)
        result["slip"] = {"game": None, "odds": None}  # typed lists have no book price
        return result

    async def test_typed_leg_list_is_matched_without_ai_and_placed_like_a_bet_caption(self):
        self.service.match_typed_slip = MagicMock(return_value=self._typed_match())
        self.bot._route_text = MagicMock(side_effect=AssertionError("typed slips must not call the AI router"))
        update = self._text_update(self.SLIP_TEXT)
        await self.bot._on_text(update, SimpleNamespace(args=[]))
        slip = self.service.match_typed_slip.call_args.args[0]
        self.assertEqual([l["type"] for l in slip["legs"]], ["player_stat", "player_stat", "spread"])
        legs, amount, max_price, anchor = self.service.quote_sports_parlay.call_args.args
        self.assertEqual((amount, max_price, anchor), (10.0, None, None))  # a typed list carries no book price
        texts = [c.args[0] for c in update.message.reply_text.await_args_list]
        self.assertTrue(any(t.startswith("Placing $5.00") for t in texts))
        self.assertIn("Parlay placed", texts[-1])

    async def test_typed_leg_list_without_a_bet_word_only_previews(self):
        self.service.match_typed_slip = MagicMock(return_value=self._typed_match())
        update = self._text_update(self.SLIP_TEXT.replace("Make this bet at $10", "$10"))
        await self.bot._on_text(update, SimpleNamespace(args=[]))
        self.service.place_sports_parlay.assert_not_called()
        self.assertIn("Parlay preview", update.message.reply_text.await_args.args[0])

    async def test_typed_leg_list_with_unmatched_legs_is_not_placed(self):
        self.service.match_typed_slip = MagicMock(return_value=self._typed_match(
            unmatched=[{"text": "Lawrence 150+", "reason": "more than one player matches"}]))
        update = self._text_update(self.SLIP_TEXT)
        await self.bot._on_text(update, SimpleNamespace(args=[]))
        self.service.place_sports_parlay.assert_not_called()
        self.assertIn("didn't place it", update.message.reply_text.await_args.args[0])

    async def test_typed_leg_list_from_another_chat_is_ignored(self):
        self.service.match_typed_slip = MagicMock()
        await self.bot._on_text(self._text_update(self.SLIP_TEXT, chat_id=999), SimpleNamespace(args=[]))
        self.service.match_typed_slip.assert_not_called()

    async def test_unknown_prop_number_and_empty_search(self):
        self.service.list_sports_props = MagicMock(return_value=[])
        update = self._update()
        await self.bot._cmd_props(update, SimpleNamespace(args=[]))
        self.service.list_sports_props.assert_not_called()
        update = self._update()
        await self.bot._cmd_parlay(update, SimpleNamespace(args="p4 + p5".split()))
        self.assertIn("No player market p4", update.message.reply_text.await_args.args[0])
        self.service.quote_sports_parlay.assert_not_called()

    def test_auto_parlay_alert_skip_summary_and_mode_line(self):
        lines = _format_auto_parlays([{"game": "BUF @ <BAL>", "legs": ["Allen 200+", "Cook 30+"], "contracts": 40,
                                       "price": 0.5, "spent": 20.0, "payout": 40.0, "warning": "check it"}])
        text = "\n".join(lines)
        self.assertIn("AUTO PARLAY BUF @ &lt;BAL&gt;: Allen 200+ + Cook 30+", text)
        self.assertIn("40 x $0.50 = $20.00 to win $40.00", text)
        self.assertIn("WARNING: check it", text)
        self.assertEqual(_format_auto_parlays(None), [])
        decisions = [{"action": "skip_buy", "reason": "no_edge_after_fees"}] * 3 + [
            {"action": "skip", "reason": "data_fetch_failed"}, {"action": "buy", "reason": "model_edge"}]
        self.assertEqual(_skip_reason_summary(decisions), "no_edge_after_fees x3, data_fetch_failed x1")
        live = SimpleNamespace(config=SimpleNamespace(paper_mode=False))
        self.assertIn("LIVE", _sports_mode_line(live, {"can_trade_live": True}))
        self.assertIn("PAPER_MODE=true", _sports_mode_line(SimpleNamespace(config=SimpleNamespace(paper_mode=True)), {}))
        broken = _sports_mode_line(live, {"can_trade_live": False})
        self.assertIn("even though PAPER_MODE=false", broken)
        self.assertIn("KALSHI_KEY_ID", broken)

    async def test_sports_shows_the_balance_by_shard(self):
        module = {"games_scanned": 1, "markets_matched": 1, "predictions": [], "buys": [], "decisions": [],
                  "can_trade_live": True, "bankroll": {"equity": 76.7, "cash": 58.61, "day_pnl": 0, "stake_scale": 1,
                  "open_exposure": 0, "exposure_budget": 38, "shards": [{"index": 0, "balance": 58.61}, {"index": 1, "balance": 0.0}]}}
        self.service.run = MagicMock(return_value={"modules": {"kalshi_sports": module}})
        self.service.config = SimpleNamespace(paper_mode=False)
        update = self._update()
        await self.bot._cmd_kalshi_sports(update, SimpleNamespace(args=[]))
        text = update.message.reply_text.await_args.args[0]
        self.assertIn("Kalshi balance by shard: #0 $58.61, #1 $0.00", text)
        self.assertIn("cash $58.61", text)

    def test_skip_examples_show_kalshi_s_own_error_text_once_each(self):
        decisions = [
            {"action": "skip_buy", "reason": "order_failed", "signals": {"error": "HTTP 404: <not found>"}},
            {"action": "skip_buy", "reason": "order_failed", "signals": {"error": "HTTP 404: <not found>"}},
            {"action": "skip_buy", "reason": "parlay_not_filled", "signals": {"why": "no quote at or under 50c"}},
            {"action": "skip_buy", "reason": "no_edge_after_fees", "signals": {}},
        ]
        self.assertEqual(_skip_examples(decisions), [
            "order_failed: HTTP 404: &lt;not found&gt;", "parlay_not_filled: no quote at or under 50c"])
        self.assertEqual(_skip_examples(None), [])

    def test_build_line_reports_the_deployed_commit(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"RAILWAY_GIT_COMMIT_SHA": "00be3ee2eb014a88e4c3234e35fd3368054a3a68",
                                     "RAILWAY_GIT_COMMIT_MESSAGE": "Place live orders through V2 <fix>\n\nbody"}):
            self.assertEqual(_build_line(), "Build: 00be3ee \u2014 Place live orders through V2 &lt;fix&gt;")
        with patch.dict(os.environ, {"RAILWAY_GIT_COMMIT_SHA": ""}):
            self.assertIn("unknown", _build_line())

    def test_storage_line_says_whether_state_survives_a_redeploy(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        service = SimpleNamespace(config=SimpleNamespace(state_path=Path("/data/data/state.json")))
        with patch("trading_bot.telegram_bot.os.path.ismount", side_effect=lambda p: str(p) == "/data"):
            line = _storage_line(service)
        self.assertIn("mounted volume /data", line)
        self.assertIn("survives redeploys", line)
        with patch("trading_bot.telegram_bot.os.path.ismount", return_value=False):
            line = _storage_line(service)
        self.assertIn("not a persistent volume", line)
        self.assertIn("read from Kalshi", line)
        self.assertEqual(_storage_line(SimpleNamespace(config=SimpleNamespace(state_path=None))), "Storage: unknown.")

    def test_action_picks_line(self):
        self.assertEqual(_action_picks_line(None), "")
        on = _action_picks_line({"enabled": True, "today": 2, "per_day": 5, "per_scan": 2})
        self.assertIn("2 of 5 used today", on)
        off = _action_picks_line({"enabled": False, "today": 0, "per_day": 5, "per_scan": 2})
        self.assertIn("OFF", off)
        self.assertIn("KALSHI_SPORTS_ACTION_BETS=false", off)

    def test_preview_and_result_wording(self):
        text = _format_parlay_preview(self.PLAN)
        self.assertIn("all must win", text)
        self.assertIn("25\u00a2", text)
        self.assertIn("no quote", text)  # fair-price ceiling warns about little action
        higher = _format_parlay_preview({**self.PLAN, "max_price": 0.32})
        self.assertIn("expected cost", higher)
        props = _format_parlay_preview({**self.PLAN, "fair": None, "reference": 0.18, "max_price": 0.18})
        self.assertIn("Kalshi's own prices put all legs hitting at about 18", props)
        self.assertIn("no view on player stats", props)
        self.assertIn("not placed", _format_parlay_result({"ok": False, "reason": "x"}).lower())
        warned = _format_parlay_result({**self.service.place_sports_parlay.return_value, "warning": "wrong side"})
        self.assertIn("wrong side", warned)


if __name__ == "__main__":
    unittest.main()


def _route(**fields):
    base = {"action": "reply", "team": "", "legs": [], "game": "", "query": "", "filter": "", "amount": None,
            "max_cents": None, "leg_count": None, "leg_percent": None, "reply_text": ""}
    base.update(fields)
    return base


class PlainTextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        config = SimpleNamespace(bot_token="token", chat_id="123", is_configured=True)
        self.bot = TradingTelegramBot(config, logging.getLogger("test_plain_text"))
        self.bot.set_service(SimpleNamespace(config=SimpleNamespace(trading_agents=SimpleNamespace(api_key="k"))))
        self.calls = []
        for name in ("games", "bet", "parlay", "stack", "props", "status", "positions", "confirm", "cancel"):
            setattr(self.bot, f"_cmd_{name}", self._recorder(name))
        self.bot._cmd_kalshi_sports = self._recorder("sports")

        async def confirm_pending(update, stake=None):
            self.calls.append(("confirm", [] if stake is None else [stake]))
        self.bot._confirm_pending = confirm_pending
        self.routes = []
        self.bot._route_text = MagicMock(side_effect=lambda text, pending: self.routes.pop(0))

    def _recorder(self, name):
        async def run(update, context):
            self.calls.append((name, list(context.args or [])))
        return run

    def _update(self, text, chat_id=123):
        return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id),
                               message=SimpleNamespace(text=text, reply_text=AsyncMock()))

    def _wait(self, label="single bet on Browns"):
        import time
        self.bot._pending_bet = {"kind": "bet", "label": label, "expires": time.monotonic() + 600}

    def _ctx(self):
        return SimpleNamespace(args=[], bot=None)

    async def test_route_to_command_builds_slash_arguments(self):
        self.assertEqual(_route_to_command(_route(action="bet", team="cleveland browns", amount=5.0)),
                         ("bet", ["cleveland", "browns", "5"]))
        self.assertEqual(_route_to_command(_route(action="parlay", legs=["hurts 200 passing", "browns"], amount=3.0,
                                                  max_cents=12.0)),
                         ("parlay", ["hurts", "200", "passing", "+", "browns", "3", "max", "12"]))
        self.assertEqual(_route_to_command(_route(action="stack", game="eagles jaguars", leg_count=9, leg_percent=85.0,
                                                  amount=2.0)),
                         ("stack", ["eagles", "jaguars", "legs", "9", "p", "85", "2"]))
        self.assertEqual(_route_to_command(_route(action="games", filter="nfl")), ("games", ["nfl"]))
        self.assertEqual(_route_to_command(_route(action="status")), ("status", []))
        self.assertIsNone(_route_to_command(_route(action="bet")))
        self.assertIsNone(_route_to_command(_route(action="parlay", legs=["only one"])))

    async def test_yes_and_no_use_the_waiting_bet_without_calling_the_model(self):
        self._wait()
        await self.bot._on_text(self._update("Yes!"), self._ctx())
        self._wait()
        await self.bot._on_text(self._update("never mind"), self._ctx())
        self.assertEqual([c[0] for c in self.calls], ["confirm", "cancel"])
        self.bot._route_text.assert_not_called()

    async def test_yes_with_nothing_waiting_or_an_expired_preview_does_not_confirm(self):
        self.routes = [_route(action="reply", reply_text="Nothing is waiting."), _route(action="confirm")]
        update = self._update("yes")
        await self.bot._on_text(update, self._ctx())
        self.assertEqual(self.calls, [])
        self._wait()
        self.bot._pending_bet["expires"] -= 1000
        update2 = self._update("yes")
        await self.bot._on_text(update2, self._ctx())
        self.assertEqual(self.calls, [])
        self.assertIn("nothing waiting", update2.message.reply_text.await_args.args[0].lower())

    async def test_a_spoken_bet_runs_the_bet_command_preview_only(self):
        self.routes = [_route(action="bet", team="browns", amount=5.0)]
        await self.bot._on_text(self._update("put five bucks on the browns"), self._ctx())
        self.assertEqual(self.calls, [("bet", ["browns", "5"])])  # a preview; /confirm still needed

    async def test_model_confirm_only_counts_when_a_bet_is_waiting(self):
        self.routes = [_route(action="confirm")]
        update = self._update("sounds good, go for it")
        await self.bot._on_text(update, self._ctx())
        self.assertEqual(self.calls, [])
        self._wait()
        self.routes = [_route(action="confirm")]
        await self.bot._on_text(self._update("sounds good, go for it"), self._ctx())
        self.assertEqual(self.calls, [("confirm", [])])

    async def test_bet_an_amount_on_this_restates_the_stake_and_places(self):
        self._wait("9-leg parlay, $2.00")
        self.routes = [_route(action="confirm", amount=10.0)]
        await self.bot._on_text(self._update("bet $10 on this parlay"), self._ctx())
        self.assertEqual(self.calls, [("confirm", [10.0])])

    async def test_pause_and_kill_are_never_run_from_text(self):
        self.routes = [_route(action="control")]
        update = self._update("shut everything down")
        await self.bot._on_text(update, self._ctx())
        self.assertEqual(self.calls, [])
        self.assertIn("/kill", update.message.reply_text.await_args.args[0])

    async def test_replies_errors_and_missing_detail_are_sent_back(self):
        self.routes = [_route(action="reply", reply_text="Try: bet $5 on the browns")]
        update = self._update("hello")
        await self.bot._on_text(update, self._ctx())
        self.assertEqual(update.message.reply_text.await_args.args[0], "Try: bet $5 on the browns")
        self.bot._route_text.side_effect = RuntimeError("needs ANTHROPIC_API_KEY")
        update = self._update("bet")
        await self.bot._on_text(update, self._ctx())
        self.assertIn("ANTHROPIC_API_KEY", update.message.reply_text.await_args.args[0])
        self.bot._route_text.side_effect = lambda text, pending: self.routes.pop(0)
        self.routes = [_route(action="bet", reply_text="Which team?")]
        update = self._update("bet something")
        await self.bot._on_text(update, self._ctx())
        self.assertEqual(update.message.reply_text.await_args.args[0], "Which team?")
        self.assertEqual(self.calls, [])

    async def test_other_chats_and_disabled_mode_are_ignored(self):
        await self.bot._on_text(self._update("yes", chat_id=999), self._ctx())
        self.bot._route_text.assert_not_called()
        self.bot.config.natural_language = False
        update = self._update("status")
        await self.bot._on_text(update, self._ctx())
        self.bot._route_text.assert_not_called()
        self.assertIn("off", update.message.reply_text.await_args.args[0])

    async def test_pending_label_is_given_to_the_router(self):
        self._wait("2-leg parlay (A + B), $2.00")
        self.routes = [_route(action="status")]
        await self.bot._on_text(self._update("how am i doing"), self._ctx())
        self.assertEqual(self.bot._route_text.call_args.args[1], "2-leg parlay (A + B), $2.00")
