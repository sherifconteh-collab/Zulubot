"""Two-way Telegram bot for monitoring and controlling the trading bot.

Outbound: trade alerts, heartbeats, risk alerts, errors
Inbound: /status, /crypto, /stocks, /polymarket, /positions, /risk,
         /pause, /resume, /kill, /config, /history, /ai, /backtest, /help
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

if TYPE_CHECKING:
    from trading_bot.config import TelegramConfig


# ── Visual helpers ─────────────────────────────────────────────────

_TG_MAX_MSG_LEN = 4096


async def _send_html_chunks(update: Update, text: str) -> None:
    """Send *text* as one or more Telegram messages, each ≤ _TG_MAX_MSG_LEN chars."""
    if len(text) <= _TG_MAX_MSG_LEN:
        await update.message.reply_text(text, parse_mode="HTML")  # type: ignore[union-attr]
        return
    lines = text.split("\n")
    chunk_lines: list[str] = []
    chunk_len = 0
    for line in lines:
        extra = len(line) + 1  # +1 for the newline that join() will add
        if chunk_lines and chunk_len + extra > _TG_MAX_MSG_LEN:
            await update.message.reply_text(  # type: ignore[union-attr]
                "\n".join(chunk_lines), parse_mode="HTML"
            )
            chunk_lines = []
            chunk_len = 0
        chunk_lines.append(line)
        chunk_len += extra
    if chunk_lines:
        await update.message.reply_text(  # type: ignore[union-attr]
            "\n".join(chunk_lines), parse_mode="HTML"
        )


def _pnl_emoji(value: float) -> str:
    if value > 0:
        return "\U0001f7e2"  # green circle
    elif value < 0:
        return "\U0001f534"  # red circle
    return "\u26aa"  # white circle


def _pnl_arrow(value: float) -> str:
    if value > 0:
        return "\u25b2"  # ▲
    elif value < 0:
        return "\u25bc"  # ▼
    return "\u25cf"  # ●


def _progress_bar(fraction: float, width: int = 10) -> str:
    """Create a text progress bar: [████░░░░░░] 40%"""
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    empty = width - filled
    bar = "\u2588" * filled + "\u2591" * empty
    return f"[{bar}]"


def _risk_status_icon(active: bool) -> str:
    return "\U0001f6d1" if active else "\u2705"  # 🛑 or ✅


def _format_currency(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value:,.2f}"
    return f"${value:.2f}"


def _fear_greed_visual(value: int | None) -> str:
    if value is None:
        return "\u2753 N/A"  # ❓
    if value <= 25:
        return f"\U0001f630 {value} Extreme Fear"  # 😰
    elif value <= 45:
        return f"\U0001f628 {value} Fear"  # 😨
    elif value <= 55:
        return f"\U0001f610 {value} Neutral"  # 😐
    elif value <= 75:
        return f"\U0001f60f {value} Greed"  # 😏
    else:
        return f"\U0001f929 {value} Extreme Greed"  # 🤩


def _clip_text(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _on_off(enabled: bool) -> str:
    return "ON" if enabled else "OFF"


class TradingTelegramBot:
    def __init__(
        self,
        config: TelegramConfig,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.logger = logger.getChild("telegram")
        self._app: Application | None = None
        self._service: Any = None
        self._paused = False
        self._trade_history: list[dict[str, Any]] = []

    @property
    def is_configured(self) -> bool:
        return self.config.is_configured

    @property
    def paused(self) -> bool:
        return self._paused

    def set_service(self, service: Any) -> None:
        self._service = service

    def record_trade(self, trade_info: dict[str, Any]) -> None:
        self._trade_history.append(trade_info)
        if len(self._trade_history) > 50:
            self._trade_history = self._trade_history[-50:]

    async def start(self) -> None:
        if not self.is_configured:
            self.logger.warning("Telegram not configured. Skipping bot startup.")
            return

        self._app = Application.builder().token(self.config.bot_token).build()
        self._register_handlers()
        await self._app.initialize()
        await self._app.start()
        if self._app.updater:
            await self._app.updater.start_polling(drop_pending_updates=True)
        self.logger.info("Telegram bot started and polling for commands.")

    async def stop(self) -> None:
        if self._app is None:
            return
        if self._app.updater:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        self.logger.info("Telegram bot stopped.")

    # ── Outbound messages ──────────────────────────────────────────

    async def send_message(self, text: str) -> None:
        if not self.is_configured or self._app is None:
            self.logger.info("[TG-OFFLINE] %s", text)
            return
        try:
            await self._app.bot.send_message(
                chat_id=self.config.chat_id,
                text=text,
                parse_mode="HTML",
            )
        except Exception as exc:
            self.logger.error("Failed to send Telegram message: %s", exc)

    @staticmethod
    def _module_icon(module_name: str) -> str:
        icons = {
            "crypto": "\U0001f4b1",
            "stocks": "\U0001f4c8",
            "polymarket": "\U0001f52e",
            "rotation": "\U0001f504",
            "predictions": "\U0001f3b2",
            "latency_arb": "\u26a1",
        }
        return icons.get(module_name, "\u2699\ufe0f")

    @staticmethod
    def _module_status_text(status: str, *, has_trades: bool) -> str:
        normalised = str(status or "unknown").strip().lower()
        if has_trades:
            return f"\u2705 {normalised}"

        status_labels = {
            "ok": "\u23f8 watching",
            "completed": "\u23f8 no fills",
            "skipped": "\u23ed skipped",
            "disabled": "\u23f8 disabled",
            "halted": "\U0001f6d1 risk halted",
            "waiting_for_seed": "\u23f3 waiting for seed",
            "no_prices": "\u26a0\ufe0f waiting for fresh prices",
            "error": "\u26a0\ufe0f error",
            "unknown": "\u2753 unknown",
        }
        return status_labels.get(normalised, f"\u23f8 {normalised}")

    @staticmethod
    def _ai_feature_summary(ai_info: dict[str, Any]) -> str:
        features: list[str] = []
        if ai_info.get("streaming_enabled"):
            features.append("streaming")
        if ai_info.get("tool_calling_enabled"):
            features.append("tools")
        if ai_info.get("vision_enabled"):
            features.append("vision")
        if ai_info.get("extended_thinking_enabled"):
            features.append("thinking")
        if ai_info.get("prompt_guard_enabled"):
            features.append("prompt-guard")
        return ", ".join(features) if features else "standard"

    def _append_generic_ai_context(self, lines: list[str], module_data: dict[str, Any]) -> None:
        ai_action = module_data.get("ai_action")
        ai_confidence = module_data.get("ai_confidence")
        ai_reasoning = module_data.get("ai_reasoning")
        if ai_action:
            confidence_text = ""
            if isinstance(ai_confidence, (int, float)):
                confidence_text = f" ({float(ai_confidence):.2f})"
            lines.append(f"   \U0001f9e0 AI: {html.escape(str(ai_action).upper())}{confidence_text}")
        if ai_reasoning:
            lines.append(f"   \u2514 {html.escape(_clip_text(ai_reasoning))}")

    def _append_module_context(self, lines: list[str], module_name: str, module_data: dict[str, Any]) -> None:
        if module_name == "crypto":
            fg = module_data.get("fear_and_greed")
            funding = module_data.get("funding_rates", {})
            sources = module_data.get("price_sources", {})
            lines.append(f"   \U0001f3ad Sentiment: {_fear_greed_visual(fg)}")
            for sym, rate in funding.items():
                src = sources.get(sym, "?")
                rate_icon = "\U0001f53b" if rate < 0 else "\U0001f53a"
                lines.append(f"   {rate_icon} {html.escape(str(sym))} funding={float(rate):.6f} ({html.escape(str(src))})")
            return

        if module_name == "stocks":
            vix = module_data.get("vix_value")
            qualified = module_data.get("qualified_count", 0)
            if vix is not None:
                vix_icon = "\U0001f525" if vix > 25 else "\u2744\ufe0f"
                lines.append(f"   {vix_icon} VIX: {float(vix):.1f}  |  Qualified: {int(qualified)}")
            next_eligible = module_data.get("next_eligible_date")
            if next_eligible:
                lines.append(f"   \u23ed Next eligible scan: {html.escape(str(next_eligible))}")
            top_targets = module_data.get("top_targets", [])
            if top_targets:
                preview = ", ".join(html.escape(str(symbol)) for symbol in top_targets[:3])
                lines.append(f"   \U0001f3af Watchlist: {preview}")
            return

        if module_name == "rotation":
            current = module_data.get("current_coin", "?")
            swapped = module_data.get("swapped_to")
            swapped_from = module_data.get("swapped_from") or module_data.get("from")
            ratio_shift = module_data.get("ratio_shift", 0)
            coins_tracked = module_data.get("coins_tracked", 0)
            holdings_synced = module_data.get("holdings_synced", 0)
            action = str(module_data.get("action", "hold"))
            lines.append(f"   \U0001f3af Holding: <code>{html.escape(str(current))}</code>")
            lines.append(f"   \U0001f4ca Tracking: {int(coins_tracked)} coins")
            if holdings_synced > 0:
                lines.append(f"   \U0001f4e5 Synced {int(holdings_synced)} holdings from Coinbase")
            if swapped:
                lines.append(
                    f"   \U0001f504 Swapped <code>{html.escape(str(swapped_from or current))}</code> -> "
                    f"<code>{html.escape(str(swapped))}</code> (shift: {float(ratio_shift):.2f}%)"
                )
            elif action == "swap_ai_blocked":
                lines.append(
                    f"   \U0001f9e0 AI vetoed swap <code>{html.escape(str(swapped_from or current))}</code> -> "
                    f"<code>{html.escape(str(module_data.get('to', '?')))}</code>"
                )
            self._append_generic_ai_context(lines, module_data)
            return

        if module_name == "predictions":
            discovered = int(module_data.get("markets_discovered", module_data.get("markets_found", 0)) or 0)
            scanned = int(module_data.get("markets_scanned", discovered) or 0)
            resolved = module_data.get("resolved_closed", [])
            can_trade = bool(module_data.get("can_trade"))
            lines.append(f"   \U0001f3b2 Markets: {scanned}/{discovered} scanned")
            execution_text = "live-enabled" if can_trade else "scan-only / paper"
            lines.append(f"   \U0001f4cc Execution: {execution_text}")
            if resolved:
                resolved_preview = ", ".join(html.escape(str(symbol)) for symbol in resolved[:3])
                lines.append(f"   \U0001f3c1 Resolved: {resolved_preview}")
            return

        if module_name == "polymarket":
            discovered = int(module_data.get("markets_discovered", 0) or 0)
            scanned = int(module_data.get("markets_scanned", 0) or 0)
            resolved = module_data.get("resolved_closed", [])
            fg = module_data.get("fear_greed")
            lines.append(f"   \U0001f50d Markets: {scanned}/{discovered} scanned")
            if fg is not None:
                lines.append(f"   \U0001f3ad Sentiment: {_fear_greed_visual(fg)}")
            if resolved:
                lines.append(
                    f"   \U0001f3c1 Resolved: {', '.join(html.escape(str(symbol)) for symbol in resolved[:3])}"
                )
            return

        if module_name == "latency_arb":
            signals = int(module_data.get("signals_detected", 0) or 0)
            scanned = int(module_data.get("markets_scanned", 0) or 0)
            regime = module_data.get("market_regime")
            stale_symbols = module_data.get("stale_symbols", [])
            can_trade_live = bool(module_data.get("can_trade_live"))
            lines.append(f"   \u26a1 Signals: {signals}  |  Markets: {scanned}")
            if regime:
                lines.append(f"   \U0001f9ed Regime: {html.escape(str(regime))}")
            if stale_symbols:
                stale_preview = ", ".join(html.escape(str(symbol)) for symbol in stale_symbols[:3])
                lines.append(f"   \u26a0\ufe0f Stale feeds: {stale_preview}")
            execution_text = "live Kalshi" if can_trade_live else "paper shadowing"
            lines.append(f"   \U0001f4cc Execution: {execution_text}")
            return

        self._append_generic_ai_context(lines, module_data)

    async def send_trade_alert(self, result: dict[str, Any]) -> None:
        module_results = result.get("modules", {})
        paper = result.get("paper_mode", True)
        mode_tag = "\U0001f4dd PAPER" if paper else "\U0001f4b0 LIVE"  # 📝 or 💰

        lines = [
            f"\U0001f916 <b>ZULUBOT TRADE CYCLE</b>  {mode_tag}",  # 🤖
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        ]

        has_trades = False
        for module_name, module_data in module_results.items():
            if not isinstance(module_data, dict):
                continue
            status = module_data.get("status", "unknown")
            buys = module_data.get("buys", [])
            sells = module_data.get("sells", [])
            error = module_data.get("error")

            icon = self._module_icon(module_name)

            if error:
                lines.append(f"\n{icon} <b>{module_name.upper()}</b>  \u26a0\ufe0f <i>{status}</i>")
                lines.append(f"   \u2514 Error: {html.escape(_clip_text(error, 100))}")
                continue

            if buys or sells:
                has_trades = True
                lines.append(f"\n{icon} <b>{module_name.upper()}</b>  {self._module_status_text(status, has_trades=True)}")
                for sym in buys:
                    lines.append(f"   \U0001f7e2 BUY  <code>{html.escape(str(sym))}</code>")
                for sym in sells:
                    lines.append(f"   \U0001f534 SELL <code>{html.escape(str(sym))}</code>")
                self.record_trade({
                    "time": result.get("timestamp", ""),
                    "module": module_name,
                    "buys": buys,
                    "sells": sells,
                })
            else:
                lines.append(f"\n{icon} <b>{module_name.upper()}</b>  {self._module_status_text(status, has_trades=False)}")

            self._append_module_context(lines, module_name, module_data)

        # ── Portfolio Summary ──
        equity = result.get("equity", 0)
        cash = result.get("cash", 0)
        pnl = result.get("realized_pnl", 0)
        positions = result.get("open_positions", [])
        kalshi_bal = result.get("kalshi_balance")

        lines.append(f"\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        lines.append(f"\U0001f4b0 <b>PORTFOLIO</b>")  # 💰
        lines.append(f"  Equity:    <b>{_format_currency(equity)}</b>")
        lines.append(f"  Cash:      {_format_currency(cash)}")
        if kalshi_bal is not None:
            lines.append(f"  \U0001f3b2 Kalshi:  {_format_currency(kalshi_bal)}")  # 🎲
        lines.append(f"  {_pnl_emoji(pnl)} P&L:     {'+' if pnl >= 0 else ''}{_format_currency(pnl)}")
        lines.append(f"  Positions: {len(positions)}")

        # ── AI Analyst Summary ──
        ai_info = result.get("ai_info")
        if ai_info:
            mode = ai_info.get("confirmation_mode", "?")
            multi = ai_info.get("multi_analyst_enabled", False)
            analysts = ai_info.get("selected_analysts", [])
            remaining = ai_info.get("budget_remaining_usd", 0)
            monthly = ai_info.get("monthly_cost_usd", 0)
            breaker = ai_info.get("circuit_breaker", {})
            features = self._ai_feature_summary(ai_info)
            registered_tools = ai_info.get("registered_tools", [])
            if multi and analysts:
                analyst_str = ", ".join(analysts)
                lines.append(f"\n\U0001f9e0 <b>AI ANALYSTS</b>: {analyst_str}")  # 🧠
            else:
                lines.append(f"\n\U0001f9e0 <b>AI</b>: {html.escape(str(mode))} mode")
            lines.append(f"   \u2514 spent=${float(monthly):.4f} | left=${float(remaining):.2f}")
            lines.append(f"   \u2514 Features: {html.escape(features)}")
            if registered_tools:
                lines.append(f"   \u2514 Tools: {html.escape(', '.join(str(tool) for tool in registered_tools))}")
            if breaker:
                state = str(breaker.get("state", "unknown")).upper()
                if state not in {"CLOSED", "DISABLED"}:
                    remaining_open = float(breaker.get("remaining_open_seconds", 0.0) or 0.0)
                    lines.append(f"   \u2514 Circuit: {html.escape(state)} ({remaining_open:.0f}s cooldown)")

        # Risk status
        risk = result.get("risk", {})
        if risk.get("kill_switch_active"):
            lines.append(f"\n\U0001f6a8 <b>KILL SWITCH ACTIVE</b> \U0001f6a8")  # 🚨
        elif risk.get("daily_halt_active"):
            lines.append(f"\n\u26a0\ufe0f <b>DAILY HALT ACTIVE</b>")  # ⚠️

        stale_price_count = result.get("stale_price_count", 0)
        if stale_price_count:
            lines.append(f"\n\u26a0\ufe0f <b>STALE PRICES</b>: {stale_price_count}")
            for item in result.get("stale_prices", [])[:3]:
                symbol = item.get("symbol", "?")
                age = item.get("age_seconds")
                age_text = "unknown age" if age is None else f"{age:.0f}s old"
                lines.append(f"   \u2514 <code>{symbol}</code> {age_text}")

        latency_snapshot = result.get("latency_ms", {})
        if latency_snapshot:
            lines.append(f"\n\u23f1\ufe0f <b>LATENCY</b>")
            for name, stats in list(latency_snapshot.items())[:4]:
                lines.append(f"   \u2514 {name}: {stats.get('last_ms', 0):.2f}ms")

        # Only send if there were actual trades, errors, or active risk events.
        has_errors = any(
            isinstance(v, dict) and v.get("error")
            for v in module_results.values()
        )
        has_risk_event = bool(risk.get("kill_switch_active") or risk.get("daily_halt_active"))
        force_send = bool(result.get("force_send"))

        if not has_trades and not has_errors and not has_risk_event:
            if not force_send:
                return
            lines.append("\nCycle complete: no trades executed this run, but the decision context above shows what the bot evaluated.")

        await self.send_message("\n".join(lines))

    async def send_heartbeat(self, status: dict[str, Any]) -> None:
        equity = status.get("equity", 0)
        cash = status.get("cash", 0)
        pnl = status.get("realized_pnl", 0)
        positions = status.get("open_positions", [])
        risk = status.get("risk", {})
        kalshi_bal = status.get("kalshi_balance")
        ai_info = status.get("ai_info")
        stale_price_count = status.get("stale_price_count", 0)
        stale_prices = status.get("stale_prices", [])
        latency_snapshot = status.get("latency_ms", {})

        dd_daily = risk.get("daily_drawdown_pct", 0)
        dd_total = risk.get("total_drawdown_pct", 0)
        peak = risk.get("peak_equity", 0)

        lines = [
            f"\U0001f916 <b>ZULUBOT HEARTBEAT</b>",  # 🤖
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            "",
            f"\U0001f4b0 <b>PORTFOLIO</b>",  # 💰
            f"  Equity:  <b>{_format_currency(equity)}</b>",
            f"  Cash:    {_format_currency(cash)}",
        ]

        # Kalshi balance
        if kalshi_bal is not None:
            lines.append(f"  \U0001f3b2 Kalshi: {_format_currency(kalshi_bal)}")  # 🎲

        lines.extend([
            f"  {_pnl_emoji(pnl)} P&L:   {'+' if pnl >= 0 else ''}{_format_currency(pnl)}",
            f"  \U0001f3c6 Peak:  {_format_currency(peak)}",  # 🏆
        ])

        # Drawdown visual bar
        if dd_daily != 0 or dd_total != 0:
            lines.append("")
            lines.append(f"\U0001f6e1 <b>RISK</b>")  # 🛡
            dd_bar = _progress_bar(abs(dd_daily) / 5.0)  # 5% = full bar
            lines.append(f"  Daily DD:  {dd_bar} {dd_daily:.1f}%")
            total_bar = _progress_bar(abs(dd_total) / 15.0)  # 15% = full bar
            lines.append(f"  Total DD:  {total_bar} {dd_total:.1f}%")

        # Positions grouped by type
        if positions:
            # Group positions by asset type
            by_type: dict[str, list[dict[str, Any]]] = {}
            for pos in positions:
                atype = pos.get("asset_type", "other")
                by_type.setdefault(atype, []).append(pos)

            type_labels = {
                "stock": ("\U0001f4c8", "STOCKS"),
                "crypto": ("\U0001f4b1", "CRYPTO"),
                "prediction": ("\U0001f3b2", "PREDICTIONS"),
            }

            for atype, type_positions in by_type.items():
                icon, label = type_labels.get(atype, ("\u2699\ufe0f", atype.upper()))
                type_pnl = sum(p.get("unrealized_pnl", 0) for p in type_positions)
                lines.append(f"\n{icon} <b>{label}</b> ({len(type_positions)}) "
                             f"{_pnl_emoji(type_pnl)} {'+' if type_pnl >= 0 else ''}{_format_currency(type_pnl)}")

                for pos in type_positions:
                    sym = pos.get("symbol", "?")
                    pnl_pos = pos.get("unrealized_pnl", 0)
                    mv = pos.get("market_value", 0)
                    entry = pos.get("entry_price", 0)
                    current = pos.get("current_price", 0)
                    pct_change = ((current - entry) / entry * 100) if entry > 0 else 0

                    lines.append(
                        f"  {_pnl_arrow(pnl_pos)} <b>{sym}</b>  "
                        f"{_format_currency(mv)} "
                        f"({'+' if pct_change >= 0 else ''}{pct_change:.1f}%) "
                        f"P&L: {'+' if pnl_pos >= 0 else ''}{_format_currency(pnl_pos)}"
                    )
        else:
            lines.append(f"\n\U0001f4e6 No open positions")

        # AI Analysts section
        if ai_info:
            multi = ai_info.get("multi_analyst_enabled", False)
            analysts = ai_info.get("selected_analysts", [])
            remaining = ai_info.get("budget_remaining_usd", 0)
            mode = ai_info.get("confirmation_mode", "?")
            monthly = ai_info.get("monthly_cost_usd", 0)
            breaker = ai_info.get("circuit_breaker", {})
            features = self._ai_feature_summary(ai_info)
            registered_tools = ai_info.get("registered_tools", [])
            lines.append("")
            if multi and analysts:
                lines.append(f"\U0001f9e0 <b>AI ANALYSTS</b> ({mode})")  # 🧠
                for a in analysts:
                    lines.append(f"  \u2022 {a}")
                lines.append(f"  \U0001f4b3 ${monthly:.4f} spent | ${remaining:.2f} left")
            else:
                lines.append(f"\U0001f9e0 <b>AI</b>: {mode} | ${remaining:.2f} left")
                lines.append(f"  \U0001f4b3 ${monthly:.4f} spent this month")
            lines.append(f"  Features: {html.escape(features)}")
            if registered_tools:
                lines.append(f"  Tools: {html.escape(', '.join(str(tool) for tool in registered_tools))}")
            if breaker and breaker.get("state") not in {None, "closed", "disabled"}:
                remaining_open = float(breaker.get("remaining_open_seconds", 0.0) or 0.0)
                lines.append(
                    f"  Circuit: {html.escape(str(breaker.get('state')).upper())} "
                    f"({remaining_open:.0f}s cooldown)"
                )

        # Risk switches
        lines.append("")
        lines.append(f"{_risk_status_icon(risk.get('daily_halt_active', False))} Daily Halt")
        lines.append(f"{_risk_status_icon(risk.get('kill_switch_active', False))} Kill Switch")

        if stale_price_count:
            lines.append("")
            lines.append(f"\u26a0\ufe0f <b>Stale Prices</b>: {stale_price_count}")
            for item in stale_prices[:3]:
                symbol = item.get("symbol", "?")
                age = item.get("age_seconds")
                age_text = "unknown age" if age is None else f"{age:.0f}s"
                lines.append(f"  <code>{symbol}</code> {age_text}")

        if latency_snapshot:
            lines.append("")
            lines.append("\u23f1\ufe0f <b>Latency</b>")
            for name, stats in list(latency_snapshot.items())[:4]:
                lines.append(f"  {name}: {stats.get('last_ms', 0):.2f}ms")

        # Module scan dates
        stock_date = status.get("last_stock_screen_date") or "never"
        kalshi_date = status.get("last_predictions_scan_date") or "never"
        poly_date = status.get("last_polymarket_scan_date") or "never"
        lines.append(f"\n\U0001f4c5 Last Scans")  # 📅
        lines.append(f"  Stocks: {stock_date}")
        lines.append(f"  \U0001f3b2 Kalshi: {kalshi_date}")  # 🎲
        lines.append(f"  \U0001f52e Polymarket: {poly_date}")  # 🔮

        if self._paused:
            lines.append(f"\n\u23f8\ufe0f <b>BOT PAUSED</b> \u2014 monitoring only")

        await self.send_message("\n".join(lines))

    async def send_risk_alert(self, snapshot: dict[str, Any]) -> None:
        lines = [
            f"\U0001f6a8\U0001f6a8\U0001f6a8 <b>RISK ALERT</b> \U0001f6a8\U0001f6a8\U0001f6a8",  # 🚨
            "",
        ]
        if snapshot.get("kill_switch_active"):
            lines.append(f"\U0001f6d1 <b>KILL SWITCH ENGAGED</b>")  # 🛑
            lines.append("All positions have been closed!")
        if snapshot.get("daily_halt_active"):
            dd = snapshot.get("daily_drawdown_pct", 0)
            lines.append(f"\u26a0\ufe0f <b>DAILY HALT</b> — Drawdown: {dd:.1f}%")
            lines.append("No new trades until tomorrow.")

        lines.append("")
        lines.append(f"\U0001f4b0 Equity: {_format_currency(snapshot.get('equity', 0))}")
        lines.append(f"\U0001f3c6 Peak: {_format_currency(snapshot.get('peak_equity', 0))}")
        await self.send_message("\n".join(lines))

    async def send_error(self, error_msg: str) -> None:
        await self.send_message(
            f"\u26a0\ufe0f <b>ERROR</b>\n"  # ⚠️
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"{error_msg}"
        )

    # ── Inbound command handlers ───────────────────────────────────

    def _register_handlers(self) -> None:
        assert self._app is not None
        commands = {
            "status": self._cmd_status,
            "crypto": self._cmd_crypto,
            "stocks": self._cmd_stocks,
            "positions": self._cmd_positions,
            "risk": self._cmd_risk,
            "latency": self._cmd_latency,
            "audit": self._cmd_audit,
            "decisions": self._cmd_decisions,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "kill": self._cmd_kill,
            "config": self._cmd_config,
            "polymarket": self._cmd_polymarket,
            "rotation": self._cmd_rotation,
            "predict": self._cmd_predictions,
            "backtest": self._cmd_backtest,
            "ai": self._cmd_ai,
            "history": self._cmd_history,
            "help": self._cmd_help,
            "start": self._cmd_help,
        }
        for name, callback in commands.items():
            self._app.add_handler(CommandHandler(name, callback))

    def _check_auth(self, update: Update) -> bool:
        if not update.effective_chat:
            return False
        return str(update.effective_chat.id) == self.config.chat_id

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        await update.message.reply_text(  # type: ignore[union-attr]
            f"\U0001f916 <b>Zulu Trading Bot</b>\n"  # 🤖
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"\n"
            f"\U0001f4ca <b>Monitor</b>\n"  # 📊
            f"  /status — Portfolio dashboard\n"
            f"  /positions — Open positions\n"
            f"  /risk — Risk metrics\n"
            f"  /latency — Module timings\n"
            f"  /history — Recent trades\n"
            f"  /audit [count] — Trade audit tail\n"
            f"  /decisions [count] — Strategy decision report\n"
            f"\n"
            f"\u26a1 <b>Execute</b>\n"  # ⚡
            f"  /crypto — Force crypto cycle\n"
            f"  /rotation — Force altcoin rotation\n"
            f"  /predict — Force prediction scan\n"
            f"  /stocks — Force stock screen\n"
            f"  /polymarket — Force Polymarket scan\n"
            f"  /backtest [range] — Run backtest\n"
            f"\n"
            f"\U0001f6e0 <b>Control</b>\n"  # 🛠
            f"  /pause — Pause trading\n"
            f"  /resume — Resume trading\n"
            f"  /kill — Emergency stop\n"
            f"\n"
            f"\u2699\ufe0f <b>Settings</b>\n"  # ⚙️
            f"  /config — Show configuration\n"
            f"  /ai — AI cost & budget\n",
            parse_mode="HTML",
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            status = self._service.status()
            await self.send_heartbeat(status)
        except Exception as exc:
            await update.message.reply_text(f"\u26a0\ufe0f Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_crypto(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        await update.message.reply_text("\U0001f4b1 Running crypto cycle...")  # type: ignore[union-attr]
        try:
            result = self._service.run("crypto")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"\u26a0\ufe0f Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_stocks(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        await update.message.reply_text("\U0001f4c8 Running stock screen (forced)...")  # type: ignore[union-attr]
        try:
            result = self._service.run("stocks", force=True)
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"\u26a0\ufe0f Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_polymarket(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        await update.message.reply_text("\U0001f52e Running Polymarket scan...")  # type: ignore[union-attr]
        try:
            result = self._service.run("polymarket")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"\u26a0\ufe0f Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_rotation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        await update.message.reply_text("\U0001f504 Running altcoin rotation cycle...")  # type: ignore[union-attr]
        try:
            result = self._service.run("rotation")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"\u26a0\ufe0f Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_predictions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        await update.message.reply_text("\U0001f3b0 Running prediction markets scan...")  # type: ignore[union-attr]
        try:
            result = self._service.run("predictions")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"\u26a0\ufe0f Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return

        timerange = None
        if context.args:
            timerange = context.args[0]

        await update.message.reply_text(  # type: ignore[union-attr]
            f"\U0001f9ea Running backtest{' (' + timerange + ')' if timerange else ''}..."  # 🧪
            f"\nThis may take a few minutes."
        )
        try:
            from trading_bot.integrations.freqtrade_adapter import FreqtradeAdapter
            adapter = FreqtradeAdapter(self._service.config)
            if not adapter.is_available:
                await update.message.reply_text(  # type: ignore[union-attr]
                    "\u26a0\ufe0f Freqtrade not available.\n"
                    "Install: pip install -r requirements-freqtrade.txt\n"
                    "Enable: FREQTRADE_ENABLED=true in .env"
                )
                return
            result = adapter.run_backtest(timerange=timerange)
            lines = ["\U0001f9ea <b>Backtest Results</b>"]
            if "strategy" in result:
                lines.append(f"\U0001f3af Strategy: {result['strategy']}")
            if "total_trades" in result:
                lines.append(f"\U0001f4ca Trades: {result['total_trades']}")
            if "profit_total_pct" in result:
                pct = result["profit_total_pct"]
                lines.append(f"{_pnl_emoji(pct)} Profit: {'+' if pct >= 0 else ''}{pct:.2f}%")
            if "max_drawdown_pct" in result:
                lines.append(f"\U0001f4c9 Max DD: {result['max_drawdown_pct']:.2f}%")
            if "sharpe_ratio" in result:
                lines.append(f"\U0001f4cf Sharpe: {result['sharpe_ratio']:.2f}")
            if "win_rate" in result:
                lines.append(f"\U0001f3c6 Win Rate: {result['win_rate']:.1%}")
            await self.send_message("\n".join(lines))
        except Exception as exc:
            await update.message.reply_text(f"\u26a0\ufe0f Backtest error: {exc}")  # type: ignore[union-attr]

    async def _cmd_ai(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        cfg = self._service.config
        if not cfg.trading_agents.enabled:
            await update.message.reply_text(  # type: ignore[union-attr]
                f"\U0001f9e0 <b>AI ANALYSTS</b>\n"  # 🧠
                f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
                f"\u23f8\ufe0f Status: DISABLED\n"
                f"\nEnable: TRADING_AGENTS_ENABLED=true",
                parse_mode="HTML",
            )
            return

        ai = getattr(self._service, "ai_service", None)
        if ai is not None and hasattr(ai, "get_monthly_cost"):
            monthly = ai.get_monthly_cost()
            remaining = ai.get_budget_remaining()
            cap = cfg.trading_agents.max_cost_per_month
            usage_pct = monthly / cap if cap > 0 else 0
            usage_bar = _progress_bar(usage_pct)
            breaker = ai.get_circuit_breaker_status() if hasattr(ai, "get_circuit_breaker_status") else None
            registered_tools = ai.get_registered_tools() if hasattr(ai, "get_registered_tools") else []

            multi = cfg.trading_agents.multi_analyst_enabled
            analysts = list(cfg.trading_agents.selected_analysts)

            lines = [
                f"\U0001f9e0 <b>AI ANALYSTS</b>",  # 🧠
                f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
                f"\u2705 Status: ACTIVE",
                f"\U0001f3af Mode: {cfg.trading_agents.confirmation_mode}",
                f"",
            ]

            # Multi-analyst team display
            if multi:
                lines.append(f"\U0001f465 <b>Analyst Team</b> (multi-perspective)")  # 👥
                try:
                    from trading_bot.integrations.analyst_personas import get_analyst
                    for key in analysts:
                        persona = get_analyst(key)
                        if persona:
                            lines.append(f"  \u2022 <b>{html.escape(persona.display_name)}</b>")
                            lines.append(f"    <i>{html.escape(persona.description)}</i>")
                        else:
                            lines.append(f"  \u2022 {html.escape(key)}")
                except ImportError:
                    for key in analysts:
                        lines.append(f"  \u2022 {html.escape(key)}")
                lines.append(f"")
                lines.append(f"\U0001f504 Signals aggregated via weighted voting")
            else:
                lines.append(f"\U0001f464 <b>Single Agent</b> (Haiku \u2192 Sonnet)")  # 👤

            lines.extend([
                f"",
                f"\U0001f4b3 <b>Budget</b>",
                f"  Spent:     ${monthly:.4f}",
                f"  Remaining: ${remaining:.2f}",
                f"  {usage_bar} {usage_pct:.0%} of ${cap:.2f}/mo",
                f"  Hard stop: {'YES' if remaining <= 0 else 'NO'}",
                f"",
                f"\U0001f916 <b>Models</b>",
                f"  Quick: {cfg.trading_agents.quick_model}",
                f"  Deep:  {cfg.trading_agents.deep_model}",
                f"  Cache: {cfg.trading_agents.cache_ttl_hours}h TTL",
                f"",
                f"\U0001f6e0 <b>Features</b>",
                f"  Streaming: {_on_off(cfg.trading_agents.streaming_enabled)}",
                f"  Tools: {_on_off(cfg.trading_agents.tool_calling_enabled)}",
                f"  Vision: {_on_off(cfg.trading_agents.vision_enabled)}",
                f"  Thinking: {_on_off(cfg.trading_agents.extended_thinking_enabled)}",
                f"  Prompt guard: {_on_off(cfg.trading_agents.prompt_injection_protection_enabled)}",
            ])
            if cfg.trading_agents.extended_thinking_enabled:
                lines.append(
                    f"  Thinking budget: {cfg.trading_agents.thinking_budget_tokens} tokens "
                    f"({cfg.trading_agents.thinking_display})"
                )
            if registered_tools:
                lines.append(f"  Registered: {', '.join(registered_tools)}")
            if breaker:
                circuit_state = str(breaker.get('state', 'unknown')).upper()
                cooldown = float(breaker.get('remaining_open_seconds', 0.0) or 0.0)
                lines.append(
                    f"  Circuit: {circuit_state} | failures={breaker.get('failure_count', 0)}"
                    f" | cooldown={cooldown:.0f}s"
                )
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]
        else:
            await update.message.reply_text(  # type: ignore[union-attr]
                f"\U0001f9e0 <b>AI ANALYSTS</b>\n"
                f"\u26a0\ufe0f Enabled but unavailable\n"
                f"Check LLM provider config (ANTHROPIC_API_KEY or OLLAMA_BASE_URL)",
                parse_mode="HTML",
            )

    async def _cmd_latency(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return

        latency = self._service.get_latency_snapshot()
        if not latency:
            await update.message.reply_text("No latency samples recorded yet.")  # type: ignore[union-attr]
            return

        lines = [
            "\u23f1\ufe0f <b>Module Latency</b>",
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        ]
        for name, stats in latency.items():
            lines.append(
                f"{html.escape(name)}: last={stats.get('last_ms', 0):.2f}ms | "
                f"avg={stats.get('avg_ms', 0):.2f}ms | "
                f"max={stats.get('max_ms', 0):.2f}ms | "
                f"samples={int(stats.get('count', 0))}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]

    async def _cmd_audit(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return

        limit = 5
        if context.args:
            try:
                limit = max(1, min(10, int(context.args[0])))
            except (TypeError, ValueError):
                limit = 5

        entries = self._service.get_recent_audit_entries(limit)
        if not entries:
            await update.message.reply_text("No trade audit entries available.")  # type: ignore[union-attr]
            return

        lines = [
            "\U0001f4dc <b>Trade Audit</b>",
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        ]
        for entry in entries:
            lines.append(
                f"{html.escape(str(entry.get('timestamp', 'n/a')))} | "
                f"{html.escape(str(entry.get('event', '?')).upper())} | "
                f"<code>{html.escape(str(entry.get('symbol', '?')))}</code> | "
                f"price={float(entry.get('price', 0.0)):.4f} | "
                f"qty={float(entry.get('quantity', 0.0)):.4f}"
            )
            reason = entry.get("reason")
            if reason:
                lines.append(f"  reason={html.escape(str(reason))}")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]

    async def _cmd_decisions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return

        limit = 10
        if context.args:
            try:
                limit = max(1, min(20, int(context.args[0])))
            except (TypeError, ValueError):
                limit = 10

        entries = self._service.get_recent_decisions(limit)
        if not entries:
            await update.message.reply_text(  # type: ignore[union-attr]
                "\U0001f4cb <b>Decision Report</b>\n\nNo decisions recorded yet.\n"
                "Run a strategy cycle first.",
                parse_mode="HTML",
            )
            return

        # Action → emoji mapping
        action_icons = {
            "buy":        "\U0001f7e2",   # 🟢
            "sell":       "\U0001f534",   # 🔴
            "swap":       "\U0001f504",   # 🔄
            "hold":       "\u26aa",       # ⚪
            "skip_buy":   "\u23f8\ufe0f", # ⏸️
            "skip_swap":  "\u23f8\ufe0f", # ⏸️
        }

        lines = [
            f"\U0001f4cb <b>Decision Report</b> (last {len(entries)})",  # 📋
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        ]

        for entry in entries:
            action = str(entry.get("action", "?"))
            icon = action_icons.get(action, "\u2753")
            symbol = html.escape(str(entry.get("symbol", "?")))
            strategy = html.escape(str(entry.get("strategy", "?")))
            reason = html.escape(str(entry.get("reason", "?")))
            ts_raw = str(entry.get("timestamp", ""))
            # Shorten timestamp: keep only HH:MM:SS portion
            ts_short = ts_raw[11:19] if len(ts_raw) >= 19 else ts_raw

            action_label = action.upper().replace("_", " ")
            lines.append(f"\n{icon} <b>{action_label}</b> <code>{symbol}</code>")
            if ts_short:
                lines.append(f"  \U0001f552 {html.escape(ts_short)}")  # 🕒
            lines.append(f"  \U0001f3af Strategy: {strategy}")
            lines.append(f"  \U0001f4ac Reason: <i>{reason}</i>")

            signals = entry.get("signals") or {}
            if signals:
                sig_parts = []
                for key, val in signals.items():
                    if val is None:
                        continue
                    sig_parts.append(f"{html.escape(key)}={html.escape(str(val))}")
                if sig_parts:
                    lines.append(f"  \U0001f4ca {' | '.join(sig_parts)}")  # 📊

        await _send_html_chunks(update, "\n".join(lines))

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        status = self._service.status()
        positions = status.get("open_positions", [])
        kalshi_bal = status.get("kalshi_balance")

        if not positions:
            no_pos_lines = [f"\U0001f4e6 <b>Positions</b>\n\nNo open positions."]
            if kalshi_bal is not None:
                no_pos_lines.append(f"\n\U0001f3b2 Kalshi Balance: {_format_currency(kalshi_bal)}")
            await update.message.reply_text(  # type: ignore[union-attr]
                "\n".join(no_pos_lines),
                parse_mode="HTML",
            )
            return

        lines = [
            f"\U0001f4e6 <b>OPEN POSITIONS ({len(positions)})</b>",  # 📦
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        ]

        # Group positions by asset type
        by_type: dict[str, list[dict[str, Any]]] = {}
        for pos in positions:
            atype = pos.get("asset_type", "other")
            by_type.setdefault(atype, []).append(pos)

        type_labels = {
            "stock": ("\U0001f4c8", "STOCKS"),
            "crypto": ("\U0001f4b1", "CRYPTO"),
            "prediction": ("\U0001f3b2", "PREDICTIONS"),
        }

        for atype, type_positions in by_type.items():
            icon, label = type_labels.get(atype, ("\u2699\ufe0f", atype.upper()))
            type_pnl = sum(p.get("unrealized_pnl", 0) for p in type_positions)
            type_value = sum(p.get("market_value", 0) for p in type_positions)
            lines.append(
                f"\n{icon} <b>{label}</b> ({len(type_positions)}) \u2014 "
                f"{_format_currency(type_value)} "
                f"{_pnl_emoji(type_pnl)} {'+' if type_pnl >= 0 else ''}{_format_currency(type_pnl)}"
            )

            for pos in type_positions:
                sym = pos["symbol"]
                qty = pos["quantity"]
                entry = pos["entry_price"]
                current = pos["current_price"]
                pnl_val = pos["unrealized_pnl"]
                mv = pos["market_value"]
                pct = ((current - entry) / entry * 100) if entry > 0 else 0

                lines.append(
                    f"\n  {_pnl_arrow(pnl_val)} <b>{sym}</b>\n"
                    f"     Entry:   ${entry:,.4f}\n"
                    f"     Current: ${current:,.4f}\n"
                    f"     Qty:     {qty:.6f}\n"
                    f"     Value:   {_format_currency(mv)}\n"
                    f"     {_pnl_emoji(pnl_val)} P&L: {'+' if pnl_val >= 0 else ''}{_format_currency(pnl_val)} "
                    f"({'+' if pct >= 0 else ''}{pct:.1f}%)"
                )

        # Kalshi balance at the bottom
        if kalshi_bal is not None:
            lines.append(f"\n\U0001f3b2 <b>Kalshi Balance:</b> {_format_currency(kalshi_bal)}")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        status = self._service.status()
        risk = status.get("risk", {})

        dd_daily = risk.get("daily_drawdown_pct", 0)
        dd_total = risk.get("total_drawdown_pct", 0)

        lines = [
            f"\U0001f6e1 <b>Risk Dashboard</b>",  # 🛡
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            f"",
            f"\U0001f4c9 Daily Drawdown",
            f"  {_progress_bar(abs(dd_daily) / 5.0)} {dd_daily:.2f}% / 5.0%",
            f"",
            f"\U0001f4c9 Total Drawdown",
            f"  {_progress_bar(abs(dd_total) / 15.0)} {dd_total:.2f}% / 15.0%",
            f"",
            f"\U0001f3c6 Peak Equity: {_format_currency(risk.get('peak_equity', 0))}",
            f"\U0001f4ca Daily Baseline: {_format_currency(risk.get('daily_baseline_equity', 0))}",
            f"",
            f"{_risk_status_icon(risk.get('daily_halt_active', False))} Daily Halt: "
            f"{'ACTIVE' if risk.get('daily_halt_active') else 'OK'}",
            f"{_risk_status_icon(risk.get('kill_switch_active', False))} Kill Switch: "
            f"{'ENGAGED' if risk.get('kill_switch_active') else 'OK'}",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        self._paused = True
        await update.message.reply_text(  # type: ignore[union-attr]
            f"\u23f8\ufe0f <b>Bot PAUSED</b>\n\n"
            f"Monitoring continues but no trades will execute.\n"
            f"Use /resume to restart trading.",
            parse_mode="HTML",
        )

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        self._paused = False
        await update.message.reply_text(  # type: ignore[union-attr]
            f"\u25b6\ufe0f <b>Bot RESUMED</b>\n\nAutomated trading is active.",
            parse_mode="HTML",
        )

    async def _cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        await update.message.reply_text(  # type: ignore[union-attr]
            f"\U0001f6a8 <b>KILL SWITCH</b>: Closing all positions...",
            parse_mode="HTML",
        )
        try:
            state = self._service.state_store.load()
            state.kill_switch_engaged = True
            self._service.broker.close_all(state, reason="manual_kill_switch")
            self._service.state_store.save(state)
            self._paused = True
            status = self._service.status()
            await self.send_heartbeat(status)
            await self.send_message(
                f"\U0001f6d1 Kill switch engaged. All positions closed. Bot paused."
            )
        except Exception as exc:
            await update.message.reply_text(f"\u26a0\ufe0f Error during kill: {exc}")  # type: ignore[union-attr]

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        cfg = self._service.config
        paper_icon = "\U0001f4dd" if cfg.paper_mode else "\U0001f4b0"

        # AI section with multi-analyst support
        ai_lines = [f"\U0001f9e0 <b>AI Analysts</b>: {'ON' if cfg.trading_agents.enabled else 'OFF'}"]
        if cfg.trading_agents.enabled:
            ai_lines.append(f"  Mode: {cfg.trading_agents.confirmation_mode}")
            if cfg.trading_agents.multi_analyst_enabled:
                analysts = ", ".join(cfg.trading_agents.selected_analysts)
                ai_lines.append(f"  \U0001f465 Multi-analyst: {analysts}")
            else:
                ai_lines.append(f"  \U0001f464 Single agent (Haiku \u2192 Sonnet)")

        lines = [
            f"\u2699\ufe0f <b>ZULUBOT CONFIG</b>",  # ⚙️
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            f"",
            f"{paper_icon} Mode: {'PAPER' if cfg.paper_mode else 'LIVE'}",
            f"\U0001f4b5 Cash: {_format_currency(cfg.initial_cash)}",
            f"\u23f8\ufe0f Paused: {'Yes' if self._paused else 'No'}",
            f"",
            f"\U0001f6e1 <b>Risk Limits</b>",
            f"  Position: {cfg.risk.max_position_fraction*100:.1f}%",
            f"  Daily Halt: {cfg.risk.daily_halt_drawdown*100:.1f}%",
            f"  Kill Switch: {cfg.risk.kill_switch_drawdown*100:.1f}%",
            f"",
            f"\U0001f4b1 <b>Crypto</b>",
            f"  {', '.join(cfg.crypto.symbols)}",
            f"  Buy: F&G < {cfg.crypto.buy_fear_threshold}",
            f"  Sell: F&G > {cfg.crypto.sell_fear_threshold}",
            f"",
            f"\U0001f4c8 <b>Stocks</b>",
            f"  P/E < {cfg.stock.max_pe_ratio} | P/B < {cfg.stock.max_pb_ratio}",
            f"  D/E < {cfg.stock.max_debt_to_equity} | VIX > {cfg.stock.vix_buy_threshold}",
            f"",
            f"\U0001f504 <b>Altcoin Rotation</b>",
            f"  {'ON' if cfg.altcoin_rotation.enabled else 'OFF'} | "
            f"Fee: {cfg.altcoin_rotation.fee_pct*100:.1f}% \u00d7 {cfg.altcoin_rotation.scout_multiplier}x",
            f"  Check: {cfg.altcoin_rotation.check_interval_minutes}m | "
            f"Min: ${cfg.altcoin_rotation.min_trade_usd}",
            f"",
            f"\U0001f3b2 <b>Kalshi Predictions</b>",
            f"  {'ON' if cfg.predictions.enabled else 'OFF'} | "
            f"YES < {cfg.predictions.buy_yes_threshold} | NO > {cfg.predictions.buy_no_threshold}",
            f"",
            f"\U0001f52e <b>Polymarket</b>",
            f"  {'ON' if cfg.polymarket_strategy.enabled else 'OFF'} | "
            f"Tags: {', '.join(cfg.polymarket_strategy.market_tags)}",
            f"",
            f"\u23f0 <b>Schedule</b>",
            f"  Crypto: {cfg.scheduler.crypto_interval_minutes}m | "
            f"Rotation: {cfg.altcoin_rotation.check_interval_minutes}m",
            f"  Stocks: {cfg.scheduler.stock_check_hour}:00 | "
            f"Heartbeat: {cfg.scheduler.heartbeat_interval_minutes}m",
            f"",
        ]
        lines.extend(ai_lines)
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if not self._trade_history:
            await update.message.reply_text(  # type: ignore[union-attr]
                f"\U0001f4dc <b>Trade History</b>\n\nNo recent trades.",
                parse_mode="HTML",
            )
            return
        lines = [
            f"\U0001f4dc <b>Recent Trades</b>",  # 📜
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        ]
        for trade in self._trade_history[-10:]:
            t = trade.get("time", "?")[:16]
            m = trade.get("module", "?")
            buys = trade.get("buys", [])
            sells = trade.get("sells", [])

            icons = {
                "crypto": "\U0001f4b1", "stocks": "\U0001f4c8", "polymarket": "\U0001f52e",
                "rotation": "\U0001f504", "predictions": "\U0001f3b0",
            }
            icon = icons.get(m, "\u2699\ufe0f")

            parts = []
            for s in buys:
                parts.append(f"\U0001f7e2 {s}")
            for s in sells:
                parts.append(f"\U0001f534 {s}")
            lines.append(f"\n{icon} <b>{t}</b>\n   {' | '.join(parts)}")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]
