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


def _format_lead_line(lead: dict[str, Any]) -> str:
    """One-line representation of a lead for the Telegram leads alert."""
    symbol = html.escape(str(lead.get("symbol", "?")))
    reason = html.escape(str(lead.get("reason", "")))
    signals = lead.get("signals") or {}
    parts: list[str] = []
    # Pull out a small set of well-known signal keys for human-readable context.
    fg = signals.get("fear_greed")
    if isinstance(fg, (int, float)):
        parts.append(f"F&G {int(fg)}")
    funding = signals.get("funding_rate")
    if isinstance(funding, (int, float)):
        parts.append(f"funding {funding:+.4%}")
    yes_price = signals.get("yes_price")
    if isinstance(yes_price, (int, float)):
        parts.append(f"YES {yes_price:.2f}")
    spread = signals.get("spread_pct")
    if isinstance(spread, (int, float)):
        parts.append(f"spread {spread:+.2%}")
    holder = signals.get("holder")
    if holder:
        parts.append(f"held by {html.escape(str(holder))}")
    price = signals.get("price")
    sma = signals.get("sma")
    if (
        isinstance(sma, (int, float)) and sma > 0
        and isinstance(price, (int, float))
    ):
        sma_days = signals.get("sma_days")
        label = f"SMA{int(sma_days)}" if isinstance(sma_days, (int, float)) else "SMA"
        parts.append(f"{(price / sma - 1) * 100:+.1f}% vs {label}")
    if not parts and isinstance(price, (int, float)):
        parts.append(f"@ ${price:,.2f}")
    context = ", ".join(parts)
    if context:
        return f"• {symbol} — {context} (skipped: {reason})"
    return f"• {symbol} (skipped: {reason})"


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
            "markov": "\U0001f9ed",
            "options": "\U0001f4ca",
            "kalshi_sports": "\U0001f3c8",
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
            crypto_cfg = getattr(getattr(self._service, "config", None), "crypto", None)
            if crypto_cfg is not None:
                weights = getattr(crypto_cfg, "symbol_weights", {}) or {}
                if weights:
                    weight_text = ", ".join(
                        f"{html.escape(str(sym))} ×{w:g}" for sym, w in weights.items()
                    )
                    lines.append(f"   ⚖️ Weights: {weight_text}")
                if getattr(crypto_cfg, "trend_filter_enabled", False):
                    lines.append(
                        f"   \U0001f4d0 Trend gate: {int(crypto_cfg.trend_sma_days)}d SMA "
                        f"(−{float(crypto_cfg.trend_tolerance_pct):.0%} tolerance)"
                    )
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
            categories = module_data.get("categories") or []
            if categories:
                lines.append(
                    "   \U0001f5c2 Categories: "
                    + ", ".join(html.escape(str(c)) for c in categories)
                )
            execution_text = "live-enabled" if can_trade else "scan-only / paper"
            lines.append(f"   \U0001f4cc Execution: {execution_text}")
            if resolved:
                resolved_preview = ", ".join(html.escape(str(symbol)) for symbol in resolved[:3])
                lines.append(f"   \U0001f3c1 Resolved: {resolved_preview}")
            return

        if module_name == "kalshi_sports":
            lines.append(
                f"   \U0001f3c8 Games: {int(module_data.get('games_scanned', 0) or 0)}  |  "
                f"Matched: {int(module_data.get('markets_matched', 0) or 0)}/"
                f"{int(module_data.get('markets_scanned', 0) or 0)}"
            )
            for pred in sorted(
                module_data.get("predictions") or [], key=lambda p: p.get("edge", 0), reverse=True
            )[:3]:
                lines.append(
                    f"   \u2022 {html.escape(str(pred.get('pick', '?')))} "
                    f"({html.escape(str(pred.get('game', '')))}) edge {float(pred.get('edge', 0)):+.1%}"
                )
            resolved = module_data.get("resolved_closed", [])
            if resolved:
                lines.append(
                    f"   \U0001f3c1 Resolved: {', '.join(html.escape(str(sym)) for sym in resolved[:3])}"
                )
            lines.append(
                "   \U0001f4cc Execution: " + ("live Kalshi" if module_data.get("can_trade_live") else "paper")
            )
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

        if module_name == "markov":
            state_name = str(module_data.get("current_state", "?"))
            signal = module_data.get("signal")
            bull = module_data.get("bull_prob")
            bear = module_data.get("bear_prob")
            status = module_data.get("status", "")
            _state_icons = {"bull": "\U0001f402", "bear": "\U0001f43b", "sideways": "\u27a1\ufe0f"}
            state_icon = _state_icons.get(state_name.lower(), "\u2753")
            lines.append(f"   {state_icon} Regime: {html.escape(state_name.upper())}")
            if signal is not None:
                arrow = "\u25b2" if float(signal) > 0 else ("\u25bc" if float(signal) < 0 else "\u25cf")
                lines.append(f"   {arrow} Signal: {float(signal):+.4f}")
            if bull is not None and bear is not None:
                lines.append(f"   \U0001f402 {float(bull):.1%}  \U0001f43b {float(bear):.1%}")
            if status == "insufficient_data":
                hist = module_data.get("history_days", 0)
                min_h = module_data.get("min_history_days", 40)
                lines.append(f"   \u23f3 Building history {int(hist)}/{int(min_h)} days")
            return

        self._append_generic_ai_context(lines, module_data)

    async def send_trade_alert(self, result: dict[str, Any]) -> None:
        """Minimal trade alert. Only sent when there are real trades, errors,
        or an active risk event. Format: plain, scannable, no emoji clutter.
        """
        module_results = result.get("modules", {})
        risk = result.get("risk", {})
        paper = result.get("paper_mode", True)
        force_send = bool(result.get("force_send"))

        # Gather trades and errors up front so we can decide whether to send
        # anything at all.
        trade_lines: list[str] = []
        error_lines: list[str] = []
        has_trades = False
        for module_name, module_data in module_results.items():
            if not isinstance(module_data, dict):
                continue
            safe_module = html.escape(str(module_name))
            buys = module_data.get("buys") or []
            sells = module_data.get("sells") or []
            err = module_data.get("error")
            if buys or sells:
                has_trades = True
                for sym in buys:
                    trade_lines.append(f"BUY  {html.escape(str(sym))}  [{safe_module}]")
                for sym in sells:
                    trade_lines.append(f"SELL {html.escape(str(sym))}  [{safe_module}]")
                self.record_trade({
                    "time": result.get("timestamp", ""),
                    "module": module_name,
                    "buys": buys,
                    "sells": sells,
                })
            if err:
                error_lines.append(
                    f"{safe_module}: {html.escape(_clip_text(err, 140))}"
                )

        kill = bool(risk.get("kill_switch_active"))
        halt = bool(risk.get("daily_halt_active"))
        has_risk_event = kill or halt

        # Silence: no trades, no errors, no risk → don't message at all
        # (unless a command explicitly forced it via force_send).
        if not has_trades and not error_lines and not has_risk_event and not force_send:
            return

        mode_tag = "PAPER" if paper else "LIVE"
        header = f"<b>Trade cycle</b> — {mode_tag}"

        lines: list[str] = [header]

        if trade_lines:
            lines.append("")
            lines.extend(trade_lines)

        if error_lines:
            lines.append("")
            lines.append("<b>Errors</b>")
            for e in error_lines:
                lines.append(e)

        # Minimal portfolio footer (single line).
        equity = result.get("equity")
        pnl = result.get("realized_pnl")
        if equity is not None:
            pnl_str = ""
            if pnl is not None:
                pnl_str = f"  P&L {'+' if pnl >= 0 else ''}{_format_currency(pnl)}"
            lines.append("")
            lines.append(f"Equity {_format_currency(equity)}{pnl_str}")

        if kill:
            lines.append("")
            lines.append("<b>KILL SWITCH ACTIVE</b>")
        elif halt:
            lines.append("")
            lines.append("<b>Daily halt active</b>")

        if force_send and not has_trades and not error_lines and not has_risk_event:
            lines.append("")
            lines.append("No trades this cycle.")

        await self.send_message("\n".join(lines))

    async def send_heartbeat(self, status: dict[str, Any]) -> None:
        """Compact heartbeat / status response. Plain text, no dividers,
        no AI/latency/scan-date blocks. Serves scheduled heartbeats (opt-in)
        and on-demand /status.
        """
        equity = status.get("equity", 0)
        cash = status.get("cash", 0)
        pnl = status.get("realized_pnl", 0)
        positions = status.get("open_positions", []) or []
        risk = status.get("risk", {}) or {}
        kalshi_bal = status.get("kalshi_balance")

        lines = [
            "<b>Status</b>",
            f"Equity {_format_currency(equity)}  "
            f"P&L {'+' if pnl >= 0 else ''}{_format_currency(pnl)}",
            f"Cash {_format_currency(cash)}"
            + (
                f"  Kalshi {_format_currency(kalshi_bal)}"
                if kalshi_bal is not None else ""
            ),
        ]

        dd_daily = risk.get("daily_drawdown_pct", 0) or 0
        dd_total = risk.get("total_drawdown_pct", 0) or 0
        if dd_daily or dd_total:
            lines.append(f"DD: day {dd_daily:.1f}%  total {dd_total:.1f}%")

        if positions:
            # Simple one-line-per-position list (no asset-type grouping, no
            # entry/current/qty detail — the user can request /positions).
            lines.append("")
            lines.append(f"<b>Positions</b> ({len(positions)})")
            for pos in positions[:10]:
                sym = pos.get("symbol", "?")
                pnl_pos = pos.get("unrealized_pnl", 0) or 0
                entry = pos.get("entry_price") or 0
                current = pos.get("current_price") or 0
                pct = ((current - entry) / entry * 100) if entry else 0
                lines.append(
                    f"{html.escape(str(sym))}  "
                    f"{'+' if pct >= 0 else ''}{pct:.1f}%  "
                    f"{'+' if pnl_pos >= 0 else ''}{_format_currency(pnl_pos)}"
                )
            if len(positions) > 10:
                lines.append(f"… +{len(positions) - 10} more (see /positions)")
        else:
            lines.append("")
            lines.append("No open positions.")

        if risk.get("kill_switch_active"):
            lines.append("")
            lines.append("<b>KILL SWITCH ACTIVE</b>")
        elif risk.get("daily_halt_active"):
            lines.append("")
            lines.append("<b>Daily halt active</b>")

        if self._paused:
            lines.append("")
            lines.append("Paused — monitoring only.")

        await self.send_message("\n".join(lines))

    async def send_risk_alert(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("kill_switch_active"):
            dd = snapshot.get("total_drawdown_pct", 0) or 0
            await self.send_message(
                f"<b>Kill switch</b> engaged. Total DD {dd:.1f}%. "
                f"All positions closed."
            )
        elif snapshot.get("daily_halt_active"):
            dd = snapshot.get("daily_drawdown_pct", 0) or 0
            await self.send_message(
                f"<b>Daily halt</b> — drawdown {dd:.1f}%. No new trades today."
            )

    async def send_error(self, error_msg: str) -> None:
        await self.send_message(f"<b>Error</b>: {html.escape(_clip_text(error_msg, 300))}")

    async def send_leads_alert(
        self,
        strategy: str,
        leads: list[dict[str, Any]],
        max_per_message: int = 10,
    ) -> None:
        """Push a per-strategy watchlist update, chunked to stay within Telegram's limit."""
        if not leads:
            return
        max_per_msg = max(1, int(max_per_message))
        shown = leads[:max_per_msg]
        header = (
            f"<b>Watching</b> ({html.escape(strategy)}) — "
            f"{len(leads)} new lead{'s' if len(leads) != 1 else ''}"
        )
        cont_header = f"{header} <i>(cont.)</i>"
        trailer = f"… +{len(leads) - max_per_msg} more" if len(leads) > max_per_msg else None

        chunks: list[str] = []
        current_lines: list[str] = [header]
        current_len: int = len(header)

        for lead in shown:
            line = _format_lead_line(lead)
            extra = len(line) + 1  # +1 for the joining newline
            if current_len + extra > _TG_MAX_MSG_LEN:
                chunks.append("\n".join(current_lines))
                current_lines = [cont_header]
                current_len = len(cont_header)
            current_lines.append(line)
            current_len += extra

        if trailer is not None:
            extra = len(trailer) + 1
            if current_len + extra > _TG_MAX_MSG_LEN:
                chunks.append("\n".join(current_lines))
                current_lines = [cont_header]
                current_len = len(cont_header)
            current_lines.append(trailer)
            current_len += extra

        chunks.append("\n".join(current_lines))

        for chunk in chunks:
            await self.send_message(chunk)

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
            "leads": self._cmd_leads,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "kill": self._cmd_kill,
            "config": self._cmd_config,
            "polymarket": self._cmd_polymarket,
            "rotation": self._cmd_rotation,
            "predict": self._cmd_predictions,
            "perps": self._cmd_kalshi_perps,
            "sports": self._cmd_kalshi_sports,
            "backtest": self._cmd_backtest,
            "ai": self._cmd_ai,
            "history": self._cmd_history,
            "btc5min": self._cmd_btc5min,
            "markov": self._cmd_markov,
            "options": self._cmd_options,
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
            "<b>Zulubot</b>\n"
            "\n"
            "<b>Monitor</b>\n"
            "/status — portfolio + positions\n"
            "/positions — open positions (detailed)\n"
            "/risk — risk metrics\n"
            "/history — recent trades\n"
            "/audit [N] — trade audit tail\n"
            "/decisions [N] — strategy decisions\n"
            "/latency — module timings\n"
            "/leads — current watchlist (skipped buys + top targets)\n"
            "\n"
            "<b>Run</b>\n"
            "/crypto /stocks /rotation /predict /perps /sports /polymarket /btc5min /markov /options\n"
            "/backtest [range]\n"
            "\n"
            "<b>Control</b>\n"
            "/pause /resume /kill\n"
            "/resume force — clear kill switch and reset peak equity\n"
            "\n"
            "<b>Settings</b>\n"
            "/config /ai",
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
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_leads(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        leads_by_strategy = self._service.get_last_leads()
        sections: list[str] = []
        total = 0
        for strategy in sorted(leads_by_strategy):
            leads = leads_by_strategy[strategy]
            if not leads:
                continue
            total += len(leads)
            sections.append(f"<b>{html.escape(strategy)}</b> ({len(leads)})")
            for lead in leads[:10]:
                sections.append(_format_lead_line(lead))
            if len(leads) > 10:
                sections.append(f"… +{len(leads) - 10} more")
            sections.append("")
        if not sections:
            await update.message.reply_text("No leads tracked yet. Strategies populate this after each run.")  # type: ignore[union-attr]
            return
        active_strategies = sum(1 for v in leads_by_strategy.values() if v)
        header = f"<b>Watchlist</b> — {total} lead{'s' if total != 1 else ''} across {active_strategies} strateg{'ies' if active_strategies != 1 else 'y'}"
        await _send_html_chunks(update, header + "\n\n" + "\n".join(sections).rstrip())

    async def _cmd_crypto(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = self._service.run("crypto")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_stocks(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = self._service.run("stocks", force=True)
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_polymarket(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = self._service.run("polymarket")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_btc5min(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = self._service.run("btc5min")
            btc = result.get("btc5min", {})
            spot = btc.get("spot_price")
            k_imp = btc.get("kalshi_implied")
            p_imp = btc.get("poly_implied")
            k_sp = btc.get("kalshi_spread_pct")
            p_sp = btc.get("poly_spread_pct")
            actions = btc.get("actions", [])

            lines = ["<b>BTC 5-min spread</b>"]
            if spot:
                lines.append(f"Spot        ${spot:,.0f}")
            if k_imp:
                sp_str = f"  ({k_sp:+.3f}%)" if k_sp is not None else ""
                lines.append(f"Kalshi      ${k_imp:,.0f}{sp_str}")
            if p_imp:
                sp_str = f"  ({p_sp:+.3f}%)" if p_sp is not None else ""
                lines.append(f"Polymarket  ${p_imp:,.0f}{sp_str}")

            if actions:
                lines.append("")
                for act in actions:
                    venue = str(act.get("venue", "")).upper()
                    action = str(act.get("action", "")).upper()
                    sp = act.get("spread_pct", 0.0)
                    lines.append(f"{action} [{venue}]  spread {sp:+.3f}%")
            else:
                threshold = btc.get("threshold_pct", 2.0)
                lines.append(f"\nHold — spreads within ±{threshold:.1f}%.")

            await update.message.reply_text(  # type: ignore[union-attr]
                "\n".join(lines), parse_mode="HTML"
            )
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_markov(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = self._service.run("markov")
            m = (result.get("modules") or {}).get("markov") or {}

            status = m.get("status", "unknown")
            if status == "insufficient_data":
                hist = m.get("history_days", 0)
                min_h = m.get("min_history_days", 40)
                sym = html.escape(str(m.get("regime_symbol", "?")))
                await update.message.reply_text(  # type: ignore[union-attr]
                    f"<b>\U0001f9ed Markov</b> — building history\n"
                    f"⏳ {sym}: {int(hist)}/{int(min_h)} days collected.\n"
                    f"Strategy will begin signalling once {int(min_h)} days of data are available.",
                    parse_mode="HTML",
                )
                return

            if status not in ("completed",):
                await update.message.reply_text(  # type: ignore[union-attr]
                    f"<b>\U0001f9ed Markov</b>: {html.escape(str(status))}",
                    parse_mode="HTML",
                )
                return

            state_name = str(m.get("current_state", "?"))
            signal = float(m.get("signal", 0.0))
            bull = float(m.get("bull_prob", 0.0))
            bear = float(m.get("bear_prob", 0.0))
            sideways = float(m.get("sideways_prob", 0.0))
            n_trans = int(m.get("n_transitions", 0))
            hist_days = int(m.get("history_days", 0))
            regime_sym = html.escape(str(m.get("regime_symbol", "?")))
            opens = m.get("opens") or []
            closes = m.get("closes") or []
            stat_dist = m.get("stationary_distribution") or {}

            _state_icons = {"bull": "\U0001f402", "bear": "\U0001f43b", "sideways": "➡️"}
            state_icon = _state_icons.get(state_name.lower(), "❓")
            signal_arrow = "▲" if signal > 0 else ("▼" if signal < 0 else "●")
            signal_bar = _progress_bar((signal + 1) / 2, width=12)

            cfg = self._service.config.markov
            entry_thr = float(cfg.entry_signal_threshold)
            exit_thr = float(cfg.exit_signal_threshold)

            if signal > entry_thr:
                action_tag = "🟢 LONG SIGNAL"
            elif signal < exit_thr:
                action_tag = "🔴 CLOSE SIGNAL"
            else:
                action_tag = "⚪ NEUTRAL — hold"

            lines = [
                f"<b>\U0001f9ed Markov Regime</b> — {regime_sym}",
                f"{state_icon} State: <b>{html.escape(state_name.upper())}</b>",
                f"{signal_arrow} Signal: <code>{signal:+.4f}</code>  {signal_bar}",
                f"→ {action_tag}",
                "",
                "<b>Next-step probabilities</b>",
                f"\U0001f402 Bull      {_progress_bar(bull, 10)} {bull:.1%}",
                f"\U0001f43b Bear      {_progress_bar(bear, 10)} {bear:.1%}",
                f"➡️  Sideways  {_progress_bar(sideways, 10)} {sideways:.1%}",
            ]

            if stat_dist:
                s_bull = float(stat_dist.get("bull", 0))
                s_bear = float(stat_dist.get("bear", 0))
                s_side = float(stat_dist.get("sideways", 0))
                lines += [
                    "",
                    "<b>Long-run (stationary)</b>",
                    f"\U0001f402 {s_bull:.1%}  \U0001f43b {s_bear:.1%}  ➡️ {s_side:.1%}",
                ]

            lines += [
                "",
                f"📅 History: {hist_days}d  |  Transitions: {n_trans}",
                f"⚙️ Entry >{entry_thr:+.2f}  Exit <{exit_thr:+.2f}",
            ]

            if opens or closes:
                lines.append("")
                if opens:
                    lines.append("✅ Opened: " + ", ".join(html.escape(str(s)) for s in opens))
                if closes:
                    lines.append("🔴 Closed: " + ", ".join(html.escape(str(s)) for s in closes))

            await _send_html_chunks(update, "\n".join(lines))
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_rotation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = self._service.run("rotation")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_options(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = await asyncio.to_thread(self._service.run, "options")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_predictions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = self._service.run("predictions")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_kalshi_perps(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = await asyncio.to_thread(self._service.run, "kalshi_perps")
            result["force_send"] = True
            await self.send_trade_alert(result)
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_kalshi_sports(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Run the sports strategy and reply with today's model picks."""
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            result = await asyncio.to_thread(self._service.run, "kalshi_sports")
            module = (result.get("modules") or {}).get("kalshi_sports") or {}
            if module.get("status") == "disabled":
                await update.message.reply_text(  # type: ignore[union-attr]
                    "Sports strategy is disabled. Set KALSHI_SPORTS_ENABLED=true."
                )
                return
            lines = [
                "<b>Sports picks</b> — "
                f"{int(module.get('games_scanned', 0) or 0)} games, "
                f"{int(module.get('markets_matched', 0) or 0)} Kalshi markets matched"
            ]
            preds = sorted(module.get("predictions") or [], key=lambda p: p.get("edge", 0), reverse=True)
            for pred in preds[:10]:
                lines.append(
                    f"\u2022 <b>{html.escape(str(pred.get('pick', '?')))}</b> "
                    f"({html.escape(str(pred.get('game', '')))})\n"
                    f"   model {float(pred.get('p_yes', 0)):.0%} (YES) | "
                    f"{html.escape(str(pred.get('side', '')).upper())} ask "
                    f"{float(pred.get('yes_ask' if pred.get('side') == 'yes' else 'no_ask', 0)):.2f} | "
                    f"edge {float(pred.get('edge', 0)):+.1%}"
                )
            if not preds:
                lines.append("No predictable games found right now.")
            for sym in module.get("buys") or []:
                lines.append(f"BUY  {html.escape(str(sym))}")
            if module.get("error"):
                lines.append(f"Error: {html.escape(_clip_text(module['error'], 140))}")
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")  # type: ignore[union-attr]

    async def _cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return

        timerange = context.args[0] if context.args else None
        await update.message.reply_text(  # type: ignore[union-attr]
            f"Running backtest{' (' + timerange + ')' if timerange else ''}…"
        )
        try:
            from trading_bot.integrations.freqtrade_adapter import FreqtradeAdapter
            adapter = FreqtradeAdapter(self._service.config)
            if not adapter.is_available:
                await update.message.reply_text(  # type: ignore[union-attr]
                    "Freqtrade not available. "
                    "Install: pip install -r requirements-freqtrade.txt. "
                    "Enable: FREQTRADE_ENABLED=true in .env"
                )
                return
            result = adapter.run_backtest(timerange=timerange)
            lines = ["<b>Backtest</b>"]
            if "strategy" in result:
                lines.append(f"Strategy   {result['strategy']}")
            if "total_trades" in result:
                lines.append(f"Trades     {result['total_trades']}")
            if "profit_total_pct" in result:
                pct = result["profit_total_pct"]
                lines.append(f"Profit     {'+' if pct >= 0 else ''}{pct:.2f}%")
            if "max_drawdown_pct" in result:
                lines.append(f"Max DD     {result['max_drawdown_pct']:.2f}%")
            if "sharpe_ratio" in result:
                lines.append(f"Sharpe     {result['sharpe_ratio']:.2f}")
            if "win_rate" in result:
                lines.append(f"Win rate   {result['win_rate']:.1%}")
            await self.send_message("\n".join(lines))
        except Exception as exc:
            await update.message.reply_text(f"Backtest error: {exc}")  # type: ignore[union-attr]

    async def _cmd_ai(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        cfg = self._service.config
        if not cfg.trading_agents.enabled:
            await update.message.reply_text(  # type: ignore[union-attr]
                "<b>AI</b>: disabled. Enable via TRADING_AGENTS_ENABLED=true.",
                parse_mode="HTML",
            )
            return

        ai = getattr(self._service, "ai_service", None)
        if ai is None or not hasattr(ai, "get_monthly_cost"):
            await update.message.reply_text(  # type: ignore[union-attr]
                "<b>AI</b>: enabled but unavailable. "
                "Check ANTHROPIC_API_KEY / OLLAMA_BASE_URL.",
                parse_mode="HTML",
            )
            return

        monthly = ai.get_monthly_cost()
        remaining = ai.get_budget_remaining()
        cap = cfg.trading_agents.max_cost_per_month
        usage_pct = (monthly / cap * 100) if cap > 0 else 0
        multi = cfg.trading_agents.multi_analyst_enabled
        analysts = list(cfg.trading_agents.selected_analysts)

        lines = [
            "<b>AI</b>",
            f"Mode       {cfg.trading_agents.confirmation_mode}",
            f"Quick      {cfg.trading_agents.quick_model}",
            f"Deep       {cfg.trading_agents.deep_model}",
            f"Spent      ${monthly:.4f} ({usage_pct:.0f}% of ${cap:.2f}/mo)",
            f"Remaining  ${remaining:.2f}",
        ]
        if hasattr(ai, "get_circuit_breaker_status"):
            breaker = ai.get_circuit_breaker_status() or {}
            b_state = str(breaker.get("state", "?"))
            if b_state == "disabled":
                lines.append("Breaker    off")
            else:
                b_line = (
                    f"Breaker    {b_state} "
                    f"({int(breaker.get('failure_count', 0))}"
                    f"/{int(breaker.get('failure_threshold', 0))} failures)"
                )
                remaining_s = float(breaker.get("remaining_open_seconds", 0) or 0)
                if remaining_s > 0:
                    b_line += f", retry in {remaining_s:.0f}s"
                lines.append(b_line)
        if multi and analysts:
            lines.append("Analysts   " + ", ".join(analysts))
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]

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

        lines = ["<b>Latency</b>"]
        for name, stats in latency.items():
            lines.append(
                f"{html.escape(name)}  "
                f"last {stats.get('last_ms', 0):.1f}ms  "
                f"avg {stats.get('avg_ms', 0):.1f}ms  "
                f"max {stats.get('max_ms', 0):.1f}ms  "
                f"n={int(stats.get('count', 0))}"
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

        lines = ["<b>Trade audit</b>"]
        for entry in entries:
            lines.append(
                f"{html.escape(str(entry.get('timestamp', 'n/a')))}  "
                f"{html.escape(str(entry.get('event', '?')).upper())}  "
                f"{html.escape(str(entry.get('symbol', '?')))}  "
                f"@{float(entry.get('price', 0.0)):.4f} "
                f"x{float(entry.get('quantity', 0.0)):.4f}"
            )
            reason = entry.get("reason")
            if reason:
                lines.append(f"  reason: {html.escape(str(reason))}")
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
                "No decisions recorded yet. Run a strategy cycle first.",
            )
            return

        lines = [f"<b>Decisions</b> (last {len(entries)})"]
        for entry in entries:
            action = str(entry.get("action", "?"))
            symbol = html.escape(str(entry.get("symbol", "?")))
            strategy = html.escape(str(entry.get("strategy", "?")))
            reason = html.escape(str(entry.get("reason", "?")))
            ts_raw = str(entry.get("timestamp", ""))
            ts_short = ts_raw[11:19] if len(ts_raw) >= 19 else ts_raw

            prefix = f"{ts_short}  " if ts_short else ""
            lines.append(
                f"\n{prefix}<b>{html.escape(action.upper())}</b> {symbol}  [{strategy}]"
            )
            lines.append(f"  {reason}")

            signals = entry.get("signals") or {}
            if signals:
                sig_parts = []
                for key, val in signals.items():
                    if val is None:
                        continue
                    sig_parts.append(f"{html.escape(str(key))}={html.escape(str(val))}")
                if sig_parts:
                    lines.append(f"  {' | '.join(sig_parts)}")

        await _send_html_chunks(update, "\n".join(lines))

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        status = self._service.status()
        positions = status.get("open_positions", []) or []
        kalshi_bal = status.get("kalshi_balance")

        if not positions:
            msg = "<b>Positions</b>\nNone open."
            if kalshi_bal is not None:
                msg += f"\nKalshi {_format_currency(kalshi_bal)}"
            await update.message.reply_text(msg, parse_mode="HTML")  # type: ignore[union-attr]
            return

        # Group by asset type for readability
        by_type: dict[str, list[dict[str, Any]]] = {}
        for pos in positions:
            by_type.setdefault(pos.get("asset_type", "other"), []).append(pos)

        lines = [f"<b>Positions</b> ({len(positions)})"]
        for atype, type_positions in by_type.items():
            type_pnl = sum(p.get("unrealized_pnl", 0) or 0 for p in type_positions)
            type_value = sum(p.get("market_value", 0) or 0 for p in type_positions)
            lines.append(
                f"\n<b>{atype.upper()}</b> ({len(type_positions)}) — "
                f"{_format_currency(type_value)}  "
                f"P&L {'+' if type_pnl >= 0 else ''}{_format_currency(type_pnl)}"
            )
            for pos in type_positions:
                sym = pos.get("symbol", "?")
                qty = pos.get("quantity", 0) or 0
                entry = pos.get("entry_price", 0) or 0
                current = pos.get("current_price", 0) or 0
                pnl_val = pos.get("unrealized_pnl", 0) or 0
                mv = pos.get("market_value", 0) or 0
                pct = ((current - entry) / entry * 100) if entry else 0
                lines.append(
                    f"{html.escape(str(sym))}  "
                    f"qty {qty:.6f}  @ ${entry:,.4f} → ${current:,.4f}  "
                    f"({'+' if pct >= 0 else ''}{pct:.1f}%)  "
                    f"value {_format_currency(mv)}  "
                    f"P&L {'+' if pnl_val >= 0 else ''}{_format_currency(pnl_val)}"
                )

        if kalshi_bal is not None:
            lines.append(f"\nKalshi {_format_currency(kalshi_bal)}")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        status = self._service.status()
        risk = status.get("risk", {}) or {}

        dd_daily = risk.get("daily_drawdown_pct", 0) or 0
        dd_total = risk.get("total_drawdown_pct", 0) or 0

        post_kill = bool(risk.get("post_kill_cooldown_active"))
        cleared_at = risk.get("kill_switch_cleared_at")
        cfg = self._service.config if self._service else None
        cooldown_h = cfg.risk.post_kill_cooldown_hours if cfg else 1.0

        lines = [
            "<b>Risk</b>",
            f"Daily DD    {dd_daily:.2f}% / 5.0%",
            f"Total DD    {dd_total:.2f}% / 15.0%",
            f"Peak        {_format_currency(risk.get('peak_equity', 0))}",
            f"Baseline    {_format_currency(risk.get('daily_baseline_equity', 0))}",
            f"Daily halt  {'ACTIVE' if risk.get('daily_halt_active') else 'ok'}",
            f"Kill switch {'ENGAGED' if risk.get('kill_switch_active') else 'ok'}",
        ]
        if post_kill:
            cleared_str = html.escape(str(cleared_at or "?")[:19])
            lines.append(f"Post-kill   🕐 cooldown active ({cooldown_h:.0f}h) — cleared {cleared_str}")
        elif cleared_at and not risk.get("kill_switch_active"):
            cleared_str = html.escape(str(cleared_at)[:19])
            lines.append(f"Post-kill   ✅ cooldown elapsed (cleared {cleared_str})")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        self._paused = True
        await update.message.reply_text(  # type: ignore[union-attr]
            "Paused. No trades will execute. /resume to restart.",
        )

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        force = bool(context.args) and str(context.args[0]).strip().lower() == "force"
        try:
            state = self._service.state_store.load()
            cleared_kill_switch = False
            peak_reset_msg = ""
            if state.kill_switch_engaged:
                summary = self._service.clear_kill_switch(force=force)
                cleared_kill_switch = True
                ks = (summary or {}).get("modules", {}).get("kill_switch", {})
                if "peak_reset_from" in ks:
                    peak_reset_msg = (
                        f" (peak reset ${ks['peak_reset_from']:.2f} → ${ks['peak_reset_to']:.2f})"
                    )
            self._paused = False
            cooldown_h = self._service.config.risk.post_kill_cooldown_hours
            if cleared_kill_switch:
                cooldown_note = (
                    f" New entries blocked for {cooldown_h:.0f}h (post-kill cooldown)."
                    if cooldown_h > 0 else ""
                )
                msg = f"Resumed. Kill switch cleared{peak_reset_msg}.{cooldown_note} Trading active."
            else:
                msg = "Resumed. Trading active."
            await update.message.reply_text(msg)  # type: ignore[union-attr]
        except Exception as exc:
            hint = "" if force else "\n\nUse /resume force to reset the high-water mark to current equity."
            await update.message.reply_text(f"Resume blocked: {exc}{hint}")  # type: ignore[union-attr]

    async def _cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        try:
            summary = self._service.engage_kill_switch(reason="manual_kill_switch")
            self._paused = True
            failed = summary.get("modules", {}).get("kill_switch", {}).get("failed", [])
            msg = "Kill switch engaged. All positions closed. Bot paused."
            if failed:
                msg += f"\nCould not auto-close: {', '.join(failed)}"
            await update.message.reply_text(msg)  # type: ignore[union-attr]
        except Exception as exc:
            await update.message.reply_text(f"Kill error: {exc}")  # type: ignore[union-attr]

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if self._service is None:
            await update.message.reply_text("Service not initialized.")  # type: ignore[union-attr]
            return
        cfg = self._service.config

        crypto_extras: list[str] = []
        if cfg.crypto.symbol_weights:
            crypto_extras.append(
                "  weights   "
                + ", ".join(f"{s} ×{w:g}" for s, w in cfg.crypto.symbol_weights.items())
            )
        if cfg.crypto.take_profit_overrides:
            crypto_extras.append(
                "  TP        "
                + ", ".join(f"{s} {p:.0%}" for s, p in cfg.crypto.take_profit_overrides.items())
                + f"  (default {cfg.crypto.take_profit_pct:.0%})"
            )
        if cfg.crypto.stop_loss_overrides:
            crypto_extras.append(
                "  SL        "
                + ", ".join(f"{s} {p:.0%}" for s, p in cfg.crypto.stop_loss_overrides.items())
                + f"  (default {cfg.crypto.stop_loss_pct:.0%})"
            )

        mem_cycles = cfg.altcoin_rotation.qty_memory_cycles
        rotation_mem = f"mem {mem_cycles}cy" if mem_cycles > 0 else "mem ∞"

        lines = [
            "<b>Config</b>",
            f"Mode         {'PAPER' if cfg.paper_mode else 'LIVE'}",
            f"Cash         {_format_currency(cfg.initial_cash)}",
            f"Paused       {'yes' if self._paused else 'no'}",
            "",
            "<b>Risk</b>",
            f"Position    {cfg.risk.max_position_fraction*100:.1f}%",
            f"Daily halt  {cfg.risk.daily_halt_drawdown*100:.1f}%",
            f"Kill        {cfg.risk.kill_switch_drawdown*100:.1f}%",
            "",
            "<b>Strategies</b>",
            f"Crypto      {', '.join(cfg.crypto.symbols)}  "
            f"buy<{cfg.crypto.buy_fear_threshold} sell>{cfg.crypto.sell_fear_threshold}"
            + ("  funding-gate" if cfg.crypto.require_negative_funding else "")
            + (f"  sma{cfg.crypto.trend_sma_days}-gate"
               if cfg.crypto.trend_filter_enabled else ""),
            *crypto_extras,
            f"Stocks      P/E<{cfg.stock.max_pe_ratio} P/B<{cfg.stock.max_pb_ratio} "
            f"D/E<{cfg.stock.max_debt_to_equity} VIX>{cfg.stock.vix_buy_threshold}",
            f"Rotation    {'on' if cfg.altcoin_rotation.enabled else 'off'}  "
            f"fee {cfg.altcoin_rotation.fee_pct*100:.1f}% x{cfg.altcoin_rotation.scout_multiplier}  "
            f"{rotation_mem}",
            f"Kalshi      {'on' if cfg.predictions.enabled else 'off'}  "
            f"YES<{cfg.predictions.buy_yes_threshold} NO>{cfg.predictions.buy_no_threshold}"
            + ("  fng-gate" if cfg.predictions.require_crypto_fear_greed else "")
            + f"  pages≤{cfg.predictions.max_pages_per_series}"
            + ("  auto-series" if cfg.predictions.discover_series else ""),
            f"Polymarket  {'on' if cfg.polymarket_strategy.enabled else 'off'}",
            "",
            "<b>Schedule</b>",
            f"Crypto      {cfg.scheduler.crypto_interval_minutes}m",
            f"Rotation    {cfg.altcoin_rotation.check_interval_minutes}m",
            f"Stocks      {cfg.scheduler.stock_check_hour}:00 daily",
            f"Heartbeat   "
            + (f"{cfg.scheduler.heartbeat_interval_minutes}m"
               if cfg.scheduler.heartbeat_enabled else "off"),
            "",
            f"<b>AI</b>: {'on' if cfg.trading_agents.enabled else 'off'}"
            + (
                f" ({cfg.trading_agents.confirmation_mode}, "
                f"breaker {'on' if cfg.trading_agents.circuit_breaker_enabled else 'off'})"
                if cfg.trading_agents.enabled else ""
            ),
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_auth(update):
            return
        if not self._trade_history:
            await update.message.reply_text(  # type: ignore[union-attr]
                "<b>Trade history</b>\nNo recent trades.",
                parse_mode="HTML",
            )
            return
        lines = ["<b>Recent trades</b>"]
        for trade in self._trade_history[-10:]:
            t = trade.get("time", "?")[:16]
            m = trade.get("module", "?")
            buys = trade.get("buys", [])
            sells = trade.get("sells", [])
            parts: list[str] = []
            for s in buys:
                parts.append(f"BUY {html.escape(str(s))}")
            for s in sells:
                parts.append(f"SELL {html.escape(str(s))}")
            lines.append(f"{t}  [{m}]  {' | '.join(parts)}")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")  # type: ignore[union-attr]
