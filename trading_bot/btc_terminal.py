"""Combined live terminal for the trading bot.

Launched via: python -m trading_bot btc-terminal

This terminal combines:
  - the BTC 5-minute spread dashboard,
  - the live daemon scheduler,
  - portfolio and risk status,
  - recent bot activity / trade logs.

Requires: rich>=13.0.0
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from trading_bot.config import AppConfig
from trading_bot.notifications import NotificationDispatcher
from trading_bot.scheduler import SchedulerService
from trading_bot.service import TradingBotService
from trading_bot.strategies.btc_5min import compute_implied_median, parse_strike
from trading_bot.telegram_bot import TradingTelegramBot

_REFRESH_SECONDS = 5
_MARKET_REFRESH_SECONDS = 30
_ACTIVITY_LIMIT = 8
_RECENT_TRADE_LIMIT = 6
_RECENT_DECISION_LIMIT = 6

_OPEN_TRADE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\w+\s+\[[^\]]+\]\s+OPEN\s+"
    r"(?P<symbol>\S+)\s+asset=(?P<asset_type>\S+)\s+strategy=(?P<strategy>\S+)\s+"
    r"qty=(?P<quantity>-?\d+(?:\.\d+)?)\s+price=(?P<price>-?\d+(?:\.\d+)?)"
)
_CLOSE_TRADE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\w+\s+\[[^\]]+\]\s+CLOSE\s+"
    r"(?P<symbol>\S+)\s+qty=(?P<quantity>-?\d+(?:\.\d+)?)\s+price=(?P<price>-?\d+(?:\.\d+)?)"
    r".*\s+reason=(?P<reason>.+)$"
)


class TerminalRuntimeState:
    def __init__(self) -> None:
        self.last_summary: dict[str, Any] | None = None
        self.last_heartbeat: dict[str, Any] | None = None
        self.last_btc_result: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.recent_events: deque[str] = deque(maxlen=12)

    def record_summary(self, result: dict[str, Any], *, source: str) -> None:
        self.last_summary = result
        modules = result.get("modules", {}) or {}
        if "btc5min" in modules:
            self.last_btc_result = modules["btc5min"]

        stamp = _format_iso_timestamp(result.get("timestamp"))
        module_names = ", ".join(modules.keys()) if modules else source
        positions = len(result.get("open_positions", []) or [])
        equity = float(result.get("equity", 0.0) or 0.0)
        self.recent_events.appendleft(f"{stamp} {module_names}  eq=${equity:,.2f}  pos={positions}")

    def record_heartbeat(self, status: dict[str, Any]) -> None:
        self.last_heartbeat = status
        self.record_summary(status, source="heartbeat")

    def record_error(self, error_msg: str) -> None:
        self.last_error = error_msg
        self.recent_events.appendleft(f"{datetime.now().strftime('%H:%M:%S')} ERROR {error_msg[:140]}")

    def record_message(self, message: str) -> None:
        self.recent_events.appendleft(f"{datetime.now().strftime('%H:%M:%S')} {message[:140]}")


class TerminalNotificationDispatcher(NotificationDispatcher):
    def __init__(self, telegram: TradingTelegramBot | None, logger: logging.Logger, terminal_state: TerminalRuntimeState) -> None:
        super().__init__(telegram, logger)
        self.terminal_state = terminal_state

    async def notify_trade(self, result: dict[str, Any]) -> None:
        self.terminal_state.record_summary(result, source="trade")
        await super().notify_trade(result)

    async def notify_heartbeat(self, status: dict[str, Any]) -> None:
        self.terminal_state.record_heartbeat(status)
        await super().notify_heartbeat(status)

    async def notify_error(self, error_msg: str) -> None:
        self.terminal_state.record_error(error_msg)
        await super().notify_error(error_msg)

    async def notify_startup(self, paper_mode: bool, vol_targeting_enabled: bool) -> None:
        mode = "LIVE" if not paper_mode else "PAPER"
        self.terminal_state.record_message(f"startup mode={mode} vol_targeting={vol_targeting_enabled}")
        await super().notify_startup(paper_mode, vol_targeting_enabled)

    async def notify_shutdown(self) -> None:
        self.terminal_state.record_message("shutdown requested")
        await super().notify_shutdown()


def _format_iso_timestamp(raw_value: Any) -> str:
    if not raw_value:
        return "--:--:--"
    try:
        return datetime.fromisoformat(str(raw_value).replace("Z", "+00:00")).strftime("%H:%M:%S")
    except ValueError:
        raw_text = str(raw_value)
        return raw_text[11:19] if len(raw_text) >= 19 else raw_text


def _spread_color(spread_pct: float | None) -> str:
    if spread_pct is None:
        return "dim"
    if spread_pct > 2.0:
        return "bold green"
    if spread_pct > 0.5:
        return "green"
    if spread_pct < -2.0:
        return "bold red"
    if spread_pct < -0.5:
        return "red"
    return "yellow"


def _price_text(price: float | None, label: str) -> Text:
    if price is None:
        return Text(f"{label}: n/a", style="dim")
    return Text(f"{label}: ${price:,.0f}", style="bold white")


def _detach_console_handlers(logger: logging.Logger) -> list[logging.Handler]:
    removed: list[logging.Handler] = []
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            removed.append(handler)
    return removed


def _restore_handlers(logger: logging.Logger, handlers: list[logging.Handler]) -> None:
    for handler in handlers:
        logger.addHandler(handler)


def _tail_interesting_log_lines(path: Path, limit: int = _ACTIVITY_LIMIT) -> list[str]:
    if limit <= 0 or not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    interesting: list[str] = []
    markers = (
        " OPEN ",
        " CLOSE ",
        "LIVE ",
        "Kalshi order",
        "Trade cycle result",
        "Heartbeat:",
        "Running ",
        "failed",
        "ERROR",
    )
    for line in reversed(lines):
        if any(marker in line for marker in markers):
            interesting.append(line[-160:])
        if len(interesting) >= limit:
            break
    interesting.reverse()
    return interesting


def _build_price_panel(
    spot: float | None,
    kalshi_implied: float | None,
    poly_implied: float | None,
    kalshi_spread: float | None,
    poly_spread: float | None,
    updated_at: str,
) -> Panel:
    lines = Text()
    lines.append(_price_text(spot, "Coinbase Spot"))
    lines.append("\n")

    if kalshi_implied is not None:
        ks_str = f"{kalshi_spread:+.2f}%" if kalshi_spread is not None else ""
        lines.append(f"Kalshi Implied:    ${kalshi_implied:,.0f}  ", style="white")
        lines.append(f"{ks_str}\n", style=_spread_color(kalshi_spread))
    else:
        lines.append("Kalshi Implied:    n/a\n", style="dim")

    if poly_implied is not None:
        ps_str = f"{poly_spread:+.2f}%" if poly_spread is not None else ""
        lines.append(f"Polymarket Implied:${poly_implied:,.0f}  ", style="white")
        lines.append(f"{ps_str}\n", style=_spread_color(poly_spread))
    else:
        lines.append("Polymarket Implied:n/a\n", style="dim")

    lines.append(f"\nUpdated: {updated_at}", style="dim")
    return Panel(lines, title="[bold cyan]BTC PRICES[/bold cyan]", border_style="cyan")


def _build_signals_panel(result: dict[str, Any] | None) -> Panel:
    if not result:
        return Panel(Text("No BTC5Min cycle yet - waiting for first scan...", style="dim"), title="[bold cyan]BTC SIGNALS[/bold cyan]", border_style="cyan")

    actions = result.get("actions", []) or []
    threshold = float(result.get("threshold_pct", 2.0) or 2.0)
    lines = Text()

    if not actions:
        ks = result.get("kalshi_spread_pct")
        ps = result.get("poly_spread_pct")
        lines.append(f"HOLD - spreads within +/-{threshold:.1f}% threshold\n", style="yellow")
        if ks is not None:
            ks_value = float(ks)
            lines.append(f"  Kalshi spread:    {ks_value:+.3f}%\n", style=_spread_color(ks_value))
        if ps is not None:
            ps_value = float(ps)
            lines.append(f"  Polymarket spread:{ps_value:+.3f}%\n", style=_spread_color(ps_value))
    else:
        for action in actions:
            venue = str(action.get("venue", "")).upper()
            action_name = str(action.get("action", "")).upper()
            spread_pct = float(action.get("spread_pct", 0.0) or 0.0)
            symbols = action.get("symbols", []) or []
            color = "green" if "YES" in action_name or "SPOT" in action_name else "red"
            lines.append(f"[{venue}] ", style="bold")
            lines.append(f"{action_name}  spread={spread_pct:+.3f}%\n", style=color)
            for symbol in symbols[:3]:
                lines.append(f"  -> {symbol}\n", style="dim")

    kalshi_n = int(result.get("kalshi_markets_used", 0) or 0)
    poly_n = int(result.get("poly_markets_used", 0) or 0)
    lines.append(f"\nMarket depth: Kalshi={kalshi_n}  Poly={poly_n}", style="dim")
    return Panel(lines, title="[bold cyan]BTC SIGNALS[/bold cyan]", border_style="cyan")


def _build_positions_panel(summary: dict[str, Any] | None) -> Panel:
    if not summary:
        return Panel(Text("No portfolio data yet", style="dim"), title="[bold cyan]POSITIONS[/bold cyan]", border_style="cyan")

    positions = summary.get("open_positions", []) or []
    if not positions:
        return Panel(Text("No open positions", style="dim"), title="[bold cyan]POSITIONS[/bold cyan]", border_style="cyan")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="white")
    table.add_column("Strategy", style="white")
    table.add_column("Value", justify="right")
    table.add_column("P&L", justify="right")

    for position in positions[:10]:
        pnl = float(position.get("unrealized_pnl", 0.0) or 0.0)
        table.add_row(
            str(position.get("symbol", "?")),
            str(position.get("strategy", "?")),
            f"${float(position.get('market_value', 0.0) or 0.0):,.2f}",
            Text(f"${pnl:+,.2f}", style="green" if pnl >= 0 else "red"),
        )

    return Panel(table, title="[bold cyan]POSITIONS[/bold cyan]", border_style="cyan")


def _build_status_panel(summary: dict[str, Any] | None, terminal_state: TerminalRuntimeState) -> Panel:
    if not summary:
        return Panel(Text("Waiting for daemon summary...", style="dim"), title="[bold cyan]BOT STATUS[/bold cyan]", border_style="cyan")

    lines = Text()
    live_mode = not bool(summary.get("paper_mode", True))
    lines.append(f"Mode: {'LIVE' if live_mode else 'PAPER'}\n", style="bold green" if live_mode else "bold yellow")
    lines.append(f"Equity: ${float(summary.get('equity', 0.0) or 0.0):,.2f}\n")
    lines.append(f"Cash: ${float(summary.get('cash', 0.0) or 0.0):,.2f}\n")
    lines.append(f"Realized P&L: ${float(summary.get('realized_pnl', 0.0) or 0.0):+,.2f}\n")
    lines.append(f"Open positions: {len(summary.get('open_positions', []) or [])}\n")

    risk = summary.get("risk", {}) or {}
    lines.append(
        f"Risk: daily={float(risk.get('daily_drawdown_pct', 0.0) or 0.0):.2f}%  "
        f"total={float(risk.get('total_drawdown_pct', 0.0) or 0.0):.2f}%\n",
        style="yellow",
    )
    if risk.get("daily_halt_active"):
        lines.append("Daily halt active\n", style="bold red")
    if risk.get("kill_switch_active"):
        lines.append("Kill switch active\n", style="bold red")
    lines.append(f"Last update: {_format_iso_timestamp(summary.get('timestamp'))}\n", style="dim")

    modules = summary.get("modules", {}) or {}
    if modules:
        lines.append("\nLast cycle modules:\n", style="bold")
        for name, data in modules.items():
            opens = len(data.get("opens", []) or [])
            closes = len(data.get("closes", []) or [])
            status = str(data.get("status", "ok"))
            suffix = f" (+{opens}/-{closes})" if opens or closes else ""
            lines.append(f"  {name}: {status}{suffix}\n")

    if terminal_state.last_error:
        lines.append(f"\nLast error: {terminal_state.last_error[:120]}", style="red")

    return Panel(lines, title="[bold cyan]BOT STATUS[/bold cyan]", border_style="cyan")


def _format_audit_entry(entry: dict[str, Any]) -> str:
    return (
        f"{_format_iso_timestamp(entry.get('timestamp'))} "
        f"{str(entry.get('event', 'trade')).upper()} "
        f"{str(entry.get('symbol', '?'))} "
        f"qty={float(entry.get('quantity', 0.0) or 0.0):.4f} "
        f"px={float(entry.get('price', 0.0) or 0.0):.4f}"
    )


def _format_trade_quantity(quantity: float) -> str:
    text = f"{quantity:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_trade_price(price: float) -> str:
    if price >= 1_000:
        return f"${price:,.2f}"
    return f"${price:,.4f}"


def _parse_trade_log_line(line: str) -> dict[str, Any] | None:
    open_match = _OPEN_TRADE_RE.match(line)
    if open_match:
        return {
            "timestamp": _format_iso_timestamp(open_match.group("timestamp")),
            "symbol": open_match.group("symbol"),
            "side": "BUY",
            "quantity": float(open_match.group("quantity")),
            "price": float(open_match.group("price")),
            "reason": open_match.group("strategy"),
        }

    close_match = _CLOSE_TRADE_RE.match(line)
    if close_match:
        return {
            "timestamp": _format_iso_timestamp(close_match.group("timestamp")),
            "symbol": close_match.group("symbol"),
            "side": "SELL",
            "quantity": float(close_match.group("quantity")),
            "price": float(close_match.group("price")),
            "reason": close_match.group("reason"),
        }

    return None


def _extract_recent_trades(log_lines: list[str], limit: int = _RECENT_TRADE_LIMIT) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    for line in reversed(log_lines):
        parsed = _parse_trade_log_line(line)
        if parsed is None:
            continue
        trades.append(parsed)
        if len(trades) >= limit:
            break
    return trades


def _clip_text(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[:width - 3]}..."


def _build_recent_trades_panel(audit_entries: list[dict[str, Any]], log_lines: list[str]) -> Panel:
    trade_rows: list[dict[str, Any]] = []
    if audit_entries:
        for entry in reversed(audit_entries[-_RECENT_TRADE_LIMIT:]):
            event = str(entry.get("event", "trade")).upper()
            side = "BUY" if event in {"OPEN", "BUY"} else "SELL" if event in {"CLOSE", "SELL"} else event
            trade_rows.append(
                {
                    "timestamp": _format_iso_timestamp(entry.get("timestamp")),
                    "symbol": str(entry.get("symbol", "?")),
                    "side": side,
                    "quantity": float(entry.get("quantity", 0.0) or 0.0),
                    "price": float(entry.get("price", 0.0) or 0.0),
                    "reason": str(entry.get("reason") or entry.get("strategy") or "trade"),
                }
            )
    else:
        trade_rows = _extract_recent_trades(log_lines, _RECENT_TRADE_LIMIT)

    if not trade_rows:
        return Panel(Text("No trades captured yet", style="dim"), title="[bold cyan]RECENT TRADES[/bold cyan]", border_style="cyan")

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Time", no_wrap=True)
    table.add_column("Symbol", no_wrap=True)
    table.add_column("Side", no_wrap=True)
    table.add_column("Qty", justify="right", no_wrap=True)
    table.add_column("Price", justify="right", no_wrap=True)
    table.add_column("Reason")

    for trade in trade_rows:
        side = str(trade.get("side", "-")).upper()
        side_style = "green" if side == "BUY" else "red" if side == "SELL" else "yellow"
        table.add_row(
            str(trade.get("timestamp", "--:--:--")),
            str(trade.get("symbol", "?")),
            Text(side, style=side_style),
            _format_trade_quantity(float(trade.get("quantity", 0.0) or 0.0)),
            _format_trade_price(float(trade.get("price", 0.0) or 0.0)),
            _clip_text(str(trade.get("reason", "trade")), 48),
        )

    return Panel(table, title="[bold cyan]RECENT TRADES[/bold cyan]", border_style="cyan")


def _build_decisions_panel(decision_entries: list[dict[str, Any]], *, ai_enabled: bool) -> Panel:
    if not ai_enabled:
        return Panel(
            Text("Trading agents are disabled. Enable TRADING_AGENTS_ENABLED=true to capture AI decisions.", style="dim"),
            title="[bold cyan]AGENT DECISIONS[/bold cyan]",
            border_style="cyan",
        )

    if not decision_entries:
        return Panel(Text("No recent decisions captured yet", style="dim"), title="[bold cyan]AGENT DECISIONS[/bold cyan]", border_style="cyan")

    ai_entries = [
        entry for entry in decision_entries
        if isinstance(entry.get("signals"), dict) and isinstance(entry.get("signals", {}).get("ai_verdict"), dict)
    ]
    visible_entries = list(reversed((ai_entries or decision_entries)[-_RECENT_DECISION_LIMIT:]))

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Time", no_wrap=True)
    table.add_column("Strategy", no_wrap=True)
    table.add_column("Symbol", no_wrap=True)
    table.add_column("Decision", no_wrap=True)
    table.add_column("Conf", justify="right", no_wrap=True)
    table.add_column("Reason")

    for entry in visible_entries:
        signals = entry.get("signals") if isinstance(entry.get("signals"), dict) else {}
        ai_verdict = signals.get("ai_verdict") if isinstance(signals, dict) and isinstance(signals.get("ai_verdict"), dict) else {}
        decision_text = str(ai_verdict.get("action") or signals.get("ai_action") or entry.get("action") or "-").upper()
        decision_style = "green" if decision_text == "BUY" else "red" if decision_text in {"SELL", "SKIP"} else "yellow"
        confidence = ai_verdict.get("confidence")
        if confidence in (None, ""):
            confidence = signals.get("ai_confidence")
        confidence_text = "-"
        if confidence not in (None, ""):
            try:
                confidence_text = f"{float(confidence):.2f}"
            except (TypeError, ValueError):
                confidence_text = str(confidence)

        reason = str(ai_verdict.get("reasoning") or signals.get("ai_reasoning") or entry.get("reason") or "decision")
        model_used = str(ai_verdict.get("model_used") or "")
        cached = bool(ai_verdict.get("cached", False))
        suffix: list[str] = []
        if model_used:
            suffix.append(_clip_text(model_used, 18))
        if cached:
            suffix.append("cached")
        if suffix:
            reason = f"{reason} [{' '.join(suffix)}]"

        table.add_row(
            _format_iso_timestamp(entry.get("timestamp")),
            _clip_text(str(entry.get("strategy", "-")).replace("_", " "), 18),
            _clip_text(str(entry.get("symbol", "-")), 18),
            Text(decision_text, style=decision_style),
            confidence_text,
            _clip_text(reason, 64),
        )

    return Panel(table, title="[bold cyan]AGENT DECISIONS[/bold cyan]", border_style="cyan")


def _build_activity_panel(
    terminal_state: TerminalRuntimeState,
    log_lines: list[str],
) -> Panel:
    lines = Text()

    if terminal_state.recent_events:
        lines.append("Recent cycles\n", style="bold")
        for event in list(terminal_state.recent_events)[:4]:
            lines.append(f"{event}\n")

    if log_lines:
        lines.append("\nRecent log lines\n", style="bold")
        for line in log_lines[-4:]:
            lines.append(f"{line}\n", style="dim")
    else:
        lines.append("\nNo activity captured yet\n", style="dim")

    return Panel(lines, title="[bold cyan]ACTIVITY[/bold cyan]", border_style="cyan")


def _build_layout(
    *,
    spot: float | None,
    kalshi_implied: float | None,
    poly_implied: float | None,
    kalshi_spread: float | None,
    poly_spread: float | None,
    updated_at: str,
    btc_result: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    terminal_state: TerminalRuntimeState,
    audit_entries: list[dict[str, Any]],
    decision_entries: list[dict[str, Any]],
    log_lines: list[str],
    ai_enabled: bool,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=12),
        Layout(name="middle", size=14),
        Layout(name="bottom"),
        Layout(name="footer", size=1),
    )
    layout["top"].split_row(Layout(name="prices", ratio=2), Layout(name="signals", ratio=2), Layout(name="status", ratio=1))
    layout["middle"].split_row(Layout(name="positions", ratio=1), Layout(name="trades", ratio=2))
    layout["bottom"].split_row(Layout(name="decisions", ratio=2), Layout(name="activity", ratio=1))

    layout["header"].update(Panel(Text("  TRADING BOT LIVE TERMINAL  ", style="bold white on blue"), border_style="blue"))
    layout["prices"].update(_build_price_panel(spot, kalshi_implied, poly_implied, kalshi_spread, poly_spread, updated_at))
    layout["signals"].update(_build_signals_panel(btc_result))
    layout["status"].update(_build_status_panel(summary, terminal_state))
    layout["positions"].update(_build_positions_panel(summary))
    layout["trades"].update(_build_recent_trades_panel(audit_entries, log_lines))
    layout["decisions"].update(_build_decisions_panel(decision_entries, ai_enabled=ai_enabled))
    layout["activity"].update(_build_activity_panel(terminal_state, log_lines))
    layout["footer"].update(Text(f" Refreshing every {_REFRESH_SECONDS}s - Ctrl+C to exit", style="dim"))
    return layout


async def run_terminal(
    config: AppConfig,
    logger: logging.Logger,
    service: TradingBotService,
) -> None:
    if not _RICH_AVAILABLE:
        print("ERROR: 'rich' library not installed. Run: pip install rich")
        return

    console = Console()
    removed_handlers = _detach_console_handlers(logger)
    terminal_state = TerminalRuntimeState()

    try:
        terminal_state.record_summary(service.status(), source="initial")
    except Exception as exc:
        terminal_state.record_error(f"initial status failed: {exc}")

    telegram = TradingTelegramBot(config.telegram, logger)
    telegram.set_service(service)
    notifier = TerminalNotificationDispatcher(telegram, logger, terminal_state)
    scheduler = SchedulerService(config, service, telegram, notifier, logger)
    scheduler_task = asyncio.create_task(scheduler.run_forever())

    btc_cfg = config.btc5min
    spot: float | None = None
    kalshi_implied: float | None = None
    poly_implied: float | None = None
    kalshi_spread: float | None = None
    poly_spread: float | None = None
    updated_at = "--:--:--"
    last_market_refresh: datetime | None = None

    def _refresh_market_data() -> None:
        nonlocal spot, kalshi_implied, poly_implied, kalshi_spread, poly_spread, updated_at, last_market_refresh

        if last_market_refresh is not None:
            elapsed = (datetime.now(timezone.utc) - last_market_refresh).total_seconds()
            if elapsed < _MARKET_REFRESH_SECONDS:
                return

        spot = None
        kalshi_implied = None
        poly_implied = None
        kalshi_spread = None
        poly_spread = None

        try:
            prices = service.coinbase_client.get_spot_prices([btc_cfg.btc_symbol])
            spot = prices.get(btc_cfg.btc_symbol)
        except Exception as exc:
            logger.getChild("btc_terminal").warning("Spot price fetch failed: %s", exc)

        pairs: list[tuple[float, float]] = []
        try:
            markets, _ = service.kalshi_client.get_markets(series_ticker="KXBTC", status="open", limit=100)
            now_ts = datetime.now(timezone.utc).timestamp()
            max_age = btc_cfg.max_strike_age_hours * 3600
            for market in markets:
                strike = parse_strike(market.get("title", "") or market.get("subtitle", ""))
                if strike is None:
                    continue
                volume = float(market.get("volume_fp") or market.get("volume") or 0)
                if volume < btc_cfg.min_kalshi_volume_usd:
                    continue
                end_date = market.get("end_date") or market.get("expiration_time", "")
                if end_date:
                    try:
                        if isinstance(end_date, (int, float)):
                            expires_ts = float(end_date)
                        else:
                            expires_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp()
                        if expires_ts - now_ts > max_age:
                            continue
                    except Exception:
                        pass
                yes_bid = float(market.get("yes_bid_dollars") or 0)
                yes_ask = float(market.get("yes_ask_dollars") or 0)
                last_price = float(market.get("last_price_dollars") or 0)
                if yes_bid or yes_ask:
                    yes_mid = (yes_bid + yes_ask) / 2
                elif last_price:
                    yes_mid = last_price
                else:
                    continue
                pairs.append((strike, yes_mid))
            kalshi_implied = compute_implied_median(pairs)
        except Exception as exc:
            logger.getChild("btc_terminal").warning("Kalshi fetch failed: %s", exc)

        pairs2: list[tuple[float, float]] = []
        try:
            markets2 = service.gamma_client.discover_markets(tags=["bitcoin"], min_volume=btc_cfg.min_poly_volume_usd, limit=50)
            now_ts2 = datetime.now(timezone.utc).timestamp()
            max_age2 = btc_cfg.max_strike_age_hours * 3600
            for market in markets2:
                strike = parse_strike(market.question)
                if strike is None:
                    continue
                if market.end_date:
                    try:
                        expires_ts2 = datetime.fromisoformat(market.end_date.replace("Z", "+00:00")).timestamp()
                        if expires_ts2 - now_ts2 > max_age2:
                            continue
                    except Exception:
                        pass
                yes_mid = market.yes_price
                if market.yes_token_id:
                    try:
                        live_price = service.clob_client.get_midpoint(market.yes_token_id)
                        if live_price is not None:
                            yes_mid = live_price
                    except Exception:
                        pass
                pairs2.append((strike, yes_mid))
            poly_implied = compute_implied_median(pairs2)
        except Exception as exc:
            logger.getChild("btc_terminal").warning("Polymarket fetch failed: %s", exc)

        if spot and spot > 0:
            kalshi_spread = ((kalshi_implied - spot) / spot * 100) if kalshi_implied else None
            poly_spread = ((poly_implied - spot) / spot * 100) if poly_implied else None

        updated_at = datetime.now(config.timezone).strftime("%H:%M:%S")
        last_market_refresh = datetime.now(timezone.utc)

    def _read_activity() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        return (
            service.get_recent_audit_entries(_RECENT_TRADE_LIMIT),
            service.get_recent_decisions(_RECENT_DECISION_LIMIT * 2),
            _tail_interesting_log_lines(config.log_dir / "trading_bot.log", _ACTIVITY_LIMIT),
        )

    console.print("[bold cyan]Trading Bot Live Terminal[/bold cyan] - loading...")

    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            loop = asyncio.get_running_loop()
            while True:
                try:
                    await loop.run_in_executor(None, _refresh_market_data)
                    audit_entries, decision_entries, log_lines = await loop.run_in_executor(None, _read_activity)
                    layout = _build_layout(
                        spot=spot,
                        kalshi_implied=kalshi_implied,
                        poly_implied=poly_implied,
                        kalshi_spread=kalshi_spread,
                        poly_spread=poly_spread,
                        updated_at=updated_at,
                        btc_result=terminal_state.last_btc_result,
                        summary=terminal_state.last_summary or terminal_state.last_heartbeat,
                        terminal_state=terminal_state,
                        audit_entries=audit_entries,
                        decision_entries=decision_entries,
                        log_lines=log_lines,
                        ai_enabled=config.trading_agents.enabled,
                    )
                    live.update(layout)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    terminal_state.record_error(f"refresh failed: {exc}")

                await asyncio.sleep(_REFRESH_SECONDS)
    finally:
        scheduler_task.cancel()
        await asyncio.gather(scheduler_task, return_exceptions=True)
        _restore_handlers(logger, removed_handlers)
