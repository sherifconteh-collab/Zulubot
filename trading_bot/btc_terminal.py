"""BTC 5-minute spread dashboard terminal.

Launched via: python -m trading_bot btc-terminal

Displays a live, auto-refreshing view of:
  - Coinbase spot BTC price
  - Kalshi KXBTC implied median price and spread
  - Polymarket bitcoin implied median price and spread
  - Open BTC / prediction market positions and unrealised P&L
  - Recent strategy signals

Requires: rich>=13.0.0
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from trading_bot.clients import CoinbaseClient
from trading_bot.clients_kalshi import KalshiClient
from trading_bot.clients_polymarket import PolymarketClobClient, PolymarketGammaClient
from trading_bot.config import AppConfig
from trading_bot.state import StateStore
from trading_bot.strategies.btc_5min import (
    parse_strike,
    compute_implied_median,
)

_REFRESH_SECONDS = 30


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
        return Panel(Text("No data yet — waiting for first scan…", style="dim"), title="[bold cyan]SIGNALS[/bold cyan]", border_style="cyan")

    actions = result.get("actions", [])
    threshold = result.get("threshold_pct", 2.0)
    lines = Text()

    if not actions:
        ks = result.get("kalshi_spread_pct")
        ps = result.get("poly_spread_pct")
        lines.append(f"HOLD — spreads within ±{threshold:.1f}% threshold\n", style="yellow")
        if ks is not None:
            lines.append(f"  Kalshi spread:    {ks:+.3f}%\n", style=_spread_color(ks))
        if ps is not None:
            lines.append(f"  Polymarket spread:{ps:+.3f}%\n", style=_spread_color(ps))
    else:
        for act in actions:
            venue = act.get("venue", "").upper()
            action = act.get("action", "").upper()
            sp = act.get("spread_pct", 0.0)
            syms = act.get("symbols", [])
            color = "green" if "yes" in action.lower() or "spot" in action.lower() else "red"
            lines.append(f"[{venue}] ", style="bold")
            lines.append(f"{action}  spread={sp:+.3f}%\n", style=color)
            for s in syms:
                lines.append(f"  → {s}\n", style="dim")

    kalshi_n = result.get("kalshi_markets_used", 0)
    poly_n = result.get("poly_markets_used", 0)
    lines.append(f"\nMarket depth: Kalshi={kalshi_n}  Poly={poly_n}", style="dim")
    return Panel(lines, title="[bold cyan]SIGNALS[/bold cyan]", border_style="cyan")


def _build_positions_panel(state_data: dict[str, Any] | None) -> Panel:
    if not state_data:
        return Panel(Text("No portfolio state", style="dim"), title="[bold cyan]POSITIONS[/bold cyan]", border_style="cyan")

    positions = state_data.get("positions", {})
    btc_positions = {
        sym: pos for sym, pos in positions.items()
        if "BTC" in sym.upper() or "KAL:" in sym or "PM:" in sym
    }

    if not btc_positions:
        return Panel(Text("No open BTC / prediction positions", style="dim"), title="[bold cyan]POSITIONS[/bold cyan]", border_style="cyan")

    table = Table.grid(padding=(0, 2))
    table.add_column("Symbol", style="white")
    table.add_column("Qty", justify="right", style="white")
    table.add_column("Entry", justify="right", style="white")
    table.add_column("Current", justify="right", style="white")
    table.add_column("P&L", justify="right")

    for sym, pos in btc_positions.items():
        entry = pos.get("entry_price", 0)
        current = pos.get("current_price", entry)
        qty = pos.get("quantity", 0)
        pnl = (current - entry) * qty
        pnl_style = "green" if pnl >= 0 else "red"
        pnl_str = f"${pnl:+,.2f}"
        table.add_row(
            sym,
            f"{qty:.6f}",
            f"${entry:,.2f}",
            f"${current:,.2f}",
            Text(pnl_str, style=pnl_style),
        )

    equity = state_data.get("equity", 0)
    cash = state_data.get("cash", 0)
    footer = Text(f"\nEquity: ${equity:,.2f}   Cash: ${cash:,.2f}", style="dim")
    content = table
    return Panel(Columns([content, footer]), title="[bold cyan]POSITIONS[/bold cyan]", border_style="cyan")


def _build_layout(
    spot: float | None,
    kalshi_implied: float | None,
    poly_implied: float | None,
    kalshi_spread: float | None,
    poly_spread: float | None,
    updated_at: str,
    last_result: dict[str, Any] | None,
    state_data: dict[str, Any] | None,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=1),
    )
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(name="prices"),
        Layout(name="signals"),
    )

    title_text = Text("  BTC 5-MIN SPREAD TERMINAL  ", style="bold white on blue")
    layout["header"].update(Panel(title_text, border_style="blue"))
    layout["prices"].update(_build_price_panel(
        spot, kalshi_implied, poly_implied, kalshi_spread, poly_spread, updated_at
    ))
    layout["signals"].update(_build_signals_panel(last_result))
    layout["right"].update(_build_positions_panel(state_data))
    layout["footer"].update(
        Text(f" Refreshing every {_REFRESH_SECONDS}s — Ctrl+C to exit", style="dim")
    )
    return layout


async def run_terminal(config: AppConfig) -> None:
    """Main entry point for the BTC terminal dashboard."""
    if not _RICH_AVAILABLE:
        print("ERROR: 'rich' library not installed. Run: pip install rich")
        return

    console = Console()
    logger = logging.getLogger("btc_terminal")

    # Instantiate only the clients we need (no full service)
    coinbase_client = CoinbaseClient(
        api_key=config.coinbase.api_key,
        api_secret=config.coinbase.api_secret,
        timeout_seconds=config.request_timeout_seconds,
        logger=logger,
    )
    kalshi_client = KalshiClient(
        key_id=config.predictions.kalshi_key_id,
        private_key_pem=config.predictions.kalshi_private_key,
        timeout_seconds=config.request_timeout_seconds,
        logger=logger,
    )
    gamma_client = PolymarketGammaClient(
        timeout_seconds=config.request_timeout_seconds,
        logger=logger,
    )
    clob_client = PolymarketClobClient(
        timeout_seconds=config.request_timeout_seconds,
        logger=logger,
    )
    state_store = StateStore(config.state_path, config.initial_cash, config.timezone)

    btc_cfg = config.btc5min

    spot: float | None = None
    kalshi_implied: float | None = None
    poly_implied: float | None = None
    kalshi_spread: float | None = None
    poly_spread: float | None = None
    last_result: dict[str, Any] | None = None
    updated_at = "—"

    def _refresh_data() -> None:
        nonlocal spot, kalshi_implied, poly_implied, kalshi_spread, poly_spread, last_result, updated_at

        # Spot price
        try:
            prices = coinbase_client.get_spot_prices([btc_cfg.btc_symbol])
            spot = prices.get(btc_cfg.btc_symbol)
        except Exception as exc:
            logger.warning("Spot price fetch failed: %s", exc)

        # Kalshi implied
        pairs: list[tuple[float, float]] = []
        try:
            markets, _ = kalshi_client.get_markets(series_ticker="KXBTC", status="open", limit=100)
            now_ts = datetime.now(timezone.utc).timestamp()
            max_age = btc_cfg.max_strike_age_hours * 3600
            for m in markets:
                strike = parse_strike(m.get("title", "") or m.get("subtitle", ""))
                if strike is None:
                    continue
                vol = float(m.get("volume_fp") or m.get("volume") or 0)
                if vol < btc_cfg.min_kalshi_volume_usd:
                    continue
                end_date = m.get("end_date") or m.get("expiration_time", "")
                if end_date:
                    try:
                        if isinstance(end_date, (int, float)):
                            exp_ts = float(end_date)
                        else:
                            exp_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp()
                        if exp_ts - now_ts > max_age:
                            continue
                    except Exception:
                        pass
                yes_bid = float(m.get("yes_bid_dollars") or 0)
                yes_ask = float(m.get("yes_ask_dollars") or 0)
                last_price = float(m.get("last_price_dollars") or 0)
                if yes_bid or yes_ask:
                    yes_mid = (yes_bid + yes_ask) / 2
                elif last_price:
                    yes_mid = last_price
                else:
                    # No active quotes and no trade history — skip to avoid noise
                    continue
                pairs.append((strike, yes_mid))
            kalshi_implied = compute_implied_median(pairs)
        except Exception as exc:
            logger.warning("Kalshi fetch failed: %s", exc)

        # Polymarket implied
        pairs2: list[tuple[float, float]] = []
        try:
            mkts = gamma_client.discover_markets(
                tags=["bitcoin"],
                min_volume=btc_cfg.min_poly_volume_usd,
                limit=50,
            )
            now_ts2 = datetime.now(timezone.utc).timestamp()
            max_age2 = btc_cfg.max_strike_age_hours * 3600
            for mkt in mkts:
                strike = parse_strike(mkt.question)
                if strike is None:
                    continue
                if mkt.end_date:
                    try:
                        exp2 = datetime.fromisoformat(mkt.end_date.replace("Z", "+00:00")).timestamp()
                        if exp2 - now_ts2 > max_age2:
                            continue
                    except Exception:
                        pass
                yes_mid = mkt.yes_price
                if mkt.yes_token_id:
                    try:
                        live = clob_client.get_midpoint(mkt.yes_token_id)
                        if live is not None:
                            yes_mid = live
                    except Exception:
                        pass
                pairs2.append((strike, yes_mid))
            poly_implied = compute_implied_median(pairs2)
        except Exception as exc:
            logger.warning("Polymarket fetch failed: %s", exc)

        # Compute spreads
        if spot and spot > 0:
            kalshi_spread = ((kalshi_implied - spot) / spot) if kalshi_implied else None
            poly_spread = ((poly_implied - spot) / spot) if poly_implied else None

        # Summarise as a "result" dict for the signals panel
        last_result = {
            "spot_price": spot,
            "kalshi_implied": kalshi_implied,
            "poly_implied": poly_implied,
            "kalshi_spread_pct": round(kalshi_spread * 100, 3) if kalshi_spread is not None else None,
            "poly_spread_pct": round(poly_spread * 100, 3) if poly_spread is not None else None,
            "kalshi_markets_used": len(pairs),
            "poly_markets_used": len(pairs2),
            "actions": [],
            "threshold_pct": btc_cfg.buy_spread_threshold * 100,
        }

        updated_at = datetime.now(config.timezone).strftime("%H:%M:%S")

    def _get_state_data() -> dict[str, Any] | None:
        try:
            state = state_store.load()
            positions_dict = {
                sym: {
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "quantity": pos.quantity,
                    "strategy": pos.strategy,
                }
                for sym, pos in state.positions.items()
            }
            equity = sum(
                pos.current_price * pos.quantity for pos in state.positions.values()
            ) + state.cash
            return {
                "positions": positions_dict,
                "cash": state.cash,
                "equity": equity,
            }
        except Exception:
            return None

    console.print("[bold cyan]BTC 5-Min Spread Terminal[/bold cyan] — loading…")

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            try:
                await asyncio.get_event_loop().run_in_executor(None, _refresh_data)
                state_data = await asyncio.get_event_loop().run_in_executor(None, _get_state_data)
                layout = _build_layout(
                    spot, kalshi_implied, poly_implied, kalshi_spread, poly_spread,
                    updated_at, last_result, state_data,
                )
                live.update(layout)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                console.print(f"[red]Refresh error: {exc}[/red]")

            await asyncio.sleep(_REFRESH_SECONDS)
