"""Turn a free-text Telegram message into one of the bot's existing commands.

The model only *chooses* an action and its arguments from a fixed list (a forced tool call);
it never runs anything. The Telegram bot then runs the same code as the slash command, so every
guard, preview and ``/confirm`` step still applies. Buying is never an available action.
"""

from __future__ import annotations

import logging
from typing import Any

from trading_bot.integrations.anthropic_errors import explain_api_failure

ACTIONS = (
    "status", "positions", "risk", "history", "sports", "leads", "config", "help",
    "games", "props", "bet", "parlay", "stack", "confirm", "cancel", "control", "reply",
)
_MAX_TEXT = 200

_SYSTEM = """You route messages sent to a personal sports-betting bot on Telegram to one of its commands.
The user texts casually. Pick exactly one action by calling the route tool. You never place bets yourself:
"bet", "parlay" and "stack" only PREVIEW a bet; the user then replies yes/confirm to place it.

Actions:
- status: portfolio, balance, how am I doing, bankroll, profit/loss.
- positions: open bets and positions. history: recent trades/results. risk: risk metrics. leads: the watchlist.
- sports: the model's own picks and analysis of today's games. config: current settings. help: what can you do.
- games: list games available to bet on (optional filter: a league like nfl/nba/mlb or a team/city).
- props: search football player-stat markets (query: player, team or stat).
- bet: preview ONE bet on a team to win. team = the team the user names (e.g. "cleveland browns"), amount = dollars if stated.
- parlay: preview a parlay. legs = each leg as short plain text ("hurts 200 passing yards", "barkley 80 rushing yards", "browns"), amount = dollars, max_cents = price ceiling in cents if stated.
- stack: auto-build a same-game player-stat parlay. game = the two teams ("eagles jaguars"), leg_count, leg_percent (chance per leg as a percent, e.g. 85), amount, max_cents.
- confirm: only when the user clearly agrees to place the previewed bet, or tells you to bet an amount on "this"/"it"/"that parlay" while one is waiting (then set amount to the dollars they say, e.g. "bet $10 on it" -> amount 10). cancel: drop the previewed bet.
- control: the user wants to pause, resume or kill/stop the bot or liquidate (never do it; the user must use the slash command).
- reply: anything else (questions, chit-chat, unclear requests). Put a short helpful answer in reply_text, and suggest what they could say.
Convert spoken amounts to numbers ("five bucks" -> 5). If an amount isn't stated leave it out. Keep every text field short."""

_TOOL = {
    "name": "route",
    "description": "Choose the bot command that matches the user's message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "team": {"type": "string"},
            "legs": {"type": "array", "items": {"type": "string"}},
            "game": {"type": "string"},
            "query": {"type": "string"},
            "filter": {"type": "string"},
            "amount": {"type": "number"},
            "max_cents": {"type": "number"},
            "leg_count": {"type": "integer"},
            "leg_percent": {"type": "number"},
            "reply_text": {"type": "string"},
        },
        "required": ["action"],
    },
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())[:_MAX_TEXT]


def _num(value: Any, low: float, high: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if low <= number <= high else None


def validate_route(raw: dict[str, Any]) -> dict[str, Any]:
    """Whitelist the action and clean every argument; anything odd becomes a plain reply."""
    action = raw.get("action")
    if action not in ACTIONS:
        return {"action": "reply", "reply_text": "I didn't understand that. Try /help for what I can do."}
    legs = [_text(l) for l in raw.get("legs", []) if _text(l)][:12] if isinstance(raw.get("legs"), list) else []
    leg_count = _num(raw.get("leg_count"), 2, 15)
    return {
        "action": action,
        "team": _text(raw.get("team")),
        "legs": legs,
        "game": _text(raw.get("game")),
        "query": _text(raw.get("query")),
        "filter": _text(raw.get("filter")),
        "amount": _num(raw.get("amount"), 0.01, 100_000),
        "max_cents": _num(raw.get("max_cents"), 1, 99),
        "leg_count": int(leg_count) if leg_count is not None else None,
        "leg_percent": _num(raw.get("leg_percent"), 5, 97),
        "reply_text": _text(raw.get("reply_text")) or "",
    }


def route_message(
    text: str,
    *,
    pending: str | None,
    api_key: str,
    model: str,
    timeout_seconds: float = 30.0,
    logger: logging.Logger | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Ask Claude which command ``text`` means. Raises RuntimeError if it can't be routed."""
    log = logger or logging.getLogger("nl_router")
    if client is None:
        if not api_key:
            raise RuntimeError("Understanding plain text needs ANTHROPIC_API_KEY. The / commands still work (/help).")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
            raise RuntimeError("The anthropic package isn't installed.") from exc
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
    context = f"A bet is waiting for confirmation: {pending}" if pending else "No bet is waiting for confirmation."
    try:
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "route"},
            messages=[{"role": "user", "content": f"{context}\n\nMessage: {text[:600]}"}],
        )
        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and isinstance(getattr(block, "input", None), dict):
                return validate_route(block.input)
        raise ValueError("no tool call in the reply")
    except Exception as exc:
        log.warning("Routing failed: %s", exc)
        raise RuntimeError(explain_api_failure(exc) or "I couldn't work out what you meant. Try /help.") from exc
