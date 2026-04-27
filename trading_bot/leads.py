"""Lead extraction from strategy results.

A "lead" is a symbol/market the bot identified as having a tradeable signal
but did NOT trade — e.g. position-size rounded to zero, max-positions cap
reached, AI veto. Leads are surfaced to Telegram so the operator can see
what's on the radar.
"""

from __future__ import annotations

from typing import Any

# Reasons that map to actionable per-symbol leads. Reasons like
# ``risk_guardrails``, ``kill_switch_active``, or ``neutral_fear_greed`` are
# intentionally excluded — they describe global state, not a per-symbol
# opportunity.
WATCH_REASONS = frozenset(
    {
        "position_size_zero",
        "max_positions_reached",
        "ai_rejected",
        "cooldown_active",
        "non_negative_funding_rate",
        "performance_memory_benched",
    }
)


def extract_leads(strategy: str, result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return per-symbol leads (skipped buys + stock top_targets) for one strategy run."""
    if not isinstance(result, dict):
        return []

    leads: list[dict[str, Any]] = []
    seen: set[str] = set()

    for decision in result.get("decisions") or []:
        if not isinstance(decision, dict):
            continue
        if decision.get("action") != "skip_buy":
            continue
        reason = str(decision.get("reason", ""))
        if reason not in WATCH_REASONS:
            continue
        symbol = str(decision.get("symbol", ""))
        if not symbol or symbol == "*":
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        signals = decision.get("signals") or {}
        leads.append(
            {
                "strategy": strategy,
                "symbol": symbol,
                "reason": reason,
                "signals": signals if isinstance(signals, dict) else {},
            }
        )

    # Stocks: surface high-ranked screen results that didn't get bought.
    # The strategy returns top_targets as a list of symbol strings.
    if strategy == "stocks":
        bought = {str(s) for s in (result.get("buys") or [])}
        for raw_symbol in result.get("top_targets") or []:
            symbol = str(raw_symbol)
            if not symbol or symbol in bought or symbol in seen:
                continue
            seen.add(symbol)
            leads.append(
                {
                    "strategy": strategy,
                    "symbol": symbol,
                    "reason": "top_target_not_bought",
                    "signals": {},
                }
            )

    return leads
