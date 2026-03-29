"""Multi-analyst persona system inspired by virattt/ai-hedge-fund.

Provides multiple AI analyst perspectives (value, contrarian, technical,
sentiment, growth) that independently evaluate signals and aggregate into
a single verdict.  This replaces the single-voice AI evaluation with a
"team of analysts" approach where each persona brings a different
investing philosophy.

Key concepts from ai-hedge-fund:
  1. Analyst Registry — clean config mapping of personas with metadata
  2. Independent Signals — each analyst produces {signal, confidence, reasoning}
  3. Signal Aggregation — weighted combination into a final verdict
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ── Signal model ───────────────────────────────────────────────────

@dataclass(frozen=True)
class AnalystSignal:
    """Output from a single analyst persona evaluation."""
    persona_key: str
    signal: str        # "bullish", "bearish", "neutral"
    confidence: float  # 0.0-1.0
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_key": self.persona_key,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


# ── Persona definition ────────────────────────────────────────────

@dataclass(frozen=True)
class AnalystPersona:
    """A named AI analyst with a specific investing philosophy."""
    key: str
    display_name: str
    description: str
    investing_style: str
    system_prompt: str
    order: int


# ── Analyst Registry ──────────────────────────────────────────────

ANALYST_REGISTRY: dict[str, AnalystPersona] = {
    "value_investor": AnalystPersona(
        key="value_investor",
        display_name="Value Investor",
        description="Benjamin Graham-style deep value analyst",
        investing_style=(
            "Insists on a margin of safety. Focuses on low P/E, low P/B, "
            "conservative debt levels, and strong free cash flow. Only buys "
            "when price is well below intrinsic value."
        ),
        system_prompt=(
            "You are a Benjamin Graham-style value investing analyst. "
            "Evaluate assets through the lens of margin of safety, intrinsic "
            "value, conservative balance sheets, and stable earnings. "
            "Be skeptical of hype and focus on proven fundamentals."
        ),
        order=0,
    ),
    "growth_investor": AnalystPersona(
        key="growth_investor",
        display_name="Growth Investor",
        description="Innovation and disruption-focused analyst",
        investing_style=(
            "Focuses on revenue growth, market disruption potential, and "
            "innovative technology. Willing to pay premium valuations for "
            "companies with exponential growth trajectories."
        ),
        system_prompt=(
            "You are a growth-focused investing analyst inspired by "
            "Cathie Wood's ARK Invest philosophy. Evaluate assets for "
            "their innovation potential, revenue growth trajectory, and "
            "market disruption capability. Consider long-term technology "
            "trends and exponential growth potential."
        ),
        order=1,
    ),
    "contrarian": AnalystPersona(
        key="contrarian",
        display_name="Contrarian Analyst",
        description="Michael Burry-style contrarian thinker",
        investing_style=(
            "Looks for market mispricing and overreaction. Questions consensus "
            "views, seeks hidden risks in popular trades, and hunts for deep "
            "value in overlooked or hated assets."
        ),
        system_prompt=(
            "You are a contrarian analyst. Your job is to challenge the "
            "consensus view. If the market is bullish, look for hidden risks. "
            "If bearish, look for overlooked value. Question every assumption "
            "and focus on what the crowd is missing. Be specific about the "
            "risks or opportunities others overlook."
        ),
        order=2,
    ),
    "technical_analyst": AnalystPersona(
        key="technical_analyst",
        display_name="Technical Analyst",
        description="Chart pattern and momentum specialist",
        investing_style=(
            "Analyzes price trends, momentum, volume patterns, and market "
            "structure. Uses support/resistance levels, moving averages, "
            "and momentum indicators to time entries and exits."
        ),
        system_prompt=(
            "You are a technical analysis specialist. Evaluate assets "
            "based on price action signals, momentum indicators, trend "
            "strength, and market microstructure. Focus on what the price "
            "data and volume tell you about near-term direction, ignoring "
            "fundamental narratives."
        ),
        order=3,
    ),
    "sentiment_analyst": AnalystPersona(
        key="sentiment_analyst",
        display_name="Sentiment Analyst",
        description="Market psychology and crowd behavior analyst",
        investing_style=(
            "Gauges crowd psychology, fear/greed cycles, insider behavior, "
            "and positioning extremes. Uses sentiment indicators to identify "
            "when markets are overly fearful or greedy."
        ),
        system_prompt=(
            "You are a market sentiment analyst. Evaluate assets based on "
            "crowd psychology, fear and greed indicators, insider activity, "
            "and market positioning. Identify when sentiment is at extremes "
            "that historically precede reversals."
        ),
        order=4,
    ),
    "risk_analyst": AnalystPersona(
        key="risk_analyst",
        display_name="Risk Analyst",
        description="Tail risk and portfolio protection specialist",
        investing_style=(
            "Focuses on downside protection, tail risk, and position sizing. "
            "Inspired by Nassim Taleb's antifragility concepts. Prioritizes "
            "asymmetric payoffs with limited downside."
        ),
        system_prompt=(
            "You are a risk-focused analyst inspired by Nassim Taleb. "
            "Evaluate the downside risk, tail risk exposure, and "
            "risk-adjusted return potential. Flag fragile positions and "
            "prefer asymmetric payoffs where the downside is capped but "
            "the upside is open-ended."
        ),
        order=5,
    ),
}


# ── Registry helpers ──────────────────────────────────────────────

def get_available_analysts() -> list[dict[str, Any]]:
    """Return metadata for all registered analyst personas."""
    return [
        {
            "key": p.key,
            "display_name": p.display_name,
            "description": p.description,
            "investing_style": p.investing_style,
            "order": p.order,
        }
        for p in sorted(ANALYST_REGISTRY.values(), key=lambda x: x.order)
    ]


def get_analyst(key: str) -> AnalystPersona | None:
    """Look up a single analyst persona by key."""
    return ANALYST_REGISTRY.get(key)


def get_default_analysts() -> tuple[str, ...]:
    """Return the default set of analyst keys for multi-analyst evaluation."""
    return ("value_investor", "contrarian", "sentiment_analyst")


# ── Signal aggregation ────────────────────────────────────────────

_SIGNAL_SCORES = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


def aggregate_analyst_signals(
    signals: list[AnalystSignal],
) -> dict[str, Any]:
    """Aggregate multiple analyst signals into a combined verdict.

    Uses confidence-weighted voting (identical to ai-hedge-fund's
    ``weighted_signal_combination`` pattern):
      1. Each signal is converted to a numeric score: bullish=+1, neutral=0, bearish=-1
      2. Weighted by confidence
      3. Final score determines action and combined confidence

    Returns a dict with keys: action, confidence, reasoning, analyst_signals.
    """
    if not signals:
        return {
            "action": "hold",
            "confidence": 0.0,
            "reasoning": "No analyst signals available.",
            "analyst_signals": [],
        }

    weighted_sum = 0.0
    total_weight = 0.0

    for sig in signals:
        score = _SIGNAL_SCORES.get(sig.signal, 0.0)
        weight = max(0.0, min(1.0, sig.confidence))
        weighted_sum += score * weight
        total_weight += weight

    if total_weight > 0:
        final_score = weighted_sum / total_weight
    else:
        final_score = 0.0

    # Convert score to action (matching Zulubot's buy/sell/hold convention).
    # The ±0.2 threshold prevents weak/split signals from triggering trades;
    # a score of 0.2 requires meaningful directional agreement among analysts.
    if final_score > 0.2:
        action = "buy"
    elif final_score < -0.2:
        action = "sell"
    else:
        action = "hold"

    combined_confidence = min(1.0, abs(final_score))

    # Build readable reasoning summary
    bullish_count = sum(1 for s in signals if s.signal == "bullish")
    bearish_count = sum(1 for s in signals if s.signal == "bearish")
    neutral_count = sum(1 for s in signals if s.signal == "neutral")

    summaries = [
        f"{s.persona_key}: {s.signal} ({s.confidence:.0%})"
        for s in signals
    ]
    reasoning = (
        f"Multi-analyst consensus ({bullish_count} bullish, "
        f"{bearish_count} bearish, {neutral_count} neutral): "
        + "; ".join(summaries)
    )

    return {
        "action": action,
        "confidence": round(combined_confidence, 4),
        "reasoning": reasoning,
        "analyst_signals": [s.to_dict() for s in signals],
    }
