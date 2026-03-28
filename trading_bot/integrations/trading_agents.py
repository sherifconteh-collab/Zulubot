"""TradingAgents wrapper service with cost-optimized LLM analysis.

Wraps the TradingAgents package from TauricResearch/TradingAgents with:
  1. Two-tier model selection (Haiku pre-screen → Sonnet deep analysis)
  2. In-memory cache with configurable TTL (default 6 hours)
  3. Per-run ($0.10) and per-month ($10) hard budget caps
  4. Three confirmation modes: log_only, boost, require

Evaluation methods and their cost profiles:
  evaluate_stock()         — Haiku→Sonnet, 6h cache, ~$0.03/call
  evaluate_crypto()        — Haiku→Sonnet, 6h cache, ~$0.03/call
  evaluate_kalshi_market() — Haiku→Sonnet, 6h cache, ~$0.03/call  (prediction markets)
  evaluate_arb_signal()    — Haiku-only,   10m cache, ~$0.002/call (latency arb)
  detect_market_regime()   — Haiku-only,   1h cache,  ~$0.002/call (regime filter)

Falls back gracefully if the tradingagents package or anthropic SDK is not installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from trading_bot.config import TradingAgentsConfig


@dataclass(frozen=True)
class AgentVerdict:
    """Result of an AI analysis."""
    action: str                         # "buy", "sell", "hold", "skip"
    confidence: float                   # 0.0-1.0
    reasoning: str
    estimated_probability: float | None = None  # For Polymarket: fair probability
    cost_usd: float = 0.0
    model_used: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "estimated_probability": self.estimated_probability,
            "cost_usd": self.cost_usd,
            "model_used": self.model_used,
            "cached": self.cached,
        }


@dataclass
class _CacheEntry:
    verdict: AgentVerdict
    expires_at: float  # Unix timestamp


class TradingAgentsService:
    """Cost-optimized LLM analysis for all 3 trading modules.

    Uses the TradingAgents framework when available, falls back to
    direct Anthropic API calls for simpler analysis.

    Cost control layers:
      1. Only called when quant signals fire (not every cycle)
      2. Haiku pre-screen (~$0.001) filters obvious no-ops
      3. Sonnet deep analysis (~$0.10-0.50) only if Haiku says "investigate"
      4. 4-hour cache prevents re-analyzing the same asset
      5. Hard caps: $0.50/run, $20/month
    """

    # Approximate costs per 1K tokens (input + output averaged)
    _MODEL_COSTS: dict[str, float] = {
        "claude-haiku-4-5-20251001": 0.001,
        "claude-sonnet-4-6": 0.015,
    }

    def __init__(self, config: TradingAgentsConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger.getChild("trading_agents")
        self._cache: dict[str, _CacheEntry] = {}
        self._month_cost: float = 0.0
        self._month_key: str = ""  # "2026-03" format
        self._anthropic_client: Any = None
        self._ta_available = False

        if not config.enabled:
            return

        # Validate API key format before attempting connection
        if not config.api_key:
            self.logger.warning("ANTHROPIC_API_KEY not set. AI analysis disabled.")
            return
        if not config.api_key.startswith("sk-ant-"):
            self.logger.warning(
                "ANTHROPIC_API_KEY looks invalid (expected 'sk-ant-...'). "
                "AI analysis may fail. Key prefix: %s***",
                config.api_key[:8],
            )

        # Try to import Anthropic SDK
        try:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=config.api_key)
            self.logger.info(
                "Anthropic SDK loaded. Models: quick=%s deep=%s budget=$%.2f/mo",
                config.quick_model, config.deep_model, config.max_cost_per_month,
            )
        except ImportError:
            self.logger.warning("anthropic package not installed. AI analysis disabled.")
            return
        except Exception as exc:
            self.logger.error("Failed to initialize Anthropic client: %s", exc)
            return

        # Try to import TradingAgents framework (optional enhancement)
        try:
            import tradingagents  # noqa: F401
            self._ta_available = True
            self.logger.info("TradingAgents framework available for enhanced multi-agent analysis.")
        except ImportError:
            self.logger.info(
                "TradingAgents package not installed. Using direct Anthropic API. "
                "Install with: pip install -r requirements-tradingagents.txt"
            )

    @property
    def is_available(self) -> bool:
        return self.config.enabled and self._anthropic_client is not None

    # ── Public evaluation methods ──────────────────────────────────

    def evaluate_stock(
        self,
        symbol: str,
        fundamentals: dict[str, Any],
        vix_value: float,
    ) -> AgentVerdict | None:
        """Evaluate a stock buy/sell signal with AI.

        Called only when quant signals (value screen + VIX threshold) fire.
        """
        if not self.is_available:
            return None

        cache_key = f"stock:{symbol}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        prompt = (
            f"You are a value investing analyst. Evaluate {symbol} for a potential trade.\n\n"
            f"Fundamentals:\n"
            f"  P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}\n"
            f"  P/B Ratio: {fundamentals.get('pb_ratio', 'N/A')}\n"
            f"  D/E Ratio: {fundamentals.get('debt_to_equity', 'N/A')}\n"
            f"  FCF Signal: {fundamentals.get('fcf_positive_signal', 'N/A')}\n"
            f"  P/FCF Ratio: {fundamentals.get('pfcf_ratio', 'N/A')}\n"
            f"  VIX: {vix_value}\n\n"
            f"Respond with JSON: {{\"action\": \"buy\"|\"sell\"|\"hold\"|\"skip\", "
            f"\"confidence\": 0.0-1.0, \"reasoning\": \"one sentence\"}}"
        )

        return self._two_tier_analysis(cache_key, prompt)

    def evaluate_crypto(
        self,
        symbol: str,
        fear_greed_value: int,
        funding_rate: float,
        price: float,
    ) -> AgentVerdict | None:
        """Evaluate a crypto buy/sell signal with AI.

        Called only when F&G + funding rate signals align.
        """
        if not self.is_available:
            return None

        cache_key = f"crypto:{symbol}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        prompt = (
            f"You are a crypto sentiment analyst. Evaluate {symbol} for a potential trade.\n\n"
            f"Signals:\n"
            f"  Fear & Greed Index: {fear_greed_value}\n"
            f"  Binance Funding Rate: {funding_rate:.6f}\n"
            f"  Current Price: ${price:,.2f}\n\n"
            f"Context: Fear < 25 with negative funding = potential buy signal. "
            f"Fear > 75 = potential sell signal.\n\n"
            f"Respond with JSON: {{\"action\": \"buy\"|\"sell\"|\"hold\"|\"skip\", "
            f"\"confidence\": 0.0-1.0, \"reasoning\": \"one sentence\"}}"
        )

        return self._two_tier_analysis(cache_key, prompt)

    def evaluate_polymarket(
        self,
        market_question: str,
        market_description: str,
        yes_price: float,
        tags: list[str],
        fear_greed_value: int | None = None,
    ) -> AgentVerdict | None:
        """Evaluate a prediction market with AI probability estimation.

        For Polymarket, AI IS the edge — it estimates fair probability
        which is compared to the market price. Bypasses Haiku pre-screen
        but still respects cache + budget.
        """
        if not self.is_available:
            return None

        cache_key = f"polymarket:{hashlib.md5(market_question.encode()).hexdigest()[:12]}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        sentiment_line = ""
        if fear_greed_value is not None:
            sentiment_line = f"  Crypto Fear & Greed: {fear_greed_value}\n"

        prompt = (
            f"You are a prediction market analyst. Estimate the fair probability "
            f"for this market.\n\n"
            f"Question: {market_question}\n"
            f"Description: {market_description[:300]}\n"
            f"Current YES price: {yes_price:.4f} (market implied probability)\n"
            f"Tags: {', '.join(tags)}\n"
            f"{sentiment_line}\n"
            f"Respond with JSON: {{\"action\": \"buy\"|\"sell\"|\"hold\"|\"skip\", "
            f"\"confidence\": 0.0-1.0, \"reasoning\": \"one sentence\", "
            f"\"estimated_probability\": 0.0-1.0}}\n\n"
            f"The estimated_probability is your assessment of the TRUE probability, "
            f"independent of the current market price."
        )

        # Polymarket bypasses Haiku pre-screen — go straight to deep analysis
        if not self._check_budget(self._estimate_cost(self.config.deep_model)):
            self.logger.warning("Monthly budget exceeded. Skipping Polymarket AI analysis.")
            return None

        verdict = self._call_model(self.config.deep_model, prompt, expect_probability=True)
        if verdict is not None:
            self._set_cache(cache_key, verdict)
        return verdict

    def evaluate_kalshi_market(
        self,
        ticker: str,
        title: str,
        mid_price: float,
        category: str,
        fear_greed_value: int | None = None,
    ) -> AgentVerdict | None:
        """Estimate fair probability for a Kalshi binary contract.

        Identical pipeline to evaluate_polymarket() but tailored for Kalshi
        market metadata. Uses two-tier analysis: Haiku pre-screen then Sonnet.
        Results cached for config.cache_ttl_hours.
        """
        if not self.is_available:
            return None

        cache_key = f"kalshi:{ticker}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        sentiment_line = ""
        if fear_greed_value is not None:
            sentiment_line = f"  Crypto Fear & Greed Index: {fear_greed_value} (0=extreme fear, 100=extreme greed)\n"

        prompt = (
            f"You are a prediction market analyst. Estimate the fair probability for this Kalshi contract.\n\n"
            f"Contract: {title}\n"
            f"Ticker: {ticker}\n"
            f"Category: {category}\n"
            f"Current YES mid-price: {mid_price:.4f} (market-implied probability)\n"
            f"{sentiment_line}\n"
            f"Assess whether the market price accurately reflects the true probability. "
            f"Consider current news, macro conditions, and base rates.\n\n"
            f"Respond with JSON only: {{\"action\": \"buy\"|\"sell\"|\"hold\"|\"skip\", "
            f"\"confidence\": 0.0-1.0, \"reasoning\": \"one sentence\", "
            f"\"estimated_probability\": 0.0-1.0}}\n\n"
            f"estimated_probability is your TRUE probability assessment, independent of market price."
        )

        verdict = self._two_tier_analysis(cache_key, prompt, expect_probability=True)
        return verdict

    def evaluate_arb_signal(
        self,
        asset: str,
        momentum_pct: float,
        contract_title: str,
        contract_price: float,
        edge: float,
    ) -> AgentVerdict | None:
        """Quick Haiku-only validation of a latency arb opportunity.

        Used to filter low-quality momentum signals before executing.
        Haiku-only (never escalates to Sonnet) — keeps cost ~$0.002/call.
        Cache TTL is 10 minutes (short, momentum is time-sensitive).
        """
        if not self.is_available:
            return None

        direction = "UP" if momentum_pct > 0 else "DOWN"
        cache_key = f"arb:{asset}:{direction}"
        cached = self._check_cache_with_ttl(cache_key, ttl_seconds=600)  # 10-min TTL
        if cached is not None:
            return cached

        haiku_cost = self._estimate_cost(self.config.quick_model)
        if not self._check_budget(haiku_cost):
            return None

        prompt = (
            f"Quick validation: Is this a genuine arbitrage signal or likely noise?\n\n"
            f"Asset: {asset}\n"
            f"Recent move: {momentum_pct:+.3f}% in 30 seconds\n"
            f"Kalshi contract: \"{contract_title}\"\n"
            f"Contract current price: {contract_price:.3f}\n"
            f"Calculated edge: {edge:.3f} ({edge*100:.1f}%)\n\n"
            f"Is this momentum move significant enough to represent a real pricing lag? "
            f"Reply JSON only: {{\"action\": \"buy\"|\"skip\", "
            f"\"confidence\": 0.0-1.0, \"reasoning\": \"one sentence\"}}"
        )

        verdict = self._call_model(self.config.quick_model, prompt)
        if verdict is not None:
            self._set_cache_with_ttl(cache_key, verdict, ttl_seconds=600)
        return verdict

    def detect_market_regime(
        self,
        btc_price: float,
        eth_price: float,
        fear_greed_value: int | None,
        btc_momentum_24h_pct: float | None = None,
    ) -> AgentVerdict | None:
        """Classify the current market regime for risk adjustment.

        Returns action = "bullish" | "bearish" | "sideways" | "volatile".
        Used by latency arb to tighten/loosen momentum thresholds.
        Haiku-only. Cache TTL 1 hour.

        Callers should treat action field as regime label (not a trade signal).
        """
        if not self.is_available:
            return None

        cache_key = "market_regime"
        cached = self._check_cache_with_ttl(cache_key, ttl_seconds=3600)  # 1-hour TTL
        if cached is not None:
            return cached

        haiku_cost = self._estimate_cost(self.config.quick_model)
        if not self._check_budget(haiku_cost):
            return None

        fg_line = f"  Fear & Greed Index: {fear_greed_value}\n" if fear_greed_value is not None else ""
        mom_line = f"  BTC 24h change: {btc_momentum_24h_pct:+.2f}%\n" if btc_momentum_24h_pct is not None else ""

        prompt = (
            f"Classify the current crypto market regime in one word.\n\n"
            f"Market data:\n"
            f"  BTC price: ${btc_price:,.0f}\n"
            f"  ETH price: ${eth_price:,.0f}\n"
            f"{fg_line}"
            f"{mom_line}\n"
            f"Choose exactly one regime: bullish, bearish, sideways, or volatile.\n"
            f"Reply JSON only: {{\"action\": \"bullish\"|\"bearish\"|\"sideways\"|\"volatile\", "
            f"\"confidence\": 0.0-1.0, \"reasoning\": \"one sentence\"}}"
        )

        verdict = self._call_model(self.config.quick_model, prompt)
        if verdict is not None:
            self._set_cache_with_ttl(cache_key, verdict, ttl_seconds=3600)
        return verdict

    def get_monthly_cost(self) -> float:
        """Get current month's total AI spend."""
        self._rotate_month()
        return self._month_cost

    def get_budget_remaining(self) -> float:
        """Get remaining budget for this month."""
        self._rotate_month()
        return max(0, self.config.max_cost_per_month - self._month_cost)

    # ── Two-tier analysis pipeline ─────────────────────────────────

    def _two_tier_analysis(
        self,
        cache_key: str,
        prompt: str,
        expect_probability: bool = False,
    ) -> AgentVerdict | None:
        """Haiku pre-screen → Sonnet deep analysis pipeline."""
        # Tier 1: Haiku quick screen
        haiku_cost = self._estimate_cost(self.config.quick_model)
        if not self._check_budget(haiku_cost):
            self.logger.warning("Monthly budget exceeded. Skipping AI analysis.")
            return None

        should_investigate = self._quick_screen(prompt)
        if not should_investigate:
            skip_verdict = AgentVerdict(
                action="skip",
                confidence=0.3,
                reasoning="Haiku pre-screen: no further investigation needed.",
                cost_usd=haiku_cost,
                model_used=self.config.quick_model,
            )
            self._set_cache(cache_key, skip_verdict)
            return skip_verdict

        # Tier 2: Sonnet deep analysis
        sonnet_cost = self._estimate_cost(self.config.deep_model)
        if not self._check_budget(sonnet_cost):
            self.logger.warning("Budget insufficient for deep analysis. Returning Haiku result.")
            return AgentVerdict(
                action="hold",
                confidence=0.4,
                reasoning="Budget limited. Haiku suggested investigation but Sonnet analysis skipped.",
                cost_usd=haiku_cost,
                model_used=self.config.quick_model,
            )

        verdict = self._call_model(self.config.deep_model, prompt, expect_probability=expect_probability)
        if verdict is not None:
            self._set_cache(cache_key, verdict)
        return verdict

    def _quick_screen(self, prompt: str) -> bool:
        """Haiku pre-screen: should we investigate further?

        Returns True if Haiku thinks the signal warrants deep analysis.
        """
        screen_prompt = (
            f"Quick assessment — should this trading signal be investigated further? "
            f"Reply with ONLY 'yes' or 'no'.\n\n{prompt}"
        )

        try:
            response = self._anthropic_client.messages.create(
                model=self.config.quick_model,
                max_tokens=10,
                messages=[{"role": "user", "content": screen_prompt}],
            )
            cost = self._estimate_cost(self.config.quick_model)
            self._record_cost(cost)

            text = response.content[0].text.strip().lower()
            result = text.startswith("yes")
            self.logger.debug("Haiku screen: '%s' → investigate=%s cost=$%.4f", text, result, cost)
            return result

        except Exception as exc:
            self.logger.error("Haiku pre-screen failed: %s", exc)
            # On failure, default to investigating (conservative)
            return True

    def _call_model(
        self,
        model: str,
        prompt: str,
        expect_probability: bool = False,
    ) -> AgentVerdict | None:
        """Call Claude model and parse the JSON verdict response."""
        try:
            response = self._anthropic_client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            cost = self._estimate_cost(model)
            self._record_cost(cost)

            text = response.content[0].text.strip()

            # Parse JSON from response (handle markdown code blocks)
            json_text = text
            if "```" in json_text:
                start = json_text.find("{")
                end = json_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_text = json_text[start:end]

            # Try direct JSON parse
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                # Try to find JSON object in the text
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(text[start:end])
                else:
                    self.logger.warning("Could not parse JSON from model response: %s", text[:200])
                    return None

            action = data.get("action", "hold")
            if action not in ("buy", "sell", "hold", "skip"):
                action = "hold"

            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            reasoning = str(data.get("reasoning", ""))[:300]

            estimated_prob = None
            if expect_probability and "estimated_probability" in data:
                estimated_prob = max(0.0, min(1.0, float(data["estimated_probability"])))

            verdict = AgentVerdict(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                estimated_probability=estimated_prob,
                cost_usd=cost,
                model_used=model,
            )
            self.logger.info(
                "AI verdict: action=%s confidence=%.2f cost=$%.4f model=%s reason='%s'",
                verdict.action, verdict.confidence, verdict.cost_usd, model, reasoning[:80],
            )
            return verdict

        except Exception as exc:
            self.logger.error("Model call failed (%s): %s", model, exc)
            return None

    # ── Cache management ───────────────────────────────────────────

    def _check_cache(self, key: str) -> AgentVerdict | None:
        """Check cache using the configured default TTL."""
        return self._check_cache_with_ttl(key)

    def _check_cache_with_ttl(self, key: str, ttl_seconds: float | None = None) -> AgentVerdict | None:
        """Check cache, returning None if the entry exceeds ttl_seconds since storage."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._cache[key]
            return None
        # If a specific TTL override is requested, check if entry is still fresh enough
        if ttl_seconds is not None:
            stored_at = entry.expires_at - (self.config.cache_ttl_hours * 3600)
            age = time.time() - stored_at
            if age > ttl_seconds:
                return None
        self.logger.debug("Cache hit for %s", key)
        return AgentVerdict(
            action=entry.verdict.action,
            confidence=entry.verdict.confidence,
            reasoning=entry.verdict.reasoning,
            estimated_probability=entry.verdict.estimated_probability,
            cost_usd=0.0,
            model_used=entry.verdict.model_used,
            cached=True,
        )

    def _set_cache(self, key: str, verdict: AgentVerdict) -> None:
        self._set_cache_with_ttl(key, verdict, ttl_seconds=self.config.cache_ttl_hours * 3600)

    def _set_cache_with_ttl(self, key: str, verdict: AgentVerdict, ttl_seconds: float) -> None:
        self._cache[key] = _CacheEntry(
            verdict=verdict,
            expires_at=time.time() + ttl_seconds,
        )

    # ── Budget management ──────────────────────────────────────────

    def _estimate_cost(self, model: str) -> float:
        """Rough cost estimate per call based on model."""
        base = self._MODEL_COSTS.get(model, 0.01)
        # Approximate 2K tokens per call
        return base * 2

    def _check_budget(self, estimated_cost: float) -> bool:
        """Check if we can afford this call within per-run and monthly caps."""
        self._rotate_month()

        if estimated_cost > self.config.max_cost_per_run:
            self.logger.warning(
                "Estimated cost $%.4f exceeds per-run cap $%.2f",
                estimated_cost, self.config.max_cost_per_run,
            )
            return False

        if (self._month_cost + estimated_cost) > self.config.max_cost_per_month:
            self.logger.warning(
                "Monthly spend $%.2f + $%.4f would exceed cap $%.2f",
                self._month_cost, estimated_cost, self.config.max_cost_per_month,
            )
            return False

        return True

    def _record_cost(self, cost: float) -> None:
        self._rotate_month()
        self._month_cost += cost
        self.logger.debug("Recorded cost $%.4f. Monthly total: $%.4f", cost, self._month_cost)

    def _rotate_month(self) -> None:
        """Reset monthly cost tracker when month changes."""
        current = datetime.now().strftime("%Y-%m")
        if current != self._month_key:
            if self._month_key:
                self.logger.info(
                    "Month rotated %s → %s. Previous month cost: $%.2f",
                    self._month_key, current, self._month_cost,
                )
            self._month_key = current
            self._month_cost = 0.0


def apply_ai_confirmation(
    verdict: AgentVerdict | None,
    confirmation_mode: str,
    base_notional: float,
    logger: logging.Logger,
) -> tuple[float, dict[str, Any]]:
    """Apply AI confirmation mode to adjust position sizing.

    Args:
        verdict: AI analysis result (None if AI unavailable).
        confirmation_mode: "log_only", "boost", or "require".
        base_notional: The base position size from risk manager.
        logger: Logger instance.

    Returns:
        (adjusted_notional, meta_dict) — adjusted size and AI metadata to store.
    """
    meta: dict[str, Any] = {}

    if verdict is None:
        return base_notional, meta

    meta["ai_verdict"] = verdict.to_dict()

    if confirmation_mode == "log_only":
        # AI opinion logged in metadata, no effect on sizing
        logger.info("AI verdict (log_only): %s confidence=%.2f", verdict.action, verdict.confidence)
        return base_notional, meta

    elif confirmation_mode == "boost":
        # AI agreement = 1.5x position, disagreement = 0.5x
        if verdict.action == "buy" and verdict.confidence >= 0.6:
            adjusted = base_notional * 1.5
            logger.info("AI boost: 1.5x (buy confidence=%.2f)", verdict.confidence)
        elif verdict.action in ("sell", "skip"):
            adjusted = base_notional * 0.5
            logger.info("AI reduction: 0.5x (action=%s confidence=%.2f)", verdict.action, verdict.confidence)
        else:
            adjusted = base_notional
        meta["ai_adjustment"] = round(adjusted / base_notional, 2) if base_notional > 0 else 1.0
        return adjusted, meta

    elif confirmation_mode == "require":
        # AI must agree or trade is skipped
        if verdict.action == "buy" and verdict.confidence >= 0.5:
            logger.info("AI confirmed trade (require mode). confidence=%.2f", verdict.confidence)
            return base_notional, meta
        else:
            logger.info(
                "AI rejected trade (require mode). action=%s confidence=%.2f",
                verdict.action, verdict.confidence,
            )
            return 0.0, meta

    # Unknown mode — treat as log_only
    return base_notional, meta
