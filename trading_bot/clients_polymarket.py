"""Polymarket API clients for market discovery and pricing.

Two public APIs (no auth required):
  - Gamma API: Market discovery, metadata, search by tags
  - CLOB API: Real-time pricing, orderbook, midpoints
"""

from __future__ import annotations

import logging
from typing import Any

from trading_bot.clients import BaseHttpClient
from trading_bot.models import PolymarketMarket


class PolymarketGammaClient(BaseHttpClient):
    """Market discovery via https://gamma-api.polymarket.com (public, no auth)."""

    def __init__(self, *, timeout_seconds: int, logger: logging.Logger) -> None:
        super().__init__(
            base_url="https://gamma-api.polymarket.com",
            timeout_seconds=timeout_seconds,
            logger=logger.getChild("polymarket_gamma"),
        )

    def discover_markets(
        self,
        *,
        tags: list[str] | None = None,
        min_volume: float = 0.0,
        min_liquidity: float = 0.0,
        limit: int = 20,
        active: bool = True,
        closed: bool = False,
    ) -> list[PolymarketMarket]:
        """Discover prediction markets matching criteria.

        Returns markets sorted by volume (descending) that meet
        minimum volume and liquidity thresholds.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": "volume24hr",
            "ascending": "false",
        }
        if tags:
            # Gamma API accepts comma-separated tag filter
            params["tag"] = ",".join(tags)

        try:
            payload = self._get_json("markets", params)
        except Exception as exc:
            self.logger.error("Failed to discover markets: %s", exc)
            return []

        if not isinstance(payload, list):
            self.logger.warning("Unexpected Gamma API response type: %s", type(payload))
            return []

        markets: list[PolymarketMarket] = []
        for item in payload:
            try:
                market = self._parse_market(item)
                if market is None:
                    continue
                if market.volume_24h < min_volume:
                    continue
                if market.liquidity < min_liquidity:
                    continue
                markets.append(market)
            except Exception as exc:
                self.logger.debug("Skipping market parse error: %s", exc)
                continue

        self.logger.info(
            "Discovered %d markets (from %d raw) matching filters",
            len(markets),
            len(payload),
        )
        return markets

    def get_market(self, condition_id: str) -> PolymarketMarket | None:
        """Get a single market by condition ID."""
        try:
            payload = self._get_json(f"markets/{condition_id}")
        except Exception as exc:
            self.logger.error("Failed to get market %s: %s", condition_id, exc)
            return None

        if not isinstance(payload, dict):
            return None
        return self._parse_market(payload)

    def _parse_market(self, data: dict[str, Any]) -> PolymarketMarket | None:
        """Parse a Gamma API market response into a PolymarketMarket."""
        condition_id = data.get("conditionId") or data.get("condition_id", "")
        if not condition_id:
            return None

        # Extract token IDs from the tokens array or clobTokenIds
        tokens = data.get("tokens", [])
        yes_token_id = ""
        no_token_id = ""
        yes_price = 0.0
        no_price = 0.0

        if isinstance(tokens, list) and len(tokens) >= 2:
            for token in tokens:
                outcome = (token.get("outcome") or "").upper()
                token_id = token.get("token_id", "")
                price = _safe_float(token.get("price", 0))
                if outcome == "YES":
                    yes_token_id = token_id
                    yes_price = price
                elif outcome == "NO":
                    no_token_id = token_id
                    no_price = price
        else:
            # Fallback: clobTokenIds field (comma-separated or list)
            clob_ids = data.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                clob_ids = [t.strip() for t in clob_ids.split(",") if t.strip()]
            if len(clob_ids) >= 2:
                yes_token_id = clob_ids[0]
                no_token_id = clob_ids[1]

            # Try outcomePrices
            outcome_prices = data.get("outcomePrices", [])
            if isinstance(outcome_prices, str):
                try:
                    import json
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []
            if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                yes_price = _safe_float(outcome_prices[0])
                no_price = _safe_float(outcome_prices[1])

        # Tags
        raw_tags = data.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
        elif isinstance(raw_tags, list):
            raw_tags = [str(t).strip().lower() for t in raw_tags]
        else:
            raw_tags = []

        return PolymarketMarket(
            condition_id=condition_id,
            question=data.get("question", ""),
            description=data.get("description", "")[:500],
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_price=yes_price,
            no_price=no_price,
            volume_24h=_safe_float(data.get("volume24hr", data.get("volume", 0))),
            liquidity=_safe_float(data.get("liquidity", 0)),
            tags=raw_tags,
            end_date=data.get("endDate", data.get("end_date_iso", "")),
            active=bool(data.get("active", True)),
            resolved=bool(data.get("resolved", False)),
            outcome=data.get("outcome"),
        )


class PolymarketClobClient(BaseHttpClient):
    """CLOB pricing via https://clob.polymarket.com (public, no auth)."""

    def __init__(self, *, timeout_seconds: int, logger: logging.Logger) -> None:
        super().__init__(
            base_url="https://clob.polymarket.com",
            timeout_seconds=timeout_seconds,
            logger=logger.getChild("polymarket_clob"),
        )

    def get_midpoint(self, token_id: str) -> float:
        """Get the midpoint price for a single token."""
        payload = self._get_json("midpoint", {"token_id": token_id})
        mid = _safe_float(payload.get("mid", 0) if isinstance(payload, dict) else 0)
        return mid

    def get_midpoints(self, token_ids: list[str]) -> dict[str, float]:
        """Get midpoint prices for multiple tokens."""
        results: dict[str, float] = {}
        for token_id in token_ids:
            try:
                results[token_id] = self.get_midpoint(token_id)
            except Exception as exc:
                self.logger.warning("Failed to get midpoint for %s: %s", token_id[:12], exc)
        return results

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """Get the full orderbook for a token."""
        try:
            payload = self._get_json("book", {"token_id": token_id})
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            self.logger.warning("Failed to get orderbook for %s: %s", token_id[:12], exc)
            return {}

    def get_market_info(self, condition_id: str) -> dict[str, Any]:
        """Get CLOB market info by condition ID."""
        try:
            payload = self._get_json("markets", {"condition_id": condition_id})
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            self.logger.warning("Failed to get CLOB market info for %s: %s", condition_id[:12], exc)
            return {}


def _safe_float(value: Any) -> float:
    """Safely convert a value to float, returning 0.0 on failure."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
