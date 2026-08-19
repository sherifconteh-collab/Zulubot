"""Kalshi prediction market API client.

Kalshi powers Coinbase's prediction markets. This client connects
directly to Kalshi's REST API for market discovery and order execution.

Market discovery (public endpoints) works WITHOUT authentication.
Trading requires RSA-PSS signed requests with a Kalshi API key.

Sign up at https://kalshi.com to get API credentials.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Any

import requests

from trading_bot.clients import request_with_retry
from trading_bot.config import HttpRetryConfig

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


# Kalshi Crypto Perpetuals series tickers.
# ⚠️ Verify exact series names against Kalshi's live API before enabling live trading.
PERP_SERIES: dict[str, str] = {
    "BTC":  "KXBTCPERP",
    "ETH":  "KXETHPERP",
    "SOL":  "KXSOLPERP",
    "XRP":  "KXXRPPERP",
    "LINK": "KXLINKPERP",
    "BCH":  "KXBCHPERP",
    "HYPE": "KXHYPEPERP",
    "SHIB": "KXSHIBPERP",
}


class KalshiClient:
    """Client for the Kalshi prediction market API."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        *,
        key_id: str = "",
        private_key_pem: str = "",
        timeout_seconds: int = 15,
        logger: logging.Logger,
        retry_config: HttpRetryConfig | None = None,
    ) -> None:
        self.key_id = key_id
        self.private_key = None
        self.timeout = timeout_seconds
        self.logger = logger.getChild("kalshi")
        self.retry_config = retry_config or HttpRetryConfig()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Load RSA private key if provided
        if private_key_pem and _HAS_CRYPTO:
            try:
                pem = private_key_pem.replace("\\n", "\n").encode("utf-8")
                self.private_key = serialization.load_pem_private_key(pem, password=None)
                self.logger.info("Kalshi RSA key loaded successfully.")
            except Exception as exc:
                self.logger.warning("Failed to load Kalshi RSA key: %s", exc)

    @property
    def can_trade(self) -> bool:
        """Whether we have credentials to place orders."""
        return bool(self.key_id and self.private_key and _HAS_CRYPTO)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """Sign a request with RSA-PSS."""
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")
        signature = self.private_key.sign(  # type: ignore[union-attr]
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate Kalshi auth headers."""
        timestamp = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(timestamp, method, path),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None, auth: bool = False) -> dict[str, Any]:
        """GET request. Public endpoints don't need auth."""
        url = f"{self.BASE_URL}{path}"
        # Sign with full URL path (e.g., /trade-api/v2/portfolio/balance)
        full_path = f"/trade-api/v2{path}"
        headers = self._auth_headers("GET", full_path) if (auth and self.can_trade) else {}
        resp = request_with_retry(
            self.session,
            "GET",
            url,
            headers=headers,
            params=params,
            timeout_seconds=self.timeout,
            logger=self.logger,
            retry_config=self.retry_config,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST request (always authenticated)."""
        if not self.can_trade:
            raise RuntimeError("Kalshi trading requires API credentials (KALSHI_KEY_ID + KALSHI_PRIVATE_KEY)")
        url = f"{self.BASE_URL}{path}"
        full_path = f"/trade-api/v2{path}"
        headers = self._auth_headers("POST", full_path)
        resp = request_with_retry(
            self.session,
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout_seconds=self.timeout,
            logger=self.logger,
            retry_config=self.retry_config,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public market data (no auth needed)
    # ------------------------------------------------------------------

    def get_markets(
        self,
        *,
        status: str = "open",
        series_ticker: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch markets. Returns (markets, next_cursor)."""
        params: dict[str, Any] = {"limit": limit, "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        result = self._get("/markets", params=params)
        markets = result.get("markets", [])
        next_cursor = result.get("cursor")
        return markets, next_cursor

    def get_market(self, ticker: str) -> dict[str, Any]:
        """Get a single market by ticker."""
        result = self._get(f"/markets/{ticker}")
        return result.get("market", result)

    def get_events(
        self,
        *,
        status: str = "open",
        series_ticker: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fetch events."""
        params: dict[str, Any] = {"limit": limit, "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        result = self._get("/events", params=params)
        return result.get("events", [])

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """Get the orderbook for a market."""
        return self._get(f"/markets/{ticker}/orderbook")

    def discover_prediction_markets(
        self,
        categories: list[str] | None = None,
        max_markets: int = 50,
    ) -> list[dict[str, Any]]:
        """Discover open prediction markets, optionally filtered by category.

        Categories: crypto, economics, politics, sports, entertainment, etc.
        Returns markets sorted by volume (most active first).
        """
        all_markets: list[dict[str, Any]] = []
        cursor = None

        for _ in range(5):  # max 5 pages
            markets, cursor = self.get_markets(status="open", limit=100, cursor=cursor)
            all_markets.extend(markets)
            if not cursor or len(all_markets) >= max_markets * 3:
                break

        # Filter by category keywords if provided
        if categories:
            cat_keywords = {c.lower() for c in categories}
            filtered = []
            for m in all_markets:
                title = (m.get("title", "") + " " + m.get("event_ticker", "")).lower()
                ticker = m.get("ticker", "").lower()
                # Match crypto keywords
                if "crypto" in cat_keywords and any(
                    k in title or k in ticker
                    for k in ("btc", "eth", "bitcoin", "ethereum", "crypto", "sol", "xrp")
                ):
                    filtered.append(m)
                    continue
                # Match economics keywords
                if "economics" in cat_keywords and any(
                    k in title or k in ticker
                    for k in ("cpi", "gdp", "fed", "fomc", "nfp", "inflation", "rate", "unemployment", "econ")
                ):
                    filtered.append(m)
                    continue
                # Match politics
                if "politics" in cat_keywords and any(
                    k in title or k in ticker
                    for k in ("president", "election", "congress", "senate", "vote", "trump", "biden", "gov")
                ):
                    filtered.append(m)
                    continue
                # If no specific categories, include everything
                if not cat_keywords - {"crypto", "economics", "politics"}:
                    continue
                filtered.append(m)
            all_markets = filtered

        # Sort by volume (most active first) and cap
        all_markets.sort(key=lambda m: int(m.get("volume", 0)), reverse=True)
        result = all_markets[:max_markets]

        self.logger.info(
            "Discovered %d prediction markets (from %d total). Top: %s",
            len(result),
            len(all_markets),
            [(m.get("ticker"), m.get("title", "")[:40]) for m in result[:5]],
        )
        return result

    # ------------------------------------------------------------------
    # Trading (requires auth)
    # ------------------------------------------------------------------

    def place_order(
        self,
        *,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str = "buy",  # "buy" or "sell"
        count: int = 1,
        yes_price: int | None = None,  # cents (1-99), None for market order
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place an order on Kalshi.

        Args:
            ticker: Market ticker (e.g., "KXBTC-26APR0117")
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            yes_price: Limit price in cents (1-99). Omit for IOC at best.
            client_order_id: Optional tracking ID.

        Returns:
            Order result dict from Kalshi.
        """
        payload: dict[str, Any] = {
            "ticker": ticker,
            "side": side.lower(),
            "action": action.lower(),
            "count": count,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "type": "limit",
        }
        if yes_price is not None:
            payload["yes_price"] = yes_price
        else:
            # Use IOC to get filled immediately at market price
            payload["time_in_force"] = "immediate_or_cancel"

        self.logger.info(
            "Kalshi order: %s %s %s x%d @ %s cents",
            action, side, ticker, count, yes_price or "market",
        )
        result = self._post("/portfolio/orders", payload)
        self.logger.info("Kalshi order result: %s", json.dumps(result, indent=2))
        return result

    def get_positions(self) -> list[dict[str, Any]]:
        """Get current portfolio positions."""
        if not self.can_trade:
            return []
        result = self._get("/portfolio/positions", auth=True)
        return result.get("market_positions", [])

    def get_balance(self) -> float:
        """Get account balance in dollars."""
        if not self.can_trade:
            return 0.0
        result = self._get("/portfolio/balance", auth=True)
        # Balance is in cents
        return float(result.get("balance", 0)) / 100.0

    # ------------------------------------------------------------------
    # Perpetuals (requires auth for trading; discovery is public)
    # ⚠️ Exact endpoints need verification against Kalshi Perps API docs.
    # All methods are designed to use _get()/_post() so the base URL,
    # path, and payload can be adjusted by changing constants only.
    # ------------------------------------------------------------------

    def get_perp_markets(
        self,
        *,
        series_ticker: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch open perpetual markets. Returns (markets, next_cursor)."""
        params: dict[str, Any] = {"limit": limit, "status": "open"}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        result = self._get("/markets", params=params)
        markets = result.get("markets", [])
        next_cursor = result.get("cursor")
        return markets, next_cursor

    def get_perp_market(self, ticker: str) -> dict[str, Any]:
        """Get a single perpetual market by ticker (for price refresh)."""
        result = self._get(f"/markets/{ticker}")
        return result.get("market", result)

    def place_perp_order(
        self,
        *,
        ticker: str,
        side: str,            # "long" or "short"
        size_usd: float,
        leverage: float = 1.0,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place a leveraged perpetual order on Kalshi.

        ⚠️ The /perp/orders endpoint and payload structure need verification.
        Adjust ``path`` and ``payload`` once you have the official API spec.
        """
        payload: dict[str, Any] = {
            "ticker": ticker,
            "side": side.lower(),
            "size_usd": round(size_usd, 2),
            "leverage": round(leverage, 2),
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "type": "market",
        }
        self.logger.info(
            "Kalshi perp order: %s %s size=$%.2f leverage=%.1fx",
            side, ticker, size_usd, leverage,
        )
        result = self._post("/perp/orders", payload)
        self.logger.info("Kalshi perp order result: %s", json.dumps(result, indent=2))
        return result

    def close_perp_position(self, ticker: str, close_side: str) -> dict[str, Any]:
        """Close an open perpetual position by placing a reverse order at market.

        ⚠️ Adjust the endpoint/payload once the official Kalshi Perps API is confirmed.
        """
        payload: dict[str, Any] = {
            "ticker": ticker,
            "side": close_side.lower(),
            "type": "market",
            "client_order_id": str(uuid.uuid4()),
        }
        self.logger.info("Closing perp %s (side=%s)", ticker, close_side)
        return self._post("/perp/orders", payload)
