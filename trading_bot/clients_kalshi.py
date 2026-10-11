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


def _levels(raw: Any) -> list[tuple[float, float]]:
    """Normalise a price-level list to [(price_dollars, size)], ascending by price.

    Handles ``["0.4300", "10.0"]`` (new ``*_dollars`` layout), cents ``[43, 10]``
    (legacy layout) and ``{"price": .., "quantity": ..}`` dict levels.
    """
    out: list[tuple[float, float]] = []
    for level in raw or []:
        try:
            if isinstance(level, dict):
                price, size = float(level.get("price", 0)), float(level.get("quantity", level.get("size", 0)))
            else:
                price, size = float(level[0]), float(level[1]) if len(level) > 1 else 0.0
        except (TypeError, ValueError, IndexError):
            continue
        out.append((price, size))
    return sorted(out)


def parse_top_of_book(book: dict[str, Any]) -> dict[str, float]:
    """Best bid/ask and size for both sides from a Kalshi orderbook response.

    Kalshi's book lists only *bids* for YES and for NO. A NO bid at ``p`` is an
    offer to sell YES at ``1 - p``, so::

        yes_ask = 1 - best_no_bid        no_ask = 1 - best_yes_bid

    Missing sides yield 0.0. Prices are in dollars (0-1).
    """
    if "orderbook_fp" in book:
        ob = book["orderbook_fp"] or {}
        yes_raw, no_raw, cents = ob.get("yes_dollars"), ob.get("no_dollars"), False
    else:
        ob = book.get("orderbook", book) or {}
        yes_raw, no_raw = ob.get("yes_dollars") or ob.get("yes"), ob.get("no_dollars") or ob.get("no")
        cents = "yes_dollars" not in ob and "no_dollars" not in ob
    yes_bids, no_bids = _levels(yes_raw), _levels(no_raw)
    if cents:  # legacy prices are integer cents
        yes_bids = [(p / 100.0, q) for p, q in yes_bids]
        no_bids = [(p / 100.0, q) for p, q in no_bids]

    yes_bid, yes_bid_size = yes_bids[-1] if yes_bids else (0.0, 0.0)
    no_bid, no_bid_size = no_bids[-1] if no_bids else (0.0, 0.0)
    return {
        "yes_bid": yes_bid,
        "yes_bid_size": yes_bid_size,
        "no_bid": no_bid,
        "no_bid_size": no_bid_size,
        # Size resting at our ask is the opposite side's best bid size.
        "yes_ask": round(1.0 - no_bid, 4) if no_bid > 0 else 0.0,
        "yes_ask_size": no_bid_size,
        "no_ask": round(1.0 - yes_bid, 4) if yes_bid > 0 else 0.0,
        "no_ask_size": yes_bid_size,
    }


def order_fill_count(result: dict[str, Any]) -> int | None:
    """Contracts filled according to an order response, or None if it can't be told.

    Looks at ``fill_count`` and the taker/maker split, in plain or ``*_fp`` (string)
    form. None means the response had no recognisable fill fields.
    """
    order = result.get("order", result) if isinstance(result, dict) else {}

    def num(*keys: str) -> float | None:
        for key in keys:
            raw = order.get(key)
            if raw not in (None, ""):
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    continue
        return None

    total = num("fill_count", "fill_count_fp")
    if total is not None:
        return int(round(total))
    taker = num("taker_fill_count", "taker_fill_count_fp")
    maker = num("maker_fill_count", "maker_fill_count_fp")
    if taker is None and maker is None:
        return None
    return int(round((taker or 0.0) + (maker or 0.0)))


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

    def _put(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.can_trade:
            raise RuntimeError("Kalshi trading requires API credentials (KALSHI_KEY_ID + KALSHI_PRIVATE_KEY)")
        resp = request_with_retry(
            self.session,
            "PUT",
            f"{self.BASE_URL}{path}",
            headers=self._auth_headers("PUT", f"/trade-api/v2{path}"),
            json=payload,
            timeout_seconds=self.timeout,
            logger=self.logger,
            retry_config=self.retry_config,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _delete(self, path: str) -> dict[str, Any]:
        if not self.can_trade:
            raise RuntimeError("Kalshi trading requires API credentials (KALSHI_KEY_ID + KALSHI_PRIVATE_KEY)")
        resp = request_with_retry(
            self.session,
            "DELETE",
            f"{self.BASE_URL}{path}",
            headers=self._auth_headers("DELETE", f"/trade-api/v2{path}"),
            timeout_seconds=self.timeout,
            logger=self.logger,
            retry_config=self.retry_config,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

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

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        """Series metadata, including ``fee_type`` and ``fee_multiplier``."""
        result = self._get(f"/series/{series_ticker}")
        return result.get("series", result)

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """Get the orderbook for a market."""
        return self._get(f"/markets/{ticker}/orderbook")

    def get_top_of_book(self, ticker: str) -> dict[str, float]:
        """Best bid/ask (dollars) and sizes for YES and NO; see ``parse_top_of_book``."""
        return parse_top_of_book(self.get_orderbook(ticker))

    def get_recent_trades(self, ticker: str, limit: int = 100) -> list[dict[str, Any]]:
        """Public trade tape for a market, newest first."""
        result = self._get("/markets/trades", params={"ticker": ticker, "limit": limit})
        return result.get("trades", [])

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
        def _volume(m: dict[str, Any]) -> float:
            try:  # current API reports ``volume_fp`` (string); older payloads ``volume``
                return float(m.get("volume_fp") or m.get("volume") or 0)
            except (TypeError, ValueError):
                return 0.0

        all_markets.sort(key=_volume, reverse=True)
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
        yes_price: int | None = None,  # cents (1-99): price of the YES side
        price_cents: int | None = None,  # cents (1-99): price of the contract *being traded*
        time_in_force: str | None = None,  # e.g. "immediate_or_cancel"
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place a limit order on Kalshi.

        Kalshi only accepts priced limit orders, so a price is required. Pass either
        ``price_cents`` (the price of the side you are trading: a NO order at 45 means
        paying 45 cents per NO contract) or ``yes_price`` (always the YES-side price).

        Args:
            ticker: Market ticker (e.g., "KXBTC-26APR0117")
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            price_cents: Limit price for the traded side, in cents (1-99).
            yes_price: Alternative: limit price on the YES side, in cents.
            time_in_force: "immediate_or_cancel" fills what it can now and cancels the rest.
            client_order_id: Optional tracking ID.

        Returns:
            Order result dict from Kalshi (see ``order_fill_count``).
        """
        side = side.lower()
        if price_cents is None and yes_price is None:
            raise ValueError("Kalshi orders need a limit price (price_cents or yes_price).")
        payload: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action.lower(),
            "count": int(count),
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "type": "limit",
        }
        if price_cents is not None:
            cents = int(min(99, max(1, price_cents)))
            payload["yes_price" if side == "yes" else "no_price"] = cents
        else:
            payload["yes_price"] = int(min(99, max(1, yes_price)))  # type: ignore[arg-type]
        if time_in_force:
            payload["time_in_force"] = time_in_force

        self.logger.info(
            "Kalshi order: %s %s %s x%d @ %s cents (%s)",
            action, side, ticker, count,
            payload.get("yes_price", payload.get("no_price")), time_in_force or "gtc",
        )
        result = self._post("/portfolio/orders", payload)
        self.logger.info("Kalshi order result: %s", json.dumps(result, indent=2))
        return result

    def sell_to_close(
        self,
        *,
        ticker: str,
        side: str,
        count: int,
        slippage_cents: int = 3,
    ) -> dict[str, Any]:
        """Sell contracts now: an immediate-or-cancel limit just below the best bid.

        Raises RuntimeError when there is no bid to sell into.
        """
        top = self.get_top_of_book(ticker)
        bid = top.get(f"{side.lower()}_bid", 0.0)
        if bid <= 0:
            raise RuntimeError(f"No {side} bid on {ticker}; cannot sell right now.")
        return self.place_order(
            ticker=ticker,
            side=side,
            action="sell",
            count=count,
            price_cents=max(1, round(bid * 100) - slippage_cents),
            time_in_force="immediate_or_cancel",
        )

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a resting order."""
        return self._delete(f"/portfolio/orders/{order_id}")

    # ------------------------------------------------------------------
    # Combos (multivariate events) and requests for quotes
    # ------------------------------------------------------------------

    def get_multivariate_collections(self, limit: int = 20) -> list[dict[str, Any]]:
        """Open combo collections, each with the events its legs may come from (public)."""
        collections: list[dict[str, Any]] = []
        cursor = None
        for _ in range(5):
            params: dict[str, Any] = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            page = self._get("/multivariate_event_collections", params)
            collections.extend(page.get("multivariate_contracts", []))
            cursor = page.get("cursor")
            if not cursor:
                break
        return collections

    def create_combo_market(self, collection_ticker: str, legs: list[dict[str, str]]) -> dict[str, Any]:
        """Create (or look up) the combo market for these legs.

        ``legs`` are ``{"market_ticker", "event_ticker", "side"}``; the combo pays $1 only if
        every leg resolves the way its side says.
        """
        return self._post(f"/multivariate_event_collections/{collection_ticker}", {"selected_markets": legs})

    def create_rfq(self, market_ticker: str, contracts: int, *, rest_remainder: bool = False) -> str:
        """Ask market makers to quote ``contracts`` of a market; returns the RFQ id."""
        result = self._post("/communications/rfqs", {
            "market_ticker": market_ticker,
            "contracts_fp": f"{int(contracts)}.00",
            "rest_remainder": rest_remainder,
        })
        return str(result.get("id", ""))

    def get_rfq_quotes(self, rfq_id: str) -> list[dict[str, Any]]:
        result = self._get("/communications/quotes", {"rfq_id": rfq_id}, auth=True)
        return result.get("quotes", [])

    def get_rfq_quote(self, rfq_id: str, quote_id: str) -> dict[str, Any]:
        result = self._get(f"/communications/rfqs/{rfq_id}/quotes/{quote_id}", auth=True)
        return result.get("quote", result)

    def accept_rfq_quote(self, rfq_id: str, quote_id: str, accepted_side: str) -> None:
        """Accept a maker's quote. ``accepted_side`` is the maker's bid being hit: accepting
        the maker's NO bid leaves the requester long YES at ``1 - no_bid``."""
        self._put(f"/communications/rfqs/{rfq_id}/quotes/{quote_id}/accept", {"accepted_side": accepted_side})

    def delete_rfq(self, rfq_id: str) -> None:
        self._delete(f"/communications/rfqs/{rfq_id}")

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
