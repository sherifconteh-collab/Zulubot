"""Coinbase Advanced Trade broker for live order execution.

This module provides the bridge between the bot's internal paper broker
and Coinbase's Advanced Trade API. In paper mode, it logs what *would*
happen. When ``paper_mode=False``, it places real market orders via
the Coinbase Advanced Trade REST API.

Requires: COINBASE_API_KEY and COINBASE_API_SECRET in .env

Uses JWT/ES256 authentication for Coinbase Cloud API keys
(the ``organizations/...`` key format).
"""

from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from datetime import datetime
from typing import Any

import jwt
import requests

from trading_bot.clients import CoinbaseClient
from trading_bot.models import PortfolioState, Position


class CoinbaseBroker:
    """Broker that can place real orders on Coinbase Advanced Trade."""

    BASE_URL = "https://api.coinbase.com"

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        coinbase_client: CoinbaseClient,
        paper_mode: bool,
        logger: logging.Logger,
    ) -> None:
        self.api_key = api_key
        # Handle escaped newlines from .env files
        self.api_secret = api_secret.replace("\\n", "\n")
        self.coinbase_client = coinbase_client
        self.paper_mode = paper_mode
        self.logger = logger.getChild("coinbase_broker")
        self.session = requests.Session()

    def _build_jwt(self, method: str, path: str) -> str:
        """Build a JWT token for Coinbase Cloud API authentication (ES256)."""
        uri = f"{method.upper()} api.coinbase.com{path}"
        now = int(time.time())
        payload = {
            "sub": self.api_key,
            "iss": "cdp",
            "nbf": now,
            "exp": now + 120,  # 2 minute expiry
            "uri": uri,
        }
        headers = {
            "kid": self.api_key,
            "nonce": secrets.token_hex(16),
            "typ": "JWT",
        }
        return jwt.encode(payload, self.api_secret, algorithm="ES256", headers=headers)

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate authorization headers with JWT bearer token."""
        token = self._build_jwt(method, path)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload)
        headers = self._auth_headers("POST", path)
        url = f"{self.BASE_URL}{path}"
        self.logger.debug("POST %s payload=%s", url, body)
        resp = self.session.post(url, headers=headers, data=body, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict[str, Any]:
        headers = self._auth_headers("GET", path)
        url = f"{self.BASE_URL}{path}"
        resp = self.session.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def place_market_buy(
        self,
        state: PortfolioState,
        *,
        binance_symbol: str,
        notional_usd: float,
        strategy: str,
        now: datetime,
        meta: dict[str, Any] | None = None,
    ) -> Position | None:
        """Place a market buy on Coinbase (or simulate in paper mode)."""
        product_id = self.coinbase_client.to_coinbase_symbol(binance_symbol)

        if self.paper_mode:
            self.logger.info(
                "[PAPER] Would place market BUY %s notional=$%.2f on Coinbase product=%s",
                binance_symbol, notional_usd, product_id,
            )
            return None  # Paper broker handles the actual position tracking

        client_order_id = str(uuid.uuid4())
        order_payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": f"{notional_usd:.2f}",
                }
            },
        }

        self.logger.info(
            "Placing LIVE market BUY on Coinbase: product=%s notional=$%.2f order_id=%s",
            product_id, notional_usd, client_order_id,
        )
        try:
            result = self._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("Coinbase order response: %s", json.dumps(result, indent=2))
            return self._log_order_result(result, binance_symbol, "BUY")
        except Exception as exc:
            self.logger.error("Coinbase BUY order failed for %s: %s", product_id, exc)
            return None

    def place_market_sell(
        self,
        state: PortfolioState,
        *,
        binance_symbol: str,
        quantity: float,
        strategy: str,
        reason: str,
    ) -> bool:
        """Place a market sell on Coinbase (or simulate in paper mode)."""
        product_id = self.coinbase_client.to_coinbase_symbol(binance_symbol)

        if self.paper_mode:
            self.logger.info(
                "[PAPER] Would place market SELL %s qty=%.6f on Coinbase product=%s reason=%s",
                binance_symbol, quantity, product_id, reason,
            )
            return True

        client_order_id = str(uuid.uuid4())
        order_payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": f"{quantity:.8f}",
                }
            },
        }

        self.logger.info(
            "Placing LIVE market SELL on Coinbase: product=%s qty=%.8f reason=%s order_id=%s",
            product_id, quantity, reason, client_order_id,
        )
        try:
            result = self._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("Coinbase SELL response: %s", json.dumps(result, indent=2))
            return result.get("success", False)
        except Exception as exc:
            self.logger.error("Coinbase SELL order failed for %s: %s", product_id, exc)
            return False

    def place_stock_buy(
        self,
        *,
        symbol: str,
        notional_usd: float,
    ) -> dict[str, Any] | None:
        """Place a market buy for a stock on Coinbase."""
        if self.paper_mode:
            self.logger.info("[PAPER] Would BUY stock %s notional=$%.2f", symbol, notional_usd)
            return None

        client_order_id = str(uuid.uuid4())
        order_payload = {
            "client_order_id": client_order_id,
            "product_id": symbol,
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": f"{notional_usd:.2f}",
                }
            },
        }
        self.logger.info("LIVE stock BUY: %s notional=$%.2f", symbol, notional_usd)
        try:
            result = self._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("Stock BUY response: %s", json.dumps(result, indent=2))
            self._log_order_result(result, symbol, "BUY")
            return result
        except Exception as exc:
            self.logger.error("Stock BUY failed for %s: %s", symbol, exc)
            return None

    def place_stock_sell(
        self,
        *,
        symbol: str,
        quantity: float,
        reason: str = "",
    ) -> bool:
        """Place a market sell for a stock on Coinbase."""
        if self.paper_mode:
            self.logger.info("[PAPER] Would SELL stock %s qty=%.6f", symbol, quantity)
            return True

        client_order_id = str(uuid.uuid4())
        order_payload = {
            "client_order_id": client_order_id,
            "product_id": symbol,
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": f"{quantity:.8f}",
                }
            },
        }
        self.logger.info("LIVE stock SELL: %s qty=%.8f reason=%s", symbol, quantity, reason)
        try:
            result = self._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("Stock SELL response: %s", json.dumps(result, indent=2))
            return result.get("success", False)
        except Exception as exc:
            self.logger.error("Stock SELL failed for %s: %s", symbol, exc)
            return False

    def place_market_buy_by_product(
        self,
        *,
        product_id: str,
        notional_usd: float,
    ) -> dict[str, Any] | None:
        """Place a market buy using Coinbase product_id directly (e.g. 'SOL-USD')."""
        if self.paper_mode:
            self.logger.info("[PAPER] Would BUY %s notional=$%.2f", product_id, notional_usd)
            return None

        client_order_id = str(uuid.uuid4())
        order_payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": f"{notional_usd:.2f}",
                }
            },
        }
        self.logger.info("LIVE market BUY: product=%s notional=$%.2f", product_id, notional_usd)
        try:
            result = self._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("BUY response: %s", json.dumps(result, indent=2))
            self._log_order_result(result, product_id, "BUY")
            return result
        except Exception as exc:
            self.logger.error("BUY order failed for %s: %s", product_id, exc)
            return None

    def place_market_sell_by_product(
        self,
        *,
        product_id: str,
        quantity: float,
        reason: str = "",
    ) -> bool:
        """Place a market sell using Coinbase product_id directly (e.g. 'SOL-USD')."""
        if self.paper_mode:
            self.logger.info("[PAPER] Would SELL %s qty=%.8f reason=%s", product_id, quantity, reason)
            return True

        client_order_id = str(uuid.uuid4())
        order_payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": f"{quantity:.8f}",
                }
            },
        }
        self.logger.info("LIVE market SELL: product=%s qty=%.8f reason=%s", product_id, quantity, reason)
        try:
            result = self._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("SELL response: %s", json.dumps(result, indent=2))
            return result.get("success", False)
        except Exception as exc:
            self.logger.error("SELL order failed for %s: %s", product_id, exc)
            return False

    def get_all_holdings(self) -> dict[str, float]:
        """Get all non-zero crypto balances. Returns {product_id: quantity}.

        Example: {"BTC-USD": 0.00045, "XRP-USD": 12.5, "SOL-USD": 0.3}
        """
        accounts = self.get_accounts()
        holdings: dict[str, float] = {}
        for acct in accounts:
            currency = acct.get("currency", "")
            if currency in ("USD", "USDC"):
                continue
            available = float(acct.get("available_balance", {}).get("value", "0"))
            if available > 0.0001:  # ignore dust
                holdings[f"{currency}-USD"] = available
        return holdings

    def place_prediction_buy(
        self,
        *,
        product_id: str,
        side: str,
        price: float,
        size: int,
    ) -> dict[str, Any] | None:
        """Buy a prediction contract (YES or NO side)."""
        if self.paper_mode:
            self.logger.info(
                "[PAPER] Would BUY prediction %s side=%s price=%.2f size=%d",
                product_id, side, price, size,
            )
            return None

        client_order_id = str(uuid.uuid4())
        order_payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": side,  # "YES" or "NO"
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": str(size),
                    "limit_price": f"{price:.2f}",
                }
            },
        }
        self.logger.info(
            "LIVE prediction BUY: product=%s side=%s price=%.2f size=%d",
            product_id, side, price, size,
        )
        try:
            result = self._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("Prediction BUY response: %s", json.dumps(result, indent=2))
            return result
        except Exception as exc:
            self.logger.error("Prediction BUY failed for %s: %s", product_id, exc)
            return None

    def place_prediction_sell(
        self,
        *,
        product_id: str,
        side: str,
        size: int,
    ) -> bool:
        """Sell/close a prediction position."""
        if self.paper_mode:
            self.logger.info(
                "[PAPER] Would SELL prediction %s side=%s size=%d",
                product_id, side, size,
            )
            return True

        # To sell a prediction, we place an opposite-side order
        client_order_id = str(uuid.uuid4())
        # Selling a YES position means placing a NO, and vice versa
        sell_side = "NO" if side == "YES" else "YES"
        order_payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": sell_side,
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": str(size),
                }
            },
        }
        self.logger.info(
            "LIVE prediction SELL: product=%s original_side=%s size=%d",
            product_id, side, size,
        )
        try:
            result = self._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("Prediction SELL response: %s", json.dumps(result, indent=2))
            return result.get("success", False)
        except Exception as exc:
            self.logger.error("Prediction SELL failed for %s: %s", product_id, exc)
            return False

    def get_accounts(self) -> list[dict[str, Any]]:
        """List Coinbase accounts (balances)."""
        result = self._get("/api/v3/brokerage/accounts")
        return result.get("accounts", [])

    def get_total_balance(self) -> tuple[float, dict[str, float]]:
        """Get total portfolio value in USD and per-asset breakdown.

        Returns:
            (total_usd, {"USD": 100.0, "BTC": 500.0, ...})
        """
        accounts = self.get_accounts()
        breakdown: dict[str, float] = {}
        total = 0.0
        for acct in accounts:
            currency = acct.get("currency", "")
            available = float(acct.get("available_balance", {}).get("value", "0"))
            hold = float(acct.get("hold", {}).get("value", "0"))
            balance = available + hold
            if balance > 0:
                # For USD, value is direct. For crypto, we need USD equivalent.
                if currency == "USD":
                    breakdown[currency] = balance
                    total += balance
                else:
                    # Use Coinbase spot price to convert
                    try:
                        import urllib.request, json as _json
                        url = f"https://api.coinbase.com/v2/prices/{currency}-USD/spot"
                        with urllib.request.urlopen(url, timeout=5) as resp:
                            data = _json.loads(resp.read())
                        usd_value = balance * float(data["data"]["amount"])
                        breakdown[currency] = usd_value
                        total += usd_value
                    except Exception:
                        # If we can't price it, skip
                        pass
        return total, breakdown

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Get details of a specific order."""
        return self._get(f"/api/v3/brokerage/orders/historical/{order_id}")

    def _log_order_result(self, result: dict[str, Any], symbol: str, side: str) -> Position | None:
        success = result.get("success", False)
        order_id = result.get("order_id", "unknown")
        if success:
            self.logger.info("Coinbase %s order SUCCESS for %s order_id=%s", side, symbol, order_id)
        else:
            failure_reason = result.get("failure_reason", "unknown")
            self.logger.error(
                "Coinbase %s order FAILED for %s: %s", side, symbol, failure_reason
            )
        return None
