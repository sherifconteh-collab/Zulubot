"""Robinhood API client for options order execution.

Uses the robin_stocks library (pip install robin-stocks).
pyotp is required only when ROBINHOOD_MFA_SECRET is set.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import robin_stocks.robinhood as rh
    _ROBIN_STOCKS_AVAILABLE = True
except ImportError:
    _ROBIN_STOCKS_AVAILABLE = False

try:
    import pyotp
    _PYOTP_AVAILABLE = True
except ImportError:
    _PYOTP_AVAILABLE = False


class RobinhoodClient:
    """Thin wrapper around robin_stocks for options order execution."""

    def __init__(
        self,
        username: str = "",
        password: str = "",
        mfa_secret: str = "",
        logger: logging.Logger | None = None,
    ) -> None:
        self.username = username
        self.password = password
        self.mfa_secret = mfa_secret
        self.logger = (logger or logging.getLogger(__name__)).getChild("robinhood")
        self._logged_in = False

    @property
    def is_available(self) -> bool:
        return _ROBIN_STOCKS_AVAILABLE and bool(self.username and self.password)

    @property
    def can_trade(self) -> bool:
        return self.is_available

    def login(self) -> bool:
        if not _ROBIN_STOCKS_AVAILABLE:
            self.logger.warning("robin_stocks not installed — Robinhood execution unavailable.")
            return False
        if not self.username or not self.password:
            self.logger.warning("Robinhood credentials not configured.")
            return False
        try:
            mfa_code: str | None = None
            if self.mfa_secret:
                if _PYOTP_AVAILABLE:
                    mfa_code = pyotp.TOTP(self.mfa_secret).now()
                else:
                    self.logger.error(
                        "ROBINHOOD_MFA_SECRET is set but pyotp is not installed. "
                        "2FA is required — install pyotp or remove ROBINHOOD_MFA_SECRET."
                    )
                    return False
            kwargs: dict[str, Any] = {"store_session": True, "pickle_name": "zulubot"}
            if mfa_code:
                kwargs["mfa_code"] = mfa_code
            rh.login(self.username, self.password, **kwargs)
            self._logged_in = True
            self.logger.info("Robinhood login successful for %s.", self.username)
            return True
        except Exception as exc:
            self.logger.error("Robinhood login failed: %s", exc)
            return False

    def _ensure_logged_in(self) -> bool:
        if not self._logged_in:
            return self.login()
        return True

    def get_expiration_dates(self, symbol: str) -> list[str]:
        if not self._ensure_logged_in():
            return []
        try:
            chain_info = rh.options.get_chains(symbol, info=None)
            if isinstance(chain_info, dict):
                return list(chain_info.get("expiration_dates") or [])
            return []
        except Exception as exc:
            self.logger.error("Failed to get expiration dates for %s: %s", symbol, exc)
            return []

    def find_options(
        self,
        symbol: str,
        expiration_date: str,
        option_type: str = "call",
    ) -> list[dict[str, Any]]:
        """Return all contracts for a symbol/expiration/type from Robinhood."""
        if not self._ensure_logged_in():
            return []
        try:
            contracts = rh.options.find_options_by_expiration(
                [symbol],
                expirationDate=expiration_date,
                optionType=option_type,
                info=None,
            )
            return [c for c in (contracts or []) if isinstance(c, dict)]
        except Exception as exc:
            self.logger.error(
                "Failed to find options for %s %s %s: %s", symbol, expiration_date, option_type, exc
            )
            return []

    def get_option_market_data(
        self,
        symbol: str,
        expiration_date: str,
        strike: str,
        option_type: str = "call",
    ) -> dict[str, Any]:
        """Return current bid/ask/greeks for a specific option contract."""
        if not self._ensure_logged_in():
            return {}
        try:
            data = rh.options.get_option_market_data(
                symbol,
                expirationDate=expiration_date,
                strike=strike,
                optionType=option_type,
                info=None,
            )
            if isinstance(data, list) and data:
                item = data[0]
                return item if isinstance(item, dict) else {}
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            self.logger.error(
                "Failed to get market data for %s %s %s %s: %s",
                symbol, expiration_date, strike, option_type, exc,
            )
            return {}

    def get_open_option_positions(self) -> list[dict[str, Any]]:
        """Return all open option positions from the Robinhood account."""
        if not self._ensure_logged_in():
            return []
        try:
            positions = rh.options.get_open_option_positions(info=None) or []
            return [p for p in positions if isinstance(p, dict)]
        except Exception as exc:
            self.logger.error("Failed to get open option positions: %s", exc)
            return []

    def place_option_buy_order(
        self,
        *,
        symbol: str,
        expiration_date: str,
        strike: str,
        option_type: str,
        quantity: int,
        limit_price: float,
    ) -> dict[str, Any] | None:
        """Place a debit buy-to-open limit order."""
        if not self._ensure_logged_in():
            return None
        try:
            order = rh.orders.order_buy_option_limit(
                positionEffect="open",
                creditOrDebit="debit",
                price=round(limit_price, 2),
                symbol=symbol,
                quantity=quantity,
                expirationDate=expiration_date,
                strike=strike,
                optionType=option_type,
                timeInForce="gtc",
            )
            if isinstance(order, dict) and "id" in order:
                self.logger.info(
                    "Placed BUY option order: %s %s %s %s qty=%d @ $%.2f id=%s",
                    symbol, expiration_date, strike, option_type, quantity, limit_price,
                    order["id"],
                )
                return order
            self.logger.error("BUY option order rejected or missing id: %s", order)
            return None
        except Exception as exc:
            self.logger.error("Failed to place option buy order for %s: %s", symbol, exc)
            return None

    def place_option_sell_order(
        self,
        *,
        symbol: str,
        expiration_date: str,
        strike: str,
        option_type: str,
        quantity: int,
        limit_price: float,
    ) -> dict[str, Any] | None:
        """Place a credit sell-to-close limit order."""
        if not self._ensure_logged_in():
            return None
        try:
            order = rh.orders.order_sell_option_limit(
                positionEffect="close",
                creditOrDebit="credit",
                price=round(limit_price, 2),
                symbol=symbol,
                quantity=quantity,
                expirationDate=expiration_date,
                strike=strike,
                optionType=option_type,
                timeInForce="gtc",
            )
            if isinstance(order, dict) and "id" in order:
                self.logger.info(
                    "Placed SELL option order: %s %s %s %s qty=%d @ $%.2f id=%s",
                    symbol, expiration_date, strike, option_type, quantity, limit_price,
                    order["id"],
                )
                return order
            self.logger.error("SELL option order rejected or missing id: %s", order)
            return None
        except Exception as exc:
            self.logger.error("Failed to place option sell order for %s: %s", symbol, exc)
            return None

    def get_account_buying_power(self) -> float:
        """Return available option buying power in the account."""
        if not self._ensure_logged_in():
            return 0.0
        try:
            profile = rh.account.load_account_profile(info=None)
            if isinstance(profile, dict):
                bp = profile.get("option_buying_power") or profile.get("buying_power") or 0.0
                return float(bp)
            return 0.0
        except Exception as exc:
            self.logger.error("Failed to get buying power: %s", exc)
            return 0.0
