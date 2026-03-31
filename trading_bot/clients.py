from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import requests
import yfinance as yf

from trading_bot.config import HttpRetryConfig
from trading_bot.models import FearGreedReading, FundingSnapshot


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (requests.Timeout, requests.ConnectionError)


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout_seconds: int,
    logger: logging.Logger,
    retry_config: HttpRetryConfig,
    **kwargs: Any,
) -> requests.Response:
    max_attempts = retry_config.max_attempts if retry_config.enabled else 1
    max_attempts = max(1, max_attempts)

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.request(method=method, url=url, timeout=timeout_seconds, **kwargs)
        except _RETRYABLE_EXCEPTIONS as exc:
            if attempt >= max_attempts:
                raise
            delay = min(
                retry_config.base_delay_seconds * (retry_config.backoff_multiplier ** (attempt - 1)),
                retry_config.max_delay_seconds,
            )
            logger.warning(
                "%s %s failed (%s) attempt %d/%d. Retrying in %.2fs.",
                method.upper(),
                url,
                exc,
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
            continue

        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts:
            delay = min(
                retry_config.base_delay_seconds * (retry_config.backoff_multiplier ** (attempt - 1)),
                retry_config.max_delay_seconds,
            )
            logger.warning(
                "%s %s returned HTTP %d attempt %d/%d. Retrying in %.2fs.",
                method.upper(),
                url,
                response.status_code,
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
            continue

        return response

    raise RuntimeError(f"Retry loop exhausted for {method.upper()} {url}")


class BaseHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: int,
        logger: logging.Logger,
        retry_config: HttpRetryConfig | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.logger = logger
        self.retry_config = retry_config or HttpRetryConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "codex-trading-bot/1.0"})

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        self.logger.debug("GET %s params=%s", url, params)
        response = request_with_retry(
            self.session,
            "GET",
            url,
            params=params,
            timeout_seconds=self.timeout_seconds,
            logger=self.logger,
            retry_config=self.retry_config,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 451:
                raise RuntimeError(
                    f"{url} returned HTTP 451. This endpoint is unavailable from the current network or legal jurisdiction."
                ) from exc
            raise RuntimeError(f"HTTP {response.status_code} returned from {url}.") from exc
        payload = response.json()
        if isinstance(payload, dict) and payload.get("Error Message"):
            raise RuntimeError(payload["Error Message"])
        return payload


class FinancialModelingPrepClient:
    """Stock data client powered by yfinance (100% free, no API key needed).

    Provides the same interface the stock strategy expects:
    - get_sp500_constituents() → list of symbols
    - get_vix_price() → VIX value
    - get_ratios_ttm_bulk() → fundamentals indexed by symbol
    - get_key_metrics_ttm_bulk() → key metrics indexed by symbol
    - get_batch_quotes() → live prices indexed by symbol
    """

    # Curated value-investing universe (40 blue-chip + growth stocks)
    DEFAULT_UNIVERSE: list[str] = [
        # Mega-cap tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # Finance
        "JPM", "BAC", "WFC", "GS", "BRK-B", "V", "MA",
        # Healthcare
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY",
        # Consumer
        "WMT", "PG", "KO", "PEP", "COST", "HD",
        # Industrial / Energy
        "XOM", "CVX", "CAT", "BA", "GE", "UPS",
        # Dividend / Defensive
        "VZ", "T", "SO", "DUK", "CL", "MMM",
    ]

    def __init__(self, *, api_key: str = "", timeout_seconds: int = 20, logger: logging.Logger) -> None:
        self.logger = logger.getChild("yfinance")
        self.api_key = api_key  # Kept for interface compat, not used
        self._yf = yf

    def get_sp500_constituents(self) -> list[str]:
        """Return curated stock universe."""
        return list(self.DEFAULT_UNIVERSE)

    def get_vix_price(self, symbol: str = "^VIX") -> float:
        """Get VIX price via yfinance."""
        # Map our symbol to yfinance ticker
        yf_symbol = "^VIX" if symbol in ("VIXCLS", "^VIX", "VIX") else symbol
        try:
            ticker = self._yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                raise RuntimeError(f"No VIX data returned for {yf_symbol}")
            price = float(hist["Close"].iloc[-1])
            self.logger.info("VIX price: %.2f (via yfinance)", price)
            return price
        except Exception as exc:
            raise RuntimeError(f"Failed to get VIX price: {exc}") from exc

    def get_ratios_ttm_bulk(self) -> dict[str, dict[str, Any]]:
        """Get fundamental ratios for all stocks in universe via yfinance."""
        return self._get_fundamentals(self.DEFAULT_UNIVERSE)

    def get_key_metrics_ttm_bulk(self) -> dict[str, dict[str, Any]]:
        """Returns empty dict — ratios_ttm_bulk already has everything we need."""
        return {}

    def get_batch_quotes(self, symbols: Iterable[str]) -> dict[str, float]:
        """Get live prices for a batch of symbols."""
        quotes: dict[str, float] = {}
        sym_list = sorted({s.strip().upper() for s in symbols if s and s.strip()})
        if not sym_list:
            return quotes
        try:
            tickers = self._yf.Tickers(" ".join(sym_list))
            for sym in sym_list:
                try:
                    ticker = tickers.tickers.get(sym)
                    if ticker is None:
                        continue
                    info = ticker.fast_info
                    price = getattr(info, "last_price", None)
                    if price is None:
                        price = getattr(info, "previous_close", None)
                    if price is not None:
                        quotes[sym] = float(price)
                except Exception as exc:
                    self.logger.debug("Quote failed for %s: %s", sym, exc)
        except Exception as exc:
            self.logger.warning("Batch quote failed: %s. Trying individually.", exc)
            for sym in sym_list:
                try:
                    ticker = self._yf.Ticker(sym)
                    price = getattr(ticker.fast_info, "last_price", None)
                    if price is not None:
                        quotes[sym] = float(price)
                except Exception:
                    continue
        self.logger.info("Got quotes for %d/%d symbols", len(quotes), len(sym_list))
        return quotes

    def _get_fundamentals(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch PE, PB, D/E, FCF for each symbol via yfinance .info dict."""
        result: dict[str, dict[str, Any]] = {}
        for sym in symbols:
            try:
                ticker = self._yf.Ticker(sym)
                info = ticker.info
                if not info or info.get("quoteType") is None:
                    continue
                # Map yfinance keys to our expected format
                entry: dict[str, Any] = {"symbol": sym}
                pe = info.get("trailingPE") or info.get("forwardPE")
                if pe is not None:
                    entry["peRatioTTM"] = float(pe)
                pb = info.get("priceToBook")
                if pb is not None:
                    entry["pbRatioTTM"] = float(pb)
                de = info.get("debtToEquity")
                if de is not None:
                    # yfinance returns D/E as percentage (e.g. 150 = 1.5x)
                    entry["debtToEquityTTM"] = float(de) / 100.0
                fcf = info.get("freeCashflow")
                if fcf is not None:
                    entry["freeCashFlowTTM"] = float(fcf)
                pfcf = info.get("priceToFreeCashflow") or info.get("priceToFreeCashFlows")
                if pfcf is not None:
                    entry["pfcfRatioTTM"] = float(pfcf)
                result[sym] = entry
                self.logger.debug(
                    "%s: PE=%.1f PB=%.1f D/E=%.2f",
                    sym,
                    entry.get("peRatioTTM", 0),
                    entry.get("pbRatioTTM", 0),
                    entry.get("debtToEquityTTM", 0),
                )
            except Exception as exc:
                self.logger.debug("Failed to get fundamentals for %s: %s", sym, exc)
                continue
        self.logger.info("Got fundamentals for %d/%d symbols", len(result), len(symbols))
        return result



class AlternativeMeClient(BaseHttpClient):
    def __init__(
        self,
        *,
        timeout_seconds: int,
        logger: logging.Logger,
        retry_config: HttpRetryConfig | None = None,
    ) -> None:
        super().__init__(
            base_url="https://api.alternative.me",
            timeout_seconds=timeout_seconds,
            logger=logger.getChild("alternative_me"),
            retry_config=retry_config,
        )

    def get_fear_and_greed(self) -> FearGreedReading:
        payload = self._get_json("fng/", {"limit": 1, "format": "json"})
        data = payload["data"][0]
        return FearGreedReading(
            value=int(data["value"]),
            classification=data["value_classification"],
            timestamp=data["timestamp"],
            time_until_update=int(data["time_until_update"]) if data.get("time_until_update") else None,
        )


class BinanceFuturesClient:
    """Crypto funding rate + price client.

    Uses Hyperliquid (free, no auth, no geo-block) for real funding rates,
    and Coinbase for spot prices. Named BinanceFuturesClient for interface
    compatibility with the crypto strategy.
    """

    # Map Binance-style symbols to Hyperliquid asset names
    SYMBOL_TO_HL: dict[str, str] = {
        "BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL",
        "ADAUSDT": "ADA", "DOTUSDT": "DOT", "LINKUSDT": "LINK",
        "AVAXUSDT": "AVAX", "MATICUSDT": "MATIC",
    }
    # Map Binance-style symbols to Coinbase product IDs
    SYMBOL_TO_CB: dict[str, str] = {
        "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
        "ADAUSDT": "ADA-USD", "DOTUSDT": "DOT-USD", "LINKUSDT": "LINK-USD",
        "AVAXUSDT": "AVAX-USD", "MATICUSDT": "MATIC-USD",
    }

    def __init__(
        self,
        *,
        timeout_seconds: int,
        logger: logging.Logger,
        retry_config: HttpRetryConfig | None = None,
    ) -> None:
        self.logger = logger.getChild("funding")
        self.timeout = timeout_seconds
        self.retry_config = retry_config or HttpRetryConfig()
        self._hl_cache: dict[str, dict[str, Any]] | None = None
        self.session = requests.Session()

    def get_mark_price_and_funding(self, symbol: str) -> FundingSnapshot:
        """Get funding rate from Hyperliquid + spot price from Coinbase."""
        funding_rate = self._get_hyperliquid_funding(symbol)
        mark_price = self._get_coinbase_price(symbol)

        self.logger.info(
            "%s: price=$%.2f funding=%.6f (hyperliquid+coinbase)",
            symbol, mark_price, funding_rate,
        )
        return FundingSnapshot(
            symbol=symbol,
            mark_price=mark_price,
            funding_rate=funding_rate,
            next_funding_time=None,
            observed_at=None,
        )

    def _get_hyperliquid_funding(self, symbol: str) -> float:
        """Fetch funding rate from Hyperliquid DEX (free, no auth, US-accessible)."""
        hl_name = self.SYMBOL_TO_HL.get(symbol, symbol.replace("USDT", ""))

        # Cache the full response (has all assets)
        if self._hl_cache is None:
            try:
                response = request_with_retry(
                    self.session,
                    "POST",
                    "https://api.hyperliquid.xyz/info",
                    timeout_seconds=self.timeout,
                    logger=self.logger,
                    retry_config=self.retry_config,
                    json={"type": "metaAndAssetCtxs"},
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                # Build name → context mapping
                meta = data[0]["universe"]
                ctxs = data[1]
                self._hl_cache = {}
                for i, m in enumerate(meta):
                    self._hl_cache[m["name"]] = ctxs[i]
            except Exception as exc:
                self.logger.warning("Hyperliquid funding fetch failed: %s", exc)
                return 0.0

        ctx = self._hl_cache.get(hl_name)
        if ctx is None:
            self.logger.warning("No Hyperliquid data for %s", hl_name)
            return 0.0

        try:
            return float(ctx["funding"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _get_coinbase_price(self, symbol: str) -> float:
        """Get spot price from Coinbase public API."""
        product_id = self.SYMBOL_TO_CB.get(symbol, f"{symbol[:3]}-USD")
        try:
            url = f"https://api.coinbase.com/v2/prices/{product_id}/spot"
            response = request_with_retry(
                self.session,
                "GET",
                url,
                timeout_seconds=self.timeout,
                logger=self.logger,
                retry_config=self.retry_config,
            )
            response.raise_for_status()
            data = response.json()
            return float(data["data"]["amount"])
        except Exception as exc:
            self.logger.error("Coinbase price failed for %s: %s", symbol, exc)
            raise RuntimeError(f"Cannot get price for {symbol}: {exc}") from exc


class CoinbaseClient(BaseHttpClient):
    """Coinbase Advanced Trade API client for spot pricing and order execution.

    Uses public market-data endpoints (no auth needed for prices).
    Keeps Binance-style symbols internally and maps to Coinbase product IDs.
    """

    SYMBOL_MAP: dict[str, str] = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "SOLUSDT": "SOL-USD",
        "ADAUSDT": "ADA-USD",
        "DOTUSDT": "DOT-USD",
        "LINKUSDT": "LINK-USD",
        "AVAXUSDT": "AVAX-USD",
        "MATICUSDT": "MATIC-USD",
    }

    REVERSE_MAP: dict[str, str] = {v: k for k, v in SYMBOL_MAP.items()}

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        timeout_seconds: int,
        logger: logging.Logger,
        retry_config: HttpRetryConfig | None = None,
    ) -> None:
        super().__init__(
            base_url="https://api.coinbase.com",
            timeout_seconds=timeout_seconds,
            logger=logger.getChild("coinbase"),
            retry_config=retry_config,
        )
        self.api_key = api_key
        self.api_secret = api_secret

    def to_coinbase_symbol(self, binance_symbol: str) -> str:
        cb = self.SYMBOL_MAP.get(binance_symbol)
        if cb is None:
            raise ValueError(f"No Coinbase mapping for Binance symbol {binance_symbol}")
        return cb

    def to_binance_symbol(self, coinbase_product: str) -> str:
        bn = self.REVERSE_MAP.get(coinbase_product)
        if bn is None:
            raise ValueError(f"No Binance mapping for Coinbase product {coinbase_product}")
        return bn

    def get_spot_price(self, binance_symbol: str) -> float:
        product_id = self.to_coinbase_symbol(binance_symbol)
        payload = self._get_json(f"api/v3/brokerage/market/products/{product_id}")
        price = _parse_float(payload.get("price"))
        if price is None:
            raise RuntimeError(f"Unable to parse Coinbase spot price for {product_id}")
        return price

    def get_spot_prices(self, binance_symbols: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for sym in binance_symbols:
            try:
                prices[sym] = self.get_spot_price(sym)
            except Exception as exc:
                self.logger.warning("Failed to get Coinbase price for %s: %s", sym, exc)
        return prices

    def get_spot_price_by_product_id(self, product_id: str) -> float:
        """Get spot price using a Coinbase product ID directly (e.g. 'BTC-USD')."""
        payload = self._get_json(f"api/v3/brokerage/market/products/{product_id}")
        price = _parse_float(payload.get("price"))
        if price is None:
            raise RuntimeError(f"Unable to parse Coinbase spot price for {product_id}")
        return price

    def get_spot_prices_by_product_ids(self, product_ids: list[str]) -> dict[str, float]:
        """Fetch prices for a list of Coinbase product IDs (e.g. 'BTC-USD', 'ETH-USD').

        Returns a dict keyed by product_id.  Missing or failed symbols are omitted.
        """
        prices: dict[str, float] = {}
        for pid in product_ids:
            try:
                prices[pid] = self.get_spot_price_by_product_id(pid)
            except Exception as exc:
                self.logger.warning("Failed to get Coinbase price for %s: %s", pid, exc)
        return prices

    def get_all_tradeable_products(self) -> list[dict[str, Any]]:
        """Fetch all active SPOT USD trading pairs from Coinbase."""
        payload = self._get_json("api/v3/brokerage/market/products", {"product_type": "SPOT"})
        products = payload.get("products", [])
        return [
            p for p in products
            if p.get("quote_currency_id") == "USD"
            and p.get("status") == "online"
        ]

    def get_all_spot_prices(self) -> dict[str, float]:
        """Fetch prices for all tradeable USD pairs. Returns {product_id: price}."""
        products = self.get_all_tradeable_products()
        prices: dict[str, float] = {}
        for p in products:
            price = _parse_float(p.get("price"))
            product_id = p.get("product_id", "")
            if price and price > 0 and product_id:
                prices[product_id] = price
        return prices

    def get_prediction_markets(self) -> list[dict[str, Any]]:
        """Fetch all active Coinbase prediction/event markets.

        Tries multiple product types and prefixes since Coinbase event
        contracts may use 'EVENT' or 'FUTURE' product type, and IDs
        may start with 'KX', 'KXBTC', or other patterns.
        """
        all_markets: list[dict[str, Any]] = []
        # Try different product types that may contain prediction markets
        for ptype in ("EVENT", "FUTURE", None):
            try:
                params: dict[str, Any] = {}
                if ptype:
                    params["product_type"] = ptype
                payload = self._get_json("api/v3/brokerage/market/products", params)
                products = payload.get("products", [])
                for p in products:
                    pid = p.get("product_id", "")
                    status = p.get("status", "").lower()
                    # Match KX-prefix OR any event/prediction-type product
                    if pid.startswith("KX") or p.get("product_type", "") in ("EVENT", "PREDICTION"):
                        if status in ("online", "active", ""):
                            all_markets.append(p)
            except Exception as exc:
                self.logger.debug("Prediction market fetch (type=%s) failed: %s", ptype, exc)

        # Deduplicate by product_id
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for m in all_markets:
            pid = m.get("product_id", "")
            if pid not in seen:
                seen.add(pid)
                unique.append(m)

        self.logger.info(
            "Prediction market discovery: found %d markets. IDs: %s",
            len(unique),
            [m.get("product_id") for m in unique[:10]],
        )
        return unique

    def get_candles(
        self, binance_symbol: str, granularity: str = "ONE_HOUR", limit: int = 24
    ) -> list[dict[str, Any]]:
        import time as _time

        product_id = self.to_coinbase_symbol(binance_symbol)
        end = int(_time.time())
        gran_seconds = {
            "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600, "SIX_HOUR": 21600, "ONE_DAY": 86400,
        }
        interval = gran_seconds.get(granularity, 3600)
        start = end - (interval * limit)
        payload = self._get_json(
            f"api/v3/brokerage/market/products/{product_id}/candles",
            {"start": str(start), "end": str(end), "granularity": granularity},
        )
        return payload.get("candles", []) if isinstance(payload, dict) else payload


class FreeMarketIntelligenceClient:
    """Free market intelligence client powered by yfinance (no API key required).

    Provides options sentiment, insider transaction activity, and short
    interest data to enrich AI stock evaluation prompts.  All data comes
    from Yahoo Finance via the yfinance library which is already a project
    dependency.  Every sub-call degrades gracefully so get_stock_signals()
    never raises.
    """

    def __init__(self, *, logger: logging.Logger) -> None:
        self.logger = logger.getChild("market_intel")
        self._yf = yf

    # ── Public API ────────────────────────────────────────────────────

    def get_stock_signals(self, ticker: str) -> dict[str, Any]:
        """Fetch and summarize free market intelligence signals for a ticker.

        Returns a structured dict safe to serialize and pass into an AI prompt.
        Keys: options_flow, insider, short_interest.
        All values default to neutral/zero on error.
        """
        # Create one Ticker instance so all helpers share the same cached session.
        t = self._yf.Ticker(ticker)
        signals = {
            "options_flow": self._get_options_summary(t),
            "insider": self._get_insider_summary(t),
            "short_interest": self._get_short_interest(t),
        }
        self.logger.debug(
            "%s market signals: options=%s insider=%s short=%s",
            ticker,
            signals["options_flow"]["net_sentiment"],
            signals["insider"]["net_sentiment"],
            signals["short_interest"]["net_sentiment"],
        )
        return signals

    # ── Private helpers ───────────────────────────────────────────────

    def _get_options_summary(self, t: yf.Ticker) -> dict[str, Any]:
        """Put/call volume and open interest from the nearest options expiry."""
        try:
            expirations = t.options
            if not expirations:
                return self._empty_options()
            chain = t.option_chain(expirations[0])
            calls_vol = int(chain.calls["volume"].fillna(0).sum())
            puts_vol = int(chain.puts["volume"].fillna(0).sum())
            calls_oi = int(chain.calls["openInterest"].fillna(0).sum())
            puts_oi = int(chain.puts["openInterest"].fillna(0).sum())
            ratio = round(puts_vol / calls_vol, 2) if calls_vol > 0 else None
            return {
                "calls_volume": calls_vol,
                "puts_volume": puts_vol,
                "calls_open_interest": calls_oi,
                "puts_open_interest": puts_oi,
                "put_call_volume_ratio": ratio,
                "net_sentiment": (
                    "bullish" if calls_vol > puts_vol
                    else "bearish" if puts_vol > calls_vol
                    else "neutral"
                ),
            }
        except Exception as exc:
            self.logger.warning("Options summary failed for %s: %s", t.ticker, exc)
            return self._empty_options()

    def _get_insider_summary(self, t: yf.Ticker) -> dict[str, Any]:
        """Recent insider buy/sell counts from SEC Form 4 disclosures via yfinance."""
        try:
            txns = t.insider_transactions
            if txns is None or (hasattr(txns, "empty") and txns.empty):
                return self._empty_insider()
            # Identify the transaction-type column (varies slightly by yfinance version)
            txn_col = next(
                (c for c in txns.columns if "transaction" in c.lower()), None
            )
            if txn_col is None:
                return self._empty_insider()
            txn_text = txns[txn_col].fillna("").str.lower()
            buys = int((txn_text.str.contains("buy|purchase")).sum())
            sells = int((txn_text.str.contains("sell|sale")).sum())
            return {
                "recent_buys": buys,
                "recent_sells": sells,
                "net_sentiment": (
                    "bullish" if buys > sells
                    else "bearish" if sells > buys
                    else "neutral"
                ),
            }
        except Exception as exc:
            self.logger.warning("Insider summary failed for %s: %s", t.ticker, exc)
            return self._empty_insider()

    def _get_short_interest(self, t: yf.Ticker) -> dict[str, Any]:
        """Short percentage of float and days-to-cover from yfinance ticker.info."""
        try:
            info = t.info
            short_pct_raw = _parse_float(info.get("shortPercentOfFloat"))
            short_ratio_raw = _parse_float(info.get("shortRatio"))
            short_pct = round(short_pct_raw * 100, 2) if short_pct_raw is not None else None
            days_to_cover = round(short_ratio_raw, 1) if short_ratio_raw is not None else None
            # Only classify when data is available; unknown → neutral
            if short_pct is None:
                net_sentiment = "neutral"
            elif short_pct > 10.0:
                net_sentiment = "bearish"
            else:
                net_sentiment = "bullish"
            return {
                "short_percent_of_float": short_pct,
                "days_to_cover": days_to_cover,
                "net_sentiment": net_sentiment,
            }
        except Exception as exc:
            self.logger.warning("Short interest failed for %s: %s", t.ticker, exc)
            return {"short_percent_of_float": None, "days_to_cover": None, "net_sentiment": "neutral"}

    @staticmethod
    def _empty_options() -> dict[str, Any]:
        return {
            "calls_volume": 0, "puts_volume": 0,
            "calls_open_interest": 0, "puts_open_interest": 0,
            "put_call_volume_ratio": None, "net_sentiment": "neutral",
        }

    @staticmethod
    def _empty_insider() -> dict[str, Any]:
        return {"recent_buys": 0, "recent_sells": 0, "net_sentiment": "neutral"}


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
