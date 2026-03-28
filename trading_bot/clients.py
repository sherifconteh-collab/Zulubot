from __future__ import annotations

import logging
from typing import Any, Iterable

import requests

from trading_bot.models import FearGreedReading, FundingSnapshot


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class BaseHttpClient:
    def __init__(self, *, base_url: str, timeout_seconds: int, logger: logging.Logger) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "codex-trading-bot/1.0"})

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        self.logger.debug("GET %s params=%s", url, params)
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
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
        import yfinance as yf
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
    def __init__(self, *, timeout_seconds: int, logger: logging.Logger) -> None:
        super().__init__(
            base_url="https://api.alternative.me",
            timeout_seconds=timeout_seconds,
            logger=logger.getChild("alternative_me"),
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

    def __init__(self, *, timeout_seconds: int, logger: logging.Logger) -> None:
        self.logger = logger.getChild("funding")
        self.timeout = timeout_seconds
        self._hl_cache: dict[str, dict[str, Any]] | None = None

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
        import json as _json, urllib.request

        hl_name = self.SYMBOL_TO_HL.get(symbol, symbol.replace("USDT", ""))

        # Cache the full response (has all assets)
        if self._hl_cache is None:
            try:
                req = urllib.request.Request(
                    "https://api.hyperliquid.xyz/info",
                    data=_json.dumps({"type": "metaAndAssetCtxs"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = _json.loads(resp.read())
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
        import json as _json, urllib.request

        product_id = self.SYMBOL_TO_CB.get(symbol, f"{symbol[:3]}-USD")
        try:
            url = f"https://api.coinbase.com/v2/prices/{product_id}/spot"
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                data = _json.loads(resp.read())
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
    ) -> None:
        super().__init__(
            base_url="https://api.coinbase.com",
            timeout_seconds=timeout_seconds,
            logger=logger.getChild("coinbase"),
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


class UnusualWhalesClient(BaseHttpClient):
    """Unusual Whales market intelligence client.

    Provides options flow alerts, dark pool transactions, insider trades,
    and congressional trading data for a given ticker.

    Requires a paid Unusual Whales API key (set UNUSUAL_WHALES_API_KEY).
    All methods return empty lists on error so callers degrade gracefully.
    """

    def __init__(self, *, api_key: str, timeout_seconds: int, logger: logging.Logger) -> None:
        super().__init__(
            base_url="https://api.unusualwhales.com",
            timeout_seconds=timeout_seconds,
            logger=logger.getChild("unusual_whales"),
        )
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def get_options_flow(
        self, ticker: str, *, min_premium: float = 50_000, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Recent unusual options flow alerts for a ticker."""
        try:
            payload = self._get_json(
                "api/option-trades/flow-alerts",
                {"ticker_symbol": ticker, "min_premium": int(min_premium), "limit": limit},
            )
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            return data if isinstance(data, list) else []
        except Exception as exc:
            self.logger.warning("Options flow failed for %s: %s", ticker, exc)
            return []

    def get_darkpool(self, ticker: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Recent dark pool transactions for a ticker."""
        try:
            payload = self._get_json(f"api/darkpool/{ticker}", {"limit": limit})
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            return data if isinstance(data, list) else []
        except Exception as exc:
            self.logger.warning("Dark pool failed for %s: %s", ticker, exc)
            return []

    def get_insider_trades(self, ticker: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Recent insider (executive/director) trades for a ticker."""
        try:
            payload = self._get_json(f"api/insider/{ticker}", {"limit": limit})
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            return data if isinstance(data, list) else []
        except Exception as exc:
            self.logger.warning("Insider trades failed for %s: %s", ticker, exc)
            return []

    def get_congress_trades(self, ticker: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Recent congressional trading disclosures for a ticker."""
        try:
            payload = self._get_json(
                "api/congress/recent-trades", {"ticker": ticker, "limit": limit}
            )
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            return data if isinstance(data, list) else []
        except Exception as exc:
            self.logger.warning("Congress trades failed for %s: %s", ticker, exc)
            return []

    def get_stock_signals(self, ticker: str) -> dict[str, Any]:
        """Fetch and summarize all Unusual Whales signals for a ticker.

        Returns a structured dict safe to serialize and pass into an AI prompt.
        All sub-calls degrade to empty lists on failure so this never raises.
        """
        flow = self.get_options_flow(ticker, min_premium=50_000, limit=20)
        darkpool = self.get_darkpool(ticker, limit=20)
        insider = self.get_insider_trades(ticker, limit=10)
        congress = self.get_congress_trades(ticker, limit=10)

        # Options flow summary
        calls = sum(1 for f in flow if str(f.get("put_call", "")).upper() == "C")
        puts = sum(1 for f in flow if str(f.get("put_call", "")).upper() == "P")
        sweeps = sum(1 for f in flow if f.get("is_sweep"))
        flow_premium = sum(float(f.get("premium", 0) or 0) for f in flow)

        # Dark pool summary
        dp_volume = sum(float(t.get("size", 0) or 0) for t in darkpool)
        dp_premium = sum(float(t.get("premium", 0) or 0) for t in darkpool)

        # Insider summary (P=purchase, S=sale)
        insider_buys = sum(
            1 for t in insider if str(t.get("transaction_code", "")).upper() == "P"
        )
        insider_sells = sum(
            1 for t in insider if str(t.get("transaction_code", "")).upper() == "S"
        )

        # Congressional summary
        congress_buys = sum(
            1 for t in congress
            if str(t.get("transaction_type", "")).lower() in ("buy", "purchase")
        )
        congress_sells = sum(
            1 for t in congress
            if str(t.get("transaction_type", "")).lower() in ("sell", "sale")
        )

        self.logger.debug(
            "%s UW signals: flow=%d(C%d/P%d) dp=%d insider=%d/%d congress=%d/%d",
            ticker, len(flow), calls, puts, len(darkpool),
            insider_buys, insider_sells, congress_buys, congress_sells,
        )

        return {
            "options_flow": {
                "calls": calls,
                "puts": puts,
                "sweeps": sweeps,
                "total_premium_usd": round(flow_premium, 2),
                "flow_count": len(flow),
                "bullish_lean": calls > puts,
            },
            "darkpool": {
                "recent_trades": len(darkpool),
                "total_volume_shares": round(dp_volume, 0),
                "total_premium_usd": round(dp_premium, 2),
            },
            "insider": {
                "recent_buys": insider_buys,
                "recent_sells": insider_sells,
                "net_sentiment": (
                    "bullish" if insider_buys > insider_sells
                    else "bearish" if insider_sells > insider_buys
                    else "neutral"
                ),
            },
            "congress": {
                "recent_buys": congress_buys,
                "recent_sells": congress_sells,
                "net_sentiment": (
                    "bullish" if congress_buys > congress_sells
                    else "bearish" if congress_sells > congress_buys
                    else "neutral"
                ),
            },
        }


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
