"""Prediction market strategy using Kalshi API.

Kalshi powers the prediction markets visible in the Coinbase app, but
Coinbase doesn't expose them through their trading API.  This strategy
connects directly to Kalshi for market discovery and order execution.

Market discovery (public, no auth) always works.
Order placement requires Kalshi API credentials (KALSHI_KEY_ID + KALSHI_PRIVATE_KEY).

Signal logic:
  BUY YES when mid_price < buy_yes_threshold (default 0.30)
  BUY NO  when mid_price > buy_no_threshold  (default 0.70)
  SELL    when price crosses take_profit_threshold (0.50) toward our side
  SELL    when price moves against by stop_loss_pct (0.15)

Cross-reference:
  For crypto-related predictions, Fear & Greed is used as an optional signal boost.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients_kalshi import KalshiClient
from trading_bot.config import PredictionConfig
from trading_bot.models import PortfolioState
from trading_bot.risk import RiskManager

try:
    from trading_bot.integrations.trading_agents import TradingAgentsService, apply_ai_confirmation
except ImportError:
    TradingAgentsService = None  # type: ignore[assignment,misc]
    apply_ai_confirmation = None  # type: ignore[assignment]


# Series tickers to scan on Kalshi
CRYPTO_SERIES = ["KXBTC", "KXETH", "KXSOL", "KXDOGE", "KXXRP"]
ECONOMICS_SERIES = ["KXCPI", "KXCPIYOY", "KXGDP", "KXFED", "KXFOMC", "KXNFP", "KXRECESSION", "KXTARIFF"]
FINANCE_SERIES = ["KXINX", "KXSPY", "KXNDX", "KXIWM", "KXGLD", "KXWTI"]
POLITICS_SERIES = ["KXTRUMP", "KXPOTUS", "KXSENATE", "KXHOUSE"]
WEATHER_SERIES = ["KXTEMP", "KXHURRICANE"]
OTHER_SERIES = ["KXNEWPOPE"]

# Minimum 24h volume (in dollars) to consider a market tradeable —
# avoids illiquid contracts with wide spreads
MIN_VOLUME_USD = 5_000

SERIES_BY_CATEGORY: dict[str, list[str]] = {
    "crypto": CRYPTO_SERIES,
    "economics": ECONOMICS_SERIES,
    "finance": FINANCE_SERIES,
    "politics": POLITICS_SERIES,
    "weather": WEATHER_SERIES,
    "other": OTHER_SERIES,
}


class CoinbasePredictionStrategy:
    """Prediction market strategy using Kalshi for discovery and execution."""

    def __init__(
        self,
        *,
        config: PredictionConfig,
        kalshi_client: KalshiClient,
        logger: logging.Logger,
        ai_service: Any | None = None,
    ) -> None:
        self.config = config
        self.kalshi = kalshi_client
        self.logger = logger.getChild("strategy.predictions")
        self.ai_service = ai_service

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
        fear_greed_value: int | None = None,
    ) -> dict[str, Any]:
        """Execute one prediction-market strategy cycle."""
        if not self.config.enabled:
            self.logger.debug("Prediction strategy is disabled.")
            return {"status": "disabled", "buys": [], "sells": [], "markets_scanned": 0}

        result: dict[str, Any] = {
            "status": "ok",
            "buys": [],
            "sells": [],
            "markets_scanned": 0,
            "markets_discovered": 0,
            "resolved_closed": [],
            "fear_greed": fear_greed_value,
            "can_trade": self.kalshi.can_trade,
        }

        try:
            # Step 1: Handle resolved/expired positions
            self._close_resolved_positions(state, broker, result)

            # Step 2: Manage existing positions (stop-loss / take-profit)
            self._manage_existing_positions(state, broker, result)

            # Step 3: Discover available prediction markets from Kalshi
            markets = self._discover_markets()
            result["markets_discovered"] = len(markets)

            if not markets:
                self.logger.info("No prediction markets found on Kalshi.")
                state.last_predictions_scan_date = now.date().isoformat()
                return result

            result["markets_scanned"] = len(markets)

            # Step 4: Scan for new entry signals
            if not state.kill_switch_engaged and not state.halt_new_trades_for_day:
                self._scan_for_entries(
                    state, markets, broker, risk_manager, now, fear_greed_value, result,
                )

            state.last_predictions_scan_date = now.date().isoformat()

        except Exception as exc:
            self.logger.error("Prediction strategy failed: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)

        self.logger.info(
            "Predictions cycle complete. scanned=%d discovered=%d buys=%s sells=%s "
            "resolved=%s can_trade=%s",
            result["markets_scanned"],
            result["markets_discovered"],
            result["buys"],
            result["sells"],
            result["resolved_closed"],
            result["can_trade"],
        )
        return result

    # ------------------------------------------------------------------
    # Market discovery (public Kalshi API, no auth needed)
    # ------------------------------------------------------------------

    def _discover_markets(self) -> list[dict[str, Any]]:
        """Fetch open markets from Kalshi for our configured categories."""
        import time

        series_to_scan: list[str] = []
        for cat in self.config.market_categories:
            series_to_scan.extend(SERIES_BY_CATEGORY.get(cat, []))
        if not series_to_scan:
            # Default: scan all known series
            for s_list in SERIES_BY_CATEGORY.values():
                series_to_scan.extend(s_list)

        all_markets: list[dict[str, Any]] = []
        for series in series_to_scan:
            try:
                markets, _ = self.kalshi.get_markets(
                    series_ticker=series, status="open", limit=20,
                )
                for m in markets:
                    m["_series"] = series
                    m["_category"] = self._series_to_category(series)
                    # Parse dollar prices
                    m["_yes_bid"] = _parse_dollars(m.get("yes_bid_dollars"))
                    m["_yes_ask"] = _parse_dollars(m.get("yes_ask_dollars"))
                    m["_no_bid"] = _parse_dollars(m.get("no_bid_dollars"))
                    m["_no_ask"] = _parse_dollars(m.get("no_ask_dollars"))
                    m["_last"] = _parse_dollars(m.get("last_price_dollars"))
                    m["_volume"] = _parse_dollars(m.get("volume_fp"))
                    m["_mid"] = self._mid_price(m)
                    # Only include markets with price data and sufficient volume
                    has_price = m["_yes_ask"] > 0.01 or m["_yes_bid"] > 0
                    has_volume = m["_volume"] >= MIN_VOLUME_USD
                    if has_price and has_volume:
                        all_markets.append(m)
                time.sleep(0.3)  # Rate limit
            except Exception as exc:
                self.logger.debug("Failed to fetch series %s: %s", series, exc)

        # Sort by volume (most active first)
        all_markets.sort(key=lambda m: m["_volume"], reverse=True)

        self.logger.info(
            "Discovered %d active prediction markets from %d series.",
            len(all_markets), len(series_to_scan),
        )
        return all_markets

    @staticmethod
    def _series_to_category(series: str) -> str:
        for cat, series_list in SERIES_BY_CATEGORY.items():
            if series in series_list:
                return cat
        return "other"

    @staticmethod
    def _mid_price(m: dict[str, Any]) -> float:
        bid = m["_yes_bid"]
        ask = m["_yes_ask"]
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return ask or bid or m["_last"]

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _get_prediction_positions(self, state: PortfolioState) -> dict[str, Any]:
        return {
            symbol: pos
            for symbol, pos in state.positions.items()
            if pos.asset_type == "prediction" and pos.strategy == "coinbase_predictions"
        }

    def _close_resolved_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        result: dict[str, Any],
    ) -> None:
        """Detect resolved/expired contracts and close them."""
        positions = self._get_prediction_positions(state)
        if not positions:
            return

        for symbol, position in list(positions.items()):
            ticker = position.meta.get("ticker", "")
            if not ticker:
                continue

            try:
                market = self.kalshi.get_market(ticker)
                status = market.get("status", "").lower()

                if status in ("settled", "finalized", "closed"):
                    result_val = market.get("result", "").lower()
                    side = position.meta.get("side", "yes")
                    if result_val == side:
                        close_price = 1.0
                    elif result_val:
                        close_price = 0.0
                    else:
                        close_price = position.current_price

                    self.logger.info(
                        "Market %s resolved: result=%s side=%s close_price=%.2f",
                        ticker, result_val, side, close_price,
                    )

                    broker.close_position(
                        state, symbol=symbol, price=close_price,
                        reason=f"resolved:{result_val or 'expired'}",
                    )
                    result["resolved_closed"].append(symbol)

            except Exception as exc:
                self.logger.warning("Failed to check resolution for %s: %s", ticker, exc)

    def _manage_existing_positions(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        result: dict[str, Any],
    ) -> None:
        """Apply take-profit and stop-loss rules to existing positions."""
        positions = self._get_prediction_positions(state)
        if not positions:
            return

        for symbol, position in list(positions.items()):
            ticker = position.meta.get("ticker", "")
            if not ticker:
                continue

            try:
                market = self.kalshi.get_market(ticker)
                yes_bid = _parse_dollars(market.get("yes_bid_dollars"))
                yes_ask = _parse_dollars(market.get("yes_ask_dollars"))
                current_price = (yes_bid + yes_ask) / 2 if yes_bid > 0 and yes_ask > 0 else (yes_bid or yes_ask)

                if current_price <= 0:
                    continue

                position.current_price = current_price
                state.last_prices[symbol] = current_price

                side = position.meta.get("side", "yes")
                entry_price = position.entry_price

                # Take profit
                if self._should_take_profit(side, entry_price, current_price):
                    self._close_position(state, broker, symbol, position, current_price, "take_profit", result)
                    continue

                # Stop loss
                if self._should_stop_loss(side, entry_price, current_price):
                    self._close_position(state, broker, symbol, position, current_price, "stop_loss", result)

            except Exception as exc:
                self.logger.warning("Failed to manage position %s: %s", symbol, exc)

    def _close_position(
        self,
        state: PortfolioState,
        broker: PaperBroker,
        symbol: str,
        position: Any,
        price: float,
        reason: str,
        result: dict[str, Any],
    ) -> None:
        """Close a prediction position (live on Kalshi + paper bookkeeping)."""
        ticker = position.meta.get("ticker", "")
        side = position.meta.get("side", "yes")

        # Sell on Kalshi if we can trade
        if self.kalshi.can_trade and ticker:
            try:
                self.kalshi.place_order(
                    ticker=ticker,
                    side=side,
                    action="sell",
                    count=int(position.quantity),
                )
            except Exception as exc:
                self.logger.warning("Kalshi sell failed for %s: %s", ticker, exc)

        closed = broker.close_position(state, symbol=symbol, price=price, reason=reason)
        if closed is not None:
            result["sells"].append(symbol)
            self.logger.info(
                "%s %s: side=%s entry=%.4f exit=%.4f",
                reason, symbol, side, position.entry_price, price,
            )

    def _should_take_profit(self, side: str, entry: float, current: float) -> bool:
        threshold = self.config.take_profit_threshold
        if side == "yes":
            return entry < threshold and current >= threshold
        else:
            no_current = 1.0 - current
            return entry < threshold and no_current >= threshold

    def _should_stop_loss(self, side: str, entry: float, current: float) -> bool:
        max_loss = self.config.stop_loss_pct
        if side == "yes":
            return (entry - current) >= max_loss
        else:
            no_current = 1.0 - current
            return (entry - no_current) >= max_loss

    # ------------------------------------------------------------------
    # Entry scanning
    # ------------------------------------------------------------------

    def _scan_for_entries(
        self,
        state: PortfolioState,
        markets: list[dict[str, Any]],
        broker: PaperBroker,
        risk_manager: RiskManager,
        now: datetime,
        fear_greed_value: int | None,
        result: dict[str, Any],
    ) -> None:
        existing_count = len(self._get_prediction_positions(state))
        if existing_count >= self.config.max_positions:
            self.logger.info("At max prediction positions (%d/%d).", existing_count, self.config.max_positions)
            return

        slots_available = self.config.max_positions - existing_count
        equity = broker.portfolio_equity(state)

        for market in markets:
            if slots_available <= 0:
                break

            ticker = market.get("ticker", "")
            if not ticker:
                continue

            # Skip if already holding
            yes_symbol = f"KX:{ticker}:yes"
            no_symbol = f"KX:{ticker}:no"
            if yes_symbol in state.positions or no_symbol in state.positions:
                continue

            mid = market["_mid"]
            if mid <= 0 or mid >= 1.0:
                continue

            # Evaluate signal
            signal = self._evaluate_signal(market, mid, fear_greed_value)
            if signal is None:
                continue

            side, price = signal

            # Position sizing
            prediction_budget = equity * self.config.capital_allocation_pct
            existing_value = sum(
                pos.market_value for pos in state.positions.values()
                if pos.strategy == "coinbase_predictions"
            )
            remaining = max(0.0, prediction_budget - existing_value)
            per_position = remaining / max(1, slots_available)
            notional = min(per_position, state.cash * 0.95)
            notional = max(notional, 0.0)

            # AI probability estimation — adjusts position size via boost/require mode
            ai_meta: dict[str, Any] = {}
            if self.ai_service is not None and apply_ai_confirmation is not None:
                try:
                    verdict = self.ai_service.evaluate_kalshi_market(
                        ticker=ticker,
                        title=market.get("title", market.get("subtitle", "")),
                        mid_price=mid,
                        category=market.get("_category", "other"),
                        fear_greed_value=fear_greed_value,
                    )
                    if verdict is not None:
                        notional, ai_meta = apply_ai_confirmation(
                            verdict=verdict,
                            confirmation_mode="boost",
                            base_notional=notional,
                            logger=self.logger,
                        )
                except Exception as exc:
                    self.logger.debug("AI prediction evaluation failed (non-blocking): %s", exc)

            if notional < self.config.min_trade_usd:
                self.logger.info("Insufficient funds for %s (notional=$%.2f).", ticker, notional)
                break

            quantity = int(notional / price) if price > 0 else 0
            if quantity <= 0:
                continue

            actual_notional = quantity * price
            symbol = f"KX:{ticker}:{side}"

            # Execute on Kalshi (live)
            if self.kalshi.can_trade:
                try:
                    yes_price_cents = int(price * 100) if side == "yes" else None
                    order = self.kalshi.place_order(
                        ticker=ticker,
                        side=side,
                        action="buy",
                        count=quantity,
                        yes_price=yes_price_cents,
                    )
                    self.logger.info("Kalshi order placed: %s", order)
                except Exception as exc:
                    self.logger.warning("Kalshi buy failed for %s: %s", ticker, exc)
                    continue
            else:
                self.logger.info(
                    "Paper-only: would buy %s %s x%d @ $%.2f (no Kalshi credentials)",
                    side, ticker, quantity, price,
                )

            # Bookkeep
            position = broker.open_position(
                state,
                symbol=symbol,
                asset_type="prediction",
                strategy="coinbase_predictions",
                price=price,
                notional=actual_notional,
                opened_at=now,
                meta={
                    "ticker": ticker,
                    "side": side,
                    "entry_price": price,
                    "category": market.get("_category", "other"),
                    "fear_greed": fear_greed_value,
                    "title": market.get("title", ""),
                    "subtitle": market.get("subtitle", ""),
                    **ai_meta,
                },
            )

            if position is not None:
                result["buys"].append(symbol)
                slots_available -= 1
                self.logger.info(
                    "Prediction entry: %s side=%s price=$%.2f qty=%d notional=$%.2f | %s",
                    symbol, side, price, quantity, actual_notional,
                    market.get("subtitle", market.get("title", "")),
                )

    def _evaluate_signal(
        self,
        market: dict[str, Any],
        mid: float,
        fear_greed_value: int | None,
    ) -> tuple[str, float] | None:
        """Evaluate whether a market has a buy signal.

        Returns (side, price) if signal found, None otherwise.
        """
        category = market.get("_category", "other")
        yes_ask = market["_yes_ask"]
        no_ask = market["_no_ask"]
        yes_bid = market["_yes_bid"]

        # Reject markets with wide bid-ask spreads (>15 cents) — poor liquidity
        if yes_ask > 0 and yes_bid > 0 and (yes_ask - yes_bid) > 0.15:
            return None

        # Crypto markets: require Fear & Greed confirmation
        if category == "crypto" and fear_greed_value is not None:
            if mid < self.config.buy_yes_threshold:
                if fear_greed_value <= 30:
                    return ("yes", yes_ask if yes_ask > 0 else mid)
            elif mid > self.config.buy_no_threshold:
                if fear_greed_value >= 70:
                    return ("no", no_ask if no_ask > 0 else (1.0 - mid))
            return None

        # Finance/economics/politics/weather: pure threshold signals
        if mid < self.config.buy_yes_threshold:
            return ("yes", yes_ask if yes_ask > 0 else mid)
        elif mid > self.config.buy_no_threshold:
            return ("no", no_ask if no_ask > 0 else (1.0 - mid))

        return None


def _parse_dollars(value: Any) -> float:
    """Parse a dollar-string field like '0.0100' to float."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
