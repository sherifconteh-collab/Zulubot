"""Mean-reversion altcoin rotation strategy.

Holds one coin at a time on Coinbase and swaps when price ratios shift
enough to overcome trading fees.  Inspired by binance-trade-bot but
adapted for Coinbase Advanced Trade.

All swaps route through USD: sell current coin -> USD -> buy target coin.
State (current coin, baseline ratios, best quantities) is persisted in
``PortfolioState.rotation_data``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from trading_bot.broker import PaperBroker
from trading_bot.clients import CoinbaseClient
from trading_bot.coinbase_broker import CoinbaseBroker
from trading_bot.config import AltcoinRotationConfig
from trading_bot.models import PortfolioState

try:
    from trading_bot.integrations.trading_agents import TradingAgentsService
except ImportError:
    TradingAgentsService = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUST_USD = 0.50  # anything under 50 cents is "dust"


def _coin_from_product(product_id: str) -> str:
    """``BTC-USD`` -> ``BTC``."""
    return product_id.split("-")[0]


def _product_from_coin(coin: str) -> str:
    """``BTC`` -> ``BTC-USD``."""
    return f"{coin}-USD"


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class AltcoinRotationStrategy:
    """One-coin-at-a-time mean-reversion rotation on Coinbase.

    Parameters
    ----------
    config:
        An :class:`AltcoinRotationConfig` instance.
    coinbase_client:
        Used for fetching spot prices (``get_all_spot_prices``).
    coinbase_broker:
        Used for placing real market orders and querying holdings.
    logger:
        Parent logger; a child ``altcoin_rotation`` logger is derived.
    """

    def __init__(
        self,
        *,
        config: AltcoinRotationConfig,
        coinbase_client: CoinbaseClient,
        coinbase_broker: CoinbaseBroker,
        logger: logging.Logger,
        ai_service: Any | None = None,
        confirmation_mode: str = "boost",
        protected_symbols: frozenset[str] = frozenset(),
    ) -> None:
        self.config = config
        self.coinbase_client = coinbase_client
        self.coinbase_broker = coinbase_broker
        self.protected_symbols = protected_symbols
        self.logger = logger.getChild("altcoin_rotation")
        self.ai_service = ai_service
        # Normalize to lowercase so "Require"/"REQUIRE" work identically to "require"
        self.confirmation_mode = confirmation_mode.strip().lower()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        state: PortfolioState,
        *,
        broker: PaperBroker,
        risk_manager: Any = None,
        now: datetime,
    ) -> dict[str, Any]:
        """Execute one scouting cycle.

        Returns a summary dict suitable for logging / Telegram.
        """
        if not self.config.enabled:
            self.logger.info("Altcoin rotation is disabled; skipping.")
            return {"status": "disabled"}

        # Ensure rotation_data bucket exists on the state object.
        rd = self._get_rotation_data(state)
        rd["cycle_count"] = rd.get("cycle_count", 0) + 1

        # 1. Fetch all Coinbase spot prices.
        try:
            all_prices = self.coinbase_client.get_all_spot_prices()  # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.error("Failed to fetch spot prices: %s", exc)
            return {"status": "error", "reason": str(exc)}

        if not all_prices:
            self.logger.warning("Price feed returned zero prices; aborting cycle.")
            return {"status": "error", "reason": "no_prices"}

        # Filter out excluded coins and cap the list.
        tracked = self._filter_coins(all_prices)
        self.logger.info(
            "Tracking %d coins (excluded %d, capped at %d).",
            len(tracked),
            len(all_prices) - len(tracked),
            self.config.max_coins_to_track,
        )

        # 2. Sync holdings from Coinbase (first-run bootstrap or drift check).
        current_coin = self._sync_holdings(state, rd, tracked, broker=broker, now=now)

        if current_coin is None:
            # No crypto holdings — try to seed from available USD/USDC cash.
            seed_result = self._seed_from_cash(state, rd, broker, tracked, now)
            if seed_result is not None:
                current_coin = seed_result
            else:
                self.logger.warning(
                    "No current rotation coin and no cash to seed. Waiting for "
                    "a deposit or recurring buy to appear on Coinbase."
                )
                self._save_rotation_data(state, rd)
                return {"status": "waiting_for_seed", "tracked_coins": len(tracked)}

        # 3. Ensure baseline ratios exist for the current coin.
        self._ensure_ratios(rd, current_coin, tracked)

        # 4. Scout for a profitable swap.
        swap_result = self._scout_swap(rd, current_coin, tracked)

        action_taken = "hold"
        swap_meta: dict[str, Any] = {}
        decisions: list[dict[str, Any]] = []

        if swap_result is not None:
            target_coin, ratio_shift = swap_result
            self.logger.info(
                "SWAP SIGNAL: %s -> %s  ratio_shift=%.4f  threshold=%.4f",
                current_coin,
                target_coin,
                ratio_shift,
                self.config.fee_pct * self.config.scout_multiplier,
            )

            # AI validation: Haiku checks for macro/momentum red flags
            ai_swap_meta: dict[str, Any] = {}
            proceed_with_swap = True
            if self.ai_service is not None:
                try:
                    current_price = tracked.get(
                        _product_from_coin(current_coin), 0.0
                    )
                    target_price = tracked.get(
                        _product_from_coin(target_coin), 0.0
                    )
                    swap_verdict = self.ai_service.evaluate_altcoin_swap(
                        current_coin=_product_from_coin(current_coin),
                        target_coin=_product_from_coin(target_coin),
                        current_price=current_price,
                        target_price=target_price,
                        ratio_shift_pct=ratio_shift * 100.0,
                        fee_threshold_pct=self.config.fee_pct * self.config.scout_multiplier * 100.0,
                    )
                    if swap_verdict is not None:
                        ai_swap_meta = {
                            "ai_action": swap_verdict.action,
                            "ai_confidence": round(swap_verdict.confidence, 3),
                            "ai_reasoning": swap_verdict.reasoning,
                            "ai_cached": swap_verdict.cached,
                        }
                        self.logger.info(
                            "AI swap check: action=%s confidence=%.2f | %s",
                            swap_verdict.action, swap_verdict.confidence,
                            swap_verdict.reasoning[:80],
                        )
                        if self.confirmation_mode == "require" and swap_verdict.action == "skip":
                            self.logger.info(
                                "AI blocked swap %s→%s in require mode.",
                                current_coin, target_coin,
                            )
                            proceed_with_swap = False
                except Exception as exc:
                    self.logger.debug("AI swap validation failed (non-blocking): %s", exc)

            if proceed_with_swap:
                success = self._execute_swap(
                    state, rd, broker, current_coin, target_coin, tracked, now,
                )
            else:
                success = False

            if success:
                action_taken = "swap"
                from_coin = current_coin
                swap_meta = {
                    "swapped_from": from_coin,
                    "swapped_to": target_coin,
                    "ratio_shift": round(ratio_shift * 100, 2),  # as percentage
                    "buys": [target_coin],
                    "sells": [from_coin],
                    **ai_swap_meta,
                }
                current_coin = target_coin
                decisions.append({
                    "strategy": "altcoin_rotation",
                    "symbol": f"{from_coin}→{target_coin}",
                    "action": "swap",
                    "reason": "ratio_shift_above_threshold",
                    "signals": {
                        "from": from_coin,
                        "to": target_coin,
                        "ratio_shift_pct": round(ratio_shift * 100, 2),
                        "threshold_pct": round(self.config.fee_pct * self.config.scout_multiplier * 100, 4),
                    },
                })
            elif not proceed_with_swap:
                action_taken = "swap_ai_blocked"
                swap_meta = {"from": current_coin, "to": target_coin, **ai_swap_meta}
                decisions.append({
                    "strategy": "altcoin_rotation",
                    "symbol": f"{current_coin}→{target_coin}",
                    "action": "skip_swap",
                    "reason": "ai_blocked_swap",
                    "signals": {
                        "from": current_coin,
                        "to": target_coin,
                        "ratio_shift_pct": round(ratio_shift * 100, 2),
                        "ai_action": ai_swap_meta.get("ai_action", "unknown"),
                        "ai_confidence": ai_swap_meta.get("ai_confidence"),
                    },
                })
            else:
                action_taken = "swap_failed"
                swap_meta = {"from": current_coin, "to": target_coin}
                decisions.append({
                    "strategy": "altcoin_rotation",
                    "symbol": f"{current_coin}→{target_coin}",
                    "action": "skip_swap",
                    "reason": "swap_execution_failed",
                    "signals": {"from": current_coin, "to": target_coin},
                })
        else:
            self.logger.info("No swap opportunity; holding %s.", current_coin)
            decisions.append({
                "strategy": "altcoin_rotation",
                "symbol": current_coin,
                "action": "hold",
                "reason": "no_swap_opportunity",
                "signals": {
                    "current_coin": current_coin,
                    "coins_tracked": len(tracked),
                    "threshold_pct": round(self.config.fee_pct * self.config.scout_multiplier * 100, 4),
                },
            })

        # 5. Absorb any extra holdings (e.g. from recurring buys).
        absorbed = self._absorb_extras(state, rd, broker, current_coin, tracked, now)

        # 6. Persist rotation data.
        self._save_rotation_data(state, rd)

        # Mark paper positions with latest prices.
        price_map = {pid: tracked[pid] for pid in tracked if pid in state.last_prices or pid in state.positions}
        broker.mark_positions(state, price_map)

        summary = {
            "status": "completed",
            "action": action_taken,
            "current_coin": current_coin,
            "coins_tracked": len(tracked),
            "holdings_synced": len(absorbed),
            "absorbed_coins": absorbed,
            "decisions": decisions,
            **swap_meta,
        }
        self.logger.info("Rotation cycle complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Rotation-data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_rotation_data(state: PortfolioState) -> dict[str, Any]:
        """Return the mutable rotation_data dict, creating it if needed."""
        if not hasattr(state, "rotation_data") or state.rotation_data is None:  # type: ignore[union-attr]
            state.rotation_data = {}  # type: ignore[attr-defined]
        rd: dict[str, Any] = state.rotation_data  # type: ignore[attr-defined]
        rd.setdefault("current_coin", None)
        rd.setdefault("ratios", {})
        rd.setdefault("coin_quantities", {})
        rd.setdefault("cycle_count", 0)
        rd.setdefault("qty_recorded_cycle", {})
        # Legacy states have quantities without timestamps; stamp them with
        # the current cycle so they start aging instead of living forever.
        for coin in rd["coin_quantities"]:
            rd["qty_recorded_cycle"].setdefault(coin, rd["cycle_count"])
        return rd

    @staticmethod
    def _save_rotation_data(state: PortfolioState, rd: dict[str, Any]) -> None:
        state.rotation_data = rd  # type: ignore[attr-defined]

    @staticmethod
    def _record_qty(rd: dict[str, Any], coin: str, qty: float) -> None:
        """Store a best-quantity memory along with the cycle it was set."""
        rd["coin_quantities"][coin] = qty
        rd.setdefault("qty_recorded_cycle", {})[coin] = rd.get("cycle_count", 0)

    def _remembered_qty(self, rd: dict[str, Any], coin: str) -> float:
        """Best-quantity memory for ``coin``, ignoring entries older than
        ``qty_memory_cycles`` so a price crash can't block re-entry forever."""
        qty = rd["coin_quantities"].get(coin, 0.0)
        limit = self.config.qty_memory_cycles
        if qty <= 0 or limit <= 0:
            return qty
        recorded = rd.get("qty_recorded_cycle", {}).get(coin)
        if recorded is None:
            return qty
        age = rd.get("cycle_count", 0) - recorded
        if age > limit:
            self.logger.info(
                "Ignoring stale best-quantity memory for %s "
                "(%.8f recorded %d cycles ago, limit %d).",
                coin, qty, age, limit,
            )
            return 0.0
        return qty

    # ------------------------------------------------------------------
    # Coin filtering
    # ------------------------------------------------------------------

    def _filter_coins(
        self, all_prices: dict[str, float]
    ) -> dict[str, float]:
        """Return prices for coins we want to track, respecting exclusions."""
        excluded = set(self.config.excluded_coins)
        filtered: dict[str, float] = {}
        for pid, price in all_prices.items():
            if not pid.endswith("-USD"):
                continue
            if pid in excluded:
                continue
            if price <= 0:
                continue
            filtered[pid] = price
            if len(filtered) >= self.config.max_coins_to_track:
                break
        return filtered

    # ------------------------------------------------------------------
    # Holdings sync (bootstrap + drift detection)
    # ------------------------------------------------------------------

    def _sync_holdings(
        self,
        state: PortfolioState,
        rd: dict[str, Any],
        prices: dict[str, float],
        broker: "PaperBroker | None" = None,
        now: "datetime | None" = None,
    ) -> str | None:
        """Discover what we actually hold on Coinbase and reconcile.

        Creates/updates Position entries so equity reflects real holdings.
        Returns the product_id of the current rotation coin, or ``None``
        if nothing meaningful is held.
        """
        try:
            holdings = self.coinbase_broker.get_all_holdings()  # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.error("Failed to fetch Coinbase holdings: %s", exc)
            # Fall back to whatever rotation_data says.
            return rd.get("current_coin")

        # Find the largest non-USD holding by USD value,
        # and sync ALL holdings into state.positions for accurate equity.
        best_coin: str | None = None
        best_usd: float = 0.0
        for pid, qty in holdings.items():
            if pid == "USD" or qty <= 0:
                continue
            product_id = pid if "-" in pid else _product_from_coin(pid)
            price = prices.get(product_id)
            if price is None:
                continue
            usd_value = qty * price

            # Create or update a Position for this holding so equity is accurate
            if product_id in state.positions:
                # Update existing position with current qty and price
                pos = state.positions[product_id]
                pos.quantity = qty
                pos.current_price = price
            elif usd_value >= _DUST_USD and broker is not None and now is not None:
                # Create a new position for this Coinbase holding
                broker.open_position(
                    state,
                    symbol=product_id,
                    asset_type="crypto",
                    strategy="altcoin_rotation",
                    price=price,
                    notional=usd_value,
                    opened_at=now,
                    meta={"synced_from_coinbase": True},
                )
                self.logger.info(
                    "Synced Coinbase holding into portfolio: %s qty=%.8f ($%.2f)",
                    product_id, qty, usd_value,
                )
                # Fix the quantity to match actual Coinbase balance
                if product_id in state.positions:
                    state.positions[product_id].quantity = qty

            if usd_value > best_usd:
                best_usd = usd_value
                best_coin = product_id

        if best_coin is None or best_usd < _DUST_USD:
            self.logger.info(
                "No significant crypto holdings found on Coinbase (best_usd=$%.2f).",
                best_usd,
            )
            return rd.get("current_coin")

        prev = rd.get("current_coin")
        if prev != best_coin:
            self.logger.info(
                "Holdings sync: current coin changed %s -> %s ($%.2f).",
                prev, best_coin, best_usd,
            )
        rd["current_coin"] = best_coin

        # Record the quantity if it is our new best for this coin.
        qty_held = holdings.get(
            _coin_from_product(best_coin),
            holdings.get(best_coin, 0.0),
        )
        prev_best = rd["coin_quantities"].get(best_coin, 0.0)
        if qty_held > prev_best:
            self._record_qty(rd, best_coin, qty_held)

        return best_coin

    # ------------------------------------------------------------------
    # Seed from available cash (first-run bootstrap)
    # ------------------------------------------------------------------

    def _seed_from_cash(
        self,
        state: PortfolioState,
        rd: dict[str, Any],
        broker: PaperBroker,
        prices: dict[str, float],
        now: datetime,
    ) -> str | None:
        """Use available USD/USDC on Coinbase to buy the first rotation coin.

        Picks the coin with the highest 24h volume from the tracked list.
        Returns the product_id of the purchased coin, or None if no cash.
        """
        # Check how much cash is available on Coinbase.
        try:
            accounts = self.coinbase_broker.get_accounts()
            cash_available = 0.0
            for acct in accounts:
                currency = acct.get("currency", "")
                if currency in ("USD", "USDC"):
                    available = float(acct.get("available_balance", {}).get("value", "0"))
                    cash_available += available
        except Exception as exc:
            self.logger.warning("Cannot check cash balance for seeding: %s", exc)
            return None

        # Only use our allocation share (e.g. 60%), leave the rest for predictions
        rotation_budget = cash_available * self.config.capital_allocation_pct
        self.logger.info(
            "Cash available: $%.2f, rotation allocation: %.0f%% = $%.2f",
            cash_available, self.config.capital_allocation_pct * 100, rotation_budget,
        )

        if rotation_budget < self.config.min_trade_usd:
            self.logger.info(
                "Not enough cash for rotation seed: $%.2f < min $%.2f",
                rotation_budget, self.config.min_trade_usd,
            )
            return None

        cash_available = rotation_budget  # Use only our share

        self.logger.info(
            "Seeding rotation with $%.2f (%.0f%% of balance). Picking best coin from %d candidates.",
            cash_available, self.config.capital_allocation_pct * 100, len(prices),
        )

        # Pick the coin with the highest price (proxy for market cap / liquidity).
        # Top coins by price tend to be BTC, ETH, SOL, etc.
        # A better heuristic: pick from a known-good list first.
        preferred = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "LINK-USD", "AVAX-USD"]
        target_coin = None
        for coin in preferred:
            if coin in prices and prices[coin] > 0:
                target_coin = coin
                break

        if target_coin is None:
            # Fall back to highest-priced coin in tracked list.
            target_coin = max(prices, key=lambda k: prices[k])

        self.logger.info(
            "Seed target: %s at $%.4f. Buying with $%.2f.",
            target_coin, prices.get(target_coin, 0), cash_available,
        )

        # Leave a tiny buffer for fees.
        notional = cash_available * 0.98
        if notional < self.config.min_trade_usd:
            return None

        buy_ok = self._place_buy(
            state, target_coin, notional, now=now,
            meta={"strategy": "altcoin_rotation", "seed": True},
        )
        if not buy_ok:
            self.logger.error("Seed buy failed for %s.", target_coin)
            return None

        # Estimate qty purchased.
        target_price = prices.get(target_coin, 1.0)
        estimated_qty = (notional / target_price) * (1.0 - self.config.fee_pct)

        # Open paper position for tracking.
        broker.open_position(
            state,
            symbol=target_coin,
            asset_type="crypto",
            strategy="altcoin_rotation",
            price=target_price,
            notional=notional,
            opened_at=now,
            meta={"seed": True},
        )

        # Update rotation data.
        rd["current_coin"] = target_coin
        self._record_qty(rd, target_coin, estimated_qty)
        self._save_rotation_data(state, rd)

        self.logger.info(
            "Seed complete: bought %s qty≈%.8f with $%.2f.",
            target_coin, estimated_qty, notional,
        )
        return target_coin

    # ------------------------------------------------------------------
    # Ratio management
    # ------------------------------------------------------------------

    def _ensure_ratios(
        self,
        rd: dict[str, Any],
        current_coin: str,
        prices: dict[str, float],
    ) -> None:
        """Initialise baseline ratios for ``current_coin`` against all others."""
        if current_coin not in rd["ratios"]:
            rd["ratios"][current_coin] = {}

        coin_ratios = rd["ratios"][current_coin]
        current_price = prices.get(current_coin)
        if current_price is None or current_price <= 0:
            return

        for pid, price in prices.items():
            if pid == current_coin or price <= 0:
                continue
            if pid not in coin_ratios:
                coin_ratios[pid] = current_price / price
                self.logger.debug(
                    "Baseline ratio %s/%s = %.8f",
                    current_coin, pid, coin_ratios[pid],
                )

    # ------------------------------------------------------------------
    # Scout for profitable swap
    # ------------------------------------------------------------------

    def _scout_swap(
        self,
        rd: dict[str, Any],
        current_coin: str,
        prices: dict[str, float],
    ) -> tuple[str, float] | None:
        """Find the best coin to swap into, if any.

        Returns ``(target_product_id, ratio_shift)`` or ``None``.
        """
        current_price = prices.get(current_coin)
        if current_price is None or current_price <= 0:
            self.logger.warning("No price for current coin %s; cannot scout.", current_coin)
            return None

        threshold = self.config.fee_pct * self.config.scout_multiplier
        coin_ratios = rd["ratios"].get(current_coin, {})

        best_target: str | None = None
        best_shift: float = 0.0

        for candidate_pid, candidate_price in prices.items():
            if candidate_pid == current_coin or candidate_price <= 0:
                continue

            baseline = coin_ratios.get(candidate_pid)
            if baseline is None or baseline <= 0:
                continue

            current_ratio = current_price / candidate_price
            ratio_shift = (current_ratio / baseline) - 1.0

            if ratio_shift <= threshold:
                continue

            # "Never go backwards" check: would we get more of candidate
            # than our previous best quantity?
            current_coin_qty = rd["coin_quantities"].get(current_coin, 0.0)
            if current_coin_qty <= 0:
                # First time; allow the swap.
                pass
            else:
                usd_proceeds = current_coin_qty * current_price * (1.0 - self.config.fee_pct)
                new_qty = (usd_proceeds / candidate_price) * (1.0 - self.config.fee_pct)
                prev_best_qty = self._remembered_qty(rd, candidate_pid)
                if prev_best_qty > 0 and new_qty < prev_best_qty:
                    self.logger.debug(
                        "Skipping %s: would get %.8f but best was %.8f (going backwards).",
                        candidate_pid, new_qty, prev_best_qty,
                    )
                    continue

            if ratio_shift > best_shift:
                best_shift = ratio_shift
                best_target = candidate_pid

        if best_target is not None:
            return (best_target, best_shift)
        return None

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _execute_swap(
        self,
        state: PortfolioState,
        rd: dict[str, Any],
        broker: PaperBroker,
        from_coin: str,
        to_coin: str,
        prices: dict[str, float],
        now: datetime,
    ) -> bool:
        """Sell ``from_coin`` and buy ``to_coin`` via USD.

        Updates rotation_data on success.  Returns ``True`` if the full
        round-trip succeeded.
        """
        from_price = prices.get(from_coin, 0.0)
        to_price = prices.get(to_coin, 0.0)
        if from_price <= 0 or to_price <= 0:
            self.logger.error(
                "Cannot execute swap %s->%s: missing prices (%.4f, %.4f).",
                from_coin, to_coin, from_price, to_price,
            )
            return False

        # Never rotate away from a protected symbol.
        from_coin_bare = _coin_from_product(from_coin)
        if from_coin in self.protected_symbols or from_coin_bare in self.protected_symbols:
            self.logger.info("Protected: skipping rotation swap away from %s.", from_coin)
            return False

        qty_to_sell = rd["coin_quantities"].get(from_coin, 0.0)
        usd_value = qty_to_sell * from_price
        if usd_value < self.config.min_trade_usd:
            self.logger.warning(
                "Swap aborted: %s holdings worth $%.2f < min $%.2f.",
                from_coin, usd_value, self.config.min_trade_usd,
            )
            return False

        # --- Sell from_coin ---
        self.logger.info(
            "SELL %s qty=%.8f (~$%.2f) for rotation swap.",
            from_coin, qty_to_sell, usd_value,
        )
        sell_ok = self._place_sell(state, from_coin, qty_to_sell, reason=f"rotation_swap_to_{to_coin}")
        if not sell_ok:
            self.logger.error("Sell leg failed for %s; aborting swap.", from_coin)
            return False

        # Close the paper position if one exists.
        if from_coin in state.positions:
            broker.close_position(
                state, symbol=from_coin, price=from_price,
                reason=f"rotation_swap_to_{to_coin}",
            )

        # Small delay so Coinbase settles the sell before the buy.
        time.sleep(1)

        # --- Buy to_coin ---
        notional_usd = usd_value * (1.0 - self.config.fee_pct)  # conservative
        if notional_usd < self.config.min_trade_usd:
            self.logger.warning(
                "Post-fee proceeds $%.2f too small to buy %s.", notional_usd, to_coin,
            )
            return False

        self.logger.info(
            "BUY %s notional=$%.2f for rotation swap.",
            to_coin, notional_usd,
        )
        buy_ok = self._place_buy(state, to_coin, notional_usd, now=now, meta={
            "strategy": "altcoin_rotation",
            "swap_from": from_coin,
            "ratio_shift": round(
                (from_price / to_price)
                / rd["ratios"].get(from_coin, {}).get(to_coin, 1.0)
                - 1.0,
                6,
            ),
        })
        if not buy_ok:
            self.logger.error(
                "Buy leg failed for %s after selling %s!  "
                "USD is now un-invested until next cycle.",
                to_coin, from_coin,
            )
            return False

        # Open paper position for tracking.
        estimated_qty = (notional_usd / to_price) * (1.0 - self.config.fee_pct)
        broker.open_position(
            state,
            symbol=to_coin,
            asset_type="crypto",
            strategy="altcoin_rotation",
            price=to_price,
            notional=notional_usd,
            opened_at=now,
            meta={"swap_from": from_coin},
        )

        # --- Update rotation data ---
        rd["current_coin"] = to_coin
        self._record_qty(rd, to_coin, estimated_qty)
        # Reset ratios for the new coin at current prices.
        rd["ratios"][to_coin] = {}
        # (They will be populated on the next _ensure_ratios call.)

        self.logger.info(
            "Swap complete: %s -> %s  estimated_qty=%.8f",
            from_coin, to_coin, estimated_qty,
        )
        return True

    # ------------------------------------------------------------------
    # Absorb extra holdings (recurring buys, airdrops, etc.)
    # ------------------------------------------------------------------

    def _absorb_extras(
        self,
        state: PortfolioState,
        rd: dict[str, Any],
        broker: PaperBroker,
        current_coin: str,
        prices: dict[str, float],
        now: datetime,
    ) -> list[str]:
        """Sell any non-rotation holdings and consolidate into ``current_coin``.

        Returns a list of product-IDs that were absorbed.
        """
        try:
            holdings = self.coinbase_broker.get_all_holdings()  # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.warning("Cannot fetch holdings for absorption check: %s", exc)
            return []

        absorbed: list[str] = []

        for asset, qty in holdings.items():
            if asset == "USD" or qty <= 0:
                continue
            product_id = asset if "-" in asset else _product_from_coin(asset)
            if product_id == current_coin:
                continue  # That is our rotation coin; skip.
            # Never absorb (sell) a protected symbol.
            if asset in self.protected_symbols or product_id in self.protected_symbols:
                self.logger.info("Protected: skipping absorption of %s.", product_id)
                continue

            price = prices.get(product_id)
            if price is None:
                continue
            usd_value = qty * price
            if usd_value < self.config.min_trade_usd:
                self.logger.debug(
                    "Skipping absorption of %s: $%.2f < min $%.2f.",
                    product_id, usd_value, self.config.min_trade_usd,
                )
                continue

            self.logger.info(
                "Absorbing extra %s: qty=%.8f (~$%.2f) -> selling to buy %s.",
                product_id, qty, usd_value, current_coin,
            )

            sell_ok = self._place_sell(
                state, product_id, qty,
                reason=f"absorb_into_{current_coin}",
            )
            if not sell_ok:
                self.logger.warning("Failed to sell extra %s; skipping.", product_id)
                continue

            time.sleep(1)

            notional = usd_value * (1.0 - self.config.fee_pct)
            if notional < self.config.min_trade_usd:
                continue

            buy_ok = self._place_buy(
                state, current_coin, notional, now=now,
                meta={"strategy": "altcoin_rotation", "absorbed_from": product_id},
            )
            if buy_ok:
                absorbed.append(product_id)
                # Update tracked quantity for current coin.
                extra_qty = (notional / prices.get(current_coin, 1.0)) * (
                    1.0 - self.config.fee_pct
                )
                self._record_qty(
                    rd,
                    current_coin,
                    rd["coin_quantities"].get(current_coin, 0.0) + extra_qty,
                )
                self.logger.info(
                    "Absorbed %s (+%.8f of %s).", product_id, extra_qty, current_coin,
                )

        return absorbed

    # ------------------------------------------------------------------
    # Low-level order wrappers (product_id native)
    # ------------------------------------------------------------------

    def _place_sell(
        self,
        state: PortfolioState,
        product_id: str,
        quantity: float,
        *,
        reason: str,
    ) -> bool:
        """Sell via CoinbaseBroker using the product_id directly.

        Falls back to the binance_symbol interface if ``place_market_sell_by_product``
        is not yet available on the broker.
        """
        if hasattr(self.coinbase_broker, "place_market_sell_by_product"):
            return self.coinbase_broker.place_market_sell_by_product(
                product_id=product_id,
                quantity=quantity,
                reason=reason,
            )

        # Fallback: try via the existing binance-symbol method.
        try:
            binance_sym = self.coinbase_client.to_binance_symbol(product_id)
        except ValueError:
            # Product not in SYMBOL_MAP; use the broker's low-level _post.
            return self._raw_sell(product_id, quantity, reason)

        return self.coinbase_broker.place_market_sell(
            state,
            binance_symbol=binance_sym,
            quantity=quantity,
            strategy="altcoin_rotation",
            reason=reason,
        )

    def _place_buy(
        self,
        state: PortfolioState,
        product_id: str,
        notional_usd: float,
        *,
        now: datetime,
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """Buy via CoinbaseBroker using the product_id directly.

        Returns ``True`` on apparent success.
        """
        if hasattr(self.coinbase_broker, "place_market_buy_by_product"):
            result = self.coinbase_broker.place_market_buy_by_product(
                product_id=product_id,
                notional_usd=notional_usd,
            )
            return result is not None or self.coinbase_broker.paper_mode

        # Fallback: try via the existing binance-symbol method.
        try:
            binance_sym = self.coinbase_client.to_binance_symbol(product_id)
        except ValueError:
            return self._raw_buy(product_id, notional_usd)

        result = self.coinbase_broker.place_market_buy(
            state,
            binance_symbol=binance_sym,
            notional_usd=notional_usd,
            strategy="altcoin_rotation",
            now=now,
            meta=meta,
        )
        # In paper mode place_market_buy returns None (paper broker tracks).
        return result is not None or self.coinbase_broker.paper_mode

    # ------------------------------------------------------------------
    # Raw order methods (bypass symbol mapping entirely)
    # ------------------------------------------------------------------

    def _raw_sell(self, product_id: str, quantity: float, reason: str) -> bool:
        """Place a sell directly via the broker's REST layer."""
        if self.coinbase_broker.paper_mode:
            self.logger.info(
                "[PAPER] Would SELL %s qty=%.8f reason=%s",
                product_id, quantity, reason,
            )
            return True

        import uuid
        order_payload = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": f"{quantity:.8f}",
                }
            },
        }
        self.logger.info(
            "RAW SELL: product=%s qty=%.8f reason=%s", product_id, quantity, reason,
        )
        try:
            result = self.coinbase_broker._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("RAW SELL response: %s", result)
            return result.get("success", False)
        except Exception as exc:
            self.logger.error("RAW SELL failed for %s: %s", product_id, exc)
            return False

    def _raw_buy(self, product_id: str, notional_usd: float) -> bool:
        """Place a buy directly via the broker's REST layer."""
        if self.coinbase_broker.paper_mode:
            self.logger.info(
                "[PAPER] Would BUY %s notional=$%.2f",
                product_id, notional_usd,
            )
            return True

        import uuid
        order_payload = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": f"{notional_usd:.2f}",
                }
            },
        }
        self.logger.info(
            "RAW BUY: product=%s notional=$%.2f", product_id, notional_usd,
        )
        try:
            result = self.coinbase_broker._post("/api/v3/brokerage/orders", order_payload)
            self.logger.info("RAW BUY response: %s", result)
            return result.get("success", False)
        except Exception as exc:
            self.logger.error("RAW BUY failed for %s: %s", product_id, exc)
            return False
