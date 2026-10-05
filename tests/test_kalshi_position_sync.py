"""Live Kalshi position sync: settled positions must be pruned, NO positions seen."""

from __future__ import annotations

import logging
import unittest
from datetime import timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from trading_bot.models import PortfolioState, Position
from trading_bot.service import TradingBotService


from trading_bot.broker import PaperBroker


def _service(positions, market=None):
    svc = object.__new__(TradingBotService)
    svc.broker = PaperBroker(False, logging.getLogger('b'))
    svc.logger = logging.getLogger("test_sync")
    svc.config = SimpleNamespace(timezone=ZoneInfo("UTC"))
    svc.kalshi_client = MagicMock()
    svc.kalshi_client.can_trade = True
    svc.kalshi_client.get_positions.return_value = positions
    svc.kalshi_client.get_market.return_value = market or {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.44",
                                                           "no_bid_dollars": "0.56", "no_ask_dollars": "0.60"}
    return svc


def _synced(ticker="KXA-1", side="yes"):
    return Position(
        symbol=f"KL:{ticker}:{side.upper()}", asset_type="prediction", strategy="kalshi_manual",
        quantity=10.0, entry_price=0.5, current_price=0.5, opened_at="x",
        meta={"ticker": ticker, "side": side, "source": "kalshi_sync", "venue": "kalshi", "cash_bucket": "kalshi"},
    )


class KalshiPositionSyncTests(unittest.TestCase):
    def test_settled_positions_are_pruned_even_when_account_is_empty(self):
        state = PortfolioState(cash=0.0)
        pos = _synced()
        state.positions[pos.symbol] = pos
        state.last_prices[pos.symbol] = 0.5
        _service([])._sync_kalshi_positions(state)
        self.assertEqual(state.positions, {})
        self.assertNotIn(pos.symbol, state.last_prices)

    def test_only_unheld_positions_are_pruned(self):
        state = PortfolioState(cash=0.0)
        for t in ("KXA-1", "KXB-2"):
            p = _synced(t)
            state.positions[p.symbol] = p
        _service([{"ticker": "KXB-2", "position": 7}])._sync_kalshi_positions(state)
        self.assertEqual(list(state.positions), ["KL:KXB-2:YES"])
        self.assertEqual(state.positions["KL:KXB-2:YES"].quantity, 7.0)

    def test_negative_position_is_a_no_holding(self):
        state = PortfolioState(cash=0.0)
        _service([{"ticker": "KXC-3", "position": -5}])._sync_kalshi_positions(state)
        pos = state.positions["KL:KXC-3:NO"]
        self.assertEqual((pos.quantity, pos.meta["side"]), (5.0, "no"))
        self.assertAlmostEqual(pos.entry_price, 0.58)   # mid of no bid/ask
        self.assertNotIn("KL:KXC-3:YES", state.positions)

    def test_fp_string_positions_supported(self):
        state = PortfolioState(cash=0.0)
        _service([{"ticker": "KXD-4", "position_fp": "12.00"}])._sync_kalshi_positions(state)
        self.assertEqual(state.positions["KL:KXD-4:YES"].quantity, 12.0)

    def test_api_failure_never_prunes(self):
        state = PortfolioState(cash=0.0)
        pos = _synced()
        state.positions[pos.symbol] = pos
        svc = _service([])
        svc.kalshi_client.get_positions.side_effect = RuntimeError("503")
        svc._sync_kalshi_positions(state)
        self.assertIn(pos.symbol, state.positions)

    def test_bot_placed_contracts_are_not_double_counted(self):
        state = PortfolioState(cash=0.0)
        bot = Position(symbol="KS:KXE-5:YES", asset_type="prediction", strategy="kalshi_sports", quantity=3.0,
                       entry_price=0.5, current_price=0.5, opened_at="x",
                       meta={"ticker": "KXE-5", "side": "yes", "venue": "kalshi"})
        state.positions[bot.symbol] = bot
        _service([{"ticker": "KXE-5", "position": 3}])._sync_kalshi_positions(state)
        self.assertEqual(list(state.positions), ["KS:KXE-5:YES"])

    def test_manual_position_not_in_sync_source_is_kept(self):
        state = PortfolioState(cash=0.0)
        bot = Position(symbol="KX:KXF-6:YES", asset_type="prediction", strategy="predictions", quantity=3.0,
                       entry_price=0.5, current_price=0.5, opened_at="x", meta={"ticker": "KXF-6", "side": "yes"})
        state.positions[bot.symbol] = bot
        _service([])._sync_kalshi_positions(state)   # bot-placed positions are closed by their strategy
        self.assertIn(bot.symbol, state.positions)


def _bot_position(qty=10.0, age_minutes=60, ticker="KXG-7", prefix="KS"):
    from datetime import datetime, timedelta
    opened = (datetime.now(ZoneInfo("UTC")) - timedelta(minutes=age_minutes)).isoformat()
    return Position(symbol=f"{prefix}:{ticker}:YES", asset_type="prediction", strategy="kalshi_sports",
                    quantity=qty, entry_price=0.5, current_price=0.5, opened_at=opened,
                    meta={"ticker": ticker, "side": "yes", "venue": "kalshi", "cash_bucket": "kalshi"})


class ReconcileBotPositionsTests(unittest.TestCase):
    def _state(self, pos):
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 50.0})
        state.positions[pos.symbol] = pos
        return state

    def test_never_filled_bet_is_removed_with_refund_bookkeeping(self):
        pos = _bot_position()
        state = self._state(pos)
        svc = _service([], market={"status": "active"})
        svc._sync_kalshi_positions(state)
        self.assertNotIn(pos.symbol, state.positions)
        self.assertEqual(state.cash_reserves["kalshi"], 50.0)   # wallet is synced from Kalshi: no double refund

    def test_recent_bet_gets_a_grace_period(self):
        pos = _bot_position(age_minutes=1)
        state = self._state(pos)
        _service([], market={"status": "active"})._sync_kalshi_positions(state)
        self.assertIn(pos.symbol, state.positions)

    def test_partial_holding_shrinks_the_bet(self):
        pos = _bot_position(qty=10.0)
        state = self._state(pos)
        _service([{"ticker": "KXG-7", "position": 4}], market={"status": "active"})._sync_kalshi_positions(state)
        self.assertEqual(state.positions[pos.symbol].quantity, 4.0)

    def test_settled_market_is_left_for_the_strategy_to_close(self):
        pos = _bot_position()
        state = self._state(pos)
        _service([], market={"status": "finalized", "result": "yes"})._sync_kalshi_positions(state)
        self.assertIn(pos.symbol, state.positions)

    def test_fully_held_bet_is_untouched(self):
        pos = _bot_position(qty=10.0)
        state = self._state(pos)
        _service([{"ticker": "KXG-7", "position": 10}])._sync_kalshi_positions(state)
        self.assertEqual(state.positions[pos.symbol].quantity, 10.0)


if __name__ == "__main__":
    unittest.main()
