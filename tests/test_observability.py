from __future__ import annotations

import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.models import PortfolioState
from trading_bot.observability import ModuleLatencyTracker, TradeAuditLog


class ModuleLatencyTrackerTests(unittest.TestCase):
    def test_snapshot_aggregates_recent_samples(self) -> None:
        tracker = ModuleLatencyTracker(window_size=3)

        tracker.record("crypto", 10.0)
        tracker.record("crypto", 20.0)
        tracker.record("crypto", 30.0)

        snapshot = tracker.snapshot()

        self.assertEqual(snapshot["crypto"]["last_ms"], 30.0)
        self.assertEqual(snapshot["crypto"]["avg_ms"], 20.0)
        self.assertEqual(snapshot["crypto"]["max_ms"], 30.0)
        self.assertEqual(snapshot["crypto"]["count"], 3.0)


class TradeAuditLogTests(unittest.TestCase):
    def test_record_and_tail_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            audit = TradeAuditLog(path, logging.getLogger("test_audit"), enabled=True)

            audit.record({"event": "open", "symbol": "BTC-USD", "price": 10.0})
            audit.record({"event": "close", "symbol": "BTC-USD", "price": 11.0})

            entries = audit.tail(2)

            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["event"], "open")
            self.assertEqual(entries[1]["event"], "close")


class BrokerObservabilityTests(unittest.TestCase):
    def test_broker_emits_trade_events_and_price_timestamps(self) -> None:
        broker = PaperBroker(True, logging.getLogger("test_broker_obs"))
        events: list[dict[str, object]] = []
        broker.set_on_trade(events.append)
        now = datetime(2026, 3, 31, 12, 0, tzinfo=ZoneInfo("UTC"))
        state = PortfolioState(cash=100.0)

        position = broker.open_position(
            state,
            symbol="BTC-USD",
            asset_type="crypto",
            strategy="crypto",
            price=10.0,
            notional=20.0,
            opened_at=now,
        )
        self.assertIsNotNone(position)
        self.assertIn("BTC-USD", state.last_price_timestamps)

        broker.close_position(state, symbol="BTC-USD", price=11.0, reason="test_close")

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event"], "open")
        self.assertEqual(events[1]["event"], "close")
        self.assertGreater(state.last_price_timestamps["BTC-USD"], 0.0)