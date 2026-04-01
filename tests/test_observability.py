from __future__ import annotations

import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_bot.broker import PaperBroker
from trading_bot.models import PortfolioState
from trading_bot.observability import DecisionLog, ModuleLatencyTracker, TradeAuditLog


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


class DecisionLogTests(unittest.TestCase):
    def _make_log(self, tmp_dir: str) -> DecisionLog:
        path = Path(tmp_dir) / "decision_log.jsonl"
        return DecisionLog(path, logging.getLogger("test_decision_log"))

    def test_record_and_tail_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._make_log(tmp)
            log.record({
                "strategy": "crypto_sentiment_funding",
                "symbol": "BTCUSDT",
                "action": "buy",
                "reason": "fear_and_greed_buy_threshold",
                "signals": {"fear_greed": 22, "funding_rate": -0.001},
            })
            log.record({
                "strategy": "value_investing",
                "symbol": "*",
                "action": "skip_buy",
                "reason": "vix_below_threshold",
                "signals": {"vix": 18.0, "vix_threshold": 20.0},
            })

            entries = log.tail(10)

            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["action"], "buy")
            self.assertEqual(entries[0]["strategy"], "crypto_sentiment_funding")
            self.assertEqual(entries[1]["action"], "skip_buy")
            self.assertEqual(entries[1]["reason"], "vix_below_threshold")

    def test_tail_respects_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._make_log(tmp)
            for i in range(15):
                log.record({"strategy": "s", "symbol": str(i), "action": "hold",
                             "reason": "test", "signals": {}})

            self.assertEqual(len(log.tail(5)), 5)
            self.assertEqual(len(log.tail(20)), 15)

    def test_tail_returns_most_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._make_log(tmp)
            for i in range(5):
                log.record({"strategy": "s", "symbol": str(i), "action": "hold",
                             "reason": "test", "signals": {}})

            entries = log.tail(3)
            symbols = [e["symbol"] for e in entries]
            self.assertEqual(symbols, ["2", "3", "4"])

    def test_timestamp_auto_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._make_log(tmp)
            log.record({"strategy": "s", "symbol": "X", "action": "buy",
                        "reason": "r", "signals": {}})

            entry = log.tail(1)[0]
            self.assertIn("timestamp", entry)
            self.assertIsInstance(entry["timestamp"], str)

    def test_record_many(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._make_log(tmp)
            log.record_many([
                {"strategy": "s", "symbol": "A", "action": "buy", "reason": "r", "signals": {}},
                {"strategy": "s", "symbol": "B", "action": "sell", "reason": "r", "signals": {}},
            ])
            entries = log.tail(10)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["symbol"], "A")
            self.assertEqual(entries[1]["symbol"], "B")

    def test_record_many_writes_all_lines_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            log = DecisionLog(path, logging.getLogger("test"))
            log.record_many([
                {"strategy": "s", "symbol": str(i), "action": "hold", "reason": "r", "signals": {}}
                for i in range(5)
            ])
            lines = [l for l in path.read_text().splitlines() if l.strip()]
            self.assertEqual(len(lines), 5)

    def test_record_many_empty_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            log = DecisionLog(path, logging.getLogger("test"))
            log.record_many([])
            self.assertFalse(path.exists())
            self.assertEqual(log.tail(10), [])

    def test_persists_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subdir" / "decisions.jsonl"
            log = DecisionLog(path, logging.getLogger("test"))
            log.record({"strategy": "s", "symbol": "X", "action": "buy",
                        "reason": "r", "signals": {}})

            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn('"action"', content)

    def test_memory_buffer_rolls_over(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            log = DecisionLog(path, logging.getLogger("test"), memory_limit=5)
            for i in range(10):
                log.record({"strategy": "s", "symbol": str(i), "action": "hold",
                             "reason": "r", "signals": {}})

            entries = log.tail(10)
            self.assertEqual(len(entries), 5)
            symbols = [e["symbol"] for e in entries]
            self.assertEqual(symbols, ["5", "6", "7", "8", "9"])

    def test_hydrates_from_file_on_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            # Write some entries using a first instance
            log1 = DecisionLog(path, logging.getLogger("test"))
            for i in range(3):
                log1.record({"strategy": "s", "symbol": str(i), "action": "hold",
                              "reason": "r", "signals": {}})
            # Second instance (simulating a restart) should pick up those entries
            log2 = DecisionLog(path, logging.getLogger("test"))
            entries = log2.tail(10)
            self.assertEqual(len(entries), 3)
            symbols = [e["symbol"] for e in entries]
            self.assertEqual(symbols, ["0", "1", "2"])

    def test_hydration_respects_memory_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            # Write 10 entries via a first instance
            log1 = DecisionLog(path, logging.getLogger("test"))
            for i in range(10):
                log1.record({"strategy": "s", "symbol": str(i), "action": "hold",
                              "reason": "r", "signals": {}})
            # Restart with a smaller memory_limit; should only keep the last N
            log2 = DecisionLog(path, logging.getLogger("test"), memory_limit=4)
            entries = log2.tail(10)
            self.assertEqual(len(entries), 4)
            symbols = [e["symbol"] for e in entries]
            self.assertEqual(symbols, ["6", "7", "8", "9"])