from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

from trading_bot.config import AppConfig
from trading_bot.scheduler import SchedulerService
from trading_bot.service import TradingBotService


def _make_service() -> TradingBotService:
    tmp = Path(tempfile.mkdtemp(prefix="zb_sched_"))
    os.environ["BOT_DATA_DIR"] = str(tmp / "data")
    os.environ["BOT_LOG_DIR"] = str(tmp / "logs")
    os.environ["PAPER_MODE"] = "true"
    os.environ["INITIAL_CASH"] = "100"
    os.environ["TRADING_AGENTS_ENABLED"] = "false"
    cfg = AppConfig.load()
    return TradingBotService(cfg, logging.getLogger("zb_test"))


def _make_scheduler(service: TradingBotService) -> SchedulerService:
    notifier = MagicMock()
    notifier.notify_leads = AsyncMock()
    telegram = MagicMock()
    return SchedulerService(
        config=service.config,
        service=service,
        telegram=telegram,
        notifier=notifier,
        logger=logging.getLogger("zb_test_sched"),
    )


CRYPTO_RESULT_TWO_LEADS = {
    "decisions": [
        {"symbol": "ETHUSDT", "action": "skip_buy", "reason": "position_size_zero", "signals": {"fear_greed": 22}},
        {"symbol": "SOLUSDT", "action": "skip_buy", "reason": "ai_rejected", "signals": {}},
    ]
}


class SchedulerLeadsDispatchTests(unittest.TestCase):
    def test_first_dispatch_sends_all_new_leads(self) -> None:
        svc = _make_service()
        sched = _make_scheduler(svc)
        now = datetime(2026, 4, 27, 9, 30, tzinfo=svc.config.timezone)

        asyncio.run(sched._dispatch_leads("crypto", CRYPTO_RESULT_TWO_LEADS, now))

        sched.notifier.notify_leads.assert_awaited_once()
        args, kwargs = sched.notifier.notify_leads.await_args
        self.assertEqual(args[0], "crypto")
        symbols = sorted(lead["symbol"] for lead in args[1])
        self.assertEqual(symbols, ["ETHUSDT", "SOLUSDT"])
        self.assertEqual(kwargs.get("max_per_message"), svc.config.leads.max_per_message)

        # Service cache populated for /leads command.
        cached = svc.get_last_leads()
        self.assertIn("crypto", cached)
        self.assertEqual(len(cached["crypto"]), 2)

        # State persisted with both symbols marked notified for today.
        state = svc.state_store.load()
        self.assertEqual(state.notified_leads_date, now.date().isoformat())
        self.assertEqual(set(state.notified_leads["crypto"]), {"ETHUSDT", "SOLUSDT"})

    def test_repeat_dispatch_same_day_does_not_re_notify(self) -> None:
        svc = _make_service()
        sched = _make_scheduler(svc)
        now = datetime(2026, 4, 27, 9, 30, tzinfo=svc.config.timezone)

        asyncio.run(sched._dispatch_leads("crypto", CRYPTO_RESULT_TWO_LEADS, now))
        sched.notifier.notify_leads.reset_mock()

        # Same result, same day → fully deduped.
        asyncio.run(sched._dispatch_leads("crypto", CRYPTO_RESULT_TWO_LEADS, now))

        sched.notifier.notify_leads.assert_not_awaited()

    def test_new_symbol_appears_after_first_dispatch_is_pushed(self) -> None:
        svc = _make_service()
        sched = _make_scheduler(svc)
        now = datetime(2026, 4, 27, 9, 30, tzinfo=svc.config.timezone)

        asyncio.run(sched._dispatch_leads("crypto", CRYPTO_RESULT_TWO_LEADS, now))
        sched.notifier.notify_leads.reset_mock()

        result_with_extra = {
            "decisions": list(CRYPTO_RESULT_TWO_LEADS["decisions"]) + [
                {"symbol": "AVAXUSDT", "action": "skip_buy", "reason": "position_size_zero"},
            ]
        }
        asyncio.run(sched._dispatch_leads("crypto", result_with_extra, now))

        sched.notifier.notify_leads.assert_awaited_once()
        args, _ = sched.notifier.notify_leads.await_args
        symbols = [lead["symbol"] for lead in args[1]]
        self.assertEqual(symbols, ["AVAXUSDT"])

    def test_dedup_resets_on_new_day(self) -> None:
        svc = _make_service()
        sched = _make_scheduler(svc)
        day_one = datetime(2026, 4, 27, 9, 30, tzinfo=svc.config.timezone)
        day_two = datetime(2026, 4, 28, 9, 30, tzinfo=svc.config.timezone)

        asyncio.run(sched._dispatch_leads("crypto", CRYPTO_RESULT_TWO_LEADS, day_one))
        sched.notifier.notify_leads.reset_mock()

        asyncio.run(sched._dispatch_leads("crypto", CRYPTO_RESULT_TWO_LEADS, day_two))

        sched.notifier.notify_leads.assert_awaited_once()
        state = svc.state_store.load()
        self.assertEqual(state.notified_leads_date, day_two.date().isoformat())

    def test_disabled_config_skips_dispatch_entirely(self) -> None:
        os.environ["LEADS_NOTIFICATION_ENABLED"] = "false"
        try:
            svc = _make_service()
        finally:
            os.environ.pop("LEADS_NOTIFICATION_ENABLED", None)
        sched = _make_scheduler(svc)
        now = datetime(2026, 4, 27, 9, 30, tzinfo=svc.config.timezone)

        asyncio.run(sched._dispatch_leads("crypto", CRYPTO_RESULT_TWO_LEADS, now))

        sched.notifier.notify_leads.assert_not_awaited()
        self.assertEqual(svc.get_last_leads(), {})  # cache also untouched


if __name__ == "__main__":
    unittest.main()
