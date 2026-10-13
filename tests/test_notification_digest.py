"""Telegram digest: routine alerts batch into one message per window; urgent ones don't wait."""

from __future__ import annotations

import asyncio
import logging
import unittest

from trading_bot.notifications import NotificationDispatcher
from trading_bot.telegram_bot import TradingTelegramBot


class FakeTelegram:
    """Mimics TradingTelegramBot's capture/send_message contract."""

    is_configured = True

    def __init__(self):
        self.sent: list[str] = []
        self._capture = None

    async def capture(self, send_fn, *args, **kwargs):
        prev, self._capture = self._capture, []
        try:
            await send_fn(*args, **kwargs)
            return list(self._capture)
        finally:
            self._capture = prev

    async def send_message(self, text):
        if self._capture is not None:
            self._capture.append(text)
        else:
            self.sent.append(text)

    async def send_trade_alert(self, result):
        await self.send_message(f"trade:{result.get('n')}")

    async def send_heartbeat(self, status):
        await self.send_message("heartbeat")

    async def send_risk_alert(self, risk):
        await self.send_message("risk:kill switch active")

    async def send_error(self, msg):
        await self.send_message(f"error:{msg}")

    async def send_leads_alert(self, strategy, leads, max_per_message=10):
        await self.send_message(f"leads:{strategy}")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _make(hours=6.0):
    tg, clock = FakeTelegram(), Clock()
    d = NotificationDispatcher(tg, logging.getLogger("t"), digest_hours=hours, clock=clock)
    return d, tg, clock


def run(coro):
    return asyncio.run(coro)


class DigestTests(unittest.TestCase):
    def test_alerts_are_held_until_the_window_passes_then_sent_as_one_message(self):
        d, tg, clock = _make()

        async def go():
            for i in range(5):
                await d.notify_trade({"n": i})
                clock.t += 3600                      # one alert per hour for 5 hours
            self.assertEqual(tg.sent, [])            # nothing sent hourly
            clock.t += 3600                          # 6 hours since start
            await d.flush_if_due()
        run(go())
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("6-hour update", tg.sent[0])
        self.assertIn("5 messages", tg.sent[0])
        for i in range(5):
            self.assertIn(f"trade:{i}", tg.sent[0])

    def test_nothing_sent_when_the_window_is_quiet(self):
        d, tg, clock = _make()
        clock.t += 6 * 3600
        run(d.flush_if_due())
        self.assertEqual(tg.sent, [])

    def test_a_new_alert_after_the_window_triggers_the_flush(self):
        d, tg, clock = _make()

        async def go():
            await d.notify_trade({"n": 1})
            clock.t += 6 * 3600 + 1
            await d.notify_trade({"n": 2})
        run(go())
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("trade:1", tg.sent[0])
        self.assertIn("trade:2", tg.sent[0])

    def test_identical_repeats_collapse(self):
        d, tg, clock = _make()

        async def go():
            for _ in range(4):
                await d.notify_risk({"kill_switch_active": True})
            clock.t += 6 * 3600
            await d.flush_if_due()
        run(go())
        self.assertEqual(tg.sent[0].count("risk:kill switch active"), 1)
        self.assertIn("3 repeats hidden", tg.sent[0])

    def test_kill_switch_trigger_is_immediate_but_other_errors_wait(self):
        d, tg, _ = _make()

        async def go():
            await d.notify_error("Kill switch triggered. Bot paused.")
            await d.notify_error("kalshi_sports failed: boom")
        run(go())
        self.assertEqual(tg.sent, ["error:Kill switch triggered. Bot paused."])

    def test_default_is_one_digest_a_day_and_a_day_of_alerts_is_kept(self):
        from trading_bot.config import TelegramConfig
        self.assertEqual(TelegramConfig().digest_hours, 24.0)
        d, tg, clock = _make(hours=24)

        async def go():
            for i in range(100):
                await d.notify_trade({"n": i})
            clock.t += 23 * 3600
            await d.flush_if_due()
            self.assertEqual(tg.sent, [])           # nothing before the day is up
            clock.t += 2 * 3600
            await d.notify_trade({"n": 100})
        run(go())
        text = "\n".join(tg.sent)
        self.assertIn("24-hour update", text)
        self.assertIn("trade:0", text)               # the first alert of the day is still there
        self.assertIn("trade:100", text)

    def test_digest_off_sends_immediately(self):
        d, tg, _ = _make(hours=0)
        run(d.notify_trade({"n": 1}))
        run(d.notify_heartbeat({"equity": 1}))
        self.assertEqual(tg.sent, ["trade:1", "heartbeat"])

    def test_shutdown_flushes_held_alerts(self):
        d, tg, _ = _make()

        async def go():
            await d.notify_trade({"n": 1})
            await d.notify_shutdown()
        run(go())
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("trade:1", tg.sent[0])

    def test_long_digests_are_split_under_the_telegram_limit(self):
        d, tg, clock = _make()
        tg.send_trade_alert = lambda result: tg.send_message("x" * 1500 + str(result["n"]))

        async def go():
            for i in range(6):
                await d.notify_trade({"n": i})
            clock.t += 6 * 3600
            await d.flush_if_due()
        run(go())
        self.assertGreater(len(tg.sent), 1)
        self.assertTrue(all(len(m) <= 4096 for m in tg.sent))

    def test_leads_and_heartbeats_are_batched_too(self):
        d, tg, clock = _make()

        async def go():
            await d.notify_leads("kalshi_sports", [{"symbol": "X"}])
            await d.notify_heartbeat({"equity": 1})
            clock.t += 6 * 3600
            await d.flush_if_due()
        run(go())
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("leads:kalshi_sports", tg.sent[0])
        self.assertIn("heartbeat", tg.sent[0])


class RealTelegramCaptureTests(unittest.TestCase):
    def test_capture_collects_instead_of_sending_and_restores_normal_sending(self):
        bot = object.__new__(TradingTelegramBot)
        bot.logger = logging.getLogger("t")
        bot._capture = None
        bot._app = None
        bot.config = type("C", (), {"bot_token": "t", "chat_id": "c", "is_configured": True})()

        async def alert():
            await bot.send_message("hello")
            await bot.send_message("world")

        captured = run(bot.capture(alert))
        self.assertEqual(captured, ["hello", "world"])
        self.assertIsNone(bot._capture)


if __name__ == "__main__":
    unittest.main()
