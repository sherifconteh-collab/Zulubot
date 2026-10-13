"""Unified notification dispatcher.

Routes notifications to configured channels (Telegram, console log, future webhooks).

Telegram digest: routine alerts are collected and sent as one message every
``digest_hours`` (default 24, i.e. once a day) instead of one message per event. Urgent alerts (the kill
switch firing) and replies to Telegram commands are never delayed.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from trading_bot.telegram_bot import TradingTelegramBot

_CHUNK_CHARS = 3600          # stay well under Telegram's 4096-character message limit
_MAX_BUFFERED = 120          # keep only the most recent messages if a digest window is very busy (a day's worth)
_SEPARATOR = "\n\n\u2014 \u2014 \u2014\n\n"


class NotificationDispatcher:
    def __init__(
        self,
        telegram: TradingTelegramBot | None,
        logger: logging.Logger,
        *,
        digest_hours: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.telegram = telegram
        self.logger = logger.getChild("notifications")
        self.digest_seconds = max(0.0, float(digest_hours)) * 3600.0
        self._clock = clock
        self._last_flush = clock()
        self._buffer: list[str] = []
        self._duplicates = 0

    # ── digest machinery ────────────────────────────────────────────

    @property
    def _telegram_ready(self) -> bool:
        return bool(self.telegram and self.telegram.is_configured)

    async def _route(self, send_fn: Any, *args: Any, urgent: bool = False, **kwargs: Any) -> None:
        """Send now (urgent or digest off), otherwise hold the message for the next digest."""
        if not self._telegram_ready:
            return
        if urgent or self.digest_seconds <= 0:
            await send_fn(*args, **kwargs)
            return
        for text in await self.telegram.capture(send_fn, *args, **kwargs):  # type: ignore[union-attr]
            if text in self._buffer:
                self._duplicates += 1
            else:
                self._buffer.append(text)
        if len(self._buffer) > _MAX_BUFFERED:
            del self._buffer[: len(self._buffer) - _MAX_BUFFERED]
        await self.flush_if_due()

    async def flush_if_due(self) -> None:
        if self._buffer and self._clock() - self._last_flush >= self.digest_seconds:
            await self.flush()

    async def flush(self) -> None:
        """Send everything held so far as one digest (split if it is long)."""
        if not self._buffer or not self._telegram_ready:
            self._buffer.clear()
            self._duplicates = 0
            self._last_flush = self._clock()
            return
        hours = self.digest_seconds / 3600.0
        header = f"<b>{hours:g}-hour update</b> \u2014 {len(self._buffer)} message{'s' if len(self._buffer) != 1 else ''}"
        if self._duplicates:
            header += f" ({self._duplicates} repeat{'s' if self._duplicates != 1 else ''} hidden)"
        chunks: list[str] = []
        current = header
        for text in self._buffer:
            if len(current) + len(_SEPARATOR) + len(text) > _CHUNK_CHARS and current != header:
                chunks.append(current)
                current = text
            else:
                current = f"{current}{_SEPARATOR}{text}"
        chunks.append(current)
        self._buffer.clear()
        self._duplicates = 0
        self._last_flush = self._clock()
        for chunk in chunks:
            await self.telegram.send_message(chunk)  # type: ignore[union-attr]

    # ── notifications ───────────────────────────────────────────────

    async def notify_trade(self, result: dict[str, Any]) -> None:
        self.logger.info("Trade cycle result: %s", result.get("modules", {}))
        if self._telegram_ready:
            await self._route(self.telegram.send_trade_alert, result)  # type: ignore[union-attr]

    async def notify_heartbeat(self, status: dict[str, Any]) -> None:
        equity = status.get("equity", 0)
        positions = len(status.get("open_positions", []))
        self.logger.info("Heartbeat: equity=$%.2f positions=%d", equity, positions)
        if self._telegram_ready:
            await self._route(self.telegram.send_heartbeat, status)  # type: ignore[union-attr]

    async def notify_risk(self, risk: dict[str, Any]) -> None:
        self.logger.warning("Risk alert: %s", risk)
        if self._telegram_ready:
            await self._route(self.telegram.send_risk_alert, risk)  # type: ignore[union-attr]

    async def notify_error(self, error_msg: str) -> None:
        self.logger.error("Bot error: %s", error_msg)
        if self._telegram_ready:
            # The kill switch firing is the one thing worth interrupting you for.
            urgent = str(error_msg).lower().startswith("kill switch")
            await self._route(self.telegram.send_error, error_msg, urgent=urgent)  # type: ignore[union-attr]

    async def notify_leads(
        self,
        strategy: str,
        leads: list[dict[str, Any]],
        max_per_message: int = 10,
    ) -> None:
        if not leads:
            return
        self.logger.info("Leads (%s): %d", strategy, len(leads))
        if self._telegram_ready:
            await self._route(  # type: ignore[union-attr]
                self.telegram.send_leads_alert, strategy, leads, max_per_message=max_per_message,
            )

    async def notify_startup(self, paper_mode: bool, vol_targeting_enabled: bool) -> None:
        msg = f"Trading bot started. Paper mode: {paper_mode}"
        self.logger.info(msg)

    async def notify_shutdown(self) -> None:
        msg = "Trading bot shutting down."
        self.logger.info(msg)
        await self.flush()  # don't lose held alerts across a restart/redeploy
