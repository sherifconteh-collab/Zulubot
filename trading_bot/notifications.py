"""Unified notification dispatcher.

Routes notifications to configured channels (Telegram, console log, future webhooks).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trading_bot.telegram_bot import TradingTelegramBot


class NotificationDispatcher:
    def __init__(self, telegram: TradingTelegramBot | None, logger: logging.Logger) -> None:
        self.telegram = telegram
        self.logger = logger.getChild("notifications")

    async def notify_trade(self, result: dict[str, Any]) -> None:
        self.logger.info("Trade cycle result: %s", result.get("modules", {}))
        if self.telegram and self.telegram.is_configured:
            await self.telegram.send_trade_alert(result)

    async def notify_heartbeat(self, status: dict[str, Any]) -> None:
        equity = status.get("equity", 0)
        positions = len(status.get("open_positions", []))
        self.logger.info("Heartbeat: equity=$%.2f positions=%d", equity, positions)
        if self.telegram and self.telegram.is_configured:
            await self.telegram.send_heartbeat(status)

    async def notify_risk(self, risk: dict[str, Any]) -> None:
        self.logger.warning("Risk alert: %s", risk)
        if self.telegram and self.telegram.is_configured:
            await self.telegram.send_risk_alert(risk)

    async def notify_error(self, error_msg: str) -> None:
        self.logger.error("Bot error: %s", error_msg)
        if self.telegram and self.telegram.is_configured:
            await self.telegram.send_error(error_msg)

    async def notify_leads(
        self,
        strategy: str,
        leads: list[dict[str, Any]],
        max_per_message: int = 10,
    ) -> None:
        if not leads:
            return
        self.logger.info("Leads (%s): %d", strategy, len(leads))
        if self.telegram and self.telegram.is_configured:
            await self.telegram.send_leads_alert(strategy, leads, max_per_message=max_per_message)

    async def notify_startup(self, paper_mode: bool, vol_targeting_enabled: bool) -> None:
        msg = f"Trading bot started. Paper mode: {paper_mode}"
        self.logger.info(msg)

    async def notify_shutdown(self) -> None:
        msg = "Trading bot shutting down."
        self.logger.info(msg)
