"""Scheduler daemon for autonomous trading bot operation.

Runs as a long-lived process, triggering strategy cycles on configurable intervals.
Designed to be launched by Claude Cowork or as a standalone daemon.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import traceback
from datetime import datetime
from typing import Any

from trading_bot.config import AppConfig, SchedulerConfig
from trading_bot.leads import extract_leads
from trading_bot.notifications import NotificationDispatcher
from trading_bot.service import TradingBotService
from trading_bot.telegram_bot import TradingTelegramBot


class SchedulerService:
    def __init__(
        self,
        config: AppConfig,
        service: TradingBotService,
        telegram: TradingTelegramBot,
        notifier: NotificationDispatcher,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.scheduler_config = config.scheduler
        self.service = service
        self.telegram = telegram
        self.notifier = notifier
        self.logger = logger.getChild("scheduler")
        self._shutdown_requested = False
        self._last_crypto_run: datetime | None = None
        self._last_btc5min_run: datetime | None = None
        self._last_polymarket_run: datetime | None = None
        self._last_rotation_run: datetime | None = None
        self._last_predictions_run: datetime | None = None
        self._last_stock_run: datetime | None = None
        self._last_heartbeat: datetime | None = None
        self._last_latency_arb_run: datetime | None = None
        self._last_markov_run: datetime | None = None
        self._last_options_run: datetime | None = None
        self._last_kalshi_perps_run: datetime | None = None
        self._last_kalshi_sports_run: datetime | None = None
        self._paused_by_kill_switch = False

        # When latency arb is active, tighten the poll loop to its scan interval
        if config.latency_arb.enabled:
            self._effective_poll_interval = min(
                config.scheduler.poll_interval_seconds,
                config.latency_arb.scan_interval_seconds,
            )
        else:
            self._effective_poll_interval = config.scheduler.poll_interval_seconds

    async def run_forever(self) -> None:
        """Main daemon loop. Runs until shutdown signal received."""
        self._install_signal_handlers()
        await self.telegram.start()
        await self.notifier.notify_startup(self.config.paper_mode, self.config.volatility.enabled)

        self.logger.info(
            "Scheduler started. crypto=%dm btc5min=%dm rotation=%dm predictions=%dm stock_hour=%d "
            "heartbeat=%dm poll=%.1fs latency_arb=%s(%.1fs)",
            self.scheduler_config.crypto_interval_minutes,
            self.scheduler_config.btc5min_interval_minutes,
            self.config.altcoin_rotation.check_interval_minutes,
            self.config.predictions.check_interval_minutes,
            self.scheduler_config.stock_check_hour,
            self.scheduler_config.heartbeat_interval_minutes,
            self._effective_poll_interval,
            "enabled" if self.config.latency_arb.enabled else "disabled",
            self.config.latency_arb.scan_interval_seconds,
        )

        try:
            while not self._shutdown_requested:
                await self._tick()
                await asyncio.sleep(self._effective_poll_interval)
        except asyncio.CancelledError:
            self.logger.info("Scheduler cancelled.")
        finally:
            await self.notifier.notify_shutdown()
            await self.telegram.stop()

    async def run_once(self) -> dict[str, Any]:
        """Run a single full cycle (all modules) and return results. For testing."""
        self.logger.info("Running single cycle...")

        # Start Telegram so messages actually get delivered
        await self.telegram.start()

        result = self.service.run("all")

        # Send notifications so Telegram gets the colorful output
        await self.notifier.notify_trade(result)

        # Also send heartbeat with current status
        try:
            status = self.service.status()
            await self.notifier.notify_heartbeat(status)
        except Exception as exc:
            self.logger.error("Heartbeat in run_once failed: %s", exc)

        # Clean shutdown
        await self.telegram.stop()

        return result

    async def _tick(self) -> None:
        now = datetime.now(self.config.timezone)
        module_results: list[dict[str, Any]] = []
        trade_notification_sent = False

        # Check if paused via Telegram
        if self.telegram.paused:
            if self._should_heartbeat(now):
                status = self.service.status()
                await self.notifier.notify_heartbeat(status)
                self._last_heartbeat = now
            # A kill-switch pause (not a manual /pause) must not stop sports betting.
            if self._sports_runs_through_kill_switch() and self._should_run_kalshi_sports(now):
                result = await self._run_module("kalshi_sports", now)
                self._last_kalshi_sports_run = now
                if result is not None:
                    await self.notifier.notify_trade(result)
            return
        self._paused_by_kill_switch = False

        # Latency arb cycle (highest priority — runs every scan_interval_seconds)
        if self._should_run_latency_arb(now):
            result = await self._run_module("latency_arb", now)
            if result is not None:
                module_results.append(result)
            self._last_latency_arb_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # Crypto cycle
        if self._should_run_crypto(now):
            result = await self._run_module("crypto", now)
            if result is not None:
                module_results.append(result)
            self._last_crypto_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # BTC 5-min spread monitor
        if self._should_run_btc5min(now):
            result = await self._run_module("btc5min", now)
            if result is not None:
                module_results.append(result)
            self._last_btc5min_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # Altcoin rotation cycle
        if self._should_run_rotation(now):
            result = await self._run_module("rotation", now)
            if result is not None:
                module_results.append(result)
            self._last_rotation_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # Markov Chain regime cycle — runs before gated modules (predictions, polymarket)
        # so that state.markov_regime_signal is fresh when those entry scans execute.
        if self._should_run_markov(now):
            result = await self._run_module("markov", now)
            if result is not None:
                module_results.append(result)
            self._last_markov_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # Predictions cycle
        if self._should_run_predictions(now):
            result = await self._run_module("predictions", now)
            if result is not None:
                module_results.append(result)
            self._last_predictions_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # Polymarket cycle
        if self._should_run_polymarket(now):
            result = await self._run_module("polymarket", now)
            if result is not None:
                module_results.append(result)
            self._last_polymarket_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # Stock cycle (only at configured hour, once per day)
        if self._should_run_stocks(now):
            result = await self._run_module("stocks", now)
            if result is not None:
                module_results.append(result)
            self._last_stock_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # Kalshi Perpetuals cycle
        if self._should_run_kalshi_perps(now):
            result = await self._run_module("kalshi_perps", now)
            if result is not None:
                module_results.append(result)
            self._last_kalshi_perps_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # Kalshi sports cycle
        if self._should_run_kalshi_sports(now):
            result = await self._run_module("kalshi_sports", now)
            if result is not None:
                module_results.append(result)
            self._last_kalshi_sports_run = now
            if (
                result is not None
                and result.get("risk", {}).get("kill_switch_active")
                and not self.config.kalshi_sports.ignore_global_halt
            ):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        # Options cycle
        if self._should_run_options(now):
            result = await self._run_module("options", now)
            if result is not None:
                module_results.append(result)
            self._last_options_run = now
            if result is not None and result.get("risk", {}).get("kill_switch_active"):
                await self._finalize_trade_notifications(module_results)
                trade_notification_sent = bool(module_results)
                await self._handle_kill_switch(result)
                return

        if module_results:
            await self._finalize_trade_notifications(module_results)
            trade_notification_sent = True

        # Heartbeat
        if self._should_heartbeat(now):
            if not trade_notification_sent:
                try:
                    status = self.service.status()
                    await self.notifier.notify_heartbeat(status)
                    risk = status.get("risk", {})
                    if risk.get("daily_halt_active") or risk.get("kill_switch_active"):
                        await self.notifier.notify_risk(risk)
                except Exception as exc:
                    self.logger.error("Heartbeat failed: %s", exc)
            self._last_heartbeat = now

    async def _run_module(self, command: str, now: datetime) -> dict[str, Any] | None:
        try:
            self.logger.info("Running %s cycle at %s", command, now.isoformat())
            result = self.service.run(command)
        except Exception as exc:
            error_msg = f"{command} module failed: {exc}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            await self.notifier.notify_error(f"{command} failed: {exc}")
            return None

        if result is not None:
            try:
                await self._dispatch_leads(command, result, now)
            except Exception as exc:
                # Lead notification must never break the trading loop.
                self.logger.error("Lead dispatch for %s failed: %s", command, exc)
        return result

    async def _dispatch_leads(self, strategy: str, result: dict[str, Any], now: datetime) -> None:
        if not self.config.leads.enabled:
            return
        modules = result.get("modules")
        module_result = modules.get(strategy, result) if isinstance(modules, dict) else result
        leads = extract_leads(strategy, module_result)
        self.service.update_last_leads(strategy, leads)
        if not leads:
            return
        state = self.service.state_store.load()
        today = now.date().isoformat()
        if state.notified_leads_date != today:
            state.notified_leads = {}
            state.notified_leads_date = today
        already = set(state.notified_leads.get(strategy, []))
        new_leads = [lead for lead in leads if lead["symbol"] not in already]
        if not new_leads:
            return
        await self.notifier.notify_leads(
            strategy,
            new_leads,
            max_per_message=self.config.leads.max_per_message,
        )
        state.notified_leads[strategy] = sorted(already | {lead["symbol"] for lead in new_leads})
        self.service.state_store.save(state)

    async def _finalize_trade_notifications(self, results: list[dict[str, Any]]) -> None:
        combined = self._combine_results(results)
        await self.notifier.notify_trade(combined)

        risk = combined.get("risk", {})
        if risk.get("daily_halt_active") or risk.get("kill_switch_active"):
            await self.notifier.notify_risk(risk)

    def _sports_runs_through_kill_switch(self) -> bool:
        cfg = self.config.kalshi_sports
        return cfg.enabled and cfg.ignore_global_halt and self._paused_by_kill_switch

    async def _handle_kill_switch(self, result: dict[str, Any]) -> None:
        risk = result.get("risk", {})
        if risk.get("kill_switch_active"):
            self.telegram._paused = True
            self._paused_by_kill_switch = True
            await self.notifier.notify_error("Kill switch triggered. Bot paused.")

    @staticmethod
    def _combine_results(results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {}

        combined = dict(results[-1])
        modules: dict[str, Any] = {}
        for result in results:
            modules.update(result.get("modules", {}))

        combined["modules"] = modules
        combined["timestamp"] = results[-1].get("timestamp")
        combined["pre_risk_equity"] = results[0].get("pre_risk_equity", combined.get("pre_risk_equity"))
        combined["post_risk_equity"] = results[-1].get("post_risk_equity", combined.get("post_risk_equity"))
        return combined

    def _should_run_latency_arb(self, now: datetime) -> bool:
        if not self.config.latency_arb.enabled:
            return False
        if self._last_latency_arb_run is None:
            return True
        elapsed = (now - self._last_latency_arb_run).total_seconds()
        return elapsed >= self.config.latency_arb.scan_interval_seconds

    def _should_run_crypto(self, now: datetime) -> bool:
        if self._last_crypto_run is None:
            return True
        elapsed = (now - self._last_crypto_run).total_seconds()
        return elapsed >= self.scheduler_config.crypto_interval_minutes * 60

    def _should_run_btc5min(self, now: datetime) -> bool:
        if not self.config.btc5min.enabled:
            return False
        if self._last_btc5min_run is None:
            return True
        elapsed = (now - self._last_btc5min_run).total_seconds()
        return elapsed >= self.scheduler_config.btc5min_interval_minutes * 60

    def _should_run_rotation(self, now: datetime) -> bool:
        if not self.config.altcoin_rotation.enabled:
            return False
        if self._last_rotation_run is None:
            return True
        elapsed = (now - self._last_rotation_run).total_seconds()
        return elapsed >= self.config.altcoin_rotation.check_interval_minutes * 60

    def _should_run_predictions(self, now: datetime) -> bool:
        if not self.config.predictions.enabled:
            return False
        if self._last_predictions_run is None:
            return True
        elapsed = (now - self._last_predictions_run).total_seconds()
        # Uses same interval as crypto (scheduler config) or prediction config, whichever is set
        return elapsed >= self.config.predictions.check_interval_minutes * 60

    def _should_run_polymarket(self, now: datetime) -> bool:
        if not self.config.polymarket_strategy.enabled:
            return False
        if self._last_polymarket_run is None:
            return True
        elapsed = (now - self._last_polymarket_run).total_seconds()
        return elapsed >= self.scheduler_config.polymarket_interval_minutes * 60

    def _should_run_stocks(self, now: datetime) -> bool:
        if now.hour != self.scheduler_config.stock_check_hour:
            return False
        if self._last_stock_run is not None and self._last_stock_run.date() == now.date():
            return False
        return True

    def _should_run_markov(self, now: datetime) -> bool:
        if not self.config.markov.enabled:
            return False
        if self._last_markov_run is None:
            return True
        elapsed = (now - self._last_markov_run).total_seconds()
        return elapsed >= self.config.markov.run_interval_minutes * 60

    def _should_run_kalshi_sports(self, now: datetime) -> bool:
        if not self.config.kalshi_sports.enabled:
            return False
        if self._last_kalshi_sports_run is None:
            return True
        elapsed = (now - self._last_kalshi_sports_run).total_seconds()
        return elapsed >= self.config.kalshi_sports.check_interval_minutes * 60

    def _should_run_kalshi_perps(self, now: datetime) -> bool:
        if not self.config.kalshi_perps.enabled:
            return False
        if self._last_kalshi_perps_run is None:
            return True
        elapsed = (now - self._last_kalshi_perps_run).total_seconds()
        return elapsed >= self.config.kalshi_perps.check_interval_minutes * 60

    def _should_run_options(self, now: datetime) -> bool:
        if not self.config.options.enabled:
            return False
        if self._last_options_run is None:
            return True
        elapsed = (now - self._last_options_run).total_seconds()
        return elapsed >= self.config.options.check_interval_minutes * 60

    def _should_heartbeat(self, now: datetime) -> bool:
        if not self.scheduler_config.heartbeat_enabled:
            return False
        if self._last_heartbeat is None:
            return True
        elapsed = (now - self._last_heartbeat).total_seconds()
        return elapsed >= self.scheduler_config.heartbeat_interval_minutes * 60

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler for all signals
                signal.signal(sig, lambda s, f: self._handle_shutdown())

    def _handle_shutdown(self) -> None:
        self.logger.info("Shutdown signal received.")
        self._shutdown_requested = True
