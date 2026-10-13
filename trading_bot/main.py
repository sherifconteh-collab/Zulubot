from __future__ import annotations

import argparse
import asyncio
import json

from trading_bot.config import AppConfig
from trading_bot.logging_utils import setup_logging
from trading_bot.runtime_lock import BotRuntimeLockError, acquire_bot_lock
from trading_bot.service import TradingBotService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three-module paper trading bot")
    parser.add_argument(
        "command",
        choices=[
            "stocks", "crypto", "polymarket", "rotation",
            "latency_arb", "kalshi_sports", "gdp_nowcast", "btc5min", "markov", "all", "status",
            "run-daemon", "backtest", "hyperopt", "download-data", "btc-terminal", "doctor",
        ],
        help=(
            "Which bot module to run. 'run-daemon' for autonomous operation. "
            "'backtest'/'hyperopt' for Freqtrade analysis. 'doctor' checks which settings are present."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a stock screen even if the weekly interval has not elapsed.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single daemon cycle then exit (for testing).",
    )
    parser.add_argument(
        "--timerange",
        type=str,
        default=None,
        help="Date range for backtest/hyperopt (e.g., 20250101-20260328).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of hyperopt epochs (default: 100).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="SharpeHyperOptLoss",
        help="Hyperopt loss function (default: SharpeHyperOptLoss).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="CryptoSentimentStrategy",
        help="Freqtrade strategy class name (default: CryptoSentimentStrategy).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe for data download (default: 1h).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        # Before AppConfig/service: it must work with an empty environment and never touch the network.
        import os
        from trading_bot.doctor import run as run_doctor
        return run_doctor(os.environ)

    config = AppConfig.load()
    logger = setup_logging(config.log_dir, config.log_level)
    service = TradingBotService(config, logger)

    if args.command == "run-daemon":
        return _run_daemon(config, service, logger, once=args.once)

    if args.command == "btc-terminal":
        from trading_bot.btc_terminal import run_terminal
        try:
            with acquire_bot_lock(config.data_dir, "btc-terminal"):
                asyncio.run(run_terminal(config, logger, service))
            return 0
        except BotRuntimeLockError as exc:
            logger.error("%s", exc)
            print(f"ERROR: {exc}")
            return 1

    if args.command in ("backtest", "hyperopt", "download-data"):
        return _run_freqtrade(config, args, logger)

    if args.command == "status":
        summary = service.status()
    else:
        summary = service.run(args.command, force=args.force)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_daemon(config: AppConfig, service: TradingBotService, logger, once: bool = False) -> int:
    from trading_bot.notifications import NotificationDispatcher
    from trading_bot.scheduler import SchedulerService
    from trading_bot.telegram_bot import TradingTelegramBot

    telegram = TradingTelegramBot(config.telegram, logger)
    telegram.set_service(service)
    notifier = NotificationDispatcher(telegram, logger, digest_hours=config.telegram.digest_hours)
    scheduler = SchedulerService(config, service, telegram, notifier, logger)

    if once:
        result = asyncio.run(scheduler.run_once())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    logger.info("Starting trading bot daemon...")
    try:
        with acquire_bot_lock(config.data_dir, "run-daemon"):
            asyncio.run(scheduler.run_forever())
        return 0
    except BotRuntimeLockError as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}")
        return 1


def _run_freqtrade(config: AppConfig, args: argparse.Namespace, logger) -> int:
    from trading_bot.integrations.freqtrade_adapter import FreqtradeAdapter

    adapter = FreqtradeAdapter(config)

    if not adapter.is_available:
        print("ERROR: Freqtrade is not available.")
        print("  1. Install: pip install -r requirements-freqtrade.txt")
        print("  2. Enable: Set FREQTRADE_ENABLED=true in .env")
        return 1

    try:
        if args.command == "download-data":
            result = adapter.download_data(
                timerange=args.timerange,
                timeframe=args.timeframe,
            )
        elif args.command == "backtest":
            result = adapter.run_backtest(
                timerange=args.timerange,
                strategy=args.strategy,
            )
        elif args.command == "hyperopt":
            result = adapter.run_hyperopt(
                epochs=args.epochs,
                timerange=args.timerange,
                strategy=args.strategy,
                loss=args.loss,
            )
        else:
            print(f"Unknown freqtrade command: {args.command}")
            return 1

        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    except Exception as exc:
        logger.error("Freqtrade command failed: %s", exc)
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
