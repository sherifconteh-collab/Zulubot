"""Freqtrade adapter — bridge between our trading bot and Freqtrade's
backtesting and hyperopt engines.

Provides a clean interface to run backtests and hyperparameter optimization
without requiring the user to understand Freqtrade's internal CLI.

Usage:
    python -m trading_bot backtest --timerange 20250101-20260328
    python -m trading_bot hyperopt --epochs 100 --loss SharpeHyperOptLoss
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_bot.config import AppConfig
from trading_bot.integrations.freqtrade_config import (
    generate_freqtrade_config,
    generate_hyperopt_config,
)

logger = logging.getLogger(__name__)

try:
    import freqtrade  # noqa: F401
    FREQTRADE_AVAILABLE = True
except ImportError:
    FREQTRADE_AVAILABLE = False


class FreqtradeAdapter:
    """Bridge to Freqtrade backtesting and hyperopt.

    Manages config generation, data download, backtest execution,
    and result parsing. Keeps all temporary files in the bot's data directory.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ft_dir = config.data_dir / "freqtrade"
        self.ft_config_path = self.ft_dir / "config.json"
        self.strategy_dir = Path(__file__).parent  # integrations/ directory
        self.results_dir = self.ft_dir / "results"

    @property
    def is_available(self) -> bool:
        return FREQTRADE_AVAILABLE and self.config.freqtrade.enabled

    def run_backtest(
        self,
        *,
        timerange: str | None = None,
        strategy: str = "CryptoSentimentStrategy",
        download_data: bool = True,
    ) -> dict[str, Any]:
        """Run a Freqtrade backtest.

        Args:
            timerange: Date range like "20250101-20260328". Defaults to last 90 days.
            strategy: Strategy class name to backtest.
            download_data: Whether to download historical data first.

        Returns:
            Backtest results dict with profit, trades, drawdown, etc.
        """
        self._ensure_available()
        self._ensure_dirs()

        # Generate config
        ft_config = generate_freqtrade_config(self.config, dry_run=True)
        self._write_config(ft_config)

        if timerange is None:
            # Default: last 90 days
            end = datetime.now()
            start = end.replace(day=1)
            start = start.replace(month=max(1, start.month - 3))
            timerange = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"

        # Download data if requested
        if download_data:
            self._download_data(timerange)

        # Run backtest
        logger.info("Running backtest: strategy=%s timerange=%s", strategy, timerange)
        cmd = [
            sys.executable, "-m", "freqtrade", "backtesting",
            "--config", str(self.ft_config_path),
            "--strategy", strategy,
            "--strategy-path", str(self.strategy_dir),
            "--timerange", timerange,
            "--export", "trades",
            "--export-filename", str(self.results_dir / "backtest_results.json"),
        ]

        result = self._run_command(cmd, "backtest")

        # Parse results
        return self._parse_backtest_results(result)

    def run_hyperopt(
        self,
        *,
        epochs: int = 100,
        timerange: str | None = None,
        strategy: str = "CryptoSentimentStrategy",
        loss: str = "SharpeHyperOptLoss",
        spaces: list[str] | None = None,
        download_data: bool = True,
    ) -> dict[str, Any]:
        """Run Freqtrade hyperparameter optimization.

        Args:
            epochs: Number of optimization epochs.
            timerange: Date range for optimization.
            strategy: Strategy class to optimize.
            loss: Loss function (SharpeHyperOptLoss, OnlyProfitHyperOptLoss, etc.).
            spaces: Search spaces (buy, sell, roi, stoploss, trailing).
            download_data: Whether to download historical data first.

        Returns:
            Hyperopt results dict with best parameters and metrics.
        """
        self._ensure_available()
        self._ensure_dirs()

        if spaces is None:
            spaces = ["buy", "sell", "roi", "stoploss"]

        # Generate configs
        ft_config = generate_freqtrade_config(self.config, dry_run=True)
        self._write_config(ft_config)
        hyperopt_config = generate_hyperopt_config(
            self.config, epochs=epochs, spaces=spaces, loss_function=loss,
        )

        if timerange is None:
            end = datetime.now()
            start = end.replace(day=1)
            start = start.replace(month=max(1, start.month - 3))
            timerange = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"

        if download_data:
            self._download_data(timerange)

        logger.info(
            "Running hyperopt: strategy=%s epochs=%d loss=%s spaces=%s timerange=%s",
            strategy, epochs, loss, spaces, timerange,
        )

        cmd = [
            sys.executable, "-m", "freqtrade", "hyperopt",
            "--config", str(self.ft_config_path),
            "--strategy", strategy,
            "--strategy-path", str(self.strategy_dir),
            "--hyperopt-loss", loss,
            "--spaces", *spaces,
            "--epochs", str(epochs),
            "--timerange", timerange,
        ]

        result = self._run_command(cmd, "hyperopt")

        return self._parse_hyperopt_results(result)

    def download_data(
        self,
        *,
        timerange: str | None = None,
        timeframe: str = "1h",
    ) -> dict[str, Any]:
        """Download historical data for backtesting.

        Args:
            timerange: Date range to download.
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d).

        Returns:
            Download result summary.
        """
        self._ensure_available()
        self._ensure_dirs()

        ft_config = generate_freqtrade_config(self.config, dry_run=True)
        self._write_config(ft_config)

        if timerange is None:
            timerange = "20250101-"

        self._download_data(timerange, timeframe)
        return {"status": "ok", "timerange": timerange, "timeframe": timeframe}

    # ── Internal methods ───────────────────────────────────────────

    def _ensure_available(self) -> None:
        if not FREQTRADE_AVAILABLE:
            raise RuntimeError(
                "Freqtrade is not installed. Install with: pip install -r requirements-freqtrade.txt"
            )
        if not self.config.freqtrade.enabled:
            raise RuntimeError(
                "Freqtrade is not enabled. Set FREQTRADE_ENABLED=true in .env"
            )

    def _ensure_dirs(self) -> None:
        self.ft_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _write_config(self, ft_config: dict[str, Any]) -> None:
        self.ft_config_path.write_text(json.dumps(ft_config, indent=2), encoding="utf-8")

    def _download_data(self, timerange: str, timeframe: str = "1h") -> None:
        logger.info("Downloading data: timerange=%s timeframe=%s", timerange, timeframe)
        cmd = [
            sys.executable, "-m", "freqtrade", "download-data",
            "--config", str(self.ft_config_path),
            "--timerange", timerange,
            "--timeframe", timeframe,
            "--datadir", str(self.ft_dir / "data"),
        ]
        self._run_command(cmd, "download-data")

    def _run_command(self, cmd: list[str], label: str) -> subprocess.CompletedProcess:
        """Run a freqtrade subprocess command."""
        logger.info("Running %s: %s", label, " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
                cwd=str(self.ft_dir),
            )
            if result.returncode != 0:
                logger.error("%s failed (exit %d):\n%s", label, result.returncode, result.stderr)
                raise RuntimeError(f"Freqtrade {label} failed: {result.stderr[:500]}")
            logger.info("%s completed successfully", label)
            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Freqtrade {label} timed out after 1 hour")

    def _parse_backtest_results(self, result: subprocess.CompletedProcess) -> dict[str, Any]:
        """Parse backtest output into a summary dict."""
        # Try to read exported results file
        results_file = self.results_dir / "backtest_results.json"
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text(encoding="utf-8"))
                # Extract key metrics from Freqtrade's results format
                strategy_results = {}
                if isinstance(data, dict) and "strategy" in data:
                    for strat_name, strat_data in data["strategy"].items():
                        strategy_results = {
                            "strategy": strat_name,
                            "total_trades": strat_data.get("total_trades", 0),
                            "profit_total": strat_data.get("profit_total", 0),
                            "profit_total_pct": strat_data.get("profit_total_abs", 0),
                            "max_drawdown": strat_data.get("max_drawdown", 0),
                            "max_drawdown_pct": strat_data.get("max_drawdown_abs", 0),
                            "sharpe_ratio": strat_data.get("sharpe", 0),
                            "sortino_ratio": strat_data.get("sortino", 0),
                            "win_rate": strat_data.get("wins", 0) / max(1, strat_data.get("total_trades", 1)),
                            "avg_profit_per_trade": strat_data.get("profit_mean", 0),
                            "holding_avg": strat_data.get("holding_avg", ""),
                        }
                        break  # Take first strategy

                return {
                    "status": "ok",
                    "results_file": str(results_file),
                    **strategy_results,
                    "stdout_tail": result.stdout[-2000:] if result.stdout else "",
                }
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Failed to parse backtest results file: %s", exc)

        # Fallback: return raw stdout
        return {
            "status": "ok",
            "results_file": str(results_file),
            "stdout": result.stdout[-3000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }

    def _parse_hyperopt_results(self, result: subprocess.CompletedProcess) -> dict[str, Any]:
        """Parse hyperopt output into a summary dict."""
        output = result.stdout or ""

        # Extract best result from stdout (Freqtrade prints it)
        best_params: dict[str, Any] = {}
        for line in output.split("\n"):
            line = line.strip()
            if "Best result" in line or "best epoch" in line.lower():
                best_params["best_line"] = line

        return {
            "status": "ok",
            "best_params": best_params,
            "stdout_tail": output[-3000:],
            "stderr_tail": (result.stderr or "")[-1000:],
        }
