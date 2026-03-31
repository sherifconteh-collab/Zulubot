from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any


class ModuleLatencyTracker:
    def __init__(self, window_size: int = 50) -> None:
        self.window_size = max(1, window_size)
        self._history: dict[str, deque[float]] = {}

    def record(self, name: str, duration_ms: float) -> None:
        samples = self._history.setdefault(name, deque(maxlen=self.window_size))
        samples.append(float(duration_ms))

    def snapshot(self) -> dict[str, dict[str, float]]:
        snapshot: dict[str, dict[str, float]] = {}
        for name, samples in self._history.items():
            if not samples:
                continue
            values = list(samples)
            snapshot[name] = {
                "last_ms": round(values[-1], 2),
                "avg_ms": round(sum(values) / len(values), 2),
                "max_ms": round(max(values), 2),
                "count": float(len(values)),
            }
        return snapshot


class TradeAuditLog:
    def __init__(self, path: Path, logger: logging.Logger, *, enabled: bool = False) -> None:
        self.path = path
        self.enabled = enabled
        self.logger = logger.getChild("trade_audit")

    def record(self, entry: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except Exception as exc:
            self.logger.warning("Failed to append trade audit entry: %s", exc)

    def tail(self, limit: int = 10) -> list[dict[str, Any]]:
        if limit <= 0 or not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            self.logger.warning("Failed to read trade audit log: %s", exc)
            return []

        entries: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
        return entries