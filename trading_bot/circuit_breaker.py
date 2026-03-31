from __future__ import annotations

import time
from typing import Any, Callable


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        cooldown_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._clock = clock or time.monotonic
        self._failure_count = 0
        self._state = "closed"
        self._open_until = 0.0

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def state(self) -> str:
        return self._state

    def remaining_open_seconds(self) -> float:
        if self._state != "open":
            return 0.0
        return max(0.0, self._open_until - self._clock())

    def allow_request(self) -> bool:
        if self._state != "open":
            return True
        if self.remaining_open_seconds() > 0.0:
            return False
        self._state = "half_open"
        return True

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "closed"
        self._open_until = 0.0

    def record_failure(self, _reason: Any | None = None) -> bool:
        if self._state == "half_open":
            self._trip_open()
            return True

        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._trip_open()
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "remaining_open_seconds": round(self.remaining_open_seconds(), 2),
        }

    def _trip_open(self) -> None:
        self._state = "open"
        self._open_until = self._clock() + self.cooldown_seconds
