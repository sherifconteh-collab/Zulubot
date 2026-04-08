from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class BotRuntimeLockError(RuntimeError):
    """Raised when another bot instance already owns the runtime lock."""


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass
class BotRuntimeLock:
    path: Path
    label: str
    _acquired: bool = False

    def _read_existing(self) -> dict[str, object]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "label": self.label,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                existing = self._read_existing()
                existing_pid = int(existing.get("pid", 0) or 0)
                if existing_pid and _pid_is_running(existing_pid):
                    existing_label = str(existing.get("label", "bot"))
                    raise BotRuntimeLockError(
                        f"Another bot instance is already running (PID {existing_pid}, mode {existing_label}). "
                        "Stop it before starting a second live bot terminal or daemon."
                    )
                with contextlib.suppress(FileNotFoundError):
                    self.path.unlink()
                continue

            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
            self._acquired = True
            return

    def release(self) -> None:
        if not self._acquired:
            return
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        self._acquired = False


@contextlib.contextmanager
def acquire_bot_lock(data_dir: Path, label: str) -> Iterator[None]:
    lock = BotRuntimeLock(data_dir / "bot_runtime.lock", label)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()