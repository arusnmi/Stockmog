from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path


def _replace_with_retry(
    temporary: Path,
    destination: Path,
    *,
    retries: int = 20,
) -> None:
    for attempt in range(retries):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt + 1 == retries:
                raise
            time.sleep(0.01)


class TrainingControl:
    """Coordinate pause, resume, and stop requests through a run-directory file."""

    def __init__(self, run_dir: str | Path, *, poll_seconds: float = 1.0) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "control.json"
        self.poll_seconds = poll_seconds

    def action(self) -> str:
        if not self.path.exists():
            return "continue"
        data = None
        for attempt in range(20):
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot read training control file {self.path}"
                ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"Invalid training control file: {self.path}")
        action = data.get("action")
        if action not in {"continue", "pause", "paused", "stop", "stopped"}:
            raise ValueError(f"Invalid training control action in {self.path}")
        return action

    def request(self, action: str) -> None:
        if action not in {"continue", "pause", "stop"}:
            raise ValueError(f"Unsupported training control request: {action}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps({"action": action}, indent=2) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(temporary, self.path)

    def checkpoint_boundary(self, save_checkpoint: Callable[[], None]) -> bool:
        """Save before pausing/stopping; return false when training should stop."""
        action = self.action()
        if action == "stop":
            save_checkpoint()
            self._write_status("stopped")
            return False
        if action != "pause":
            return True

        save_checkpoint()
        self._write_status("paused")
        while True:
            action = self.action()
            if action == "continue":
                return True
            if action == "stop":
                save_checkpoint()
                self._write_status("stopped")
                return False
            time.sleep(self.poll_seconds)

    def _write_status(self, action: str) -> None:
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps({"action": action}, indent=2) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(temporary, self.path)
