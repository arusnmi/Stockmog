from __future__ import annotations

import os
import time
from typing import Callable


def format_duration(seconds: float) -> str:
    """Format a duration using the most readable unit."""
    if seconds != seconds or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes:02d}m"


def hardware_summary(torch_module, device) -> str:
    import psutil

    memory = psutil.virtual_memory()
    parts = [
        f"CPU {psutil.cpu_count(logical=False) or '?'} cores/"
        f"{os.cpu_count()} threads",
        f"RAM {memory.total / 1024**3:.1f} GiB",
    ]
    if device.type == "cuda":
        properties = torch_module.cuda.get_device_properties(device)
        parts.append(
            f"GPU {properties.name} {properties.total_memory / 1024**3:.1f} GiB"
        )
    else:
        parts.append("GPU none (CPU training)")
    return ", ".join(parts)


def benchmark_seconds_per_batch(
    step: Callable[[], None],
    *,
    warmup: int = 3,
    measured: int = 12,
    synchronize: Callable[[], None] = lambda: None,
) -> float:
    """Time real training steps on this machine."""
    for _ in range(warmup):
        step()
    synchronize()
    started = time.perf_counter()
    for _ in range(measured):
        step()
    synchronize()
    return (time.perf_counter() - started) / measured


class RoundProgress:
    """Reports time left for a fixed-maximum number of boosting rounds."""

    def __init__(
        self,
        *,
        label: str,
        total_rounds: int,
        estimated_seconds_per_round: float,
        report_interval_seconds: float = 60.0,
    ) -> None:
        self.label = label
        self.total_rounds = total_rounds
        self.seconds_per_round = estimated_seconds_per_round
        self.report_interval = report_interval_seconds
        self._started = time.perf_counter()
        self._last_report = self._started
        self._rounds_timed = 0

    def describe_estimate(self, rounds_done: int = 0) -> str:
        remaining = max(0, self.total_rounds - rounds_done)
        return (
            f"{self.label} estimate: {self.seconds_per_round:.2f}s per round, "
            f"up to {format_duration(remaining * self.seconds_per_round)} for "
            f"{remaining} rounds (early stopping may finish sooner)"
        )

    def round_done(self, rounds_done: int) -> str | None:
        now = time.perf_counter()
        self._rounds_timed += 1
        if self._rounds_timed >= 5:
            self.seconds_per_round = (now - self._started) / self._rounds_timed
        if now - self._last_report < self.report_interval:
            return None
        self._last_report = now
        left = max(0, self.total_rounds - rounds_done) * self.seconds_per_round
        return (
            f"{self.label} round {rounds_done}/{self.total_rounds} "
            f"{100 * rounds_done / self.total_rounds:.1f}% | "
            f"max remaining {format_duration(left)}"
        )


class EpochProgress:
    """Tracks measured speed and reports time left in the epoch and run."""

    def __init__(
        self,
        *,
        label: str,
        total_epochs: int,
        batches_per_epoch: int,
        estimated_seconds_per_batch: float,
        estimated_validation_seconds: float,
        report_interval_seconds: float = 60.0,
    ) -> None:
        self.label = label
        self.total_epochs = total_epochs
        self.batches_per_epoch = max(1, batches_per_epoch)
        self.seconds_per_batch = estimated_seconds_per_batch
        self.validation_seconds = estimated_validation_seconds
        self.report_interval = report_interval_seconds
        self._epoch_started = time.perf_counter()
        self._last_report = self._epoch_started
        self._batches_done = 0
        self._measured_batches = 0
        self._measured_seconds = 0.0
        self._last_batch_time = self._epoch_started

    def estimated_epoch_seconds(self) -> float:
        return (
            self.batches_per_epoch * self.seconds_per_batch
            + self.validation_seconds
        )

    def describe_estimate(self) -> str:
        epoch = self.estimated_epoch_seconds()
        return (
            f"{self.label} estimate: {format_duration(epoch)} per epoch, up to "
            f"{format_duration(epoch * self.total_epochs)} for "
            f"{self.total_epochs} epochs (early stopping may finish sooner)"
        )

    def start_epoch(self, batches_already_done: int = 0) -> None:
        now = time.perf_counter()
        self._epoch_started = now
        self._last_report = now
        self._last_batch_time = now
        self._batches_done = batches_already_done

    def batch_done(self, epoch_index: int) -> None:
        now = time.perf_counter()
        self._measured_seconds += now - self._last_batch_time
        self._measured_batches += 1
        self._last_batch_time = now
        self._batches_done += 1
        if self._measured_batches >= 20:
            self.seconds_per_batch = (
                self._measured_seconds / self._measured_batches
            )
        if now - self._last_report < self.report_interval:
            return
        self._last_report = now
        remaining_batches = max(0, self.batches_per_epoch - self._batches_done)
        epoch_left = remaining_batches * self.seconds_per_batch + (
            self.validation_seconds
        )
        later_epochs = max(0, self.total_epochs - epoch_index - 1)
        total_left = epoch_left + later_epochs * self.estimated_epoch_seconds()
        print(
            f"{self.label} epoch {epoch_index + 1}/{self.total_epochs} "
            f"{100 * self._batches_done / self.batches_per_epoch:.1f}% | "
            f"epoch left {format_duration(epoch_left)} | "
            f"max remaining {format_duration(total_left)}",
            flush=True,
        )

    def finish_epoch(self, epoch_index: int, validation_seconds: float) -> str:
        self.validation_seconds = validation_seconds
        later_epochs = max(0, self.total_epochs - epoch_index - 1)
        elapsed = time.perf_counter() - self._epoch_started
        return (
            f"epoch time {format_duration(elapsed)}; max remaining "
            f"{format_duration(later_epochs * self.estimated_epoch_seconds())}"
        )
