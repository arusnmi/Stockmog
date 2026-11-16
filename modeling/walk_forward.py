from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .data import DatasetSplit, TrainingDataset, _datetime_to_ns


@dataclass(frozen=True)
class WalkForwardFold:
    split: DatasetSplit
    prediction_end: datetime | None
    policy_holdout: bool


def _timestamps_ns(values: tuple[datetime, ...] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype != np.dtype("int64"):
            raise ValueError("Cached timestamps must be a one-dimensional int64 array")
        return values
    return np.asarray([_datetime_to_ns(value) for value in values], dtype=np.int64)


def _utc_datetime(value: np.int64) -> datetime:
    seconds, nanoseconds = divmod(int(value), 1_000_000_000)
    if nanoseconds % 1_000:
        raise ValueError("Fold boundaries must have at most microsecond precision")
    return datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(
        microseconds=nanoseconds // 1_000
    )


def walk_forward_splits(
    dataset: TrainingDataset,
    *,
    folds: int = 3,
    warmup_fraction: float = 0.5,
    holdout_fraction: float = 0.15,
    validation_fraction: float = 0.2,
) -> tuple[WalkForwardFold, ...]:
    """Create purged expanding-window folds and a final policy holdout.

    The validation period is before each prediction period, including the
    holdout. Labels ending on a boundary are purged, not used for fitting.
    """
    if folds < 1:
        raise ValueError("folds must be at least one")
    if not 0 < warmup_fraction < 1 or not 0 < holdout_fraction < 1:
        raise ValueError("warmup and holdout fractions must be between zero and one")
    if warmup_fraction + holdout_fraction >= 1:
        raise ValueError("warmup and holdout fractions must leave a prediction period")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    timestamps = _timestamps_ns(dataset.timestamps)
    ends = _timestamps_ns(dataset.label_end_timestamps)
    if len(timestamps) != len(ends) or len(timestamps) != len(dataset.labels):
        raise ValueError("Event timestamps, label ends and labels must have equal lengths")
    if np.any(ends < timestamps):
        raise ValueError("A label end cannot precede its event timestamp")
    unique = np.unique(timestamps)
    first_prediction = int(len(unique) * warmup_fraction)
    holdout_start = int(len(unique) * (1.0 - holdout_fraction))
    if first_prediction < 2 or holdout_start >= len(unique):
        raise ValueError("Not enough distinct timestamps for warmup and holdout")
    if holdout_start - first_prediction < folds:
        raise ValueError("Not enough distinct timestamps for the requested prediction folds")

    boundaries = [
        first_prediction + (holdout_start - first_prediction) * index // folds
        for index in range(folds + 1)
    ]
    results = []
    for fold_index, start_position in enumerate(boundaries):
        start = unique[start_position]
        validation_position = int(start_position * (1.0 - validation_fraction))
        validation_position = min(max(validation_position, 1), start_position - 1)
        validation_start = unique[validation_position]
        prediction_end = (
            unique[boundaries[fold_index + 1]]
            if fold_index < folds
            else None
        )
        train = np.flatnonzero(
            (timestamps < validation_start) & (ends < validation_start)
        )
        validation = np.flatnonzero(
            (timestamps >= validation_start) & (timestamps < start) & (ends < start)
        )
        prediction_mask = timestamps >= start
        if prediction_end is not None:
            prediction_mask &= timestamps < prediction_end
        prediction = np.flatnonzero(prediction_mask)
        if min(len(train), len(validation), len(prediction)) == 0:
            raise ValueError(
                f"Walk-forward fold {fold_index + 1} is empty after purging labels; "
                "add more history or reduce the label horizon"
            )
        if len(np.unique(dataset.labels[train])) < 2:
            raise ValueError(
                f"Walk-forward fold {fold_index + 1} training period must contain "
                "at least two outcome classes"
            )
        results.append(
            WalkForwardFold(
                split=DatasetSplit(
                    train=train,
                    validation=validation,
                    test=prediction,
                    validation_start=_utc_datetime(validation_start),
                    test_start=_utc_datetime(start),
                ),
                prediction_end=(
                    _utc_datetime(prediction_end) if prediction_end is not None else None
                ),
                policy_holdout=fold_index == folds,
            )
        )
    return tuple(results)
