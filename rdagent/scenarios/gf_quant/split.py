"""Leakage-safe configurable date splits for the GF Quant scenario."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence


def _date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _ordered_dates(values: Iterable[str | date | datetime]) -> list[date]:
    dates = sorted({_date(value) for value in values})
    if not dates:
        raise ValueError("At least one trading date is required")
    return dates


@dataclass(frozen=True)
class Period:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Period start {self.start} is after end {self.end}")

    @classmethod
    def parse(cls, start: str | date, end: str | date) -> "Period":
        return cls(_date(start), _date(end))

    def as_list(self) -> list[str]:
        return [self.start.isoformat(), self.end.isoformat()]


@dataclass(frozen=True)
class DateSplit:
    train: Period
    valid: Period
    test: Period

    def __post_init__(self) -> None:
        if not self.train.end < self.valid.start:
            raise ValueError("Train and valid periods overlap or are not chronological")
        if not self.valid.end < self.test.start:
            raise ValueError("Valid and test periods overlap or are not chronological")

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": self.train.as_list(), "valid": self.valid.as_list(), "test": self.test.as_list()}


def build_ratio_split(
    trading_dates: Sequence[str | date | datetime],
    *,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    min_observations: int = 60,
) -> DateSplit:
    """Build one split from the dates actually present in the data.

    The test ratio is the remainder. Boundaries always land on observed trading
    dates and no calendar-year constants are embedded in the code.
    """

    dates = _ordered_dates(trading_dates)
    if not (0 < train_ratio < 1 and 0 < valid_ratio < 1 and train_ratio + valid_ratio < 1):
        raise ValueError("train_ratio and valid_ratio must be positive and sum to less than one")
    if len(dates) < min_observations:
        raise ValueError(f"Need at least {min_observations} dates, got {len(dates)}")
    train_end_index = max(0, min(len(dates) - 3, int(len(dates) * train_ratio) - 1))
    valid_end_index = max(train_end_index + 1, min(len(dates) - 2, int(len(dates) * (train_ratio + valid_ratio)) - 1))
    return DateSplit(
        train=Period(dates[0], dates[train_end_index]),
        valid=Period(dates[train_end_index + 1], dates[valid_end_index]),
        test=Period(dates[valid_end_index + 1], dates[-1]),
    )


def build_rolling_splits(
    trading_dates: Sequence[str | date | datetime],
    *,
    train_size: int,
    valid_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding_train: bool = True,
) -> list[DateSplit]:
    """Build walk-forward splits using counts of observed trading dates."""

    dates = _ordered_dates(trading_dates)
    sizes = (train_size, valid_size, test_size)
    if any(size <= 0 for size in sizes):
        raise ValueError("All window sizes must be positive")
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")
    total = sum(sizes)
    if len(dates) < total:
        raise ValueError(f"Need at least {total} dates, got {len(dates)}")

    result: list[DateSplit] = []
    offset = 0
    while offset + total <= len(dates):
        train_start = 0 if expanding_train else offset
        train_end = offset + train_size - 1
        valid_start = train_end + 1
        valid_end = valid_start + valid_size - 1
        test_start = valid_end + 1
        test_end = test_start + test_size - 1
        result.append(
            DateSplit(
                train=Period(dates[train_start], dates[train_end]),
                valid=Period(dates[valid_start], dates[valid_end]),
                test=Period(dates[test_start], dates[test_end]),
            )
        )
        offset += step
    return result
