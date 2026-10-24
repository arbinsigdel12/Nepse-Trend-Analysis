"""Settings in one place. Change numbers here, not deep inside the code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Modelling settings ----
HORIZON = 5                 # predict the return over the next 5 trading days
TOP_K = 10                  # evaluate and show the top 10 picks
MIN_AVG_TURNOVER = 500_000  # NPR per day (20-day average). Below this a stock is "too thin"
MIN_TRADED_DAYS_20 = 12     # must have traded on at least 12 of the last 20 days
ROUND_TRIP_COST = 0.012     # PLACEHOLDER: 1.2% total cost of buying and selling.
                            # Replace with your real broker fee + SEBON + DP fee + tax estimate.

# ---- NEPSE trading rules ----
# Daily price limit for one stock. NEPSE raised it from 10% to 15% on 2026-04-20.
LIMIT_CHANGE_DATE = date(2026, 4, 20)
OLD_LIMIT = 0.10
NEW_LIMIT = 0.15
NEPAL_UTC_OFFSET_MINUTES = 5 * 60 + 45
FETCH_AFTER_NPT = (15, 30)       # market closes about 3 PM; we wait until 3:30 PM to fetch today
SCHEDULE_CHANGE_DATE = date(2026, 4, 6)   # NEPSE moved from Sunday-Thursday to Monday-Friday


def closed_weekdays(day) -> tuple:
    """Weekdays (Monday = 0) when NEPSE is normally closed on this date."""
    return (4, 5) if day < SCHEDULE_CHANGE_DATE else (5, 6)

# ---- Data cleaning ----
ADJUST_TOLERANCE = 0.001    # |prev_close / yesterday_close - 1| above this = corporate action


def price_limit_series(dates: pd.Series) -> pd.Series:
    """Daily price limit in force on each date."""
    cutoff = pd.Timestamp(LIMIT_CHANGE_DATE)
    return pd.Series(np.where(dates >= cutoff, NEW_LIMIT, OLD_LIMIT), index=dates.index)


@dataclass
class Paths:
    """Every file the project reads or writes. Set NEPSE_DATA_DIR to move them."""
    base: Path | None = None

    def __post_init__(self) -> None:
        if self.base is None:
            self.base = Path(os.environ.get("NEPSE_DATA_DIR", PROJECT_ROOT / "data"))
        self.base = Path(self.base)

    @property
    def raw_dir(self) -> Path: return self.base / "raw" / "prices"
    @property
    def imported_dir(self) -> Path: return self.base / "imported"
    @property
    def news_dir(self) -> Path: return self.base / "news" / "raw"
    @property
    def prices_raw(self) -> Path: return self.base / "clean" / "prices_raw.parquet"
    @property
    def prices_clean(self) -> Path: return self.base / "clean" / "prices_clean.parquet"
    @property
    def validation(self) -> Path: return self.base / "clean" / "validation.json"
    @property
    def features(self) -> Path: return self.base / "features" / "features.parquet"
    @property
    def oos(self) -> Path: return self.base / "models" / "oos_predictions.parquet"
    @property
    def model(self) -> Path: return self.base / "models" / "model.txt"
    @property
    def model_meta(self) -> Path: return self.base / "models" / "model_meta.json"
    @property
    def metrics(self) -> Path: return self.base / "models" / "metrics.json"
    @property
    def daily_eval(self) -> Path: return self.base / "models" / "daily_eval.parquet"
    @property
    def latest(self) -> Path: return self.base / "predictions" / "latest.parquet"

    def ensure(self) -> None:
        for d in (self.raw_dir, self.imported_dir, self.prices_raw.parent,
                  self.features.parent, self.model.parent, self.latest.parent):
            d.mkdir(parents=True, exist_ok=True)
