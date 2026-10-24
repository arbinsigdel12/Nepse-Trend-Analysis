"""Saving and loading raw data files."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .config import Paths
from .normalize import STANDARD, normalize_frame, rows_to_frame


def raw_day_path(p: Paths, day: date) -> Path:
    return p.raw_dir / f"{day.isoformat()}.json"


def raw_day_exists(p: Paths, day: date) -> bool:
    return raw_day_path(p, day).exists()


def save_raw_day(p: Paths, day: date, rows: list) -> None:
    p.ensure()
    raw_day_path(p, day).write_text(json.dumps(rows), encoding="utf-8")


def list_raw_days(p: Paths) -> list[date]:
    if not p.raw_dir.exists():
        return []
    return sorted(date.fromisoformat(f.stem) for f in p.raw_dir.glob("*.json"))


def import_csv(p: Paths, csv_path) -> tuple[int, dict]:
    """Read a CSV you downloaded yourself, standardise it, and keep a copy."""
    p.ensure()
    df = pd.read_csv(csv_path)
    clean, report = normalize_frame(df)
    if clean.empty:
        raise ValueError("No usable rows. Check that the CSV has symbol, date and close columns.")
    out = p.imported_dir / (Path(csv_path).stem + ".parquet")
    clean.to_parquet(out, index=False)
    return len(clean), report


def build_prices_raw(p: Paths) -> pd.DataFrame:
    """Combine every saved day (API) and every imported CSV into one standard table."""
    p.ensure()
    frames = []
    if p.imported_dir.exists():
        for f in sorted(p.imported_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(f))
    for day in list_raw_days(p):
        rows = json.loads(raw_day_path(p, day).read_text(encoding="utf-8"))
        df, _ = rows_to_frame(rows, default_date=day)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No price data found. Run `fetch` or `import-csv` first.")
    df = pd.concat(frames, ignore_index=True)[STANDARD]
    df = (df.sort_values(["symbol", "date"])
            .drop_duplicates(["symbol", "date"], keep="last")
            .reset_index(drop=True))
    df.to_parquet(p.prices_raw, index=False)
    return df


def import_folder(p: Paths, folder) -> tuple[int, int]:
    """Load a folder of per-stock CSV files (one file per symbol, name = symbol)."""
    import numpy as np
    p.ensure()
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        df["symbol"] = f.stem
        if "per_change" in df.columns and "close" in df.columns:
            # the exchange's own previous close, rebuilt from its percentage change.
            # On bonus/right days the exchange resets it, which reveals the price drop.
            pc = pd.to_numeric(df["per_change"], errors="coerce")
            prev = pd.to_numeric(df["close"], errors="coerce") / (1 + pc / 100)
            df["previous_close"] = prev.where(np.isfinite(prev))
        clean, _ = normalize_frame(df)
        if not clean.empty:
            frames.append(clean)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(p.imported_dir / "nepse_data_company_wise.parquet", index=False)
    return len(out), len(frames)
