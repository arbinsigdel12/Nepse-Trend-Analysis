"""Check the data, fix corporate actions, and fill gaps so every stock has a full history."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import ADJUST_TOLERANCE, Paths, price_limit_series

DATA_START = "2019-01-01"   # before this, the dataset previous-close is unreliable


def build_clean_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw standard table -> one row per stock per trading day, with adjusted prices.

    1. Removes duplicates and repairs impossible highs/lows.
    2. Adds a row for days a stock did not trade (price carried forward, volume zero).
    3. Finds bonus/right/dividend price drops. NEPSE resets 'previous close' on those
       days, so (previous close / yesterday's close) not equal to 1 reveals them.
       Older prices are scaled so the history is continuous (adj_close).
    """
    df = (raw.sort_values(["symbol", "date"])
             .drop_duplicates(["symbol", "date"], keep="last").copy())
    for c in ("open", "high", "low"):
        df[c] = df[c].where(df[c] > 0).fillna(df["close"])
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    for c in ("volume", "turnover", "trades"):
        df[c] = df[c].fillna(0.0)

    df["prev_close"] = df["prev_close"].where(df["prev_close"] > 0)
    all_dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
    n = len(all_dates)
    frames = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.set_index("date")
        first = all_dates.get_loc(g.index.min())
        last = all_dates.get_loc(g.index.max())
        end = n - 1 if last >= n - 20 else last      # still-listed stocks run to the latest date
        g = g.reindex(all_dates[first:end + 1])
        g["symbol"] = sym
        g["traded"] = g["close"].notna().astype(int)
        g["close"] = g["close"].ffill()
        for c in ("open", "high", "low"):
            g[c] = g[c].fillna(g["close"])
        for c in ("volume", "turnover", "trades"):
            g[c] = g[c].fillna(0.0)
        g["prev_close"] = g["prev_close"].fillna(g["close"].shift(1))
        frames.append(g.rename_axis("date").reset_index())
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    grp = panel.groupby("symbol", sort=False)
    yesterday_close = grp["close"].shift(1)
    factor = (panel["prev_close"] / yesterday_close).where(panel["traded"] == 1)
    factor = factor.where(np.isfinite(factor), 1.0).fillna(1.0)
    is_action = factor < (1.0 - ADJUST_TOLERANCE)   # drops only; rises are treated as noise
    factor = factor.where(is_action, 1.0)
    panel["corp_action"] = is_action.astype(int)
    panel["action_factor"] = factor

    # price before an action day x factor = price on a comparable basis with after
    rev_cum = panel.groupby("symbol", sort=False)["action_factor"].transform(
        lambda s: s[::-1].cumprod()[::-1])
    panel["adj_factor"] = (rev_cum / panel["action_factor"]).astype(float)
    for c in ("open", "high", "low", "close"):
        panel[f"adj_{c}"] = panel[c] * panel["adj_factor"]
    return panel


def validate(raw: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Simple health checks. Returns a dictionary you can read."""
    issues, warnings = [], []
    dup = int(raw.duplicated(["symbol", "date"]).sum())
    if dup:
        warnings.append(f"{dup} duplicate (symbol, date) rows were dropped.")
    bad_price = int((raw[["open", "high", "low", "close"]].fillna(1) <= 0).any(axis=1).sum())
    if bad_price:
        warnings.append(f"{bad_price} rows had a zero or negative open/high/low (replaced by the close).")
    hl = int((raw["high"] < raw["low"]).sum())
    if hl:
        warnings.append(f"{hl} rows had high below low (repaired).")

    dates = pd.DatetimeIndex(np.sort(raw["date"].unique()))
    gaps = pd.Series(dates[1:] - dates[:-1], index=dates[1:])
    big_gaps = gaps[gaps > pd.Timedelta(days=10)]
    if len(big_gaps):
        warnings.append("Gaps of more than 10 days with no data (missed days or long holidays): "
                        + ", ".join(str(d.date()) for d in big_gaps.index[:5]))

    moves = (raw["close"] / raw["prev_close"].where(raw["prev_close"] > 0) - 1).abs()
    limit = price_limit_series(raw["date"])
    too_big = raw[(moves > limit + 0.02)]
    if len(too_big):
        warnings.append(f"{len(too_big)} rows moved more than the daily limit. "
                        f"Example: {too_big.iloc[0]['symbol']} on {too_big.iloc[0]['date'].date()}")

    actions = panel[panel["corp_action"] == 1]
    extreme = actions[(actions["action_factor"] < 0.3) | (actions["action_factor"] > 3)]
    if len(extreme):
        issues.append(f"{len(extreme)} corporate-action factors look extreme (below 0.3 or above 3). "
                      "Check these rows by hand.")
    last_day = panel["date"].max()
    n_latest = int((panel.loc[panel["date"] == last_day, "traded"] == 1).sum())
    return {
        "rows": int(len(raw)), "symbols": int(raw["symbol"].nunique()),
        "first_date": str(raw["date"].min().date()), "last_date": str(raw["date"].max().date()),
        "trading_days": int(len(dates)), "stocks_traded_on_last_day": n_latest,
        "corporate_actions_found": int(len(actions)),
        "issues": issues, "warnings": warnings, "ok": not issues,
    }


def run_clean(raw: pd.DataFrame, p: Paths | None = None) -> tuple[pd.DataFrame, dict]:
    p = p or Paths()
    p.ensure()
    raw = raw[raw["date"] >= pd.Timestamp(DATA_START)].copy()
    panel = build_clean_panel(raw)
    report = validate(raw, panel)
    panel.to_parquet(p.prices_clean, index=False)
    p.validation.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return panel, report
