"""Feature engineering: turn price history into numbers a model can learn from.

GOLDEN RULE: a feature on date D may only use information available at the close of
day D. Nothing from the future.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import HORIZON, MIN_AVG_TURNOVER, MIN_TRADED_DAYS_20, Paths, price_limit_series

FEATURE_COLS = [
    # momentum: how much the price moved recently
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    # trend position
    "dist_ma20", "dist_ma50", "dist_hi_252", "pos_20",
    # risk and daily shape
    "vol_20", "vol_60", "range_14", "gap", "close_pos",
    # liquidity and activity
    "log_turn_20", "turn_ratio", "amihud_20", "traded_days_20",
    # NEPSE price-limit behaviour
    "up_10", "dn_10",
    # this stock versus the market
    "ex_ret_5", "ex_ret_20",
    # where this stock ranks today among tradable stocks (0 to 1)
    "cs_ret_5", "cs_ret_20", "cs_vol_20", "cs_turn_ratio", "cs_dist_ma20",
    "cs_dist_hi", "cs_log_turn_20",
]
RANK_SOURCES = {"cs_ret_5": "ret_5", "cs_ret_20": "ret_20", "cs_vol_20": "vol_20",
                "cs_turn_ratio": "turn_ratio", "cs_dist_ma20": "dist_ma20",
                "cs_dist_hi": "dist_hi_252", "cs_log_turn_20": "log_turn_20"}


def _roll(g, col: str, window: int, fn: str, min_periods: int | None = None) -> pd.Series:
    r = getattr(g[col].rolling(window, min_periods=min_periods or window), fn)()
    return r.reset_index(level=0, drop=True)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    g = df.groupby("symbol", sort=False)
    adj = df["adj_close"]

    for k in (1, 3, 5, 10, 20, 60):
        df[f"ret_{k}"] = adj / g["adj_close"].shift(k) - 1

    df["ma20"] = _roll(g, "adj_close", 20, "mean")
    df["ma50"] = _roll(g, "adj_close", 50, "mean")
    df["dist_ma20"] = adj / df["ma20"] - 1
    df["dist_ma50"] = adj / df["ma50"] - 1
    df["hi_252"] = _roll(g, "adj_high", 252, "max", min_periods=60)
    df["dist_hi_252"] = adj / df["hi_252"] - 1
    df["lo_20"] = _roll(g, "adj_low", 20, "min")
    df["hi_20"] = _roll(g, "adj_high", 20, "max")
    df["pos_20"] = (adj - df["lo_20"]) / (df["hi_20"] - df["lo_20"] + 1e-9)

    df["vol_20"] = _roll(g, "ret_1", 20, "std")
    df["vol_60"] = _roll(g, "ret_1", 60, "std", min_periods=30)
    df["rng"] = (df["adj_high"] - df["adj_low"]) / adj
    df["range_14"] = _roll(g, "rng", 14, "mean")
    df["gap"] = df["adj_open"] / g["adj_close"].shift(1) - 1
    df["close_pos"] = (adj - df["adj_low"]) / (df["adj_high"] - df["adj_low"] + 1e-9)

    df["turn_5"] = _roll(g, "turnover", 5, "mean")
    df["turn_20"] = _roll(g, "turnover", 20, "mean")
    df["log_turn_20"] = np.log1p(df["turn_20"])
    df["turn_ratio"] = df["turn_5"] / (df["turn_20"] + 1.0)
    df["illiq"] = df["ret_1"].abs() / (df["turnover"] + 1.0) * 1e6
    df["amihud_20"] = _roll(g, "illiq", 20, "mean")
    df["traded_days_20"] = _roll(g, "traded", 20, "sum")

    limit = price_limit_series(df["date"])
    df["hit_up"] = ((df["ret_1"] >= limit * 0.97) & (df["traded"] == 1)).astype(float)
    df["hit_dn"] = ((df["ret_1"] <= -limit * 0.97) & (df["traded"] == 1)).astype(float)
    g = df.groupby("symbol", sort=False)
    df["up_10"] = _roll(g, "hit_up", 10, "sum", min_periods=1)
    df["dn_10"] = _roll(g, "hit_dn", 10, "sum", min_periods=1)

    # who counts as "tradable": enough turnover and trading days
    df["eligible"] = ((df["turn_20"] >= MIN_AVG_TURNOVER)
                      & (df["traded_days_20"] >= MIN_TRADED_DAYS_20))

    # market-wide numbers for each date (tradable stocks only)
    elig_ret = df["ret_1"].where(df["eligible"])
    mkt = elig_ret.groupby(df["date"]).mean()
    adv = (df["ret_1"] > 0).astype(float).where(df["eligible"]).groupby(df["date"]).mean()
    mkt_df = pd.DataFrame({"mkt_ret_1": mkt, "breadth": adv}).sort_index()
    mkt_df["mkt_ret_5"] = mkt_df["mkt_ret_1"].rolling(5, min_periods=5).sum()
    mkt_df["mkt_ret_20"] = mkt_df["mkt_ret_1"].rolling(20, min_periods=20).sum()
    for c in mkt_df.columns:
        df[c] = df["date"].map(mkt_df[c])

    for k in (5, 20):
        m = df[f"ret_{k}"].where(df["eligible"]).groupby(df["date"]).transform("mean")
        df[f"ex_ret_{k}"] = df[f"ret_{k}"] - m
    for new, src in RANK_SOURCES.items():
        df[new] = df[src].where(df["eligible"]).groupby(df["date"]).rank(pct=True)

    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float).replace([np.inf, -np.inf], np.nan)
    return df


def add_labels(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """What we want to predict (used for training and testing only, never as a feature).

    We assume you could buy at the NEXT day's close and sell `horizon` trading days later.
      fwd_ret      = return over that holding period
      fwd_excess   = fwd_ret minus the average tradable stock's fwd_ret that day
      label_rank   = where fwd_excess ranks among tradable stocks that day (0 to 1)
    """
    out = df.copy()
    g = out.groupby("symbol", sort=False)["adj_close"]
    out["fwd_ret"] = g.shift(-(1 + horizon)) / g.shift(-1) - 1
    elig_fwd = out["fwd_ret"].where(out["eligible"])
    out["fwd_excess"] = out["fwd_ret"] - elig_fwd.groupby(out["date"]).transform("mean")
    out["label_rank"] = elig_fwd.groupby(out["date"]).rank(pct=True)
    return out


NON_SHARE_WORDS = r"debenture|bond|fund|scheme|mutual|promoter|preferred"


def non_share_symbols(symbols, p: Paths) -> set[str]:
    """Symbols that are not ordinary shares: debentures, bonds, mutual funds, promoter shares.

    Uses the security names in the downloaded NEPSE files, plus a simple rule:
    any symbol containing a digit (debentures and many funds) is excluded.
    """
    import json
    names: dict[str, str] = {}
    for f in sorted(p.raw_dir.glob("*.json")):
        try:
            rows = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in rows:
            if isinstance(r, dict) and r.get("symbol") and r.get("securityName"):
                names[str(r["symbol"]).upper()] = str(r["securityName"]).lower()
    out = set()
    for s in symbols:
        if any(ch.isdigit() for ch in s) or re.search(NON_SHARE_WORDS, names.get(s, "")):
            out.add(s)
    return out


def run_features(panel: pd.DataFrame, p: Paths | None = None) -> pd.DataFrame:
    p = p or Paths()
    p.ensure()
    excluded = non_share_symbols(panel["symbol"].unique(), p)
    print(f"Excluding {len(excluded)} non-share instruments (debentures, funds, promoter shares)")
    panel = panel[~panel["symbol"].isin(excluded)]
    df = add_labels(build_features(panel))
    df.to_parquet(p.features, index=False)
    return df
