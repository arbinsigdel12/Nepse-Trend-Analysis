"""A permanent record of every daily ranking, and a check of how those rankings actually did.

Backtests can flatter a model. This keeps the rankings exactly as they were published each
day, before the outcome was known, so that after a few weeks the model can be judged on
its real track record. The first ranking saved for a date is never overwritten.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .config import HORIZON, TOP_K, Paths
from .evaluate import ic_by_date, topk_by_date


def history_path(p: Paths):
    return p.base / "predictions" / "history.parquet"


def append_history(latest: pd.DataFrame, p: Paths | None = None) -> int:
    """Save the ranking for the latest date unless that date is already in the log."""
    p = p or Paths()
    meta = json.loads(p.model_meta.read_text(encoding="utf-8")) if p.model_meta.exists() else {}
    new = latest[["date", "symbol", "rank", "score", "score_pct", "close"]].copy()
    new["model_trained_through"] = meta.get("trained_through")
    new["logged_at"] = pd.Timestamp.now().floor("s")
    path = history_path(p)
    if path.exists():
        old = pd.read_parquet(path)
        new = new[~new["date"].isin(old["date"].unique())]
        if new.empty:
            return 0
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return int(new["date"].nunique())


def run_live_check(p: Paths | None = None) -> None:
    p = p or Paths()
    if not history_path(p).exists():
        print("No saved rankings yet. They are saved each time `predict` runs.")
        return
    hist = pd.read_parquet(history_path(p))
    feats = pd.read_parquet(p.features, columns=["date", "symbol", "fwd_excess"])
    d = hist.merge(feats, on=["date", "symbol"], how="left")
    ready = d.dropna(subset=["fwd_excess"]).copy()
    print(f"Rankings saved: {hist['date'].nunique()} day(s), "
          f"{pd.Timestamp(hist['date'].min()).date()} to {pd.Timestamp(hist['date'].max()).date()}")
    print(f"Rankings whose {HORIZON}-day outcome is now known: {ready['date'].nunique()}")
    if ready["date"].nunique() < 5:
        print("\nToo few results to judge anything yet. Outcomes arrive about "
              f"{HORIZON + 1} trading days after each ranking. Check again in a few weeks.")
        return
    ready["score_rank"] = -ready["rank"]
    ic = ic_by_date(ready, "score_rank")
    tk = topk_by_date(ready, "score_rank", TOP_K)
    ic_i, tk_i = ic.iloc[::HORIZON], tk.iloc[::HORIZON]
    n = len(ic_i)
    t = float(ic_i.mean() / (ic_i.std(ddof=1) / math.sqrt(n))) if n > 2 and ic_i.std(ddof=1) > 0 else float("nan")
    print(f"\nLive results over {ready['date'].nunique()} ranked days ({n} independent {HORIZON}-day periods):")
    print(f"  average rank correlation (IC):        {ic.mean():+.3f}   (backtest 2020-2026: about +0.095)")
    print(f"  t-statistic of the IC:                {t:.2f}   (above 2 is visible; very few periods give wide noise)")
    print(f"  top-{TOP_K} picks beat the market by:     {tk_i['topk_excess'].mean() * 100:+.2f}% per period before costs   (backtest: about +0.45%)")
    print(f"  top-{TOP_K} picks beat the market in:     {tk_i['topk_hit'].mean() * 100:.0f}% of picks   (chance level: {tk_i['base_hit'].mean() * 100:.0f}%)")
    print("\nA few weeks of results are mostly noise. Judge the model on months, not days.")
