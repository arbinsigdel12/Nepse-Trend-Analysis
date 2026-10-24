"""Training with walk-forward testing, then fitting the final model and scoring today."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import HORIZON, Paths
from .features import FEATURE_COLS

BASELINE_COLS = ["fwd_ret", "fwd_excess", "label_rank", "ret_5", "ret_20",
                 "turn_ratio", "vol_20", "dist_ma20"]


def _new_model(objective: str):
    import lightgbm as lgb
    common = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                  random_state=42, verbose=-1)
    if objective == "lambdarank":
        return lgb.LGBMRanker(objective="lambdarank", **common)
    return lgb.LGBMRegressor(objective="regression", **common)


def _fit(train: pd.DataFrame, features: list[str], objective: str):
    train = train.sort_values(["date", "symbol"])
    model = _new_model(objective)
    if objective == "lambdarank":
        y = np.clip(np.floor(train["label_rank"] * 5), 0, 4).astype(int)
        groups = train.groupby("date", sort=True).size().to_numpy()
        model.fit(train[features], y, group=groups)
    else:
        model.fit(train[features], train["label_rank"])
    return model


def walk_forward(df: pd.DataFrame, features: list[str] = FEATURE_COLS,
                 objective: str = "regression", min_train_days: int = 250,
                 test_days: int = 20, embargo: int = HORIZON + 1,
                 max_folds: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Honest testing: train on the past, predict the next block, slide forward, repeat.

    `embargo` leaves a gap between training and test so labels that overlap the test
    period cannot leak in.
    """
    d = df[df["eligible"] & df["label_rank"].notna()]
    dates = np.sort(d["date"].unique())
    starts = list(range(min_train_days + embargo, len(dates), test_days))
    if max_folds:
        starts = starts[-max_folds:]
    if not starts:
        raise ValueError(f"Not enough history: need more than {min_train_days + embargo} "
                         f"labelled trading days, have {len(dates)}.")
    parts = []
    for n, i in enumerate(starts, 1):
        train_end = dates[i - embargo]
        test_dates = dates[i:i + test_days]
        tr = d[d["date"] <= train_end]
        te = d[(d["date"] >= test_dates[0]) & (d["date"] <= test_dates[-1])].copy()
        model = _fit(tr, features, objective)
        te["pred"] = model.predict(te[features])
        te["fold"] = n
        parts.append(te[["date", "symbol", "pred", "fold"] + BASELINE_COLS])
        if verbose:
            print(f"  fold {n}/{len(starts)}: trained on {len(tr):,} rows through "
                  f"{pd.Timestamp(train_end).date()}, tested {len(te):,} rows")
    return pd.concat(parts, ignore_index=True)


def train_final(df: pd.DataFrame, features: list[str] = FEATURE_COLS,
                objective: str = "regression", p: Paths | None = None) -> dict:
    p = p or Paths()
    p.ensure()
    d = df[df["eligible"] & df["label_rank"].notna()]
    model = _fit(d, features, objective)
    booster = model.booster_
    booster.save_model(str(p.model))
    imp = dict(zip(features, booster.feature_importance(importance_type="gain").tolist()))
    meta = {"features": features, "objective": objective, "rows": int(len(d)),
            "trained_through": str(pd.Timestamp(d["date"].max()).date()),
            "importance_gain": imp}
    p.model_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def score_latest(df: pd.DataFrame, p: Paths | None = None) -> pd.DataFrame:
    """Score every tradable stock on the most recent date and rank them."""
    import lightgbm as lgb
    p = p or Paths()
    p.ensure()
    meta = json.loads(p.model_meta.read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(p.model))
    last = df["date"].max()
    today = df[(df["date"] == last) & df["eligible"]].copy()
    if today.empty:
        raise ValueError("No tradable stocks on the latest date. Check your data and the "
                         "MIN_AVG_TURNOVER setting in config.py.")
    today["score"] = booster.predict(today[meta["features"]])
    today["rank"] = today["score"].rank(ascending=False, method="first").astype(int)
    today["score_pct"] = today["score"].rank(pct=True) * 100
    cols = ["date", "symbol", "rank", "score", "score_pct", "close", "traded",
            "ret_1", "ret_5", "ret_20", "turn_20", "turn_ratio", "vol_20"]
    out = today.sort_values("rank")[cols].reset_index(drop=True)
    out.to_parquet(p.latest, index=False)
    return out
