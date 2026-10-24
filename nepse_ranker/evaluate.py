"""Judging the model honestly: does it beat simple rules, after costs?"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .config import HORIZON, ROUND_TRIP_COST, TOP_K, Paths


def ic_by_date(df: pd.DataFrame, score: str, target: str = "fwd_excess") -> pd.Series:
    """Rank correlation (Spearman) between score and what really happened, per day."""
    out = {}
    for d, g in df.groupby("date"):
        if g[score].nunique() < 3 or g[target].nunique() < 3:
            continue
        out[d] = g[score].rank().corr(g[target].rank())
    return pd.Series(out, dtype=float).sort_index()


def topk_by_date(df: pd.DataFrame, score: str, k: int = TOP_K) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("date"):
        if len(g) < k * 2:
            continue
        top = g.nlargest(k, score)
        rows.append({"date": d, "topk_excess": top["fwd_excess"].mean(),
                     "topk_hit": (top["fwd_excess"] > 0).mean(),
                     "base_hit": (g["fwd_excess"] > 0).mean(), "n": len(g)})
    return pd.DataFrame(rows).set_index("date").sort_index()


def summarize(df: pd.DataFrame, score: str, k: int = TOP_K, horizon: int = HORIZON,
              cost: float = ROUND_TRIP_COST) -> dict:
    ic = ic_by_date(df, score)
    tk = topk_by_date(df, score, k)
    ic_ind = ic.iloc[::horizon]              # every 5th day: independent, non-overlapping periods
    tk_ind = tk.iloc[::horizon]
    n = len(ic_ind)
    t_stat = float(ic_ind.mean() / (ic_ind.std(ddof=1) / math.sqrt(n))) if n > 2 and ic_ind.std(ddof=1) > 0 else float("nan")
    net = tk_ind["topk_excess"] - cost
    equity = (1 + net).cumprod()
    drawdown = float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan")
    return {
        "mean_ic": float(ic.mean()), "ic_t_stat": t_stat,
        "ic_positive_share": float((ic > 0).mean()),
        "topk_gross_excess": float(tk_ind["topk_excess"].mean()),
        "topk_net_excess": float(net.mean()),
        "topk_hit_rate": float(tk_ind["topk_hit"].mean()),
        "base_hit_rate": float(tk_ind["base_hit"].mean()),
        "periods": int(n), "max_drawdown_net": drawdown,
        "final_equity_net": float(equity.iloc[-1]) if len(equity) else float("nan"),
    }


def verdict(summary: pd.DataFrame) -> dict:
    m = summary.loc["model"]
    baselines = summary.drop(index=["model", "random"], errors="ignore")
    best_base = float(baselines["topk_net_excess"].max()) if len(baselines) else float("-inf")
    strong = (m["mean_ic"] > 0.02) and (m["ic_t_stat"] > 2) and (m["topk_net_excess"] > 0)
    if strong and m["topk_net_excess"] > best_base:
        level, text = "promising", ("The model passes the basic screen: positive rank correlation, "
            "statistically visible, positive after costs, and better than the simple rules. "
            "This is NOT proof. Confirm with live paper trading before risking money.")
    elif strong:
        level, text = "mixed", ("The model passes the basic screen but a simple rule did as well or better. "
            "Prefer the simple rule until the model clearly beats it.")
    elif m["mean_ic"] > 0 and m["ic_t_stat"] > 1:
        level, text = "weak", ("There is a faint signal, but not strong enough to trust after costs. "
            "Improve features or collect more data before relying on it.")
    else:
        level, text = "none", ("No reliable edge detected. The ranking should be treated as a screening "
            "list only, not a prediction. This is a normal and useful result.")
    return {"level": level, "text": text, "best_baseline_net_excess": best_base}


def evaluate_all(oos: pd.DataFrame, k: int = TOP_K, p: Paths | None = None) -> dict:
    p = p or Paths()
    p.ensure()
    d = oos.copy()
    rng = np.random.default_rng(0)
    d["random"] = rng.random(len(d))
    d["neg_vol_20"] = -d["vol_20"]
    scores = {"model": "pred", "momentum_5d": "ret_5", "momentum_20d": "ret_20",
              "volume_surge": "turn_ratio", "low_volatility": "neg_vol_20", "random": "random"}
    summary = pd.DataFrame({name: summarize(d, col, k) for name, col in scores.items()}).T
    daily = pd.DataFrame({"ic": ic_by_date(d, "pred")}).join(topk_by_date(d, "pred", k), how="outer")
    daily.index.name = "date"
    daily.reset_index().to_parquet(p.daily_eval, index=False)
    v = verdict(summary)
    result = {"summary": summary.round(5).to_dict(orient="index"), "verdict": v,
              "k": k, "horizon": HORIZON, "round_trip_cost": ROUND_TRIP_COST,
              "test_start": str(pd.Timestamp(d["date"].min()).date()),
              "test_end": str(pd.Timestamp(d["date"].max()).date())}
    p.metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"summary": summary, "verdict": v, "daily": daily}
