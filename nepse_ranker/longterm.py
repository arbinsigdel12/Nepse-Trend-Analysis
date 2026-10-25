"""Long-term screen: risk, history, dilution and red flags for every ordinary share.

It does NOT value companies: earnings and book value are not in the data yet, and the NEPSE
feed has no cash dividends. A stock passing the checks is not a recommendation to buy it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .company import _read, company_dir
from .config import Paths

TRADING_DAYS = 240
RED_FLAG_EVENTS = ["lock_in_end", "share_sale_intent", "project_disruption", "regulatory_action"]
CHECK_NAMES = {"c_history": "short history", "c_liquid": "thin trading", "c_drawdown": "deep past fall",
               "c_vol": "high volatility", "c_dilution": "repeated right issues", "c_flags": "red-flag news"}


def screen_path(p: Paths):
    return p.base / "longterm" / "screen.parquet"


def _fy_start(s):
    try:
        return int(str(s)[:4])
    except ValueError:
        return None


def build_longterm(p: Paths | None = None, min_turnover: float = 1_000_000.0) -> pd.DataFrame:
    p = p or Paths()
    d = company_dir(p)
    cls = {str(r["symbol"]).upper(): r for r in _read(d / "classification.json", [])}
    actions = _read(d / "corporate_actions.json", {})
    sectors = _read(d / "sectors.json", {})
    if not sectors:
        raise FileNotFoundError("No company data. Run: python -m nepse_ranker company-fetch")
    px = pd.read_parquet(p.prices_clean, columns=["date", "symbol", "close", "adj_close", "turnover", "traded"])
    px = px[px["symbol"].isin(sectors)]
    last = px["date"].max()
    fy_all = [_fy_start(r.get("fiscalYear")) for v in actions.values() for r in v]
    cur_fy = max([x for x in fy_all if x] or [0])

    flag_counts = pd.Series(dtype=int)
    nf = p.news_dir.parent / "news_table.parquet"
    if nf.exists():
        cols = ["evt_" + e for e in RED_FLAG_EVENTS]
        news = pd.read_parquet(nf, columns=["symbol", "published"] + cols)
        news = news[news["published"] >= last - pd.Timedelta(days=365)]
        flag_counts = news[news[cols].any(axis=1)].groupby("symbol").size()

    dp = company_dir(p) / "dividends.parquet"
    div_by, fy_cur = {}, 0
    if dp.exists():
        dv = pd.read_parquet(dp)
        dv["paid"] = (dv["cash_pct"].fillna(0) > 0) | (dv["bonus_pct"].fillna(0) > 0)
        fy_cur = int(dv["fy_start"].max())
        div_by = {s: g for s, g in dv.groupby("symbol")}

    def div_metrics(sym, last_close):
        out = {"dividend_years_5fy": np.nan, "cash_avg_3fy_pct": np.nan, "cash_known_3fy": 0, "cash_yield": np.nan}
        if not div_by:
            return out
        ds = div_by.get(sym)
        if ds is None:
            out["dividend_years_5fy"] = 0
            return out
        done = ds[(ds["fy_start"] >= fy_cur - 5) & (ds["fy_start"] <= fy_cur - 1)]
        out["dividend_years_5fy"] = int(done.loc[done["paid"], "fy_start"].nunique())
        c3 = ds[(ds["fy_start"] >= fy_cur - 3) & (ds["fy_start"] <= fy_cur - 1)]["cash_pct"].dropna()
        out["cash_known_3fy"] = len(c3)
        if len(c3):
            out["cash_avg_3fy_pct"] = float(c3.mean())
            face = float(cls.get(sym, {}).get("faceValue") or 100)
            out["cash_yield"] = out["cash_avg_3fy_pct"] / 100 * face / last_close
        return out

    dp = company_dir(p) / "dividends.parquet"
    div_by, fy_cur = {}, 0
    if dp.exists():
        dv = pd.read_parquet(dp)
        dv["paid"] = (dv["cash_pct"].fillna(0) > 0) | (dv["bonus_pct"].fillna(0) > 0)
        fy_cur = int(dv["fy_start"].max())
        div_by = {s: g for s, g in dv.groupby("symbol")}

    def div_metrics(sym, last_close):
        out = {"dividend_years_5fy": np.nan, "cash_avg_3fy_pct": np.nan, "cash_known_3fy": 0, "cash_yield": np.nan}
        if not div_by:
            return out
        ds = div_by.get(sym)
        if ds is None:
            out["dividend_years_5fy"] = 0
            return out
        done = ds[(ds["fy_start"] >= fy_cur - 5) & (ds["fy_start"] <= fy_cur - 1)]
        out["dividend_years_5fy"] = int(done.loc[done["paid"], "fy_start"].nunique())
        c3 = ds[(ds["fy_start"] >= fy_cur - 3) & (ds["fy_start"] <= fy_cur - 1)]["cash_pct"].dropna()
        out["cash_known_3fy"] = len(c3)
        if len(c3):
            out["cash_avg_3fy_pct"] = float(c3.mean())
            face = float(cls.get(sym, {}).get("faceValue") or 100)
            out["cash_yield"] = out["cash_avg_3fy_pct"] / 100 * face / last_close
        return out

    rows = []
    for sym, g in px.groupby("symbol"):
        g = g.sort_values("date")
        c = g["adj_close"].to_numpy(dtype=float)
        n = len(c)
        if n < 60:
            continue

        def cagr(years):
            m = int(years * TRADING_DAYS)
            return (c[-1] / c[-1 - m]) ** (1 / years) - 1 if n > m else np.nan

        w5 = c[-min(n, 5 * TRADING_DAYS):]
        w3 = c[-min(n, 3 * TRADING_DAYS):]
        daily = np.diff(w3) / w3[:-1]
        recent = g.tail(120)
        listing = pd.to_datetime(cls.get(sym, {}).get("listingDate"), errors="coerce")
        acts = actions.get(sym, [])
        recent_acts = [r for r in acts if (_fy_start(r.get("fiscalYear")) or 0) >= cur_fy - 4]
        bonus = [float(r["bonusPercentage"]) for r in recent_acts if r.get("bonusPercentage") not in (None, "")]
        rights = [r for r in recent_acts if r.get("rightPercentage") not in (None, "")]
        rows.append({
            "symbol": sym, "sector": sectors.get(sym), "last_close": float(g["close"].iloc[-1]),
            "listed_years": (last - listing).days / 365.25 if pd.notna(listing) else np.nan,
            "price_years": n / TRADING_DAYS,
            "cagr_3y": cagr(3), "cagr_5y": cagr(5),
            "vol_ann": float(np.std(daily, ddof=1) * np.sqrt(TRADING_DAYS)),
            "max_dd_5y": float((w5 / np.maximum.accumulate(w5) - 1).min()),
            "below_3y_high": float(w3[-1] / w3.max() - 1),
            "median_turnover_120d": float(recent["turnover"].median()),
            "traded_share_1y": float(g["traded"].tail(TRADING_DAYS).mean()),
            "bonus_5fy_total_pct": float(sum(bonus)), "bonus_years_5fy": len(bonus),
            "rights_issues_5fy": len(rights), "red_flags_1y": int(flag_counts.get(sym, 0)),
            **div_metrics(sym, float(g["close"].iloc[-1])),
        })
    df = pd.DataFrame(rows)
    df["c_history"] = (df["listed_years"] >= 5) & (df["price_years"] >= 3)
    df["c_liquid"] = df["median_turnover_120d"] >= min_turnover
    df["c_drawdown"] = df["max_dd_5y"] > -0.60
    df["c_vol"] = df["vol_ann"] < 0.50
    df["c_dilution"] = df["rights_issues_5fy"] <= 1
    df["c_flags"] = df["red_flags_1y"] <= 1
    checks = list(CHECK_NAMES)
    df["checks_passed"] = df[checks].sum(axis=1)
    df["fails"] = df[checks].apply(lambda r: ", ".join(CHECK_NAMES[c] for c in checks if not r[c]), axis=1)
    df = df.sort_values(["checks_passed", "cagr_5y"], ascending=[False, False]).reset_index(drop=True)
    out = screen_path(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def run_longterm_build(p: Paths | None = None) -> pd.DataFrame:
    p = p or Paths()
    df = build_longterm(p)
    print(f"Screened {len(df)} ordinary shares.")
    print("\nChecks passed (out of 6):")
    print(df["checks_passed"].value_counts().sort_index(ascending=False).to_string())
    print("\nWhat fails most often:")
    for k, name in CHECK_NAMES.items():
        print(f"  {name:24s} {int((~df[k]).sum()):4d} stocks")
    best = df[df["checks_passed"] == 6]
    print(f"\nPassing all 6 checks: {len(best)}. By sector:")
    print(best["sector"].value_counts().to_string())
    show = df.head(12)[["symbol", "sector", "checks_passed", "cagr_5y", "vol_ann", "max_dd_5y"]].copy()
    for c in ("cagr_5y", "vol_ann", "max_dd_5y"):
        show[c] = (show[c] * 100).round(1)
    print("\nTop of the list (cagr_5y, vol_ann, max_dd_5y in percent):")
    print(show.to_string(index=False))
    print(f"\nSaved to {screen_path(p)}")
    return df
