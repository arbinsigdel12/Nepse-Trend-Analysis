"""Pull cash-dividend and bonus declarations out of the announcement text.

The NEPSE corporate-action feed has bonus and right shares only. Newer dividend announcements
state the amounts in a fixed pattern, for example:
    Dividend Declaration of NTC Fiscal Year: 2080-2081 Cash Dividend: 30%
Older ones often just point to an attached PDF, so coverage is partial. This module reports
exactly how partial, so nothing is built on numbers that are not there.
"""
from __future__ import annotations

import re

import pandas as pd

from .company import _read, company_dir
from .config import Paths

FY = re.compile(r"fiscal\s*year\s*[:\-]?\s*(\d{4})\s*[-/]\s*(\d{2,4})", re.I)
CASH = re.compile(r"cash\s*dividend[^0-9]{0,25}([\d]+(?:\.\d+)?)\s*%", re.I)
BONUS = re.compile(r"bonus\s*shares?[^0-9]{0,25}([\d]+(?:\.\d+)?)\s*%", re.I)


def dividends_path(p: Paths):
    return company_dir(p) / "dividends.parquet"


def _first(rx, text):
    m = rx.search(text)
    return float(m.group(1)) if m else None


def parse_declarations(news: pd.DataFrame) -> pd.DataFrame:
    n = news[news["source"] == "company_news"].copy()
    text = n["title"].fillna("") + " " + n["body"].fillna("")
    n = n[text.str.contains(r"dividend|bonus", case=False, regex=True)].copy()
    text = n["title"].fillna("") + " " + n["body"].fillna("")
    fy = text.str.extract(FY)
    n["fy_start"] = pd.to_numeric(fy[0], errors="coerce")
    n["cash_pct"] = text.map(lambda t: _first(CASH, t))
    n["bonus_pct"] = text.map(lambda t: _first(BONUS, t))
    return n


def run_dividends_build(p: Paths | None = None) -> pd.DataFrame:
    p = p or Paths()
    news = pd.read_parquet(p.news_dir.parent / "news_table.parquet")
    n = parse_declarations(news)
    n = n[n["symbol"].notna()]
    print(f"Announcements mentioning dividend or bonus: {len(n):,}")
    print(f"  with a fiscal year stated: {int(n['fy_start'].notna().sum()):,}")
    print(f"  with a cash dividend %:    {int(n['cash_pct'].notna().sum()):,}")
    print(f"  with a bonus %:            {int(n['bonus_pct'].notna().sum()):,}")
    known = n[n["fy_start"].notna() & (n["cash_pct"].notna() | n["bonus_pct"].notna())]
    per_year = (known.groupby(["symbol", "fy_start"])
                .agg(cash_pct=("cash_pct", "max"), bonus_pct_text=("bonus_pct", "max"),
                     declared=("published", "min")).reset_index())

    actions = _read(company_dir(p) / "corporate_actions.json", {})
    feed = []
    for sym, rows in actions.items():
        for r in rows:
            if r.get("bonusPercentage") not in (None, ""):
                try:
                    feed.append((sym, int(str(r["fiscalYear"])[:4]), float(r["bonusPercentage"])))
                except ValueError:
                    pass
    feed = pd.DataFrame(feed, columns=["symbol", "fy_start", "bonus_pct_feed"]).groupby(
        ["symbol", "fy_start"], as_index=False).max()
    out = per_year.merge(feed, on=["symbol", "fy_start"], how="outer")
    out["bonus_pct"] = out["bonus_pct_feed"].fillna(out["bonus_pct_text"])
    out["fy_start"] = out["fy_start"].astype(int)
    out = out.sort_values(["symbol", "fy_start"]).reset_index(drop=True)
    dividends_path(p).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dividends_path(p), index=False)

    print("\nStock-years with information, by fiscal year (cash = cash dividend known from text):")
    cov = out.groupby("fy_start").agg(stocks=("symbol", "nunique"), cash_known=("cash_pct", lambda s: int(s.notna().sum())),
                                      bonus_known=("bonus_pct", lambda s: int(s.notna().sum())))
    print(cov.to_string())
    for sym in ["NABIL", "NTC", "NBL", "EBL"]:
        s = out[out["symbol"] == sym]
        if len(s):
            print(f"\n{sym}:")
            print(s[["fy_start", "cash_pct", "bonus_pct"]].to_string(index=False))
    miss = n[n["fy_start"].notna() & n["cash_pct"].isna() & n["bonus_pct"].isna()]
    print(f"\nDeclarations with a fiscal year but no amounts in the text: {len(miss)} (amounts are probably in attached PDFs).")
    print("A few of them:")
    for _, r in miss.sample(min(5, len(miss)), random_state=1).iterrows():
        print(f"  {r['symbol']} {r['published'].date()}: {r['body'][:160]}")
    nofy = n[n["fy_start"].isna()]
    print(f"\nDeclarations with no fiscal year in the text: {len(nofy)}. A few of them:")
    for _, r in nofy.sample(min(5, len(nofy)), random_state=1).iterrows():
        print(f"  {r['symbol']} {r['published'].date()}: {str(r['title'])[:70]} | {r['body'][:110]}")
    print(f"\nSaved to {dividends_path(p)}")
    return out
