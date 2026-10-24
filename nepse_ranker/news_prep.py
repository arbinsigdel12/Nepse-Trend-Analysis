"""Turn raw NEPSE news into one clean table, link each item to a stock, tag the kind of
event, and check whether each kind of event is followed by unusual price moves."""
from __future__ import annotations

import html
import math
import re

import numpy as np
import pandas as pd

from .config import Paths
from .news import load_saved

CLOSE_MINUTES = 15 * 60     # the market closes at about 3 PM Nepal time

# Rules are applied to the headline (lower-case). One headline can match several rules.
# These are a first draft: `news-build` prints what they miss so we can improve them.
EVENT_RULES = {
    "dividend": r"dividend",
    "bonus_share": r"bonus",
    "right_share": r"\brights?\b",
    "book_closure": r"book ?clos",
    "agm_sgm": r"\bagm\b|\bsgm\b|general meeting",
    "board_meeting": r"board meeting|board decision|bod meeting",
    "financial_result": r"financial|quarter|unaudited|audited|annual report|\bq[1-4]\b|profit",
    "merger_acquisition": r"merger|acquisition|amalgamat|\bmou\b",
    "public_issue": r"\bipo\b|\bfpo\b|public (?:issue|offering)|allot|issue manager|subscription",
    "management_change": r"director|\bceo\b|chief executive|appoint|resign|chairman|chairperson|\bbod\b|board member|company secretary|auditor",
    "regulatory_action": r"suspend|halt|revoke|penalt|\bfine\b|\bban\b|delist|warning|restrict",
    "debenture_bond": r"debenture|bond",
    "auction_promoter": r"auction|promoter",
    "price_adjusted": r"price adjust",
    "share_sale_intent": r"intention (?:for|to) sale|sale of (?:share|promoter)|share sale",
    "lock_in_end": r"lock ?-?in",
    "meeting_minutes": r"\bminutes?\b",
    "project_disruption": r"shut ?down|heavy rain|flood|landslide|damage|outage|\bfire\b",
}
SYMBOL_PATTERN = re.compile(r"[\[\(]([A-Z][A-Z0-9]{1,11})[\]\)]")


def strip_html(s) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", str(s or "")))
    return re.sub(r"\s+", " ", s).strip()


def _symbols_in(title: str, universe: set) -> list[str]:
    return [s for s in SYMBOL_PATTERN.findall(title or "") if s in universe]


def build_news_table(p: Paths | None = None, universe: set | None = None) -> pd.DataFrame:
    p = p or Paths()
    if universe is None:
        universe = set(pd.read_parquet(p.prices_clean, columns=["symbol"])["symbol"].unique())
    rows = []
    for r in load_saved(p, "company_news"):
        title = str(r.get("newsHeadline") or "").strip()
        sym = str(r.get("symbol") or "").strip().upper()
        if sym not in universe:
            found = _symbols_in(title, universe)
            sym = found[0] if found else None
        rows.append(("company_news", r.get("id"), r.get("publishedDate"), sym, title,
                     strip_html(r.get("newsBody"))[:2000]))
    for r in load_saved(p, "news_alerts"):
        title = str(r.get("messageTitle") or "").strip()
        syms = _symbols_in(title, universe) or [None]
        for sym in syms:
            rows.append(("alert", r.get("id"), r.get("addedDate"), sym, title,
                         strip_html(r.get("messageBody"))[:2000]))
    df = pd.DataFrame(rows, columns=["source", "id", "published", "symbol", "title", "body"])
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"]).sort_values("published").reset_index(drop=True)
    df["date"] = df["published"].dt.normalize()
    low = df["title"].str.lower()
    for name, pat in EVENT_RULES.items():
        df["evt_" + name] = low.str.contains(pat, regex=True)
    df["evt_none"] = ~df[["evt_" + n for n in EVENT_RULES]].any(axis=1)
    return df


def _templates(titles: pd.Series, symbols: pd.Series | None = None) -> pd.Series:
    if symbols is not None:      # remove each headline's own stock symbol, so similar headlines group together
        titles = pd.Series([re.sub(rf"\b{re.escape(str(s))}\b", " ", ti) if s and s == s else ti
                            for ti, s in zip(titles, symbols)], index=titles.index)
    t = titles.str.replace(r"\[[^\]]*\]|\([^)]*\)", " ", regex=True).str.lower()
    t = t.str.replace(r"[^a-z ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return t.str[:70].value_counts()


def run_news_build(p: Paths | None = None) -> pd.DataFrame:
    p = p or Paths()
    df = build_news_table(p)
    out = p.news_dir.parent / "news_table.parquet"
    df.to_parquet(out, index=False)
    print(f"News items: {len(df):,}  ({df['published'].min().date()} to {df['published'].max().date()})")
    print(df.groupby("source").agg(items=("title", "size"),
                                   linked_to_a_stock=("symbol", lambda s: int(s.notna().sum()))).to_string())
    flags = [c for c in df.columns if c.startswith("evt_")]
    print("\nHeadlines per event type (one headline can have several):")
    print(df[flags].sum().astype(int).sort_values(ascending=False).to_string())
    cn = df[df["source"] == "company_news"]
    print("\nMost common headline wording, company news (stock names removed):")
    print(_templates(cn["title"], cn["symbol"]).head(25).to_string())
    print("\nMost common wording the rules did NOT tag:")
    print(_templates(df.loc[df["evt_none"], "title"], df.loc[df["evt_none"], "symbol"]).head(20).to_string())
    print(f"\nSaved to {out}")
    return df


def reaction_positions(published: pd.Series, trading_dates: pd.DatetimeIndex) -> np.ndarray:
    """For each news time, the position of the first trading day whose CLOSE comes after it."""
    day = published.dt.normalize()
    minutes = published.dt.hour * 60 + published.dt.minute
    pos = trading_dates.searchsorted(day.to_numpy(), side="left")
    is_trading_day = np.zeros(len(pos), dtype=bool)
    ok = pos < len(trading_dates)
    is_trading_day[ok] = trading_dates[pos[ok]] == day.to_numpy()[ok]
    after_close = (minutes >= CLOSE_MINUTES).to_numpy()
    return np.where(is_trading_day & after_close, pos + 1, pos)


def run_event_study(p: Paths | None = None, seed: int = 0) -> pd.DataFrame:
    """Average unusual return around each kind of announcement.

    Day 0 = the first trading day whose close comes after the announcement (the market's
    reaction). 'drift' = what happened AFTER that reaction (days +1 to +5), which is the
    part someone reading the news could still have traded on.
    """
    p = p or Paths()
    news = pd.read_parquet(p.news_dir.parent / "news_table.parquet")
    f = pd.read_parquet(p.features, columns=["date", "symbol", "ret_1", "mkt_ret_1", "eligible"])
    f["xret"] = (f["ret_1"] - f["mkt_ret_1"]).where(f["eligible"])
    wide = f.pivot(index="date", columns="symbol", values="xret").sort_index()
    dates, values = wide.index, wide.to_numpy()
    col = {s: i for i, s in enumerate(wide.columns)}

    news = news[news["symbol"].isin(col)].copy()
    news["E"] = reaction_positions(news["published"], dates)
    news = news[news["E"] < len(dates)]
    ks = list(range(-5, 11))
    k0 = ks.index(0)

    def matrix(ev):
        si = ev["symbol"].map(col).to_numpy()
        ei = ev["E"].to_numpy()
        M = np.full((len(ev), len(ks)), np.nan)
        for j, k in enumerate(ks):
            pos = ei + k
            ok = (pos >= 0) & (pos < len(dates))
            M[ok, j] = values[pos[ok], si[ok]]
        return M

    def summarize(M):
        pre = M[:, :k0]
        post = M[:, k0 + 1:k0 + 6]
        res = {"events": len(M), "run_up_-5to-1_%": np.nanmean(pre.sum(axis=1)) * 100,
               "reaction_day0_%": np.nanmean(M[:, k0]) * 100, "day+1_%": np.nanmean(M[:, k0 + 1]) * 100}
        full = np.isfinite(post).all(axis=1)
        d = post[full].sum(axis=1)
        res["drift_+1to+5_%"] = d.mean() * 100 if len(d) else np.nan
        res["drift_t"] = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 else np.nan
        return res

    rows = {}
    for name in EVENT_RULES:
        ev = news[news["evt_" + name]].drop_duplicates(["symbol", "E"])
        if len(ev) >= 30:
            rows[name] = summarize(matrix(ev))
    ev_all = news[~news["evt_none"]].drop_duplicates(["symbol", "E"])
    rows["ALL tagged"] = summarize(matrix(ev_all))
    ev_none = news[news["evt_none"]].drop_duplicates(["symbol", "E"])
    if len(ev_none) >= 30:
        rows["untagged"] = summarize(matrix(ev_none))
    rng = np.random.default_rng(seed)
    fake = ev_all.copy()
    fake["E"] = rng.integers(5, len(dates) - 11, len(fake))
    rows["PLACEBO (random days)"] = summarize(matrix(fake))
    out = pd.DataFrame(rows).T
    out["events"] = out["events"].astype(int)
    pd.set_option("display.width", 200)
    print("Returns are beyond the market average, in percent. drift_t above about 3 deserves attention;")
    print("events cluster on the same dates, so treat the t values as optimistic.\n")
    print(out.round(2).sort_values("events", ascending=False).to_string())
    out.round(4).to_csv(p.news_dir.parent / "event_study.csv")
    return out
