"""Turn tagged news into model features, using only news that was already public.

A news item counts from its REACTION DAY onward: the first trading day whose close comes
after the item was published. The features for a day therefore never include anything the
market had not yet had a chance to see by that day's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .news_prep import reaction_positions

CAP = 40    # "days since" is capped here; 40 also means "no such news recently"
EVENT_FEATURES = ["right_share", "price_adjusted", "dividend", "bonus_share", "lock_in_end",
                  "merger_acquisition", "share_sale_intent", "project_disruption"]
NEGATIVE_EVENTS = ["right_share", "price_adjusted", "lock_in_end", "merger_acquisition",
                   "share_sale_intent", "project_disruption"]
NEWS_COLS = [f"since_{e}" for e in EVENT_FEATURES] + ["news_count_5", "neg_events_10"]


def _window_sum(count: np.ndarray, w: int) -> np.ndarray:
    cs = np.cumsum(count, axis=0)
    lag = np.vstack([np.zeros((w, count.shape[1])), cs[:-w]])
    return cs - lag


def add_news_features(df: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Add the NEWS_COLS columns to the feature table."""
    dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
    symbols = pd.Index(sorted(df["symbol"].unique()))
    T, S = len(dates), len(symbols)
    news = news[news["symbol"].isin(symbols)].copy()
    news["E"] = reaction_positions(news["published"], dates)
    news = news[news["E"] < T]
    e_pos = news["E"].to_numpy()
    s_pos = symbols.get_indexer(news["symbol"])
    row = np.arange(T)[:, None]
    out = {}

    for e in EVENT_FEATURES:
        sel = news["evt_" + e].to_numpy()
        flag = np.zeros((T, S), dtype=bool)
        flag[e_pos[sel], s_pos[sel]] = True
        last = np.maximum.accumulate(np.where(flag, row, -1), axis=0)
        out["since_" + e] = np.where(last >= 0, np.minimum(row - last, CAP), CAP).astype(float)

    count = np.zeros((T, S))
    np.add.at(count, (e_pos, s_pos), 1)
    out["news_count_5"] = _window_sum(count, 5)
    neg = news[["evt_" + e for e in NEGATIVE_EVENTS]].any(axis=1).to_numpy()
    neg_count = np.zeros((T, S))
    np.add.at(neg_count, (e_pos[neg], s_pos[neg]), 1)
    out["neg_events_10"] = _window_sum(neg_count, 10)

    di = dates.get_indexer(df["date"])
    si = symbols.get_indexer(df["symbol"])
    df = df.copy()
    for name, M in out.items():
        df[name] = M[di, si]
    return df


FLAG_LABELS = {"right_share": "right-share announcement", "price_adjusted": "ex-date price adjustment",
               "lock_in_end": "lock-in expiry", "merger_acquisition": "merger/acquisition news",
               "share_sale_intent": "insider share-sale notice", "project_disruption": "project disruption"}


def attach_news_flags(p) -> pd.DataFrame:
    """Add news_flag and news_note columns to the saved latest ranking.

    A stock is flagged when negative-type news (see FLAG_LABELS) reached the market within the
    last 5 trading days. It is a caution, and it does not change the ranking order.
    """
    latest = pd.read_parquet(p.latest)
    table = p.news_dir.parent / "news_table.parquet"
    latest["news_flag"] = False
    latest["news_note"] = ""
    if table.exists():
        df = pd.read_parquet(p.features, columns=["date", "symbol"])
        feat = add_news_features(df, pd.read_parquet(table))
        today = feat[feat["date"] == feat["date"].max()].set_index("symbol")
        notes = {}
        for sym, row in today.iterrows():
            hits = [FLAG_LABELS[e] for e in NEGATIVE_EVENTS if row["since_" + e] <= 5]
            if hits:
                notes[sym] = ", ".join(hits)
        latest["news_note"] = latest["symbol"].map(notes).fillna("")
        latest["news_flag"] = latest["news_note"] != ""
    latest.to_parquet(p.latest, index=False)
    return latest
