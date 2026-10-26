"""Broker activity for every stock and day, computed straight from the saved floorsheet files.

Uses DuckDB, which reads the Parquet files directly and stays fast on tens of millions of trades.
"""
from __future__ import annotations

import pandas as pd

from .config import Paths

BIG_TRADE_NPR = 1_000_000     # a single trade of at least NPR 10 lakh counts as a block trade


def features_path(p: Paths):
    return p.base / "floorsheet_features.parquet"


def build_floorsheet_features(p: Paths | None = None) -> pd.DataFrame:
    import duckdb
    p = p or Paths()
    files = sorted((p.base / "floorsheet").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No floorsheet files. Run: python -m nepse_ranker floorsheet-collect")
    glob = (p.base / "floorsheet" / "*.parquet").as_posix()
    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW t AS
        SELECT collected_for AS day, stockSymbol AS symbol,
               CAST(buyerMemberId AS BIGINT) AS buyer, CAST(sellerMemberId AS BIGINT) AS seller,
               CAST(contractQuantity AS DOUBLE) AS qty, CAST(contractAmount AS DOUBLE) AS amt
        FROM read_parquet('{glob}')
    """)
    df = con.execute(f"""
        WITH tot AS (
            SELECT day, symbol, count(*) AS trades, sum(amt) AS turnover, sum(qty) AS shares,
                   count(DISTINCT buyer) AS buyers, count(DISTINCT seller) AS sellers,
                   sum(CASE WHEN buyer = seller THEN amt ELSE 0 END) AS self_amt,
                   sum(CASE WHEN amt >= {BIG_TRADE_NPR} THEN amt ELSE 0 END) AS big_amt
            FROM t GROUP BY day, symbol),
        b AS (SELECT day, symbol, buyer AS broker, sum(amt) AS buy_amt FROM t GROUP BY day, symbol, buyer),
        s AS (SELECT day, symbol, seller AS broker, sum(amt) AS sell_amt FROM t GROUP BY day, symbol, seller),
        br AS (
            SELECT coalesce(b.day, s.day) AS day, coalesce(b.symbol, s.symbol) AS symbol,
                   coalesce(b.broker, s.broker) AS broker,
                   coalesce(buy_amt, 0) AS buy_amt, coalesce(sell_amt, 0) AS sell_amt
            FROM b FULL OUTER JOIN s ON b.day = s.day AND b.symbol = s.symbol AND b.broker = s.broker),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY day, symbol ORDER BY buy_amt DESC) AS buy_rank,
                      row_number() OVER (PARTITION BY day, symbol ORDER BY sell_amt DESC) AS sell_rank,
                      buy_amt - sell_amt AS net
            FROM br),
        agg AS (
            SELECT r.day, r.symbol,
                   sum(CASE WHEN buy_rank <= 3 THEN buy_amt END) AS top3_buy,
                   sum(CASE WHEN sell_rank <= 3 THEN sell_amt END) AS top3_sell,
                   max(CASE WHEN buy_rank = 1 THEN buy_amt END) AS top1_buy,
                   max(CASE WHEN sell_rank = 1 THEN sell_amt END) AS top1_sell,
                   sum(power(buy_amt / tot.turnover, 2)) AS hhi_buy,
                   sum(power(sell_amt / tot.turnover, 2)) AS hhi_sell,
                   max(net) AS max_net_buy, min(net) AS max_net_sell
            FROM ranked r JOIN tot ON r.day = tot.day AND r.symbol = tot.symbol
            GROUP BY r.day, r.symbol)
        SELECT tot.day, tot.symbol, trades, turnover, shares, buyers, sellers,
               turnover / trades AS avg_trade_npr,
               top1_buy / turnover AS top1_buy_share, top3_buy / turnover AS top3_buy_share,
               top1_sell / turnover AS top1_sell_share, top3_sell / turnover AS top3_sell_share,
               hhi_buy, hhi_sell, self_amt / turnover AS self_trade_share, big_amt / turnover AS block_trade_share,
               max_net_buy / turnover AS top_net_buyer_share, -max_net_sell / turnover AS top_net_seller_share
        FROM tot JOIN agg ON tot.day = agg.day AND tot.symbol = agg.symbol
        ORDER BY tot.day, tot.symbol
    """).df()
    df["day"] = pd.to_datetime(df["day"])
    df.to_parquet(features_path(p), index=False)
    return df


def run_floorsheet_build(p: Paths | None = None) -> pd.DataFrame:
    import duckdb
    p = p or Paths()
    df = build_floorsheet_features(p)
    last = df["day"].max()
    print(f"Floorsheet days: {df['day'].nunique()}, stock-days: {len(df):,}, latest day {last.date()}")
    from .company import _read, company_dir
    shares = set(_read(company_dir(p) / "sectors.json", {}))
    d = df[(df["day"] == last) & (df["turnover"] >= 10_000_000) & (df["trades"] >= 100)].copy()
    if shares:
        d = d[d["symbol"].isin(shares)]
    print(f"\nOrdinary shares on {last.date()} with at least NPR 1 crore turnover and 100 trades: {len(d)}")
    pd.set_option("display.width", 200)
    show = ["symbol", "trades", "turnover", "top3_buy_share", "top3_sell_share", "self_trade_share", "block_trade_share"]

    def table(title, col, n=8):
        print(f"\n{title}")
        t = d.sort_values(col, ascending=False).head(n)[show].copy()
        t["turnover"] = (t["turnover"] / 1e7).round(1)
        for c in show[3:]:
            t[c] = (t[c] * 100).round(1)
        print(t.rename(columns={"turnover": "turnover_crore"}).to_string(index=False))

    table("Most concentrated buying (top 3 brokers' share of turnover, %):", "top3_buy_share")
    table("Highest same-broker trade share (%):", "self_trade_share")
    table("Most turnover in block trades (%):", "block_trade_share")
    files = (p.base / "floorsheet" / "*.parquet").as_posix()
    con = duckdb.connect()
    top = con.execute(f"""
        SELECT buyerBrokerName AS broker, sum(CAST(contractAmount AS DOUBLE)) / 1e7 AS bought_crore, count(*) AS trades
        FROM read_parquet('{files}') WHERE collected_for = '{last.date()}'
        GROUP BY buyerBrokerName ORDER BY bought_crore DESC LIMIT 8""").df()
    print(f"\nBiggest buying brokers on {last.date()}:")
    print(top.round(1).to_string(index=False))
    print(f"\nSaved to {features_path(p)}")
    return df
