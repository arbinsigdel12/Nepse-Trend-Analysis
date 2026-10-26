"""Command line: python -m nepse_ranker <command>"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date

import pandas as pd

from .config import Paths


def _build(p):
    from .clean import run_clean
    from .store import build_prices_raw
    raw = build_prices_raw(p)
    panel, report = run_clean(raw, p)
    print(f"Rows: {report['rows']:,} | stocks: {report['symbols']} | days: {report['trading_days']}")
    print(f"Dates: {report['first_date']} to {report['last_date']}")
    print(f"Corporate actions found: {report['corporate_actions_found']}")
    for w in report["warnings"]:
        print("WARNING:", w)
    for i in report["issues"]:
        print("ISSUE:", i)
    return panel


def _features(p):
    from .features import run_features
    panel = pd.read_parquet(p.prices_clean)
    df = run_features(panel, p)
    n_elig = int(df["eligible"].sum())
    print(f"Feature table: {len(df):,} rows, {n_elig:,} tradable stock-days")
    return df


def _train(p, objective, max_folds):
    from .evaluate import evaluate_all
    from .model import train_final, walk_forward
    df = pd.read_parquet(p.features)
    print("Walk-forward testing (train on the past, test on the next block):")
    oos = walk_forward(df, objective=objective, max_folds=max_folds)
    oos.to_parquet(p.oos, index=False)
    res = evaluate_all(oos, p=p)
    cols = ["mean_ic", "ic_t_stat", "topk_gross_excess", "topk_net_excess",
            "topk_hit_rate", "base_hit_rate", "periods"]
    print("\nHow the model compares with simple rules:")
    print(res["summary"][cols].round(4).to_string())
    print("\nVerdict:", res["verdict"]["level"].upper())
    print(res["verdict"]["text"])
    meta = train_final(df, objective=objective, p=p)
    print(f"\nFinal model trained on {meta['rows']:,} rows through {meta['trained_through']}")


def _predict(p):
    from .model import score_latest
    df = pd.read_parquet(p.features)
    out = score_latest(df, p)
    from .news_features import attach_news_flags
    out = attach_news_flags(p)
    print(f"Ranking for {out['date'].iloc[0].date()}  ({len(out)} tradable stocks)")
    show = out.head(10)[["rank", "symbol", "score_pct", "close", "ret_1", "ret_5", "turn_20"]]
    print(show.round(3).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nepse_ranker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download prices from NEPSE")
    f.add_argument("--start", required=True, help="YYYY-MM-DD")
    f.add_argument("--end", help="YYYY-MM-DD (default: today)")
    sub.add_parser("update", help="download any days missed since the last run")
    sub.add_parser("news-fetch", help="download official NEPSE news and disclosures")
    sub.add_parser("news-inspect", help="show what the saved news looks like")
    sub.add_parser("floorsheet-build", help="broker concentration and trade-pattern measures from the floorsheet")
    sub.add_parser("status", help="health check: how fresh is every dataset")
    sub.add_parser("dividends-build", help="extract cash dividend and bonus declarations from announcements")
    sub.add_parser("longterm-build", help="build the long-term risk and history screen")
    sub.add_parser("company-fetch", help="download sectors and dividend/bonus/right history")
    sub.add_parser("company-inspect", help="summarise the downloaded company data")
    sub.add_parser("floorsheet-collect", help="save today's floorsheet (every trade) after the close")
    sub.add_parser("floorsheet-inspect", help="show what has been collected so far")
    sub.add_parser("news-build", help="clean the news, link it to stocks, tag event types")
    sub.add_parser("news-study", help="check what happens to prices after each kind of news")
    sub.add_parser("inspect-raw", help="show what the downloaded data looks like")
    c = sub.add_parser("import-csv", help="load a CSV you downloaded yourself")
    c.add_argument("path")
    fo = sub.add_parser("import-folder", help="load a folder of per-stock CSV files")
    fo.add_argument("path")
    sub.add_parser("build", help="combine raw data, check it, adjust for bonus shares")
    sub.add_parser("features", help="create features and labels")
    for name in ("train", "run-all"):
        t = sub.add_parser(name)
        t.add_argument("--objective", default="regression", choices=["regression", "lambdarank"])
        t.add_argument("--max-folds", type=int, default=None, help="test only the latest N folds (faster)")
    sub.add_parser("predict", help="rank stocks for the latest date")
    sub.add_parser("dashboard", help="open the dashboard in your browser")
    a = ap.parse_args(argv)
    p = Paths()

    if a.cmd == "fetch":
        from .fetch import backfill
        end = date.fromisoformat(a.end) if a.end else None
        print(backfill(date.fromisoformat(a.start), end, p))
    elif a.cmd == "update":
        from .fetch import update
        print(update(p))
    elif a.cmd == "news-fetch":
        from .news import fetch_news
        fetch_news(p)
    elif a.cmd == "news-build":
        from .news_prep import run_news_build
        run_news_build(p)
    elif a.cmd == "news-study":
        from .news_prep import run_event_study
        run_event_study(p)
    elif a.cmd == "floorsheet-collect":
        from .floorsheet import collect_floorsheet
        collect_floorsheet(p)
    elif a.cmd == "floorsheet-inspect":
        from .floorsheet import inspect_floorsheet
        print(inspect_floorsheet(p))
    elif a.cmd == "company-fetch":
        from .company import fetch_company_data
        fetch_company_data(p)
    elif a.cmd == "company-inspect":
        from .company import inspect_company
        print(inspect_company(p))
    elif a.cmd == "longterm-build":
        from .longterm import run_longterm_build
        run_longterm_build(p)
    elif a.cmd == "dividends-build":
        from .dividends import run_dividends_build
        run_dividends_build(p)
    elif a.cmd == "status":
        from .status import run_status
        run_status(p)
    elif a.cmd == "floorsheet-build":
        from .floorsheet_features import run_floorsheet_build
        run_floorsheet_build(p)
    elif a.cmd == "news-inspect":
        from .news import inspect_news
        print(inspect_news(p))
    elif a.cmd == "inspect-raw":
        from .fetch import inspect_raw
        print(inspect_raw(p))
    elif a.cmd == "import-csv":
        from .store import import_csv
        n, rep = import_csv(p, a.path)
        print(f"Imported {n:,} rows. Recognised: {rep['mapped']}  Missing: {rep['missing']}")
    elif a.cmd == "import-folder":
        from .store import import_folder
        n, k = import_folder(p, a.path)
        print(f"Imported {n:,} rows from {k} stocks.")
    elif a.cmd == "build":
        _build(p)
    elif a.cmd == "features":
        _features(p)
    elif a.cmd == "train":
        _train(p, a.objective, a.max_folds)
    elif a.cmd == "predict":
        _predict(p)
    elif a.cmd == "run-all":
        _build(p)
        _features(p)
        _train(p, a.objective, a.max_folds)
        _predict(p)
    elif a.cmd == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], check=False)


if __name__ == "__main__":
    main()
