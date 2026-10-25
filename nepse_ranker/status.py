"""One-screen health check: how fresh is every dataset, and what needs attention."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd

from .config import FETCH_AFTER_NPT, Paths, closed_weekdays
from .fetch import nepal_now


def _weekdays_after(start: date, end: date) -> int:
    n, d = 0, start + timedelta(days=1)
    while d <= end:
        if d.weekday() not in closed_weekdays(d):
            n += 1
        d += timedelta(days=1)
    return n


def _max_date(path, column) -> date | None:
    if not path.exists():
        return None
    s = pd.read_parquet(path, columns=[column])[column]
    return pd.Timestamp(s.max()).date() if len(s) else None


def _age_days(path, today: date) -> int | None:
    if not path.exists():
        return None
    return (today - datetime.fromtimestamp(path.stat().st_mtime).date()).days


def run_status(p: Paths | None = None) -> list:
    p = p or Paths()
    now = nepal_now()
    today = now.date()
    expected_last = today if (now.hour, now.minute) >= FETCH_AFTER_NPT else today - timedelta(days=1)
    rows = []

    def add(flag, name, text):
        rows.append((flag, name, text))

    # prices
    last_clean = _max_date(p.prices_clean, "date")
    if last_clean is None:
        add("MISSING", "Prices", "no cleaned prices yet. Run: python -m nepse_ranker build")
    else:
        behind = _weekdays_after(last_clean, expected_last)
        if behind == 0:
            add("OK", "Prices", f"up to date, last trading day {last_clean}")
        else:
            add("CHECK", "Prices", f"last day {last_clean}, {behind} trading weekday(s) behind. "
                "Normal during a market holiday, otherwise run: update, then build")
    # features, model, ranking
    last_feat = _max_date(p.features, "date")
    if last_feat is None:
        add("MISSING", "Features", "none yet. Run: python -m nepse_ranker features")
    elif last_clean and last_feat < last_clean:
        add("CHECK", "Features", f"last day {last_feat} is behind the prices ({last_clean}). Run: features")
    else:
        add("OK", "Features", f"last day {last_feat}")
    if p.model_meta.exists():
        trained = date.fromisoformat(json.loads(p.model_meta.read_text(encoding="utf-8"))["trained_through"])
        age = (today - trained).days
        add("OK" if age <= 45 else "CHECK", "Model",
            f"trained through {trained} ({age} days ago)" + ("" if age <= 45 else ". Retrain: python -m nepse_ranker train"))
    else:
        add("MISSING", "Model", "not trained yet. Run: python -m nepse_ranker train")
    last_rank = _max_date(p.latest, "date")
    if last_rank is None:
        add("MISSING", "Ranking", "none yet. Run: python -m nepse_ranker predict")
    elif last_feat and last_rank < last_feat:
        add("CHECK", "Ranking", f"ranking is for {last_rank}, features go to {last_feat}. Run: predict")
    else:
        add("OK", "Ranking", f"for {last_rank}")
    # news
    table = p.news_dir.parent / "news_table.parquet"
    last_news = _max_date(table, "published")
    raw_news = p.news_dir / "company_news.json"
    if last_news is None:
        add("MISSING", "News", "no news table. Run: python -m nepse_ranker news-fetch, then news-build")
    else:
        text = f"latest item {last_news}"
        flag = "OK"
        if raw_news.exists() and (today - datetime.fromtimestamp(raw_news.stat().st_mtime).date()).days > 6:
            flag, text = "CHECK", text + ". news-fetch has not run recently"
        elif raw_news.exists() and raw_news.stat().st_mtime > table.stat().st_mtime + 60:
            flag, text = "CHECK", text + ". New items downloaded but not processed: run news-build"
        add(flag, "News", text)
    # floorsheet
    fdir = p.base / "floorsheet"
    files = sorted(fdir.glob("*.parquet")) if fdir.exists() else []
    if not files:
        add("MISSING", "Floorsheet", "nothing collected. Run: python -m nepse_ranker floorsheet-collect")
    else:
        days = [date.fromisoformat(f.stem) for f in files]
        have = set(days)
        missed = []
        if last_clean:
            cal = pd.read_parquet(p.prices_clean, columns=["date", "traded"])
            cal = cal[cal["traded"] == 1]["date"].dt.date.unique()
            missed = [d for d in sorted(cal) if days[0] <= d <= days[-1] and d not in have]
        behind = _weekdays_after(days[-1], expected_last)
        flag = "OK" if behind == 0 and not missed else "CHECK"
        text = f"{len(days)} day(s) saved, latest {days[-1]}"
        if missed:
            text += f". {len(missed)} trading day(s) missed since collection began (cannot be recovered), e.g. {missed[-1]}"
        if behind:
            text += f". {behind} trading weekday(s) behind (normal during a market holiday): run floorsheet-collect after 3:30 PM"
        add(flag, "Floorsheet", text)
    # company data and long-term screen
    for name, path, fix in [("Company data", p.base / "company" / "corporate_actions.json", "company-fetch"),
                            ("Dividends", p.base / "company" / "dividends.parquet", "dividends-build"),
                            ("Long-term screen", p.base / "longterm" / "screen.parquet", "longterm-build")]:
        age = _age_days(path, today)
        if age is None:
            add("MISSING", name, f"not built. Run: python -m nepse_ranker {fix}")
        else:
            add("OK" if age <= 45 else "CHECK", name,
                f"built {age} day(s) ago" + ("" if age <= 45 else f". Refresh: python -m nepse_ranker {fix}"))
    # daily job log
    log = p.base.parent / "logs" / "daily.log"
    if not log.exists():
        add("CHECK", "Daily job", "no log yet. Has run_daily.ps1 ever run?")
    else:
        runs = [ln for ln in log.read_text(encoding="utf-8", errors="ignore").splitlines() if " RUN " in ln]
        add("OK" if runs and runs[-1].rstrip().endswith("OK") else "CHECK", "Daily job",
            runs[-1].strip() if runs else "log exists but no completed run")
    # disk
    size = sum(f.stat().st_size for f in p.base.rglob("*") if f.is_file()) / 1e9
    add("OK" if size < 20 else "CHECK", "Disk used", f"{size:.2f} GB in {p.base}")

    print(f"\nNEPSE Ranker status, {now:%Y-%m-%d %H:%M} Nepal time\n")
    for flag, name, text in rows:
        print(f"[{flag:7s}] {name:17s} {text}")
    need = sum(1 for r in rows if r[0] != "OK")
    print(f"\n{need} item(s) need attention." if need else "\nEverything is up to date.")
    return rows
