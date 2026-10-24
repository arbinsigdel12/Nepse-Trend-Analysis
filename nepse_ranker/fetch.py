"""Download daily prices from NEPSE using the unofficial `nepsepy` library.

This is an unofficial, community-built client for NEPSE's public website data.
It paces its own requests. NEPSE may change its site at any time, which can break
this step. If that happens, use `import-csv` instead.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone

from .config import FETCH_AFTER_NPT, NEPAL_UTC_OFFSET_MINUTES, Paths, closed_weekdays
from .normalize import rows_to_frame
from .store import list_raw_days, raw_day_exists, raw_day_path, save_raw_day


def nepal_now() -> datetime:
    return datetime.now(timezone(timedelta(minutes=NEPAL_UTC_OFFSET_MINUTES)))


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _load_client():
    try:
        from nepsepy import NepseClient, RateLimitedError
    except ImportError as e:
        raise SystemExit("The 'nepsepy' package is missing. Run: pip install nepsepy") from e
    return NepseClient, RateLimitedError


def fetch_day(client, day: date, tries: int = 3) -> list:
    """Ask NEPSE for one date's price table. Retries on timeouts and on empty answers."""
    last_error = None
    for attempt in range(1, tries + 1):
        try:
            rows = client.today_price_all(business_date=day.isoformat())
            if isinstance(rows, dict):
                rows = rows.get("content") or rows.get("data") or []
            if rows:
                return rows
            last_error = None   # empty answer: maybe a holiday, maybe a hiccup
        except Exception as e:
            last_error = e
        if attempt < tries:
            time.sleep(2 * attempt)
    if last_error is not None:
        raise last_error
    return []


def backfill(start: date, end: date | None = None, p: Paths | None = None) -> dict:
    """Download every missing trading day between start and end (safe to re-run)."""
    p = p or Paths()
    p.ensure()
    NepseClient, RateLimitedError = _load_client()
    now = nepal_now()
    end = end or now.date()
    stats = {"downloaded": 0, "empty_days": 0, "skipped_existing": 0, "errors": 0, "stopped": ""}
    consecutive_errors = 0

    with NepseClient() as client:
        for day in _daterange(start, end):
            if day.weekday() in closed_weekdays(day):
                continue
            if raw_day_exists(p, day):
                stats["skipped_existing"] += 1
                continue
            if day == now.date() and (now.hour, now.minute) < FETCH_AFTER_NPT:
                print(f"{day}: market may still be open, skipping today. Run again after 3:30 PM Nepal time.")
                continue
            try:
                rows = fetch_day(client, day)
            except RateLimitedError:
                stats["stopped"] = "NEPSE asked us to slow down. Wait a few minutes and run again."
                break
            except Exception as e:
                stats["errors"] += 1
                consecutive_errors += 1
                print(f"{day}: error: {type(e).__name__}: {e}")
                if consecutive_errors >= 5:
                    stats["stopped"] = "5 errors in a row. Stopping so we do not hammer the server."
                    break
                continue
            consecutive_errors = 0
            save_raw_day(p, day, rows)
            if rows:
                stats["downloaded"] += 1
                print(f"{day}: {len(rows)} rows")
            else:
                stats["empty_days"] += 1
                print(f"{day}: no trading (holiday or no data)")
    return stats


def update(p: Paths | None = None) -> dict:
    """Catch up from the day after the last saved day until today."""
    p = p or Paths()
    days = list_raw_days(p)
    if not days:
        raise SystemExit("Nothing downloaded yet. Run `fetch --start YYYY-MM-DD` first.")
    return backfill(days[-1] + timedelta(days=1), None, p)


def inspect_raw(p: Paths | None = None) -> str:
    """Show what the newest saved file looks like and which columns we understood."""
    p = p or Paths()
    days = [d for d in list_raw_days(p) if json.loads(raw_day_path(p, d).read_text())]
    if not days:
        return "No saved days with data yet. Run `fetch` first."
    rows = json.loads(raw_day_path(p, days[-1]).read_text())
    df, report = rows_to_frame(rows, default_date=days[-1])
    lines = [f"File: {raw_day_path(p, days[-1])}",
             f"Rows in file: {len(rows)}",
             f"Columns in file: {sorted(rows[0].keys())}",
             f"Recognised as: {report['mapped']}",
             f"NOT recognised: {report['missing']}",
             f"Sample row: {rows[0]}",
             f"Rows after cleaning: {len(df)}"]
    if "close" in report["missing"] or "symbol" in report["missing"]:
        lines.append("PROBLEM: symbol or close price was not recognised. "
                     "Add the real column names to ALIASES in nepse_ranker/normalize.py.")
    return "\n".join(lines)
