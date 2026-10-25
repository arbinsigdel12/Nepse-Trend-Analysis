"""Collect the market-wide floorsheet (every individual trade) once per day after the close.

The NEPSE client can only return the LATEST trading day's floorsheet, so history cannot be
downloaded later: each day you skip is lost. Run this every trading day after 3:30 PM Nepal
time. Each day is saved to its own compressed file and is never fetched twice.
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from .config import FETCH_AFTER_NPT, Paths
from .fetch import nepal_now


def _load_client():
    try:
        from nepsepy import NepseClient
    except ImportError as e:
        raise SystemExit("The 'nepsepy' package is missing. Run: pip install nepsepy") from e
    return NepseClient


def floorsheet_dir(p: Paths):
    return p.base / "floorsheet"


def _page_rows(resp) -> tuple[list, int | None, int | None]:
    """Return (rows, total_pages, reported_total_trades) from one floorsheet response."""
    if not isinstance(resp, dict):
        return [], None, None
    inner = resp.get("floorsheets", resp)
    rows = inner.get("content") if isinstance(inner, dict) else inner
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    pages = inner.get("totalPages") if isinstance(inner, dict) else None
    total = resp.get("totalTrades")
    return rows, pages, total


def _business_date(rows: list, fallback: str) -> str:
    for r in rows[:50]:
        for k, v in r.items():
            if re.fullmatch(r"(?i)(business|trade|trading)?_?date", k) and v:
                return str(v)[:10]
    return fallback


def collect_floorsheet(p: Paths | None = None, page_size: int = 500, max_pages: int = 3000) -> dict:
    p = p or Paths()
    out_dir = floorsheet_dir(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    NepseClient = _load_client()
    with NepseClient() as client:
        as_of = str(client.market_status().get("asOf", ""))[:10]
        now = nepal_now()
        if as_of == now.date().isoformat() and (now.hour, now.minute) < FETCH_AFTER_NPT:
            print("The market may still be open. Run this again after 3:30 PM Nepal time.")
            return {"saved": False, "reason": "market may be open"}
        if as_of and (out_dir / f"{as_of}.parquet").exists():
            print(f"{as_of}: already collected.")
            return {"saved": False, "reason": "already collected", "date": as_of}

        rows, reported, total_pages = [], None, None
        for page in range(1, max_pages + 1):
            chunk, pages, total = _page_rows(client.floorsheets(page=page, size=page_size))
            reported = reported if reported is not None else total
            total_pages = total_pages or pages
            if not chunk:
                break
            rows.extend(chunk)
            if page % 10 == 0:
                print(f"  {len(rows):,} trades so far ...", flush=True)
            if total_pages and page >= total_pages:
                break

    if not rows:
        print("No floorsheet rows came back.")
        return {"saved": False, "reason": "empty"}
    day = _business_date(rows, as_of or date.today().isoformat())
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df.insert(0, "collected_for", day)
    target = out_dir / f"{day}.parquet"
    if target.exists():
        print(f"{day}: already collected.")
        return {"saved": False, "reason": "already collected", "date": day}
    df.to_parquet(target, index=False)
    msg = f"{day}: saved {len(df):,} trades to {target}"
    if reported is not None and int(reported) != len(df):
        msg += f"  WARNING: NEPSE reports {int(reported):,} trades, we saved {len(df):,}"
    print(msg)
    return {"saved": True, "date": day, "rows": len(df), "reported": reported}


def inspect_floorsheet(p: Paths | None = None) -> str:
    p = p or Paths()
    files = sorted(floorsheet_dir(p).glob("*.parquet"))
    if not files:
        return "No floorsheet collected yet. Run: python -m nepse_ranker floorsheet-collect"
    lines = [f"{len(files)} day(s) collected, {files[0].stem} to {files[-1].stem}"]
    total = 0
    for f in files[-5:]:
        n = len(pd.read_parquet(f, columns=["collected_for"]))
        total += n
        lines.append(f"  {f.stem}: {n:,} trades")
    df = pd.read_parquet(files[-1])
    lines.append(f"columns: {list(df.columns)}")
    lines.append("sample row: " + str({k: (str(v)[:40]) for k, v in df.iloc[0].items()}))
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    lines.append(f"disk used so far: {size_mb:.1f} MB")
    return "\n".join(lines)
