"""Download official NEPSE news: company disclosures, news and alerts, press releases, notices.

These come from the same NEPSE website data as the prices, through the same client, so
if price download works, this should too. Each run ADDS new items to what you already
saved (nothing is overwritten), so history builds up over time.
"""
from __future__ import annotations

import hashlib
import json
import re

from .config import Paths

# source name -> (how to fetch page N, maximum pages to try)
SOURCES = {
    "company_news": (lambda c, page: c.company_news_list() if page == 1 else [], 1),
    "news_alerts": (lambda c, page: c.news_alerts(page), 40),
    "press_releases": (lambda c, page: c.press_releases(page), 20),
    "notices": (lambda c, page: c.notices(page), 20),
}


def _rows(obj) -> list[dict]:
    """Pull the list of items out of whatever shape the API returned."""
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for k in ("content", "data", "result", "items"):
            v = obj.get(k)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
    return []


def _key(row: dict) -> str:
    if row.get("id") is not None:
        return f"id:{row['id']}"
    return "h:" + hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()


def _load_client():
    try:
        from nepsepy import NepseClient, RateLimitedError
    except ImportError as e:
        raise SystemExit("The 'nepsepy' package is missing. Run: pip install nepsepy") from e
    return NepseClient, RateLimitedError


def news_path(p: Paths, source: str):
    return p.news_dir / f"{source}.json"


def load_saved(p: Paths, source: str) -> list[dict]:
    f = news_path(p, source)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def fetch_news(p: Paths | None = None) -> dict:
    """Download every news source and merge it into the saved files."""
    p = p or Paths()
    p.news_dir.mkdir(parents=True, exist_ok=True)
    NepseClient, RateLimitedError = _load_client()
    stats = {}
    with NepseClient() as client:
        for name, (fetch_page, max_pages) in SOURCES.items():
            saved = load_saved(p, name)
            seen = {_key(r) for r in saved}
            added, error = 0, ""
            for page in range(1, max_pages + 1):
                try:
                    rows = _rows(fetch_page(client, page))
                except RateLimitedError:
                    error = "rate limited: wait a few minutes and run again"
                    break
                except Exception as e:
                    error = f"{type(e).__name__}: {str(e)[:100]}"
                    break
                if not rows:
                    break
                new = [r for r in rows if _key(r) not in seen]
                for r in new:
                    seen.add(_key(r))
                saved.extend(new)
                added += len(new)
                if not new and page > 1:
                    break           # nothing new on this page: we are caught up
            news_path(p, name).write_text(json.dumps(saved), encoding="utf-8")
            stats[name] = {"total": len(saved), "new": added, "error": error}
            print(f"{name}: {len(saved):,} items ({added:,} new)" + (f"  ERROR {error}" if error else ""))
    return stats


def inspect_news(p: Paths | None = None) -> str:
    """Show what each saved news source looks like, so we can map its fields correctly."""
    p = p or Paths()
    lines = []
    for name in SOURCES:
        rows = load_saved(p, name)
        lines.append(f"=== {name}: {len(rows):,} items")
        if not rows:
            continue
        keys = sorted({k for r in rows[:200] for k in r})
        lines.append(f"fields: {keys}")
        date_keys = [k for k in keys if re.search("date|time", k, re.I)]
        for k in date_keys:
            vals = sorted(str(r[k]) for r in rows if r.get(k))
            if vals:
                lines.append(f"  {k}: from {vals[0][:19]} to {vals[-1][:19]}")
        sample = {k: (str(v)[:110] + "...") if len(str(v)) > 110 else v for k, v in rows[0].items()}
        lines.append(f"sample: {sample}")
    return "\n".join(lines)
