"""Download company-level facts from NEPSE: sector, listing date, instrument type, and the
bonus / cash dividend / right-share history of every ordinary share.

Safe to re-run: stocks already downloaded are skipped, so an interrupted run just resumes.
"""
from __future__ import annotations

import json
from collections import Counter

from .config import Paths

SHOW_SYMBOLS = ["NABIL", "NTC", "NBL", "SHIVM", "UPPER"]


def company_dir(p: Paths):
    return p.base / "company"


def _load_client():
    try:
        from nepsepy import NepseClient
    except ImportError as e:
        raise SystemExit("The 'nepsepy' package is missing. Run: pip install nepsepy") from e
    return NepseClient


def _rows(obj) -> list[dict]:
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for k in ("content", "data", "result"):
            if isinstance(obj.get(k), list):
                return [r for r in obj[k] if isinstance(r, dict)]
    return []


def _is_equity(row: dict) -> bool:
    it = row.get("instrumentType")
    code = it.get("code") if isinstance(it, dict) else it
    return str(code).upper() == "EQ"


def _read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def fetch_company_data(p: Paths | None = None) -> dict:
    p = p or Paths()
    d = company_dir(p)
    d.mkdir(parents=True, exist_ok=True)
    NepseClient = _load_client()
    with NepseClient() as c:
        cls = []
        for page in range(1, 60):
            resp = c.classification(page=page, size=100)
            rows = _rows(resp)
            if not rows:
                break
            cls.extend(rows)
            if isinstance(resp, dict) and resp.get("last"):
                break
        if cls:
            (d / "classification.json").write_text(json.dumps(cls), encoding="utf-8")
        else:
            cls = _read(d / "classification.json", [])
        universe = {str(r["symbol"]).upper(): int(r["id"]) for r in cls
                    if _is_equity(r) and str(r.get("isPromoter", "N")).upper() != "Y"
                    and r.get("symbol") and r.get("id") is not None}
        print(f"Ordinary shares found in the classification list: {len(universe)}")

        actions = _read(d / "corporate_actions.json", {})
        sectors = _read(d / "sectors.json", {})

        def save():
            (d / "corporate_actions.json").write_text(json.dumps(actions), encoding="utf-8")
            (d / "sectors.json").write_text(json.dumps(sectors), encoding="utf-8")

        errors, done = 0, 0
        for i, (sym, sid) in enumerate(sorted(universe.items()), 1):
            if sym in actions and sym in sectors:
                continue
            try:
                acts = _rows(c.corporate_actions(sid))
                det = c.security_detail(sid)
                sector = (det.get("securityData") or {}).get("sector") if isinstance(det, dict) else None
            except Exception as e:
                errors += 1
                print(f"{sym}: {type(e).__name__}: {str(e)[:80]}")
                if errors >= 8:
                    print("8 errors in a row. Stopping; run the same command again later to resume.")
                    break
                continue
            errors = 0
            actions[sym], sectors[sym] = acts, sector
            done += 1
            if done % 25 == 0:
                save()
                print(f"  {len(actions)} of {len(universe)} stocks done ...", flush=True)
        save()
    print(f"Saved: {len(actions)} stocks with corporate-action history, {len(sectors)} with a sector.")
    return {"stocks": len(actions), "universe": len(universe)}


def inspect_company(p: Paths | None = None) -> str:
    p = p or Paths()
    d = company_dir(p)
    cls = _read(d / "classification.json", [])
    actions = _read(d / "corporate_actions.json", {})
    sectors = _read(d / "sectors.json", {})
    if not actions:
        return "Nothing downloaded yet. Run: python -m nepse_ranker company-fetch"
    rows = [r for v in actions.values() for r in v]
    lines = [f"Classification entries: {len(cls)} | stocks with downloaded history: {len(actions)} | "
             f"records in total: {len(rows):,}"]
    lines.append("Stocks with no records at all: " + str(sum(1 for v in actions.values() if not v)))
    lines.append("Sectors: " + str(dict(Counter(sectors.values()).most_common())))
    dates = sorted(str(r.get("listingDate"))[:10] for r in cls if r.get("listingDate"))
    if dates:
        lines.append(f"Listing dates: {dates[0]} to {dates[-1]}")
    lines.append("Record types (activeStatus): " + str(dict(Counter(r.get("activeStatus") for r in rows).most_common())))
    for field in ("bonusPercentage", "cashDividend", "rightPercentage"):
        lines.append(f"records with a {field} value: {sum(1 for r in rows if r.get(field) not in (None, ''))}")
    years = Counter(str(r.get("fiscalYear")) for r in rows)
    lines.append("Records per fiscal year: " + str(dict(sorted(years.items()))))
    for sym in SHOW_SYMBOLS:
        if sym in actions:
            lines.append(f"\n--- {sym}: {len(actions[sym])} records")
            for r in sorted(actions[sym], key=lambda r: str(r.get("fiscalYear"))):
                lines.append("  " + str({k: r.get(k) for k in ["fiscalYear", "activeStatus", "bonusPercentage",
                             "cashDividend", "rightPercentage", "rightAmountPerShare"]}))
    return "\n".join(lines)
