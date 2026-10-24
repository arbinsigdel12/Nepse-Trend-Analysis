"""Turn whatever column names a data source uses into one standard table.

Standard columns: date, symbol, open, high, low, close, prev_close, volume, turnover, trades
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

STANDARD = ["date", "symbol", "open", "high", "low", "close",
            "prev_close", "volume", "turnover", "trades"]
NUMERIC = ["open", "high", "low", "close", "prev_close", "volume", "turnover", "trades"]

# For each standard column, the names we recognise (lower-case, letters and digits only).
# The first match in this order wins. If your source uses another name, add it here.
ALIASES: dict[str, list[str]] = {
    "date": ["businessdate", "publisheddate", "date", "tradedate", "tradingdate", "asofdate"],
    "symbol": ["symbol", "securitysymbol", "stocksymbol", "ticker", "scrip"],
    "open": ["openprice", "open"],
    "high": ["highprice", "high", "maxprice"],
    "low": ["lowprice", "low", "minprice"],
    "close": ["closeprice", "close", "closingprice", "ltp", "lasttradedprice", "lastupdatedprice"],
    "prev_close": ["previousdaycloseprice", "previousdayclose", "previouscloseprice",
                   "previousclose", "previousclosing", "prevclose"],
    "volume": ["totaltradedquantity", "volume", "tradedshares", "tradedquantity",
               "sharestraded", "quantity", "qty", "totalshares"],
    "turnover": ["totaltradedvalue", "tradedamount", "turnover", "tradedvalue", "amount", "value"],
    "trades": ["totaltrades", "trades", "totaltransactions", "transactions",
               "nooftransactions", "numberoftransactions"],
}


def _key(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_mapping(columns) -> dict[str, str]:
    """Return {standard_name: original_column_name} for the columns we can recognise."""
    by_key: dict[str, Any] = {}
    for c in columns:
        by_key.setdefault(_key(c), c)
    mapping = {}
    for std, names in ALIASES.items():
        for n in names:
            if n in by_key:
                mapping[std] = by_key[n]
                break
    return mapping


def _to_number(s: pd.Series) -> pd.Series:
    if s.dtype == object or str(s.dtype).startswith("string"):
        s = s.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalize_frame(df: pd.DataFrame, default_date=None) -> tuple[pd.DataFrame, dict]:
    """Standardise a raw table. Returns (clean_table, report)."""
    mapping = find_mapping(df.columns)
    out = pd.DataFrame(index=df.index)
    for std in STANDARD:
        out[std] = df[mapping[std]] if std in mapping else float("nan")

    if "date" not in mapping and default_date is not None:
        out["date"] = pd.Timestamp(default_date)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if getattr(out["date"].dt, "tz", None) is not None:
        out["date"] = out["date"].dt.tz_localize(None)
    out["date"] = out["date"].dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out.loc[out["symbol"].isin(["NAN", "NONE", ""]), "symbol"] = None
    for c in NUMERIC:
        out[c] = _to_number(out[c])
    out = out.dropna(subset=["date", "symbol", "close"])
    out = out[out["close"] > 0].reset_index(drop=True)
    missing = [s for s in STANDARD if s not in mapping and not (s == "date" and default_date is not None)]
    return out, {"mapped": mapping, "missing": missing}


def rows_to_frame(rows: Any, default_date=None) -> tuple[pd.DataFrame, dict]:
    """Accept a list of dicts (what the NEPSE API returns) and standardise it."""
    if isinstance(rows, dict):
        rows = rows.get("content") or rows.get("data") or rows.get("result") or []
    if not rows:
        return pd.DataFrame(columns=STANDARD), {"mapped": {}, "missing": list(STANDARD)}
    return normalize_frame(pd.DataFrame(rows), default_date=default_date)
