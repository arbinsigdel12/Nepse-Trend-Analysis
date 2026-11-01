import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402

WL = ui.P.base / "watchlist" / "fundamentals.csv"
COLS = ["symbol", "as_of", "eps", "bvps", "npl_pct", "note"]


def render():
    c = ui.core()
    latest = c["latest"]
    ui.page_header("Watchlist and valuation",
                   "NEPSE does not publish earnings or book value as data, so you enter them here from each company's "
                   "quarterly report. Use EPS for a full year (the annualised figure, or the sum of the last four quarters). "
                   "The numbers are only as good as what you type in.")
    if latest is None:
        ui.empty_state("No data yet. Run:", "python -m nepse_ranker predict")
        return
    if WL.exists():
        sheet = pd.read_csv(WL)
        for col in COLS:
            if col not in sheet.columns:
                sheet[col] = None
        sheet = sheet[COLS]
    else:
        sheet = pd.DataFrame({"symbol": ["NABIL", "NTC"], "as_of": [None, None], "eps": [None, None],
                              "bvps": [None, None], "npl_pct": [None, None], "note": [None, None]})
    for col in ("eps", "bvps", "npl_pct"):
        sheet[col] = pd.to_numeric(sheet[col], errors="coerce")
    for col in ("symbol", "as_of", "note"):
        sheet[col] = sheet[col].astype("object")
    scr = ui.get(ui.P.base / "longterm" / "screen.parquet")
    options = sorted(scr["symbol"]) if scr is not None else sorted(latest["symbol"])
    with st.container(border=True):
        edited = ui.show_editor(sheet, num_rows="dynamic", key="watchlist", column_config={
            "symbol": st.column_config.SelectboxColumn("Stock", options=options, required=True),
            "as_of": st.column_config.TextColumn("Report period", help="for example 2082/83 Q1"),
            "eps": st.column_config.NumberColumn("EPS, full year (NPR)", format="%.2f"),
            "bvps": st.column_config.NumberColumn("Book value per share (NPR)", format="%.2f"),
            "npl_pct": st.column_config.NumberColumn("Bad loans % (banks)", format="%.2f"),
            "note": st.column_config.TextColumn("Note")})
        if st.button("Save to file", icon=":material/save:"):
            WL.parent.mkdir(parents=True, exist_ok=True)
            edited.to_csv(WL, index=False)
            st.success(f"Saved to {WL}")
    f1, f2, f3 = st.columns(3, gap="large")
    max_pe = f1.slider("Flag PE above", 5, 60, 25)
    max_pb = f2.slider("Flag PB above", 1.0, 10.0, 3.0)
    max_npl = f3.slider("Flag bad loans above (%)", 1.0, 15.0, 5.0)

    ed = edited.dropna(subset=["symbol"]).copy()
    for col in ("eps", "bvps", "npl_pct"):
        ed[col] = pd.to_numeric(ed[col], errors="coerce")
    ed = ed[ed["eps"].notna() | ed["bvps"].notna()].drop_duplicates("symbol", keep="last")
    if ed.empty:
        st.info("Enter EPS and book value for at least one stock above to see its valuation.")
        return
    px_all = ui.get(ui.P.prices_clean, columns=["date", "symbol", "close"])
    last_close = px_all.sort_values("date").groupby("symbol")["close"].last()
    ed["price"] = ed["symbol"].map(last_close)
    face = {}
    cls_file = ui.P.base / "company" / "classification.json"
    if cls_file.exists():
        for r in json.loads(cls_file.read_text(encoding="utf-8")):
            try:
                face[str(r["symbol"]).upper()] = float(r.get("faceValue") or 100)
            except (TypeError, ValueError):
                pass
    div_file = ui.P.base / "company" / "dividends.parquet"
    if div_file.exists():
        dv = pd.read_parquet(div_file).dropna(subset=["cash_pct"]).sort_values("fy_start")
        lastdiv = dv.groupby("symbol").last()
        ed["cash_pct"], ed["div_fy"] = ed["symbol"].map(lastdiv["cash_pct"]), ed["symbol"].map(lastdiv["fy_start"])
    else:
        ed["cash_pct"], ed["div_fy"] = np.nan, np.nan
    ed["dps"] = ed["cash_pct"] / 100 * ed["symbol"].map(face).fillna(100)
    ed["pe"] = np.where(ed["eps"] > 0, ed["price"] / ed["eps"], np.nan)
    ed["pb"] = np.where(ed["bvps"] > 0, ed["price"] / ed["bvps"], np.nan)
    ed["earn_yield"] = ed["eps"] / ed["price"] * 100
    ed["roe"] = np.where(ed["bvps"] > 0, ed["eps"] / ed["bvps"] * 100, np.nan)
    ed["cash_yield"] = ed["dps"] / ed["price"] * 100
    ed["payout"] = np.where(ed["eps"] > 0, ed["dps"] / ed["eps"] * 100, np.nan)

    def flags(r):
        out = []
        if pd.notna(r["eps"]) and r["eps"] <= 0:
            out.append("loss-making")
        if pd.notna(r["pe"]) and r["pe"] > max_pe:
            out.append("PE high")
        if pd.notna(r["pb"]) and r["pb"] > max_pb:
            out.append("PB high")
        if pd.notna(r["payout"]) and r["payout"] > 100:
            out.append("payout above earnings")
        if pd.notna(r["npl_pct"]) and r["npl_pct"] > max_npl:
            out.append("bad loans high")
        return ", ".join(out)

    ed["flags"] = ed.apply(flags, axis=1)
    sector_map = c["sectors"]
    view = pd.DataFrame({
        "Stock": ed["symbol"], "Sector": ed["symbol"].map(sector_map), "Price (NPR)": ed["price"], "PE": ed["pe"],
        "PB": ed["pb"], "Earnings yield %": ed["earn_yield"], "ROE %": ed["roe"],
        "Latest cash dividend %": ed["cash_pct"], "Dividend year (BS start)": ed["div_fy"],
        "Cash yield %": ed["cash_yield"], "Payout of EPS %": ed["payout"], "Bad loans %": ed["npl_pct"], "Flags": ed["flags"]})
    ui.show_df(view, hide_index=True, column_config={
        "Price (NPR)": st.column_config.NumberColumn(format="%.1f"), "PE": st.column_config.NumberColumn(format="%.1f"),
        "PB": st.column_config.NumberColumn(format="%.2f"), "Earnings yield %": st.column_config.NumberColumn(format="%.1f"),
        "ROE %": st.column_config.NumberColumn(format="%.1f"),
        "Latest cash dividend %": st.column_config.NumberColumn(format="%.1f"),
        "Dividend year (BS start)": st.column_config.NumberColumn(format="%d"),
        "Cash yield %": st.column_config.NumberColumn(format="%.1f"),
        "Payout of EPS %": st.column_config.NumberColumn(format="%.0f"),
        "Bad loans %": st.column_config.NumberColumn(format="%.1f")})
    st.caption("PE = price / EPS. PB = price / book value. ROE = EPS / book value. Cash yield and payout use the latest "
               "cash dividend found in announcements, as a percentage of a Rs 100 face value, before tax. A blank means "
               "unknown. A flag is a prompt to look closer, not a verdict, and what counts as cheap differs between sectors.")


render()
