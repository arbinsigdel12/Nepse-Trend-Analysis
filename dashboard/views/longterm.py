import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402

SORTS = {"Lowest volatility": ("vol_ann", True), "Smallest past fall": ("max_dd_5y", False),
         "Most traded": ("median_turnover_120d", False), "Most checks passed": ("checks_passed", False),
         "Highest 5-year return (past performance only)": ("cagr_5y", False)}


def render():
    scr = ui.get(ui.P.base / "longterm" / "screen.parquet")
    ui.page_header("Long-term screen",
                   "A risk and history screen, not a valuation. It checks how long a stock has traded, how easily it trades, "
                   "how deep its past falls were, how volatile it is, whether it keeps issuing right shares and whether it has "
                   "recent red-flag news. Earnings and book value are not in the data, and cash dividends are only partly known "
                   "(read from announcement text; blank means unknown). Passing is not a reason to buy.")
    if scr is None:
        ui.empty_state("No long-term screen yet. Run this in PowerShell, then refresh:", "python -m nepse_ranker longterm-build")
        return
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1.2, 1.8, 1.2])
        pick = c1.multiselect("Sectors (empty means all)", sorted(scr["sector"].dropna().unique()))
        min_checks = c2.slider("Checks passed", 0, 6, 5)
        sort_by = c3.selectbox("Sort by", list(SORTS))
        n_show = c4.slider("Rows", 10, 150, 40)
    d = scr[scr["checks_passed"] >= min_checks]
    if pick:
        d = d[d["sector"].isin(pick)]
    col, asc = SORTS[sort_by]
    d = d.sort_values(col, ascending=asc, na_position="last")

    ui.kpi_grid([
        ("Stocks shown", f"{len(d)}", f"of {len(scr)} screened"),
        ("Pass all 6", f"{int((scr['checks_passed'] == 6).sum())}", "checks passed"),
        ("Median volatility", f"{d['vol_ann'].median() * 100:.0f}%" if len(d) else "n/a", "yearly"),
        ("Median worst fall", f"{d['max_dd_5y'].median() * 100:.0f}%" if len(d) else "n/a", "over up to 5 years"),
    ])
    ui.gap(6)

    plot = d.dropna(subset=["cagr_5y", "vol_ann"]).head(300).copy()
    if len(plot):
        ui.section("Risk and past return", "Each dot is a stock. Left is calmer, up is better past return. Hover for details.")
        plot["vol_pct"], plot["ret_pct"] = plot["vol_ann"] * 100, plot["cagr_5y"] * 100
        plot["size"] = np.sqrt(plot["median_turnover_120d"].clip(lower=1e5))
        fig = px.scatter(plot, x="vol_pct", y="ret_pct", color="sector", size="size", size_max=24, hover_name="symbol",
                         hover_data={"vol_pct": ":.0f", "ret_pct": ":.1f", "size": False, "checks_passed": True})
        fig.update_xaxes(title="Yearly volatility, %")
        fig.update_yaxes(title="Return per year over 5 years, % (bonus-adjusted, no cash dividends)")
        st.plotly_chart(ui.style_fig(fig, 420, hover="closest"))

    table = d.head(n_show)
    import pandas as pd
    view = pd.DataFrame({
        "Symbol": table["symbol"], "Sector": table["sector"], "Checks": table["checks_passed"],
        "Years listed": table["listed_years"], "Return/yr 5y %": table["cagr_5y"] * 100,
        "Volatility % (yearly)": table["vol_ann"] * 100, "Worst fall 5y %": table["max_dd_5y"] * 100,
        "Median turnover (NPR m)": table["median_turnover_120d"] / 1e6,
        "Dividend years (5 FY)": table["dividend_years_5fy"] if "dividend_years_5fy" in table else None,
        "Avg cash div % (3 FY)": table["cash_avg_3fy_pct"] if "cash_avg_3fy_pct" in table else None,
        "Cash yield %": table["cash_yield"] * 100 if "cash_yield" in table else None,
        "Rights issues (5 FY)": table["rights_issues_5fy"], "Red flags (1y)": table["red_flags_1y"],
        "Failed checks": table["fails"]})
    fmt = {k: st.column_config.NumberColumn(format="%.1f") for k in
           ("Years listed", "Return/yr 5y %", "Volatility % (yearly)", "Worst fall 5y %", "Median turnover (NPR m)",
            "Avg cash div % (3 FY)", "Cash yield %")}
    fmt["Checks"] = st.column_config.ProgressColumn("Checks", min_value=0, max_value=6, format="%d")
    ui.show_df(view, hide_index=True, column_config=fmt)
    with st.expander("What the six checks are"):
        st.markdown("""
1. **History:** listed at least 5 years, with at least 3 years of price data here.
2. **Trading:** median daily turnover over the last 120 days of at least NPR 1 million.
3. **Past falls:** worst peak-to-trough fall over up to 5 years better than -60%.
4. **Volatility:** yearly price swings below 50%.
5. **Dilution:** at most one right-share issue in the last 5 fiscal years.
6. **News:** at most one red-flag item in the last year.
""")


render()
