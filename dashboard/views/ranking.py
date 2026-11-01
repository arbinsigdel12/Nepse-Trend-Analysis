import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from nepse_ranker.config import HORIZON  # noqa: E402


def sparklines(symbols, n=30):
    px = ui.get(ui.P.prices_clean, columns=["date", "symbol", "adj_close"])
    if px is None:
        return {}
    px = px[px["symbol"].isin(symbols)].sort_values(["symbol", "date"])
    return {s: g["adj_close"].tail(n).round(2).tolist() for s, g in px.groupby("symbol")}


def render():
    c = ui.core()
    latest, sectors = c["latest"], c["sectors"]
    if latest is None:
        ui.page_header("Stock ranking")
        ui.empty_state("No ranking found yet. Run this in PowerShell, then refresh:", "python -m nepse_ranker predict")
        return
    asof = pd.Timestamp(latest["date"].iloc[0])
    ui.page_header("Stock ranking", f"Model ordering of {len(latest)} tradable stocks as of {asof:%d %B %Y}. Score 100 is "
                   "the model's favourite. It is a ranking aid, not a price forecast. Click a row to see its chart.")

    d = latest.copy()
    d["sector"] = d["symbol"].map(sectors).fillna("Other")
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([2, 1.2, 1.4, 1.1])
        pick = f1.multiselect("Sectors (empty means all)", sorted(d["sector"].unique()))
        min_turn = f2.number_input("Min avg turnover (NPR m)", 0.0, 500.0, 0.0, step=1.0)
        query = f3.text_input("Find a symbol").strip().upper()
        hide_flag = f4.toggle("Hide stocks with bad news", value=False)
    if pick:
        d = d[d["sector"].isin(pick)]
    d = d[d["turn_20"] / 1e6 >= min_turn]
    if query:
        d = d[d["symbol"].str.contains(query, regex=False)]
    if hide_flag and "news_flag" in d:
        d = d[~d["news_flag"]]
    n = st.slider("Rows to show", 10, 100, 25)
    top = d.sort_values("rank").head(n)
    spark = sparklines(list(top["symbol"]))
    view = pd.DataFrame({
        "Rank": top["rank"], "Symbol": top["symbol"], "Sector": top["sector"], "Score": top["score_pct"],
        "Price (NPR)": top["close"], "1d %": top["ret_1"] * 100, "5d %": top["ret_5"] * 100,
        "20d %": top["ret_20"] * 100, "30-day trend": top["symbol"].map(spark),
        "Avg turnover (NPR m)": top["turn_20"] / 1e6,
        "Bad news": top["news_note"] if "news_note" in top else ""}).reset_index(drop=True)
    cfg = {"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
           "Price (NPR)": st.column_config.NumberColumn(format="%.1f"),
           "1d %": st.column_config.NumberColumn(format="%+.2f"), "5d %": st.column_config.NumberColumn(format="%+.2f"),
           "20d %": st.column_config.NumberColumn(format="%+.2f"),
           "30-day trend": st.column_config.LineChartColumn("30-day trend"),
           "Avg turnover (NPR m)": st.column_config.NumberColumn(format="%.1f")}
    event = ui.show_df(view, hide_index=True, on_select="rerun", selection_mode="single-row",
                       key="rank_table", column_config=cfg)
    try:
        rows = list(event.selection.rows)
    except Exception:
        rows = []
    if rows and rows[0] < len(view):
        sym = view.iloc[rows[0]]["Symbol"]
        r = d[d["symbol"] == sym].iloc[0]
        with st.container(border=True):
            h1, h2 = st.columns([3, 1])
            h1.markdown(f"### {sym}  &nbsp; {ui.pct_chip(r['ret_1'])} &nbsp; {ui.chip(r['sector'], 'accent')} "
                        + (ui.chip("bad news: " + r["news_note"], "warn") if r.get("news_flag", False) else ""),
                        unsafe_allow_html=True)
            if h2.button(f"Open {sym} page", icon=":material/open_in_new:"):
                ui.go_to_stock(sym)
            px = ui.get(ui.P.prices_clean, columns=["date", "symbol", "adj_close"])
            s = px[px["symbol"] == sym].sort_values("date").tail(180)
            fig = go.Figure(go.Scatter(x=s["date"], y=s["adj_close"], mode="lines", fill="tozeroy",
                                       line=dict(color=ui.ACCENT, width=2), fillcolor="rgba(15,118,110,.12)"))
            fig.update_yaxes(range=[s["adj_close"].min() * 0.97, s["adj_close"].max() * 1.02])
            st.plotly_chart(ui.style_fig(fig, 260))
    else:
        st.caption("Tip: click a row to see its chart and open its page.")

    with st.expander("Lowest scores, the stocks the model likes least"):
        low = d.sort_values("rank", ascending=False).head(15)
        ui.show_df(pd.DataFrame({"Rank": low["rank"], "Symbol": low["symbol"], "Sector": low["sector"],
                                   "Score": low["score_pct"], "5d %": low["ret_5"] * 100,
                                   "Volatility % (daily)": low["vol_20"] * 100}), hide_index=True, column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "5d %": st.column_config.NumberColumn(format="%+.2f"),
            "Volatility % (daily)": st.column_config.NumberColumn(format="%.2f")})
        oos = ui.get(ui.P.oos, columns=["date", "pred", "fwd_excess"])
        if oos is not None:
            dec = np.ceil(oos.groupby("date")["pred"].rank(pct=True) * 10).clip(1, 10)
            g = oos.groupby(dec)["fwd_excess"].mean() * 100
            st.caption(f"In past tests the lowest-scored tenth did {g.loc[1]:+.2f}% against the market average over "
                       f"{HORIZON} days and the highest-scored tenth {g.loc[10]:+.2f}%. Avoiding the bottom is where "
                       "the ranking has been most useful.")
    st.download_button("Download this table (CSV)", view.drop(columns=["30-day trend"]).to_csv(index=False).encode("utf-8"),
                       file_name=f"nepse_ranking_{asof:%Y-%m-%d}.csv", mime="text/csv", icon=":material/download:")


render()
