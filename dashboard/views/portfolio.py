import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402


def render():
    c = ui.core()
    latest = c["latest"]
    ui.page_header("Portfolio planner", "Enter what you hold, or plan to hold. The planner checks concentration and replays "
                   "how this mix would have behaved in past price history.")
    scr = ui.get(ui.P.base / "longterm" / "screen.parquet")
    if scr is not None:
        options, sector_of = sorted(scr["symbol"]), dict(zip(scr["symbol"], scr["sector"]))
    elif latest is not None:
        options, sector_of = sorted(latest["symbol"]), c["sectors"]
    else:
        ui.empty_state("No data yet. Run:", "python -m nepse_ranker predict")
        return
    starter = pd.DataFrame({"symbol": options[:2], "amount_npr": [100000.0, 100000.0]})
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        ui.section("Your holdings", "Add rows with the + at the bottom of the table.")
        held = ui.show_editor(starter, num_rows="dynamic", key="holdings", column_config={
            "symbol": st.column_config.SelectboxColumn("Stock", options=options, required=True),
            "amount_npr": st.column_config.NumberColumn("Amount invested (NPR)", min_value=0.0, step=1000.0)})
        l1, l2 = st.columns(2)
        max_stock = l1.slider("Maximum per stock (%)", 5, 50, 10)
        max_sector = l2.slider("Maximum per sector (%)", 10, 100, 30)
    held = held.dropna(subset=["symbol"])
    held = held[held["amount_npr"].fillna(0) > 0]
    held = held.groupby("symbol", as_index=False)["amount_npr"].sum()
    if held.empty:
        with right:
            st.info("Add at least one holding in the table.")
        return
    held["weight"] = held["amount_npr"] / held["amount_npr"].sum() * 100
    held["sector"] = held["symbol"].map(sector_of).fillna("Unknown")
    by_sector = held.groupby("sector")["weight"].sum().sort_values(ascending=False)
    with right:
        ui.section("Where the money is", "Share of the portfolio by sector")
        fig = go.Figure(go.Pie(labels=list(by_sector.index), values=list(by_sector.values), hole=0.62,
                               textinfo="label+percent", sort=False))
        fig.update_layout(showlegend=False)
        st.plotly_chart(ui.style_fig(fig, 300, hover="closest"))
    for r in held[held["weight"] > max_stock].itertuples():
        st.warning(f"{r.symbol} is {r.weight:.0f}% of the portfolio, above your {max_stock}% limit.")
    for sec, w in by_sector.items():
        if w > max_sector:
            st.warning(f"{sec} is {w:.0f}% of the portfolio, above your {max_sector}% limit.")

    prices = ui.get(ui.P.prices_clean, columns=["date", "symbol", "adj_close"])
    px = prices[prices["symbol"].isin(held["symbol"])].pivot(index="date", columns="symbol", values="adj_close").sort_index().dropna()
    if len(px) < 120:
        st.info("Not enough shared price history for these stocks to replay the past.")
        return
    wts = (held.set_index("symbol")["weight"] / 100).reindex(px.columns)
    daily = px.pct_change().dropna() @ wts
    curve = (1 + daily).cumprod()
    fall = curve / curve.cummax() - 1
    ui.kpi_grid([
        ("History used", f"{len(px) / 240:.1f} yrs", "shared price history"),
        ("Worst fall from a peak", f"{fall.min() * 100:.0f}%", "deepest drawdown", "down"),
        ("Worst 20 trading days", f"{curve.pct_change(20).min() * 100:.0f}%", "about one month", "down"),
        ("Yearly volatility", f"{daily.std() * (240 ** 0.5) * 100:.0f}%", "std. dev. of returns"),
    ], min_width=200)
    ui.gap(6)
    ui.section("How this mix would have behaved", "Fall from the previous peak")
    ff = go.Figure(go.Scatter(x=fall.index, y=fall * 100, fill="tozeroy", line=dict(color=ui.DOWN),
                              fillcolor="rgba(220,38,38,.15)"))
    ff.update_yaxes(title="%")
    st.plotly_chart(ui.style_fig(ff, 300))
    st.caption("Uses past price changes only (bonus-adjusted, no cash dividends), with these weights held constant. The "
               "past is not a promise: the next fall can be deeper than any shown here.")


render()
