import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import ui  # noqa: E402

PERIODS = {"3M": 63, "6M": 126, "1Y": 240, "3Y": 720, "All": None}


def price_tab(sym):
    px = ui.get(ui.P.prices_clean, columns=["date", "symbol", "adj_open", "adj_high", "adj_low", "adj_close", "volume"])
    if px is None:
        ui.empty_state("Price history not found. Run:", "python -m nepse_ranker build")
        return
    period = st.radio("Period", list(PERIODS), index=1, horizontal=True, label_visibility="collapsed")
    s = px[px["symbol"] == sym].sort_values("date")
    if PERIODS[period]:
        s = s.tail(PERIODS[period])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=s["date"], open=s["adj_open"], high=s["adj_high"], low=s["adj_low"],
                                 close=s["adj_close"], name="Price", increasing_line_color=ui.UP,
                                 decreasing_line_color=ui.DOWN), row=1, col=1)
    full = px[px["symbol"] == sym].sort_values("date")
    for win, color in ((20, "#D97706"), (50, "#6366F1")):
        ma = full["adj_close"].rolling(win).mean().loc[s.index]
        fig.add_trace(go.Scatter(x=s["date"], y=ma, name=f"{win}-day average", line=dict(color=color, width=1.3)), row=1, col=1)
    up = (s["adj_close"] >= s["adj_open"]).to_numpy()
    fig.add_trace(go.Bar(x=s["date"], y=s["volume"], name="Shares traded",
                         marker_color=np.where(up, "rgba(22,163,74,.55)", "rgba(220,38,38,.55)")), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(ui.style_fig(fig, 540))
    st.caption("Prices are adjusted for bonus and right shares so the history is continuous.")


def model_tab(sym, r):
    if r.get("news_flag", False):
        st.warning(f"Fresh negative news in the last 5 trading days: {r['news_note']}. In past data, stocks in this "
                   "situation trailed the market by about 0.6% over the following 5 days.")
    hist = ui.get(ui.P.oos, columns=["date", "symbol", "pred"])
    if hist is not None:
        hist = hist.assign(pct=hist.groupby("date")["pred"].rank(pct=True) * 100)
        h = hist[hist["symbol"] == sym].sort_values("date")
        if len(h):
            ui.section("How the model has scored this stock", "0 is the lowest-ranked tenth, 100 the highest")
            fig = go.Figure(go.Scatter(x=h["date"], y=h["pct"], mode="lines", line=dict(color=ui.ACCENT, width=1.8)))
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(ui.style_fig(fig, 260))
    nf = ui.P.base / "news" / "news_table.parquet"
    if nf.exists():
        nt = ui.get(nf, columns=["published", "symbol", "title"])
        recent = nt[nt["symbol"] == sym].sort_values("published", ascending=False).head(15)
        ui.section("Recent official announcements")
        if len(recent):
            ui.show_df(recent.assign(published=recent["published"].dt.strftime("%Y-%m-%d %H:%M"))[["published", "title"]],
                       hide_index=True)
        else:
            st.caption("No announcements on record.")


def broker_tab(sym):
    ff_path = ui.P.base / "floorsheet_features.parquet"
    if not ff_path.exists():
        ui.empty_state("No floorsheet data yet. It is collected daily after the close:", "python -m nepse_ranker floorsheet-build")
        return
    ffd = ui.get(ff_path)
    sf = ffd[ffd["symbol"] == sym].sort_values("day")
    if sf.empty:
        st.caption("No floorsheet trades recorded for this stock yet.")
        return
    r = sf.iloc[-1]
    day_txt = f"{r['day']:%Y-%m-%d}"
    st.caption(f"Latest collected day: {day_txt}. History starts on {ffd['day'].min():%Y-%m-%d}, so this covers only "
               "the days collected so far.")
    n_tr, top3, same, block = int(r["trades"]), r["top3_buy_share"], r["self_trade_share"], r["block_trade_share"]
    ui.kpi_grid([
        ("Trades", f"{n_tr:,}", "on the latest day"),
        ("Top 3 buyers", f"{top3 * 100:.0f}%", "share of turnover"),
        ("Same broker both sides", f"{same * 100:.0f}%", "share of turnover"),
        ("Block trades", f"{block * 100:.0f}%", "share of turnover"),
    ], min_width=200)
    if n_tr >= 100 and same > 0.20:
        st.warning("A large share of this stocks turnover on the latest day was between the same broker on both sides. "
                   "That can be legitimate, but part of the volume may not reflect real interest. One day of data is "
                   "not evidence of anything on its own.")
    try:
        import duckdb
        glob = (ui.P.base / "floorsheet" / "*.parquet").as_posix()
        con = duckdb.connect()
        b1, b2 = st.columns(2, gap="large")
        for box, (side, col) in zip((b1, b2), [("Biggest buyers", "buyerBrokerName"), ("Biggest sellers", "sellerBrokerName")]):
            q = con.execute(
                f"SELECT {col} AS broker, sum(CAST(contractAmount AS DOUBLE)) / 1e5 AS lakh, count(*) AS trades "
                f"FROM read_parquet('{glob}') WHERE collected_for = '{day_txt}' AND stockSymbol = '{sym}' "
                f"GROUP BY 1 ORDER BY 2 DESC LIMIT 5").df()
            with box:
                ui.section(side, "turnover in NPR lakh")
                ui.show_df(q.round(1), hide_index=True)
    except Exception as e:
        st.caption(f"Broker table unavailable: {e}")


def income_tab(sym):
    dv = ui.get(ui.P.base / "company" / "dividends.parquet")
    if dv is not None and (dv["symbol"] == sym).any():
        d = dv[dv["symbol"] == sym].sort_values("fy_start")
        ui.section("Dividend history", "Percent of the Rs 100 face value. Blank means not stated in the announcement.")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["fy_start"].astype(str), y=d["cash_pct"], name="Cash dividend %", marker_color=ui.ACCENT))
        fig.add_trace(go.Bar(x=d["fy_start"].astype(str), y=d["bonus_pct"], name="Bonus shares %", marker_color="#6366F1"))
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="Fiscal year (start, Bikram Sambat)")
        st.plotly_chart(ui.style_fig(fig, 300))
    else:
        st.caption("No dividend declarations found for this stock.")
    scr = ui.get(ui.P.base / "longterm" / "screen.parquet")
    if scr is not None and (scr["symbol"] == sym).any():
        row = scr[scr["symbol"] == sym].iloc[0]
        ui.section("Long-term screen")
        st.markdown(ui.kpi("Checks passed", f"{int(row['checks_passed'])} of 6",
                           row["fails"] if row["fails"] else "no checks failed"), unsafe_allow_html=True)
    wl = ui.P.base / "watchlist" / "fundamentals.csv"
    if wl.exists():
        w = pd.read_csv(wl)
        w = w[w["symbol"] == sym]
        if len(w):
            w = w.iloc[-1]
            price = float(ui.core()["latest"].set_index("symbol").loc[sym, "close"])
            ui.section("Your watchlist numbers")
            eps, bv = pd.to_numeric(w.get("eps"), errors="coerce"), pd.to_numeric(w.get("bvps"), errors="coerce")
            ui.kpi_grid([
                ("PE", f"{price / eps:.1f}" if eps and eps > 0 else "n/a", f"EPS {eps}"),
                ("PB", f"{price / bv:.2f}" if bv and bv > 0 else "n/a", f"Book value {bv}"),
                ("ROE", f"{eps / bv * 100:.1f}%" if eps and bv and bv > 0 else "n/a", "EPS / book value"),
            ], min_width=200)


def render():
    c = ui.core()
    latest, sectors, names = c["latest"], c["sectors"], c["names"]
    if latest is None:
        ui.empty_state("No ranking found yet. Run:", "python -m nepse_ranker predict")
        return
    options = sorted(latest["symbol"])
    default = ui.current_symbol(options)
    sym = st.selectbox("Stock", options, index=options.index(default), key="stock_select")
    st.session_state["current_symbol"] = sym
    r = latest[latest["symbol"] == sym].iloc[0]

    px = ui.get(ui.P.prices_clean, columns=["date", "symbol", "adj_close"])
    hl = ""
    if px is not None:
        s52 = px[px["symbol"] == sym].sort_values("date").tail(240)["adj_close"]
        lo, hi = float(s52.min()), float(s52.max())
        pos = 0 if hi == lo else (float(s52.iloc[-1]) - lo) / (hi - lo) * 100
        hl = (f'<div class="kpi-label">52-week range</div><div class="scorebar"><div style="width:{pos:.0f}%"></div></div>'
              f'<div class="kpi-sub" style="padding-top:6px">{lo:,.0f} to {hi:,.0f}</div>')
    name = html.escape(names.get(sym) or "")
    st.markdown(
        f'<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:18px 32px">'
        f'<div style="min-width:180px;flex:1 1 220px"><div class="kpi-label">{html.escape(sectors.get(sym, "Other"))}</div>'
        f'<div style="font-size:2.1rem;font-weight:700;line-height:1.15;letter-spacing:-0.02em">{sym}</div>'
        f'<div class="kpi-sub" style="padding-top:0">{name}</div></div>'
        f'<div style="min-width:200px;flex:1 1 240px;max-width:360px">{hl}</div>'
        f'<div style="text-align:right;flex:0 0 auto"><div style="font-size:2.1rem;font-weight:700;line-height:1.15;letter-spacing:-0.02em">NPR {r["close"]:,.1f}</div>'
        f'<div style="margin-top:4px">{ui.pct_chip(r["ret_1"])} <span class="kpi-sub">today</span></div></div></div></div>',
        unsafe_allow_html=True)
    ui.gap(4)
    mv = lambda x: "up" if x > 0 else "down" if x < 0 else ""  # noqa: E731
    ui.kpi_grid([
        ("Model rank", f"{int(r['rank'])}", f"of {len(latest)} stocks"),
        ("Score", f"{r['score_pct']:.0f}", "0 to 100"),
        ("5-day move", f"{r['ret_5'] * 100:+.1f}%", "price change", mv(r["ret_5"])),
        ("20-day move", f"{r['ret_20'] * 100:+.1f}%", "price change", mv(r["ret_20"])),
        ("Daily volatility", f"{r['vol_20'] * 100:.1f}%", "20-day average"),
    ])
    ui.gap(4)
    t1, t2, t3, t4 = st.tabs(["Price", "Model and news", "Broker activity", "Income and value"])
    with t1:
        price_tab(sym)
    with t2:
        model_tab(sym, r)
    with t3:
        broker_tab(sym)
    with t4:
        income_tab(sym)


render()
