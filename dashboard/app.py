"""NEPSE Ranker dashboard.  Start it with:  python -m nepse_ranker dashboard"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nepse_ranker.config import HORIZON, Paths  # noqa: E402

P = Paths()
POS, NEG, MUTED = "#0F766E", "#B4443C", "#7A8699"
RULE_NAMES = {"model": "Model", "momentum_5d": "Rule: last week's winners",
              "momentum_20d": "Rule: last month's winners", "volume_surge": "Rule: volume spike",
              "low_volatility": "Rule: calmest stocks", "random": "Random picks"}

st.set_page_config(page_title="NEPSE Ranker", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, mtime: float, columns):
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def get(path: Path, columns=None):
    if not path.exists():
        return None
    return _read_parquet(str(path), path.stat().st_mtime, tuple(columns) if columns else None)


def get_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def style_fig(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08, x=0))
    return fig


latest = get(P.latest)
metrics = get_json(P.metrics)

st.title("NEPSE Ranker")
if latest is None:
    st.error("No ranking found yet. In PowerShell, run:  python -m nepse_ranker run-all")
    st.stop()

asof = pd.Timestamp(latest["date"].iloc[0])
st.caption(f"Data through {asof:%A, %d %B %Y}  ·  {len(latest)} tradable stocks ranked  ·  "
           "a research tool, not investment advice")
if (pd.Timestamp.today().normalize() - asof).days > 5:
    st.warning("The data is more than 5 days old. NEPSE may have been closed for a holiday, "
               "otherwise run:  python -m nepse_ranker update  then  run-all")

if metrics:
    v = metrics["verdict"]
    box = {"promising": st.success, "mixed": st.info, "weak": st.warning, "none": st.error}[v["level"]]
    box(f"**Model check: {v['level'].upper()}.** {v['text']}")

t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Today's ranking", "Stock", "Model check", "Data health", "How to read this", "Long-term", "Portfolio"])

# ---------------------------------------------------------------- ranking
with t1:
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Stocks to show", 5, 50, 20)
    min_turn = c2.number_input("Minimum average daily turnover (NPR million)", 0.0, 500.0, 0.0, step=1.0)
    query = c3.text_input("Find a symbol").strip().upper()

    def build_table(d):
        return pd.DataFrame({
            "Rank": d["rank"], "Symbol": d["symbol"], "Score": d["score_pct"],
            "Close (NPR)": d["close"], "1 day %": d["ret_1"] * 100, "5 day %": d["ret_5"] * 100,
            "20 day %": d["ret_20"] * 100, "Avg daily turnover (NPR m)": d["turn_20"] / 1e6,
            "Volatility % (daily)": d["vol_20"] * 100,
            "Fresh negative news": d["news_note"] if "news_note" in d else ""})

    cfg = {"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
           "Close (NPR)": st.column_config.NumberColumn(format="%.1f"),
           "1 day %": st.column_config.NumberColumn(format="%.2f"),
           "5 day %": st.column_config.NumberColumn(format="%.2f"),
           "20 day %": st.column_config.NumberColumn(format="%.2f"),
           "Avg daily turnover (NPR m)": st.column_config.NumberColumn(format="%.1f"),
           "Volatility % (daily)": st.column_config.NumberColumn(format="%.2f")}

    d = latest.copy()
    d = d[d["turn_20"] / 1e6 >= min_turn]
    if query:
        d = d[d["symbol"].str.contains(query, regex=False)]
    st.subheader("Highest scores")
    st.dataframe(build_table(d.sort_values("rank").head(n)), hide_index=True, column_config=cfg)
    if "news_flag" in d.columns:
        k = int(d.sort_values("rank").head(n)["news_flag"].sum())
        if k:
            st.caption(f"{k} of these {n} stocks carry fresh negative news (see the last column).")
    st.caption("Score 100 = the model's favourite among today's tradable stocks. It is a ranking, "
               "not a price forecast. See the 'Model check' tab for how reliable the ranking has been.")

    with st.expander("Lowest scores (stocks the model likes least)"):
        st.dataframe(build_table(d.sort_values("rank", ascending=False).head(n)), hide_index=True,
                     column_config=cfg)
        oos = get(P.oos, columns=["date", "pred", "fwd_excess"])
        if oos is not None:
            dec = np.ceil(oos.groupby("date")["pred"].rank(pct=True) * 10).clip(1, 10)
            g = oos.groupby(dec)["fwd_excess"].mean() * 100
            st.caption(f"In past tests, the lowest-scored tenth of stocks did {g.loc[1]:+.2f}% versus the "
                       f"market average over {HORIZON} days, and the highest-scored tenth {g.loc[10]:+.2f}%. "
                       "Avoiding the bottom is where the ranking has been most useful.")
    st.download_button("Download full ranking (CSV)", latest.to_csv(index=False).encode("utf-8"),
                       file_name=f"nepse_ranking_{asof:%Y-%m-%d}.csv", mime="text/csv")

# ---------------------------------------------------------------- stock
with t2:
    sym = st.selectbox("Stock", sorted(latest["symbol"]))
    row = latest[latest["symbol"] == sym].iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rank today", f"{int(row['rank'])} of {len(latest)}")
    m2.metric("Score", f"{row['score_pct']:.0f}")
    m3.metric("5-day move", f"{row['ret_5'] * 100:+.1f}%")
    m4.metric("20-day move", f"{row['ret_20'] * 100:+.1f}%")
    m5.metric("Daily volatility", f"{row['vol_20'] * 100:.1f}%")
    note = row["news_note"] if "news_note" in row.index else ""
    if note:
        st.warning(f"Fresh negative news in the last 5 trading days: {note}. In past data, stocks in this "
                   "situation trailed the market by about 0.6% over the following 5 days.")
    news_file = P.base / "news" / "news_table.parquet"
    if news_file.exists():
        nt = get(news_file, columns=["published", "symbol", "title"])
        recent = nt[nt["symbol"] == sym].sort_values("published", ascending=False).head(12)
        with st.expander(f"Recent official announcements for {sym}", expanded=bool(note)):
            if len(recent):
                shown = recent.assign(published=recent["published"].dt.strftime("%Y-%m-%d %H:%M"))
                st.dataframe(shown[["published", "title"]], hide_index=True)
            else:
                st.write("No announcements on record.")

    prices = get(P.prices_clean, columns=["date", "symbol", "adj_open", "adj_high", "adj_low",
                                          "adj_close", "volume"])
    if prices is None:
        st.info("Price history not found. Run: python -m nepse_ranker build")
    else:
        s = prices[prices["symbol"] == sym].sort_values("date").tail(250)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                            vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=s["date"], open=s["adj_open"], high=s["adj_high"],
                                     low=s["adj_low"], close=s["adj_close"], name="Price",
                                     increasing_line_color=POS, decreasing_line_color=NEG), row=1, col=1)
        fig.add_trace(go.Scatter(x=s["date"], y=s["adj_close"].rolling(20).mean(), name="20-day average",
                                 line=dict(color="#C58B00", width=1.4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=s["date"], y=s["adj_close"].rolling(50).mean(), name="50-day average",
                                 line=dict(color="#5B6CFF", width=1.4)), row=1, col=1)
        fig.add_trace(go.Bar(x=s["date"], y=s["volume"], name="Shares traded",
                             marker_color=MUTED), row=2, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(style_fig(fig, 520))
        st.caption("Prices are adjusted for bonus and right shares so the history is continuous. "
                   "Last 250 trading days.")

    hist = get(P.oos, columns=["date", "symbol", "pred"])
    if hist is not None:
        hist = hist.assign(pct=hist.groupby("date")["pred"].rank(pct=True) * 100)
        h = hist[hist["symbol"] == sym].sort_values("date")
        if len(h):
            f2 = go.Figure(go.Scatter(x=h["date"], y=h["pct"], mode="lines", line=dict(color=POS, width=1.6)))
            f2.update_yaxes(title="Model score (0 to 100)", range=[0, 100])
            st.subheader("How the model has scored this stock over time")
            st.plotly_chart(style_fig(f2, 260))

# ---------------------------------------------------------------- model check
with t3:
    if not metrics:
        st.info("No test results yet. Run: python -m nepse_ranker train")
    else:
        st.write(f"Honest out-of-sample test: **{metrics['test_start']}** to **{metrics['test_end']}**. "
                 f"The model always predicted blocks of days it had never seen. Top {metrics['k']} picks, "
                 f"{metrics['horizon']}-day holding, assumed round-trip cost "
                 f"**{metrics['round_trip_cost'] * 100:.1f}%** (change it in config.py).")
        s = pd.DataFrame(metrics["summary"]).T
        show = pd.DataFrame({
            "Rank skill (IC)": s["mean_ic"], "IC t-stat": s["ic_t_stat"],
            "Top picks, % per period (before cost)": s["topk_gross_excess"] * 100,
            "Top picks, % per period (after cost)": s["topk_net_excess"] * 100,
            "Top picks beat the market": s["topk_hit_rate"], "Chance level": s["base_hit_rate"]})
        show.index = [RULE_NAMES.get(i, i) for i in show.index]
        st.dataframe(show.round(3))
        st.caption("IC is how well the predicted order matches the real order: 0 is no skill, "
                   "above about 0.05 steadily is useful. 'After cost' is what matters for trading.")

        daily = get(P.daily_eval)
        if daily is not None:
            daily = daily.sort_values("date")
            f3 = go.Figure(go.Scatter(x=daily["date"], y=daily["ic"].rolling(60, min_periods=20).mean(),
                                      line=dict(color=POS, width=1.8), name="IC, 60-day average"))
            f3.add_hline(y=0, line_color=MUTED, line_dash="dot")
            f3.update_yaxes(title="Rank skill (IC)")
            st.subheader("Is the skill steady over time?")
            st.plotly_chart(style_fig(f3, 300))

            ind = daily.dropna(subset=["topk_excess"]).iloc[::HORIZON]
            cost = metrics["round_trip_cost"]
            gross = (1 + ind["topk_excess"]).cumprod() - 1
            net = (1 + ind["topk_excess"] - cost).cumprod() - 1
            f4 = go.Figure()
            f4.add_trace(go.Scatter(x=ind["date"], y=gross * 100, name="Before cost", line=dict(color=POS)))
            f4.add_trace(go.Scatter(x=ind["date"], y=net * 100, name="After cost", line=dict(color=NEG)))
            f4.add_hline(y=0, line_color=MUTED, line_dash="dot")
            f4.update_yaxes(title="Cumulative % versus the market average")
            st.subheader("What the top picks earned beyond the market, trading every period")
            st.plotly_chart(style_fig(f4, 320))

        oos = get(P.oos, columns=["date", "pred", "fwd_excess"])
        if oos is not None:
            dec = np.ceil(oos.groupby("date")["pred"].rank(pct=True) * 10).clip(1, 10).astype(int)
            g = (oos.groupby(dec)["fwd_excess"].mean() * 100)
            f5 = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g.values,
                                  marker_color=[POS if x >= 0 else NEG for x in g.values]))
            f5.update_xaxes(title="Score group (10 = highest scored tenth of stocks)")
            f5.update_yaxes(title=f"Average {HORIZON}-day result vs market, %")
            st.subheader("Do higher scores really do better?")
            st.plotly_chart(style_fig(f5, 320))

        yearly = P.base / "models" / "stability_by_year.csv"
        if yearly.exists():
            st.subheader("Year by year")
            st.dataframe(pd.read_csv(yearly).set_index("year").round(3))

# ---------------------------------------------------------------- data health
with t4:
    val = get_json(P.validation)
    if val:
        a, b, c, d_ = st.columns(4)
        a.metric("Stocks in history", val["symbols"])
        b.metric("Trading days", val["trading_days"])
        c.metric("Last date", val["last_date"])
        d_.metric("Stocks traded on last day", val["stocks_traded_on_last_day"])
        st.write(f"First date {val['first_date']} · corporate actions detected: {val['corporate_actions_found']}")
        for w in val["warnings"]:
            st.warning(w)
        for i in val["issues"]:
            st.error(i)
        if not val["warnings"] and not val["issues"]:
            st.success("No data problems found.")
    prices = get(P.prices_clean, columns=["date", "traded"])
    if prices is not None:
        per_day = prices.groupby("date")["traded"].sum().tail(60)
        f6 = go.Figure(go.Bar(x=per_day.index, y=per_day.values, marker_color=MUTED))
        f6.update_yaxes(title="Stocks traded")
        st.subheader("Stocks traded per day, last 60 trading days")
        st.plotly_chart(style_fig(f6, 260))
        st.caption("NEPSE trades Monday to Friday (since April 2026) except public holidays. "
                   "A sudden drop here can mean missing data.")

# ---------------------------------------------------------------- guide
with t5:
    st.markdown(f"""
**What the score means.** Every trading day, the model looks at each tradable stock's recent price
moves, volatility, trading activity and price-limit history, and puts the stocks in order. A score of
100 means the model expects that stock to do best relative to the others over the next {HORIZON} trading
days. It is an *ordering*, not a forecast of how many rupees a stock will gain.

**What the model has found.** In past tests, calm and not-overheated stocks did slightly better than
volatile stocks and stocks that had just spiked or hit a daily price limit. The skill is real but small.

**Limits you should respect**
- Costs matter more than the signal. The average edge of the top picks is smaller than the cost of
  trading every week for most people. Check the 'after cost' numbers in the Model check tab.
- Past tests are not the future. Markets change; NEPSE changed its price limit and trading days in 2026.
- It only uses prices. Company results, dividends, mergers and news are not in it yet.
- Stocks locked at the daily price limit may be impossible to buy at the quoted price.
- It ranks only stocks that trade often enough. Thin stocks are left out on purpose.

**Better uses than weekly trading:** treat low-scored stocks as a list to avoid, and use the ranking as
one input next to company fundamentals if you invest for the long term.

This is general information, not financial advice.
""")



# ---------------------------------------------------------------- long-term screen
SCREEN = P.base / "longterm" / "screen.parquet"
with t6:
    scr = get(SCREEN)
    if scr is None:
        st.info("No long-term screen yet. In PowerShell run:  python -m nepse_ranker longterm-build")
    else:
        st.markdown("**A risk and history screen, not a valuation.** It checks how long a stock has traded, "
                    "how easily it trades, how deep its past falls were, how volatile it is, whether it keeps "
                    "issuing right shares, and whether it has recent red-flag news. Earnings, book value and "
                    "cash dividends are only partly known (read from announcement text; blank means unknown), so it "
                    "cannot say whether a stock is cheap. Passing is not a reason to buy.")
        c1, c2, c3 = st.columns(3)
        sector_list = sorted(scr["sector"].dropna().unique())
        pick = c1.multiselect("Sectors", sector_list, default=sector_list)
        min_checks = c2.slider("Checks passed (out of 6)", 0, 6, 5)
        n_show = c3.slider("Stocks to show", 10, 150, 40)
        sort_by = st.selectbox("Sort by", ["Lowest volatility", "Smallest past fall", "Most traded",
                                           "Highest 5-year return (past performance only)"])
        order = {"Lowest volatility": ("vol_ann", True), "Smallest past fall": ("max_dd_5y", False),
                 "Most traded": ("median_turnover_120d", False),
                 "Highest 5-year return (past performance only)": ("cagr_5y", False)}[sort_by]
        d = scr[scr["sector"].isin(pick) & (scr["checks_passed"] >= min_checks)]
        d = d.sort_values(order[0], ascending=order[1], na_position="last").head(n_show)
        table = pd.DataFrame({
            "Symbol": d["symbol"], "Sector": d["sector"], "Checks passed": d["checks_passed"],
            "Years listed": d["listed_years"], "Return per year, 5y %": d["cagr_5y"] * 100,
            "Volatility % (yearly)": d["vol_ann"] * 100, "Worst fall, 5y %": d["max_dd_5y"] * 100,
            "Median turnover (NPR m)": d["median_turnover_120d"] / 1e6,
            "Bonus, last 5 FY (%)": d["bonus_5fy_total_pct"], "Rights issues, 5 FY": d["rights_issues_5fy"],
            "Dividend years, last 5 FY": d.get("dividend_years_5fy"),
            "Avg cash dividend % (3 FY)": d.get("cash_avg_3fy_pct"),
            "Cash yield %": d["cash_yield"] * 100 if "cash_yield" in d.columns else None,
            "Dividend years, last 5 FY": d.get("dividend_years_5fy"),
            "Avg cash dividend % (3 FY)": d.get("cash_avg_3fy_pct"),
            "Cash yield %": d["cash_yield"] * 100 if "cash_yield" in d.columns else None,
            "Red flags, 1y": d["red_flags_1y"], "Failed checks": d["fails"]})
        st.dataframe(table, hide_index=True, column_config={
            "Years listed": st.column_config.NumberColumn(format="%.1f"),
            "Return per year, 5y %": st.column_config.NumberColumn(format="%.1f"),
            "Volatility % (yearly)": st.column_config.NumberColumn(format="%.1f"),
            "Worst fall, 5y %": st.column_config.NumberColumn(format="%.1f"),
            "Median turnover (NPR m)": st.column_config.NumberColumn(format="%.1f"),
            "Bonus, last 5 FY (%)": st.column_config.NumberColumn(format="%.1f"),
            "Avg cash dividend % (3 FY)": st.column_config.NumberColumn(format="%.1f"),
            "Cash yield %": st.column_config.NumberColumn(format="%.1f")})
        st.caption(f"{len(scr)} ordinary shares screened. Returns include bonus-share adjustment but not cash "
                   "dividends. The turnover check uses NPR 1 million a day; change it in longterm.py.")
        with st.expander("What the six checks are"):
            st.markdown("""
1. **History:** listed at least 5 years, with at least 3 years of price data here.
2. **Trading:** median daily turnover over the last 120 days of at least NPR 1 million.
3. **Past falls:** worst peak-to-trough fall over up to 5 years better than -60%.
4. **Volatility:** yearly price swings below 50%.
5. **Dilution:** at most one right-share issue in the last 5 fiscal years.
6. **News:** at most one red-flag item (lock-in expiry, insider sale notice, project disruption, regulatory action) in the last year.
""")

# ---------------------------------------------------------------- portfolio planner
with t7:
    st.markdown("Enter what you hold, or plan to hold. The planner checks concentration and replays how "
                "this mix would have behaved in past price history.")
    scr_all = get(SCREEN)
    options = sorted(scr_all["symbol"]) if scr_all is not None else sorted(latest["symbol"])
    sector_of = dict(zip(scr_all["symbol"], scr_all["sector"])) if scr_all is not None else {}
    starter = pd.DataFrame({"symbol": options[:2], "amount_npr": [100000.0, 100000.0]})
    held = st.data_editor(starter, num_rows="dynamic", key="holdings", column_config={
        "symbol": st.column_config.SelectboxColumn("Stock", options=options, required=True),
        "amount_npr": st.column_config.NumberColumn("Amount invested (NPR)", min_value=0.0, step=1000.0)})
    l1, l2 = st.columns(2)
    max_stock = l1.slider("Maximum per stock (%)", 5, 50, 10)
    max_sector = l2.slider("Maximum per sector (%)", 10, 100, 30)
    held = held.dropna(subset=["symbol"])
    held = held[held["amount_npr"].fillna(0) > 0]
    held = held.groupby("symbol", as_index=False)["amount_npr"].sum()
    if held.empty:
        st.info("Add at least one holding in the table above.")
    else:
        held["weight"] = held["amount_npr"] / held["amount_npr"].sum() * 100
        held["sector"] = held["symbol"].map(sector_of).fillna("Unknown")
        st.dataframe(held.rename(columns={"symbol": "Stock", "amount_npr": "Amount (NPR)",
                                          "weight": "Weight %", "sector": "Sector"}).round(1), hide_index=True)
        by_sector = held.groupby("sector")["weight"].sum().sort_values(ascending=False)
        for r in held[held["weight"] > max_stock].itertuples():
            st.warning(f"{r.symbol} is {r.weight:.0f}% of the portfolio, above your {max_stock}% limit.")
        for sec, w in by_sector.items():
            if w > max_sector:
                st.warning(f"{sec} is {w:.0f}% of the portfolio, above your {max_sector}% limit.")
        fs = go.Figure(go.Bar(x=by_sector.values, y=by_sector.index, orientation="h", marker_color=POS))
        fs.update_xaxes(title="Share of portfolio, %")
        st.plotly_chart(style_fig(fs, 60 + 40 * len(by_sector)))

        prices = get(P.prices_clean, columns=["date", "symbol", "adj_close"])
        px = prices[prices["symbol"].isin(held["symbol"])].pivot(
            index="date", columns="symbol", values="adj_close").sort_index().dropna()
        if len(px) < 120:
            st.info("Not enough shared price history for these stocks to replay the past.")
        else:
            wts = (held.set_index("symbol")["weight"] / 100).reindex(px.columns)
            daily = px.pct_change().dropna() @ wts
            curve = (1 + daily).cumprod()
            fall = curve / curve.cummax() - 1
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Years of history used", f"{len(px) / 240:.1f}")
            m2.metric("Worst fall from a peak", f"{fall.min() * 100:.0f}%")
            m3.metric("Worst 20 trading days", f"{(curve.pct_change(20).min()) * 100:.0f}%")
            m4.metric("Yearly volatility", f"{daily.std() * (240 ** 0.5) * 100:.0f}%")
            ff = go.Figure(go.Scatter(x=fall.index, y=fall * 100, fill="tozeroy", line=dict(color=NEG)))
            ff.update_yaxes(title="Fall from the previous peak, %")
            st.subheader("How this mix would have behaved")
            st.plotly_chart(style_fig(ff, 300))
            st.caption("Uses past price changes only (bonus-adjusted, no cash dividends), with these weights held "
                       "constant. The past is not a promise: the next fall can be deeper than any shown here.")
