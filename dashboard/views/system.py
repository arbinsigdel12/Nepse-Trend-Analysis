import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from nepse_ranker.config import HORIZON  # noqa: E402
from nepse_ranker.status import run_status  # noqa: E402

KIND = {"OK": "up", "CHECK": "warn", "MISSING": "down"}


def render():
    ui.page_header("Data and guide", "Health of the data pipeline, details of what is stored, and how to read the results.")
    t1, t2, t3 = st.tabs(["Health", "Data details", "How to read this"])
    with t1:
        st.caption("The same checks as `python -m nepse_ranker status`. CHECK lines during a market holiday are normal.")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rows = run_status(ui.P)
        except Exception as e:
            st.error(f"Could not run the health check: {e}")
            rows = []
        need = sum(1 for r in rows if r[0] != "OK")
        st.markdown(f'<div class="card">{ui.chip("Everything is up to date", "up") if not need else ui.chip(f"{need} item(s) need attention", "warn")}</div>',
                    unsafe_allow_html=True)
        ui.gap(6)
        st.markdown('<div class="card tight">' + "".join(
            f'<div class="row"><span class="sym" style="min-width:150px">{name}</span>'
            f'<span class="muted" style="flex:1">{text}</span>{ui.chip(flag, KIND.get(flag, "flat"))}</div>'
            for flag, name, text in rows) + "</div>", unsafe_allow_html=True)
    with t2:
        val = ui.get_json(ui.P.validation)
        if val:
            ui.kpi_grid([
                ("Stocks in history", f"{val['symbols']}", "symbols with prices"),
                ("Trading days", f"{val['trading_days']:,}", "in the price history"),
                ("Last date", val["last_date"], "latest trading day stored"),
                ("Traded on last day", f"{val['stocks_traded_on_last_day']}", "stocks with a trade"),
            ], min_width=200)
            st.caption(f"First date {val['first_date']}. Corporate actions detected: {val['corporate_actions_found']}.")
            for w in val["warnings"]:
                st.warning(w)
            for i in val["issues"]:
                st.error(i)
        prices = ui.get(ui.P.prices_clean, columns=["date", "traded"])
        if prices is not None:
            per_day = prices.groupby("date")["traded"].sum().tail(90)
            ui.section("Stocks traded per day", "Last 90 trading days. A sudden drop can mean missing data.")
            fig = go.Figure(go.Bar(x=per_day.index, y=per_day.values, marker_color=ui.MUTED))
            st.plotly_chart(ui.style_fig(fig, 260, hover="closest"))
        hist = ui.get(ui.P.base / "predictions" / "history.parquet", columns=["date"])
        if hist is not None:
            st.markdown(ui.kpi("Saved daily rankings", f"{hist['date'].nunique()}",
                               "run python -m nepse_ranker live-check after a few weeks"), unsafe_allow_html=True)
    with t3:
        st.markdown(f"""
**What the score means.** Every trading day the model looks at each tradable stock's recent price moves, volatility,
trading activity and price-limit history, and puts the stocks in order. A score of 100 means the model expects that
stock to do best relative to the others over the next {HORIZON} trading days. It is an ordering, not a forecast of how
many rupees a stock will gain.

**What the model has found.** In past tests, calm and not-overheated stocks did slightly better than volatile stocks and
stocks that had just spiked or hit a daily price limit. The skill is real but small.

**Limits you should respect**
- Costs matter more than the signal. The edge of the top picks is smaller than the cost of trading every week for most people.
- Past tests are not the future. NEPSE changed its price limit and trading days in 2026.
- It uses prices, announcements and floorsheet data. Company earnings are only what you enter in the Watchlist.
- Stocks locked at the daily price limit may be impossible to buy at the quoted price.
- It ranks only stocks that trade often enough. Thin stocks are left out on purpose.

**Better uses than weekly trading:** treat low-scored stocks and stocks with fresh bad news as a list to be careful with,
and use the Long-term screen, Watchlist and Portfolio pages for decisions you plan to hold for years.

This is general information, not financial advice.
""")


render()
