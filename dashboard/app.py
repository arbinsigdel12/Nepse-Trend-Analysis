"""NEPSE Trend Analysis dashboard.  Start it with:  python -m nepse_ranker dashboard"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import ui  # noqa: E402

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="NEPSE Trend Analysis", page_icon=":material/monitoring:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

try:  # brand block above the navigation (needs a reasonably recent Streamlit)
    st.logo(str(HERE / "assets" / "logo.svg"), size="large")
except Exception:
    pass

# One flat list: every page is always visible in the sidebar, in the order you would use them.
# (Group headers made the Markets and Research sections collapse and hide their pages.)
nav = st.navigation([
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/ranking.py", title="Ranking", icon=":material/leaderboard:"),
    st.Page("views/stock.py", title="Stock", icon=":material/candlestick_chart:"),
    st.Page("views/longterm.py", title="Long-term screen", icon=":material/verified:"),
    st.Page("views/watchlist.py", title="Watchlist", icon=":material/visibility:"),
    st.Page("views/portfolio.py", title="Portfolio", icon=":material/pie_chart:"),
    st.Page("views/model_check.py", title="Model check", icon=":material/fact_check:"),
    st.Page("views/system.py", title="Data and guide", icon=":material/settings:"),
])


def _jump():
    sym = st.session_state.get("jump")
    if sym:
        st.session_state["current_symbol"] = sym
        st.session_state["jump"] = None
        st.switch_page("views/stock.py")


latest = ui.get(ui.P.latest, columns=["symbol"])
with st.sidebar:
    st.divider()
    if latest is not None:
        st.selectbox("Find a stock", sorted(latest["symbol"]), index=None, key="jump", on_change=_jump,
                     placeholder="Type a symbol ...")
    st.caption("NEPSE Trend Analysis. A research tool, not investment advice.")

nav.run()
