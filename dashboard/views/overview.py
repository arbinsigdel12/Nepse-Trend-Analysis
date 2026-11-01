import html
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402


def movers(df, col, n=6):
    rows = "".join(
        f'<div class="row"><span class="sym">{r.symbol}</span><span class="muted">{r.close:,.1f}</span>'
        f"{ui.pct_chip(getattr(r, col))}</div>" for r in df.head(n).itertuples())
    return f'<div class="card tight">{rows}</div>'


def render():
    c = ui.core()
    latest, metrics, sectors, names = c["latest"], c["metrics"], c["sectors"], c["names"]
    if latest is None:
        ui.empty_state("No ranking found yet. Run this in PowerShell, then refresh:", "python -m nepse_ranker predict")
        return
    asof = pd.Timestamp(latest["date"].iloc[0])
    age = (pd.Timestamp.today().normalize() - asof).days
    chips = [ui.chip(f"{len(latest)} tradable stocks")]
    chips.append(ui.chip(f"Data is {age} days old (holiday?)", "warn") if age > 5 else ui.chip("Data up to date"))
    if metrics:
        level = metrics["verdict"]["level"]
        chips.append(ui.chip(f"Model check: {level}", "warn" if level in ("weak", "none") else ""))
    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    st.markdown(f'<div class="hero"><h1>{greet}. NEPSE at a glance</h1>'
                f'<p>Last trading day: {asof:%A, %d %B %Y}</p>{"".join(chips)}</div>', unsafe_allow_html=True)

    day = ui.get(ui.P.prices_clean, columns=["date", "symbol", "turnover"])
    turnover = float(day[(day["date"] == asof) & day["symbol"].isin(latest["symbol"])]["turnover"].sum()) if day is not None else float("nan")
    adv, dec = int((latest["ret_1"] > 0).sum()), int((latest["ret_1"] < 0).sum())
    move = float(latest["ret_1"].mean())
    flagged = int(latest["news_flag"].sum()) if "news_flag" in latest else 0
    ic = metrics["summary"]["model"]["mean_ic"] if metrics else None

    ui.kpi_grid([
        ("Average stock today", f"{move * 100:+.2f}%", "equal-weighted, tradable stocks", "up" if move > 0 else "down" if move < 0 else ""),
        ("Advancing / declining", f"{adv} / {dec}", f"{len(latest) - adv - dec} unchanged"),
        ("Turnover", f"{turnover / 1e7:,.0f} cr", "NPR crore, tradable stocks"),
        ("Fresh bad-news flags", f"{flagged}", "in the last 5 trading days"),
        ("Ranking skill", f"{ic:+.2f}" if ic is not None else "n/a", "backtest rank correlation; small but steady"),
    ])

    ui.gap(6)
    left, right = st.columns([1.75, 1], gap="large")
    with left:
        ui.section("Market map", "Box size is trading activity, colour is today's move. Click a sector to zoom in.")
        st.plotly_chart(ui.market_treemap(latest, sectors, height=590))
    with right:
        ui.section("Biggest gainers", "Today, tradable stocks")
        st.markdown(movers(latest.sort_values("ret_1", ascending=False), "ret_1"), unsafe_allow_html=True)
        ui.gap(6)
        ui.section("Biggest decliners", "Today, tradable stocks")
        st.markdown(movers(latest.sort_values("ret_1"), "ret_1"), unsafe_allow_html=True)

    ui.gap(6)
    ui.section("Top of the model ranking", "A screening aid, not a trading signal. See Model check for how reliable it has been.")
    cards = []
    for r in latest.sort_values("rank").head(5).itertuples():
        warn = ui.chip("bad news", "warn") if getattr(r, "news_flag", False) else ""
        cards.append(
            f'<div class="card pick"><div class="rank">RANK {int(r.rank)}</div><div class="sym">{r.symbol}</div>'
            f'<div class="name">{html.escape(names.get(r.symbol) or "") or "&nbsp;"}</div>'
            f'<div class="foot">{ui.pct_chip(r.ret_1)} {warn}'
            f'<div class="scorebar"><div style="width:{r.score_pct:.0f}%"></div></div></div></div>')
    ui.card_grid(cards)
    ui.link("views/ranking.py", "Open the full ranking")

    if flagged:
        ui.gap(6)
        ui.section("Stocks with fresh bad news", "Right-share, ex-date, lock-in, insider-sale or project-disruption news")
        bad = latest[latest["news_flag"]].sort_values("rank")
        st.markdown('<div class="card">' + " ".join(
            ui.chip(f"{r.symbol}: {r.news_note}", "warn") for r in bad.itertuples()) + "</div>", unsafe_allow_html=True)


render()
