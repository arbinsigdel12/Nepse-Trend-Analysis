import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from nepse_ranker.config import HORIZON  # noqa: E402

RULE_NAMES = {"model": "Model", "momentum_5d": "Rule: last week's winners", "momentum_20d": "Rule: last month's winners",
              "volume_surge": "Rule: volume spike", "low_volatility": "Rule: calmest stocks", "random": "Random picks"}
VERDICT_KIND = {"promising": "up", "mixed": "accent", "weak": "warn", "none": "down"}


def render():
    metrics = ui.core()["metrics"]
    ui.page_header("Model check", "How reliable the ranking has been when tested only on days it had never seen.")
    if not metrics:
        ui.empty_state("No test results yet. Run this in PowerShell, then refresh:", "python -m nepse_ranker train")
        return
    v = metrics["verdict"]
    st.markdown(f'<div class="card"><span class="kpi-label">Verdict</span><br>'
                f'{ui.chip(v["level"].upper(), VERDICT_KIND.get(v["level"], "flat"))} &nbsp; {v["text"]}</div>',
                unsafe_allow_html=True)
    ui.gap(6)
    s = pd.DataFrame(metrics["summary"]).T
    m = s.loc["model"]
    tone = lambda x: "up" if x > 0 else "down" if x < 0 else ""  # noqa: E731
    ui.kpi_grid([
        ("Rank skill (IC)", f"{m['mean_ic']:+.3f}", "0 is no skill; steady above 0.05 is useful"),
        ("Top picks, before costs", f"{m['topk_gross_excess'] * 100:+.2f}%", f"per {HORIZON}-day period vs the market", tone(m['topk_gross_excess'])),
        ("After assumed costs", f"{m['topk_net_excess'] * 100:+.2f}%", f"cost assumed {metrics['round_trip_cost'] * 100:.1f}% per trade", tone(m['topk_net_excess'])),
        ("Picks beating the market", f"{m['topk_hit_rate'] * 100:.0f}%", f"chance level {m['base_hit_rate'] * 100:.0f}%"),
    ], min_width=200)
    st.caption(f"Test period {metrics['test_start']} to {metrics['test_end']}. Top {metrics['k']} picks, {metrics['horizon']}-day "
               "holding. Change the assumed cost in nepse_ranker/config.py.")
    show = pd.DataFrame({"Rank skill (IC)": s["mean_ic"], "IC t-stat": s["ic_t_stat"],
                         "Top picks % (before cost)": s["topk_gross_excess"] * 100,
                         "Top picks % (after cost)": s["topk_net_excess"] * 100,
                         "Beat the market": s["topk_hit_rate"], "Chance level": s["base_hit_rate"]})
    show.index = [RULE_NAMES.get(i, i) for i in show.index]
    ui.section("Model against simple rules")
    ui.show_df(show.round(3))

    daily = ui.get(ui.P.daily_eval)
    c1, c2 = st.columns(2, gap="large")
    if daily is not None:
        daily = daily.sort_values("date")
        with c1:
            ui.section("Is the skill steady?", "60-day average of rank skill")
            f3 = go.Figure(go.Scatter(x=daily["date"], y=daily["ic"].rolling(60, min_periods=20).mean(),
                                      line=dict(color=ui.ACCENT, width=2)))
            f3.add_hline(y=0, line_color=ui.MUTED, line_dash="dot")
            st.plotly_chart(ui.style_fig(f3, 300))
        with c2:
            ind = daily.dropna(subset=["topk_excess"]).iloc[::HORIZON]
            cost = metrics["round_trip_cost"]
            ui.section("What the top picks earned", "Cumulative, versus the market average")
            f4 = go.Figure()
            f4.add_trace(go.Scatter(x=ind["date"], y=((1 + ind["topk_excess"]).cumprod() - 1) * 100, name="Before cost",
                                    line=dict(color=ui.ACCENT)))
            f4.add_trace(go.Scatter(x=ind["date"], y=((1 + ind["topk_excess"] - cost).cumprod() - 1) * 100, name="After cost",
                                    line=dict(color=ui.DOWN)))
            f4.add_hline(y=0, line_color=ui.MUTED, line_dash="dot")
            st.plotly_chart(ui.style_fig(f4, 300))
    oos = ui.get(ui.P.oos, columns=["date", "pred", "fwd_excess"])
    if oos is not None:
        dec = np.ceil(oos.groupby("date")["pred"].rank(pct=True) * 10).clip(1, 10).astype(int)
        g = oos.groupby(dec)["fwd_excess"].mean() * 100
        ui.section("Do higher scores really do better?", f"Average {HORIZON}-day result against the market, by score group")
        f5 = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g.values,
                              marker_color=[ui.UP if x >= 0 else ui.DOWN for x in g.values]))
        f5.update_xaxes(title="Score group (10 is the highest-scored tenth)")
        st.plotly_chart(ui.style_fig(f5, 300, hover="closest"))
    yearly = ui.P.base / "models" / "stability_by_year.csv"
    if yearly.exists():
        ui.section("Year by year")
        ui.show_df(pd.read_csv(yearly).set_index("year").round(3))


render()
