"""Shared look and data access for the dashboard pages."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from nepse_ranker.config import HORIZON, Paths  # noqa: E402

P = Paths()
ACCENT, UP, DOWN, MUTED = "#0F766E", "#16A34A", "#DC2626", "#7A8699"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---- Fonts: Inter for text, but NEVER for icon glyphs or code ------------------------------ */
html, body,
body *:not([data-testid="stIconMaterial"]):not([class*="material"]):not(code):not(pre):not(kbd) {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}
code, pre, kbd, code *, pre * {
    font-family: ui-monospace, 'JetBrains Mono', 'Cascadia Code', Consolas, Menlo, monospace !important;
}
[data-testid="stIconMaterial"], .material-icons, [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-weight: 400 !important; font-style: normal !important; letter-spacing: normal !important;
    text-transform: none !important; font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important; white-space: nowrap; direction: ltr;
}

/* ---- Page frame ---------------------------------------------------------------------------- */
.block-container { padding-top: 4.6rem; padding-bottom: 3rem; padding-left: 2.5rem; padding-right: 2.5rem;
                   max-width: 1400px; }
#MainMenu, footer, .stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stDeployButton"] {
    display: none !important; }
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 700; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(127,127,127,.18); }
[data-testid="stSidebar"] [data-testid="stSelectbox"] label p { font-weight: 600; font-size: .85rem; }

/* ---- Cards and KPI grids (CSS grid => every card in a row has the same height) ------------- */
.card { border: 1px solid rgba(127,127,127,.22); border-radius: 16px; padding: 16px 18px; box-sizing: border-box;
        background: rgba(127,127,127,.05); box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.card.tight { padding: 4px 16px; }
.kpi-grid, .pick-grid { display: grid; gap: 14px; margin: 2px 0 6px; }
.kpi-grid > .card, .pick-grid > .card { height: 100%; display: flex; flex-direction: column; }
.kpi-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .62; font-weight: 600; }
.kpi-grid.long-labels .kpi-label { min-height: 2.4em; }
.kpi-value { font-size: 1.8rem; font-weight: 700; line-height: 1.15; margin-top: 6px; letter-spacing: -0.02em; }
.kpi-value.up { color: #16A34A; } .kpi-value.down { color: #DC2626; }
.kpi-sub { font-size: .82rem; opacity: .68; margin-top: auto; padding-top: 4px; line-height: 1.35; }

.chip { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
        white-space: nowrap; }
.chip.up { background: rgba(22,163,74,.15); color: #16A34A; }
.chip.down { background: rgba(220,38,38,.15); color: #DC2626; }
.chip.flat { background: rgba(127,127,127,.18); }
.chip.warn { background: rgba(217,119,6,.17); color: #D97706; }
.chip.accent { background: rgba(20,184,166,.17); color: #14B8A6; }

/* ---- Hero ----------------------------------------------------------------------------------- */
.hero { background: radial-gradient(900px 260px at 92% -30%, rgba(94,234,212,.28), transparent 70%),
                    linear-gradient(135deg, #0F766E 0%, #134E4A 100%);
        color: #fff; border-radius: 20px; padding: 26px 30px 24px; margin-bottom: 18px; }
.hero h1 { color: #fff; margin: 0 0 6px 0 !important; padding: 0 !important; font-size: 1.9rem; line-height: 1.2; }
.hero p { margin: 0 !important; opacity: .88; }
.hero .chip { background: rgba(255,255,255,.17); color: #fff; margin: 16px 8px 0 0; }
.hero .chip.warn { background: rgba(251,191,36,.28); color: #FEF3C7; }

/* ---- Lists, picks, bars --------------------------------------------------------------------- */
.row { display: flex; justify-content: space-between; align-items: center; padding: 9px 2px;
       border-bottom: 1px solid rgba(127,127,127,.14); gap: 10px; }
.row:last-child { border-bottom: none; }
.row .sym { font-weight: 700; min-width: 70px; }
.row .muted { opacity: .65; font-size: .86rem; margin-left: auto; font-variant-numeric: tabular-nums; }
.row .chip { min-width: 78px; text-align: center; }
.scorebar { height: 6px; border-radius: 999px; background: rgba(127,127,127,.2); margin-top: 10px; overflow: hidden; }
.scorebar > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #14B8A6, #0F766E); }
.pick .rank { font-size: .72rem; opacity: .6; font-weight: 600; letter-spacing: .06em; }
.pick .sym { font-size: 1.25rem; font-weight: 700; }
.pick .name { font-size: .78rem; opacity: .6; min-height: 1.1rem; overflow: hidden; white-space: nowrap;
              text-overflow: ellipsis; }
.pick .foot { margin-top: auto; padding-top: 8px; }
.section-title { font-size: 1.05rem; font-weight: 700; margin: 6px 0 2px 0; }
.section-sub { font-size: .85rem; opacity: .65; margin-bottom: 10px; }
.page-head { margin-bottom: 14px; }
.page-head h1 { font-size: 1.75rem; margin: 0 !important; padding: 0 !important; line-height: 1.2; }
.page-head p { opacity: .68; max-width: 920px; margin: 6px 0 0 !important; font-size: .92rem; line-height: 1.55; }

/* ---- Tabs as pills ---------------------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding: 6px 16px; height: auto; }
.stTabs [aria-selected="true"] { background: rgba(15,118,110,.14); }

@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .hero { padding: 20px 18px; } .hero h1 { font-size: 1.5rem; }
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def _read_parquet(path: str, mtime: float, columns):
    return pd.read_parquet(path, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get(path: Path, columns=None):
    if not path.exists():
        return None
    return _read_parquet(str(path), path.stat().st_mtime, tuple(columns) if columns else None)


def get_json(path: Path):
    if not path.exists():
        return None
    return _read_json(str(path), path.stat().st_mtime)


def core() -> dict:
    latest = get(P.latest)
    cls = get_json(P.base / "company" / "classification.json") or []
    return {
        "latest": latest,
        "metrics": get_json(P.metrics),
        "sectors": get_json(P.base / "company" / "sectors.json") or {},
        "names": {str(r.get("symbol")).upper(): r.get("securityName") for r in cls if isinstance(r, dict)},
    }


def current_symbol(options) -> str:
    sym = st.session_state.get("current_symbol")
    return sym if sym in options else options[0]


# ---------------------------------------------------------------- small html pieces
def pct_chip(x, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return '<span class="chip flat">n/a</span>'
    kind = "up" if x > 0.0005 else "down" if x < -0.0005 else "flat"
    arrow = "&#9650;" if kind == "up" else "&#9660;" if kind == "down" else "&bull;"
    return f'<span class="chip {kind}">{arrow} {abs(x) * 100:.{digits}f}%</span>'


def chip(text: str, kind: str = "flat") -> str:
    return f'<span class="chip {kind}">{text}</span>'


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = f" {tone}" if tone in ("up", "down") else ""
    return (f'<div class="card"><div class="kpi-label">{label}</div><div class="kpi-value{cls}">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpi_grid(items, min_width: int = 175) -> None:
    """Render KPI cards in one CSS grid so they are all the same height and wrap nicely on small screens.

    items: tuples of (label, value[, sub[, tone]]) where tone is "up" or "down".
    """
    cells = "".join(kpi(*it) for it in items)
    long_labels = " long-labels" if max(len(it[0]) for it in items) > 19 else ""
    st.markdown(f'<div class="kpi-grid{long_labels}" style="grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr))">'
                f'{cells}</div>', unsafe_allow_html=True)


def card_grid(cards, min_width: int = 170) -> None:
    """Render ready-made card html strings in an equal-height grid."""
    st.markdown(f'<div class="pick-grid" style="grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr))">'
                f'{"".join(cards)}</div>', unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div><div class="section-sub">{sub}</div>',
                unsafe_allow_html=True)


def page_header(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="page-head"><h1>{title}</h1>{f"<p>{sub}</p>" if sub else ""}</div>',
                unsafe_allow_html=True)


def gap(px: int = 8) -> None:
    st.markdown(f'<div style="height:{px}px"></div>', unsafe_allow_html=True)


def show_df(data, **kw):
    """st.dataframe that fills the width on every Streamlit version."""
    try:
        return st.dataframe(data, width="stretch", **kw)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kw)


def show_editor(data, **kw):
    """st.data_editor that fills the width on every Streamlit version."""
    try:
        return st.data_editor(data, width="stretch", **kw)
    except TypeError:
        return st.data_editor(data, use_container_width=True, **kw)


def link(path: str, label: str, icon: str = ":material/arrow_forward:") -> None:
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:
        pass


def go_to_stock(sym: str) -> None:
    st.session_state["current_symbol"] = sym
    st.switch_page("views/stock.py")


def style_fig(fig, height: int = 380, hover: str = "x unified"):
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=28, b=4), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hovermode=hover,
                      font=dict(family="Inter, system-ui, sans-serif"),
                      legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(gridcolor="rgba(127,127,127,.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(127,127,127,.15)", zeroline=False)
    return fig


# ---------------------------------------------------------------- market map
_RED, _NEUTRAL, _GREEN = (200, 45, 45), (63, 72, 86), (22, 150, 80)


def heat_color(pct: float, limit: float = 3.0) -> str:
    """Red for falls, slate for flat, green for gains. Saturates at +/- `limit` percent."""
    t = max(-1.0, min(1.0, 0.0 if pct is None or pd.isna(pct) else pct / limit))
    end = _GREEN if t >= 0 else _RED
    mix = tuple(round(a + (b - a) * abs(t)) for a, b in zip(_NEUTRAL, end))
    return "rgb(%d,%d,%d)" % mix


def treemap_nodes(latest: pd.DataFrame, sectors: dict) -> dict:
    """Plain lists for go.Treemap. Labels are pre-formatted strings, so nothing depends on %{color} templates."""
    t = latest[["symbol", "ret_1", "turn_20"]].copy()
    t["sector"] = t["symbol"].map(sectors).fillna("Other")
    t["size"] = t["turn_20"].fillna(0).clip(lower=1)
    t["move"] = t["ret_1"].fillna(0) * 100
    root = "NEPSE"
    n = {k: [] for k in ("ids", "labels", "parents", "values", "colors", "text", "hover")}

    def add(i, label, parent, value, move, text, hover):
        n["ids"].append(i); n["labels"].append(label); n["parents"].append(parent); n["values"].append(float(value))
        n["colors"].append(heat_color(move) if parent else "#0B1220"); n["text"].append(text); n["hover"].append(hover)

    mkt = (t["move"] * t["size"]).sum() / t["size"].sum()
    add(root, root, "", t["size"].sum(), mkt, root, f"<b>NEPSE</b><br>Activity-weighted move: {mkt:+.2f}%")
    for sec, g in t.groupby("sector"):
        mv = (g["move"] * g["size"]).sum() / g["size"].sum()
        sid = f"sector::{sec}"
        add(sid, sec, root, g["size"].sum(), mv, f"{sec}  {mv:+.1f}%",
            f"<b>{sec}</b><br>{len(g)} stocks<br>Activity-weighted move: {mv:+.2f}%")
        n["colors"][-1] = "#111A2B"  # sector headers: dark band, so the tiles carry the colour
        for r in g.itertuples():
            add(r.symbol, r.symbol, sid, r.size, r.move, f"<b>{r.symbol}</b><br>{r.move:+.1f}%",
                f"<b>{r.symbol}</b> ({sec})<br>Today: {r.move:+.2f}%")
    return n


def market_treemap(latest: pd.DataFrame, sectors: dict, height: int = 600):
    import plotly.graph_objects as go
    n = treemap_nodes(latest, sectors)
    fig = go.Figure(go.Treemap(
        ids=n["ids"], labels=n["labels"], parents=n["parents"], values=n["values"], branchvalues="total",
        text=n["text"], textinfo="text", hovertext=n["hover"], hoverinfo="text",
        marker=dict(colors=n["colors"], line=dict(width=1.5, color="#0B1220")),
        textfont=dict(color="#FFFFFF", size=13), tiling=dict(pad=3),
        pathbar=dict(visible=True, thickness=24), root_color="#0B1220"))
    style_fig(fig, height, hover="closest")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), uniformtext=dict(minsize=10, mode="hide"))
    return fig


def empty_state(message: str, command: str = "") -> None:
    extra = f"<br><code>{command}</code>" if command else ""
    st.markdown(f'<div class="card">{message}{extra}</div>', unsafe_allow_html=True)
