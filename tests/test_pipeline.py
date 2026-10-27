"""Automated checks for the parts of the pipeline where a silent mistake would be costly.

Run from the project folder with:   python -m pytest -q
They use made-up practice data, so they need no internet and take about a minute.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from nepse_ranker.clean import build_clean_panel
from nepse_ranker.config import HORIZON, closed_weekdays, price_limit_series
from nepse_ranker.dividends import _infer_fy, parse_declarations
from nepse_ranker.features import FEATURE_COLS, add_labels, build_features
from nepse_ranker.floorsheet import _page_rows
from nepse_ranker.model import walk_forward
from nepse_ranker.news_features import EVENT_FEATURES, NEGATIVE_EVENTS, NEWS_COLS, add_news_features
from nepse_ranker.news_prep import reaction_positions
from nepse_ranker.normalize import rows_to_frame
from nepse_ranker.synthetic import generate_synthetic


@pytest.fixture(scope="module")
def raw():
    return generate_synthetic(n_symbols=25, n_days=420, seed=3)


@pytest.fixture(scope="module")
def panel(raw):
    return build_clean_panel(raw)


@pytest.fixture(scope="module")
def feats(panel):
    return add_labels(build_features(panel))


# ---------------------------------------------------------------- data handling
def test_nepse_field_names_are_recognised():
    rows = [{"businessDate": "2026-10-20", "symbol": "NABIL", "openPrice": 1000, "highPrice": 1020,
             "lowPrice": 990, "closePrice": 1010, "previousDayClosePrice": 1000,
             "totalTradedQuantity": 5000, "totalTradedValue": 5050000, "totalTrades": 120}]
    df, report = rows_to_frame(rows)
    assert report["missing"] == []
    assert df.loc[0, "symbol"] == "NABIL" and df.loc[0, "close"] == 1010


def test_bonus_adjustment_keeps_returns_continuous(panel):
    grp = panel.groupby("symbol")
    adjusted = panel["adj_close"] / grp["adj_close"].shift(1) - 1
    reported = panel["close"] / panel["prev_close"] - 1
    ok = (panel["traded"] == 1) & adjusted.notna()
    assert (adjusted[ok] - reported[ok]).abs().max() < 1e-9
    assert panel["corp_action"].sum() > 0


def test_every_stock_has_a_row_for_every_trading_day(panel):
    counts = panel.groupby("symbol")["date"].agg(["count", "min", "max"])
    all_days = panel["date"].nunique()
    full = counts[counts["max"] == panel["date"].max()]
    assert (full["count"] <= all_days).all()
    for sym, g in panel.groupby("symbol"):
        assert g["date"].is_monotonic_increasing and not g["date"].duplicated().any()


# ---------------------------------------------------------------- trading rules
def test_trading_week_and_price_limit_follow_the_rule_changes():
    assert closed_weekdays(date(2026, 3, 1)) == (4, 5)      # Friday and Saturday closed (old week)
    assert closed_weekdays(date(2026, 5, 1)) == (5, 6)      # Saturday and Sunday closed (new week)
    s = price_limit_series(pd.Series(pd.to_datetime(["2026-04-19", "2026-04-20"])))
    assert list(s) == [0.10, 0.15]


# ---------------------------------------------------------------- no peeking at the future
def test_features_never_use_the_future(panel):
    cut_date = np.sort(panel["date"].unique())[300]
    full = build_features(panel)
    cut = build_features(panel[panel["date"] <= cut_date])
    a = cut[cut["date"] <= cut_date].set_index(["symbol", "date"])[FEATURE_COLS].sort_index()
    b = full[full["date"] <= cut_date].set_index(["symbol", "date"])[FEATURE_COLS].sort_index()
    a, b = a.align(b, join="inner")
    assert len(a) > 1000
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_label_is_the_return_from_next_close_over_the_horizon(feats):
    s = feats[feats["symbol"] == feats["symbol"].iloc[0]].reset_index(drop=True)
    i = 100
    expected = s.loc[i + 1 + HORIZON, "adj_close"] / s.loc[i + 1, "adj_close"] - 1
    assert s.loc[i, "fwd_ret"] == pytest.approx(expected)


def test_labels_exist_only_for_tradable_stocks_and_stay_between_0_and_1(feats):
    assert feats["label_rank"].dropna().between(0, 1).all()
    assert feats.loc[~feats["eligible"], "label_rank"].isna().all()


def test_walk_forward_test_blocks_are_in_order_and_do_not_overlap(feats):
    oos = walk_forward(feats, min_train_days=120, test_days=40, max_folds=3, verbose=False)
    spans = oos.groupby("fold")["date"].agg(["min", "max"]).sort_index()
    assert len(spans) == 3
    assert (spans["min"].iloc[1:].to_numpy() > spans["max"].iloc[:-1].to_numpy()).all()
    assert oos["pred"].notna().all()


# ---------------------------------------------------------------- news
def test_news_is_assigned_to_the_first_close_after_it_was_published():
    trading_days = pd.DatetimeIndex(["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-12"])
    published = pd.Series(pd.to_datetime(["2026-10-06 10:00", "2026-10-06 16:00", "2026-10-10 12:00",
                                          "2026-10-07 15:00"]))
    assert list(reaction_positions(published, trading_days)) == [1, 2, 3, 3]


def test_news_features_never_use_the_future(panel):
    rng = np.random.default_rng(0)
    syms = sorted(panel["symbol"].unique())[:5]
    days = np.sort(panel["date"].unique())
    news = pd.DataFrame({
        "symbol": [syms[i % 5] for i in range(80)],
        "published": [pd.Timestamp(days[int(rng.integers(20, len(days)))]) + pd.Timedelta(hours=int(rng.integers(9, 19)))
                      for _ in range(80)]})
    for e in set(EVENT_FEATURES) | set(NEGATIVE_EVENTS):
        news["evt_" + e] = rng.random(len(news)) < 0.4
    base = panel[["date", "symbol"]]
    cut_date = days[300]
    full = add_news_features(base, news)
    cut = add_news_features(base[base["date"] <= cut_date], news)
    a = cut.set_index(["symbol", "date"])[NEWS_COLS].sort_index()
    b = full[full["date"] <= cut_date].set_index(["symbol", "date"])[NEWS_COLS].sort_index()
    a, b = a.align(b, join="inner")
    assert np.allclose(a.to_numpy(float), b.to_numpy(float))


# ---------------------------------------------------------------- dividends
def test_dividend_text_is_read_correctly():
    news = pd.DataFrame({
        "source": "company_news",
        "symbol": ["NTC", "NABIL", "SMB", "OLDX", "SGIC"],
        "published": pd.to_datetime(["2024-12-21", "2026-09-06", "2024-11-22", "2024-12-21", "2024-01-25"]),
        "title": ["Dividend Declaration [NTC]", "Declaration of Dividend-Nabil Bank [NABIL]",
                  "Book Close Declared  [SMB]", "Dividend Declaration [OLDX]", "No Dividend  [SGIC]"],
        "body": ["Dividend Declaration of NTC Fiscal Year: 2080-2081 Cash Dividend: 30%",
                 "Dividend Declaration of NABIL Fiscal Year: 2082-2083 Cash Dividend: 10.8% Bonus Shares: 5%",
                 "Book Close of SMB Book Close Date : 11/29/2024 Cash Dividend: 0.725% Bonus Shares: 13.775%",
                 "Dividend Declaration of OLDX Fiscal Year: 2020-21 Cash Dividend: 40%",
                 "No Dividend Declaration of SGIC"]})
    out = parse_declarations(news).set_index("symbol")
    assert out.loc["NTC", "cash_pct"] == 30 and out.loc["NTC", "fy_start"] == 2080
    assert out.loc["NABIL", "cash_pct"] == 10.8 and out.loc["NABIL", "bonus_pct"] == 5
    assert out.loc["SMB", "fy_start"] == 2080 and out.loc["SMB", "bonus_pct"] == pytest.approx(13.775)
    assert out.loc["OLDX", "fy_start"] == 2077            # fiscal year given in AD is converted
    assert out.loc["SGIC", "cash_pct"] == 0               # an explicit "no dividend" notice means zero


def test_fiscal_year_is_inferred_from_the_announcement_date():
    assert _infer_fy(pd.Timestamp("2024-11-22")) == 2080
    assert _infer_fy(pd.Timestamp("2025-01-15")) == 2080
    assert _infer_fy(pd.Timestamp("2026-10-12")) == 2082


# ---------------------------------------------------------------- floorsheet
def test_floorsheet_page_is_unpacked():
    resp = {"totalTrades": 1234, "floorsheets": {"content": [{"contractId": 1}, {"contractId": 2}], "totalPages": 3}}
    rows, pages, total = _page_rows(resp)
    assert len(rows) == 2 and pages == 3 and total == 1234
    assert _page_rows("not a dict") == ([], None, None)


# ---------------------------------------------------------------- live track record
def test_ranking_log_keeps_the_first_ranking_saved_for_a_date(tmp_path):
    from nepse_ranker.config import Paths
    from nepse_ranker.livelog import append_history, history_path
    p = Paths(tmp_path)
    p.ensure()
    day = pd.Timestamp("2026-10-16")
    first = pd.DataFrame({"date": day, "symbol": ["A", "B"], "rank": [1, 2], "score": [0.9, 0.1],
                          "score_pct": [100.0, 50.0], "close": [10.0, 20.0]})
    changed = first.assign(rank=[2, 1])
    assert append_history(first, p) == 1
    assert append_history(changed, p) == 0                  # a rerun must not rewrite history
    saved = pd.read_parquet(history_path(p)).sort_values("symbol")
    assert list(saved["rank"]) == [1, 2]
