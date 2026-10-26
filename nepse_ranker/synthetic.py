"""Practice data. Fake stocks that behave a bit like NEPSE so you can run the whole
pipeline today. The numbers are made up: results on this data mean NOTHING about
the real market. Its only job is to prove the plumbing works."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic(n_symbols: int = 80, n_days: int = 900, seed: int = 7,
                       start: str = "2022-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cal = pd.date_range(start, periods=int(n_days * 7 / 5) + 20, freq="D")
    dates = cal[~cal.weekday.isin([4, 5])][:n_days]          # trade Sunday to Thursday
    S, T = n_symbols, len(dates)

    symbols = np.array([f"S{chr(65 + i // 26)}{chr(65 + i % 26)}X" for i in range(S)])
    sector = rng.integers(0, 6, S)
    beta = rng.uniform(0.6, 1.4, S)
    idio = rng.uniform(0.012, 0.03, S)
    base_turn = np.exp(rng.normal(np.log(2_000_000), 0.9, S))
    p_skip = np.where(rng.random(S) < 0.3, 0.12, 0.01)       # some stocks trade less often

    mkt = rng.normal(0.0003, 0.008, T)
    sec = rng.normal(0, 0.004, (T, 6))
    rets = np.zeros((T, S))
    close_prev = np.exp(rng.uniform(np.log(150), np.log(1500), S))
    events = {s: rng.choice(np.arange(100, T - 50), 2, replace=False) for s in range(S)}

    rows = []
    for t in range(T):
        sig = rets[max(0, t - 5):t].sum(axis=0) if t > 0 else np.zeros(S)
        r = beta * mkt[t] + sec[t, sector] + idio * rng.normal(size=S) + 0.10 * sig   # weak planted signal
        r = np.clip(r, -0.095, 0.095)
        traded = rng.random(S) > p_skip
        r = np.where(traded, r, 0.0)
        rets[t] = r
        base = close_prev.copy()
        for s in range(S):
            if t in events[s]:
                base[s] = close_prev[s] / 1.15            # a 15% bonus: exchange resets previous close
        close = np.round(base * (1 + r), 1)
        close = np.where(traded, close, close_prev)
        gap = rng.normal(0, 0.003, S)
        opn = np.round(base * (1 + gap), 1)
        hi = np.round(np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.004, S))), 1)
        lo = np.round(np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.004, S))), 1)
        turn = base_turn * np.exp(rng.normal(0, 0.5, S)) * (1 + 6 * np.abs(r))
        vol = np.round(turn / close)
        trades = np.maximum(1, np.round(vol / np.exp(rng.normal(np.log(45), 0.4, S))))
        for s in np.flatnonzero(traded):
            rows.append((dates[t], symbols[s], opn[s], hi[s], lo[s], close[s],
                         round(base[s], 1), vol[s], round(turn[s], 1), trades[s]))
        close_prev = np.where(traded, close, close_prev)

    df = pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close",
                                     "prev_close", "volume", "turnover", "trades"])
    return df
