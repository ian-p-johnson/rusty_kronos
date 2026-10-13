#!/usr/bin/env python3
"""Generate the shared probe input for the per-layer parity check.

All three models are probed with the same 512-row window from
`tests/data/regression_input.csv`, normalised and clipped exactly as
`KronosPredictor.predict` does. One probe file serves all three.

Usage:
    python parity/gen_probe.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "tests" / "data" / "regression_input.csv"
OUT = REPO / "parity" / "fixtures"
CTX_LEN = 512
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def main() -> int:
    df = pd.read_csv(INPUT, parse_dates=["timestamps"])
    ctx = df.iloc[:CTX_LEN]
    raw = ctx[FEATURES].to_numpy().astype(np.float32)
    mean, std = raw.mean(0), raw.std(0)
    x = np.ascontiguousarray(np.clip((raw - mean) / (std + 1e-5), -5.0, 5.0)[None]).astype(np.float32)
    ts = pd.to_datetime(ctx["timestamps"])
    stamp = np.ascontiguousarray(
        np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1)
    ).astype(np.float32)[None]

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "probe_x.npy", x)
    np.save(OUT / "probe_stamp.npy", stamp)
    print(f"probe_x:     {x.shape}  f_order={x.flags['F_CONTIGUOUS']}  range [{x.min():.3f}, {x.max():.3f}]")
    print(f"probe_stamp: {stamp.shape}  f_order={stamp.flags['F_CONTIGUOUS']}  range [{stamp.min():.0f}, {stamp.max():.0f}]")
    print(f"wrote to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())