#!/usr/bin/env python3
"""Benchmark the frozen PyTorch reference and record it as the port's baseline.

The Rust port's only meaningful speed claim is "faster than this file's
output", so the baseline has to be produced by an identical procedure for all
three models and committed.

Usage:
    python parity/bench_py.py                 # write parity/baseline_pytorch.json
    python parity/bench_py.py --quick         # fewer configs / iterations
    python parity/bench_py.py --dtype fp16    # also/instead measure reduced precision
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from parity.gen_reference import MODELS, SEED, set_seed  # noqa: E402
from model import Kronos, KronosTokenizer  # noqa: E402
from model.kronos import auto_regressive_inference  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT = REPO / "parity" / "baseline_pytorch.json"

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# (ctx_len, pred_len, sample_count)
#
# N is capped at 10 deliberately. Large-batch configs are pure GEMM work where
# the port's upside is ~1.2x, and they take minutes each on this box -- which
# heats the GPU and pollutes every measurement that follows. Use
# `--big-batch` when you specifically want the throughput regime.
CONFIGS = [
    (400, 120, 1),
    (400, 120, 5),
    (512, 120, 1),
    (512, 120, 10),
    (512, 48, 10),
]
BIG_CONFIGS = [
    (512, 120, 64),
    (512, 48, 256),
]

COOLDOWN_S = 10.0  # let the GPU settle so configs are comparable to each other


def timeit(fn, warmup: int, iters: int) -> float:
    """Return the minimum wall time in seconds (min is the right statistic for
    latency: it is the least noise-contaminated estimate of the true cost)."""
    for _ in range(warmup):
        fn()
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench(name: str, dtype: torch.dtype, configs, iters: int) -> list[dict]:
    spec = MODELS[name]
    tok = (
        KronosTokenizer.from_pretrained(spec["tokenizer"], revision=spec["tokenizer_rev"])
        .to(DEVICE).eval().to(dtype)
    )
    mdl = Kronos.from_pretrained(spec["model"], revision=spec["model_rev"]).to(DEVICE).eval().to(dtype)

    rows = []
    for ctx, pred, n in configs:
        x = np.random.default_rng(0).standard_normal((n, ctx, 6)).astype(np.float32)
        xs = np.zeros((n, ctx, 5), np.float32)
        ys = np.zeros((n, pred, 5), np.float32)
        xt = torch.from_numpy(np.clip(x, -5, 5)).to(DEVICE).to(dtype)
        st = torch.from_numpy(xs).to(DEVICE).to(dtype)
        yt = torch.from_numpy(ys).to(DEVICE).to(dtype)

        def run():
            set_seed(SEED)
            with torch.no_grad():
                # sample_count=1: the batch dim ALREADY carries n. Passing n here
                # too would make auto_regressive_inference square the batch.
                return auto_regressive_inference(tok, mdl, xt, st, yt, 512, pred, 5, 0.6, 0, 0.9, 1, False)

        time.sleep(COOLDOWN_S)  # thermal settling, see COOLDOWN_S
        try:
            t = timeit(run, warmup=1, iters=iters)
        except RuntimeError as e:  # e.g. bf16 .numpy() in the frozen reference
            rows.append({"model": name, "ctx": ctx, "pred_len": pred, "batch": n,
                         "dtype": str(dtype).split(".")[-1], "error": str(e)[:160]})
            print(f"  {name:12s} ctx={ctx:3d} pred={pred:3d} N={n:3d} {dtype} -> "
                  f"ERROR: {str(e)[:60]}", flush=True)
            continue
        rows.append({
            "model": name, "ctx": ctx, "pred_len": pred, "batch": n,
            "dtype": str(dtype).split(".")[-1],
            "total_ms": round(t * 1e3, 3),
            "per_step_us": round(t / pred * 1e6, 1),
            "us_per_sample_step": round(t / (pred * n) * 1e6, 2),
        })
        print(f"  {name:12s} ctx={ctx:3d} pred={pred:3d} N={n:3d} "
              f"{str(dtype).split('.')[-1]:8s} {t*1e3:9.2f} ms  {t/pred*1e6:9.1f} us/step",
              flush=True)

    del tok, mdl
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 iteration instead of 3")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    ap.add_argument("--big-batch", action="store_true",
                    help="also measure N=64/256 (minutes per config; heats the GPU)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0) if DEVICE.startswith("cuda") else None
    env = {
        "gpu": p.name if p else "cpu",
        "sm_count": p.multi_processor_count if p else None,
        "capability": f"{p.major}.{p.minor}" if p else None,
        "vram_gib": round(p.total_memory / 2**30, 2) if p else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cooldown_s": COOLDOWN_S,
        "iters": 1 if args.quick else 3,
        "statistic": "min wall time of N iterations, after 1 warmup",
    }
    print(f"device: {env['gpu']} (SMs={env['sm_count']}, cc={env['capability']}), "
          f"torch {torch.__version__}")
    print("NOTE: this is a *Laptop* RTX 5070 Ti. Desktop parts have more SMs and higher\n"
          "      clocks; see PORTING_FROM_PYTHON.md S6.6 before extrapolating.\n")

    configs = CONFIGS + (BIG_CONFIGS if args.big_batch else [])
    iters = 1 if args.quick else 3

    rows: list[dict] = []
    for name in MODELS:
        print(f"[{name}]", flush=True)
        rows += bench(name, DTYPES[args.dtype], configs, iters)

    payload = {
        "note": "Frozen PyTorch reference baseline. Generated by parity/bench_py.py. "
                "The Rust port is measured against exactly this procedure. "
                "Consumer GPUs throttle under sustained load: always re-measure Python "
                "and Rust in the SAME session for any speed claim, rather than "
                "comparing against a stale copy of this file.",
        "env": env,
        "config": {"models": list(MODELS), "configs": [list(c) for c in configs]},
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())