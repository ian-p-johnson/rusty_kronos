#!/usr/bin/env python3
"""Compare the Rust weight loader's per-tensor manifest against PyTorch.

This is the Phase 1 exit gate. Shape equality is exact. `sum`/`absmax` are
compared with a relative tolerance rather than exactly, because the reduction
order differs between torch and candle -- the point is to prove the RIGHT
TENSOR was loaded, not to verify float addition associativity.

Usage:
    python parity/check_weights.py <rust-manifest-dir> [--rtol 1e-5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "parity" / "fixtures" / "weights.json"

# rust manifest stem -> (model key, "model" | "tokenizer")
PAIRS = {
    "small": ("kronos-small", "model"),
    "base": ("kronos-base", "model"),
    "mini": ("kronos-mini", "model"),
    "tok-base": ("kronos-small", "tokenizer"),  # Kronos-Tokenizer-base
    "tok-2k": ("kronos-mini", "tokenizer"),     # Kronos-Tokenizer-2k
}


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_dir", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args()

    ref = json.loads(WEIGHTS.read_text())
    failures: list[str] = []
    checked = 0

    for stem, (model_key, kind) in PAIRS.items():
        path = args.manifest_dir / f"{stem}.json"
        if not path.exists():
            failures.append(f"{stem}: missing rust manifest {path}")
            continue
        got = json.loads(path.read_text())
        want = ref[model_key][kind]["tensors"]

        got_keys, want_keys = set(got["tensors"]), set(want)
        if got_keys != want_keys:
            failures.append(
                f"{stem}: key set mismatch "
                f"(+rust-only {sorted(got_keys - want_keys)[:5]}, "
                f"-missing {sorted(want_keys - got_keys)[:5]})"
            )
            continue

        worst_sum, worst_abs, worst_name = 0.0, 0.0, ""
        for name in sorted(want_keys):
            g, w = got["tensors"][name], want[name]
            checked += 1
            if g["shape"] != w["shape"]:
                failures.append(f"{stem}/{name}: shape {g['shape']} != {w['shape']}")
                continue
            # basis buffers are integer-valued; candle keeps them as u32/i64
            if isinstance(g.get("sum"), (int, float)) and w["n"] > 1:
                rs = rel(float(g["sum"]), float(w["sum"]))
                ra = rel(float(g["absmax"]), float(w["absmax"]))
                if rs > worst_sum:
                    worst_sum, worst_name = rs, name
                worst_abs = max(worst_abs, ra)
                if rs > args.rtol:
                    failures.append(
                        f"{stem}/{name}: sum rel {rs:.3e} > {args.rtol:.0e} "
                        f"(rust {g['sum']} vs torch {w['sum']})"
                    )
                if ra > args.rtol:
                    failures.append(
                        f"{stem}/{name}: absmax rel {ra:.3e} > {args.rtol:.0e}"
                    )
        status = "FAIL" if failures else "ok"
        print(f"  {stem:9s} {len(want_keys):4d} tensors  {status:4s}  "
              f"worst sum rel {worst_sum:.2e} ({worst_name or '-'})  "
              f"worst absmax rel {worst_abs:.2e}")

    print(f"\nchecked {checked} tensors across {len(PAIRS)} checkpoints")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures[:40]:
            print(f"  - {f}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        return 1
    print("PASS: every tensor name, shape and value matches the PyTorch manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())