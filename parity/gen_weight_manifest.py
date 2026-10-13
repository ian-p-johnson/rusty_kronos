#!/usr/bin/env python3
"""Add a per-tensor inventory to parity/fixtures/weights.json.

A safetensors sha256 proves the *file* is the right one; it does not prove the
loader consumed the right *tensors*. This adds shape + dtype + a float64 sum and
absmax for every tensor, which is what the Rust loader must reproduce.

Fast: reads safetensors directly, does not load any model.

Usage:
    python parity/gen_weight_manifest.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "parity" / "fixtures"
WEIGHTS = FIX / "weights.json"


def inventory(path: Path) -> dict:
    out = {}
    with safe_open(str(path), framework="numpy") as f:
        for name in sorted(f.keys()):
            t = f.get_tensor(name).astype(np.float64, copy=False)
            out[name] = {
                "shape": list(t.shape),
                "sum": float(t.sum()),
                "absmax": float(np.abs(t).max()),
                "n": int(t.size),
            }
    return out


def main() -> int:
    idx = json.loads(WEIGHTS.read_text())
    for name in sorted(idx):
        for kind in ("model", "tokenizer"):
            entry = idx[name][kind]
            print(f"[{name}/{kind}] {entry['repo']}@{entry['revision'][:8]} ...", end="", flush=True)
            # resolve via the HF cache rather than re-downloading
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    entry["repo"],
                    revision=entry["revision"],
                    allow_patterns=["*.safetensors"],
                )
            ) / entry["safetensors"]["filename"]
            inv = inventory(root)
            entry["tensors"] = inv
            print(f" {len(inv)} tensors", flush=True)
    WEIGHTS.write_text(json.dumps(idx, indent=2, sort_keys=True))
    total = sum(len(e["tensors"]) for m in idx.values() for e in m.values())
    print(f"\nwrote {WEIGHTS} ({total} tensors across {len(idx)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())