#!/usr/bin/env python3
"""Generate parity reference fixtures from the FROZEN PyTorch reference.

This is the oracle for the Rust port. It reads `model/` and `tests/` without
modifying them (see parity/check_frozen.sh) and writes:

  parity/fixtures/golden/<model>_ctx<N>_pred<P>.csv
        End-to-end deterministic (greedy) predictions, same CSV shape as the
        upstream tests/data/regression_output_*.csv.

  parity/fixtures/layers/<model>/<tensor>.npy  + index.json
        Per-stage activations captured with forward hooks, so a Rust parity
        failure can be localised to a single op instead of a single number.
        Tensor names are stable and are the contract the Rust side codes against.

  parity/fixtures/weights.json
        Pinned HF revisions + per-tensor checksums, so the Rust loader can prove
        it loaded exactly these bytes.

Usage:
    python parity/gen_reference.py --all
    python parity/gen_reference.py --model kronos-small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

FIX = REPO / "parity" / "fixtures"
INPUT_CSV = REPO / "tests" / "data" / "regression_input.csv"
FEATURES = ["open", "high", "low", "close", "volume", "amount"]

# ---------------------------------------------------------------------------
# Model registry. Revisions are pinned: the upstream regression test pins
# Kronos-small and Kronos-Tokenizer-base; mini/base/2k were resolved on
# 2026-10-13 and are pinned here so fixtures stay reproducible.
# ---------------------------------------------------------------------------
MODELS = {
    "kronos-small": {
        "model": "NeoQuasar/Kronos-small",
        "model_rev": "901c26c1332695a2a8f243eb2f37243a37bea320",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "tokenizer_rev": "0e0117387f39004a9016484a186a908917e22426",
        "max_context": 512,
    },
    "kronos-base": {
        "model": "NeoQuasar/Kronos-base",
        "model_rev": "2b554741eca47781b64468546e77fef3e85130e6",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "tokenizer_rev": "0e0117387f39004a9016484a186a908917e22426",
        "max_context": 512,
    },
    "kronos-mini": {
        "model": "NeoQuasar/Kronos-mini",
        "model_rev": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_rev": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        # Kronos-mini pairs with the 2k tokenizer and supports a 2048 window;
        # we exercise 512 for cross-model comparability.
        "max_context": 512,
    },
}

CTX_LENS = [256, 512]
PRED_LEN = 8
SEED = 123
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Weight inventory
# ---------------------------------------------------------------------------
def dump_weights(spec: dict) -> dict:
    from huggingface_hub import snapshot_download

    out = {}
    for kind in ("model", "tokenizer"):
        repo, rev = spec[kind], spec[f"{kind}_rev"]
        root = Path(snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors"]))
        cfg = json.loads((root / "config.json").read_text())
        st = root / "model.safetensors"
        out[kind] = {
            "repo": repo,
            "revision": rev,
            "config": cfg,
            "safetensors": {
                "filename": st.name,
                "bytes": st.stat().st_size,
                "sha256": sha256(st),
            },
        }
    return out


# ---------------------------------------------------------------------------
# Per-stage capture
# ---------------------------------------------------------------------------
class Capture:
    """Collect forward-hook activations under stable prefixed names.

    Prefixes matter: both `Kronos` and `KronosTokenizer` own modules called
    `head` and `embed`. Everything is namespaced `pred.*` / `tok.*` so the
    contract the Rust side codes against is unambiguous.
    """

    def __init__(self) -> None:
        self.tensors: dict[str, np.ndarray] = {}
        self._handles: list = []

    def attach(self, root: torch.nn.Module, prefix: str, names: list[str], pick=None) -> None:
        """Attach hooks to `names` (dotted paths relative to `root`).

        `pick` optionally maps a module output -> the tuple element to capture
        (needed for BSQuantizer, which returns (loss, quantized, indices)).
        """
        pick = pick or {}
        lookup = dict(root.named_modules())
        for name in names:
            mod = lookup.get(name)
            if mod is None:
                raise KeyError(f"module not found: {prefix}{name}")
            key = f"{prefix}{name}"

            def make(key: str, idx: int | None):
                def hook(_m, _inp, out):
                    t = out[idx] if idx is not None else (out[0] if isinstance(out, tuple) else out)
                    if isinstance(t, torch.Tensor):
                        self.tensors[key] = t.detach().float().cpu().numpy()

                return hook

            self._handles.append(mod.register_forward_hook(make(key, pick.get(name))))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def capture_layers(name: str, model, tokenizer, ctx_len: int) -> tuple[dict, dict]:
    """Run the deterministic forward (encode -> decode_s1 -> decode_s2) plus the
    tokenizer decode pass, capturing per-stage activations under `pred.*` and
    `tok.*` prefixes.

    Returns (tensors, extra) where `extra` carries integer token ids and the
    config the Rust side needs to reproduce the same computation.
    """
    cap = Capture()

    # Top-level submodules for every layer; per-projection captures are
    # restricted to layer 0 to keep fixture size sane while still validating
    # the projection maths exactly.
    pred_names = (
        ["embedding", "time_emb", "norm", "dep_layer", "head"]
        + [f"transformer.{i}" for i in range(model.n_layers)]
        + [
            "transformer.0.norm1",
            "transformer.0.self_attn.q_proj",
            "transformer.0.self_attn.k_proj",
            "transformer.0.self_attn.v_proj",
            "transformer.0.self_attn.out_proj",
            "transformer.0.norm2",
            "transformer.0.ffn.w1",
            "transformer.0.ffn.w2",
            "transformer.0.ffn.w3",
            "embedding.emb_s1",
            "embedding.emb_s2",
            "embedding.fusion_proj",
            "dep_layer.cross_attn",
            "dep_layer.norm",
            "head.proj_s1",
            "head.proj_s2",
        ]
    )
    cap.attach(model, "pred.", pred_names)

    n_enc = len(tokenizer.encoder)
    n_dec = len(tokenizer.decoder)
    tok_names = (
        ["embed", "quant_embed", "tokenizer.bsq", "post_quant_embed", "post_quant_embed_pre", "head"]
        + [f"encoder.{i}" for i in range(n_enc)]
        + [f"decoder.{i}" for i in range(n_dec)]
    )
    # BinarySphericalQuantizer.forward returns (zq, loss, metrics) -> capture zq.
    cap.attach(tokenizer, "tok.", tok_names, pick={"tokenizer.bsq": 0})

    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamps"])
    ctx = df.iloc[:ctx_len]
    raw = ctx[FEATURES].to_numpy().astype(np.float32)
    mean, std = raw.mean(0), raw.std(0)
    x = np.clip((raw - mean) / (std + 1e-5), -5.0, 5.0)[None].astype(np.float32)
    ts = pd.to_datetime(ctx["timestamps"])
    x_stamp = np.stack(
        [ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1
    ).astype(np.float32)[None]

    xt = torch.from_numpy(x).to(DEVICE)
    stt = torch.from_numpy(x_stamp).to(DEVICE)

    extra: dict = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "ctx_len": ctx_len,
        "pred_len": PRED_LEN,
        "seed": SEED,
        "d_model": model.d_model,
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "ff_dim": model.ff_dim,
        "s1_bits": model.s1_bits,
        "s2_bits": model.s2_bits,
        "learn_te": bool(model.learn_te),
    }

    with torch.no_grad():
        # --- tokenizer encode (half=True => two 10-bit subtokens) ---
        z = tokenizer.encode(xt, half=True)
        extra["token_s1"] = z[0].cpu().numpy().astype(np.int64)
        extra["token_s2"] = z[1].cpu().numpy().astype(np.int64)

        # --- tokenizer decode round-trip (exercises post_quant_embed, decoder, head) ---
        recon = tokenizer.decode(z, half=True)
        extra["tokenizer_recon"] = recon.float().cpu().numpy()

        # --- predictor decode_s1 over the full context ---
        s1_ids, s2_ids = z[0][:, :ctx_len].contiguous(), z[1][:, :ctx_len].contiguous()
        s1_logits, context = model.decode_s1(s1_ids, s2_ids, stt)
        extra["s1_logits_last"] = s1_logits[:, -1, :].float().cpu().numpy()
        extra["context_last"] = context[:, -1, :].float().cpu().numpy()

        # --- predictor decode_s2 conditioned on the greedy coarse token ---
        pre_arg = s1_logits[:, -1, :].argmax(-1, keepdim=True)
        extra["pre_argmax"] = pre_arg.cpu().numpy().astype(np.int64)
        s2_logits = model.decode_s2(context, pre_arg)
        extra["s2_logits_last"] = s2_logits[:, -1, :].float().cpu().numpy()

    cap.detach()
    return cap.tensors, extra


def dedupe(tensors: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Drop exact-duplicate captures, keeping the most specific name.

    Several hooks alias the same tensor (e.g. `Kronos.head` *is* `head.proj_s1`;
    `DependencyAwareLayer` returns `self.norm(...)`; `HierarchicalEmbedding`
    returns `fusion_proj(...)`). Captures are ranked by dotted-depth descending
    so `head__proj_s1` wins over `head`, then compared by content hash.
    """
    kept: dict[str, np.ndarray] = {}
    seen: dict[bytes, str] = {}
    aliases: dict[str, str] = {}
    for key in sorted(tensors, key=lambda k: (-k.count("."), k)):
        arr = tensors[key]
        digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).digest()
        if digest in seen:
            aliases[key] = seen[digest]
        else:
            seen[digest] = key
            kept[key] = arr
    return kept, aliases


# ---------------------------------------------------------------------------
# Golden CSVs (end-to-end deterministic predictions)
# ---------------------------------------------------------------------------
def gen_golden(name: str, spec: dict, model, tokenizer, df: pd.DataFrame) -> None:
    (FIX / "golden").mkdir(parents=True, exist_ok=True)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=spec["max_context"])
    for ctx_len in CTX_LENS:
        out_path = FIX / "golden" / f"{name}_ctx{ctx_len}_pred{PRED_LEN}.csv"
        expected_rows = PRED_LEN
        src = df.iloc[: ctx_len + expected_rows]
        if src.shape[0] < ctx_len + expected_rows:
            src = df
        ctx = src.iloc[:ctx_len]
        for rec, dtype in (("fp32", torch.float32),):
            set_seed(SEED)
            with torch.no_grad():
                pred = predictor.predict(
                    df=ctx[FEATURES].reset_index(drop=True),
                    x_timestamp=ctx["timestamps"].reset_index(drop=True),
                    y_timestamp=src["timestamps"].iloc[ctx_len : ctx_len + expected_rows].reset_index(drop=True),
                    pred_len=expected_rows,
                    T=1.0,
                    top_k=1,
                    top_p=1.0,
                    verbose=False,
                    sample_count=1,
                )
            out = pd.DataFrame(index=pd.Index(src["timestamps"].iloc[ctx_len : ctx_len + expected_rows].reset_index(drop=True)))
            for c in FEATURES:
                out[c] = pred[c].to_numpy()
            out.index.name = "timestamps"
            out.to_csv(out_path, float_format="%.9g")
            print(f"    golden {out_path.name}  ({dtype})")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", choices=sorted(MODELS))
    args = ap.parse_args()
    targets = sorted(MODELS) if args.all or not args.model else [args.model]

    FIX.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamps"])
    weights_index = json.loads((FIX / "weights.json").read_text()) if (FIX / "weights.json").exists() else {}

    for name in targets:
        spec = MODELS[name]
        print(f"[{name}] downloading/verifying weights ...")
        weights_index[name] = dump_weights(spec)
        set_seed(SEED)

        tokenizer = (
            KronosTokenizer.from_pretrained(spec["tokenizer"], revision=spec["tokenizer_rev"])
            .to(DEVICE)
            .eval()
        )
        model = (
            Kronos.from_pretrained(spec["model"], revision=spec["model_rev"]).to(DEVICE).eval()
        )
        print(f"    params: model={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
              f"tokenizer={sum(p.numel() for p in tokenizer.parameters())/1e6:.2f}M")

        gen_golden(name, spec, model, tokenizer, df)

        print(f"[{name}] capturing layer activations ...")
        tensors, extra = capture_layers(name, model, tokenizer, CTX_LENS[-1])
        n_raw = len(tensors)
        tensors, aliases = dedupe(tensors)
        print(f"    {n_raw} hooks -> {len(tensors)} unique tensors "
              f"({n_raw - len(tensors)} exact duplicates dropped: {aliases})")
        out_dir = FIX / "layers" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for k in list(out_dir.glob("*.npy")):
            k.unlink()
        for key, arr in sorted(tensors.items()):
            np.save(out_dir / f"{key.replace('.', '__')}.npy", arr)
        for key, arr in extra.items():
            if isinstance(arr, np.ndarray):
                np.save(out_dir / f"{key}.npy", arr)
        (out_dir / "index.json").write_text(
            json.dumps(
                {
                    "model": name,
                    "device": DEVICE,
                    "deduped_aliases": aliases,
                    "tensors": {k: list(v.shape) for k, v in sorted(tensors.items())},
                    "extra": {
                        k: (list(v.shape) if isinstance(v, np.ndarray) else v)
                        for k, v in extra.items()
                    },
                },
                indent=2,
            )
        )
        print(f"    wrote {len(tensors)} layer tensors -> {out_dir}")
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (FIX / "weights.json").write_text(json.dumps(weights_index, indent=2, sort_keys=True))
    print(f"\nfixtures written to {FIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())