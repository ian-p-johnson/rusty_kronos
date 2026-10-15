# Porting Kronos from Python/PyTorch to Rust (or Mojo) + GPU

**Status:** **implemented.** Phases 0–6 of the plan below are built and green; the port runs on `tch`/libtorch with no Python runtime. **Current head: `afeae9d`.** Read **§11 (implementation log)** for what actually shipped, the substrate decision, build recipe, live gate status and the one open performance question. Sections 1–10 are the original assessment and are kept as written; where later work superseded a recommendation it is flagged inline.
**Scope of study:** this repo at `67b630e`, plus `2508.02739v1.pdf` (AAAI 2026, "Kronos: A Foundation Model for the Language of Financial Markets")
**Measurements:** taken on the local machine (NVIDIA GeForce RTX 5070 Ti **Laptop**, 46 SMs, CC 12.0, 12 GB, torch 2.11.0+cu128). Projections to desktop RTX 5070 Ti / RTX 3090 are labelled as such.

---

## 0. Verdict (read this first)

| Question | Answer |
|---|---|
| Is it amenable to a Rust/Mojo port? | **Yes, strongly.** The entire GPU workload is **1,232 lines** (`model/module.py` 570 + `model/kronos.py` 662) built from *standard* PyTorch primitives. No custom CUDA kernels, no custom autograd on the active path, no flash-attn, no exotic ops, no FSDP/AMP. |
| What hands off to the GPU today? | Only `model/kronos.py` + `model/module.py`. All ~9,900 other lines of Python (examples, finetune, webui) contain **zero** device work — the only "GPU" line in the entire peripheral tree is the *string literal* `device="cuda:0"` passed to the predictor constructor. |
| Rust or Mojo? | **Rust**, for everything. Mojo only as an optional spike for 3–4 fused kernels. See §5. |
| GPU required? | The workload is GPU-bound at every useful batch size; it is pointless without one. |
| Realistic speedup | **≈4–6× end-to-end**, not 50×. See §6. The dominant lever is *tensor-core precision* (~3×), not the language. |
| Biggest surprise | **An exact KV cache is impossible.** Kronos re-encodes a rolling window with window-*relative* RoPE positions, so every cached key is invalidated every step. This removes the usual 10–100× LLM-decoding lever. Proven empirically in §4.3. |

**The single most important strategic finding:** ~60% of the achievable latency win is available *in Python* by enabling tensor cores and removing wasted passes (§6.4). A full port is justified by deployment properties (latency floor, process footprint, a real REST/streaming server, no Python runtime), not by raw speed alone.

---

## 1. Scope audit — what is actually Kronos

Total Python: **11,189 lines**, 27 files.

| Area | Files | LOC | GPU work? | Port verdict |
|---|---|---|---|---|
| **Core model** | `model/kronos.py`, `model/module.py` | **1,232** | **Yes — all of it** | **PORT THIS** |
| Tests | `tests/test_kronos_regression.py` + data | 140 | inherits | **PORT as the parity harness** |
| Finetune (qlib) | `finetune/*` | 1,292 | training only | Keep Python (§7.3) |
| Finetune (csv) | `finetune_csv/*.py` | 1,455 | training only | Keep Python (§7.3) |
| Examples | `examples/*.py`, `examples/yuce/*` | 4,698 | **none** | Keep Python (§7.2) |
| WebUI | `webui/app.py`, `run.py` | 797 | **none** | Optional; reimplement the ~700-line Flask/Plotly surface if you want a Rust server (§7.2) |

Corroborating evidence: `examples/`, `finetune*/` and `webui/` import **zero** `torch` symbols. The peripheral files are pandas/matplotlib/akshare/Flask/Plotly glue plus a hand-tuned post-hoc heuristic rescaler in `examples/prediction_new.py:702-825` and `examples/prediction_new_GUI.py:997-1139` that is **not part of Kronos** and uses `np.random` (a reproducibility hazard — recommend dropping it, see §8).

Repo hygiene issues found while auditing (all worth fixing before any port):
- `examples/get_akshare_date_2024-2025_x.py` is a ~92% duplicate of `examples/get_date_new.py`; `prediction_new_GUI.py` is ~85% duplicated from `prediction_new.py`.
- Undeclared dependencies: `akshare`, `baostock`, `requests`, `tkinter`, `pyarrow` (needed by `webui/app.py` via `pd.read_feather`).
- `webui/app.py:31` imports `numpy` but never uses it.
- **bf16 output is broken in the reference implementation**: `model/kronos.py:466` calls `z.cpu().numpy()`, which raises `TypeError: unsupported ScalarType BFloat16`. Any bf16 path must cast first (reproduced; see §6.5).

---

## 2. Architecture (from the paper + verified against real HF configs)

Verified against the downloaded `config.json` files — note the paper says "3 encoder/decoder layers" for the tokenizer but the released `Kronos-Tokenizer-base` actually uses **4+4 with `ff_dim=512`, `group_size=4`**. Trust the config, not the paper.

**Stage 1 — K-line tokenizer** (`KronosTokenizer`, 3.96 M params, d=256):
`Linear(6→256)` → 4 × TransformerBlock → `Linear(256→20)` → **BSQuantizer** → `Linear(20→256)` → 4 × TransformerBlock → `Linear(256→6)`.

**Stage 2 — autoregressive predictor** (decoder-only):

| | Kronos-mini | Kronos-small | Kronos-base | Kronos-large |
|---|---|---|---|---|
| layers | 4 | 8 | 12 | 18 |
| d_model | 256 | 512 | 832 | 1664 |
| d_ff | 512 | 1024 | 2048 | 3072 |
| heads | 4 | 8 | 16 | 32 |
| params (measured) | 4.11 M | **24.74 M** | 102.20 M | 499.02 M |
| max context | 2048 | 512 | 512 | 512 |

Param counts were reproduced analytically from the architecture and match the paper's Table 1 (24.7 M / 102.3 M / 499.2 M) to <0.1%.

**Tokenisation scheme** (the paper's central trick, and the thing most likely to break a port):
- Each D=6 OHLCVA vector → L2-normalised → **BSQ**: sign function → a 20-bit ±1 code → scaled by `1/√20`.
- The 20 bits are split into two 10-bit **subtokens** (`s1_bits = s2_bits = 10`) → vocab **1,024** each.
- Subtokens are predicted *sequentially*: `p(b_t) = p(b_t^c | b_<t) · p(b_t^f | b_<t, b_t^c)`.
- `b_t^c` is fed into a **cross-attention "dependency-aware layer"** whose query is `emb_s1(b̂_t^c)` and whose key/value is the transformer hidden state — so `decode_s2` is a genuine second pass.
- Measured codebook usage: 97.66 % coarse, 85.25 % fine (paper Table 11).

This factorisation is why `Kronos-base` is 102 M and not 1.8 B (paper Table 12: n=1 would be 1,842 M params vs 102.3 M at n=2) — and why inference costs **2 sequential steps per K-line**.

---

## 3. What hands off to the GPU today

Everything happens inside `KronosPredictor.predict` → `KronosPredictor.generate` → `auto_regressive_inference` (`model/kronos.py:389-469`):

1. CPU: z-score normalise, clip to ±5, build 5 time features (`minute/hour/weekday/day/month`).
2. CPU→GPU: three `np.float32` arrays.
3. GPU: `tokenizer.encode(x, half=True)` — one 4-layer forward over the context.
4. GPU: **a Python `for` loop of `pred_len` iterations**, each doing:
   - `model.decode_s1(...)` — **a full forward pass over the entire window**, then discarding every position but the last.
   - `sample_from_logits` — divide by T, top-p (`torch.sort` + `cumsum` + `scatter` over a 1,024-wide vocab), softmax, `torch.multinomial`.
   - `model.decode_s2(...)` — cross-attention over the whole window.
   - two `torch.roll` calls to shift the ring buffer.
5. GPU→CPU: `tokenizer.decode(...)` over the whole 512-token window, `np.mean` over samples, only then the last `pred_len` rows are used.
6. CPU: de-normalise, build a DataFrame.

**Training** (`finetune/`, `finetune_csv/`) is plain `torchrun` + DDP + NCCL + gradient accumulation. No AMP, no FSDP, no `accelerate`, no `apex`, no `triton`, no `flash_attn`, no custom CUDA. Both trees are ~90 % duplicates of each other.

---

## 4. Portability assessment

### 4.1 What is easy

> **Superseded (Phase 6).** The `candle-core` weight-loading assumption held for Phases 1–5 and was then retired: the final engine uses `tch::Tensor::read_safetensors`. The rest of this section still stands.

- **Weight loading.** Weights are `safetensors` + `config.json` on the HF Hub. Rust's `candle-core` has a native `VarBuilder::from_mmaped_safetensors` + `hf-hub` client — a direct, zero-friction match. `config.json` is 11 scalar keys per model.
- **The architecture.** `Linear`, `Embedding`, `RMSNorm`, `SiLU`, RoPE, `scaled_dot_product_attention` — all first-class in `candle-core`/`cudarc` (cuBLAS, cuBLASLt, cuDNN) or trivially hand-written.
- **Attention.** `F.scaled_dot_product_attention` with `is_causal=True` → cuDNN/cuBLAS or a FlashAttention kernel. No masking subtleties: the only mask used is the optional `key_padding_mask`, which **`auto_regressive_inference` never passes**.
- **einops.** Used in exactly 3 places (`module.py:135,176,193`), all trivially unrolled.
- **Sampling.** `top_k_top_p_filtering` + `sample_from_logits` is 50 lines; becomes a single fused kernel.
- **The tokenizer's inference path.** `BSQuantizer` at inference is only: `F.normalize` → `sign` → `×1/√20` → `bits_to_indices` (`model/kronos.py:142-159`). ~20 lines.

### 4.2 What is awkward

- **`DifferentiableEntropyFunction` (`model/module.py:10-32`)** is a hand-written `torch.autograd.Function`. **You do not need to port it.** `BinarySphericalQuantizer.__init__` hardcodes `soft_entropy=True` (`module.py:42`), so `forward` always takes the `soft_entropy_loss` branch and `codebook_entropy` is dead. A port should assert this and delete the dead path.
- **`DifferentiableEntropyFunction` is also numerically suspicious** if anyone ever flips that flag: `torch.zeros(2**K)` with `K=20` allocates 1 M elements and `scatter_reduce`s a batch into it; the `backward` indexes `grad_array[zi]` with `zi ∈ [0, 2^20)`. It is training-only and unreachable. Flag it upstream; do not port it.
- **Training-only BSQ surface.** If you ever port training you need `soft_entropy_loss`: a grouped einsum against `group_codebook` (2^4 = 16 entries × group_size 4), then softmax/sigmoid entropies. That is portable but it is the fiddliest ~40 lines in the repo.
- **Steady-state ergonomics.** `sample_from_logits` has a latent bug when `top_k is None` and `top_p is None` (the `if top_k > 0` inside `top_k_top_p_filtering` would `TypeError`). The caller always passes both, so it is unreachable today; the port should normalise the arguments once at the API boundary.

### 4.3 What is fatal — and it changes the whole performance story

**There is no KV cache, and parity forbids adding one.**

`auto_regressive_inference` keeps a ring buffer and rolls it by −1 each step (`model/kronos.py:451-454`), re-running the **entire** window through **every** transformer layer each step. RoPE positions are always re-assigned `0..L-1` (`module.py:296-301`), i.e. **window-relative, not absolute**.

At step *i*, slot *j* holds token `u_{i-L+1+j}` attending to slots `0..j`.
At step *i+1*, slot *j* holds token `u_{i-L+2+j}` attending to slots `0..j`.

The **prefix changes every step**, so every cached key and every hidden state is stale. This is not a subtle RoPE argument — it was measured directly. Tracking one physical token across two consecutive steps:

```
step 0 -> 1   token# 294  slot   4 ->   5   max|d hidden|=1.3742e+00   max|d logits|=6.4789e+00
step 0 -> 1   token#  81  slot  99 -> 100   max|d hidden|=1.0982e+00   max|d logits|=1.1538e+01
step 0 -> 1   token# 943  slot 299 -> 300   max|d hidden|=1.8773e+00   max|d logits|=8.1355e+00
step 0 -> 1   token# 907  slot 510 -> 511   max|d hidden|=7.1180e-01   max|d logits|=9.5653e+00
```

Hidden states move by O(1). Caching is invalid.

**Why this is architecturally sensible** (and why you should not "fix" it casually): the model is deliberately a *fixed-window, position-recycled* decoder — `max_context` is capped at 512 precisely so that positions cycle and the distribution never sees position drift. Switching to absolute RoPE would make a KV cache legal and give ~10–50×, but **it is a different model**. It would need retraining and a fresh accuracy evaluation. Out of scope for a parity port. Recorded as future work in §9.

**Consequence for the port:** per-step cost is Θ(batch × L × d²) and that is a hard floor. Per-step FLOPs, Kronos-small, L=512, d=512, d_ff=1024, 8 layers:

```
q,k,v      3 × 2·(B·512)·512·512
attn       2 × 2·B·8·512²·64
out        2·(B·512)·512·512
ffn        3 × 2·(B·512)·512·1024
        × 8 layers  + proj_s1 (2·B·512·1024) + dep-layer (3 × 2·B·512·512²) + proj_s2
   =  238 GFLOP per step at B = 10          (validated against the profiler: predicted 24.8 ms, measured 27.5 ms)
```

---

## 5. Rust vs Mojo

> **Outcome (recorded for the record).** Rust won, as recommended — but not via `candle`. The substrate is `tch` (libtorch bindings); see §11.2. Mojo was never spiked, because the §7 Phase-5 finding (candle's *kernels*, not the language, were the ceiling) made the Rust-plus-libtorch route strictly better than anything a Mojo kernel spike could have delivered. The `cutile-rs`/`cuda-oxide` route was evaluated and rejected: they are NVIDIA's experimental Rust kernel DSLs — excellent for hand-written kernels, useless as a drop-in for ATen's tuned GEMM/SDPA/softmax library.

Both are viable. They are **not** equivalent, and the deciding factor is not raw kernel speed.

| | **Rust** | **Mojo** |
|---|---|---|
| Language state (2026) | Stable, mature | 1.1.0 stable (Sep 2026), **fully open source Apache-2.0** (Aug 2026) |
| ML framework | `candle-core` (inference-first, cuBLAS/cuDNN), `burn`+`cubecl` (autodiff), `cudarc` (safe CUDA driver/NVRTC/cuBLASLt/NCCL bindings, CUDA 11.4–13.4) | MAX (`nn`, `kernels`, autotuned `def` GPU functions) |
| Safetensors + HF Hub | **Native** (`VarBuilder`, `hf-hub`) — the weights format matches exactly | **Hand-written** |
| Reverse-mode autodiff | Yes (`burn`, `candle`) | **Not mature** — training port impractical |
| Fused-kernel authoring | `cuTile` (JIT CUDA in Rust) or `cudarc` + NVRTC + `cu` | **Best-in-class** — `Tile`/SIMD + compile-time autotune is Mojo's strongest suit |
| Serving / IO ecosystem | axum, tokio, serde, polars, csv, parquet | **None** — you write HTTP, CSV, JSON from scratch |
| Measured kernel perf | cuBLAS for GEMMs (near-optimal here) | Competitive on memory-bound; ~10–20 % gaps on fast-math compute-bound kernels on NVIDIA (arXiv 2509.21039) |
| Governance / risk | Multi-vendor, Apache/MIT | Now under **Qualcomm** (Modular acquired Jun 2026); ABI explicitly **not** stable |

**Recommendation: Rust for everything.**

1. The decisive factor is **autodiff**. Mojo has no production-grade reverse mode, so a Mojo port would be *inference-only* — which is fine for the primary goal, but it means Mojo cannot be the answer if fine-tuning ever needs to move too.
2. At batch ≥ 10 the workload is **GEMM-bound and already near-optimal** (§6.3): the profiler shows cutlass kernels running at ~81 % of the part's bf16 peak. Mojo cannot beat cuBLAS there. Mojo's advantage — fast fused elementwise kernels — is real but confined to the ~40 % of device time that is *not* GEMM, and only at small batch.
3. Safetensors + HF Hub + a REST server + Parquet are all one-crate solutions in Rust and all hand-written in Mojo. For a quant stack that has to run as a service, that difference dominates.
4. A third-party experience report (arXiv 2609.25008) documents five `candle` 0.11 training-path defects and a `burn` 0.20 backward pass measured at ~3 % of theoretical matmul throughput. **Validate this yourself** before committing — it is one author's report, not a benchmark suite. It reinforces the recommendation: *use candle for inference only, and keep training in Python.*

**When Mojo would win:** if §7 Phase 5 shows the fused elementwise kernels are the remaining bottleneck at your target batch size and you are willing to own the IO layer. Realistically: run it as a 2-week spike on *one* kernel (fused RMSNorm+RoPE, or fused top-p + multinomial) and compare against the Rust `cuTile` version. Do not port the project to Mojo on the strength of that.

---

## 6. Performance

All "Measured" rows are on the **local RTX 5070 Ti Laptop** (46 SMs). Projections to the desktop parts are **scaled estimates**, not measurements — see §6.6 for the method and the uncertainty.

### 6.1 Measured baseline (current code, fp32, torch 2.11)

| Model | ctx | pred_len | N (= batch) | End-to-end | Per step |
|---|---|---|---|---|---|
| Kronos-small | 400 | 120 | 1 | **538 ms** | 4.49 ms |
| Kronos-small | 400 | 120 | 5 | **1,780 ms** | 14.8 ms |
| Kronos-small | 512 | 120 | 10 | **3,822 ms** | 31.8 ms |
| Kronos-mini | 400 | 120 | 10 | **670 ms** | 5.59 ms |

Component split (Kronos-small, B=10, ctx 512):

| Component | Time | Share |
|---|---|---|
| `tokenizer.encode` | 3.2 ms | 0.1 % |
| **AR loop (120 × 27.2 ms)** | **3,263 ms** | **85 %** |
| `tokenizer.decode` | 3.3 ms | 0.1 % |

### 6.2 Measured per-step cost by precision (L = 512)

| Batch | fp32 | bf16 | fp16 | bf16 + CUDA graph |
|---|---|---|---|---|
| B = 1 | 4.23 ms | 2.17 ms | 2.13 ms | **1.47 ms** |
| B = 10 | 27.45 ms | **9.30 ms** | 9.60 ms | 9.07 ms |
| B = 64 | 196.96 ms | 74.97 ms | 76.47 ms | 76.32 ms |
| B = 256 (full-window) | 863.6 ms | 312.9 ms | — | — |

Achieved throughput at B=10: **9.6 TFLOP/s fp32 → 28.3 TFLOP/s bf16** (≈3.0×; bf16 tensor peak for this part ≈ 57.7 TFLOP/s, so bf16 runs at ~49 % of peak — respectable for FP32-accumulate GEMMs).

### 6.3 Where the time actually goes (profiler, B=1, bf16)

```
366 CUDA kernels per step, 1.51 ms device time
  cutlass GEMMs            51 kernels    567 us    37.6 %
  flash attention          10 kernels    133 us     8.8 %
  elementwise / reduce    285 kernels    810 us    53.6 %
```

Two conclusions:

1. **At B ≥ 10 the step is GEMM-bound and near-optimal.** The GEMMs are already at ~81 % of the part's peak. A hand-written engine using cuBLAS cannot beat this; the port's upside here is ~5 %, not 5×.
2. **At B = 1, 62 % of device time is elementwise/reduce glue** spread over 285 kernels. This is where a Rust engine with fused kernels wins — and it is the *latency-critical* case (single symbol, single forecast). CUDA-graph replay alone only gives 1.47× because it removes the ~0.6 ms CPU launch gap but still pays GPU-side scheduling for 366 kernels. **Fusing 366 → ~80 kernels is the real lever.**

### 6.4 The honest speedup budget

| Lever | Where | Gain |
|---|---|---|
| fp32 → **bf16/fp16 with fp32 accumulate** | any batch | **~3.0× (measured)** |
| Fuse 285 elementwise/reduce kernels → ~20 | B = 1 | ~1.5–2.5× |
| Fuse + CUDA-graph the step | B = 10 | ~1.1–1.4× |
| Truncated tokenizer decode (only `pred_len` rows are used, but 512 are decoded — and the decoder is causal over a *static* sequence, so a prefix KV cache **is** valid here) | per forecast | −3 to −4 ms absolute |
| Cache the encoder output across rolling-window calls (contexts overlap heavily in production) | per forecast | −3.3 ms absolute |
| Shorter lookback (cost is ~linear in L for L ≳ 256) | user knob | up to 2× |
| **Realistic total vs the fp32 baseline** | | **≈4–6×** |

There is **no 10× lever left** beyond this without changing the model (§4.3).

### 6.5 Precision vs parity

Parity was verified two ways.

**(a) Greedy decoding is reproducible and already tested.** The repo's own regression test uses `top_k=1, top_p=1.0`, which degenerates `sample_from_logits` to argmax. Run on GPU against the committed golden CSVs:

| ctx_len | max abs err | max rel err | passes `rtol=1e-5` |
|---|---|---|---|
| 512 | 6.25e-01 | **1.205e-06** | ✅ |
| 256 | 6.875e-01 | **2.054e-06** | ✅ |

Run-to-run output with a fixed seed is **bitwise identical**. So there is a working, committed, deterministic parity harness. Build the Rust port against it from day one.

**(b) Reduced precision diverges slowly, not catastrophically.** Single forward pass, identical fp32 inputs, B=10, 512-token window:

| | s1 logits max rel err | s2 logits max rel err | top-1 disagreement |
|---|---|---|---|
| **fp16** (fp32 accum) | 4.75e-04 | 6.99e-04 | **0 %** |
| **bf16** | 3.63e-03 | 4.81e-03 | 0 % |

fp16 is **~7× more accurate than bf16** here (10 vs 8 mantissa bits) and this model has no dynamic-range problem — inputs are z-scored and clipped to ±5, so fp16's limited exponent range is irrelevant. **Prefer fp16 over bf16.**

Caveats that must be measured in Phase 4, not assumed:
- This is **one** step. Errors compound across 120 autoregressive steps fed back into the decoder. Measure full-trajectory divergence before committing.
- **Stochastic sampling cannot be bit-exact.** `torch.multinomial` consumes cuRAND Philox stream. A port must define parity for stochastic mode as *distributional* (χ² on token frequencies over many seeds + the repo's own MSE metric), not bitwise. Optionally reimplement torch's exact Philox offset scheme for bit-exact sampling — feasible, fragile, and only worth it if you need to reproduce a specific backtest.
- `model/kronos.py:466` (`z.cpu().numpy()`) **crashes on bf16 output** — reproduced. Fix in the reference before using it as a comparison baseline.

### 6.6 Projection to desktop RTX 5070 Ti / RTX 3090

Scaling factors used (desktop 5070 Ti has 70 SMs vs 46; the part also clocks higher):

| | RTX 5070 Ti Laptop (measured) | RTX 5070 Ti (desktop) | RTX 3090 |
|---|---|---|---|
| SMs | 46 | 70 (×1.52) | 82 (×1.78) |
| FP32 peak | ~23.7 TFLOP/s | 43.94 (×1.85) | 35.58 (×1.50) |
| BF16/FP16 tensor, FP32-accum (dense) | ~57.7 | 87.88 (×1.52) | **71.16** (×1.23) |
| TF32 tensor (dense) | — | 43.94 | **35.58** |
| FP16 tensor, FP16-accum (dense) | — | 175.8 | 142.3 |
| Memory bandwidth | ~700 GB/s | 896 GB/s | 936 GB/s |

Note the **RTX 3090 is ~23 % *slower* than the desktop 5070 Ti for this workload** in bf16 tensor terms (Ampere consumer halves FP32-accumulate rate), despite having more SMs. Its only real advantages here are 24 GB VRAM and 936 GB/s bandwidth. For Kronos (a ≤100 MB model) VRAM is irrelevant.

**Projected end-to-end latency, Kronos-small, ctx 512, pred_len 120** (±30 %):

| Scenario | N=1 | N=10 | Batch of 256 series |
|---|---|---|---|
| **Current PyTorch fp32** — 5070 Ti | ~400–470 ms | ~1.9–2.4 s | ~90–110 s |
| **Current PyTorch fp32** — 3090 | ~430–510 ms | ~2.3–2.9 s | ~110–130 s |
| **PyTorch + fp16** (no port) — 5070 Ti | ~170–200 ms | ~0.55–0.70 s | ~25–30 s |
| **Ported Rust, fp16, fused** — 5070 Ti | **~70–90 ms** | **~0.45–0.60 s** | ~15–20 s |
| **Ported Rust, fp16, fused** — 3090 | **~85–110 ms** | **~0.55–0.70 s** | ~19–25 s |

**Recommended operating point for a live 5-minute-bar strategy** (ctx 512, **pred_len 48**, N=10): ~**180–240 ms** ported. For a 1-minute bar the arithmetic is unchanged — it is `pred_len` that drives cost, not frequency.

Interpretation:
- **Latency floor ≈ 50–100 µs of pure GPU work per step** is *not* reachable, because each step is a full 512-token forward (238 GFLOP at B=10). What you can reach is near the GEMM roofline.
- The port takes you from "seconds" to "tens of milliseconds" for single-symbol, and from "tens of seconds" to "seconds" for large batches. That is the real deliverable.

---

## 7. Staged plan

Seven phases. Each has a hard exit criterion. **Do not start Phase N+1 until Phase N's criterion is met.** Estimated effort assumes one engineer familiar with CUDA and Rust; adjust to taste.

### Phase 0 — Freeze the reference (1–2 days) ✅ *highest value per hour in this entire document*

Before writing a line of Rust, make the Python baseline trustworthy and fast enough to be a fair comparator.

1. Fix the bf16 `.numpy()` crash (`model/kronos.py:466`).
2. Add a proper benchmark harness: `bench/` script sweeping `{model} × {ctx} × {pred_len} × {sample_count} × {batch} × {dtype}`, emitting JSON. Record the numbers in §6.1 as the committed baseline.
3. Add deterministic-mode parity tests to CI (they already exist — just run them on GPU too; `tests/test_kronos_regression.py:34` hardcodes `DEVICE="cpu"`).
4. **Free Python's own 3×**: wrap `auto_regressive_inference` in `torch.autocast("cuda", torch.float16)` with fp32 accumulation, behind a flag. Measure.
5. Truncate the final `tokenizer.decode` to the last `pred_len` positions using a prefix KV cache (valid — the decoder is causal over a static sequence).
6. Memoize `tokenizer.encode` on the context tensor.

**Exit:** committed, reproducible fp32 golden outputs + a measured fp16 number + a green benchmark. **If Phase 0 alone meets your latency requirement, stop here** — you have ~60 % of the achievable win for ~5 % of the effort.

### Phase 1 — Rust skeleton + weight loading (2–3 days)

Workspace: `rust/` (cargo workspace, `crates/kronos-core`, `crates/kronos-cuda`, `crates/kronos-cli`).

- `candle-core` with `cuda` + `cudnn` features; `hf-hub` for downloads; `safetensors` mmap.
- `KronosConfig` mirroring `config.json` (11 scalars) — **serde with `#[serde(default)]`**, validated at load time (e.g. `d_model % n_heads == 0`, `max_context ≤ model max`).
- Loader that maps `safetensors` keys → struct fields and **fails loudly on any unexpected or missing key**. A silent partial load here is the single most likely cause of a mysterious parity failure.
- Round-trip test: load in Python, load in Rust, assert every tensor is bitwise equal.

**Exit:** a CLI prints per-tensor checksums matching Python; a Rust unit test proves it in CI.

### Phase 2 — Forward-parity of the static pieces (3–5 days)

Port, in order, verifying **bitwise-or-1e-6 parity at each step**:

1. `RMSNorm` → fused kernel
2. `RotaryPositionalEmbedding` (incl. `_rotate_half` and the cos/sin cache invalidation at `module.py:294`) → fused into the QKV epilogue
3. `FeedForward` (SwiGLU: `w2(silu(w1(x)) * w3(x))`) → **one fused GEMM** for `w1|w3` (concatenate the weights at load time — a free 1.3× on the FFN, which is 60 % of step FLOPs)
4. `TransformerBlock`
5. `HierarchicalEmbedding` (`s1_emb * √d`, `s2_emb * √d`, concat, `fusion_proj` → one GEMM)
6. `TemporalEmbedding` (5 lookups, summed)
7. `RMSNorm` + `DualHead` + `DependencyAwareLayer` (single-query cross-attention)

Parity test: feed the 512-token regression input, compare hidden states and logits layer-by-layer against Python dumps.

**Exit:** max relative logit error < 1e-5 in fp32 against `tests/data/regression_input.csv`. This is where parity bugs are cheapest to find — do not proceed past a single failing layer.

### Phase 3 — The tokenizer (2–3 days)

- **Inference path only:** `F.normalize` → `sign` → `×1/√20` → `bits_to_indices`, and `indices_to_bits` for decode. `group_size=4`, `codebook_dim=20`, `s1_bits=s2_bits=10` — verified from the released config, **not** the paper's "group_size 9 / 3 layers".
- Bit-pack to `u16` (`(bits * 2^arange).sum(-1)`) — note this is a **weighted sum, not a shift**; for 10 bits it coincides with `sum(bits << i)` and is bit-identical, but keep it explicit.
- Confirm identical `(s1, s2)` index tensors vs Python for every position of the regression input.

**Exit:** identical token ids on the full regression input; round-trip `decode` within 1e-6 of Python.

### Phase 4 — Deterministic full pipeline + precision decision (3–5 days) ← **the parity gate**

- Port the exact ring-buffer/roll semantics (`model/kronos.py:408-454`), including the `current_seq_len <= max_context` warm-up branch. Getting this wrong is silent.
- Port `top_k_top_p_filtering` + `sample_from_logits`; assert `top_k=1, top_p=1.0` → argmax → bitwise-identical output.
- Run `tests/test_kronos_regression.py`'s two golden CSVs in Rust.
- **Then run the fp16 experiment**: measure full 120-step trajectory divergence (max abs/rel on OHLCVA, token flip count, and MSE against ground truth using the repo's own `MSE_EXPECTED = [0.008979, 0.003741]`).

**Exit:** fp32 greedy parity passes `rtol=1e-5` on both golden CSVs, **and** a written fp16-vs-fp32 divergence report decides fp16-accept / fp16-reject. This is a decision gate, not a formality — do not defer it.

### Phase 5 — Performance engineering (5–10 days)

In priority order, measuring after each:

1. **Fused QKV** (one GEMM instead of three) and **fused SwiGLU** (one GEMM for `w1|w3`).
2. **Fused RMSNorm + RoPE** as a single epilogue kernel.
3. **CUDA graph capture of one AR step**, keyed on `(batch, dtype)` — re-instantiate on shape change. Expect ~1.1–1.4× at B=10, ~1.5× at B=1.
4. **Fused sampling**: single-kernel top-p + categorical sample (softmax → threshold → prefix-sum → one draw), eliminating the `sort`/`cumsum`/`scatter` chain and the host sync.
5. **Persistent megakernel** for the B=1 case (optional, only if Phase 6 says B=1 matters). Target: 366 kernels → ~80.
6. Persistent staging buffer + pinned host memory so the whole 120-step loop never leaves the device (currently `preds.cpu().numpy()` forces one sync per call).

**Exit:** documented speedup per item; measured on both target GPUs. Kill any item that does not pay for itself.

### Phase 6 — Production surface (5–8 days)

- `axum` REST server reproducing `webui/app.py`'s contract (`/api/load-data`, `/api/predict`, `/api/available-models`, …). Keep the Plotly JSON shape identical so the existing front-end works unchanged.
- Request batching / micro-batching with a bounded queue — this is where a large batch actually wins (§6.4).
- Graceful degradation: OOM → smaller batch; missing GPU → CPU path (use `candle`'s CPU backend for tests).
- Structured tracing (the `nvtx` feature in `cudarc`).

**Exit:** `load_data → predict` round-trip through the Rust server matches Python's output on the regression fixture.

### Phase 7 — Optional: training (10–20 days) — **only if justified**

Only port training if you have a concrete need. Measured reasoning: Kronos-base fwd+bwd at batch 32 × 512 tokens ≈ 9 TFLOP/step → ~0.4 s/step on either GPU, i.e. ~12–15 min/epoch. **PyTorch is already adequate.** If you do port:

- Use **DDP over NCCL only** — matches the current code (no FSDP, no AMP, no activation checkpointing anywhere in the repo).
- You do **not** need `DifferentiableEntropyFunction` (§4.2). Assert `soft_entropy == true`.
- Predictor training needs only `collect_metrics=false` encode + two cross-entropies. **Precompute and cache the tokenized dataset** — the reference re-runs the tokenizer forward *every* step (`finetune/train_predictor.py:101`); caching alone is a large win.
- Reproducibility hazards to settle first (all pre-existing, see §8): the two finetune trees disagree on `set_epoch_seed`, on AdamW `betas` (omitted in `finetune_csv/finetune_tokenizer.py`), and on normalisation window (lookback-only vs full-window).

---

## 8. Parity definition (settle this before writing code)

"Parity" is ambiguous and the answer changes the design. Adopt this definition explicitly:

| Mode | Definition | How to verify |
|---|---|---|
| **Deterministic (greedy)** | `top_k=1, top_p=1.0` → argmax. | **Bitwise or `rtol=1e-5`** vs `tests/data/regression_output_{256,512}.csv`. Already passes on GPU at 1.2e-6. |
| **Stochastic** | Same distribution, different RNG stream. | Distributional: χ² / KL on token frequencies over ≥1,000 seeds, plus the repo's own MSE targets. **Do not attempt bitwise.** |
| **fp16 tolerance** | Bounded divergence, measured not assumed. | Full 120-step trajectory; report max abs/rel OHLCVA error, token-flip count, and MSE vs ground truth. Gate at Phase 4. |
| **Training** | Not applicable. | Reproducibility is already inconsistent across the two finetune trees — fix the Python first, then match. |

Pre-existing Python behaviours to decide on (do not silently inherit them):
- `finetune/dataset.py:95` — `__getitem__` ignores `idx` and RNG-draws, so `DistributedSampler`'s sharding is largely defeated.
- `finetune/dataset.py:109-113` normalises on the **lookback slice only**; `finetune_csv/finetune_base_model.py:125` normalises over the **full window**. Train/test skew.
- `finetune_csv/finetune_tokenizer.py` omits `betas` from AdamW → silently uses (0.9, 0.999) instead of the configured (0.9, 0.95).
- `finetune/train_tokenizer.py:131-148` — gradient accumulation with **no `no_sync` context**, so it all-reduces every micro-batch (correctness-neutral, pure waste).
- `examples/prediction_new*.py` apply a hand-tuned `adjustment_factor ∈ [0.85, 1.15]` on top of model output, using `np.random`/`np.polyfit`. **Not Kronos. Recommend dropping rather than porting.**

---

## 9. Future work (explicitly out of scope)

1. **Absolute RoPE positions → a legal KV cache.** Would cut per-step cost from Θ(L) to Θ(1) and deliver a genuine 10–50×. Requires retraining and a fresh accuracy evaluation. This is the only route to that class of speedup, and it is a research project, not a port.
2. **Subtoken factorisation `n`.** The paper's Table 12 shows n=2 → 4 cuts vocab params by only 3.2 M while *doubling* sequential steps per token. n=2 is right; n=3+ is a latency regression. Don't touch.
3. **CUDA Graphs over the whole 120-step loop** rather than one step. Only viable once Phase 5 has cut the kernel count substantially; otherwise the graph is enormous for a ~1 ms/step saving.
4. **Mojo spike** on 1–2 fused kernels (`Tile` autotuned RMSNorm+RoPE, and fused top-p sampling) to settle §5 empirically.
5. **Upstream fixes.** The `top_k is None` TypeError in `sample_from_logits`; the bf16 `.numpy()` crash; `DifferentiableEntropyFunction`'s 2^20 scatter buffer; the three duplicated finetune trees. Small, and they help the Python reference too.

---

## 10. Appendix — methodology and raw data

**Environment.** torch 2.11.0+cu128, numpy/pandas/einops/huggingface_hub/safetensors, CUDA 13.1 driver.
**Hardware.** NVIDIA GeForce RTX 5070 Ti **Laptop GPU**, 46 SMs, CC 12.0, 11.5 GiB, measured stream bandwidth ~1.04 TB/s.
**Models.** `NeoQuasar/Kronos-small` rev `901c26c1…`, `NeoQuasar/Kronos-mini` rev `f4e68697…`, `NeoQuasar/Kronos-Tokenizer-base` rev `0e011738…`.

**Param counts measured from loaded modules:**

| | total | non-embedding | non-emb weight bytes @fp32 |
|---|---|---|---|
| Kronos-mini | 4.11 M | 3.58 M | 14 MB |
| Kronos-small | 24.74 M | 23.69 M | 95 MB |
| Tokenizer-base | 3.96 M | 3.96 M | 16 MB |

**Tokenizer passes** (ctx 512, fp32): encode/decode = 1.01/0.93 ms at B=1; 3.34/4.27 ms at B=10; 28.1/27.9 ms at B=64; 112.8/119.1 ms at B=256.

**Parity (greedy, fp32, GPU):** ctx 512 → max rel 1.205e-06 ✅; ctx 256 → max rel 2.054e-06 ✅; run-to-run bitwise identical with fixed seed.

**Precision divergence** (single forward, B=10, ctx 512, identical fp32 inputs):
fp16 → s1 4.745e-04, s2 6.988e-04, 0 % top-1 flip. bf16 → s1 3.632e-03, s2 4.806e-03, 0 % top-1 flip.

**KV-cache invalidation** — hidden-state delta for the same physical token across consecutive AR steps: 0.71–1.88 (hidden), 6.5–11.5 (logits). See §4.3.

**Uncertainty on projections.** Desktop-vs-laptop scaling is derived from SM count × clock, not measured — budget **±30 %**. The one assumption worth re-checking on real hardware is the achieved GEMM fraction (~81 % at B=1, ~49 % at B=10 of bf16 peak); if the desktop part sustains a higher fraction, Phase 5 items 1–3 will pay less than estimated and item 5 will pay more.

**Reproducing these numbers.** The benchmark scripts used for this assessment (`bench_kronos.py`, `bench_kv.py`, `bench_opt.py`, `prof_kernels.py`, `verify_parity.py`, `prec3.py`, `kvcache_proof.py`) are small and self-contained; fold them into `bench/` as the Phase 0 deliverable.

---

## 11. Implementation log (live status)

Everything below is the *as-built* record. §1–10 are the plan; this is what happened.

### 11.1 Commit map

One commit per phase, linear history, local fork `ian-p-johnson/rusty_kronos`, branch `master`. Nothing pushed.

| Commit | Phase | Content |
|---|---|---|
| `4c0fccf` | 0 | Freeze the PyTorch oracle: golden-CSV generation, probe dumps, weight manifest, benchmark harness (`parity/`), `frozen.lock` + `check_frozen.sh` |
| `4d73673` | 1 | Cargo workspace, `KronosConfig` (11 scalars, serde), **strict** safetensors loader (fails loudly on any missing/unexpected key), CLI `info`/`manifest` |
| `b9be690`, `46afc7b`, `6516d1f` | 2a/2b | Low-level ops with per-op parity tests; then the full model maths (RMSNorm, RoPE, SwiGLU, attention, embeddings, dual head, dependency layer) with per-stage parity on small/base/mini |
| `a5ee1f1` | 4 | AR ring-buffer loop, `top_k`/`top_p` sampler, golden gate, fp16 divergence report |
| `2f6a4f8` | 5.1 | Batch-throughput diagnostics; sorted-space sampler (no scatter); causal-mask cache — 21.4 s → ~14.5–16.3 s |
| `7de8342` | 5.2 | Custom online-softmax attention kernel via Candle's custom-CUDA API (`.cu` → NVRTC → PTX): 8.3 ms → 2.4 ms per attention call |
| `2ad77fd` | 5.3 | GEMM-algo investigation, per-step/per-op probes, bit-identical GEMM recipe, step anatomy |
| `afeae9d` | 6 | **Substrate swap: candle → `tch`/libtorch**, whole engine re-expressed; all gates green |

**Phase-numbering note.** The commit stream reuses the numbers in §7 but not the boundaries: §7's "Phase 5 — performance engineering" was executed as commits `2f6a4f8`/`7de8342`/`2ad77fd` (all of the fusion work except CUDA graphs and a persistent megakernel), and this repo's "Phase 6" is the **substrate swap**, which §7 does not have a phase for at all. §7's own Phase 6 — the axum server, request batching, OOM degradation — has **not** been started and is still described correctly in §7 and §11.7.

Code size: **4,505 lines** of Rust across `kronos-core` (3,515) and `kronos-cli` (990), plus ~1,000 lines of integration tests.

### 11.2 The substrate decision (the one real fork in the road)

Phase 5 established the ceiling precisely, and it was not the language:

- Every *isolated* op in the candle engine was **slower than the same op in Python**, sometimes by 3–6× (softmax 4.7 ms vs 0.9 ms; an untuned `mul` at 1.6 ms vs 0.04 ms).
- One GEMM was **10× off peak** because candle silently routed through a deprecated cuBLAS algo.
- Writing kernels by hand *worked* — the fused attention kernel was a 3.5× win — but each kernel is days of work, and there are hundreds of `mul`/`add`/`reshape` calls per step.

So the question became: can Rust drive **the exact libtorch the oracle uses**, with no Python? Answer: yes — `tch` 0.24 + `torch-sys`, linking the pip torch 2.11.0 installation. This preserves every tuned ATen kernel (cuBLAS/cuBLASLt GEMMs, fused RMSNorm, mem-efficient/flash SDPA, cumsum/softmax/sort chains) while keeping all model logic, the sampler, the CLI and the parity harness in Rust. It is a *substrate* change, not a cheat: the arithmetic still has to be written, and it still has to hit the frozen oracle.

Rejected alternatives: `burn`+`cubecl` (no mature SDPA/quantizer story, would rewrite the kernels by hand); `candle`+`cuda-oxide`/`cutile` (authoring DSLs, no library kernels — the 5.1/5.2 data above says hand-writing is the slow path).

### 11.3 Build recipe (this bit is fiddly; don't rediscover it)

```bash
# 1. libtorch = the pip torch 2.11.0 already used for the oracle
export T=/tmp/opencode/kronosenv/lib/python3.12/site-packages
export LIBTORCH=$T/torch
export LD_LIBRARY_PATH="$T/torch/lib:$T/nvidia/{cublas,cudnn,cudart,nccl,cusparse,cufft,cusolver,nvtx,nvjitlink,curand,nvshmem}/lib:$LD_LIBRARY_PATH"
source /tmp/opencode/tchspike/env.sh   # all of the above, in one line
```

- **`tch = "0.24"` exactly.** torch-sys 0.24 pins torch 2.11.0 (bit-identical to the oracle env). 0.26 targets a newer torch and fails against the 2.11 headers.
- No feature flags: `tch` has no `cuda` feature — libtorch CUDA is picked up from `LIBTORCH`.
- Both `build.rs` files re-emit `-Wl,--no-as-needed -Wl,-ltorch_cuda -Wl,-rpath,$LIBTORCH/lib`. Without this the linker drops `libtorch_cuda.so` (nothing in the Rust code references its symbols) and ATen's CUDA dispatch comes up unregistered.
- **`source` the env before every cargo command**, or `torch-sys` builds against the wrong ABI.

### 11.4 Porting notes — the traps that actually cost time

1. **dtype is not `DType`.** Candle's `DType::U32` has no torch analogue; token ids are `Kind::Int` (i64) and the packed-token pipeline widens to i64 when it walks embedding tables.
2. **Implicit-conversion rules that saved the port.** `f_multiply_scalar`, `f_add_scalar` and friends accept `impl Into<Scalar>`, so `Tensor + &Tensor` (candle idiom) becomes `&a + &b` with no `f_add` wrapper.
3. **Scalars must live on the device.** `Tensor::from_slice(&[1e-5])` is a CPU tensor; `(&std + &eps)` then dies with *"two devices, cuda:0 and cpu"*. Build it with `.to_device(dev)`.
4. **2-D tensors will not flatten implicitly.** `try_into::<Vec<f32>>()` on a `[1, 1]` errors — add `.reshape([-1])`.
5. **A missing `?` can still compile.** Every ported op that can touch the device returns `Result<Tensor>`; dropping the `?` gives you a `Result`, and because `Result` *also* has `map_err`, the compiler stays quiet and then fails a line later with *"no method named `unsqueeze` found for struct `Result`"*. Prefer fixing the call site over casting.
6. **The final-decode window is `min(total, max_context)`, not `max_context`.** Narrowing by `max_context` unconditionally crashes every ctx256/pred8 golden with `start (0) + length (512) exceeds dimension size (264)` — the ring buffer never filled because `initial + pred_len < max_context`.
7. **Do not carry candle's literal dim numbers across.** `argmax(dim)` → `f_argmax(-1, false)` (last axis), `to_vec1::<u32>()` → `to_kind(Kind::Int)` + `try_into()`. A hard-coded `f_argmax(2, …)` on a `[1, 1024]` tensor only fails at runtime, with *"dimension index 2 out of range"*, and only on the code path that hits it.
8. **The sampler audit's analytic p must be computed at the sampling temperature.** Softmaxing the raw logits while drawing at `T=0.6` reports 0.1028 against an empirical 0.2148 and fails a correct sampler. Softmax `logits / T`.
9. **Audit the parity of the *audit*, not just the model.** Every one of the bugs above was found by a gate, never by inspection.
10. **Retiring candle is a real deletion, not a feature flag.** `flash.rs`, `flash_attn.ptx` and the cuBLASLt GEMM wrappers went; `rust/cuda/flash_attn.cu` survives only as history under `7de8342`.

### 11.5 Gate status at `afeae9d` — all green

| Gate | Command | Status |
|---|---|---|
| Build (release) | `cargo build --release` | clean, 0 warnings |
| Lints | `cargo clippy --all-targets` | 0 warnings |
| Unit tests | `cargo test --release --lib` | 30 passed |
| Integration suites | `cargo test --release --tests` | phase4 (8), bits (2), decode, linear, weights, gemm — all passed |
| Oracle frozen | `./parity/check_frozen.sh` | `model/` and `tests/` byte-identical to `frozen.lock` |
| Golden CSVs (tests) | `phase4` | 6/6 pass at `rtol=1e-5` — worst rel err per model/ctx: small 6.14e-7 (ctx256) / 4.76e-7 (ctx512), base 6.14e-7 / 2.45e-7, mini 3.55e-7 / 1.83e-7 |
| Golden CSVs (CLI spot-check) | `kronos-cli predict --golden` | matches the above through the real CSV→forecast→de-normalise path |
| Sampler audit | `kronos-cli sampler-audit` | 3/3 assertions pass |

Deterministic (greedy) parity is bitwise reproducible run-to-run, as in Python.

### 11.6 Performance status — measured, with the caveat that matters

**The measurement protocol for anything that goes in a report:** liaise with the user first (they must be able to stop whatever else is running on the laptop), then take the *minimum* of ≥3 iterations with a cooldown, and record clocks/temperature. Consumer GPUs throttle under sustained load, and the Python baseline was taken the same way (`parity/baseline_pytorch.json`: min-of-3, 10 s cooldown).

**Liaised per-op comparison, `tch` vs Python, Kronos-small fp32** (this is the trustworthy one):

| Op | shape | Python | Rust (`tch`) |
|---|---|---|---|
| linear | GEMM in the model's own shapes | 0.54 ms | 0.68 ms |
| rmsnorm | fused | 0.072 ms | **0.030 ms** |
| softmax | `[10, 4274]` | 0.004 ms | 0.004 ms |
| softmax | `[10, 2²¹]` | 0.91 ms | 0.91 ms |
| **sdpa** | `[10, 8, 512, 64]` causal | 0.399 ms | 0.403 ms |

So: SDPA is exactly at parity (and is now the fused mem-efficient kernel in fp32, FlashAttention in fp16 — the custom candle kernel it replaced took 2.4 ms), softmax is at parity, RMSNorm is 2.4× *faster* than Python's in-line call, and only the linear GEMM is behind (0.68 vs 0.54 ms), which is worth a second look but is not a 3× problem.

**End-to-end is still unexplained, and I am not going to pretend otherwise.** Kronos-small, ctx512, pred120, B=10, fp32, against the frozen baseline (`parity/baseline_pytorch.json`, min-of-3 with 10 s cooldown):

| Engine | AR loop | per step |
|---|---|---|
| PyTorch oracle (liaised, min-of-3) | 3,260 ms | **39.4 ms** |
| Rust candle | ~14,200 ms | ~118 ms |
| Rust `tch` | ~15,800 ms | ~131 ms |

Reference points from the same frozen file: small/512/48/B=10 = **1,940 ms**, mini/512/120/B=10 = **877 ms** (the cheapest large-ish target).

The substrate swap did **not** move end-to-end, even though it moved every isolated op to parity. Three candidate explanations, none yet tested:

1. **The E2E runs above were not liaised.** They were taken on a contended laptop. The per-op table *was* liaised and came out green. This is the cheapest thing to rule out and the most likely explanation — if a 4× E2E gap is real it cannot hide behind ops that each measure at parity.
2. **Launch-bound rather than device-bound.** `decode_s1` is ~350 small aten calls; if per-call CPU dispatch through the tch FFI costs ~100 µs, that is 35 ms of pure CPU with the GPU idle. Distinguishable cheaply: if per-step time is the same at B=1 and B=10, we are launch-bound; if it scales ~10×, we are device-bound. (`nvidia-smi dmon` during the run settles it outright.)
3. **Op-count inflation** — extra `.contiguous()` / `reshape` / `transpose` traffic that Python does not pay (the attention path materialises 3 contiguous copies per layer that Python's strided SDPA accepts as-is).

**Next step, in order:** liaised E2E fp32 run of small/512/120/B=10 → if the gap survives a quiet machine, run the B=1 vs B=10 scaling test with `nvidia-smi dmon`, then attack whichever of (2)/(3) it names. Then the fp16 run, which is the more interesting number anyway (§6.4 predicts ~3× from precision alone; an earlier session measured the oracle's own bf16 path at ~1.1 s for this config, though that row is *not* in the frozen baseline and should be re-taken).

### 11.7 What is left

- **[blocking]** Establish whether the E2E gap is real (§11.6) — one liaised run answers it.
- Port `cuTile`/fused-kernel ideas onto `tch` only if (2)/(3) come back clean, i.e. only if the GPU really is busy.
- The tokenizer's causal decode **does** admit a prefix KV cache (§4.3's exception): ~3–4 ms per forecast, and it is valid, but it changes numerics slightly and needs its own parity decision.
- Phase 7 of the original plan — the axum server, request batching, graceful OOM degradation — is untouched. Everything above is library + CLI.
- Training stays in Python (§5.1, §7.7). Not revisited.