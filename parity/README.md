# parity/ — PyTorch oracle and parity harness

Everything the Rust port is measured against lives here. Nothing here is
imported by the Rust code at runtime; it is fixtures + comparison tooling.

## The freeze

`model/` and `tests/` are the **parity oracle** and are byte-frozen at upstream
commit `67b630e`. Run `./check_frozen.sh` (or let CI) to verify:

```
sha256sum --check parity/frozen.lock
```

Rationale: every parity claim we make is only worth something if the
reference is immutable. Patching `model/kronos.py` to make a comparison
convenient destroys the ability to distinguish "the Rust is wrong" from
"someone moved the goalposts".

Bugs found in the oracle are **documented, not fixed** — see
`PORTING_FROM_PYTHON.md` §1 and §8:

| File:line | Issue | Why not fixed |
|---|---|---|
| `model/kronos.py:466` | `z.cpu().numpy()` raises `TypeError: unsupported ScalarType BFloat16` — the reference cannot run in bf16 at all | blocks nothing; cast in the harness instead |
| `model/kronos.py:377` | `if top_k > 0` inside `top_k_top_p_filtering` will `TypeError` if `top_k is None`; unreachable from `sample_from_logits` today | port normalises args at the API boundary |
| `model/module.py:10-32` | `DifferentiableEntropyFunction` allocates `2**20` and scatter-reduces; also **dead** — `soft_entropy=True` is hardcoded at `module.py:42` | do not port the dead path |
| `model/kronos.py:451-454` | window is rolled and re-encoded every step with window-relative RoPE ⇒ **no valid KV cache** | architectural, not a bug; see `PORTING_FROM_PYTHON.md` §4.3 |

## Fixtures

Regenerate everything with:

```bash
python parity/gen_reference.py --all          # needs torch + CUDA
```

### `fixtures/golden/<model>_ctx<N>_pred8.csv`

End-to-end **deterministic** predictions (`T=1.0, top_k=1, top_p=1.0`,
`sample_count=1`, `seed=123`), same column shape as the upstream
`tests/data/regression_output_*.csv`.

`kronos-small` is cross-checked against the upstream CSVs and agrees at
`rtol=1e-5` (max rel err 1.2e-6 @ ctx 512, 2.1e-6 @ ctx 256). The mini/base
CSVs were generated here because upstream ships no fixtures for them.

### `fixtures/layers/<model>/`

Per-stage activations captured with forward hooks from a single
`ctx_len=512` deterministic pass. 33–41 tensors per model, ~108 MB total.

Names are namespaced `pred.*` (the `Kronos` predictor) and `tok.*` (the
`KronosTokenizer`) because **both models own modules called `head` and
`embed`** — without the prefix they collide and silently overwrite.

Filenames are the tensor name with `.` → `__`. Format is `.npy` v1.0
(`<f4` little-endian float32), chosen because the header is trivial to parse
from Rust and needs no extra dependency.

Captured stages:

| Stage | Tensor | Validates |
|---|---|---|
| tokenize | `tok.embed`, `tok.encoder.{0..2}`, `tok.quant_embed`, `tok.tokenizer.bsq` | encoder + BSQ quantiser + `F.normalize` |
| detokenize | `tok.post_quant_embed`, `tok.decoder.{0..2}`, `tok.head` | `indices_to_bits` + decoder |
| embed | `pred.embedding.{emb_s1,emb_s2,fusion_proj}`, `pred.time_emb` | `*sqrt(d_model)`, `W_fuse`, 5 temporal lookups |
| block 0 internals | `pred.transformer.0.{norm1,norm2,self_attn.{q,k,v,out}_proj,ffn.{w1,w2,w3}}` | RMSNorm, RoPE-free projections, SwiGLU |
| blocks 0..N | `pred.transformer.{i}` | residual structure per layer |
| heads | `pred.norm`, `pred.head.{proj_s1,proj_s2}`, `pred.dep_layer.{cross_attn,norm}` | RMSNorm, dual head, dependency-aware cross-attention |
| ints | `token_s1`, `token_s2`, `pre_argmax` | token ids must match **exactly** |
| outputs | `s1_logits_last`, `s2_logits_last`, `context_last`, `tokenizer_recon` | end-of-pipeline |

`tok.post_quant_embed_pre` is **intentionally absent**: it is only used in
`KronosTokenizer.forward` (the training path), never in `decode()`.

`index.json` lists shapes plus the non-tensor config (`mean`/`std` of the
normalisation window, `ctx_len`, `seed`, and the model hyperparameters). It
also records `deduped_aliases` — hooks that provably returned identical
tensors and were dropped:

```
pred.head        -> pred.head.proj_s1        # DualHead.forward returns proj_s1(x)
pred.embedding   -> pred.embedding.fusion_proj
pred.dep_layer   -> pred.dep_layer.norm      # DependencyAwareLayer returns norm(...)
```

### `fixtures/weights.json`

Pinned HF repo + revision, the full `config.json`, and the **sha256 of
`model.safetensors`** for every model/tokenizer. The Rust loader asserts
against these so a silent partial or mismatched load is impossible.

> Note the released tokenizers are **not** identical:
> `Kronos-Tokenizer-base` has `group_size=4`, `Kronos-Tokenizer-2k` has
> `group_size=5` (`num_groups` 5 vs 4). `Kronos-mini` pairs with the 2k
> tokenizer. The paper describes "3 layers, ff_dim 1024, group_size 9" for
> the tokenizer — **trust `config.json`, not the paper**.

## Parity tiers

| Tier | Definition | Gate |
|---|---|---|
| deterministic | `top_k=1, top_p=1.0` ⇒ argmax | **`rtol=1e-5` vs golden CSVs** — strict |
| stage | per-tensor activations | `rtol=1e-5` fp32, per tensor |
| token ids | `token_s1`/`token_s2` | **exact integer equality** |
| stochastic | nucleus sampling | 3 cheap assertions, **not** a statistical test |
| precision | fp16 vs fp32 | divergence report, bounded, not bit-exact |

The `tok.post_quant_embed_pre` stage is intentionally absent: it is used only by
`KronosTokenizer.forward` (the training path), never by `decode()`.

## Running the per-layer parity check (Phase 2)

```bash
python parity/gen_probe.py          # shared 512-row probe input
for m in small base mini; do ...; done
```

`rust/target/release/kronos-cli parity` runs the tokenizer encode, the tokenizer
decode round-trip, the predictor's `decode_s1` and `decode_s2`, and compares
every captured stage against `fixtures/layers/<model>/`. Current result on all
three models:

| stage | kronos-small | kronos-base | kronos-mini |
|---|---|---|---|
| token ids (`s1`/`s2`/argmax) | **exact** | **exact** | **exact** |
| `tokenizer_recon` | 4.3e-6 | 4.3e-6 | 3.8e-6 |
| `context_last` | 4.5e-7 | — | — |
| `s1_logits_last` | 1.2e-5 | — | — |
| `s2_logits_last` | 8.6e-6 | — | — |

Result: **7/7 stages pass for all three released models at `rtol=1e-5`**.

### Bugs Phase 2b found (all via fixtures, none via unit tests alone)

1. **Fortran-order `.npy`.** `np.save` wrote `probe_x.npy` with
   `fortran_order: True` because the array was built with `np.stack([...], 1)`.
   The reader silently treated it as C-order, so the "input" was actually
   `x[0,0,0], x[1,0,0], x[2,0,0], ...` — three similar `open` prices repeated —
   and every downstream stage was poisoned. Fixed both sides: the reader now
   handles Fortran order, and `gen_probe.py` writes contiguous arrays.
2. **int64 read as 4 bytes.** `'<i8'` is 8 bytes per element; the reader read
   4 and widened the low half, shifting the timestamp axis.
3. `(1,1)` shape treated as 2 elements (`max(len)`) instead of the product.
4. **`indices_to_bits` mask direction inverted.** Dividing by `2^-i` *multiplies*
   by `2^i`, broadcasting the low bit of the index to every position. This was
   caught by a full-codebook round-trip test after a fixture-only mismatch was
   not enough to localise it.
5. **RMSNorm eps in the wrong place.** `rsqrt(mean + eps)` is not
   `rsqrt(mean) + eps`. The latter amplifies a near-zero mean by 1e5 and made
   the encoder blow up to 1e19 and then NaN.
6. **Pre-LN residual wired backwards.** `x + attn(norm(x))`, not
   `norm(x) + attn(x)`.

Stochastic parity is distributional by agreement. Rather than a χ² gate we use
three assertions that catch essentially every realistic sampler bug at
negligible cost — because `top_k=1` degenerates to argmax and therefore
**never executes** `sort` / the cumulative threshold / `multinomial`:

1. top-1 token frequency within a few percent of Python's over many draws
2. **no** sample ever lands outside Python's nucleus
3. the RNG produces different draws across steps (not a constant)

See `../PORTING_FROM_PYTHON.md` §6.5 and §8 for the fp16 measurement.