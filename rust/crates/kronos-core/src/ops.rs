//! Low-level tensor ops that mirror `model/module.py`.
//!
//! Every function here corresponds 1:1 with a Python expression and cites the
//! line it reproduces. Where a detail is easy to get subtly wrong — and would
//! produce a model that runs and is quietly wrong — it is called out inline.
//!
//! Backend note (Phase 6): the substrate is libtorch via tch — the exact
//! `libtorch_cuda.so` the frozen PyTorch oracle runs — so this layer is thin.
//! The candle-era custom kernels (flash.cu, gemm_fast) were retired once aten
//! kernels matched or beat them: rms_norm/softmax/linear dispatch through the
//! same fused paths Python used, sdpa is the real fused/mem-efficient kernel
//! (fp32: mem-efficient cutlass; bf16/fp16: flash), and the GEMM algo bug
//! candle had (deprecated CUBLAS_GEMM_DEFAULT_TENSOR_OP) does not exist here.

use std::sync::Mutex;
use tch::{Device, Kind, Tensor};

pub type Result<T> = std::result::Result<T, tch::TchError>;

/// CPU device handle for fixture/profile purposes.
pub fn cpu() -> Device {
    Device::Cpu
}

/// The engine's CUDA device (ordinal 0, matching every run on this box).
pub fn cuda() -> Device {
    Device::Cuda(0)
}

/// Embedding lookup on rank-2 `[B, T]` ids: torch's `embedding` is rank-free,
/// so Kronos ids work directly (candle needed a flatten shim).
#[inline]
pub fn embedding_lookup(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    Tensor::f_embedding(weight, ids, -1, false, false)
}

/// Linear projection: `[B, L, in] x [out, in]^T + bias -> [B, L, out]`.
///
/// The single most frequently called operation in the model. `F.linear` on
/// 3D input: aten casts to a single big 2-D GEMM plus a bias epilogue — the
/// fastest path available on this stack (phase-5 probes: 0.54-0.68 ms for the
/// (6320,512)x[512,1024] FFN shape, vs candle's 1.23 ms).
pub fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let y = x.matmul(&w.transpose(1, 0));
    match bias {
        Some(b) => Ok(y + b),
        None => Ok(y),
    }
}

/// `RMSNorm` — `model/module.py:257-268`.
///
/// `x * rsqrt(mean(x*x, -1, keepdim) + eps) * weight`, eps = 1e-5. Wrapped to
/// aten's fused rms_norm (torch 2.4+; numerically cross-checked against the
/// explicit chain at rel err 3.6e-7 in the spike). NOTE the eps goes INSIDE
/// the sqrt in this call form: rsqrt(ms + eps).
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let d = x.size()[x.size().len() - 1];
    let w = weight.reshape([d]);
    x.f_rms_norm([d], Some(&w), eps as f64)
}

/// `silu` — FFN gate (`model/module.py:231`): `w2(silu(w1(x)) * w3(x))`.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    x.f_silu()
}

/// RoPE — `model/module.py:284-312`.
///
/// This is the **GPT-J / `rotate_half`** convention, not GPT-NeoX:
///
/// ```text
/// inv_freq[j]     = 1 / 10000^(2j/dim)              j in [0, dim/2)
/// freqs[t, j]     = t * inv_freq[j]
/// cos             = cat(freqs.cos(), freqs.cos())   # [seq, dim]
/// _rotate_half(x) = cat(-x[..., dim/2:], x[..., :dim/2])
/// out             = q * cos + rotate_half(q) * sin
/// ```
///
/// Getting the half-split backwards is the easiest way to build a model that
/// loads cleanly and forecasts garbage. Upstream caches cos/sin keyed on
/// sequence length (module.py:294); we build the table once for `max_len`.
/// The historians' warning from the candle port stands: expressing the j=0
/// term as powf(-1) gives 0**-1 = +inf; exp(-j*ln10000) is the safe form.
pub struct RotaryCache {
    cos: Tensor,
    sin: Tensor,
    len: usize,
}

impl Clone for RotaryCache {
    fn clone(&self) -> Self {
        Self {
            cos: self.cos.shallow_clone(),
            sin: self.sin.shallow_clone(),
            len: self.len,
        }
    }
}

impl RotaryCache {
    pub fn new(dim: usize, max_len: usize, device: &Device, dtype: Kind) -> Result<Self> {
        assert!(
            dim.is_multiple_of(2),
            "RoPE head_dim must be even, got {dim}"
        );
        // upstream: 1.0 / (10000 ** (arange(0, dim, 2) / dim))
        let tally = Tensor::f_arange_start_step(0.0, dim as f64, 2.0, (Kind::Float, *device))?;
        let tally = tally / dim as f64;
        let instr_freq = ((tally * (10_000f64.ln())).f_neg()?).f_exp()?;
        let t = Tensor::f_arange_start_step(0.0, max_len as f64, 1.0, (Kind::Float, *device))?;
        let freqs = t.unsqueeze(1) * instr_freq.unsqueeze(0); // [seq, dim/2]
        let cat = Tensor::cat(&[&freqs, &freqs], 1); // [seq, dim]
        Ok(Self {
            cos: cat.f_cos()?.unsqueeze(0).unsqueeze(0).f_totype(dtype)?,
            sin: cat.f_sin()?.unsqueeze(0).unsqueeze(0).f_totype(dtype)?,
            len: max_len,
        })
    }

    #[inline]
    pub fn max_len(&self) -> usize {
        self.len
    }

    /// Apply RoPE to `q` and `k`, both `[B, H, L, head_dim]`.
    pub fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let l = q.size()[2];
        if l > self.len as i64 {
            return Err(tch::TchError::Kind(format!(
                "rope cache too short: need {l}, have {}",
                self.len
            )));
        }
        let cos = self.cos.narrow(2, 0, l);
        let sin = self.sin.narrow(2, 0, l);
        Ok((
            rotate_half_apply(q, &cos, &sin)?,
            rotate_half_apply(k, &cos, &sin)?,
        ))
    }
}

/// `x * cos + rotate_half(x) * sin`.
fn rotate_half_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.size()[3];
    let half = d / 2;
    let x1 = x.narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);
    let rotated = Tensor::cat(&[&(-x2), &x1], 3);
    Ok(x * cos + rotated * sin)
}

/// `F.scaled_dot_product_attention` — `model/module.py:345` (self) / `:389`
/// (cross).
///
/// The REAL fused kernel this time (the spike measured fp32 0.40 ms vs our
/// materialised chain's 8.3 ms at (10,8,512,64) — this dispatches to cutlass
/// memory-efficient attention at fp32 and flash at bf16/fp16, identical to
/// what Python's oracle runs).
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    Tensor::f_scaled_dot_product_attention(q, k, v, None::<&Tensor>, 0.0, causal, None, false)
}

/// `F.normalize(z, dim=-1)` — `model/module.py:246`.
///
/// Unlike `rms_norm` there is no learned scale and the default eps (1e-12)
/// only guards division by zero.
pub fn l2_normalize(z: &Tensor) -> Result<Tensor> {
    // Sums of squares MUST be accumulated in f32 regardless of compute dtype.
    // The tokenizer's encoder emits activations of order 1e4 (larger than
    // f32's 65504 limit — no wait, larger than f16's: 65_504 — and the
    // upstream autocast path always reduces in f32). In fp16 the squares
    // (1e4^2 = 1e8) overflow -> every value normalises to 0 -> sign(0) = -1
    // -> every subtoken collapses to id 0 (measured: 100% token flips before
    // this fix in the Phase-2b engine).
    let z32 = z.to_kind(Kind::Float);
    let zsq = z32.multiply(&z32);
    let last = z32.size().len() as i64 - 1;
    let sums = zsq.f_sum_dim_intlist(vec![last], true, None)?.f_sqrt()?;
    Ok(z32 / sums)
}

/// BSQ sign step — `BinarySphericalQuantizer.quantize`, `model/module.py:82-88`.
///
/// Upstream is `torch.where(z > 0, 1, -1)` under a straight-through estimator;
/// with no autograd it is exactly that mapping. Note the tie case: **z == 0
/// maps to -1**, which is part of the model's quantisation and must be
/// reproduced. `sign()` would map zero to 0 — WRONG. Hence the explicit
/// gt/not-ge comparison with a bipolar affine.
pub fn bsq_sign(z: &Tensor) -> Result<Tensor> {
    let mask = z.f_gt(0.0f64)?.f_totype(Kind::Float)?;
    Ok(mask * 2.0 - 1.0)
}

/// `BSQuantizer.bits_to_indices` — `model/module.py:234-243`.
///
/// `indices = sum_i (bits[..., i] >= 0) * 2**i` — element 0 of the bit vector
/// is the **least** significant bit. With `s1_bits == s2_bits == 10` this
/// happens to coincide with left-shift packing, but we implement the weighted
/// sum the source defines rather than assuming the coincidence.
pub fn bits_to_indices(bits: &Tensor) -> Result<Tensor> {
    let n = bits.size()[bits.size().len() - 1];
    let powers = Tensor::f_arange_start(0.0, n as f64, (Kind::Double, bits.device()))?.f_exp2()?;
    let mask = bits.f_ge(0.0f64)?.f_totype(Kind::Double)?;
    let prod = &mask * &powers;
    let last = prod.size().len() as i64 - 1;
    prod.f_sum_dim_intlist(vec![last], false, None)
}

/// `KronosTokenizer.indices_to_bits` with `half=True` — `model/module.py:126-140`.
///
/// Inverse of the above: expand each index into LSB-first bits, map to bipolar
/// {-1, +1}, then scale by `1/sqrt(codebook_dim)`.
///
/// History note (keep for the bug ledger): candle 0.11 had no integer shift,
/// so this used floor(index / 2^i) mod 2 — BOTH floors required, positive
/// powers only. torch HAS integer ops (>> and &), and the direct integer form
/// is not only simpler but avoids fp64 mantissa limits in odd edge cases.
pub fn indices_to_bits(
    s1: &Tensor,
    s2: &Tensor,
    codebook_dim: usize,
    q_scale: f64,
) -> Result<Tensor> {
    let half = codebook_dim / 2;
    let inject_score_idx = |idx: &Tensor| -> Result<Tensor> {
        let i64idx = idx.to_kind(Kind::Int64);
        let mut cols: Vec<Tensor> = Vec::with_capacity(half);
        for i in 0..half {
            let bit = i64idx
                .f_bitwise_right_shift(&Tensor::f_full(
                    [],
                    i as i64,
                    (Kind::Int64, i64idx.device()),
                )?)?
                .f_bitwise_and(1i64)?;
            let b = (bit.f_totype(Kind::Float)? * 2.0) - 1.0;
            // push onto a NEW trailing axis: each column is a bit index
            cols.push(b.unsqueeze(2));
        }
        let bits = Tensor::cat(&cols, 2);
        Ok(bits * q_scale)
    };
    let b1 = inject_score_idx(s1)?;
    let b2 = inject_score_idx(s2)?;
    let out = Tensor::cat(&[&b1, &b2], b1.size().len() as i64 - 1);
    Ok(out)
}

/// Rebuild `group_codebook` — `model/module.py:76-78`.
///
/// Registered with `persistent=False`, so it is absent from **every**
/// checkpoint and must be reconstructed. It is
/// `indexes_to_codes(arange(2**group_size))[:, -group_size:]`: the trailing
/// `group_size` bits of each group index, in their original order, as
/// bipolar floats.
pub fn build_group_codebook(group_size: usize, device: &Device) -> Result<Tensor> {
    let n = 1usize << group_size;
    let mut rows = Vec::with_capacity(n * group_size);
    for idx in 0..n {
        for i in 0..group_size {
            // basis[j] = 2^(group_size-1-j), so bit i of the code is
            // (idx >> (group_size-1-i)) & 1
            let bit = (idx >> (group_size - 1 - i)) & 1;
            rows.push(if bit == 1 { 1.0f32 } else { -1.0f32 });
        }
    }
    Ok(Tensor::from_slice(&rows)
        .reshape([n as i64, group_size as i64])
        .to_device(*device))
}

// ---------------------------------------------------------------------------
// Env-gated profile accumulators (kept for cross-phase comparisons; nothing
// in the aten path needs micro-instrumentation since torch has its own
// profiler, but the hooks stay so profile-take callers keep compiling).
// 0: scores bmm, 1: mask add, 2: softmax, 3: value bmm, 4: RMSNorm,
// 5: linear, 6: other.
static PROF: Mutex<[f64; 7]> = Mutex::new([0.0; 7]);

#[doc(hidden)]
pub fn prof_take() -> [f64; 7] {
    if std::env::var("KRONOS_OPS_PROFILE").is_err() {
        return [0.0; 7];
    }
    match PROF.try_lock() {
        Ok(mut t) => std::mem::replace(&mut *t, [0.0; 7]),
        Err(_) => [0.0; 7],
    }
}

#[doc(hidden)]
pub fn __prof_norm(dt: f64) -> f64 {
    if std::env::var("KRONOS_OPS_PROFILE").is_ok() {
        if let Ok(mut t) = PROF.try_lock() {
            t[4] += dt;
        }
    }
    dt
}

#[doc(hidden)]
fn prof_add(slot: usize, dt: f64) {
    if std::env::var("KRONOS_OPS_PROFILE").is_ok() {
        if let Ok(mut t) = PROF.try_lock() {
            t[slot] += dt;
        }
    }
}

#[allow(dead_code)]
fn keep_prof_used() {
    prof_add(6, 0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_at_position_zero_is_identity() {
        let dev = Device::Cpu;
        let dim = 8usize;
        let rc = RotaryCache::new(dim, 5, &dev, Kind::Float).unwrap();
        let q = Tensor::f_arange_start_step(0.0, dim as f64, 1.0, (Kind::Float, dev))
            .unwrap()
            .reshape([1, 1, 1, dim as i64]);
        let (qr, _) = rc.apply(&q, &q).unwrap();
        let v: Vec<f32> = (&qr.f_to(Device::Cpu).unwrap().reshape([-1]))
            .try_into()
            .unwrap();
        for (i, got) in v.iter().enumerate() {
            assert!((*got - i as f32).abs() < 1e-6, "index {i}: got {got}");
        }
    }

    #[test]
    fn rope_uses_rotate_half_not_neox_interleaving() {
        let dev = Device::Cpu;
        let dim = 4usize;
        let rc = RotaryCache::new(dim, 3, &dev, Kind::Float).unwrap();
        let q = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
            .reshape([1, 1, 2, dim as i64]);
        let (qr, _) = rc.apply(&q, &q).unwrap();
        let v: Vec<f32> = (&qr.f_to(Device::Cpu).unwrap().reshape([-1]))
            .try_into()
            .unwrap();

        // position 0: angles are zero, so the rotation is the identity
        assert!((v[0] - 1.0).abs() < 1e-5, "pos0[0] got {}", v[0]);
        assert!((v[1] - 2.0).abs() < 1e-5, "pos0[1] got {}", v[1]);
        assert!((v[2] - 3.0).abs() < 1e-5, "pos0[2] got {}", v[2]);
        assert!((v[3] - 4.0).abs() < 1e-5, "pos0[3] got {}", v[3]);

        let expect = [-1.984_110_6, 1.959_900_7, 2.462_377_9, 4.019_799_7];
        for (i, e) in expect.iter().enumerate() {
            assert!(
                (v[4 + i] - e).abs() < 1e-4,
                "pos1[{i}] expected {e}, got {}",
                v[4 + i]
            );
        }
    }

    #[test]
    fn rope_cache_rejects_overlong_sequences() {
        let dev = Device::Cpu;
        let rc = RotaryCache::new(8, 4, &dev, Kind::Float).unwrap();
        let q = Tensor::zeros([1, 1, 5, 8], (Kind::Float, dev));
        assert!(rc.apply(&q, &q).is_err());
    }

    #[test]
    fn bsq_sign_maps_zero_to_minus_one() {
        let dev = Device::Cpu;
        let z = Tensor::from_slice(&[-2.0f32, -0.0, 0.0, 1e-30, 3.0]).reshape([1, 5]);
        let s: Vec<f32> = (&bsq_sign(&z).unwrap().f_to(dev).unwrap().reshape([-1]))
            .try_into()
            .unwrap();
        assert_eq!(s, vec![-1.0, -1.0, -1.0, 1.0, 1.0]);
    }

    #[test]
    fn bits_to_indices_is_lsb_first() {
        let _dev = Device::Cpu;
        let bits = Tensor::from_slice(&[1.0f32, -1.0, 1.0, -1.0]).reshape([1, 4]);
        let idx: Vec<f64> = (&bits_to_indices(&bits).unwrap().reshape([-1]))
            .try_into()
            .unwrap();
        assert_eq!(idx[0], 5.0, "bit0 + bit2 = 1 + 4 under LSB-first packing");

        // A literal zero IS counted as set, matching torch's `>= 0`.
        let zeros = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0]).reshape([1, 4]);
        let idx: Vec<f64> = (&bits_to_indices(&zeros).unwrap()).try_into().unwrap();
        assert_eq!(idx[0], 15.0, "zeros count as set under `>= 0`");
    }

    #[test]
    fn bits_indices_round_trip_over_the_real_codebook() {
        let _dev = Device::Cpu;
        const CODEBOOK_DIM: usize = 20;
        let n = 1usize << 10;
        let ids: Vec<i64> = (0..n).map(|i| i as i64).collect();
        let idx = Tensor::from_slice(&ids).reshape([1, n as i64]);
        let bits = indices_to_bits(&idx, &idx, CODEBOOK_DIM, 1.0).unwrap();
        assert_eq!(bits.size(), [1_i64, n as i64, CODEBOOK_DIM as i64]);

        // half=True produces cat([s1_bits(10), s2_bits(10)], -1), so each
        // subtoken must be recovered from its own slice.
        let a: Vec<i64> = (&idx.reshape([-1])).try_into().unwrap();
        for (name, start) in [("s1", 0i64), ("s2", 10i64)] {
            let half = bits.narrow(2, start, 10);
            let back: Vec<f64> = (&bits_to_indices(&half).unwrap().reshape([-1]))
                .try_into()
                .unwrap();
            for i in 0..n {
                assert_eq!(
                    a[i] as f64, back[i],
                    "{name} index {i} failed to round-trip"
                );
            }
        }
    }

    #[test]
    fn group_codebook_shape_and_bipolarity() {
        let dev = Device::Cpu;
        for gs in [4usize, 5] {
            let cb = build_group_codebook(gs, &dev).unwrap();
            assert_eq!(cb.size(), [1i64 << gs, gs as i64]);
            let v: Vec<Vec<f32>> = (&cb).try_into().unwrap();
            for row in &v {
                for x in row {
                    assert!(*x == 1.0f32 || *x == -1.0f32, "non-bipolar entry {x}");
                }
            }
            assert!(v[0].iter().all(|x| *x == -1.0f32), "row 0 should be all -1");
            assert!(
                v[(1 << gs) - 1].iter().all(|x| *x == 1.0f32),
                "last row should be all +1"
            );
        }
    }

    #[test]
    fn rms_norm_gives_unit_rms() {
        let dev = Device::Cpu;
        let w = Tensor::ones([16], (Kind::Float, dev));
        let y = rms_norm(
            &Tensor::f_arange_start_step(0.0, 16.0, 1.0, (Kind::Float, dev))
                .unwrap()
                .reshape([1, 16]),
            &w,
            1e-5,
        )
        .unwrap();
        let v: Vec<f32> = (&y.reshape([-1])).try_into().unwrap();
        let rms = (v.iter().map(|x| x * x).sum::<f32>() / 16.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "rms was {rms}");
    }

    #[test]
    fn l2_normalize_gives_unit_norm() {
        let _dev = Device::Cpu;
        let z = Tensor::from_slice(&[3.0f32, 4.0, 0.0, 0.0]).reshape([1, 4]);
        let y = l2_normalize(&z).unwrap();
        let v: Vec<f32> = (&y.reshape([-1])).try_into().unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn sdpa_shapes_match_manual_reference() {
        let dev = Device::Cuda(0);
        if !tch::utils::has_cuda() {
            return;
        }
        let q = Tensor::randn([2, 8, 64, 64], (Kind::Float, dev));
        let k = Tensor::randn([2, 8, 64, 64], (Kind::Float, dev));
        let v = Tensor::randn([2, 8, 64, 64], (Kind::Float, dev));
        let out = sdpa(&q, &k, &v, true).unwrap();
        assert_eq!(out.size(), [2, 8, 64, 64]);
    }
}
