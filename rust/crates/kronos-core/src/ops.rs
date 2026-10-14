//! Low-level tensor ops that mirror `model/module.py`.
//!
//! Every function here corresponds 1:1 with a Python expression and cites the
//! line it reproduces. Where a detail is easy to get subtly wrong — and would
//! produce a model that runs and is quietly wrong — it is called out inline.

use candle_core::{DType, Device, Result, Tensor};

/// Reduce over the last axis of `t`.
#[inline]
fn last_dim(t: &Tensor) -> Vec<usize> {
    vec![t.dims().len() - 1]
}

/// Embedding lookup that accepts rank-2 index tensors.
///
/// Candle's `Tensor::embedding` only takes rank-1 ids. Kronos ids are `[B, T]`,
/// so we flatten, look up, and reshape back. This is called on every forward
/// pass so it is kept minimal.
#[inline]
pub fn embedding_lookup(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let dims = ids.dims().to_vec();
    let flat = ids.flatten_all()?;
    let out = weight.embedding(&flat)?;
    let mut shape = dims.clone();
    shape.push(weight.dim(1)?);
    out.reshape(&shape[..])
}

/// Linear projection: `[B, L, in] x [out, in]^T + bias -> [B, L, out]`.
///
/// Candle's `matmul` requires equal rank on both sides, so we flatten the
/// leading two dims, do the 2-D product, and restore. This is the single most
/// frequently called operation in the model.
#[inline]
pub fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let (b, l, din) = (dims[0], dims[1], dims[2]);
    let dout = w.dim(0)?;
    let x2 = x.reshape((b * l, din))?;
    let y2 = x2.matmul(&w.t()?)?;
    let y = y2.reshape((b, l, dout))?;
    match bias {
        Some(bias) => y.broadcast_add(bias),
        None => Ok(y),
    }
}

/// `RMSNorm` — `model/module.py:257-268`.
///
/// `x * rsqrt(mean(x*x, -1, keepdim) + eps) * weight`, eps = 1e-5.
///
/// The upstream `.float()` / `.type_as(x)` round trip means the sum of squares
/// is always accumulated in fp32 even under autocast; we do the same.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let ms = x32.sqr()?.mean_keepdim(last_dim(&x32))?;
    // upstream: x * rsqrt(mean(x^2) + eps) * weight
    // the eps goes INSIDE the sqrt: rsqrt(ms + eps), not rsqrt(ms) + eps.
    // Without this, a near-zero ms would produce a huge rsqrt and blow up.
    let denom = ms.affine(1.0f64, eps as f64)?.sqrt()?.recip()?;
    x32.broadcast_mul(&denom)?
        .to_dtype(x.dtype())?
        .broadcast_mul(weight)
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
#[derive(Clone)]
pub struct RotaryCache {
    cos: Tensor,
    sin: Tensor,
    len: usize,
}

impl RotaryCache {
    pub fn new(dim: usize, max_len: usize, device: &Device, dtype: DType) -> Result<Self> {
        assert!(
            dim.is_multiple_of(2),
            "RoPE head_dim must be even, got {dim}"
        );
        // upstream: 1.0 / (10000 ** (arange(0, dim, 2) / dim))
        // expressed as exp(-(j/dim)*ln10000) because powf(-1) on the j=0 term
        // would be 0**-1 = +inf and poison the whole table with NaN.
        let inv_freq = Tensor::arange_step(0f32, dim as f32, 2.0, device)?
            .broadcast_div(&Tensor::from_vec(vec![dim as f32], (), device)?)?
            .affine(10000f64.ln(), 0.0)?
            .neg()?
            .exp()?;
        let t = Tensor::arange(0f32, max_len as f32, device)?;
        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [seq, dim/2]
        let cat = Tensor::cat(&[&freqs, &freqs], 1)?; // [seq, dim]
        Ok(Self {
            cos: cat.cos()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)?,
            sin: cat.sin()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)?,
            len: max_len,
        })
    }

    #[inline]
    pub fn max_len(&self) -> usize {
        self.len
    }

    /// Apply RoPE to `q` and `k`, both `[B, H, L, head_dim]`.
    pub fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let l = q.dim(2)?;
        if l > self.len {
            return Err(candle_core::Error::Msg(format!(
                "rope cache too short: need {l}, have {}",
                self.len
            )));
        }
        let cos = self.cos.narrow(2, 0, l)?;
        let sin = self.sin.narrow(2, 0, l)?;
        Ok((
            rotate_half_apply(q, &cos, &sin)?,
            rotate_half_apply(k, &cos, &sin)?,
        ))
    }
}

/// `x * cos + rotate_half(x) * sin`.
fn rotate_half_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(3)?;
    let half = d / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
    x.broadcast_mul(cos)?
        .broadcast_add(&rotated.broadcast_mul(sin)?)
}

/// `F.scaled_dot_product_attention` — `model/module.py:345` (self) / `:389` (cross).
///
/// Candle's `sdpa` custom op requires cuDNN on CUDA, which is not installed
/// here. We implement the attention manually using standard tensor ops, which
/// is mathematically identical (up to float ordering) and portable.
///
/// Shapes: `q`, `k`, `v` are `[B, H, L, hd]`. `causal` applies a lower-triangular
/// mask so position `i` can only attend to positions `0..=i`.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let hd = q.dim(3)? as f64;
    let scale = (1.0 / hd.sqrt()) as f32;
    let (b, h, lq, _) = q.dims4()?;
    let lk = k.dim(2)?;

    // scores = (Q * K^T) * scale
    let kt = k.transpose(2, 3)?.contiguous()?; // [B, H, hd, Lk]
    let mut scores = q.matmul(&kt)?; // [B, H, Lq, Lk]
    if scale != 1.0 {
        scores = scores.affine(scale as f64, 0.0)?;
    }

    if causal {
        // mask[i][j] = -inf when j > i (cannot attend to the future)
        let mask = Tensor::from_vec(
            (0..lq * lk)
                .map(|idx| {
                    let i = idx / lk;
                    let j = idx % lk;
                    if j > i {
                        f64::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
            (lq, lk),
            q.device(),
        )?
        .to_dtype(q.dtype())? // match the scores dtype (f32 or f16)
        .broadcast_as((b, h, lq, lk))?;
        scores = scores.broadcast_add(&mask)?;
    }

    // softmax over the last axis, then multiply by V
    // (manual because candle's softmax_last_dim also requires cuDNN on CUDA)
    let last = scores.dims().len() - 1;
    let max = scores.max_keepdim(last)?;
    let shifted = scores.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(last)?;
    let probs = exp.broadcast_div(&sum)?;
    probs.matmul(&v.contiguous()?)
}

/// `F.normalize(z, dim=-1)` — `model/module.py:246`.
///
/// Unlike `rms_norm` there is no learned scale and the default eps (1e-12)
/// only guards division by zero.
pub fn l2_normalize(z: &Tensor) -> Result<Tensor> {
    // Sums of squares MUST be accumulated in f32 regardless of compute dtype.
    // The tokenizer's encoder emits activations of order 1e4 (larger than
    // f32's 65504 limit for f16 squares: 1e4^2 = 1e8 -> inf -> every value
    // normalises to 0 -> sign(0) = -1 -> every subtoken collapses to id 0
    // (measured: 100% token flips before this fix).
    let sums = z
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_keepdim(last_dim(z))?
        .sqrt()?;
    z.to_dtype(DType::F32)?
        .broadcast_div(&sums)?
        .to_dtype(z.dtype())
}

/// BSQ sign step — `BinarySphericalQuantizer.quantize`, `model/module.py:82-88`.
///
/// Upstream is `torch.where(z > 0, 1, -1)` under a straight-through estimator;
/// with no autograd it is exactly `sign(z)`. Note the tie case: **z == 0 maps to
/// -1**, which is part of the model's quantisation and must be reproduced.
pub fn bsq_sign(z: &Tensor) -> Result<Tensor> {
    // candle comparison ops return U8; cast before the bipolar mapping or the
    // affine would saturate in integer space.
    // `gt`, not `ge`: upstream is torch.where(z > 0, 1, -1), so an exact zero
    // must map to -1. Using `ge` would flip every zero (and -0.0) to +1.
    let zero = Tensor::zeros_like(z)?;
    z.gt(&zero)?.to_dtype(z.dtype())?.affine(2.0, -1.0)
}

/// `BSQuantizer.bits_to_indices` — `model/module.py:234-243`.
///
/// `indices = sum_i (bits[..., i] >= 0) * 2**i` — element 0 of the bit vector is
/// the **least** significant bit. With `s1_bits == s2_bits == 10` this happens
/// to coincide with left-shift packing, but we implement the weighted sum the
/// source defines rather than assuming the coincidence.
pub fn bits_to_indices(bits: &Tensor) -> Result<Tensor> {
    let n = bits.dim(last_dim(bits)[0])?;
    let powers = Tensor::from_vec(
        (0..n).map(|i| (1u64 << i) as f64).collect::<Vec<_>>(),
        n,
        bits.device(),
    )?;
    let mask = bits.ge(&Tensor::zeros_like(bits)?)?.to_dtype(DType::F64)?;
    mask.broadcast_mul(&powers)?.sum(last_dim(bits))
}

/// `KronosTokenizer.indices_to_bits` with `half=True` — `model/module.py:126-140`.
///
/// Inverse of the above: expand each index into LSB-first bits, map to bipolar
/// {-1, +1}, then scale by `1/sqrt(codebook_dim)`.
pub fn indices_to_bits(
    s1: &Tensor,
    s2: &Tensor,
    codebook_dim: usize,
    q_scale: f64,
) -> Result<Tensor> {
    let half = codebook_dim / 2;
    let device = s1.device();
    // Bit i of the index is (index >> i) & 1. We have no integer shift or
    // remainder in candle 0.11, so we compute floor(index / 2^i) and take its
    // low bit -- both floor() calls are required (floor first, then mod 2), or
    // fractional quotients would produce non-binary "bits".
    //
    // NOTE: using `index * 2^i mod 2` instead would be wrong: it yields the low
    // bit of `index` for every position, i.e. a broadcast, not a bit field.
    // Bit i of the index is (index >> i) & 1. We have no integer shift in
    // candle 0.11, so: q = floor(index / 2^i), then take q's low bit with
    // q - 2*floor(q/2). BOTH floors are required, and the masks must be
    // POSITIVE powers of two -- dividing by 2^-i would multiply by 2^i and
    // broadcast the low bit of `index` to every position instead.
    let masks = Tensor::from_vec(
        (0..half).map(|i| (1u64 << i) as f64).collect::<Vec<_>>(),
        half,
        device,
    )?;
    let two = Tensor::from_vec(vec![2.0f64], (), device)?;
    let expand = |idx: &Tensor| -> Result<Tensor> {
        let unsq = idx.unsqueeze(idx.dims().len())?.to_dtype(DType::F64)?;
        let q = unsq.broadcast_div(&masks)?.floor()?;
        let bit = q.broadcast_sub(&q.broadcast_div(&two)?.floor()?.broadcast_mul(&two)?)?;
        bit.to_dtype(DType::F32)?.affine(2.0, -1.0)
    };
    let b1 = expand(s1)?;
    let b2 = expand(s2)?;
    let out = Tensor::cat(&[&b1, &b2], b1.dims().len() - 1)?;
    out.affine(q_scale, 0.0)
}

/// Rebuild `group_codebook` — `model/module.py:76-78`.
///
/// Registered with `persistent=False`, so it is absent from **every** checkpoint
/// and must be reconstructed. It is `indexes_to_codes(arange(2**group_size))
/// [:, -group_size:]`: the trailing `group_size` bits of each group index, in
/// their original order, as bipolar floats.
pub fn build_group_codebook(group_size: usize, device: &Device) -> Result<Tensor> {
    let n = 1usize << group_size;
    let mut rows = Vec::with_capacity(n * group_size);
    for idx in 0..n {
        for i in 0..group_size {
            // basis[j] = 2^(group_size-1-j), so bit i of the code is (idx >> (group_size-1-i)) & 1
            let bit = (idx >> (group_size - 1 - i)) & 1;
            rows.push(if bit == 1 { 1.0f32 } else { -1.0f32 });
        }
    }
    Tensor::from_vec(rows, (n, group_size), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_at_position_zero_is_identity() {
        let dev = Device::Cpu;
        let dim = 8usize;
        let rc = RotaryCache::new(dim, 5, &dev, DType::F32).unwrap();
        let q = Tensor::arange(0f32, dim as f32, &dev)
            .unwrap()
            .reshape((1, 1, 1, dim))
            .unwrap();
        let (qr, _) = rc.apply(&q, &q).unwrap();
        let v = qr.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, got) in v.iter().enumerate() {
            assert!((*got - i as f32).abs() < 1e-6, "index {i}: got {got}");
        }
    }

    /// Distinguishes the GPT-J `rotate_half` convention from GPT-NeoX pair
    /// interleaving. The expected values are computed independently:
    ///
    /// ```text
    /// dim = 4  ->  arange(0,4,2)/dim = [0, 0.5]
    ///             inv_freq = 1/10000^[0,0.5] = [1, 0.01]
    /// position 1  ->  freqs = [1*1, 1*0.01] = [1, 0.01]
    ///             cos = [.540302306, .999950000, .540302306, .999950000]
    ///             sin = [.841470985, .009999833, .841470985, .009999833]
    /// q = [1,2,3,4];  rotate_half(q) = cat(-[3,4], [1,2]) = [-3,-4,1,2]
    /// out = q*cos + rotate_half(q)*sin
    /// ```
    #[test]
    fn rope_uses_rotate_half_not_neox_interleaving() {
        let dev = Device::Cpu;
        let dim = 4usize;
        let rc = RotaryCache::new(dim, 3, &dev, DType::F32).unwrap();
        // position 0 (identity) and position 1 (non-trivial): 8 elements
        let q = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            (1, 1, 2, dim),
            &dev,
        )
        .unwrap();
        let (qr, _) = rc.apply(&q, &q).unwrap();
        let v = qr.flatten_all().unwrap().to_vec1::<f32>().unwrap();

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
        let rc = RotaryCache::new(8, 4, &dev, DType::F32).unwrap();
        let q = Tensor::zeros((1, 1, 5, 8), DType::F32, &dev).unwrap();
        assert!(rc.apply(&q, &q).is_err());
    }

    #[test]
    fn bsq_sign_maps_zero_to_minus_one() {
        let dev = Device::Cpu;
        let z = Tensor::from_vec(vec![-2.0f32, -0.0, 0.0, 1e-30, 3.0], (1, 5), &dev).unwrap();
        let s = bsq_sign(&z).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(s[0], vec![-1.0, -1.0, -1.0, 1.0, 1.0]);
    }

    #[test]
    fn bits_to_indices_is_lsb_first() {
        let dev = Device::Cpu;
        // Bits arrive as +/- q_scale, never a literal zero. Upstream tests
        // `bits >= 0`, so a literal 0.0 would count as set -> use bipolar input.
        let bits = Tensor::from_vec(vec![1.0f32, -1.0, 1.0, -1.0], (1, 4), &dev).unwrap();
        // torch's .sum(-1) drops the axis, so the result is rank-1
        let idx = bits_to_indices(&bits).unwrap().to_vec1::<f64>().unwrap();
        assert_eq!(idx[0], 5.0, "bit0 + bit2 = 1 + 4 under LSB-first packing");

        // A literal zero IS counted as set, matching torch's `>= 0`.
        let zeros = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0], (1, 4), &dev).unwrap();
        let idx = bits_to_indices(&zeros).unwrap().to_vec1::<f64>().unwrap();
        assert_eq!(idx[0], 15.0, "zeros count as set under `>= 0`");
    }

    /// Every released model uses s1_bits = s2_bits = 10, so codebook_dim = 20
    /// and each index spans 0..1024. Round-tripping the full range of both
    /// subtoken spaces is the check that matters.
    #[test]
    fn bits_indices_round_trip_over_the_real_codebook() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        const CODEBOOK_DIM: usize = 20;
        let n = 1usize << 10;
        let idx =
            Tensor::from_vec((0..n).map(|i| i as u32).collect::<Vec<_>>(), (1, n), &dev).unwrap();
        let bits = indices_to_bits(&idx, &idx, CODEBOOK_DIM, 1.0).unwrap();
        assert_eq!(bits.dims(), &[1, n, CODEBOOK_DIM]);

        // half=True produces cat([s1_bits(10), s2_bits(10)], -1), so each subtoken
        // must be recovered from its own slice -- summing the whole 20 columns
        // would just give s1 + 1024*s2.
        let a = idx.to_vec2::<u32>().unwrap();
        for (name, start) in [("s1", 0usize), ("s2", 10usize)] {
            let half = bits.narrow(2, start, 10)?;
            let back = bits_to_indices(&half)?.to_vec2::<f64>().unwrap();
            for i in 0..n {
                assert_eq!(
                    a[0][i] as f64, back[0][i],
                    "{name} index {i} failed to round-trip"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn group_codebook_shape_and_bipolarity() {
        let dev = Device::Cpu;
        for gs in [4usize, 5] {
            let cb = build_group_codebook(gs, &dev).unwrap();
            assert_eq!(cb.dims(), &[1usize << gs, gs]);
            let v: Vec<Vec<f32>> = cb.to_vec2::<f32>().unwrap();
            for row in &v {
                for x in row {
                    assert!(*x == 1.0f32 || *x == -1.0f32, "non-bipolar entry {x}");
                }
            }
            // index 0 is all -1; index 2**gs - 1 is all +1
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
        let x = Tensor::arange(0f32, 16.0, &dev)
            .unwrap()
            .reshape((1, 16))
            .unwrap();
        let w = Tensor::ones((16,), DType::F32, &dev).unwrap();
        let y = rms_norm(&x, &w, 1e-5).unwrap();
        let v = y.to_vec2::<f32>().unwrap();
        let rms = (v[0].iter().map(|x| x * x).sum::<f32>() / 16.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "rms was {rms}");
    }

    #[test]
    fn l2_normalize_gives_unit_norm() {
        let dev = Device::Cpu;
        let z = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 0.0], (1, 4), &dev).unwrap();
        let y = l2_normalize(&z).unwrap().to_vec2::<f32>().unwrap();
        assert!((y[0][0] - 0.6).abs() < 1e-6);
        assert!((y[0][1] - 0.8).abs() < 1e-6);
    }
}
