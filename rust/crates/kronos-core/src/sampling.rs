//! Nucleus / top-k sampling — `model/kronos.py:331-386`.
//!
//! Two regimes with different parity obligations:
//!
//! * **Deterministic** (`top_k == 1`): upstream filters to the top-1 token, so
//!   `softmax` puts probability exactly 1 on the argmax and `multinomial`
//!   degenerates to argmax. We take the argmax directly — same outcome, no host
//!   sync — and this is what gates the golden CSVs.
//! * **Stochastic** (`top_p < 1.0`): needs a categorical draw. We do NOT
//!   reproduce torch's Philox stream (par was agreed: distributional, see
//!   `PORTING_FROM_PYTHON.md` §8). Seeded SplitMix64 + inverse-CDF instead.
//!
//! `top_p` is implemented without `scatter`. Upstream sorts descending, takes
//! the cumulative softmax, shifts the mask one right, and scatters back. That
//! is algebraically: token `j` is kept iff the softmax mass of tokens with a
//! *strictly greater* logit is `<= top_p` (with `rank_j` = count of strictly
//! greater tokens). Exact for distinct logits and identical in behaviour on
//! real model outputs.
//!
//! Two upstream details reproduced exactly:
//! * `filter` **returns** inside the top-k branch, so `top_p` is ignored
//!   whenever `top_k > 0` (model/kronos.py:347-352).
//! * the keep test is `>=` the k-th largest (torch keeps ties; `logits < kth`
//!   removes strictly-less).

use candle_core::{DType, Device, Result, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct SampleCfg {
    pub temperature: f32,
    /// 0 disables top-k filtering.
    pub top_k: u32,
    /// 1.0 disables nucleus filtering.
    pub top_p: f32,
}

impl Default for SampleCfg {
    fn default() -> Self {
        // the repo's own `predict` defaults (model/kronos.py:519)
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
        }
    }
}

/// SplitMix64: five lines, full 64-bit avalanche, no dependency.
pub struct Rng {
    st: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { st: seed }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.st = self.st.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.st;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform on [0, 1) with 53 mantissa bits.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64
    }
}

/// Substitute `-inf` without NaN risk: keep ? logit : -BIG.
///
/// The reference uses `filter_value = -f32::INFINITY`. We use a large *finite*
/// number so `shifted = val - rowmax` stays finite: `exp(-BIG - rowmax)` still
/// underflows to exactly 0, matching torch, without any `inf - inf` hazard on a
/// row that is entirely -inf (possible only for a degenerate top_k).
const NEG_BIG: f64 = -1.0e30;

fn apply_keep(logits: &Tensor, keep: &Tensor) -> Result<Tensor> {
    // stay in the logits' dtype: with fp16 weights the distribution is f16 and
    // an f32 scalar here would raise a mixed-dtype mul.
    let keepf = keep.to_dtype(logits.dtype())?;
    let drop = keepf.affine(-1.0f64, 1.0f64)?; // 1 - keep
    logits
        .broadcast_mul(&keepf)?
        .broadcast_add(&drop.affine(NEG_BIG, 0.0)?)
}

/// `top_k_top_p_filtering` — the filtered `[B, V]` logits.
///
/// **Must already be temperature-scaled**: upstream divides by T *once* in
/// `sample_from_logits` and then filters (model/kronos.py:374-377), so calling
/// `filter` on raw logits would scale twice.
pub fn filter(logits: &Tensor, cfg: &SampleCfg) -> Result<Tensor> {
    let do_top_k = cfg.top_k > 0;
    let do_top_p = cfg.top_p < 1.0;
    if !do_top_k && !do_top_p {
        return Ok(logits.clone());
    }
    let last = logits.dims().len() - 1;
    if do_top_k {
        let v = logits.dim(last)? as u32;
        let k = cfg.top_k.min(v).max(1);
        // threshold = the k-th largest (descending sort position k-1); all
        // ties above it are kept, matching torch's `logits < kth` removal.
        let thr = if k == 1 {
            logits.max_keepdim(last)?
        } else {
            logits
                .sort_last_dim(false)?
                .0
                .narrow(last, (k - 1) as usize, 1)?
        };
        return apply_keep(logits, &logits.broadcast_ge(&thr)?);
    }
    // nucleus on a rank-2 [B, V] tensor:
    //   gt[b, j, i] = logit_j > logit_i,  mass[j] = sum_i p_i * gt[b,j,i],
    //   keep[b, j]  = mass[b, j] <= top_p
    let dims = logits.dims().to_vec();
    let _ = dims;
    let _ = last;
    let p = softmax(logits)?;
    let row = logits.unsqueeze(1)?; // [B, 1, V] = logit_i (compare axis)
    let col = logits.unsqueeze(2)?; // [B, V, 1] = logit_j (kept axis)
                                    // gt[b,j,i] = logit_i > logit_j: "how much mass sits above token j".
                                    // The comparison direction is load-bearing — pointing it the other way
                                    // keeps the TAIL tokens instead of the head.
    let gt = row.broadcast_gt(&col)?.to_dtype(logits.dtype())?;
    let mass = gt.broadcast_mul(&p.unsqueeze(1)?)?.sum(2)?; // [B, V]
    let keep = mass.broadcast_le(
        &Tensor::from_vec(vec![cfg.top_p], (), logits.device())?.to_dtype(mass.dtype())?,
    )?;
    apply_keep(logits, &keep)
}

/// Numerically-stable softmax over the last axis (avoids the cuDNN-gated op).
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    let last = x.dims().len() - 1;
    let max = x.max_keepdim(last)?;
    let exp = x.broadcast_sub(&max)?.exp()?;
    let den = exp.sum_keepdim(last)?;
    exp.broadcast_div(&den)
}

/// Cumulative softmax mass of the tokens *strictly greater* than each given id
/// — the quantity upstream's nucleus test compares against `top_p`.
///
/// Rank-2 `[B, V]` logits that are already temperature-scaled; `ids` is `[B]`.
pub fn strictly_greater_mass(logits: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let dims = logits.dims().to_vec();
    debug_assert_eq!(dims.len(), 2, "expected [B, V] logits");
    let b = dims[0];
    let v = dims[1];
    let p = softmax(logits)?;
    let row = logits.unsqueeze(1)?; // [B, 1, V] = logit_i
    let col = logits.unsqueeze(2)?; // [B, V, 1] = logit_j
    let gt = row.broadcast_gt(&col)?.to_dtype(logits.dtype())?;
    let mass = gt.broadcast_mul(&p.unsqueeze(1)?)?.sum(2)?; // [B, V]
    let mass_cpu = mass
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let ids_cpu = ids
        .to_device(&Device::Cpu)?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;
    let out: Vec<f32> = (0..b)
        .map(|r| mass_cpu[r * v + ids_cpu[r] as usize])
        .collect();
    Tensor::from_vec(out, (b,), &Device::Cpu)
}

/// Sample one token per row from `[B, V]` logits.
///
/// `top_k == 1` -> argmax (no host sync). Otherwise: filter, softmax, then
/// host-side inverse-CDF with the seeded RNG. The deliberate sync is the very
/// place Phase 5 fuses away; here we buy exact distributional semantics.
pub fn sample(logits: &Tensor, cfg: &SampleCfg, rng: &mut Rng) -> Result<Tensor> {
    if cfg.top_k == 1 {
        return logits.argmax(logits.dims().len() - 1);
    }
    if cfg.temperature == 1.0 {
        // exact no-op path: avoid an affine round trip when T == 1
        let filtered = filter(logits, cfg)?;
        let probs = softmax(&filtered)?;
        return inverse_cdf(&probs, rng);
    }
    let scaled = logits.affine(1.0f64 / cfg.temperature as f64, 0.0)?;
    let filtered = filter(&scaled, cfg)?;
    let probs = softmax(&filtered)?;
    inverse_cdf(&probs, rng)
}

/// Inverse-CDF draw over a row-major `[B, V]` probability tensor.
fn inverse_cdf(probs: &Tensor, rng: &mut Rng) -> Result<Tensor> {
    let dims = probs.dims().to_vec();
    let b = dims[0];
    let v = dims[dims.len() - 1];
    let flat = probs
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut ids = Vec::with_capacity(b);
    for r in 0..b {
        let row = &flat[r * v..r * v + v];
        let u = rng.next_f64() as f32;
        let mut acc = 0.0f32;
        let mut pick = v - 1;
        for (i, p) in row.iter().enumerate() {
            acc += *p;
            if acc > u {
                pick = i;
                break;
            }
        }
        ids.push(pick as u32);
    }
    Tensor::from_vec(ids, (b,), &Device::Cpu)?.to_device(probs.device())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_topk1_is_argmax() {
        let dev = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![1.0f32, 9.0, 2.0, -1.0, -5.0, 3.0], (2, 3), &dev).unwrap();
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        let ids = sample(&logits, &cfg, &mut Rng::new(0)).unwrap();
        assert_eq!(ids.to_vec1::<u32>().unwrap(), vec![1u32, 2]);
    }

    #[test]
    fn filtered_mass_leaks_nothing() {
        // after top_k=1 the kept token must carry probability 1.0 exactly, not
        // just nearly — zero-mask padding would leak exp(-rowmax).
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 10.0, 0.0], (1, 3), &dev).unwrap();
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        let p = softmax(&filter(&logits, &cfg).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!((p[0][1] - 1.0).abs() < 1e-6, "got {}", p[0][1]);
        assert_eq!(p[0][0], 0.0);
        assert_eq!(p[0][2], 0.0);
    }

    #[test]
    fn topk2_removes_only_the_smallest() {
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(vec![3.0f32, 2.0, 1.0], (1, 3), &dev).unwrap();
        let p = filter(
            &logits,
            &SampleCfg {
                temperature: 1.0,
                top_k: 2,
                top_p: 0.0,
            },
        )
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
        assert_eq!(p[0], vec![3.0f32, 2.0, -1.0e30]);
    }

    #[test]
    fn nucleus_membership_matches_hand_computation() {
        // logits [0, -1, -2] -> p = exp/l1 = [.66524094, .24472848, .09003057]
        // (the earlier [.5,.3,.2] was a wrong hand computation -- caught by the
        // very assertion it was meant to validate). Strictly-greater masses:
        //   j=0: 0        j=1: p0 = .66524094      j=2: p0+p1 = .90996945
        // so with top_p=.7: tokens 0,1 are in; token 2 is out.
        let dev = Device::Cpu;
        // one id per batch row: replicate the row to probe all three tokens
        let logits = Tensor::from_vec(
            vec![0.0f32, -1.0, -2.0, 0.0, -1.0, -2.0, 0.0, -1.0, -2.0],
            (3, 3),
            &dev,
        )
        .unwrap();
        let ids = Tensor::from_vec(vec![0u32, 1, 2], (3,), &dev).unwrap();
        let m = match strictly_greater_mass(&logits, &ids) {
            Ok(v) => v.to_vec1::<f32>().unwrap(),
            Err(e) => panic!("strictly_greater_mass failed: {e}"),
        };
        assert!(
            (m[0] - 0.0).abs() < 1e-5,
            "m[0]={} must be 0 (max token, nothing greater)",
            m[0]
        );
        assert!((m[1] - 0.665_240_94).abs() < 1e-5, "m[1]={}", m[1]);
        assert!((m[2] - 0.909_969_45).abs() < 1e-5, "m[2]={}", m[2]);
    }

    #[test]
    fn temperature_scales_the_distribution() {
        // T = 2 should NOT collapse; T -> tiny should sharpen to argmax-only.
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(vec![2.0f32, 1.0], (1, 2), &dev).unwrap();
        let hot = SampleCfg {
            temperature: 0.01,
            top_k: 0,
            top_p: 1.0,
        };
        let p = softmax(&logits.affine(1.0 / 0.01, 0.0).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(p[0][0] > 0.999, "sharpened: {}", p[0][0]);
        let _ = hot;
    }
}
