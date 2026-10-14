//! Nucleus / top-k sampling — `model/kronos.py:331-386`.
//!
//! Two regimes with different parity obligations:
//!
//! * **Deterministic** (`top_k == 1`): upstream filters to the top-1 token, so
//!   `softmax` puts probability exactly 1 on the argmax and `multinomial`
//!   degenerates to argmax. We take the argmax directly — same outcome, no host
//!   sync — and this is what gates the golden CSVs.
//! * **Stochastic** (`top_p < 1.0` or `top_k > 1`): a categorical draw. We do
//!   NOT reproduce torch's Philox stream (the agreed-par is *distributional*,
//!   see `PORTING_FROM_PYTHON.md` §8); uniforms come from a seeded SplitMix64
//!   and are drawn up front, once per trajectory.
//!
//! Phase 5 (batch throughput) moved sampling into **sorted space**, which
//! replaced two Phase-4 costs with one sort:
//!
//! * the Phase-4 nucleus filter built a `[B, V, V]` strict-greater table every
//!   step — 42 MB of intermediate traffic at B=10 — now the filter is a prefix
//!   count over the descending sort, O(B·V·logV);
//! * the Phase-4 sampler moved probabilities to the host per step to run the
//!   inverse CDF — now the whole draw is GPU-side from pre-drawn uniforms.
//!
//! Upstream semantics preserved:
//!  * `filter` RETURNS inside the top-k branch, so `top_p` is ignored whenever
//!    `top_k > 0` (model/kronos.py:347-352);
//!  * the kept set is the **contiguous prefix** of the descending sort: rank r
//!    is kept iff the cumulative mass before it is `<= top_p` — upstream's
//!    one-right-shift rule verbatim (module-style see model/kronos.py:359-369);
//!  * renormalising over that kept prefix is torch's multinomial over the
//!    FILTERED distribution;
//!  * ties: `logits < kth` removal keeps whole tie groups; a descending sort
//!    groups them too, and the k-th-largest threshold lands on the same set.
//!
//! Traversal order is probability-sorted rather than token-id order. The kept
//! set and its renormalised probabilities are identical, so the distribution is
//! unchanged, but individual draws differ from a host-side traversal of the
//! same uniforms. This is precisely the distributional parity we committed to.

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
    /// A `[steps, batch]` table of uniforms, drawn in the exact order the loop
    /// consumes them (step-major, row-minor) so a seeded run is reproducible.
    pub fn uniform_table(&mut self, steps: usize, batch: usize, dev: &Device) -> Result<Tensor> {
        let mut v = Vec::with_capacity(steps * batch);
        for _ in 0..steps {
            for _ in 0..batch {
                v.push(self.next_f64() as f32);
            }
        }
        Tensor::from_vec(v, (steps, batch), dev)
    }
}

/// Numerically-stable softmax over the last axis (avoids the cuDNN-gated op).
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    let last = x.dims().len() - 1;
    let max = x.max_keepdim(last)?;
    let exp = x.broadcast_sub(&max)?.exp()?;
    let den = exp.sum_keepdim(last)?;
    exp.broadcast_div(&den)
}

/// One rank-position helper: `[V]` of 0.0, 1.0, ... as f32.
fn rank_positions(v: u32, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        (0..v).map(|i| i as f32).collect::<Vec<_>>(),
        (v as usize,),
        dev,
    )
}

/// Sorted-space categorical sampling over `[B, V]` logits.
///
/// `draws` must be `[B]` uniforms on the same device — the row order matches
/// the batch (see `Rng::uniform_table`).
pub fn sample(logits: &Tensor, cfg: &SampleCfg, draws: &Tensor) -> Result<Tensor> {
    if cfg.top_k == 1 {
        // argmax: the degenerate multinomial. No host sync, and this is the
        // path that gates the golden CSVs.
        return logits.argmax(logits.dims().len() - 1);
    }
    let last = logits.dims().len() - 1;
    let shape = logits.dims().to_vec();
    let (b, v) = (shape[0], shape[last]);

    // callers usually hand a strided last-position slice; sort needs contiguous
    let scaled = if cfg.temperature == 1.0 {
        logits.contiguous()?
    } else {
        logits
            .affine(1.0f64 / cfg.temperature as f64, 0.0)?
            .contiguous()?
    };

    // descending sort: rank 0 = highest logit
    let (vals, idx) = scaled.sort_last_dim(false)?;
    let probs = softmax(&vals)?;
    let cdf = probs.cumsum(last)?;

    // ---- kept-prefix length K (one scalar per row) ----
    let cdf_excl = Tensor::cat(
        &[
            &Tensor::zeros((b, 1), probs.dtype(), probs.device())?,
            &cdf.narrow(last, 0, v - 1)?,
        ],
        last,
    )?;
    let k_counts = if cfg.top_k > 0 {
        let k = cfg.top_k.min(v as u32) as f32;
        // ranks < k are kept -> prefix length min(k, V)
        k_counts_from_positions(&rank_positions(v as u32, probs.device())?, k as u32, (b, v))?
    } else {
        // rank r kept iff cdf_excl[r] <= top_p; the prefix property then makes
        // the kept set ranks 0..K-1
        let keep = cdf_excl.broadcast_le(
            &Tensor::from_vec(vec![cfg.top_p], (), probs.device())?.to_dtype(probs.dtype())?,
        )?;
        keep.to_dtype(probs.dtype())?.sum(last)?
    };

    // ---- sample the rank within the kept prefix ----
    // renormalise: norm[r] = cdf[r] / cdf[K-1]; pick the LAST rank whose norm
    // is still below the draw (inverse-CDF), clamped into the prefix against
    // rounding at u -> 1.
    let positions = rank_positions(v as u32, probs.device())?.to_dtype(probs.dtype())?;
    let kpos = positions
        .unsqueeze(0)?
        .broadcast_as((b, v))?
        .broadcast_eq(
            &k_counts
                .affine(1.0f64, -1.0)?
                .unsqueeze(1)?
                .to_dtype(probs.dtype())?,
        )?
        .to_dtype(probs.dtype())?; // [B, V] one-hot at K-1
    let total_kept = cdf.broadcast_mul(&kpos)?.sum(last)?.unsqueeze(1)?; // [B, 1]
    let norm = cdf.broadcast_div(&total_kept)?;
    let lt = norm.broadcast_lt(
        &draws
            .to_device(probs.device())?
            .to_dtype(probs.dtype())?
            .unsqueeze(1)?,
    )?;
    let mut rank = lt.to_dtype(probs.dtype())?.sum(last)?; // [B] f32
                                                           // clamp to [0, K-1]
    rank = rank.broadcast_minimum(&k_counts.affine(1.0f64, -1.0)?)?;
    rank = rank.broadcast_maximum(&Tensor::zeros((b,), probs.dtype(), probs.device())?)?;

    // ---- map rank -> original token id through the sort permutation ----
    let onehot_rank = positions
        .unsqueeze(0)?
        .broadcast_as((b, v))?
        .broadcast_eq(&rank.unsqueeze(1)?.to_dtype(probs.dtype())?)?
        .to_dtype(probs.dtype())?;
    // idx values are token ids < 2^24 -> exact in f32
    let picked = idx
        .to_dtype(probs.dtype())?
        .broadcast_mul(&onehot_rank)?
        .sum(last)?;
    picked.to_dtype(DType::U32)
}

/// Position-mask prefix count when the kept set is "the first k ranks".
fn k_counts_from_positions(positions: &Tensor, k: u32, (b, v): (usize, usize)) -> Result<Tensor> {
    let mask = positions
        .unsqueeze(0)?
        .broadcast_as((b, v))?
        .to_dtype(DType::F32)?
        .broadcast_lt(&Tensor::from_vec(vec![k as f32], (), positions.device())?)?;
    mask.to_dtype(DType::F32)?.sum(1)
}

/// Cumulative softmax mass of the tokens *strictly greater* than each given id
/// — the quantity upstream's nucleus test compares against `top_p`.
/// Audit-only (used by the sampler assertions, not in the loop).
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
                                    // gt[b,j,i] = logit_i > logit_j: "how much mass sits above token j".
                                    // The comparison direction is load-bearing — pointing it the other way
                                    // keeps the TAIL tokens instead of the head.
    let gt = row.broadcast_gt(&col)?.to_dtype(logits.dtype())?;
    let mass = gt.broadcast_mul(&p.unsqueeze(1)?)?.sum(2)?.flatten_all()?; // [B, V]
    let mass_cpu = mass
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
        let tiny = Rng::new(0).uniform_table(1, 2, &dev).unwrap();
        let ids = sample(&logits, &cfg, &tiny).unwrap();
        assert_eq!(ids.to_vec1::<u32>().unwrap(), vec![1u32, 2]);
    }

    #[test]
    fn inverse_cdf_picks_the_expected_token() {
        // logits [0,-1,-2] on T=1: p = [.66524094, .24472848, .09003057];
        // cdf rising = [.6652, .9099, 1.0]. u = 0.8 -> rank 1 = token 1.
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, -1.0, -2.0], (1, 3), &dev).unwrap();
        let draws = Tensor::from_vec(vec![0.8f32], (1,), &dev).unwrap();
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let id = sample(&logits, &cfg, &draws)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        assert_eq!(id[0], 1, "u=0.8 must land in token 1's cumulative cell");
        let draws = Tensor::from_vec(vec![0.99f32], (1,), &dev).unwrap();
        let id = sample(&logits, &cfg, &draws)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        assert_eq!(id[0], 2, "u=0.99 must land in token 2's cell");
    }

    #[test]
    fn nucleus_excludes_the_tail() {
        // p = [.6652, .2447, .0900]; cdf_excl = [0, .6652, .9099].
        // top_p = 0.7: ranks 0,1 kept (0 and .6652 <= .7); rank 2 excluded.
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, -1.0, -2.0], (1, 3), &dev).unwrap();
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.7,
        };
        let _draws = Rng::new(1).uniform_table(64, 1, &dev).unwrap();
        for u in 0..64u64 {
            let draw = Tensor::from_vec(vec![(u as f64 + 0.5) / 64.0], (1,), &dev).unwrap();
            let id = sample(&logits, &cfg, &draw)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_ne!(id[0], 2, "token 2 sits outside a top_p=0.7 nucleus");
        }
    }

    #[test]
    fn nucleus_never_drops_the_head() {
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, -1.0, -2.0], (1, 3), &dev).unwrap();
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.7,
        };
        // u=0.0 + eps must still land on rank 0 (the only token under any
        // strict-first-kept-cdf reading)
        let draw = Tensor::from_vec(vec![0.001f32], (1,), &dev).unwrap();
        let id = sample(&logits, &cfg, &draw)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        assert_eq!(id[0], 0);
    }

    #[test]
    fn topk2_removes_the_smallest_only() {
        // logits [3,2,1], top_k=2 -> kept set {3.0, 2.0}; token 2 must NEVER
        // appear over a sweep of all uniforms.
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(vec![3.0f32, 2.0, 1.0], (1, 3), &dev).unwrap();
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        for i in 0..64u64 {
            let draw = Tensor::from_vec(vec![(i as f64 + 0.5) / 64.0], (1,), &dev).unwrap();
            let id = sample(&logits, &cfg, &draw)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert!(id[0] == 0 || id[0] == 1, "draw {i}: got token {}", id[0]);
        }
    }

    #[test]
    fn filtered_renormalisation_covers_the_kept_set() {
        // top_p just under the cumulative boundary: kept = {0,1}; draws must
        // still cover BOTH kept tokens all the way to u -> 1 - eps.
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, -1.0, -2.0], (1, 3), &dev).unwrap();
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.7,
        };
        let draw = Tensor::from_vec(vec![0.999f32], (1,), &dev).unwrap();
        let id = sample(&logits, &cfg, &draw)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        assert_eq!(
            id[0], 1,
            "renormalised tail of the kept set must be reachable"
        );
    }

    #[test]
    fn uniform_table_shape_and_reproducibility() {
        let dev = Device::Cpu;
        let a = Rng::new(123).uniform_table(3, 4, &dev).unwrap();
        let b = Rng::new(123).uniform_table(3, 4, &dev).unwrap();
        assert_eq!(a.dims(), &[3, 4]);
        assert_eq!(a.to_vec2::<f32>().unwrap(), b.to_vec2::<f32>().unwrap());
        // row-major step-major order matches the loop's consumption order
        let first_step = a
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(first_step.iter().all(|x| (0.0..1.0).contains(x)));
    }
}
