//! Nucleus / top-k sampling — `model/kronos.py:331-386`.
//!
//! Two regimes with different parity obligations:
//!
//! * **Deterministic** (`top_k == 1`): upstream filters to the top-1 token, so
//!   `softmax` puts probability exactly 1 on the argmax and `multinomial`
//!   degenerates to argmax. We take the argmax directly — same outcome, no host
//!   sync — and this is what gates the golden CSVs.
//! * **Stochastic** (`top_p < 1.0` or `top_k > 1`): a categorical draw. We do
//!   NOT reproduce torch's Philox stream (the agreed parity is *distributional*,
//!   see `PORTING_FROM_PYTHON.md` §8); uniforms come from a seeded SplitMix64
//!   and are drawn up front, once per trajectory.
//!
//! Phase 5 moved sampling into **sorted space** (O(B·V·logV), GPU-side draws);
//! Phase 6 keeps that algorithm on the torch substrate. Upstream semantics
//! preserved:
//!  * `filter` RETURNS inside the top-k branch, so `top_p` is ignored whenever
//!    `top_k > 0` (model/kronos.py:347-352);
//!  * the kept set is the **contiguous prefix** of the descending sort: rank r
//!    is kept iff the cumulative mass before it is `<= top_p` (module-style
//!    one-right-shift rule, model/kronos.py:359-369);
//!  * renormalising over that kept prefix is torch's multinomial over the
//!    FILTERED distribution;
//!  * ties: `logits < kth` removal keeps whole tie groups; a descending sort
//!    groups them too, and the k-th-largest threshold lands on the same set.
//!
//! Traversal order is probability-sorted rather than token-id order, so
//! individual draws differ from a host-side traversal of the same uniforms —
//! the distributional parity we committed to.

use crate::ops::Result;
use tch::{Device, Kind, Tensor};

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
        Ok(Tensor::from_slice(&v)
            .reshape([steps as i64, batch as i64])
            .to_device(*dev))
    }
}

/// Numerically-stable softmax over the last axis (longhand; the fused
/// form is what torch's own ops call — this is for clarity and audits).
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    let last = x.size().len() as i64 - 1;
    let max = x.f_amax(vec![last], true)?;
    let exp = (x - max).f_exp()?;
    let den = exp.f_sum_dim_intlist(vec![last], true, None)?;
    Ok(exp / den)
}

/// One rank-position helper: `[V]` of 0.0, 1.0, ... as f32.
fn rank_positions(v: u32, dev: &Device) -> Result<Tensor> {
    Tensor::f_arange_start_step(0.0, v as f64, 1.0, (Kind::Float, *dev))
}

/// Sorted-space categorical sampling over `[B, V]` logits.
///
/// `draws` must be `[B]` uniforms on the same device — the row order matches
/// the batch (see `Rng::uniform_table`).
pub fn sample(logits: &Tensor, cfg: &SampleCfg, draws: &Tensor) -> Result<Tensor> {
    if cfg.top_k == 1 {
        // argmax: the degenerate multinomial. No host sync, and this is the
        // path that gates the golden CSVs. Kind::Int because torch's id
        // vocabulary here is i32/i64, not u32.
        return Ok(logits.f_argmax(-1, false)?.to_kind(Kind::Int));
    }
    let dim = (logits.size().len() - 1) as i64;
    let size = logits.size();
    let b = size[0];
    let v = size[dim as usize];

    // callers usually hand a strided last-position slice; sort/kernels want
    // contiguous input
    let scaled = if cfg.temperature == 1.0 {
        logits.contiguous()
    } else {
        (logits * (1.0f64 / cfg.temperature as f64)).contiguous()
    };

    // descending sort: rank 0 = highest logit
    let (vals, idx) = scaled.f_sort(dim, true)?;
    let probs = softmax(&vals)?;
    let cdf = probs.f_cumsum(dim, None)?;

    // ---- kept-prefix length K (one scalar per row) ----
    let cdf_excl = Tensor::cat(
        &[
            &Tensor::zeros([b, 1], (probs.kind(), probs.device())),
            &cdf.narrow(dim, 0, v - 1),
        ],
        dim,
    );
    let k_counts = if cfg.top_k > 0 {
        let k = cfg.top_k.min(v as u32) as f64;
        // ranks < k are kept -> prefix length min(k, V)
        let positions = rank_positions(v as u32, &probs.device())?;
        let mask = positions.unsqueeze(0).expand([b, v], false);
        mask.f_lt(k)?.f_sum_dim_intlist(vec![dim], false, None)?
    } else {
        // rank r kept iff cdf_excl[r] <= top_p; the prefix property then makes
        // the kept set ranks 0..K-1
        cdf_excl
            .f_le(cfg.top_p as f64)?
            .f_sum_dim_intlist(vec![dim], false, None)?
    };
    let kcounts_f = k_counts.to_kind(Kind::Float);

    // ---- sample the rank within the kept prefix ----
    // renormalise: norm[r] = cdf[r] / cdf[K-1]; pick the LAST rank whose norm
    // is still below the draw (inverse-CDF), clamped into the prefix against
    // rounding at u -> 1.
    let positions = rank_positions(v as u32, &probs.device())?;
    // one-hot of kept_last (K-1) over the rank axis: [B, V]
    // `kpos` MUST be [B, 1]: a bare [B] right-aligns to [1, B] and only
    // broadcasts legally when B == 1, which is all the old gates ever ran.
    let kpos = (&kcounts_f - 1.0).unsqueeze(1);
    let konehot = positions
        .unsqueeze(0)
        .expand([b, v], false)
        .f_eq_tensor(&kpos)?
        .to_kind(Kind::Float);
    let total_kept = cdf
        .multiply(&konehot)
        .f_sum_dim_intlist(vec![dim], false, None)?
        .unsqueeze(1); // [B, 1]
    let norm = cdf / total_kept;
    let dval = draws
        .f_to(probs.device())?
        .to_kind(Kind::Float)
        .unsqueeze(1);
    let lt = norm.f_lt_tensor(&dval)?;
    let rank = lt
        .to_kind(Kind::Float)
        .f_sum_dim_intlist(vec![dim], false, None)?;
    // clamp to [0, K-1]
    let kmax = kcounts_f - 1.0;
    let rank = rank.f_clamp_max_tensor(&kmax)?;
    let rank = rank.f_clamp_min(0.0)?;

    // ---- map rank -> original token id through the sort permutation ----
    let onehot_rank = positions
        .unsqueeze(0)
        .expand([b, v], false)
        .f_eq_tensor(&rank.unsqueeze(1))?
        .to_kind(Kind::Float);
    // idx values are token ids < 2^24 -> exact in f32
    let picked = idx
        .to_kind(Kind::Float)
        .multiply(&onehot_rank)
        .f_sum_dim_intlist(vec![dim], false, None)?;
    Ok(picked.to_kind(Kind::Int))
}

/// One-hot equality flag per row: `[B, V]` with 1 at the given rank.
fn _kept_used_docs() {}

/// Position-mask prefix count when the kept set is "the first k ranks".
#[allow(dead_code)]
fn k_counts_from_positions(positions: &Tensor, k: u32, b: i64, v: i64) -> Result<Tensor> {
    positions
        .unsqueeze(0)
        .expand([b, v], false)
        .f_lt(k as f64)?
        .f_sum_dim_intlist(vec![1], false, None)
}

/// Cumulative softmax mass of the tokens *strictly greater* than each given id
/// — the quantity upstream's nucleus test compares against `top_p`.
/// Audit-only (used by the sampler assertions, not in the loop).
///
/// Rank-2 `[B, V]` logits that are already temperature-scaled; `ids` is `[B]`.
pub fn strictly_greater_mass(logits: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let ids_cpu = ids.to_device(Device::Cpu).to_kind(Kind::Int);
    let ids_v: Vec<i32> = (&ids_cpu).try_into().map_err(tch_err)?;
    let dims = logits.size();
    debug_assert_eq!(dims.len(), 2, "expected [B, V] logits");
    let b = dims[0];
    let v = dims[1];
    let p = softmax(logits)?;
    let row = logits.unsqueeze(1); // [B, 1, V] = logit_i
    let col = logits.unsqueeze(2); // [B, V, 1] = logit_j
                                   // gt[b,j,i] = logit_i > logit_j: "how much mass sits above token j".
                                   // The comparison direction is load-bearing — pointing it the other way
                                   // keeps the TAIL tokens instead of the head.
    let gt = row.f_gt_tensor(&col)?.to_kind(logits.kind());
    let p_exp = p.unsqueeze(1);
    let mass = gt
        .multiply(&p_exp)
        .f_sum_dim_intlist(vec![2_i64], false, None)?
        .reshape([-1]); // [B, V]
    let mass_cpu = mass.to_kind(Kind::Float).to_device(Device::Cpu);
    let mass_v: Vec<f32> = (&mass_cpu).try_into().map_err(tch_err)?;
    let out: Vec<f32> = (0..b as usize)
        .map(|r| mass_v[r * v as usize + ids_v[r] as usize])
        .collect();
    Ok(Tensor::from_slice(&out))
}

/// Adapter for the `TryFrom<&Tensor>` conversion error into `TchError`.
fn tch_err(e: tch::TchError) -> tch::TchError {
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_sampling_is_row_independent() {
        // Regression gate for a latent cross-row bug: `kpos` was [B] where the
        // comparison needs [B, 1], which broadcasts legally only when B == 1.
        // Every pre-tch gate ran B == 1 (goldens are top_k=1; the audit draws
        // 256 rows of ONE series), so the bug shipped. Sampling a row alone
        // must equal sampling it inside a batch, for every row and every cfg.
        let (b, v) = (8i64, 64i64);
        // deterministic, well-separated logits
        let mut data = Vec::with_capacity((b * v) as usize);
        for i in 0..(b * v) {
            data.push(((i * 37 % 19) as f32) - 9.0 + (i % b) as f32 * 0.5);
        }
        let logits = Tensor::from_slice(&data).reshape([b, v]);
        let draws = Tensor::from_slice(&[0.05f32, 0.21, 0.33, 0.44, 0.55, 0.66, 0.77, 0.91]);
        for cfg in [
            SampleCfg {
                temperature: 0.6,
                top_k: 0,
                top_p: 0.9,
            },
            SampleCfg {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.7,
            },
            SampleCfg {
                temperature: 1.0,
                top_k: 5,
                top_p: 0.9,
            },
            SampleCfg {
                temperature: 1.3,
                top_k: 0,
                top_p: 1.0,
            },
        ] {
            let batched = sample(&logits, &cfg, &draws).unwrap();
            let vids: Vec<i32> = (&batched).try_into().unwrap();
            for i in 0..b {
                let row = sample(
                    &logits.shallow_clone().narrow(0, i, 1),
                    &cfg,
                    &draws.shallow_clone().narrow(0, i, 1),
                )
                .unwrap();
                let v1: Vec<i32> = (&row).try_into().unwrap();
                assert_eq!(
                    vids[i as usize], v1[0],
                    "row {i} differs between batch and single-row sampling (T={} top_k={} top_p={})",
                    cfg.temperature, cfg.top_k, cfg.top_p
                );
                assert!(
                    (0..v as i32).contains(&vids[i as usize]),
                    "sampled id out of range for row {i}"
                );
            }
        }
    }

    #[test]
    fn deterministic_topk1_is_argmax() {
        let dev = Device::Cpu;
        let logits = Tensor::from_slice(&[1.0f32, 9.0, 2.0, -1.0, -5.0, 3.0]).reshape([2, 3]);
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        let tiny = Rng::new(0).uniform_table(1, 2, &dev).unwrap();
        let ids = sample(&logits, &cfg, &tiny).unwrap();
        let v: Vec<i32> = (&ids).try_into().unwrap();
        assert_eq!(v, vec![1i32, 2]);
    }

    #[test]
    fn inverse_cdf_picks_the_expected_token() {
        // logits [0,-1,-2] on T=1: p = [.66524094, .24472848, .09003057];
        // cdf rising = [.6652, .9099, 1.0]. u = 0.8 -> rank 1 = token 1.
        let _dev = Device::Cpu;
        let logits = Tensor::from_slice(&[0.0f32, -1.0, -2.0]).reshape([1, 3]);
        let draws = Tensor::from_slice(&[0.8f32]);
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let id = sample(&logits, &cfg, &draws).unwrap();
        let v: Vec<i32> = (&id).try_into().unwrap();
        assert_eq!(v[0], 1, "u=0.8 must land in token 1's cumulative cell");
        let draws = Tensor::from_slice(&[0.99f32]);
        let id = sample(&logits, &cfg, &draws).unwrap();
        let v: Vec<i32> = (&id).try_into().unwrap();
        assert_eq!(v[0], 2, "u=0.99 must land in token 2's cell");
    }

    #[test]
    fn nucleus_excludes_the_tail() {
        // p = [.6652, .2447, .0900]; cdf_excl = [0, .6652, .9099].
        // top_p = 0.7: ranks 0,1 kept (0 and .6652 <= .7); rank 2 excluded.
        let _dev = Device::Cpu;
        let logits = Tensor::from_slice(&[0.0f32, -1.0, -2.0]).reshape([1, 3]);
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.7,
        };
        for i in 0..64u64 {
            let draw = Tensor::from_slice(&[((i as f64 + 0.5) / 64.0) as f32]);
            let id = sample(&logits, &cfg, &draw).unwrap();
            let v: Vec<i32> = (&id).try_into().unwrap();
            assert_ne!(v[0], 2, "token 2 sits outside a top_p=0.7 nucleus");
        }
    }

    #[test]
    fn nucleus_never_drops_the_head() {
        let _dev = Device::Cpu;
        let logits = Tensor::from_slice(&[0.0f32, -1.0, -2.0]).reshape([1, 3]);
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.7,
        };
        // u=0.0 + eps must still land on rank 0 (the only token under any
        // strict-first-kept-cdf reading)
        let draw = Tensor::from_slice(&[0.001f32]);
        let id = sample(&logits, &cfg, &draw).unwrap();
        let v: Vec<i32> = (&id).try_into().unwrap();
        assert_eq!(v[0], 0);
    }

    #[test]
    fn topk2_removes_the_smallest_only() {
        // logits [3,2,1], top_k=2 -> kept set {3.0, 2.0}; token 2 must NEVER
        // appear over a sweep of all uniforms.
        let _dev = Device::Cpu;
        let logits = Tensor::from_slice(&[3.0f32, 2.0, 1.0]).reshape([1, 3]);
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        for i in 0..64u64 {
            let draw = Tensor::from_slice(&[((i as f64 + 0.5) / 64.0) as f32]);
            let id = sample(&logits, &cfg, &draw).unwrap();
            let v: Vec<i32> = (&id).try_into().unwrap();
            assert!(v[0] == 0 || v[0] == 1, "draw {i}: got token {}", v[0]);
        }
    }

    #[test]
    fn filtered_renormalisation_covers_the_kept_set() {
        // top_p just under the cumulative boundary: kept = {0,1}; draws must
        // still cover BOTH kept tokens all the way to u -> 1 - eps.
        let _dev = Device::Cpu;
        let logits = Tensor::from_slice(&[0.0f32, -1.0, -2.0]).reshape([1, 3]);
        let cfg = SampleCfg {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.7,
        };
        let draw = Tensor::from_slice(&[0.999f32]);
        let id = sample(&logits, &cfg, &draw).unwrap();
        let v: Vec<i32> = (&id).try_into().unwrap();
        assert_eq!(
            v[0], 1,
            "renormalised tail of the kept set must be reachable"
        );
    }

    #[test]
    fn uniform_table_shape_and_reproducibility() {
        let dev = Device::Cpu;
        let a = Rng::new(123).uniform_table(3, 4, &dev).unwrap();
        let b = Rng::new(123).uniform_table(3, 4, &dev).unwrap();
        assert_eq!(a.size(), [3, 4]);
        let ac = a.reshape([-1]).to_device(Device::Cpu);
        let bc = b.reshape([-1]).to_device(Device::Cpu);
        let av: Vec<f32> = (&ac).try_into().unwrap();
        let bv: Vec<f32> = (&bc).try_into().unwrap();
        assert_eq!(av, bv);
        assert!(av.iter().all(|x| (0.0..1.0).contains(x)));
    }
}
