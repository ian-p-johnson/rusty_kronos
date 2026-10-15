//! Autoregressive generation — `model/kronos.py:389-469`.
//!
//! Parity-relevant property (documented in `PORTING_FROM_PYTHON.md` §4.3):
//! upstream maintains a ring buffer and rolls it by -1 each step, so RoPE
//! positions are **re-assigned 0..L-1 to a different prefix every step**. A KV
//! cache is therefore impossible, and the roll must be reproduced exactly.
//!
//! We avoid `torch.roll` and `slice_assign` by keeping the token window itself
//! (a sliding stream), which produces the identical sequence of windows — the
//! `windows_match_ring_buffer` test pins that down.
//!
//! Upstream branch structure, reproduced verbatim:
//!
//! | condition | input this step | buffer update |
//! |---|---|---|
//! | `current <= max_context` | `buffer[:, :window_len]` | append at index `current` |
//! | `current <  max_context` | (same)          | (same) |
//! | `current >= max_context` | full buffer     | **drop oldest, append new** |

use crate::data::Kline;
use crate::ops::Result;
use crate::predictor::Kronos;
use crate::sampling::{sample, Rng, SampleCfg};
use crate::tokenizer::KronosTokenizer;
use tch::{Device, Kind, Tensor};

/// Everything the loop needs beyond the two models.
#[derive(Debug, Clone, Copy)]
pub struct ArParams {
    pub max_context: usize,
    pub pred_len: usize,
    pub clip: f32,
    pub cfg: SampleCfg,
    pub sample_count: usize,
}

impl ArParams {
    /// Convenience constructor used by the CLI.
    pub fn new(max_context: usize, pred_len: usize, samples: usize, cfg: SampleCfg) -> Self {
        Self {
            max_context,
            pred_len,
            clip: 5.0,
            cfg,
            sample_count: samples,
        }
    }
}

impl Default for ArParams {
    fn default() -> Self {
        Self {
            max_context: 512,
            pred_len: 8,
            clip: 5.0,
            cfg: SampleCfg::default(),
            sample_count: 1,
        }
    }
}

/// Repeat row-block copies: `[n, T, D] -> [n*S, T, D]`, row-major over `(n, s)`,
/// so the S samples of one series are contiguous. Matches
/// `x.unsqueeze(1).repeat(1, S, 1, 1).reshape(-1, T, D)`.
fn repeat_series(t: &Tensor, s: usize) -> Result<Tensor> {
    if s == 1 {
        return Ok(t.shallow_clone());
    }
    let n = t.size()[0];
    let mut idx: Vec<i64> = Vec::with_capacity(n as usize * s);
    for i in 0..n {
        for _ in 0..s {
            idx.push(i);
        }
    }
    let idx = Tensor::from_slice(&idx).to_device(t.device());
    Ok(t.index_select(0, &idx).contiguous())
}

/// Inputs shaped exactly as `KronosPredictor.generate` produces them.
pub struct ArInput {
    /// normalised, clipped `[1, T, 6]`
    pub x: Tensor,
    /// `[1, T, 5]`
    pub x_stamp: Tensor,
    /// `[1, pred_len, 5]`
    pub y_stamp: Tensor,
}

/// The full `predict` result: the de-normalised forecast plus the exact
/// sampled subtoken ids (consumed by the precision report's path lens).
pub struct PredictOutput {
    /// `[n, pred_len, 6]` de-normalised OHLCVA
    pub forecast: Tensor,
    /// `[n*S, pred_len]` sampled coarse-subtoken ids
    pub sampled_pre: Tensor,
    /// `[n*S, pred_len]` sampled fine-subtoken ids
    pub sampled_post: Tensor,
}

pub struct ArOutput {
    /// `[n, window, 6]` — the decoded K-lines, sample-averaged
    pub z: Tensor,
    /// `[n*S, pred_len]` coarse-subtoken ids, exactly as sampled
    pub sampled_pre: Tensor,
    /// `[n*S, pred_len]` fine-subtoken ids, exactly as sampled
    pub sampled_post: Tensor,
}

/// Gap between the two best logits at one position: the near-tie detector.
/// A greedy token flip between two engines is only concerning when this gap is
/// large; a flip at a gap of ~1e-3 on a logit scale of tens is arithmetic order.
fn top2_margin(logits: &Tensor) -> Result<f32> {
    let last = (logits.size().len() - 1) as i64;
    let (vals, _) = logits.f_sort(last, true)?; // descending: rank0 = highest
    let host = vals
        .reshape([-1])
        .to_device(Device::Cpu)
        .to_kind(Kind::Float);
    let v: Vec<f32> = (&host).try_into()?;
    Ok(v[0] - v[1])
}

/// The loop. Verbatim port of `auto_regressive_inference`.
pub fn generate(
    tok: &KronosTokenizer,
    model: &Kronos,
    input: &ArInput,
    p: &ArParams,
    rng: &mut Rng,
) -> Result<ArOutput> {
    // Cast the input to the models' compute dtype: with fp16 weights the F32
    // activations would otherwise hit a mixed-dtype matmul and error out.
    let dt = tok.dtype();
    let x = repeat_series(&input.x, p.sample_count)?.to_kind(dt);
    let x_stamp = repeat_series(&input.x_stamp, p.sample_count)?.to_kind(dt);
    let y_stamp = repeat_series(&input.y_stamp, p.sample_count)?.to_kind(dt);

    let profile = std::env::var("KRONOS_PROFILE").is_ok();
    let t0 = std::time::Instant::now();
    let x_tokens = tok.encode(&x)?; // (s1, s2) ids [B, T]
    if profile {
        eprintln!(
            "[profile] encode {:.1} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    // Pre-draw every uniform for the whole trajectory and upload ONCE: the
    // Phase-4 sampler read them off the host per step (one sync each). The
    // row order matches the loop's batch order, step-major.
    let dev = x.device();
    let batch = x_tokens.0.size()[0];
    let draws_tab = if p.cfg.top_k == 1 {
        Tensor::zeros([p.pred_len as i64, batch], (Kind::Float, dev))
    } else {
        rng.uniform_table(p.pred_len, batch as usize, &dev)?
            .to_device(dev)
    };
    let initial = x.size()[1];
    let total = initial + p.pred_len as i64;
    let full_stamp = Tensor::cat(&[&x_stamp, &y_stamp], 1).contiguous();
    let t_loop = std::time::Instant::now();
    let (mut t_samp, mut t_rest_total) = (0.0f64, 0.0f64);
    let (mut t_s1, mut t_s2) = (0.0f64, 0.0f64);

    // Ring buffer: the initial window holds only the LAST min(T, max_ctx) tokens.
    let buffer_len = initial.min(p.max_context as i64);
    let start_idx = (initial - p.max_context as i64).max(0);
    let mut s1 = x_tokens.0.narrow(1, start_idx, buffer_len);
    let mut s2 = x_tokens.1.narrow(1, start_idx, buffer_len);
    let mut generated_pre: Vec<Tensor> = Vec::with_capacity(p.pred_len);
    let mut generated_post: Vec<Tensor> = Vec::with_capacity(p.pred_len);

    for i in 0..p.pred_len {
        let current = initial + i as i64;
        let window_len = s1.size()[1];
        // stamp window corresponds exactly to the token window
        let cs = (current - p.max_context as i64).max(0);
        let stamp = full_stamp.narrow(1, cs, window_len);

        let ts = std::time::Instant::now();
        let d1 = model.decode_s1(&s1, &s2, &stamp)?;
        let dt_s1 = ts.elapsed().as_secs_f64();
        t_rest_total += dt_s1;
        t_s1 += dt_s1;
        let s1_last = d1.s1_logits.narrow(1, window_len - 1, 1).squeeze_dim(1);
        let trace = std::env::var("KRONOS_TRACE_TOKENS").is_ok();
        if trace {
            let m = top2_margin(&s1_last)?;
            let a = s1_last
                .f_argmax(-1, false)?
                .to_device(Device::Cpu)
                .to_kind(Kind::Int);
            let a0: Vec<i32> = (&a).try_into()?;
            eprintln!("  step {i:3}: s1 top1={} margin={m:.4}", a0[0]);
        }
        let draws = draws_tab.narrow(0, i as i64, 1).squeeze_dim(0);
        let ts = std::time::Instant::now();
        let sample_pre = sample(&s1_last, &p.cfg, &draws)?.unsqueeze(1); // [B, 1]
        t_samp += ts.elapsed().as_secs_f64();

        let ts = std::time::Instant::now();
        let s2_logits = model.decode_s2(&d1.context, &sample_pre)?;
        let dt_s2 = ts.elapsed().as_secs_f64();
        t_rest_total += dt_s2;
        t_s2 += dt_s2;
        let s2_last = s2_logits.narrow(1, window_len - 1, 1).squeeze_dim(1);
        if trace {
            let m = top2_margin(&s2_last)?;
            let a = s2_last
                .f_argmax(-1, false)?
                .to_device(Device::Cpu)
                .to_kind(Kind::Int);
            let a0: Vec<i32> = (&a).try_into()?;
            eprintln!("  step {i:3}: s2 top1={} margin={m:.4}", a0[0]);
        }
        let ts = std::time::Instant::now();
        let sample_post = sample(&s2_last, &p.cfg, &draws)?.unsqueeze(1); // [B, 1]
        t_samp += ts.elapsed().as_secs_f64();

        generated_pre.push(sample_pre.shallow_clone());
        generated_post.push(sample_post.shallow_clone());

        // buffer update: append while the window still has room, else slide
        if (current as usize) < p.max_context {
            s1 = Tensor::cat(&[&s1, &sample_pre], 1).contiguous();
            s2 = Tensor::cat(&[&s2, &sample_post], 1).contiguous();
        } else {
            let w1 = s1.size()[1];
            let w2 = s2.size()[1];
            s1 = Tensor::cat(&[&s1.narrow(1, 1, w1 - 1), &sample_pre], 1).contiguous();
            s2 = Tensor::cat(&[&s2.narrow(1, 1, w2 - 1), &sample_post], 1).contiguous();
        }
    }

    if profile {
        let _ = t_rest_total;
        eprintln!(
            "[profile] AR loop  {:.1} ms  |  decode_s1 {:.1}  decode_s2 {:.1}  |  sampling {:.1} ms",
            t_loop.elapsed().as_secs_f64() * 1e3,
            t_s1 * 1e3,
            t_s2 * 1e3,
            t_samp * 1e3
        );
        let [a, b, c, d, e, f, g] = crate::ops::prof_take();
        let sp = crate::block::attn_prof_take();
        eprintln!(
            "[attn] qkv={:.2} splits={:.2} rope={:.2} scores={:.2} out={:.2}",
            sp.qkv, sp.splits, sp.rope, sp.scores, sp.out
        );
        eprintln!(
            "[prof] scores={a:.2} mask={b:.2} softmax={c:.2} vbmm={d:.2} norm={e:.2} linear={f:.2} other={g:.2}"
        );
    }
    let t_dec = std::time::Instant::now();
    // Final decode over the last max_context positions of the full stream.
    let mut all_pre = x_tokens.0.shallow_clone();
    for g in &generated_pre {
        all_pre = Tensor::cat(&[&all_pre, g], 1);
    }
    let mut all_post = x_tokens.1.shallow_clone();
    for g in &generated_post {
        all_post = Tensor::cat(&[&all_post, g], 1);
    }
    // the candle-era clamping: when the full stream is SHORTER than
    // max_context (ctx256/pred8 runs), the decode-window is the whole stream
    let width = p.max_context as i64;
    let start = total.saturating_sub(width).max(0);
    let width = width.min(total - start);
    let z_in_s1 = all_pre.narrow(1, start, width).contiguous();
    let z_in_s2 = all_post.narrow(1, start, width).contiguous();
    let z = tok.decode(&(z_in_s1, z_in_s2))?;

    if profile {
        eprintln!(
            "[profile] decode {:.1} ms",
            t_dec.elapsed().as_secs_f64() * 1e3
        );
    }
    // mean over the sample axis: [n*S, L, 6] -> [n, L, 6]
    let w = z.size()[1];
    let d_in = z.size()[2];
    let n = batch / p.sample_count as i64;
    let z = z
        .reshape([p.sample_count as i64, n, w, d_in])
        .f_mean_dim(vec![0_i64], false, None)?;
    let sampled_pre = Tensor::cat(&generated_pre, 1).contiguous();
    let sampled_post = Tensor::cat(&generated_post, 1).contiguous();
    Ok(ArOutput {
        z,
        sampled_pre,
        sampled_post,
    })
}

/// `KronosPredictor.predict` — per-feature normalise over the context window,
/// clip, generate, de-normalise (`model/kronos.py:519-559`).
///
/// The variance is a **two-pass** computation. The reference uses
/// `np.std(axis=0)` whose ddof-0 default would also be satisfied by
/// `E[x^2] - E[x]^2`, but volume is ~1e6 here, so the one-pass form loses
/// precision in f32 and shows up as a few-percent error in the amount column.
pub fn predict(
    tok: &KronosTokenizer,
    model: &Kronos,
    rows: &[Kline],
    p: &ArParams,
    rng: &mut Rng,
    dev: &Device,
) -> Result<PredictOutput> {
    debug_assert!(rows.len() > p.pred_len);
    let ctx = rows.len() - p.pred_len;
    let eps = Tensor::from_slice(&[1e-5f32]).to_device(*dev);

    let ctx_rows = &rows[..ctx];
    let y_rows = &rows[ctx..];

    let xs = crate::data::stamps(ctx_rows, dev)?;
    let ys = crate::data::stamps(y_rows, dev)?;
    let v = crate::data::values(ctx_rows, dev)?; // [ctx, 6]

    // per-feature mean over the window: [1, 6]
    let mu = v.f_mean_dim(vec![0_i64], true, None)?;
    let diff = &v - &mu;
    let var = diff.multiply(&diff).f_mean_dim(vec![0_i64], true, None)?;
    let std = var.f_sqrt()?;

    let normed = (&v - &mu) / (&std + &eps);
    let normed = normed
        .f_clamp_min(-(p.clip as f64))?
        .f_clamp_max(p.clip as f64)?;

    let input = ArInput {
        x: normed.unsqueeze(0).contiguous(),
        x_stamp: xs.unsqueeze(0),
        y_stamp: ys.unsqueeze(0),
    };
    let out = generate(tok, model, &input, p, rng)?;
    let mut res = out.z.to_kind(Kind::Float); // de-normalise in f32, like the predictor
    let w = res.size()[1];
    res = res.narrow(1, w - p.pred_len as i64, p.pred_len as i64);
    // de-normalise with the SAME (std + 1e-5) the reference uses
    let scale = &std + &eps;
    let forecast = res * scale + &mu;
    Ok(PredictOutput {
        forecast,
        sampled_pre: out.sampled_pre,
        sampled_post: out.sampled_post,
    })
}
