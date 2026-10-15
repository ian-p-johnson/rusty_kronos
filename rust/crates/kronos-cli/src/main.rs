//! CLI for the Rust Kronos port.
//!
//! Phase 1 exposes weight-inspection subcommands only. Model subcommands land
//! in Phase 2+; they are intentionally absent rather than stubbed so that a
//! green build never implies working inference.

use anyhow::{bail, Context, Result};
use kronos_core::predictor::Kronos;
use kronos_core::tokenizer::KronosTokenizer;
use kronos_core::{
    expected_predictor_keys, expected_tokenizer_keys, load_predictor_weights,
    load_tokenizer_weights, PredictorConfig, TokenizerConfig, Weights,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use tch::Tensor;
use tch::{Device, Kind as DType};

fn usage() -> &'static str {
    "kronos — Rust port of the Kronos financial K-line foundation model

USAGE:
  kronos info      --config <config.json> --weights <model.safetensors> [--tokenizer]
  kronos manifest  --config <config.json> --weights <model.safetensors> --out <f.json>
                    [--tokenizer] [--dtype fp32|fp16]

  kronos predict   --data <csv> --model-dir <d> --tok-dir <d>
                    [--ctx N] [--pred-len N] [--samples N] [--top-k N] [--top-p F]
                    [--temperature F] [--max-context N] [--seed N]
                    [--golden <csv>] [--out <csv>]

  kronos microbench --samples N            # per-op overhead probe\n  kronos sampler-audit --data <csv> --model-dir <d> --tok-dir <d> [--ctx N] [--seed N]
  kronos precision-report  --data <csv> --model-dir <d> --tok-dir <d>
                    [--ctx N] [--pred-len N] [--samples N]
"
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print!("{}", usage());
        return Ok(());
    }
    match args[0].as_str() {
        "info" => cmd_info(&args[1..]),
        "manifest" => cmd_manifest(&args[1..]),
        "parity" => cmd_parity(&args[1..]),
        "predict" => cmd_predict(&args[1..]),
        "sampler-audit" => cmd_sampler_audit(&args[1..]),
        "precision-report" => cmd_precision(&args[1..]),
        "microbench" => cmd_microbench(&args[1..]),
        "-h" | "--help" | "help" => {
            print!("{}", usage());
            Ok(())
        }
        other => bail!("unknown subcommand `{other}`\n\n{}", usage()),
    }
}

struct Args {
    config: PathBuf,
    weights: PathBuf,
    out: Option<PathBuf>,
    tokenizer: bool,
    dtype: DType,
}

fn parse_args(args: &[String]) -> Result<Args> {
    let mut a = Args {
        config: PathBuf::new(),
        weights: PathBuf::new(),
        out: None,
        tokenizer: false,
        dtype: DType::Float,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                a.config = PathBuf::from(args.get(i).context("--config needs a value")?);
            }
            "--weights" => {
                i += 1;
                a.weights = PathBuf::from(args.get(i).context("--weights needs a value")?);
            }
            "--out" => {
                i += 1;
                a.out = Some(PathBuf::from(args.get(i).context("--out needs a value")?));
            }
            "--tokenizer" => a.tokenizer = true,
            "--dtype" => {
                i += 1;
                a.dtype = match args.get(i).map(|s| s.as_str()) {
                    Some("fp32") | Some("f32") => DType::Float,
                    Some("fp16") | Some("f16") => DType::Half,
                    Some("bf16") => DType::BFloat16,
                    _ => bail!("--dtype must be fp32, fp16 or bf16"),
                };
            }
            other => bail!("unknown flag `{other}`\n\n{}", usage()),
        }
        i += 1;
    }
    if a.config.as_os_str().is_empty() {
        bail!("--config is required\n\n{}", usage());
    }
    if a.weights.as_os_str().is_empty() {
        bail!("--weights is required\n\n{}", usage());
    }
    Ok(a)
}

/// Per-tensor statistics. The purpose is to prove the *right* tensor was
/// loaded, not to verify reduction order, so these are compared with a
/// relative tolerance rather than exactly.
///
/// Stats are computed on CPU in f64: candle's CUDA backend does not support
/// every f64 reduction, and this is a startup diagnostic where a copy of
/// <=100 MB is irrelevant.
fn tensor_stats(w: &Weights) -> serde_json::Map<String, serde_json::Value> {
    let mut out = serde_json::Map::new();
    for (name, t) in w.iter_sorted() {
        let shape: Vec<usize> = t.size().iter().map(|&d| d as usize).collect();
        let n = t.numel();
        let host = t
            .reshape([-1])
            .to_device(Device::Cpu)
            .to_kind(DType::Double);
        let v: Option<Vec<f64>> = (&host).try_into().ok();
        let (sum, absmax) = match v {
            Some(v) => (
                Some(v.iter().sum::<f64>()),
                Some(v.iter().map(|x| x.abs()).fold(0.0f64, f64::max)),
            ),
            None => (None, None),
        };
        out.insert(
            (*name).clone(),
            json!({ "shape": shape, "n": n, "sum": sum, "absmax": absmax }),
        );
    }
    out
}

fn cmd_info(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    let cfg_raw = std::fs::read_to_string(&a.config)?;
    let dev = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let dtype = a.dtype;

    let (n_expected, cfg_desc, n_params, keys) = if a.tokenizer {
        let c = TokenizerConfig::from_json(&cfg_raw)?;
        c.validate()?;
        let keys = expected_tokenizer_keys(&c);
        let w = load_tokenizer_weights(&a.weights, &c, dtype, &dev)?;
        (keys.len(), c.to_string(), w.numel(), keys)
    } else {
        let c = PredictorConfig::from_json(&cfg_raw)?;
        c.validate()?;
        let keys = expected_predictor_keys(&c);
        let w = load_predictor_weights(&a.weights, &c, dtype, &dev)?;
        (keys.len(), c.to_string(), w.numel(), keys)
    };

    println!("config   : {}", a.config.display());
    println!("weights  : {}", a.weights.display());
    println!("dtype    : {dtype:?}   device: {dev:?}");
    println!("{cfg_desc}");
    println!("tensors  : {n_expected} (all present, all consumed, shapes verified)");
    println!("params   : {n_params} ({:.2} M)", n_params as f64 / 1e6);
    let _ = keys;
    Ok(())
}

fn cmd_manifest(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    let out = a.out.clone().context("--out is required for `manifest`")?;
    let cfg_raw = std::fs::read_to_string(&a.config)?;
    let dev = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let w: Weights = if a.tokenizer {
        let c = TokenizerConfig::from_json(&cfg_raw)?;
        c.validate()?;
        load_tokenizer_weights(&a.weights, &c, a.dtype, &dev)?
    } else {
        let c = PredictorConfig::from_json(&cfg_raw)?;
        c.validate()?;
        load_predictor_weights(&a.weights, &c, a.dtype, &dev)?
    };

    let payload = json!({
        "config_path": a.config,
        "weights_path": a.weights,
        "kind": if a.tokenizer { "tokenizer" } else { "predictor" },
        "dtype": format!("{:?}", a.dtype),
        "numel": w.numel(),
        "tensors": tensor_stats(&w),
    });
    std::fs::write(&out, serde_json::to_string_pretty(&payload)?)
        .with_context(|| format!("writing {}", out.display()))?;
    println!("wrote {} ({} tensors)", out.display(), w.len());
    Ok(())
}

/// Shared loader for the three Phase 4 subcommands.
struct Phase4 {
    tok: kronos_core::tokenizer::KronosTokenizer,
    mdl: kronos_core::predictor::Kronos,
    rows: Vec<kronos_core::Kline>,
    dev: Device,
}

fn load_phase4(
    data: &Path,
    model_dir: &Path,
    tok_dir: &Path,
    ctx: usize,
    pred_len: usize,
) -> anyhow::Result<Phase4> {
    let dev = tch::Device::cuda_if_available();
    let dtype = DType::Float;
    let tok_cfg =
        TokenizerConfig::from_json(&std::fs::read_to_string(tok_dir.join("config.json"))?)?;
    tok_cfg.validate()?;
    let tok_w = load_tokenizer_weights(&tok_dir.join("model.safetensors"), &tok_cfg, dtype, &dev)?;
    let mdl_cfg =
        PredictorConfig::from_json(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    mdl_cfg.validate()?;
    let mdl_w =
        load_predictor_weights(&model_dir.join("model.safetensors"), &mdl_cfg, dtype, &dev)?;
    let mut rows = kronos_core::load_csv(data)?;
    if rows.len() < ctx + pred_len {
        anyhow::bail!(
            "CSV has {} rows; need at least ctx({}) + pred_len({})",
            rows.len(),
            ctx,
            pred_len
        );
    }
    rows.truncate(ctx + pred_len);
    let tok = kronos_core::tokenizer::KronosTokenizer::load(&tok_w, &tok_cfg, 512, &dev, dtype)?;
    let mdl = Kronos::load(&mdl_w, &mdl_cfg, 512, &dev, dtype)?;
    Ok(Phase4 {
        tok,
        mdl,
        rows,
        dev,
    })
}

struct CommonFlags {
    data: PathBuf,
    model_dir: PathBuf,
    tok_dir: PathBuf,
    ctx: usize,
    pred_len: usize,
    samples: usize,
    top_k: u32,
    top_p: f32,
    temperature: f32,
    max_context: usize,
    seed: u64,
    out: Option<PathBuf>,
    golden: Option<PathBuf>,
}

fn parse_common(args: &[String]) -> anyhow::Result<CommonFlags> {
    let mut f = CommonFlags {
        data: PathBuf::new(),
        model_dir: PathBuf::new(),
        tok_dir: PathBuf::new(),
        ctx: 512,
        pred_len: 8,
        samples: 1,
        top_k: 1,
        top_p: 1.0,
        temperature: 1.0,
        max_context: 512,
        seed: 123,
        out: None,
        golden: None,
    };
    let mut i = 0;
    let get = |k: &str, args: &[String], i: &mut usize| -> anyhow::Result<String> {
        *i += 1;
        args.get(*i)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("--{k} needs a value"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--data" => f.data = PathBuf::from(get("data", args, &mut i)?),
            "--model-dir" => f.model_dir = PathBuf::from(get("model-dir", args, &mut i)?),
            "--tok-dir" => f.tok_dir = PathBuf::from(get("tok-dir", args, &mut i)?),
            "--out" => f.out = Some(PathBuf::from(get("out", args, &mut i)?)),
            "--golden" => f.golden = Some(PathBuf::from(get("golden", args, &mut i)?)),
            "--ctx" => f.ctx = get("ctx", args, &mut i)?.parse()?,
            "--pred-len" => f.pred_len = get("pred-len", args, &mut i)?.parse()?,
            "--samples" => f.samples = get("samples", args, &mut i)?.parse()?,
            "--top-k" => f.top_k = get("top-k", args, &mut i)?.parse()?,
            "--top-p" => f.top_p = get("top-p", args, &mut i)?.parse()?,
            "--temperature" => f.temperature = get("temperature", args, &mut i)?.parse()?,
            "--max-context" => f.max_context = get("max-context", args, &mut i)?.parse()?,
            "--seed" => f.seed = get("seed", args, &mut i)?.parse()?,
            other => anyhow::bail!("unknown flag `{other}`"),
        }
        i += 1;
    }
    if f.data.as_os_str().is_empty()
        || f.model_dir.as_os_str().is_empty()
        || f.tok_dir.as_os_str().is_empty()
    {
        anyhow::bail!("--data, --model-dir and --tok-dir are all required");
    }
    Ok(f)
}

/// `kronos predict`: the golden path. Reproduces `KronosPredictor.predict`
/// end-to-end from a raw CSV, and optionally asserts against a committed
/// golden CSV (same schema) with rtol=1e-5, exiting 1 on mismatch.
fn cmd_predict(args: &[String]) -> Result<()> {
    let f = parse_common(args)?;
    let ph = load_phase4(&f.data, &f.model_dir, &f.tok_dir, f.ctx, f.pred_len)?;
    let p = kronos_core::ar::ArParams::new(
        f.max_context,
        f.pred_len,
        f.samples,
        kronos_core::sampling::SampleCfg {
            temperature: f.temperature,
            top_k: f.top_k,
            top_p: f.top_p,
        },
    );
    let mut rng = kronos_core::sampling::Rng::new(f.seed);
    let out = kronos_core::ar::predict(&ph.tok, &ph.mdl, &ph.rows, &p, &mut rng, &ph.dev)?;
    let preds = out.forecast;
    let rows = preds.size()[0] as usize;
    let n = preds.size()[1] as usize;
    let d = preds.size()[2] as usize;
    let flat: Vec<f32> = (&preds
        .reshape([-1])
        .to_device(Device::Cpu)
        .to_kind(DType::Float))
        .try_into()?;

    if let Some(gold) = &f.golden {
        let text = std::fs::read_to_string(gold)?;
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() - 1 != rows * n || lines.len() < 2 {
            anyhow::bail!(
                "golden has {} rows, got {}x{} predictions",
                lines.len() - 1,
                rows,
                n
            );
        }
        let header: Vec<&str> = lines[0].split(',').map(|c| c.trim()).collect();
        let names = ["open", "high", "low", "close", "volume", "amount"];
        let mut cols = Vec::with_capacity(6);
        for name in names {
            match header.iter().position(|h| *h == name) {
                Some(j) => cols.push(j),
                None => anyhow::bail!("golden header missing column `{name}`"),
            }
        }
        let rtol = 1e-5f64;
        let mut worst = (0.0f64, 0usize);
        for r in 0..(rows * n) {
            let line = lines
                .get(r + 1)
                .ok_or_else(|| anyhow::anyhow!("missing golden row {r}"))?;
            let parts: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
            for (k, j) in cols.iter().enumerate() {
                let want: f64 = parts[*j].parse()?;
                let got: f64 = flat[r * d + k] as f64;
                let rel = (got - want).abs() / want.abs().max(1e-12);
                if rel > worst.0 {
                    worst = (rel, r);
                }
                if (got - want).abs() > rtol * want.abs().max(1e-12) {
                    anyhow::bail!(
                        "row {r} col {} diverged: got {got}, want {want} (rel {rel:.3e})",
                        ["open", "high", "low", "close", "volume", "amount"][k]
                    );
                }
            }
        }
        println!(
            "GOLDEN OK rtol={} worst rel err {:.2e} (row {})",
            rtol, worst.0, worst.1
        );
    } else {
        println!("rows={rows} cols={n} values={d}");
        for r in 0..rows.min(2) {
            for k in 0..6 {
                print!("{:>14.6} ", flat[r * d + k]);
            }
            println!();
        }
    }
    if let Some(out) = &f.out {
        let mut s = String::from("open,high,low,close,volume,amount\n");
        for r in 0..rows * n {
            for k in 0..d {
                s.push_str(&format!("{:.9}", flat[r * d + k]));
                if k + 1 < d {
                    s.push(',');
                }
            }
            s.push('\n');
        }
        std::fs::write(out, s)?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

/// The three agreed sampler assertions (parity/README.md "stochastic" tier).
fn cmd_sampler_audit(args: &[String]) -> Result<()> {
    let f = parse_common(args)?;
    let ph = load_phase4(&f.data, &f.model_dir, &f.tok_dir, f.ctx, 1)?;
    let ctx = &ph.rows[..f.ctx];
    let xs = kronos_core::data::stamps(ctx, &ph.dev)?.unsqueeze(0);
    let v = kronos_core::data::values(ctx, &ph.dev)?;
    let mu = v.f_mean_dim(vec![0], true, None)?;
    let diff = &v - &mu;
    let std = diff
        .multiply(&diff)
        .f_mean_dim(vec![0], true, None)?
        .f_sqrt()?;
    let eps = Tensor::from_slice(&[1e-5f32]).to_device(ph.dev);
    let normed = (&v - &mu) / (&std + &eps);
    let normed = normed.f_clamp_min(-5.0)?.f_clamp_max(5.0)?.unsqueeze(0);

    let toks = ph.tok.encode(&normed)?;
    let d1 = ph.mdl.decode_s1(&toks.0, &toks.1, &xs)?;
    let last = d1.s1_logits.narrow(1, f.ctx as i64 - 1, 1).squeeze_dim(1); // [1, V]
    let argmax = last.f_argmax(-1, false)?;
    let flat_arg: Vec<i32> = (&argmax.to_device(Device::Cpu).to_kind(DType::Int)).try_into()?;

    // assertion 3 needs the identity of the deterministic token: compare it
    // against the committed fixture when present.
    println!("argmax token = {}", flat_arg[0]);

    let draws = 256;
    let cfg_hot = kronos_core::sampling::SampleCfg {
        temperature: 0.6,
        top_k: 0,
        top_p: 0.9,
    };
    let tab = kronos_core::sampling::Rng::new(f.seed).uniform_table(draws, 1, &last.device())?;
    let mut ids = Vec::with_capacity(draws);
    for i in 0..draws {
        let d = tab.narrow(0, i as i64, 1).squeeze_dim(0);
        let id = kronos_core::sampling::sample(&last, &cfg_hot, &d)?;
        let iv: Vec<i32> = (&id.to_device(Device::Cpu).to_kind(DType::Int)).try_into()?;
        ids.push(iv[0]);
    }
    // (1) no draw outside the nucleus
    let mass = kronos_core::sampling::strictly_greater_mass(
        &last,
        &Tensor::from_slice(&[flat_arg[0]]).to_device(ph.dev),
    )?;
    let mv0: Vec<f32> = (&mass
        .to_device(Device::Cpu)
        .to_kind(DType::Float)
        .reshape([-1]))
        .try_into()?;
    let p_analytic = 1.0f64 - mv0[0] as f64;
    // the argmax token's probability from the softmax itself
    // the audit compares draws taken at T=0.6, so the reference p must come
    // from the SAME temperature-scaled distribution (the candle-era code
    // softmaxed the scaled logits — a rewording of `model/kronos.py:344`).
    let scaled = (&last * (1.0f64 / 0.6)).contiguous();
    let probs = kronos_core::sampling::softmax(&scaled)?;
    let pv: Vec<f32> = (&probs
        .narrow(1, flat_arg[0] as i64, 1)
        .to_device(Device::Cpu)
        .to_kind(DType::Float)
        .reshape([-1]))
        .try_into()?;
    let p_max = pv[0] as f64;
    let n_in = ids.iter().filter(|x| **x == flat_arg[0]).count();
    let freq = n_in as f64 / draws as f64;
    println!("argmax token prob (softmax) = {p_analytic:.4} / argmax-readback {p_max:.4}");
    println!("empirical freq over {draws} draws = {freq:.4}");

    let mut failures = 0usize;
    // assertion 1: frequency close to the analytic probability
    if (freq - p_max).abs() > 0.10f64.max(0.10 * p_max) {
        eprintln!("FAIL (1) top-1 frequency {freq:.4} vs analytic {p_max:.4}");
        failures += 1;
    }
    // assertion 2: every draw inside the nucleus (mass of strictly-greater <= top_p)
    let member = kronos_core::sampling::strictly_greater_mass(
        &last,
        &Tensor::from_slice(&ids).to_device(ph.dev),
    )?;
    let mv: Vec<f32> = (&member.to_device(Device::Cpu).to_kind(DType::Float)).try_into()?;
    if let Some((i, m)) = mv.iter().enumerate().find(|(_, m)| **m > 0.9 + 1e-6) {
        eprintln!("FAIL (2) draw {i} outside nucleus (strictly-greater mass {m:.4})");
        failures += 1;
    }
    // assertion 3: draws are not a constant
    if ids.windows(2).all(|w| w[0] == w[1]) {
        eprintln!("FAIL (3) draws are constant");
        failures += 1;
    }
    if failures == 0 {
        println!("SAMPLER AUDIT OK (draws={draws}, top_p=0.9, T=0.6)");
    } else {
        println!("SAMPLER AUDIT FAILED: {failures} assertion(s)");
        std::process::exit(1);
    }
    Ok(())
}

/// fp32 vs fp16 divergence across the full trajectory: **reported, not gated**.
///
/// Three lenses, because "the output drifted" conflates arithmetic with path
/// divergence once sampling feeds back:
///   1. arithmetic — one decode_s1 pass on the identical context, fp16 vs fp32,
///      max abs difference over the 1024 coarse-subtoken logits;
///   2. path — how many of the sampled AR tokens differ over `pred_len` steps;
///   3. terminal — the de-normalised OHLCVA forecast, max relative error per column.
fn cmd_precision(args: &[String]) -> Result<()> {
    let f = parse_common(args)?;
    let dev = tch::Device::cuda_if_available();

    struct Run {
        logits0: Vec<f32>,
        pre: Vec<i32>,
        post: Vec<i32>,
        cols: Vec<f32>,
    }
    let run = |dtype: DType, tag: &str, ms: &mut f64| -> anyhow::Result<Run> {
        let tok_cfg = TokenizerConfig::from_json(
            &std::fs::read_to_string(f.tok_dir.join("config.json")).map_err(anyhow::Error::from)?,
        )?;
        tok_cfg.validate()?;
        let mdl_cfg = PredictorConfig::from_json(
            &std::fs::read_to_string(f.model_dir.join("config.json"))
                .map_err(anyhow::Error::from)?,
        )?;
        mdl_cfg.validate()?;
        let tok_w =
            load_tokenizer_weights(&f.tok_dir.join("model.safetensors"), &tok_cfg, dtype, &dev)?;
        let mdl_w = load_predictor_weights(
            &f.model_dir.join("model.safetensors"),
            &mdl_cfg,
            dtype,
            &dev,
        )?;
        let tok = KronosTokenizer::load(&tok_w, &tok_cfg, f.max_context, &dev, dtype)?;
        let mdl = Kronos::load(&mdl_w, &mdl_cfg, f.max_context, &dev, dtype)?;
        let mut rows = kronos_core::load_csv(&f.data)?;
        rows.truncate(f.ctx + f.pred_len);
        let t0 = std::time::Instant::now();

        // lens 1: arithmetic — the same context, one coarse-token head pass
        let v = kronos_core::data::values(&rows[..f.ctx], &dev)?;
        let mu = v.f_mean_dim(vec![0], true, None)?;
        let mu2 = &mu * &mu;
        let std = v.multiply(&v).f_sum_dim_intlist(vec![0], true, None)? / (f.ctx as f64);
        let std = (std - mu2).f_sqrt()?;
        let eps = Tensor::from_slice(&[1e-5f32]);
        let normed = (&v - &mu) / (&std + &eps);
        let normed = normed.f_clamp_min(-5.0)?.f_clamp_max(5.0)?;
        let xs = kronos_core::data::stamps(&rows[..f.ctx], &dev)?.unsqueeze(0);
        let toks = tok.encode(&normed.unsqueeze(0).to_kind(dtype))?;
        let d1 = mdl.decode_s1(&toks.0, &toks.1, &xs)?;
        let logits0: Vec<f32> = (&d1
            .s1_logits
            .narrow(1, f.ctx as i64 - 1, 1)
            .reshape([-1])
            .to_device(Device::Cpu)
            .to_kind(DType::Float))
            .try_into()?;

        // lens 2+3: the full stochastic trajectory
        let p = kronos_core::ar::ArParams::new(
            f.max_context,
            f.pred_len,
            f.samples,
            kronos_core::sampling::SampleCfg {
                temperature: f.temperature,
                top_k: f.top_k,
                top_p: f.top_p,
            },
        );
        let mut rng = kronos_core::sampling::Rng::new(f.seed);
        let out = kronos_core::ar::predict(&tok, &mdl, &rows, &p, &mut rng, &dev)?;
        let pre: Vec<i32> = (&out
            .sampled_pre
            .reshape([-1])
            .to_device(Device::Cpu)
            .to_kind(DType::Int))
            .try_into()?;
        let post: Vec<i32> = (&out
            .sampled_post
            .reshape([-1])
            .to_device(Device::Cpu)
            .to_kind(DType::Int))
            .try_into()?;
        let cols: Vec<f32> = (&out
            .forecast
            .reshape([-1])
            .to_device(Device::Cpu)
            .to_kind(DType::Float))
            .try_into()?;
        *ms = t0.elapsed().as_millis() as f64;
        println!("  {tag}: full trajectory {ms:.1} ms");
        Ok(Run {
            logits0,
            pre,
            post,
            cols,
        })
    };

    let mut ms32 = 0.0;
    let r32 = run(DType::Float, "fp32", &mut ms32)?;
    let mut ms16 = 0.0;
    let r16 = run(DType::Half, "fp16", &mut ms16)?;

    // lens 1
    let d0 = r32
        .logits0
        .iter()
        .zip(&r16.logits0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // lens 2
    let flips_pre = r32.pre.iter().zip(&r16.pre).filter(|(a, b)| a != b).count();
    let flips_post = r32
        .post
        .iter()
        .zip(&r16.post)
        .filter(|(a, b)| a != b)
        .count();
    // lens 3
    let names = ["open", "high", "low", "close", "volume", "amount"];
    let n = r32.cols.len() / 6;
    let mut worst = [(0.0f32, 0usize); 6];
    for (c, _) in names.iter().enumerate() {
        for r in 0..n {
            let a = r32.cols[r * 6 + c] as f64;
            let b = r16.cols[r * 6 + c] as f64;
            let rel = (a - b).abs() / a.abs().max(1e-9);
            if rel as f32 > worst[c].0 {
                worst[c] = (rel as f32, r);
            }
        }
    }
    // lens 2b: GREEDY (top_k=1) path — a token flips only when the fp16
    // arithmetic actually crosses the argmax boundary, so this isolates the
    // arithmetic from the sampler's RNG sensitivity on flat distributions.
    let greedy = |dtype: DType| -> anyhow::Result<(Vec<i32>, Vec<i32>, Vec<f32>)> {
        let tok_cfg = TokenizerConfig::from_json(
            &std::fs::read_to_string(f.tok_dir.join("config.json")).map_err(anyhow::Error::from)?,
        )?;
        let mdl_cfg = PredictorConfig::from_json(
            &std::fs::read_to_string(f.model_dir.join("config.json"))
                .map_err(anyhow::Error::from)?,
        )?;
        let tok_w =
            load_tokenizer_weights(&f.tok_dir.join("model.safetensors"), &tok_cfg, dtype, &dev)?;
        let mdl_w = load_predictor_weights(
            &f.model_dir.join("model.safetensors"),
            &mdl_cfg,
            dtype,
            &dev,
        )?;
        let tok = KronosTokenizer::load(&tok_w, &tok_cfg, f.max_context, &dev, dtype)?;
        let mdl = Kronos::load(&mdl_w, &mdl_cfg, f.max_context, &dev, dtype)?;
        let rows = kronos_core::load_csv(&f.data)?;
        let used = rows.len().min(f.ctx + f.pred_len);
        let p = kronos_core::ar::ArParams::new(
            f.max_context,
            f.pred_len,
            1,
            kronos_core::sampling::SampleCfg {
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
            },
        );
        let mut rng = kronos_core::sampling::Rng::new(f.seed);
        let o = kronos_core::ar::predict(&tok, &mdl, &rows[..used], &p, &mut rng, &dev)?;
        Ok((
            (&o.sampled_pre
                .reshape([-1])
                .to_device(Device::Cpu)
                .to_kind(DType::Int))
                .try_into()?,
            (&o.sampled_post
                .reshape([-1])
                .to_device(Device::Cpu)
                .to_kind(DType::Int))
                .try_into()?,
            (&o.forecast
                .reshape([-1])
                .to_device(Device::Cpu)
                .to_kind(DType::Float))
                .try_into()?,
        ))
    };
    let (g32_pre, g32_post, g32_fc) = greedy(DType::Float)?;
    let (g16_pre, g16_post, g16_fc) = greedy(DType::Half)?;
    let gpre = g32_pre.iter().zip(&g16_pre).filter(|(a, b)| a != b).count();
    let gpost = g32_post
        .iter()
        .zip(&g16_post)
        .filter(|(a, b)| a != b)
        .count();

    println!(
        "fp32 {:.1} ms | fp16 {:.1} ms | {} trajectory steps, samples={}, seed={}",
        ms32, ms16, n, f.samples, f.seed
    );
    println!("  lens 1  decode_s1 logits step 0      : max abs diff {d0:.4e}");
    println!(
        "  lens 2  sampled tokens (T={:.1}, top_p={:.2}) : coarse flips {}/{}  fine flips {}/{}",
        f.temperature,
        f.top_p,
        flips_pre,
        r32.pre.len(),
        flips_post,
        r32.post.len()
    );
    println!(
        "  lens 2b greedy tokens (top_k=1)              : coarse flips {}/{}  fine flips {}/{}",
        gpre,
        g32_pre.len(),
        gpost,
        g32_post.len()
    );
    for (c, name) in names.iter().enumerate() {
        println!(
            "  lens 3  {:<7} max rel err {:.3e} (row {})",
            name, worst[c].0, worst[c].1
        );
        if g32_fc.len() == g16_fc.len() {
            let mut rworst = 0.0f64;
            let n_g = g32_fc.len() / 6;
            for r in 0..n_g {
                let a = g32_fc[r * 6 + c] as f64;
                let b = g16_fc[r * 6 + c] as f64;
                rworst = rworst.max((a - b).abs() / a.abs().max(1e-9));
            }
            println!(
                "  lens 3b {:<7} GREEDY fp32-vs-fp16 max rel err {:.3e}",
                name, rworst
            );
        }
    }
    Ok(())
}

/// Per-op overhead probe: run a chain of tiny ops at a realistic shape and
/// report wall time per op. Diagnosed the Phase-5 finding that the port was
/// overhead-bound (~0.6 ms/op) rather than FLOP-bound.
fn cmd_microbench(args: &[String]) -> Result<()> {
    let mut batch = 10usize;
    let mut reps = 2000usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--samples" => {
                i += 1;
                batch = args[i].parse()?;
            }
            "--reps" => {
                i += 1;
                reps = args[i].parse()?;
            }
            other => anyhow::bail!("unknown flag `{other}`"),
        }
        i += 1;
    }
    let dev = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let t = Tensor::randn([batch as i64, 512, 512], (DType::Float, dev));
    let _sync = || {
        let _ = t.to_device(Device::Cpu).to_kind(DType::Float);
    };
    // warmup
    for _ in 0..50 {
        let _x = (&t * &t + 1.0).shallow_clone();
    }
    _sync();
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let _x = (&t * &t + 2.0).shallow_clone();
    }
    _sync();
    let per = t0.elapsed().as_secs_f64() / reps as f64;
    println!(
        "sqr+affine at ({batch},512,512) f32: {per:.3} ms/op over {reps} reps  (gpu {:?})",
        p_name(&dev)
    );
    // and a big-GEMM reference for contrast
    let a = Tensor::randn([(batch * 512) as i64, 512], (DType::Float, dev));
    let w = Tensor::randn([1024, 512], (DType::Float, dev));
    let w = w.transpose(0, 1);
    for _ in 0..10 {
        let _y = a.matmul(&w);
    }
    _sync();
    let t1 = std::time::Instant::now();
    for _ in 0..reps {
        let _y = a.matmul(&w);
    }
    _sync();
    let per = t1.elapsed().as_secs_f64() / reps as f64;
    println!("matmul {batch}x512x1024 f32: {per:.3} ms/call");

    // ---- SDPA head-to-head at the real attention shape ----
    let lq = 512usize;
    let q = Tensor::randn([batch as i64, 8, lq as i64, 64], (DType::Float, dev));
    let k = Tensor::randn([batch as i64, 8, lq as i64, 64], (DType::Float, dev));
    let v = Tensor::randn([batch as i64, 8, lq as i64, 64], (DType::Float, dev));
    let fused = || -> anyhow::Result<Tensor> {
        Tensor::f_scaled_dot_product_attention(&q, &k, &v, None::<&Tensor>, 0.0, true, None, false)
            .map_err(anyhow::Error::from)
    };
    match fused() {
        Ok(_) => {
            let t = std::time::Instant::now();
            for _ in 0..50 {
                let _ = fused()?;
            }
            _sync();
            println!(
                "sdpa(fused kernel): {:.3} ms/call",
                t.elapsed().as_secs_f64() / 50.0 * 1e3
            );
        }
        Err(e) => println!("sdpa(fused kernel) FAILED: {e}"),
    }
    let manual = |q: &Tensor| -> anyhow::Result<f64> {
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let kt = k.transpose(2, 3).contiguous();
            let mut scores = q.matmul(&kt) * 0.125;
            let _mask = Tensor::f_full([lq as i64, lq as i64], 0f64, (DType::Float, dev))?;
            let ones = Tensor::ones([lq as i64, lq as i64], (DType::Bool, dev));
            let up = Tensor::f_triu(&ones, 1)?;
            let mut up_f = up.to_kind(DType::Float);
            up_f *= -1e30f64;
            let up_f = up_f.expand([batch as i64, 8, lq as i64, lq as i64], false);
            scores += up_f;
            let last = scores.size().len() as i64 - 1;
            let max = scores.f_amax(vec![last], true)?;
            let expd = (&scores - &max).exp();
            let sm = expd.f_sum_dim_intlist(vec![last], true, None)?;
            let pr = expd / sm;
            let _o = pr.matmul(&v.contiguous());
        }
        _sync();
        Ok(t.elapsed().as_secs_f64() / 10.0 * 1e3)
    };
    println!("sdpa(manual chain): {:.3} ms/call", manual(&q)?);
    Ok(())
}

#[allow(dead_code)]
fn p_name(dev: &Device) -> String {
    format!("{dev:?}")
}

/// Small helper so `info` can work without a GPU (CI on CPU-only runners).
#[allow(dead_code)]
trait DeviceExt {
    fn new_cuda_or_cpu() -> Device;
}
impl DeviceExt for Device {
    fn new_cuda_or_cpu() -> Device {
        tch::Device::cuda_if_available()
    }
}

fn cmd_parity(args: &[String]) -> Result<()> {
    use kronos_core::{load_predictor_weights, load_tokenizer_weights, parity};

    let mut model_dir = PathBuf::new();
    let mut tok_dir = PathBuf::new();
    let mut probe_dir = PathBuf::new();
    let mut fixture_dir = PathBuf::new();
    let mut rtol = 1e-5f64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = PathBuf::from(args.get(i).context("--model-dir needs a value")?);
            }
            "--tok-dir" => {
                i += 1;
                tok_dir = PathBuf::from(args.get(i).context("--tok-dir needs a value")?);
            }
            "--probe" => {
                i += 1;
                probe_dir = PathBuf::from(args.get(i).context("--probe needs a value")?);
            }
            "--fixture-dir" => {
                i += 1;
                fixture_dir = PathBuf::from(args.get(i).context("--fixture-dir needs a value")?);
            }
            "--rtol" => {
                i += 1;
                rtol = args
                    .get(i)
                    .context("--rtol needs a value")?
                    .parse()
                    .context("--rtol must be a float")?;
            }
            other => bail!("unknown flag `{other}`"),
        }
        i += 1;
    }

    let dev = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let dtype = DType::Float;

    let tok_cfg_raw = std::fs::read_to_string(tok_dir.join("config.json"))?;
    let tok_cfg = TokenizerConfig::from_json(&tok_cfg_raw)?;
    tok_cfg.validate()?;
    let tok_w = load_tokenizer_weights(&tok_dir.join("model.safetensors"), &tok_cfg, dtype, &dev)?;

    let mdl_cfg_raw = std::fs::read_to_string(model_dir.join("config.json"))?;
    let mdl_cfg = PredictorConfig::from_json(&mdl_cfg_raw)?;
    mdl_cfg.validate()?;
    let mdl_w =
        load_predictor_weights(&model_dir.join("model.safetensors"), &mdl_cfg, dtype, &dev)?;

    let x = kronos_core::npy::read_npy(&probe_dir.join("probe_x.npy"))?.to_device(dev);
    let stamp = kronos_core::npy::read_npy(&probe_dir.join("probe_stamp.npy"))?.to_device(dev);
    let ctx_len = x.size()[1] as usize;
    let max_len = ctx_len.max(mdl_cfg.n_layers); // over-cover for the dep-layer rope

    let tok = KronosTokenizer::load(&tok_w, &tok_cfg, max_len, &dev, dtype)?;
    let mdl = Kronos::load(&mdl_w, &mdl_cfg, max_len, &dev, dtype)?;

    let out = parity::run_probe(&tok, &mdl, &x, &stamp)?;
    let reports = parity::compare_stages(&out, &fixture_dir, rtol);
    let fails = parity::print_report(&reports);

    let json = serde_json::to_string_pretty(&reports)?;
    let report_path = fixture_dir.join("parity_report.json");
    std::fs::write(&report_path, &json)?;
    println!(
        "\n{}/{} stages pass (rtol={:.0e})  -> {}",
        reports.len() - fails,
        reports.len(),
        rtol,
        report_path.display()
    );
    if fails > 0 {
        std::process::exit(1);
    }
    Ok(())
}

#[allow(dead_code)]
fn is_cuda(dev: &Device) -> bool {
    matches!(dev, Device::Cuda(_))
}

#[allow(dead_code)]
fn path_str(p: &Path) -> String {
    p.display().to_string()
}
