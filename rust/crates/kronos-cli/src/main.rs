//! CLI for the Rust Kronos port.
//!
//! Phase 1 exposes weight-inspection subcommands only. Model subcommands land
//! in Phase 2+; they are intentionally absent rather than stubbed so that a
//! green build never implies working inference.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use kronos_core::predictor::Kronos;
use kronos_core::tokenizer::KronosTokenizer;
use kronos_core::{
    expected_predictor_keys, expected_tokenizer_keys, load_predictor_weights,
    load_tokenizer_weights, PredictorConfig, TokenizerConfig, Weights,
};
use serde_json::json;
use std::path::{Path, PathBuf};

fn usage() -> &'static str {
    "kronos — Rust port of the Kronos financial K-line foundation model

USAGE:
  kronos info      --config <config.json> --weights <model.safetensors> [--tokenizer]
  kronos manifest  --config <config.json> --weights <model.safetensors> --out <f.json>
                    [--tokenizer] [--dtype fp32|fp16]

  kronos predict   ...   (Phase 2+)
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
        dtype: DType::F32,
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
                    Some("fp32") | Some("f32") => DType::F32,
                    Some("fp16") | Some("f16") => DType::F16,
                    Some("bf16") => DType::BF16,
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
        let shape = t.shape().dims().to_vec();
        let n = t.elem_count();

        // flatten -> host -> f64, then reduce. Any failure yields nulls rather
        // than aborting: this is a diagnostic, and a null is a visible signal.
        let stats = t
            .flatten_all()
            .and_then(|f| f.to_device(&Device::Cpu))
            .and_then(|f| f.to_dtype(DType::F64))
            .ok()
            .map(|f64| {
                let sum = f64.sum_all().ok().and_then(|s| s.to_scalar::<f64>().ok());
                let absmax = f64
                    .abs()
                    .and_then(|a| a.max(0))
                    .ok()
                    .and_then(|m| m.to_scalar::<f64>().ok());
                (sum, absmax)
            })
            .unwrap_or((None, None));

        out.insert(
            name.clone(),
            json!({ "shape": shape, "n": n, "sum": stats.0, "absmax": stats.1 }),
        );
    }
    out
}

fn cmd_info(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    let cfg_raw = std::fs::read_to_string(&a.config)?;
    let dev = Device::new_cuda_or_cpu();
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
    let dev = Device::new_cuda_or_cpu();

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

/// Small helper so `info` can work without a GPU (CI on CPU-only runners).
trait DeviceExt {
    fn new_cuda_or_cpu() -> Device;
}
impl DeviceExt for Device {
    fn new_cuda_or_cpu() -> Device {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
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

    let dev = Device::new_cuda_or_cpu();
    let dtype = DType::F32;

    let tok_cfg_raw = std::fs::read_to_string(tok_dir.join("config.json"))?;
    let tok_cfg = TokenizerConfig::from_json(&tok_cfg_raw)?;
    tok_cfg.validate()?;
    let tok_w = load_tokenizer_weights(&tok_dir.join("model.safetensors"), &tok_cfg, dtype, &dev)?;

    let mdl_cfg_raw = std::fs::read_to_string(model_dir.join("config.json"))?;
    let mdl_cfg = PredictorConfig::from_json(&mdl_cfg_raw)?;
    mdl_cfg.validate()?;
    let mdl_w =
        load_predictor_weights(&model_dir.join("model.safetensors"), &mdl_cfg, dtype, &dev)?;

    let x = kronos_core::npy::read_npy(&probe_dir.join("probe_x.npy"))?.to_device(&dev)?;
    let stamp = kronos_core::npy::read_npy(&probe_dir.join("probe_stamp.npy"))?.to_device(&dev)?;
    let ctx_len = x.dim(1)?;
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
