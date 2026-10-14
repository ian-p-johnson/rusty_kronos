//! Phase 4 gates.
//!
//! Run with the checkpoints present:
//!
//! ```sh
//! cargo test -p kronos-core --test phase4
//! KRONOS_STRICT_TESTS=1 cargo test -p kronos-core --test phase4   # goldens must run
//! ```
//!
//! Three gates, in order of cost:
//! 1. **stamp** — the CSV loader and calendar features against
//!    `parity/fixtures/probe_stamp.npy` (Monday=0..Sunday=6, exactly pandas).
//! 2. **golden** — the six deterministic goldens (3 models × ctx 256/512,
//!    top_k=1 ⇒ argmax) at `rtol=1e-5`, end-to-end from the raw CSV.
//! 3. **sampler** — the three stochastic assertions, in-process.

use candle_core::{DType, Device, Tensor};
use kronos_core::ar;
use kronos_core::predictor::Kronos;
use kronos_core::sampling::{self, SampleCfg};
use kronos_core::tokenizer::KronosTokenizer;
use kronos_core::{
    load_csv, load_predictor_weights, load_tokenizer_weights, PredictorConfig, TokenizerConfig,
};
use std::path::PathBuf;

const RTOL: f64 = 1e-5;

fn repo() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/rust/crates/kronos-core at compile time,
    // three levels up is the repo root regardless of the runtime cwd.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn fixtures() -> PathBuf {
    repo().join("parity/fixtures")
}

fn csv() -> PathBuf {
    repo().join("tests/data/regression_input.csv")
}

fn hf_cache() -> PathBuf {
    std::env::var("KRONOS_HF_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
                .join(".cache/huggingface/hub")
        })
}

fn snapshot(repo_name: &str) -> Option<PathBuf> {
    let dir = hf_cache().join(format!("models--{}", repo_name.replace('/', "--")));
    let mut best: Option<PathBuf> = None;
    if let Ok(entries) = std::fs::read_dir(dir.join("snapshots")) {
        for e in entries.flatten() {
            if e.path().join("model.safetensors").exists() {
                best = Some(e.path());
                break;
            }
        }
    }
    best
}

fn require(path: &std::path::Path, what: &str) -> PathBuf {
    match path.exists() {
        true => path.to_path_buf(),
        false => {
            if std::env::var("KRONOS_STRICT_TESTS").is_ok() {
                panic!("{what} not found ({})", path.display());
            }
            eprintln!("SKIP: {what} not available at {}", path.display());
            std::process::exit(0);
        }
    }
}

struct Loaded {
    tok: KronosTokenizer,
    mdl: Kronos,
    dev: Device,
}

fn load(model_key: &str, tok_repo: &str, dtype: DType) -> Loaded {
    let dev = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let model_snap = require(
        &snapshot(&format!("NeoQuasar/Kronos-{model_key}"))
            .unwrap_or_else(|| PathBuf::from("<missing>")),
        &format!("Kronos-{model_key}"),
    );
    let tok_snap = require(
        &snapshot(tok_repo).unwrap_or_else(|| PathBuf::from("<missing>")),
        tok_repo,
    );
    let tcfg =
        TokenizerConfig::from_json(&std::fs::read_to_string(tok_snap.join("config.json")).unwrap())
            .unwrap();
    tcfg.validate().unwrap();
    let mcfg = PredictorConfig::from_json(
        &std::fs::read_to_string(model_snap.join("config.json")).unwrap(),
    )
    .unwrap();
    mcfg.validate().unwrap();
    let tw =
        load_tokenizer_weights(&tok_snap.join("model.safetensors"), &tcfg, dtype, &dev).unwrap();
    let mw =
        load_predictor_weights(&model_snap.join("model.safetensors"), &mcfg, dtype, &dev).unwrap();
    Loaded {
        tok: KronosTokenizer::load(&tw, &tcfg, 512, &dev, dtype).unwrap(),
        mdl: Kronos::load(&mw, &mcfg, 512, &dev, dtype).unwrap(),
        dev,
    }
}

// ---------------------------------------------------------------- gate 1
/// The CSV loader + calendar features must reproduce the captured stamp
/// fixture exactly (integers in f32, compared for exact equality).
#[test]
fn stamps_match_the_fixture() {
    require(&csv(), "regression_input.csv");
    require(&fixtures().join("probe_stamp.npy"), "probe_stamp.npy");
    let rows = load_csv(&csv()).unwrap();
    assert!(rows.len() >= 512, "fixture CSV has {} rows", rows.len());
    let dev = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let got = kronos_core::data::stamps(&rows[..512], &dev).unwrap();
    let want = kronos_core::npy::read_npy(&fixtures().join("probe_stamp.npy")).unwrap();

    let g = got
        .flatten_all()
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let w = want
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(g.len(), w.len());
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert_eq!(a, b, "stamp[{i}]: rust {a} != oracle {b}");
    }
    println!("stamps: 512×5 calendar features reproduce the oracle exactly");
}

// ---------------------------------------------------------------- gate 2
fn run_golden(model_key: &str, tok_repo: &str, ctx: usize) {
    let pred_len = 8;
    let golden = fixtures().join(format!(
        "golden/kronos-{model_key}_ctx{ctx}_pred{pred_len}.csv"
    ));
    let golden = require(&golden, &format!("golden for {model_key} ctx {ctx}"));

    let loaded = load(model_key, tok_repo, DType::F32);
    let rows = load_csv(&csv()).unwrap();
    let mut p = ar::ArParams::new(
        512,
        pred_len,
        1,
        SampleCfg {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        },
    );
    let _ = &mut p;
    p.pred_len = pred_len;
    let used = rows.len().min(ctx + pred_len);
    let slice = &rows[..used];
    let mut rng = sampling::Rng::new(123);
    let out = ar::predict(&loaded.tok, &loaded.mdl, slice, &p, &mut rng, &loaded.dev).unwrap();

    let d = out.forecast.dim(2).unwrap();
    let flat = out
        .forecast
        .flatten_all()
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let text = std::fs::read_to_string(&golden).unwrap();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let header: Vec<&str> = lines[0].split(',').map(|c| c.trim()).collect();
    let names = ["open", "high", "low", "close", "volume", "amount"];
    let n_rows = lines.len() - 1;
    assert_eq!(n_rows, pred_len, "golden rows");

    let mut worst = 0.0f64;
    for r in 0..n_rows {
        let parts: Vec<&str> = lines[1 + r].split(',').map(|c| c.trim()).collect();
        for (k, name) in names.iter().enumerate() {
            let j = header.iter().position(|h| h == name).expect("column");
            let want: f64 = parts[j].parse().expect("float");
            let got = flat[r * d + k] as f64;
            let rel = (got - want).abs() / want.abs().max(1e-12);
            worst = worst.max(rel);
            assert!(
                (got - want).abs() <= RTOL * want.abs().max(1e-12),
                "{model_key} ctx={ctx} row {r} {name}: rust {got} vs oracle {want} (rel {rel:.3e})"
            );
        }
    }
    println!(
        "golden {:<12} ctx={:<3}  worst rel = {:.2e}  (rtol {:.0e})",
        model_key, ctx, worst, RTOL
    );
}

#[test]
fn golden_kronos_small_ctx512() {
    run_golden("small", "NeoQuasar/Kronos-Tokenizer-base", 512);
}

#[test]
fn golden_kronos_small_ctx256() {
    run_golden("small", "NeoQuasar/Kronos-Tokenizer-base", 256);
}

#[test]
fn golden_kronos_base_ctx512() {
    run_golden("base", "NeoQuasar/Kronos-Tokenizer-base", 512);
}

#[test]
fn golden_kronos_base_ctx256() {
    run_golden("base", "NeoQuasar/Kronos-Tokenizer-base", 256);
}

#[test]
fn golden_kronos_mini_ctx512() {
    run_golden("mini", "NeoQuasar/Kronos-Tokenizer-2k", 512);
}

#[test]
fn golden_kronos_mini_ctx256() {
    run_golden("mini", "NeoQuasar/Kronos-Tokenizer-2k", 256);
}

// ---------------------------------------------------------------- gate 3
/// The three agreed stochastic assertions over one real distribution.
#[test]
fn sampler_audit() {
    require(&csv(), "regression_input.csv");
    let loaded = load("small", "NeoQuasar/Kronos-Tokenizer-base", DType::F32);
    let rows = load_csv(&csv()).unwrap();
    let dev = loaded.dev;
    let ctx = &rows[..512];

    let xs = kronos_core::data::stamps(ctx, &dev)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let v = kronos_core::data::values(ctx, &dev).unwrap();
    let mu = v.mean_keepdim(0).unwrap();
    let std = v
        .broadcast_sub(&mu)
        .unwrap()
        .sqr()
        .unwrap()
        .mean_keepdim(0)
        .unwrap()
        .sqrt()
        .unwrap();
    let eps = Tensor::from_vec(vec![1e-5f32], (), &dev).unwrap();
    let norm = v
        .broadcast_sub(&mu)
        .unwrap()
        .broadcast_div(&std.broadcast_add(&eps).unwrap())
        .unwrap();
    let norm = norm
        .broadcast_maximum(&Tensor::from_vec(vec![-5.0f32], (), &dev).unwrap())
        .unwrap()
        .broadcast_minimum(&Tensor::from_vec(vec![5.0f32], (), &dev).unwrap())
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let toks = loaded.tok.encode(&norm).unwrap();
    let d1 = loaded.mdl.decode_s1(&toks.0, &toks.1, &xs).unwrap();
    let logits = d1.s1_logits.narrow(1, 511, 1).unwrap().squeeze(1).unwrap(); // [1, V]

    // Deterministic identity: the argmax is the committed pre_argmax fixture.
    let argmax = logits
        .argmax(logits.dims().len() - 1)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
    let fixture = fixtures()
        .join("layers/kronos-small")
        .join("pre_argmax.npy");
    if fixture.exists() {
        let want = kronos_core::npy::read_npy(&fixture).unwrap();
        let want = want
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            argmax[0] as f32, want[0],
            "argmax token must match the oracle exactly (got {}, want {})",
            argmax[0], want[0]
        );
    }

    // stochastic regime
    let cfg = SampleCfg {
        temperature: 0.6,
        top_k: 0,
        top_p: 0.9,
    };
    let draws = 512usize;
    let mut rng = sampling::Rng::new(123);
    let mut ids: Vec<u32> = Vec::with_capacity(draws);
    for _ in 0..draws {
        let id = sampling::sample(&logits, &cfg, &mut rng).unwrap();
        ids.push(
            id.to_device(&Device::Cpu)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()[0],
        );
    }

    // (2) every draw inside the nucleus: replicate the single-row distribution
    // `draws` times so the rank-2 [B, V] contract holds with one id per row.
    let scaled = logits.affine(1.0 / 0.6, 0.0).unwrap();
    let scaled_rep =
        Tensor::cat(&(0..draws).map(|_| scaled.clone()).collect::<Vec<_>>(), 0).unwrap();
    let ids_t = Tensor::from_vec(ids.clone(), (draws,), &dev).unwrap();
    let mass = sampling::strictly_greater_mass(&scaled_rep, &ids_t).unwrap();
    let mass = mass
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let n_outside = mass.iter().filter(|m| **m > 0.9 + 1e-6).count();
    assert_eq!(
        n_outside, 0,
        "assertion 2 failed: {n_outside} draws landed outside the top_p=0.9 nucleus"
    );

    // (1) top-1 empirical frequency vs the analytic probability
    let probs = sampling::softmax(&scaled).unwrap();
    let p_max = probs
        .narrow(0, 0, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[argmax[0] as usize] as f64;
    let n_in = ids.iter().filter(|x| **x == argmax[0]).count();
    let freq = n_in as f64 / draws as f64;
    assert!(
        (freq - p_max).abs() <= 0.10f64.max(0.10 * p_max),
        "assertion 1 failed: empirical {freq:.4} vs analytic {p_max:.4}"
    );

    // (3) draws are not constant
    assert!(
        ids.windows(2).any(|w| w[0] != w[1]),
        "assertion 3 failed: draws are constant"
    );

    println!(
        "sampler audit OK: {draws} draws, argmax {} p={p_max:.4} freq={freq:.4}, 0 outside nucleus, non-constant",
        argmax[0]
    );
}
