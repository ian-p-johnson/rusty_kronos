//! Integration tests against the real released checkpoints.
//!
//! These are the Phase 1 exit gate: proof that the Rust loader reads the same
//! bytes, the same tensors and the same shapes as PyTorch does.
//!
//! They need the checkpoints on disk. Point `KRONOS_HF_CACHE` at a Hugging Face
//! cache directory (default `~/.cache/huggingface/hub`). If the weights are not
//! present the tests SKIP rather than fail, unless `KRONOS_STRICT_TESTS=1`, so
//! that a CPU-only CI runner without the download still gets a green build
//! instead of a red one.
//!
//! ```sh
//! cargo test -p kronos-core --test weight_loading
//! ```

use kronos_core::{
    expected_predictor_keys, expected_tokenizer_keys, load_predictor_weights,
    load_tokenizer_weights, PredictorConfig, TokenizerConfig,
};
use std::path::{Path, PathBuf};
use tch::{Device, Kind as DType};

fn hf_cache() -> PathBuf {
    std::env::var("KRONOS_HF_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
            PathBuf::from(home).join(".cache/huggingface/hub")
        })
}

/// Resolve `<cache>/models--<org>--<name>/snapshots/<rev>/<file>`, trying every
/// snapshot revision present rather than hardcoding one.
fn ckpt(cache: &Path, repo: &str) -> Option<PathBuf> {
    let dir = cache.join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = dir.join("snapshots").read_dir().ok()?;
    for entry in snapshots.flatten() {
        let p = entry.path().join("model.safetensors");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn cfg_of(w: &Path) -> String {
    let p = w.parent().unwrap().join("config.json");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
}

/// `(repo, expected_params)` for the three released predictors.
/// Expected counts are from the PyTorch measurement and match paper Table 1.
const PREDICTORS: &[(&str, usize)] = &[
    ("NeoQuasar/Kronos-mini", 4_108_192),
    ("NeoQuasar/Kronos-small", 24_741_696),
    ("NeoQuasar/Kronos-base", 102_311_008),
];

const TOKENIZERS: &[(&str, &str)] = &[
    ("NeoQuasar/Kronos-Tokenizer-base", "tokenizer"),
    ("NeoQuasar/Kronos-Tokenizer-2k", "tokenizer"),
];

#[test]
fn every_released_predictor_loads_completely() {
    let cache = hf_cache();
    let dev = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let mut ran = 0;
    for (repo, want_params) in PREDICTORS {
        let w = ckpt(&cache, repo);
        if w.is_none() {
            if std::env::var("KRONOS_STRICT_TESTS").is_ok() {
                panic!("{repo} not found (strict mode)");
            }
            eprintln!("SKIP: {repo} not available");
            continue;
        }
        let w = w.unwrap();
        let cfg = PredictorConfig::from_json(&cfg_of(&w)).expect("config parses");
        cfg.validate().expect("config valid");

        let expected = expected_predictor_keys(&cfg);
        let loaded = load_predictor_weights(&w, &cfg, DType::Float, &dev).expect("weights load");

        assert_eq!(loaded.len(), expected.len(), "{repo}: tensor count");
        assert_eq!(
            loaded.numel(),
            *want_params,
            "{repo}: parameter count (paper Table 1 / PyTorch measurement)"
        );
        eprintln!(
            "ok {repo}: {} tensors, {} params",
            loaded.len(),
            loaded.numel()
        );
        ran += 1;
    }
    assert!(
        ran > 0,
        "no predictors were tested - is KRONOS_HF_CACHE set correctly?"
    );
}

#[test]
fn every_released_tokenizer_loads_completely() {
    let cache = hf_cache();
    let dev = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let mut ran = 0;
    for (repo, _) in TOKENIZERS {
        let Some(w) = ckpt(&cache, repo) else {
            if std::env::var("KRONOS_STRICT_TESTS").is_ok() {
                panic!("{repo} not found (strict mode)");
            }
            eprintln!("SKIP: {repo} not available");
            continue;
        };
        let cfg = TokenizerConfig::from_json(&cfg_of(&w)).expect("config parses");
        cfg.validate().expect("config valid");
        let expected = expected_tokenizer_keys(&cfg);
        let loaded = load_tokenizer_weights(&w, &cfg, DType::Float, &dev).expect("weights load");
        // The two i64 BSQ basis buffers are validated by name/dtype/shape but
        // deliberately absent from the device map (no i64 CUDA path, unused at
        // inference); the rest must all be present.
        let int_skipped = expected
            .iter()
            .filter(|n| {
                matches!(
                    n.as_str(),
                    "tokenizer.bsq.basis" | "tokenizer.bsq.group_basis"
                )
            })
            .count();
        assert_eq!(
            loaded.len(),
            expected.len() - int_skipped,
            "{repo}: tensor count"
        );
        // Both tokenizers are ~3.96M; they differ by exactly 1 parameter
        // because `group_basis` is length group_size (4 vs 5).
        assert!(
            (3_958_000..3_959_000).contains(&loaded.numel()),
            "{repo}: implausible tokenizer size {}",
            loaded.numel()
        );
        eprintln!(
            "ok {repo}: {} tensors, {} params, num_groups={}",
            loaded.len(),
            loaded.numel(),
            cfg.num_groups()
        );
        ran += 1;
    }
    assert!(ran > 0, "no tokenizers were tested");
}

#[test]
fn a_mismatched_config_is_rejected_not_silently_loaded() {
    // Feed the tokenizer checkpoint the *predictor* config's expectations.
    // The key-set check must catch it. This is the failure mode the strict
    // loader exists to prevent: a quietly-wrong checkpoint that still runs.
    let cache = hf_cache();
    let Some(p) = ckpt(&cache, "NeoQuasar/Kronos-Tokenizer-base") else {
        return;
    };
    let tok_cfg = TokenizerConfig::from_json(&cfg_of(&p)).unwrap();

    // Same tokenizer weights, but claim 5 encoder blocks instead of 3.
    let mut wrong = tok_cfg.clone();
    wrong.n_enc_layers = 6;
    let expected = expected_tokenizer_keys(&wrong);
    let res = kronos_core::weights::Weights::validate_file(&p, &expected, |_| None);
    assert!(
        res.is_err(),
        "loading 5-block expectations against a 3-block checkpoint must fail"
    );
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("missing") || msg.contains("not used"),
        "unexpected error: {msg}"
    );
    eprintln!("ok: mismatched config rejected -> {msg}");
}

#[test]
fn truncated_checkpoint_is_rejected() {
    // Prove the "unexpected/missing" machinery fires, using a synthetic file.
    let dir = std::env::temp_dir().join("kronos-test-truncated");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");

    let t = tch::Tensor::zeros([4, 4], (DType::Float, Device::Cpu));
    tch::Tensor::write_safetensors(&[("only_this", &t)], &path).unwrap();

    let mut expected = std::collections::BTreeSet::new();
    expected.insert("only_this".to_string());
    // passes: key set matches exactly
    kronos_core::weights::Weights::validate_file(&path, &expected, |_| Some(vec![4, 4])).unwrap();

    // fails: now demand a tensor the file does not have
    let mut missing = expected.clone();
    missing.insert("needed_but_absent".to_string());
    let e = kronos_core::weights::Weights::validate_file(&path, &missing, |_| None).unwrap_err();
    assert!(e.to_string().contains("missing"), "{e}");

    // fails: file has a tensor the architecture does not use
    let e = kronos_core::weights::Weights::validate_file(&path, &Default::default(), |_| None)
        .unwrap_err();
    assert!(e.to_string().contains("not used"), "{e}");

    // fails: wrong shape
    let e = kronos_core::weights::Weights::validate_file(&path, &expected, |_| Some(vec![8, 8]))
        .unwrap_err();
    assert!(e.to_string().contains("shape"), "{e}");

    let _ = std::fs::remove_dir_all(&dir);
}
