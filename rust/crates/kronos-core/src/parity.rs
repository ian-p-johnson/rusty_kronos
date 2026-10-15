//! Per-stage parity harness.
//!
//! Reproduces exactly the sequence `parity/gen_reference.py::capture_layers`
//! ran, then compares every captured tensor against the committed fixture.
//!
//! The point of comparing *stages* rather than just the final output is that a
//! RoPE-convention error, a wrong head count, or a swapped bias shows up as a
//! mismatch on one specific stage, which names the bug. Comparing only the end
//! number would show "something is slightly wrong" and nothing else.

use crate::npy::{fixture_path, read_npy};
use crate::ops::Result;
use crate::predictor::Kronos;
use crate::tokenizer::KronosTokenizer;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use tch::{Device, Kind, Tensor};

/// One tensor's parity verdict.
#[derive(Serialize, Debug)]
pub struct StageReport {
    pub name: String,
    pub shape: Vec<usize>,
    pub max_abs: f64,
    pub max_rel: f64,
    pub pass: bool,
}

/// All the stages a single probe run produced.
#[derive(Debug, Default)]
pub struct ProbeOutput {
    pub tensors: BTreeMap<String, Tensor>,
}

impl ProbeOutput {
    pub fn insert(&mut self, name: &str, t: Tensor) {
        self.tensors.insert(name.to_string(), t);
    }
}

/// Run one forward pass through the tokenizer and the predictor, capturing the
/// same stages the reference did.
///
/// `x` is `[1, T, 6]` normalised OHLCVA; `stamp` is `[1, T, 5]` calendar features.
pub fn run_probe(
    tok: &KronosTokenizer,
    model: &Kronos,
    x: &Tensor,
    stamp: &Tensor,
) -> Result<ProbeOutput> {
    let mut out = ProbeOutput::default();
    let (s1_ids, s2_ids) = tok.encode(x)?;
    out.insert("token_s1", s1_ids.to_kind(Kind::Double));
    out.insert("token_s2", s2_ids.to_kind(Kind::Double));

    let recon = tok.decode(&(s1_ids.shallow_clone(), s2_ids.shallow_clone()))?;
    out.insert("tokenizer_recon", recon.to_kind(Kind::Double));

    let d1 = model.decode_s1(&s1_ids, &s2_ids, stamp)?;
    let last = |t: &Tensor| -> Result<Tensor> {
        Ok(t.narrow(1, t.size()[1] - 1, 1)
            .squeeze_dim(1)
            .to_kind(Kind::Double))
    };
    out.insert("s1_logits_last", last(&d1.s1_logits)?);
    out.insert("context_last", last(&d1.context)?);

    let pre_arg = d1
        .s1_logits
        .narrow(1, d1.s1_logits.size()[1] - 1, 1)
        .f_argmax(2, false)?;
    out.insert("pre_argmax", pre_arg.to_kind(Kind::Double));

    let s2_logits = model.decode_s2(&d1.context, &pre_arg)?;
    out.insert("s2_logits_last", last(&s2_logits)?);
    Ok(out)
}

/// Compare every captured stage against the committed fixture.
///
/// Floating-point stages are accepted at `rtol` (relative, matching the
/// upstream `np.testing.assert_allclose` budget). Integer stages — the token
/// ids and the argmax — must match **exactly**, because a one-off token id is a
/// categorical error, not a rounding error.
pub fn compare_stages(out: &ProbeOutput, fixture_dir: &Path, rtol: f64) -> Vec<StageReport> {
    let mut reports = Vec::new();
    for (name, got) in &out.tensors {
        let path = fixture_path(fixture_dir, name);
        if !path.exists() {
            reports.push(StageReport {
                name: name.clone(),
                shape: got.size().iter().map(|&d| d as usize).collect(),
                max_abs: f64::NAN,
                max_rel: f64::NAN,
                pass: false,
            });
            continue;
        }
        let want = match read_npy(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("warn: could not read {}: {e}", path.display());
                continue;
            }
        };
        // fixtures are stored on CPU; bring ours over and cast to the same dtype
        let got_cpu = got
            .f_to(Device::Cpu)
            .unwrap_or_else(|_| got.shallow_clone());
        let want_f64 = want.to_kind(Kind::Double);
        let got_f64 = got_cpu.to_kind(Kind::Double);

        if got_f64.size() != want_f64.size() {
            reports.push(StageReport {
                name: name.clone(),
                shape: got_f64.size().iter().map(|&d| d as usize).collect(),
                max_abs: f64::INFINITY,
                max_rel: f64::INFINITY,
                pass: false,
            });
            continue;
        }

        let (max_abs, max_rel) = diff_stats(&got_f64, &want_f64);
        let integer_stage = name.contains("token_s") || name.contains("pre_argmax");
        let pass = if integer_stage {
            max_abs == 0.0
        } else {
            max_rel <= rtol || max_abs <= rtol
        };
        reports.push(StageReport {
            name: name.clone(),
            shape: got_f64.size().iter().map(|&d| d as usize).collect(),
            max_abs,
            max_rel,
            pass,
        });
    }
    reports.sort_by(|a, b| a.name.cmp(&b.name));
    reports
}

fn diff_stats(a: &Tensor, b: &Tensor) -> (f64, f64) {
    let af = a.to_kind(Kind::Double).reshape([-1]);
    let bf = b.to_kind(Kind::Double).reshape([-1]);
    let (Ok(af_vec), Ok(bf_vec)) = (Vec::<f64>::try_from(&af), Vec::<f64>::try_from(&bf)) else {
        return (f64::NAN, f64::NAN);
    };
    let av = af_vec;
    let bv = bf_vec;
    if av.len() != bv.len() {
        return (f64::INFINITY, f64::INFINITY);
    }
    let mut max_abs = 0.0f64;
    let mut max_val = 1e-12f64;
    let mut nan_count = 0usize;
    for (x, y) in av.iter().zip(&bv) {
        if x.is_nan() || y.is_nan() {
            nan_count += 1;
            max_abs = f64::INFINITY;
            continue;
        }
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let m = x.abs().max(y.abs());
        if m > max_val {
            max_val = m;
        }
    }
    if nan_count > 0 {
        eprintln!("  note: {nan_count} NaN values in comparison");
        return (f64::INFINITY, f64::INFINITY);
    }
    (max_abs, max_abs / max_val)
}

pub fn print_report(reports: &[StageReport]) -> usize {
    let mut fails = 0usize;
    println!(
        "{:<34} {:>12} {:>12} {:>8}",
        "stage", "max_abs", "max_rel", "verdict"
    );
    println!("{}", "-".repeat(70));
    for r in reports {
        let verdict = if r.pass { "ok" } else { "FAIL" };
        if !r.pass {
            fails += 1;
        }
        println!(
            "{:<34} {:>12.4e} {:>12.4e} {:>8}",
            r.name, r.max_abs, r.max_rel, verdict
        );
    }
    fails
}
