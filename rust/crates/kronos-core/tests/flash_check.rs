//! The flash kernel must reproduce ops::sdpa's manual chain (whose semantics
//! are empirically backed against PyTorch) within our 1e-5 rtol gate.
use candle_core::{Device, Tensor};

fn dev() -> Device {
    // mirrors the harness device choice; one RTX 5070 in this box
    Device::new_cuda(0).unwrap()
}

fn make(b: usize, h: usize, lq: usize, lk: usize, hd: usize) -> (Tensor, Tensor, Tensor) {
    // deterministic values created on the CPU then moved to CUDA, so the
    // comparison is repeatable on any driver / GPU clock state
    let d = Device::Cpu;
    let mut xs = vec![7.0f32; b * h * lq * hd];
    let mut xd = vec![3.0f32; b * h * lk * hd];
    let mut u = 17u64;
    for x in &mut xs {
        u = u.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(123);
        *x = ((u >> 33) % 2000) as f32 / 1000.0 - 1.0;
    }
    for x in &mut xd {
        u = u.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(123);
        *x = ((u >> 33) % 2000) as f32 / 1000.0 - 1.0;
    }
    let q = Tensor::from_vec(xs, (b, h, lq, hd), &d)
        .unwrap()
        .to_device(&dev())
        .unwrap();
    let k = Tensor::from_vec(xd.clone(), (b, h, lk, hd), &d)
        .unwrap()
        .to_device(&dev())
        .unwrap();
    let v = Tensor::from_vec(xd, (b, h, lk, hd), &d)
        .unwrap()
        .to_device(&dev())
        .unwrap();
    (q, k, v)
}

fn case(b: usize, h: usize, lq: usize, lk: usize, hd: usize, causal: bool) {
    let (q, k, v) = make(b, h, lq, lk, hd);
    std::env::set_var("KRONOS_NO_FLASH", "1");
    let manual = kronos_core::ops::sdpa(&q, &k, &v, causal).unwrap();
    std::env::remove_var("KRONOS_NO_FLASH");
    let fused = kronos_core::flash::fused(&q, &k, &v, causal).unwrap();
    let m = manual
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let f = fused
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // repo parity convention (parity.rs): rel <= rtol OR abs <= rtol — on the
    // softmax path both accumulation orders carry O(1 ULP) noise on
    // small-magnitude outputs (1e-3-scale tails), which rel-only metrics
    // would flag as catastrophic.
    let mut max_rel = 0f32;
    let mut max_abs = 0f32;
    let mut wv = (0f32, 0f32);
    for (a, c) in m.iter().zip(f.iter()) {
        let d = (a - c).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / c.abs().max(1e-6));
        if d / c.abs().max(1e-6) > max_rel {
            wv = (*a, *c);
        }
    }
    if max_rel >= 1e-5 && std::env::var("KRONOS_FLASH_DEBUG").is_ok() {
        println!(
            "worst-rel manual={:.9} fused={:.9} abs={} rel={:.2e}",
            wv.0, wv.1, max_abs, max_rel
        );
    }
    assert!(
        max_rel < 1e-5 || max_abs < 1e-5,
        "b={b} h={h} lq={lq} lk={lk} hd={hd} causal={causal}: rel {max_rel:.2e} abs {max_abs:.2e}"
    );
}

#[test]
fn flash_matches_manual() {
    // the distinct head dims / shapes the predictor and tokenizer use
    case(2, 8, 632, 512, 64, true); // small predictor s1 self-attention
    case(3, 8, 1, 512, 64, true); // s2 step: one query row
    case(2, 4, 631, 512, 52, true); // base head dim
    case(1, 4, 256, 512, 64, false); // tokenizer cross / non-causal
    case(1, 4, 1024, 512, 64, true); // tokenizer big window
    case(4, 16, 77, 77, 64, true); // mini
}
