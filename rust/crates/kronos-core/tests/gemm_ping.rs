//! gemm numerics: compare ops::linear against plain tch matmul bits.
use kronos_core::ops;
use tch::{Device, Tensor};

#[test]
fn gemm_fast_engaged() -> Result<(), Box<dyn std::error::Error>> {
    if !tch::utils::has_cuda() {
        return Ok(());
    }
    let dev = Device::Cuda(0);
    let _d = Device::Cpu;
    let m = 632i64;
    let k = 512i64;
    let n = 1024i64;
    let mut u = 5u64;
    let mut mk = move |len: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            u = u.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            out.push(((u >> 33) % 2000) as f32 / 1000.0 - 1.0);
        }
        out
    };
    let x2 = Tensor::from_slice(&mk((m * k) as usize))
        .reshape([m, k])
        .to_device(dev);
    let w = Tensor::from_slice(&mk((n * k) as usize))
        .reshape([n, k])
        .to_device(dev);
    // ops::linear torch-linear form: x @ w^T (w is [n, k])
    let got = ops::linear(&x2.reshape([1, m, k]), &w, None)?.reshape([m, n]);
    let ref_t = x2.matmul(&w.transpose(1, 0));
    let a: Vec<f32> = (&ref_t.to_device(Device::Cpu).reshape([-1])).try_into()?;
    let c: Vec<f32> = (&got.to_device(Device::Cpu).reshape([-1])).try_into()?;
    let err = a
        .iter()
        .zip(c.iter())
        .map(|(p, q)| ((*p - *q).abs() / p.abs().max(1e-3)) as f64)
        .fold(0f64, f64::max);
    println!("gemm_fast vs candle matmul: rel err {err:.2e}");
    assert!(err < 1e-6, "gemm_fast mismatch");
    Ok(())
}
