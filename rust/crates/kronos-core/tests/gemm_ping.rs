//! gemm_fast engagement + numerics: compare new path vs candle matmul bits.
use candle_core::{Device, Tensor};
use kronos_core::ops;

#[test]
fn gemm_fast_engaged() -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    let d = Device::Cpu;
    let m = 632;
    let k = 512;
    let n = 1024;
    let mut u = 5u64;
    let mut mk = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                u = u.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                ((u >> 33) % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    };
    let x2 = Tensor::from_vec(mk(m * k), (m, k), &d)?.to_device(&dev)?;
    let w = Tensor::from_vec(mk(n * k).into_iter().collect::<Vec<f32>>(), (n, k), &d)?
        .to_device(&dev)?;
    // torch-linear form uses w as [n, k]; candle matmul path needs [k, n]
    let ref_t = x2.matmul(&w.t()?)?;
    let got = ops::linear(&x2.reshape((1, m, k))?.contiguous()?, &w, None)?.reshape((m, n))?;
    let a = ref_t.flatten_all()?.to_vec1::<f32>()?;
    let c = got.flatten_all()?.to_vec1::<f32>()?;
    let mut err = 0f64;
    for (p, q) in a.iter().zip(c.iter()) {
        err = err.max(((*p - *q).abs() / p.abs().max(1e-3)) as f64);
    }
    println!("gemm_fast vs candle matmul: rel err {err:.2e}");
    assert!(err < 1e-6, "gemm_fast mismatch");
    Ok(())
}
