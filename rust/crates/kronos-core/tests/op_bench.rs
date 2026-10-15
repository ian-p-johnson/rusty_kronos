//! Same per-op probe as /tmp/op_probe_torch.py, through our engine.
use candle_core::{DType, Device, Tensor};
use kronos_core::ops;

fn bench<F: FnMut() -> candle_core::Result<Tensor>>(
    mut f: F,
    reps: u32,
) -> candle_core::Result<f64> {
    for _ in 0..10 {
        let _ = warm(&mut f);
    }
    let dev = Device::new_cuda(0).unwrap();
    dev.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    let mut last = None;
    for _ in 0..reps {
        last = Some(f()?);
    }
    dev.synchronize()?;
    drop(last);
    Ok(t0.elapsed().as_secs_f64() / f64::from(reps) * 1e3)
}

fn softmax_manual(t: &Tensor) -> candle_core::Result<Tensor> {
    // exactly ops::sdpa's softmax chain (max, sub, exp, sum keepdim, div)
    let last = t.dims().len() - 1;
    let max = t.max_keepdim(last)?;
    let mut e = t.broadcast_sub(&max)?.exp()?;
    let s = e.sum_keepdim(last)?;
    e = e.broadcast_div(&s)?;
    Ok(e)
}

fn warm<F: FnMut() -> candle_core::Result<Tensor>>(f: &mut F) -> candle_core::Result<()> {
    let _ = f()?;
    Ok(())
}

#[test]
fn op_bench() -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    let d = &Device::Cpu;
    let (b, l, dm, ff) = (10usize, 632usize, 512usize, 1024usize);
    let mk = |n: usize, dev: &Device| -> candle_core::Result<Tensor> {
        let v: Vec<f32> = (0..n).map(|i| ((i % 101) as f32) * 0.01 - 0.5).collect();
        Tensor::from_vec(v, n, dev)
    };
    let x = mk(b * l * dm, d)?.reshape((b, l, dm))?.to_device(&dev)?;
    let w1 = mk(dm * ff, d)?.reshape((ff, dm))?.to_device(&dev)?;
    let bb = mk(ff, d)?.to_device(&dev)?;
    let rw = mk(dm, d)?.to_device(&dev)?;
    let lg = mk(b * 4274, d)?.reshape((b, 4274))?.to_device(&dev)?;
    let dbg = mk(b * (1 << 21), d)?
        .reshape((b, 1 << 21))?
        .to_device(&dev)?;

    let q = mk(b * l * dm, d)?.reshape((b, l, dm))?.to_device(&dev)?;
    let kw = mk(dm * dm, d)?.reshape((dm, dm))?.to_device(&dev)?;
    let vb = mk(dm * dm, d)?.reshape((dm, dm))?.to_device(&dev)?;

    let reps = 200;
    println!(
        "rust   linear 10x632x512->1024: {:8.3} ms",
        bench(|| ops::linear(&x, &w1, Some(&bb)), reps)?
    );
    println!(
        "rust   rmsnorm 10x632x512:      {:8.3} ms",
        bench(|| ops::rms_norm(&x, &rw, 1e-6), reps)?
    );
    println!(
        "rust   softmax [10,4274]:       {:8.3} ms",
        bench(|| softmax_manual(&lg), reps)?
    );
    println!(
        "rust   softmax [10,2097152]:    {:8.3} ms",
        bench(|| softmax_manual(&dbg), reps)?
    );
    println!(
        "rust   linear 10x512x512 q:     {:8.3} ms",
        bench(|| ops::linear(&q, &kw, None), reps)?
    );
    let _ = (DType::F32, vb);
    Ok(())
}
