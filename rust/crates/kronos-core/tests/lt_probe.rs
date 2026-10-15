//! cuBLASLt probe: tensor-core (TF32) GEMM vs candle's classic-handle linear
//! at the FFN shape, plus the numeric error TF32 introduces (vs fp32
//! reference) — establishes whether an Lt-based GEMM path is the lever.
use candle_core::{Device, Tensor};
use cudarc::cublas::CudaBlas;
use cudarc::cublaslt::safe::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use std::sync::Arc;

fn slice_u32_clone(t: &Tensor) -> candle_core::Result<cudarc::driver::CudaSlice<f32>> {
    let (st, _) = t.storage_and_layout();
    match &*st {
        candle_core::Storage::Cuda(s) => match &s.slice {
            candle_core::cuda_backend::CudaStorageSlice::F32(v) => Ok(v.clone()),
            _ => Err(candle_core::Error::Msg("not f32".into())),
        },
        _ => Err(candle_core::Error::Msg("not cuda".into())),
    }
}

#[test]
fn lt_probe() -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    let d = Device::Cpu;
    let (b, l, k, n) = (10usize, 632usize, 512usize, 1024usize);

    let mk = |len: usize, seed: u64| -> Vec<f32> {
        let mut u = seed;
        (0..len)
            .map(|_| {
                u = u.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                ((u >> 33) % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    };
    let x = Tensor::from_vec(mk(b * l * k, 11), (b, l, k), &d)?.to_device(&dev)?;
    let w = Tensor::from_vec(mk(n * k, 22), (n, k), &d)?.to_device(&dev)?;

    // candle reference (classic cublas, true fp32)
    let ref_ts = kronos_core::ops::linear(&x, &w, None)?;

    let cdev = match dev {
        Device::Cuda(ref dd) => dd.clone(),
        _ => unreachable!(),
    };
    let blaslt =
        CudaBlasLT::new(Arc::clone(&cdev.cuda_stream())).map_err(candle_core::Error::msg)?;
    let xs = slice_u32_clone(&x)?;
    let ws = slice_u32_clone(&w)?;
    let mut cs: cudarc::driver::CudaSlice<f32> =
        unsafe { cdev.alloc(b * l * n).map_err(candle_core::Error::msg)? };

    // C(m=k?no) — cfg the col-major recipe for row-major x[l,k] * w[n,k]^T:
    // C row-major [l,n] viewed col-major (n, l); A = W row-major [n,k] viewed
    // col-major (k, n), transa=T -> (n, k); B = X row-major [l,k] viewed
    // col-major (k, l), no trans -> (k, l). C_col = A^T_col_op... i.e. m=n=k?
    let cfg = MatmulConfig {
        transa: true,  // op(A): (k,n) -> (n,k)
        transb: false, // op(B): (k,l)
        transc: false,
        m: n as u64,
        n: (l * b) as u64,
        k: k as u64,
        alpha: 1.0,
        beta: 0.0,
        lda: k as i64, // A [n,k] row-major = (k,n) col-major, lda=k(opA=T: lda=row dim k)
        ldb: k as i64, // B [l,k] row-major = (k,l) col-major, ldb=k
        ldc: n as i64, // C [l,n] row-major = (n,l) col-major, ldc=n
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };
    unsafe {
        blaslt
            .matmul(cfg, &ws, &xs, &mut cs, None, None)
            .map_err(|e| candle_core::Error::msg(format!("lt: {e:?}")))?;
    }
    let host = cdev
        .cuda_stream()
        .clone_dtoh(&cs)
        .map_err(|e| candle_core::Error::msg(format!("{e:?}")))?;
    let out = Tensor::from_vec(host, (b, l, n), &d)?;

    // compare on a batch slice
    let a = ref_ts.flatten_all()?;
    let c = out.flatten_all()?;
    let n_total = a.elem_count();
    let stride = n_total / 2048;
    let mut max_err = 0f64;
    for i in (0..n_total).step_by(stride.max(1)) {
        let av = a.to_device(&d)?.to_vec1::<f32>()?[i] as f64;
        let cv = c.to_device(&d)?.to_vec1::<f32>()?[i] as f64;
        max_err = max_err.max((av - cv).abs() / av.abs().max(1e-3) as f64);
    }
    println!("lt-vs-candle max rel err (sampled): {max_err:.2e}");

    // ---- timing: single-shot Lt vs candle linear ---
    let reps = 200u32;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        unsafe {
            blaslt
                .matmul(cfg, &ws, &xs, &mut cs, None, None)
                .map_err(|e| candle_core::Error::msg(format!("lt: {e:?}")))?;
        }
    }
    cdev.cuda_stream()
        .synchronize()
        .map_err(candle_core::Error::msg)?;
    let lt_ms = t0.elapsed().as_secs_f64() / f64::from(reps) * 1e3;

    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let _ = kronos_core::ops::linear(&x, &w, None)?;
    }
    cdev.cuda_stream()
        .synchronize()
        .map_err(candle_core::Error::msg)?;
    let cd_ms = t0.elapsed().as_secs_f64() / f64::from(reps) * 1e3;

    // flatten the batch to one 2-D gemm (what torch's broadcast matmul does)
    let x2 = x.reshape((l * b, k))?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let _ = ops_lin_flat(&x2, &w)?;
    }
    cdev.cuda_stream()
        .synchronize()
        .map_err(candle_core::Error::msg)?;
    let flat_ms = t0.elapsed().as_secs_f64() / f64::from(reps) * 1e3;
    // ---- raw cublas classic with tiny scope-numbed harness: single [6320]X[512] gemm
    // with CUBLAS_GEMM_DEFAULT vs candle's CUBLAS_GEMM_DEFAULT_TENSOR_OP choice.
    use cudarc::cublas::sys;
    let cublas = CudaBlas::new(Arc::clone(&cdev.cuda_stream())).map_err(candle_core::Error::msg)?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        // same shapes as candle's flattened 2d: m=1024,n=632,k=512,op transa
        let (lhs, _) = x2.storage_and_layout();
        let gpu_lhs = match &*lhs {
            candle_core::Storage::Cuda(cs) => clone_f32(&cs.slice)?,
            _ => unreachable!(),
        };
        let (rw, _) = w.storage_and_layout();
        let gpu_w = match &*rw {
            candle_core::Storage::Cuda(cs) => clone_f32(&cs.slice)?,
            _ => unreachable!(),
        };
        let mut acc: cudarc::driver::CudaSlice<f32> =
            unsafe { cdev.alloc(l * b * n).map_err(candle_core::Error::msg)? };
        let one: f32 = 1.0;
        let zero: f32 = 0.0;
        {
            let stream2 = cdev.cuda_stream();
            let (a_ptr, _a) = gpu_lhs.device_ptr(&stream2);
            let (b_ptr, _b) = gpu_w.device_ptr(&stream2);
            let (c_ptr, _c) = acc.device_ptr_mut(&stream2);
            unsafe {
                cudarc::cublas::result::gemm_strided_batched_ex(
                    *cublas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    1024,
                    632,
                    512,
                    &one as *const f32 as *const _,
                    b_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    512,
                    0,
                    a_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    512,
                    (632 * 512) as i64,
                    &zero as *const f32 as *const _,
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    1024,
                    (632 * 1024) as i64,
                    10,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(|e| candle_core::Error::msg(format!("cublas: {e:?}")))?;
            }
        }
    }
    cdev.cuda_stream()
        .synchronize()
        .map_err(candle_core::Error::msg)?;
    let raw_ms = t0.elapsed().as_secs_f64() / f64::from(reps) * 1e3;
    println!(
        "lt {lt_ms:.3}  candle-batched {cd_ms:.3}  candle-flat2d {flat_ms:.3}  raw-default-algo {raw_ms:.3}  (torch 0.54)"
    );
    Ok(())
}

fn clone_f32(
    s: &candle_core::cuda_backend::CudaStorageSlice,
) -> candle_core::Result<cudarc::driver::CudaSlice<f32>> {
    match s {
        candle_core::cuda_backend::CudaStorageSlice::F32(v) => Ok(v.clone()),
        _ => Err(candle_core::Error::msg("not f32")),
    }
}

fn ops_lin_flat(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    x.matmul(&w.transpose(0, 1)?)
}
