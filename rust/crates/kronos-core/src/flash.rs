//! Fused single-pass attention (online softmax) for the CUDA backend.
//!
//! Launches one PTX kernel (`rust/cuda/flash_attn.cu`, precompiled to
//! `flash_attn.ptx` at development time) through candle's custom-module
//! machinery (`CudaDevice::get_or_load_custom_func`, loaded once and cached
//! by candle inside the device). Scores never touch HBM, which removes the
//! dominant cost of the manual chain in `ops.rs` (~8.3 ms/call at
//! (10,8,512,64)); PyTorch gets the same effect from its memory-efficient
//! kernel at fp32 (0.38 ms). Shapes not covered by the kernel (non-fp32,
//! hd > 64, CPU) fall back to `None` so the caller can take the manual path.

use candle_core::cuda_backend::{CudaDType, CudaStorageSlice};
use candle_core::op::BackpropOp;
use candle_core::{DType, Device, Result, Shape, Storage, Tensor};
use cudarc::driver::PushKernelArg;

const PTX: &str = include_str!("flash_attn.ptx");
const MODULE_NAME: &str = "kronos-flash";
const KERNEL_NAME: &str = "flash_causal";
const TILE_W: usize = 64; // shared-tile width in the kernel (hd <= this)
                          // Tile/geometry knobs. The kernel takes them at runtime; the loader fixes the
                          // PTX (built once) but the tiling is decided per launch. ldb = rows per block
                          // == one warp per row; keys per tile == rows (shared tile is square 64x64).
                          // Defaults picked empirically on the RTX 5070 Ti Laptop (see microbench).
fn rows_per_block_and_tile() -> (u32, u32) {
    let rows = std::env::var("KRONOS_FLASH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let rows = rows.clamp(1, TILE_W as u32);
    (rows, rows)
}

fn not_supported(kind: &str) -> candle_core::Error {
    candle_core::Error::Msg(format!("flash: not supported: {kind}"))
}

fn slice_of(storage: &Storage) -> Result<&cudarc::driver::CudaSlice<f32>> {
    let Storage::Cuda(cs) = storage else {
        return Err(not_supported("not cuda"));
    };
    match &cs.slice {
        CudaStorageSlice::F32(data) => Ok(data),
        _ => Err(not_supported("dtype slice")),
    }
}

/// Kernel entry point: same semantics as `ops::sdpa` for fp32 CUDA tensors.
/// `q/k/v` are contiguous `[B, H, L, hd]` (std layout; verbatim from the row-
/// major concatenation the attention path passes) and the outputs match
/// torch's `is_causal` semantics. Exposed for tests and the microbench
/// head-to-head; the main dispatch is [`try_fused`].
pub fn fused(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    if q.dtype() != DType::F32 {
        return Err(not_supported("only f32"));
    }
    let (b, h, lq, hd) = q.dims4()?;
    let (kb, kh, lk, khd) = k.dims4()?;
    let (vb, vh, lv, vhd) = v.dims4()?;
    if kb != b || kh != h || khd != hd || vb != b || vh != h || lv != lk || vhd != hd {
        return Err(not_supported("shapes"));
    }

    let dev = match q.device() {
        Device::Cuda(d) => d,
        _ => return Err(not_supported("device")),
    };

    // Input storage read-guards keep the device memory alive through the
    // launch; they live to the end of this function.
    let (qs, _ql) = q.storage_and_layout();
    let (ks, _kl) = k.storage_and_layout();
    let (vs, _vl) = v.storage_and_layout();
    let q_slice = slice_of(&qs)?;
    let k_slice = slice_of(&ks)?;
    let v_slice = slice_of(&vs)?;

    // Output allocated through the candle device's own allocator, launched,
    // and only then moved into a Tensor (after `launch` returns, no live
    // references remain; same-stream ordering protects the asynchronous
    // writes against later users).
    let out_len = b * h * lq * hd;
    let out_slice: cudarc::driver::CudaSlice<f32> = unsafe { dev.alloc(out_len)? };
    let (rows, _) = rows_per_block_and_tile();
    let func = dev.get_or_load_custom_func(KERNEL_NAME, MODULE_NAME, PTX)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((lq as u32).div_ceil(rows), (b * h) as u32, 1),
        block_dim: (rows * 32, 1, 1),
        shared_mem_bytes: 0, // the kernel uses only STATIC shared (sK + sV, 32 KiB)
    };
    let causal_i32 = i32::from(causal);
    unsafe {
        func.builder()
            .arg(q_slice)
            .arg(k_slice)
            .arg(v_slice)
            .arg(&out_slice)
            .arg(&(lq as i32))
            .arg(&(lk as i32))
            .arg(&(hd as i32))
            .arg(&causal_i32)
            .launch(cfg)
            .map_err(|e| candle_core::Error::Msg(format!("flash kernel launch: {e:?}")))?;
    }

    let out_storage = <f32 as CudaDType>::wrap_cuda_slice(out_slice, dev.clone());
    Ok(Tensor::from_storage(
        Storage::Cuda(out_storage),
        Shape::from((b, h, lq, hd)),
        BackpropOp::none(),
        true,
    ))
}

/// Dispatch into the fused path when supported; `Ok(None)` means the caller
/// should run the manual chain.
pub fn try_fused(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Option<Tensor>> {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var("KRONOS_NO_FLASH").is_ok()) {
        return Ok(None);
    }
    if !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
        note_fallback("non-contiguous tensor");
        return Ok(None);
    }
    if q.dtype() != DType::F32 || !matches!(q.device(), Device::Cuda(_)) {
        note_fallback("dtype/device");
        return Ok(None);
    }
    let hd = q.dims4().map(|(_, _, _, d)| d).unwrap_or(0);
    let vhd = v.dims4().map(|(_, _, _, d)| d).unwrap_or(0);
    let hd_ok = hd > 0 && hd <= TILE_W; // all model head dims (4, 52, 64)
    if !hd_ok || vhd != hd {
        note_fallback(&format!("hd={hd} vhd={vhd}"));
        return Ok(None);
    }
    match fused(q, k, v, causal) {
        Ok(t) => Ok(Some(t)),
        Err(e) => {
            note_fallback(&format!("kernel err: {e}"));
            Ok(None)
        }
    }
}

/// One-shot diagnostic (first fallback reason only) when KRONOS_FLASH_MISSING
/// is set: prints why the fused path declined a shape/errors.
fn note_fallback(why: &str) {
    if std::env::var("KRONOS_FLASH_MISSING").is_ok() {
        eprintln!("[flash] fallback: {why}");
    }
}
