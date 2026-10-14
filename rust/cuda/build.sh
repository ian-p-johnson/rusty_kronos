#!/bin/bash
# Regenerates crates/kronos-core/src/flash_attn.ptx from flash_attn.cu.
#
# PTX (not cubin) so cudarc JITs to whatever device is present; virtual arch
# compute_80 JITs cleanly onto anything newer (this box is sm_120). No NVRTC
# or nvcc needed at Rust build time — the repo carries the compiled PTX.
set -euo pipefail
cd "$(dirname "$0")"
nvcc --ptx \
    -arch=compute_80 -code=compute_80 \
    -lineinfo \
    -o ../crates/kronos-core/src/flash_attn.ptx flash_attn.cu
echo "wrote crates/kronos-core/src/flash_attn.ptx ($(wc -c < ../crates/kronos-core/src/flash_attn.ptx) bytes)"
