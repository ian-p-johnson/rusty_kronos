// Single-pass (online-softmax) attention for Kronos shapes.
//
// Semantics (must match torch F.scaled_dot_product_attention):
//   scores[b,h,i,j] = dot(q[b,h,i,:], k[b,h,j,:]) * rsqrt(hd); (j>i) -> -inf if causal
//   out[b,h,i,:]    = softmax_j(scores[b,h,i,:]) @ v[b,h,:,:]
//
// Layout: one warp per query row; ROWS warps per block share the K/V tiles
// (so the global K/V traffic is re-read lq/ROWS times, not once per row).
// Each lane owns dims (lane, lane+32) of its row; scores are never
// materialised: K/V tiles live in shared memory, probs are consumed online.
// This is the same algorithm family as cutlass "memory-efficient attention"
// (which is what PyTorch dispatches to at fp32 on this GPU) — flash
// attention is the fp16 sibling. Regenerating PTX: rust/cuda/build.sh
// (nvcc 13.1, one invocation, no templates — lq/lk/hd/causal are runtime
// args, tile width is fixed 64).

#define TW 64          // tile width and max tile keys (models use hd 52 and 64)

extern "C" __global__ void flash_causal(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        float* __restrict__ O,
                                        int lq, int lk, int hd, int causal) {
    const int rows = (blockDim.x >> 5); // q rows per block (runtime)
    const int bk = rows;                // square shared tile: keys per tile == rows
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = blockIdx.x * rows + warp; // query row
    const size_t bh = blockIdx.y;           // b*H
    const size_t qoff = bh * (size_t)lq * hd;
    const size_t koff = bh * (size_t)lk * hd;

    const int d0 = lane, d1 = lane + 32;
    float q0 = 0.f, q1 = 0.f;
    if (g < lq && d0 < hd) q0 = Q[qoff + (size_t)g * hd + d0];
    if (g < lq && d1 < hd) q1 = Q[qoff + (size_t)g * hd + d1];

    const float scale = rsqrtf((float)hd);
    float acc0 = 0.f, acc1 = 0.f;
    float m = -INFINITY, l = 0.f;

    __shared__ float sK[TW][TW]; // bk rows x TW; host bounds bk <= TW
    __shared__ float sV[TW][TW];

    const int t = threadIdx.x;
    const unsigned full = 0xffffffffu;
    // causal: tiles entirely past the diagonal (all keys j > g) can be skipped
    int kb_end = lk;
    if (causal && g < lq) {
        kb_end = ((g / bk) + 1) * bk;
        if (kb_end > lk) kb_end = lk;
    }
    for (int kb = 0; kb < kb_end; kb += bk) {
        const int nkey = (lk - kb) < bk ? (lk - kb) : bk;
        // cooperative tile load, zero-fill outside (lk, hd)
        for (int e = t; e < bk * TW; e += (int)blockDim.x) {
            const int j = e / TW, d = e % TW;
            const int gj = kb + j;
            if (gj < lk && d < hd) {
                sK[j][d] = K[koff + (size_t)gj * hd + d];
                sV[j][d] = V[koff + (size_t)gj * hd + d];
            } else {
                sK[j][d] = 0.f; // keeps q*0 == 0 for masked/disabled dims
                sV[j][d] = 0.f;
            }
        }
        __syncthreads();

        if (g < lq) {
            for (int j = 0; j < nkey; j++) {
                const int gj = kb + j;
                float p = q0 * sK[j][d0] + q1 * sK[j][d1];
                p += __shfl_xor_sync(full, p, 16);
                p += __shfl_xor_sync(full, p, 8);
                p += __shfl_xor_sync(full, p, 4);
                p += __shfl_xor_sync(full, p, 2);
                p += __shfl_xor_sync(full, p, 1);
                p *= scale;
                if ((causal && gj > g) || gj >= lk) p = -INFINITY;
                const float mn = fmaxf(m, p);
                // exp visibility: every lane runs the same values, so compute
                // both exponentials once on the warp leader and broadcast
                // (32x fewer SFU calls than computing them per lane).
                float e = 0.f, r = 1.f;
                if (lane == 0) { e = expf(p - mn); r = expf(m - mn); }
                e = __shfl_sync(full, e, 0);
                r = __shfl_sync(full, r, 0);
                acc0 = acc0 * r + e * sV[j][d0];
                acc1 = acc1 * r + e * sV[j][d1];
                l = l * r + e;
                m = mn;
            }
        }
        __syncthreads(); // tiles reused next iteration
    }

    if (g < lq && d0 < hd) O[qoff + (size_t)g * hd + d0] = acc0 / l;
    if (g < lq && d1 < hd) O[qoff + (size_t)g * hd + d1] = acc1 / l;
}
