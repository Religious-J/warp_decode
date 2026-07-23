// warp_decode: gate_up kernel for BF16 MoE decode.
//
// Layout (matches sglang fused_moe / bench_bf16_fuse_moe.py):
//   X   : [B, hidden]                  bf16
//   W   : [E, 2 * inter, hidden]       bf16       (rows [0, inter) = gate == W1,
//                                                  rows [inter, 2*inter) = up  == W3)
//   ids : [B, topk]                    int32
//   out : [B * topk, inter]            bf16       (== silu(gate) * up, fused)
//
// Parallelism (warp-per-neuron):
//   gridDim.x = B * topk                 (one CTA per (token, topk_slot))
//   gridDim.y = inter / WARPS_PER_CTA    (each CTA covers WARPS_PER_CTA neurons)
//   blockDim.x = WARPS_PER_CTA * 32
//   Each warp owns one output neuron n \in [0, inter), computes
//       g = sum_k x[token, k] * W[expert, n,        k]
//       u = sum_k x[token, k] * W[expert, n+inter,  k]
//       ou2u(g) * u
//
// Notes:
//   * All 8 warps in a CTA share the same x[token] row -> stage it into smem once.
//   * hidden must be a multiple of 32 * VEC = 256 (VEC = 8 bf16 per uint4 load).
//   * inter must be a multiple of WARPS_PER_CTA (=8 by default).


#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace warp_decode {

constexpr int WARPS_PER_CTA   = 8;
constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;
constexpr int VEC             = 8;   // 8 bf16 == 16 B == uint4

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // full warp reduction via xor butterfly
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__global__ void moe_gate_up_3d_batched_kernel(
    const __nv_bfloat16* __restrict__ X,        // [B, hidden]
    const __nv_bfloat16* __restrict__ W,        // [E, 2*inter, hidden]
    const int*           __restrict__ topk_ids, // [B, topk]
    __nv_bfloat16*       __restrict__ out,      // [B*topk, inter]
    int B, int E, int topk, int hidden, int inter)
{
    const int flat_tok = blockIdx.x;                 // in [0, B*topk)
    const int token_id = flat_tok / topk;
    const int slot     = flat_tok - token_id * topk;

    const int warp_id  = threadIdx.x >> 5;
    const int lane_id  = threadIdx.x & 31;

    const int neuron_n = blockIdx.y * WARPS_PER_CTA + warp_id;
    // Boundary guard for inter % WARPS_PER_CTA != 0 (currently we assert host-side).
    if (neuron_n >= inter) return;

    const int expert_id = topk_ids[token_id * topk + slot];

    // ---- 1. Stage x[token] into shared memory (shared across all 8 warps) ----
    extern __shared__ __nv_bfloat16 smem_x[];  // hidden bf16 elements
    const __nv_bfloat16* __restrict__ x_ptr = X + (size_t)token_id * hidden;

    #pragma unroll 1
    for (int i = threadIdx.x * VEC; i < hidden; i += THREADS_PER_CTA * VEC) {
        *reinterpret_cast<uint4*>(&smem_x[i]) =
            *reinterpret_cast<const uint4*>(&x_ptr[i]);
    }
    __syncthreads();

    // ---- 2. Warp streams its two weight rows, dot-products with x -----------
    const size_t expert_stride = (size_t)(2 * inter) * hidden;
    const __nv_bfloat16* __restrict__ gate_ptr =
        W + (size_t)expert_id * expert_stride + (size_t)neuron_n * hidden;
    const __nv_bfloat16* __restrict__ up_ptr   =
        gate_ptr + (size_t)inter * hidden;

    float acc_gate = 0.f;
    float acc_up   = 0.f;

    #pragma unroll 1
    for (int k = lane_id * VEC; k < hidden; k += 32 * VEC) {
        uint4 xv = *reinterpret_cast<const uint4*>(&smem_x[k]);
        uint4 gv = *reinterpret_cast<const uint4*>(&gate_ptr[k]);
        uint4 uv = *reinterpret_cast<const uint4*>(&up_ptr[k]);

        const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xv);
        const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(&gv);
        const __nv_bfloat16* uh = reinterpret_cast<const __nv_bfloat16*>(&uv);

        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            float xf = __bfloat162float(xh[i]);
            acc_gate += xf * __bfloat162float(gh[i]);
            acc_up   += xf * __bfloat162float(uh[i]);
        }
    }

    acc_gate = warp_reduce_sum(acc_gate);
    acc_up   = warp_reduce_sum(acc_up);

    if (lane_id == 0) {
        float y = silu_f32(acc_gate) * acc_up;
        out[(size_t)flat_tok * inter + neuron_n] = __float2bfloat16(y);
    }
}

void launch_moe_gate_up_3d_batched(
    const __nv_bfloat16* X,
    const __nv_bfloat16* W,
    const int*           topk_ids,
    __nv_bfloat16*       out,
    int B, int E, int topk, int hidden, int inter,
    cudaStream_t stream)
{
    dim3 grid(B * topk, inter / WARPS_PER_CTA);
    dim3 block(THREADS_PER_CTA);
    size_t smem_bytes = (size_t)hidden * sizeof(__nv_bfloat16);

    moe_gate_up_3d_batched_kernel<<<grid, block, smem_bytes, stream>>>(
        X, W, topk_ids, out, B, E, topk, hidden, inter);
}

}  // namespace warp_decode
