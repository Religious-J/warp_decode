// warp_decode: down kernel for BF16 MoE decode.
//
// Layout:
//   buffer     : [B, topk, inter]      bf16   (== [B*topk, inter], gate_up output)
//   W_down     : [E, hidden, inter]    bf16
//   topk_ids   : [B, topk]             int32
//   topk_scale : [B, topk]             fp32
//   out        : [B, hidden]           bf16
//
// Parallelism (warp-per-output-group, no shared memory):
//   Each warp owns OUTS_PER_WARP adjacent output dims (h .. h+OUTS_PER_WARP-1)
//   for one token. One input read `xv` feeds OUTS_PER_WARP independent dot
//   products, so input-side arithmetic intensity is OUTS_PER_WARP.
//   buffer[token, :, :] fits in L1/L2 (3 KB @ Qwen3-TP8), so no smem staging.
//
//   1D grid over (token, h-group). blockDim = 256 (8 warps/CTA).
//
// Constraints:
//   * inter  % 64 == 0                    (half_k = inter/2, bf16x2 vec)
//   * hidden % OUTS_PER_WARP == 0

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace warp_decode {

constexpr int WARP_SIZE_DOWN       = 32;
constexpr int WARPS_PER_CTA_DOWN   = 8;
constexpr int THREADS_PER_CTA_DOWN = WARPS_PER_CTA_DOWN * WARP_SIZE_DOWN;

// Number of output dims each warp computes. Higher = better input reuse, more
// register pressure. 8 is the sweet spot on H20 for Qwen3 shapes.
constexpr int OUTS_PER_WARP = 8;

__device__ __forceinline__ float warp_reduce_sum_d(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

template<int OUTS>
__global__ void moe_down_3d_batched_kernel(
    const __nv_bfloat16* __restrict__ buffer,     // [B, topk, inter]
    const __nv_bfloat16* __restrict__ W,          // [E, hidden, inter]
    const int*           __restrict__ topk_ids,   // [B, topk]
    const float*         __restrict__ topk_scale, // [B, topk]
    __nv_bfloat16*       __restrict__ out,        // [B, hidden]
    int B, int E, int topk, int hidden, int inter)
{
    const size_t global_tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t global_warp = global_tid / WARP_SIZE_DOWN;
    const int    lane_id     = threadIdx.x & (WARP_SIZE_DOWN - 1);

    const int    groups_per_token = hidden / OUTS;
    const size_t total_warps      = (size_t)B * groups_per_token;
    if (global_warp >= total_warps) return;

    const int token_id  = (int)(global_warp / groups_per_token);
    const int group_idx = (int)(global_warp - (size_t)token_id * groups_per_token);
    const int h_base    = group_idx * OUTS;

    const int half_k = inter / 2;

    // Accumulators (one per output h in this warp's group).
    float acc[OUTS];
    #pragma unroll
    for (int j = 0; j < OUTS; ++j) acc[j] = 0.f;

    #pragma unroll 1
    for (int s = 0; s < topk; ++s) {
        const int   expert_id = topk_ids  [token_id * topk + s];
        const float scale     = topk_scale[token_id * topk + s];

        const __nv_bfloat162* __restrict__ in_ptr =
            reinterpret_cast<const __nv_bfloat162*>(
                buffer + ((size_t)token_id * topk + s) * inter);

        // Precompute weight-row pointers for the OUTS outputs.
        const __nv_bfloat16* w_base =
            W + (size_t)expert_id * hidden * inter + (size_t)h_base * inter;
        const __nv_bfloat162* w_ptrs[OUTS];
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) {
            w_ptrs[j] = reinterpret_cast<const __nv_bfloat162*>(w_base + j * inter);
        }

        // Per-expert local accumulators (allow the compiler to overlap loads).
        float local[OUTS];
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) local[j] = 0.f;

        // 32 lanes cover half_k in (half_k/32) iters. For Qwen3 half_k=96 -> 3 iters.
        #pragma unroll
        for (int i = lane_id; i < half_k; i += WARP_SIZE_DOWN) {
            __nv_bfloat162 xv = in_ptr[i];
            #pragma unroll
            for (int j = 0; j < OUTS; ++j) {
                __nv_bfloat162 wv = w_ptrs[j][i];
                float2 p = __bfloat1622float2(__hmul2(xv, wv));
                local[j] += p.x + p.y;
            }
        }

        #pragma unroll
        for (int j = 0; j < OUTS; ++j) acc[j] += scale * local[j];
    }

    #pragma unroll
    for (int j = 0; j < OUTS; ++j) acc[j] = warp_reduce_sum_d(acc[j]);

    if (lane_id == 0) {
        __nv_bfloat16* out_ptr = out + (size_t)token_id * hidden + h_base;
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) {
            out_ptr[j] = __float2bfloat16(acc[j]);
        }
    }
}

void launch_moe_down_3d_batched(
    const __nv_bfloat16* buffer,
    const __nv_bfloat16* W,
    const int*           topk_ids,
    const float*         topk_scale,
    __nv_bfloat16*       out,
    int B, int E, int topk, int hidden, int inter,
    cudaStream_t stream)
{
    const size_t total_warps = (size_t)B * (hidden / OUTS_PER_WARP);
    const size_t ctas = (total_warps + WARPS_PER_CTA_DOWN - 1) / WARPS_PER_CTA_DOWN;

    dim3 grid((unsigned)ctas);
    dim3 block(THREADS_PER_CTA_DOWN);

    moe_down_3d_batched_kernel<OUTS_PER_WARP><<<grid, block, 0, stream>>>(
        buffer, W, topk_ids, topk_scale, out, B, E, topk, hidden, inter);
}

}  // namespace warp_decode
