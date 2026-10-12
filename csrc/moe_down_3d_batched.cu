// warp_decode: down kernel for blockwise FP8 MoE decode.
//
// Layout (Qwen3.6-35B-A3B, TP=1):
//   buffer       : [B*topk, inter]             fp8 e4m3
//   W_down       : [E, hidden, inter]          fp8 e4m3
//   buffer_scale : [B*topk, inter / 128]       fp32
//   w_scale      : [E, hidden / 128, inter / 128] fp32
//   topk_ids     : [B, topk]                   int32
//   topk_scale   : [B, topk]                   fp32
//   out          : [B, hidden]                bf16
//
// Parallelism (warp-per-output-group, no shared memory):
//   Each warp owns four adjacent output dimensions for one token.
//   Input values are reused across independent FP32 dot products. Block scales
//   and routing weights are applied in FP32 before one final BF16 conversion.

#include "moe_3d_batched.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace warp_decode {
namespace {

constexpr int HIDDEN_SIZE          = 2048;
constexpr int INTER_SIZE           = 512;
constexpr int NUM_EXPERTS          = 256;
constexpr int TOPK                 = 8;
constexpr int SCALE_GROUP          = 128;
constexpr int WARP_SIZE_DOWN       = 32;
constexpr int WARPS_PER_CTA_DOWN   = 4;
constexpr int THREADS_PER_CTA_DOWN = WARPS_PER_CTA_DOWN * WARP_SIZE_DOWN;
constexpr int OUTS_PER_WARP        = 4;

using fp8 = __nv_fp8_e4m3;

__device__ __forceinline__ float warp_reduce_sum_d(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__device__ __forceinline__ float2 unpack_fp8_pair_d(uint16_t packed) {
    return __half22float2(static_cast<__half2>(
        __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3)));
}

template<int OUTS>
__global__ void moe_down_3d_batched_kernel(
    const fp8*     buffer,       // [B*topk, inter]
    const fp8*     W,            // [E, hidden, inter]
    const float*   buffer_scale, // [B*topk, inter / SCALE_GROUP]
    const float*   w_scale,      // [E, hidden / SCALE_GROUP, inter / SCALE_GROUP]
    const int*     topk_ids,     // [B, topk]
    const float*   topk_scale,   // [B, topk]
    __nv_bfloat16* out,          // [B, hidden]
    int B)
{
    const size_t global_tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t global_warp = global_tid / WARP_SIZE_DOWN;
    const int lane_id = threadIdx.x & (WARP_SIZE_DOWN - 1);
    const int groups_per_token = HIDDEN_SIZE / OUTS;
    const size_t total_warps = (size_t)B * groups_per_token;
    if (global_warp >= total_warps) return;

    const int token_id  = (int)(global_warp / groups_per_token);
    const int group_idx = (int)(global_warp - (size_t)token_id * groups_per_token);
    const int h_base    = group_idx * OUTS;

    float acc[OUTS]{};
    #pragma unroll 1
    for (int s = 0; s < TOPK; ++s) {
        const int flat_tok  = token_id * TOPK + s;
        const int expert_id = topk_ids[flat_tok];
        if (expert_id < 0 || expert_id >= NUM_EXPERTS) {
            if (lane_id == 0) {
                #pragma unroll
                for (int j = 0; j < OUTS; ++j) {
                    out[(size_t)token_id * HIDDEN_SIZE + h_base + j] = __float2bfloat16_rn(nanf(""));
                }
            }
            return;
        }

        const fp8* in_ptr = buffer + (size_t)flat_tok * INTER_SIZE;
        const fp8* w_base = W + ((size_t)expert_id * HIDDEN_SIZE + h_base) * INTER_SIZE;
        const float* scale_ptr = w_scale +
            ((size_t)expert_id * (HIDDEN_SIZE / SCALE_GROUP) + h_base / SCALE_GROUP) *
            (INTER_SIZE / SCALE_GROUP);
        float local[OUTS]{};

        // Four FP8 elements per lane cover one scale group per iteration.
        #pragma unroll
        for (int k = lane_id * 4; k < INTER_SIZE; k += SCALE_GROUP) {
            uint32_t xv = *reinterpret_cast<const uint32_t*>(&in_ptr[k]);
            const float2 x_lo = unpack_fp8_pair_d(uint16_t(xv));
            const float2 x_hi = unpack_fp8_pair_d(uint16_t(xv >> 16));
            const float scale =
                buffer_scale[(size_t)flat_tok * (INTER_SIZE / SCALE_GROUP) + k / SCALE_GROUP] *
                scale_ptr[k / SCALE_GROUP];

            #pragma unroll
            for (int j = 0; j < OUTS; ++j) {
                uint32_t wv = *reinterpret_cast<const uint32_t*>(&w_base[j * INTER_SIZE + k]);
                const float2 w_lo = unpack_fp8_pair_d(uint16_t(wv));
                const float2 w_hi = unpack_fp8_pair_d(uint16_t(wv >> 16));
                float dot = x_lo.x * w_lo.x;
                dot = fmaf(x_lo.y, w_lo.y, dot);
                dot = fmaf(x_hi.x, w_hi.x, dot);
                dot = fmaf(x_hi.y, w_hi.y, dot);
                local[j] = fmaf(dot, scale, local[j]);
            }
        }

        #pragma unroll
        for (int j = 0; j < OUTS; ++j) {
            acc[j] = fmaf(local[j], topk_scale[flat_tok], acc[j]);
        }
    }

    #pragma unroll
    for (int j = 0; j < OUTS; ++j) acc[j] = warp_reduce_sum_d(acc[j]);

    if (lane_id == 0) {
        __nv_bfloat16* out_ptr = out + (size_t)token_id * HIDDEN_SIZE + h_base;
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) {
            out_ptr[j] = __float2bfloat16_rn(acc[j]);
        }
    }
}

}  // anonymous namespace

void launch_moe_down_3d_batched(
    const void*  buffer,
    const void*  W,
    const float* buffer_scale,
    const float* w_scale,
    const int*   topk_ids,
    const float* topk_scale,
    void*        out,
    int B,
    cudaStream_t stream)
{
    if (!B) return;
    const size_t total_warps = (size_t)B * (HIDDEN_SIZE / OUTS_PER_WARP);
    const size_t ctas = (total_warps + WARPS_PER_CTA_DOWN - 1) / WARPS_PER_CTA_DOWN;
    dim3 grid(ctas);
    dim3 block(THREADS_PER_CTA_DOWN);
    moe_down_3d_batched_kernel<OUTS_PER_WARP><<<grid, block, 0, stream>>>(
        static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
        topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out), B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace warp_decode
