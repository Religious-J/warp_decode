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
// Each lane loads 16 adjacent FP8 values and reuses the decoded input across
// four outputs. B <= 2 distributes the eight experts across eight warps;
// larger batches process experts in one warp. All reductions use FP32.

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
constexpr int OUTS_PER_WARP        = 4;
constexpr int VEC_DOWN             = 16;

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

template<bool SPLIT_TOPK>
__global__ void moe_down_3d_batched_kernel(
    const fp8*     buffer,
    const fp8*     W,
    const float*   buffer_scale,
    const float*   w_scale,
    const int*     topk_ids,
    const float*   topk_scale,
    __nv_bfloat16* out,
    int B)
{
    const int warp_id = threadIdx.x / WARP_SIZE_DOWN;
    const int lane_id = threadIdx.x & (WARP_SIZE_DOWN - 1);
    const int token_id = blockIdx.x % B;
    constexpr int GROUPS_PER_CTA = SPLIT_TOPK ? 1 : WARPS_PER_CTA_DOWN;
    const int group_idx = (blockIdx.x / B) * GROUPS_PER_CTA +
                          (SPLIT_TOPK ? 0 : warp_id);
    const int h_base = group_idx * OUTS_PER_WARP;
    const int k = lane_id * VEC_DOWN;
    const int scale_group = k / SCALE_GROUP;
    float acc[OUTS_PER_WARP]{};

    #pragma unroll 1
    for (int i = 0; i < (SPLIT_TOPK ? 1 : TOPK); ++i) {
        const int s = SPLIT_TOPK ? warp_id : i;
        const int flat_tok = token_id * TOPK + s;
        const int expert_id = topk_ids[flat_tok];
        if (expert_id < 0 || expert_id >= NUM_EXPERTS) {
            #pragma unroll
            for (int j = 0; j < OUTS_PER_WARP; ++j) acc[j] = nanf("");
            break;
        }

        const fp8* x_ptr = buffer + (size_t)flat_tok * INTER_SIZE + k;
        const fp8* w_ptr = W + ((size_t)expert_id * HIDDEN_SIZE + h_base) * INTER_SIZE + k;
        const uint4 xv = *reinterpret_cast<const uint4*>(x_ptr);
        const uint32_t xpacked[4] = {xv.x, xv.y, xv.z, xv.w};
        float x[VEC_DOWN];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 lo = unpack_fp8_pair_d(uint16_t(xpacked[p]));
            const float2 hi = unpack_fp8_pair_d(uint16_t(xpacked[p] >> 16));
            x[4 * p] = lo.x;
            x[4 * p + 1] = lo.y;
            x[4 * p + 2] = hi.x;
            x[4 * p + 3] = hi.y;
        }
        const size_t weight_scale_idx =
            ((size_t)expert_id * (HIDDEN_SIZE / SCALE_GROUP) + h_base / SCALE_GROUP) *
            (INTER_SIZE / SCALE_GROUP) + scale_group;
        const float scale =
            buffer_scale[(size_t)flat_tok * (INTER_SIZE / SCALE_GROUP) + scale_group] *
            w_scale[weight_scale_idx];

        #pragma unroll
        for (int j = 0; j < OUTS_PER_WARP; ++j) {
            const uint4 wv = *reinterpret_cast<const uint4*>(w_ptr + j * INTER_SIZE);
            const uint32_t wpacked[4] = {wv.x, wv.y, wv.z, wv.w};
            float dot = 0.f;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float2 lo = unpack_fp8_pair_d(uint16_t(wpacked[p]));
                const float2 hi = unpack_fp8_pair_d(uint16_t(wpacked[p] >> 16));
                dot = fmaf(x[4 * p], lo.x, dot);
                dot = fmaf(x[4 * p + 1], lo.y, dot);
                dot = fmaf(x[4 * p + 2], hi.x, dot);
                dot = fmaf(x[4 * p + 3], hi.y, dot);
            }
            const float scaled = dot * scale;
            if constexpr (SPLIT_TOPK) {
                acc[j] = scaled;
            } else {
                acc[j] = fmaf(scaled, topk_scale[flat_tok], acc[j]);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < OUTS_PER_WARP; ++j) acc[j] = warp_reduce_sum_d(acc[j]);

    if constexpr (SPLIT_TOPK) {
        __shared__ float partial[TOPK][OUTS_PER_WARP];
        if (lane_id == 0) {
            #pragma unroll
            for (int j = 0; j < OUTS_PER_WARP; ++j) partial[warp_id][j] = acc[j];
        }
        // Invalid experts also reach this barrier and propagate NaNs.
        __syncthreads();
        if (threadIdx.x < OUTS_PER_WARP) {
            float value = 0.f;
            #pragma unroll
            for (int s = 0; s < TOPK; ++s) {
                value = fmaf(partial[s][threadIdx.x],
                             topk_scale[token_id * TOPK + s], value);
            }
            out[(size_t)token_id * HIDDEN_SIZE + h_base + threadIdx.x] =
                __float2bfloat16_rn(value);
        }
    } else if (lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < OUTS_PER_WARP; ++j) {
            out[(size_t)token_id * HIDDEN_SIZE + h_base + j] = __float2bfloat16_rn(acc[j]);
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
    if (B <= 2) {
        const size_t ctas = (size_t)B * (HIDDEN_SIZE / OUTS_PER_WARP);
        moe_down_3d_batched_kernel<true><<<ctas, TOPK * WARP_SIZE_DOWN, 0, stream>>>(
            static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
            topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out), B);
    } else {
        const size_t ctas = (size_t)B * (HIDDEN_SIZE / OUTS_PER_WARP / WARPS_PER_CTA_DOWN);
        moe_down_3d_batched_kernel<false><<<ctas, WARPS_PER_CTA_DOWN * WARP_SIZE_DOWN, 0, stream>>>(
            static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
            topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out), B);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace warp_decode
