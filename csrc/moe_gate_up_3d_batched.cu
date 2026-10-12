// warp_decode: gate_up kernel for blockwise FP8 MoE decode.
//
// Layout (Qwen3.6-35B-A3B, TP=1):
//   X       : [B, hidden]                  fp8 e4m3
//   W       : [E, 2 * inter, hidden]       fp8 e4m3 (gate rows, then up rows)
//   x_scale : [B, hidden / 128]            fp32
//   w_scale : [E, 2 * inter / 128, hidden / 128] fp32
//   ids     : [B, topk]                    int32
//   out     : [B * topk, inter]            bf16 (silu(gate) * up, fused)
//
// Parallelism (two neurons per warp):
//   gridDim.x = B * topk; gridDim.y = inter / (WARPS_PER_CTA * OUTS_PER_WARP).
//   Each warp reuses one decoded input vector across two gate/up weight rows.
//   FP8 pairs are widened before FP32 accumulation; each vector stays within
//   one 128-element scale group. Gate/up rounds to BF16 before activation.

#include "moe_3d_batched.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace warp_decode {
namespace {

constexpr int HIDDEN_SIZE     = 2048;
constexpr int INTER_SIZE      = 512;
constexpr int NUM_EXPERTS     = 256;
constexpr int TOPK            = 8;
constexpr int SCALE_GROUP     = 128;
constexpr int WARPS_PER_CTA   = 4;
constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;
constexpr int VEC             = 16;  // 16 FP8 values == 16 B == uint4
constexpr int OUTS_PER_WARP   = 2;

static_assert(SCALE_GROUP % OUTS_PER_WARP == 0);
static_assert(INTER_SIZE % (WARPS_PER_CTA * OUTS_PER_WARP) == 0);

using fp8 = __nv_fp8_e4m3;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // full warp reduction via xor butterfly
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int d = 16; d; d /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, d));
    }
    return v;
}

__device__ __forceinline__ float2 unpack_fp8_pair(uint16_t packed) {
    return __half22float2(static_cast<__half2>(
        __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3)));
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void quantize_fp8_blockwise_kernel(
    const __nv_bfloat16* X,    // [rows, cols]
    fp8*                out,  // [rows, cols]
    float*              scale, // [rows, cols / SCALE_GROUP]
    int rows, int cols)
{
    const int lane_id = threadIdx.x & 31;
    const size_t group = (size_t)blockIdx.x * WARPS_PER_CTA + threadIdx.x / 32;
    if (group >= (size_t)rows * (cols / SCALE_GROUP)) return;

    // Each warp quantizes one group; four adjacent BF16 elements per lane.
    const size_t offset = group * SCALE_GROUP + lane_id * 4;
    uint2 packed = *reinterpret_cast<const uint2*>(X + offset);
    const auto* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
    float xf[4];
    float max_abs = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        xf[i] = __bfloat162float(values[i]);
        max_abs = fmaxf(max_abs, fabsf(xf[i]));
    }

    const float group_scale = __fdiv_rn(fmaxf(warp_reduce_max(max_abs), 1.e-10f), 448.f);
    if (lane_id == 0) scale[group] = group_scale;
    const float inv_scale = __fdiv_rn(1.f, group_scale);
    __nv_fp8x2_e4m3 lo(make_float2(xf[0] * inv_scale, xf[1] * inv_scale));
    __nv_fp8x2_e4m3 hi(make_float2(xf[2] * inv_scale, xf[3] * inv_scale));
    *reinterpret_cast<uint32_t*>(out + offset) = uint32_t(lo.__x) | (uint32_t(hi.__x) << 16);
}

__global__ void moe_gate_up_3d_batched_kernel(
    const fp8*     X,        // [B, hidden]
    const fp8*     W,        // [E, 2*inter, hidden]
    const float*   x_scale,  // [B, hidden / SCALE_GROUP]
    const float*   w_scale,  // [E, 2*inter / SCALE_GROUP, hidden / SCALE_GROUP]
    const int*     topk_ids, // [B, topk]
    __nv_bfloat16* out)      // [B*topk, inter]
{
    const int flat_tok = blockIdx.x;
    const int token_id = flat_tok / TOPK;
    const int warp_id  = threadIdx.x >> 5;
    const int lane_id  = threadIdx.x & 31;
    const int neuron_n = (blockIdx.y * WARPS_PER_CTA + warp_id) * OUTS_PER_WARP;
    const int expert_id = topk_ids[flat_tok];

    if (expert_id < 0 || expert_id >= NUM_EXPERTS) {
        if (lane_id == 0) {
            #pragma unroll
            for (int j = 0; j < OUTS_PER_WARP; ++j) {
                out[(size_t)flat_tok * INTER_SIZE + neuron_n + j] = __float2bfloat16_rn(nanf(""));
            }
        }
        return;
    }

    const fp8* x_ptr = X + (size_t)token_id * HIDDEN_SIZE;
    const size_t expert_stride = (size_t)(2 * INTER_SIZE) * HIDDEN_SIZE;
    const fp8* gate_ptr = W + (size_t)expert_id * expert_stride + (size_t)neuron_n * HIDDEN_SIZE;
    const fp8* up_ptr   = gate_ptr + (size_t)INTER_SIZE * HIDDEN_SIZE;
    const float* gate_scale = w_scale +
        ((size_t)expert_id * (2 * INTER_SIZE / SCALE_GROUP) + neuron_n / SCALE_GROUP) *
        (HIDDEN_SIZE / SCALE_GROUP);
    const float* up_scale = gate_scale + (INTER_SIZE / SCALE_GROUP) * (HIDDEN_SIZE / SCALE_GROUP);

    float acc_gate[OUTS_PER_WARP]{};
    float acc_up[OUTS_PER_WARP]{};

    #pragma unroll
    for (int k = lane_id * VEC; k < HIDDEN_SIZE; k += 32 * VEC) {
        uint4 xv = *reinterpret_cast<const uint4*>(&x_ptr[k]);
        const auto* x_pairs = reinterpret_cast<const uint16_t*>(&xv);
        float2 xf[VEC / 2];
        #pragma unroll
        for (int i = 0; i < VEC / 2; ++i) xf[i] = unpack_fp8_pair(x_pairs[i]);
        const float scale = x_scale[(size_t)token_id * (HIDDEN_SIZE / SCALE_GROUP) + k / SCALE_GROUP];
        const float gate_factor = scale * gate_scale[k / SCALE_GROUP];
        const float up_factor   = scale * up_scale[k / SCALE_GROUP];

        #pragma unroll
        for (int j = 0; j < OUTS_PER_WARP; ++j) {
            uint4 gv = *reinterpret_cast<const uint4*>(&gate_ptr[j * HIDDEN_SIZE + k]);
            uint4 uv = *reinterpret_cast<const uint4*>(&up_ptr[j * HIDDEN_SIZE + k]);
            const auto* g_pairs = reinterpret_cast<const uint16_t*>(&gv);
            const auto* u_pairs = reinterpret_cast<const uint16_t*>(&uv);
            float local_gate = 0.f;
            float local_up   = 0.f;

            #pragma unroll
            for (int i = 0; i < VEC / 2; ++i) {
                const float2 gf = unpack_fp8_pair(g_pairs[i]);
                const float2 uf = unpack_fp8_pair(u_pairs[i]);
                local_gate = fmaf(xf[i].x, gf.x, local_gate);
                local_gate = fmaf(xf[i].y, gf.y, local_gate);
                local_up   = fmaf(xf[i].x, uf.x, local_up);
                local_up   = fmaf(xf[i].y, uf.y, local_up);
            }
            acc_gate[j] = fmaf(local_gate, gate_factor, acc_gate[j]);
            acc_up[j]   = fmaf(local_up,   up_factor,   acc_up[j]);
        }
    }

    #pragma unroll
    for (int j = 0; j < OUTS_PER_WARP; ++j) {
        acc_gate[j] = warp_reduce_sum(acc_gate[j]);
        acc_up[j]   = warp_reduce_sum(acc_up[j]);
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < OUTS_PER_WARP; ++j) {
            // Match the BF16 GEMM boundary before SGLang's SiLU-and-multiply.
            const float gate = __bfloat162float(__float2bfloat16_rn(acc_gate[j]));
            const float up   = __bfloat162float(__float2bfloat16_rn(acc_up[j]));
            const float y = silu_f32(gate) * up;
            out[(size_t)flat_tok * INTER_SIZE + neuron_n + j] = __float2bfloat16_rn(y);
        }
    }
}

}  // anonymous namespace

void launch_quantize_fp8_blockwise(
    const void* X,
    void*       out,
    float*      scale,
    int rows, int cols,
    cudaStream_t stream)
{
    if (!rows || !cols) return;
    const size_t total_groups = (size_t)rows * (cols / SCALE_GROUP);
    dim3 grid((total_groups + WARPS_PER_CTA - 1) / WARPS_PER_CTA);
    dim3 block(THREADS_PER_CTA);
    quantize_fp8_blockwise_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(X), static_cast<fp8*>(out), scale, rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_moe_gate_up_3d_batched(
    const void*  X,
    const void*  W,
    const float* x_scale,
    const float* w_scale,
    const int*   topk_ids,
    void*        out,
    int B,
    cudaStream_t stream)
{
    if (!B) return;
    dim3 grid(B * TOPK, INTER_SIZE / (WARPS_PER_CTA * OUTS_PER_WARP));
    dim3 block(THREADS_PER_CTA);
    moe_gate_up_3d_batched_kernel<<<grid, block, 0, stream>>>(
        static_cast<const fp8*>(X), static_cast<const fp8*>(W), x_scale, w_scale,
        topk_ids, static_cast<__nv_bfloat16*>(out));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace warp_decode
