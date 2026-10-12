#pragma once

#include <cuda_runtime.h>

namespace warp_decode {

void launch_quantize_fp8_blockwise(
    const void* x,
    void*       out,
    float*      x_scale,
    int rows, int cols,
    cudaStream_t stream);

void launch_moe_gate_up_3d_batched(
    const void*  X,
    const void*  W,
    const float* x_scale,
    const float* w_scale,
    const int*   topk_ids,
    void*        out,
    int B,
    cudaStream_t stream);

void launch_moe_down_3d_batched(
    const void*  buffer,
    const void*  W,
    const float* buffer_scale,
    const float* w_scale,
    const int*   topk_ids,
    const float* topk_scale,
    void*        out,
    int B,
    cudaStream_t stream);

}  // namespace warp_decode
