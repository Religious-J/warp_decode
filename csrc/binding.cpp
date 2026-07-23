// Torch bindings for warp_decode kernels.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

namespace warp_decode {

void launch_moe_gate_up_3d_batched(
    const __nv_bfloat16* X,
    const __nv_bfloat16* W,
    const int*           topk_ids,
    __nv_bfloat16*       out,
    int B, int E, int topk, int hidden, int inter,
    cudaStream_t stream);

void launch_moe_down_3d_batched(
    const __nv_bfloat16* buffer,
    const __nv_bfloat16* W,
    const int*           topk_ids,
    const float*         topk_scale,
    __nv_bfloat16*       out,
    int B, int E, int topk, int hidden, int inter,
    cudaStream_t stream);

}  // namespace warp_decode

namespace {

#define CHECK_CUDA(x)     TORCH_CHECK((x).is_cuda(),           #x " must be CUDA")
#define CHECK_CONTIG(x)   TORCH_CHECK((x).is_contiguous(),     #x " must be contiguous")
#define CHECK_BF16(x)     TORCH_CHECK((x).scalar_type() == at::kBFloat16, #x " must be bf16")
#define CHECK_F32(x)      TORCH_CHECK((x).scalar_type() == at::kFloat,    #x " must be fp32")
#define CHECK_I32(x)      TORCH_CHECK((x).scalar_type() == at::kInt,      #x " must be int32")

torch::Tensor moe_gate_up_3d_batched(
    torch::Tensor x,           // [B, hidden]           bf16
    torch::Tensor w,           // [E, 2*inter, hidden]  bf16
    torch::Tensor topk_ids)    // [B, topk]             int32
{
    CHECK_CUDA(x);     CHECK_CONTIG(x);     CHECK_BF16(x);
    CHECK_CUDA(w);     CHECK_CONTIG(w);     CHECK_BF16(w);
    CHECK_CUDA(topk_ids); CHECK_CONTIG(topk_ids); CHECK_I32(topk_ids);

    TORCH_CHECK(x.dim() == 2 && w.dim() == 3 && topk_ids.dim() == 2);

    const int B      = x.size(0);
    const int hidden = x.size(1);
    const int E      = w.size(0);
    const int two_i  = w.size(1);
    TORCH_CHECK(two_i % 2 == 0, "w.size(1) must be 2 * inter");
    const int inter  = two_i / 2;
    TORCH_CHECK(w.size(2) == hidden, "w.size(2) must == hidden");
    TORCH_CHECK(topk_ids.size(0) == B);
    const int topk   = topk_ids.size(1);

    TORCH_CHECK(hidden % 256 == 0, "hidden must be a multiple of 256 (32 lanes * 8 bf16)");
    TORCH_CHECK(inter  %   8 == 0, "inter must be a multiple of 8 warps per CTA");

    auto out = torch::empty({B * topk, inter},
                            x.options().dtype(at::kBFloat16));

    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_moe_gate_up_3d_batched(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        topk_ids.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        B, E, topk, hidden, inter,
        stream.stream());

    return out;
}

}  // anonymous namespace

namespace {

torch::Tensor moe_down_3d_batched(
    torch::Tensor buffer,       // [B*topk, inter]        bf16
    torch::Tensor w,            // [E, hidden, inter]     bf16
    torch::Tensor topk_ids,     // [B, topk]              int32
    torch::Tensor topk_scale,   // [B, topk]              fp32
    int64_t       B_arg)        // batch size (buffer is [B*topk, inter])
{
    CHECK_CUDA(buffer);     CHECK_CONTIG(buffer);     CHECK_BF16(buffer);
    CHECK_CUDA(w);          CHECK_CONTIG(w);          CHECK_BF16(w);
    CHECK_CUDA(topk_ids);   CHECK_CONTIG(topk_ids);   CHECK_I32(topk_ids);
    CHECK_CUDA(topk_scale); CHECK_CONTIG(topk_scale); CHECK_F32(topk_scale);

    TORCH_CHECK(buffer.dim() == 2 && w.dim() == 3
                && topk_ids.dim() == 2 && topk_scale.dim() == 2);

    const int B      = static_cast<int>(B_arg);
    const int inter  = buffer.size(1);
    const int E      = w.size(0);
    const int hidden = w.size(1);
    TORCH_CHECK(w.size(2) == inter, "w.size(2) must == inter");
    TORCH_CHECK(topk_ids.size(0) == B);
    const int topk   = topk_ids.size(1);
    TORCH_CHECK(topk_scale.size(0) == B && topk_scale.size(1) == topk);
    TORCH_CHECK(buffer.size(0) == B * topk, "buffer rows must be B*topk");

    TORCH_CHECK(inter  % 64 == 0, "inter must be a multiple of 64 (bf16x2 vec, 32 lanes)");
    TORCH_CHECK(hidden %  8 == 0, "hidden must be a multiple of OUTS_PER_WARP");

    auto out = torch::empty({B, hidden}, buffer.options().dtype(at::kBFloat16));

    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_moe_down_3d_batched(
        reinterpret_cast<const __nv_bfloat16*>(buffer.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        topk_ids.data_ptr<int>(),
        topk_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        B, E, topk, hidden, inter,
        stream.stream());

    return out;
}

}  // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_gate_up_3d_batched", &moe_gate_up_3d_batched,
          "Warp-decode BF16 gate_up + silu_and_mul (fused)");
    m.def("moe_down_3d_batched", &moe_down_3d_batched,
          "Warp-decode BF16 down projection + topk-weighted sum");
}
