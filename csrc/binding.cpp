// Torch bindings for warp_decode kernels.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <cstdint>
#include "moe_3d_batched.h"

namespace {

constexpr int H = 2048, I = 512, E = 256, K = 8, Q = 128;
using OptionalTensor = std::optional<torch::Tensor>;
using Quantized = std::tuple<torch::Tensor, torch::Tensor>;

void check(
    const torch::Tensor& t,
    at::ScalarType dtype,
    const torch::Device& device,
    const char* name)
{
    TORCH_CHECK(t.is_cuda() && t.device() == device, name, " must be on the same CUDA device");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has an incorrect dtype");
}

void shape(
    const torch::Tensor& t,
    std::initializer_list<int64_t> expected,
    const char* name)
{
    TORCH_CHECK(t.sizes() == at::IntArrayRef(expected), name, " has an incorrect shape; expected ",
                at::IntArrayRef(expected), ", got ", t.sizes());
}

void alignment(const torch::Tensor& t, int bytes, const char* name) {
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % bytes == 0,
                name, " must have a ", bytes, "-byte-aligned data pointer");
}

Quantized quantize_fp8_blockwise(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    check(x, at::kBFloat16, x.device(), "x");
    alignment(x, 8, "x");
    TORCH_CHECK(x.dim() == 2 && x.size(1) % Q == 0,
                "x must be [rows, cols] with cols a multiple of 128");
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() &&
                x.size(1) <= std::numeric_limits<int>::max(), "x is too large");

    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x, x.options().dtype(at::kFloat8_e4m3fn));
    auto x_scale = torch::empty({x.size(0), x.size(1) / Q},
                               x.options().dtype(at::kFloat));

    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_quantize_fp8_blockwise(
        x.data_ptr(),
        out.data_ptr(),
        x_scale.data_ptr<float>(),
        x.size(0), x.size(1),
        stream.stream());

    return {out, x_scale};
}

Quantized prepare_activation(
    torch::Tensor x,
    OptionalTensor x_scale,
    int64_t rows,
    int cols)
{
    shape(x, {rows, cols}, "activation");
    if (x.scalar_type() == at::kBFloat16) {
        TORCH_CHECK(!x_scale.has_value(), "activation scale is only accepted for prequantized FP8 input");
        return quantize_fp8_blockwise(x);
    }
    check(x, at::kFloat8_e4m3fn, x.device(), "activation");
    alignment(x, 16, "FP8 activation");
    TORCH_CHECK(x_scale.has_value(), "prequantized FP8 activation requires its FP32 scale");
    check(*x_scale, at::kFloat, x.device(), "activation scale");
    shape(*x_scale, {rows, cols / Q}, "activation scale");
    return {x, *x_scale};
}

int routing_check(torch::Tensor topk_ids, const torch::Device& device) {
    check(topk_ids, at::kInt, device, "topk_ids");
    TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(1) == K, "topk_ids must be [B,8]");
    TORCH_CHECK(topk_ids.size(0) <= std::numeric_limits<int>::max() / K, "batch is too large");
    return topk_ids.size(0);
}

void weight_check(
    torch::Tensor w,
    torch::Tensor w_scale,
    const torch::Device& device,
    int n,
    int k)
{
    check(w, at::kFloat8_e4m3fn, device, "weight");
    alignment(w, 16, "FP8 weight");
    check(w_scale, at::kFloat, device, "weight scale");
    shape(w, {E, n, k}, "weight (Qwen3.6 TP1 only)");
    shape(w_scale, {E, n / Q, k / Q}, "weight scale");
}

torch::Tensor moe_gate_up_3d_batched(
    torch::Tensor x,                 // [B, hidden]                         bf16 or fp8
    torch::Tensor w,                 // [E, 2*inter, hidden]                 fp8
    torch::Tensor w_scale,           // [E, 2*inter/128, hidden/128]         fp32
    torch::Tensor topk_ids,          // [B, topk]                           int32
    OptionalTensor x_scale = std::nullopt)  // [B, hidden/128]              fp32
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    c10::cuda::CUDAGuard guard(x.device());

    const int B = routing_check(topk_ids, x.device());
    weight_check(w, w_scale, x.device(), 2 * I, H);
    auto [x_fp8, activation_scale] = prepare_activation(x, x_scale, B, H);

    auto out = torch::empty({int64_t(B) * K, I},
                            x.options().dtype(at::kBFloat16));

    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_moe_gate_up_3d_batched(
        x_fp8.data_ptr(),
        w.data_ptr(),
        activation_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        topk_ids.data_ptr<int>(),
        out.data_ptr(),
        B,
        stream.stream());

    return out;
}

torch::Tensor moe_down_3d_batched(
    torch::Tensor buffer,            // [B*topk, inter]                     bf16 or fp8
    torch::Tensor w,                 // [E, hidden, inter]                   fp8
    torch::Tensor w_scale,           // [E, hidden/128, inter/128]           fp32
    torch::Tensor topk_ids,          // [B, topk]                           int32
    torch::Tensor topk_scale,        // [B, topk]                           fp32
    OptionalTensor buffer_scale = std::nullopt)  // [B*topk, inter/128]     fp32
{
    TORCH_CHECK(buffer.is_cuda(), "buffer must be CUDA");
    c10::cuda::CUDAGuard guard(buffer.device());

    const int B = routing_check(topk_ids, buffer.device());
    check(topk_scale, at::kFloat, buffer.device(), "topk_scale");
    shape(topk_scale, {B, K}, "topk_scale");
    weight_check(w, w_scale, buffer.device(), H, I);
    auto [buffer_fp8, activation_scale] = prepare_activation(buffer, buffer_scale, int64_t(B) * K, I);

    auto out = torch::empty({B, H}, buffer.options().dtype(at::kBFloat16));

    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_moe_down_3d_batched(
        buffer_fp8.data_ptr(),
        w.data_ptr(),
        activation_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        topk_ids.data_ptr<int>(),
        topk_scale.data_ptr<float>(),
        out.data_ptr(),
        B,
        stream.stream());

    return out;
}

torch::Tensor moe_ffn_3d_batched(
    torch::Tensor x,
    torch::Tensor gate_up_w,
    torch::Tensor down_w,
    torch::Tensor gate_up_scale,
    torch::Tensor down_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_scale,
    OptionalTensor x_scale = std::nullopt)
{
    // Validate both projections before enqueueing either one.
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    const int B = routing_check(topk_ids, x.device());
    weight_check(down_w, down_scale, x.device(), H, I);
    check(topk_scale, at::kFloat, x.device(), "topk_scale");
    shape(topk_scale, {B, K}, "topk_scale");

    auto buffer = moe_gate_up_3d_batched(x, gate_up_w, gate_up_scale, topk_ids, x_scale);
    return moe_down_3d_batched(buffer, down_w, down_scale, topk_ids, topk_scale);
}

}  // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("quantize_fp8_blockwise", &quantize_fp8_blockwise,
          "BF16 -> E4M3, FP32 scale per row/group of 128",
          py::arg("x"));
    m.def("moe_gate_up_3d_batched", &moe_gate_up_3d_batched,
          "Qwen3.6 TP1 FP8 gate/up + SiLU, BF16 output",
          py::arg("x"), py::arg("w"), py::arg("w_scale"), py::arg("topk_ids"),
          py::arg("x_scale") = py::none());
    m.def("moe_down_3d_batched", &moe_down_3d_batched,
          "Qwen3.6 TP1 FP8 down + FP32 top-k weighted sum",
          py::arg("buffer"), py::arg("w"), py::arg("w_scale"), py::arg("topk_ids"),
          py::arg("topk_scale"), py::arg("buffer_scale") = py::none());
    m.def("moe_ffn_3d_batched", &moe_ffn_3d_batched,
          "Qwen3.6 TP1 blockwise FP8 W8A8 MoE FFN",
          py::arg("x"), py::arg("gate_up_w"), py::arg("down_w"), py::arg("gate_up_scale"),
          py::arg("down_scale"), py::arg("topk_ids"), py::arg("topk_scale"),
          py::arg("x_scale") = py::none());
}
