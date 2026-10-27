// Torch bindings for warp_decode kernels.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <tuple>
#include "moe_3d_batched.h"

namespace {

constexpr int SCALE_GROUP = 128;
using OptionalTensor = std::optional<torch::Tensor>;
using QuantizedActivation = std::tuple<torch::Tensor, torch::Tensor>;

void check_tensor(
    const torch::Tensor& t,
    at::ScalarType dtype,
    const torch::Device& device,
    const char* name)
{
    TORCH_CHECK(t.is_cuda() && t.device() == device, name, " must be on the same CUDA device");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has an incorrect dtype");
}

void check_shape(
    const torch::Tensor& t,
    std::initializer_list<int64_t> expected,
    const char* name)
{
    TORCH_CHECK(t.sizes() == at::IntArrayRef(expected), name, " has an incorrect shape; expected ",
                at::IntArrayRef(expected), ", got ", t.sizes());
}

void check_alignment(const torch::Tensor& t, int bytes, const char* name) {
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % bytes == 0,
                name, " must have a ", bytes, "-byte-aligned data pointer");
}

QuantizedActivation quantize_fp8_blockwise(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    check_tensor(x, at::kBFloat16, x.device(), "x");
    check_alignment(x, 8, "x");
    TORCH_CHECK(x.dim() == 2 && x.size(1) % SCALE_GROUP == 0,
                "x must be [rows, cols] with cols a multiple of 128");
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() &&
                x.size(1) <= std::numeric_limits<int>::max(), "x is too large");

    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x, x.options().dtype(at::kFloat8_e4m3fn));
    auto x_scale = torch::empty({x.size(0), x.size(1) / SCALE_GROUP},
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

QuantizedActivation prepare_activation(
    torch::Tensor x,
    OptionalTensor x_scale,
    int64_t rows,
    int cols)
{
    check_shape(x, {rows, cols}, "activation");
    if (x.scalar_type() == at::kBFloat16) {
        TORCH_CHECK(!x_scale.has_value(), "activation scale is only accepted for prequantized FP8 input");
        return quantize_fp8_blockwise(x);
    }
    check_tensor(x, at::kFloat8_e4m3fn, x.device(), "activation");
    check_alignment(x, 16, "FP8 activation");
    TORCH_CHECK(x_scale.has_value(), "prequantized FP8 activation requires its FP32 scale");
    check_tensor(*x_scale, at::kFloat, x.device(), "activation scale");
    check_shape(*x_scale, {rows, cols / SCALE_GROUP}, "activation scale");
    return {x, *x_scale};
}

int check_routing(torch::Tensor topk_ids, const torch::Device& device) {
    check_tensor(topk_ids, at::kInt, device, "topk_ids");
    TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(1) > 0 && topk_ids.size(1) <= 32,
                "topk_ids must be [B, topk] with 1 <= topk <= 32");
    const int topk = topk_ids.size(1);
    TORCH_CHECK(topk_ids.size(0) <= std::numeric_limits<int>::max() / topk, "batch is too large");
    return topk_ids.size(0);
}

struct MoeShape {
    int hidden;
    int inter;
    int experts;
};

enum class Projection { GateUp, Down };

MoeShape check_weights(
    const torch::Tensor& w,
    const torch::Tensor& w_scale,
    const torch::Device& device,
    int topk,
    Projection projection)
{
    check_tensor(w, at::kFloat8_e4m3fn, device, "weight");
    check_alignment(w, 16, "FP8 weight");
    check_tensor(w_scale, at::kFloat, device, "weight scale");
    TORCH_CHECK(w.dim() == 3 && w.size(0) > 0 && w.size(1) > 0 && w.size(2) > 0,
                "weight must be [experts, output, input] with positive dimensions");
    const bool gate_up = projection == Projection::GateUp;
    const int64_t hidden = w.size(gate_up ? 2 : 1);
    const int64_t inter = gate_up ? w.size(1) / 2 : w.size(2);
    TORCH_CHECK((!gate_up || w.size(1) % 2 == 0) &&
                hidden % SCALE_GROUP == 0 && inter % SCALE_GROUP == 0,
                "hidden and per-rank intermediate must be multiples of 128");
    // The output dimension is the y dimension of the CUDA grid.
    TORCH_CHECK(hidden <= 65535 * 4 && inter <= 65535 * 8 &&
                w.size(0) <= std::numeric_limits<int>::max(), "weight shape is too large");
    TORCH_CHECK(topk <= w.size(0), "topk must not exceed the number of experts");
    check_shape(w_scale,
                {w.size(0), w.size(1) / SCALE_GROUP, w.size(2) / SCALE_GROUP},
                "weight scale");
    return {int(hidden), int(inter), int(w.size(0))};
}

bool should_fuse_down_quantization(int batch_size, int hidden, int inter, int topk) {
    // Inline quantization saves one launch, but repeats across output groups.
    // Keep the measured small-batch cases; other shapes quantize once up front.
    return batch_size == 1 && (topk == 8 || topk == 10) &&
           ((inter == 256 && (hidden == 2048 || hidden == 3072)) ||
            (inter == 128 && hidden == 4096 && topk == 10));
}

torch::Tensor moe_gate_up_3d_batched(
    torch::Tensor x,                 // [B, hidden]                 bf16 or fp8
    torch::Tensor w,                 // [E, 2*inter, hidden]        fp8
    torch::Tensor w_scale,           // [E, 2*inter/128, hidden/128] fp32
    torch::Tensor topk_ids,          // [B, topk]                   int32
    OptionalTensor x_scale = std::nullopt)  // [B, hidden/128]       fp32
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    c10::cuda::CUDAGuard guard(x.device());

    const int batch_size = check_routing(topk_ids, x.device());
    const int topk = topk_ids.size(1);
    const auto dims = check_weights(w, w_scale, x.device(), topk, Projection::GateUp);
    const int hidden = dims.hidden, inter = dims.inter;
    auto [x_fp8, activation_scale] = prepare_activation(x, x_scale, batch_size, hidden);

    auto out = torch::empty({int64_t(batch_size) * topk, inter},
                            x.options().dtype(at::kBFloat16));

    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_moe_gate_up_3d_batched(
        x_fp8.data_ptr(),
        w.data_ptr(),
        activation_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        topk_ids.data_ptr<int>(),
        out.data_ptr(),
        batch_size, hidden, inter, dims.experts, topk,
        stream.stream());

    return out;
}

torch::Tensor moe_down_3d_batched(
    torch::Tensor buffer,            // [B*topk, inter]             bf16 or fp8
    torch::Tensor w,                 // [E, hidden, inter]          fp8
    torch::Tensor w_scale,           // [E, hidden/128, inter/128]   fp32
    torch::Tensor topk_ids,          // [B, topk]                   int32
    torch::Tensor topk_scale,        // [B, topk]                   fp32
    OptionalTensor buffer_scale = std::nullopt)  // [B*topk, inter/128] fp32
{
    TORCH_CHECK(buffer.is_cuda(), "buffer must be CUDA");
    c10::cuda::CUDAGuard guard(buffer.device());

    const int batch_size = check_routing(topk_ids, buffer.device());
    const int topk = topk_ids.size(1);
    check_tensor(topk_scale, at::kFloat, buffer.device(), "topk_scale");
    check_shape(topk_scale, {batch_size, topk}, "topk_scale");
    const auto dims = check_weights(w, w_scale, buffer.device(), topk, Projection::Down);
    const int hidden = dims.hidden, inter = dims.inter;
    const int64_t routed_rows = int64_t(batch_size) * topk;

    if (should_fuse_down_quantization(batch_size, hidden, inter, topk) &&
        buffer.scalar_type() == at::kBFloat16 && !buffer_scale.has_value()) {
        check_tensor(buffer, at::kBFloat16, buffer.device(), "buffer");
        check_alignment(buffer, 8, "BF16 buffer");
        check_shape(buffer, {routed_rows, inter}, "buffer");
        auto out = torch::empty({batch_size, hidden}, buffer.options().dtype(at::kBFloat16));
        warp_decode::launch_moe_down_3d_batched_bf16(
            buffer.data_ptr(),
            w.data_ptr(),
            w_scale.data_ptr<float>(),
            topk_ids.data_ptr<int>(),
            topk_scale.data_ptr<float>(),
            out.data_ptr(),
            batch_size, hidden, inter, dims.experts, topk,
            at::cuda::getCurrentCUDAStream().stream());
        return out;
    }
    auto [buffer_fp8, activation_scale] = prepare_activation(
        buffer, buffer_scale, routed_rows, inter);

    auto out = torch::empty({batch_size, hidden}, buffer.options().dtype(at::kBFloat16));

    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_moe_down_3d_batched(
        buffer_fp8.data_ptr(),
        w.data_ptr(),
        activation_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        topk_ids.data_ptr<int>(),
        topk_scale.data_ptr<float>(),
        out.data_ptr(),
        batch_size, hidden, inter, dims.experts, topk,
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
    const int batch_size = check_routing(topk_ids, x.device());
    const int topk = topk_ids.size(1);
    const auto gate = check_weights(gate_up_w, gate_up_scale, x.device(), topk, Projection::GateUp);
    const auto down = check_weights(down_w, down_scale, x.device(), topk, Projection::Down);
    TORCH_CHECK(gate.hidden == down.hidden && gate.inter == down.inter &&
                gate.experts == down.experts,
                "gate/up and down weight shapes must describe the same MoE");
    check_tensor(topk_scale, at::kFloat, x.device(), "topk_scale");
    check_shape(topk_scale, {batch_size, topk}, "topk_scale");

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
          "Blockwise FP8 gate/up + SiLU, BF16 output",
          py::arg("x"), py::arg("w"), py::arg("w_scale"), py::arg("topk_ids"),
          py::arg("x_scale") = py::none());
    m.def("moe_down_3d_batched", &moe_down_3d_batched,
          "Blockwise FP8 down + FP32 top-k weighted sum",
          py::arg("buffer"), py::arg("w"), py::arg("w_scale"), py::arg("topk_ids"),
          py::arg("topk_scale"), py::arg("buffer_scale") = py::none());
    m.def("moe_ffn_3d_batched", &moe_ffn_3d_batched,
          "Blockwise FP8 W8A8 MoE FFN",
          py::arg("x"), py::arg("gate_up_w"), py::arg("down_w"), py::arg("gate_up_scale"),
          py::arg("down_scale"), py::arg("topk_ids"), py::arg("topk_scale"),
          py::arg("x_scale") = py::none());
}
