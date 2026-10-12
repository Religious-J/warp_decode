# warp_decode

An **unofficial re-implementation** of Cursor's *warp decode* MoE inference
technique ([blog](https://cursor.com/blog/warp-decode)), targeting **blockwise FP8 W8A8
precision on NVIDIA Hopper GPUs (H20 / H100)**.

Cursor's original work targets Blackwell and MXFP8; the public blog describes
the high-level idea (organize the kernel around outputs instead of experts,
give each warp one output neuron, avoid all staging) but not the concrete
kernel shape. This repository is an independent implementation that arrives at
similar conclusions on Hopper by working the design out from a roofline
argument. This version targets Qwen3.6-35B-A3B at TP=1.

The two MoE kernels in this repository are:

| Kernel                          | Fused operation                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `moe_gate_up_3d_batched`        | gate + up projections, `silu(gate) * up` activation, all in one launch          |
| `moe_down_3d_batched`           | down projection with topk-weighted sum-reduce, in one launch                    |

Both take unsorted `topk_ids` directly, so **no `moe_align_block_size` /
scatter / permute is needed**. Router scoring and top-k selection are excluded.

---

## Performance on Hopper

### Setup

- **GPU:** NVIDIA H20 (Hopper, sm_90), with dynamic clocks.
- **Software:** PyTorch 2.13.0+cu130, CUDA toolkit 13.0.88, Triton 3.7.1,
  SGLang 0.5.21, sglang-kernel 0.4.7.
- **Shape:** Qwen3.6-35B-A3B MoE FFN under TP=1:
  `hidden=2048`, `inter=512`, `num_experts=256`, `topk=8`.
- **Reference:** SGLang's standard Triton FP8 configuration lookup and
  `fused_experts` end-to-end path.
- **Routing:** eight distinct experts per token, with normalized FP32 routing weights.
- **Timing:** CUDA Events over 16 calls per graph and 200 graph replays per sample.
  Weights and routing are fixed; repeated replay can benefit from caching.
- **Precision:** E4M3 activations and weights, FP32 accumulators. Activations
  use one scale per row and 128 elements; weights use one scale per 128×128 block.

### Results

The current FP8 implementation needs a complete benchmark run before recording
new performance tables. The benchmark measures dynamic activation quantization,
gate/up, SiLU, down, and expert combination in one call. Both implementations
receive precomputed top-k IDs and weights, so router computation is excluded.

---

## Design & optimizations

The blog gives one useful headline — *"assign each warp to one output value,
stream weights, avoid staging"*. Both kernels retain this output-oriented layout
and apply FP8 block scales within the dot-product loop.

### `moe_gate_up_3d_batched`

**Shape.** `X:[B, hidden]` × `W:[E, 2*inter, hidden]` → `[B*topk, inter]`,
where the output is `silu(gate) * up` fused.

**Kernel structure:**

- 2D grid: `(B*topk, inter / WARPS_PER_CTA)`, `blockDim = 128` = 4 warps.
- Each CTA handles one `(token, topk_slot)` pair.
- The 4 warps in a CTA together cover 4 adjacent output neurons.
- All 4 warps in a CTA read the **same** `x[token]` row directly from
  global memory through the cache hierarchy, without shared-memory staging.
- Each warp streams two weight rows (`gate` and `up` for its neuron) directly
  from HBM using vectorized `uint4` loads (`VEC = 16` FP8 per lane per iter),
  and accumulates the two dot products in registers.
- Each vector stays within one 128-element scale group. FP8 pairs are widened
  to FP32, accumulated, and multiplied by the activation and weight scales.
- Gate/up results round to BF16 before FP32 `silu(g) * u`; the final BF16
  write-out happens at the end in a single lane.

**Why this specific layout:**

- The reduction axis is `hidden = 2048`, giving four vector-load iterations.
- 4 warps per CTA provide warp-level parallelism to hide weight-loading latency.
- Each warp performs its own reduction, without a CTA barrier.

### `moe_down_3d_batched`

**Shape.** `buffer:[B, topk, inter]` × `W:[E, hidden, inter]` → `[B, hidden]`,
with per-slot `topk_scale` folded in and a sum-reduce over `topk`.

**Kernel structure:**

- **1D grid** over `(token, h-group)`. Each warp owns 4 adjacent output dims.
- No shared-memory staging. The input buffer is reused through the cache hierarchy.
- For each of the `topk` experts, the warp reads 4 FP8 buffer elements per lane
  and reuses them across its output weight rows.
- FP8 operands are widened before FP32 multiplication and accumulation;
  activation and weight block scales are applied within the reduction.
- Independent FP32 accumulators (one per output `h`) live in registers per
  lane; the topk-weighted sum folds `topk_scale` in FP32 before the final
  BF16 conversion.
- The output and reduction loops are `#pragma unroll`-ed.

### Numerical accuracy

Both kernels use E4M3 activations and weights with FP32 accumulators.
Dequantization is `value = fp8_value * scale`. BF16 inputs are dynamically
quantized per row and 128 elements, with `scale = max(abs(group), 1e-10) / 448`.
The BF16 SiLU output is requantized to FP8 before the down projection.
The down kernel applies `topk_scale` and sums across experts in FP32 before
one final BF16 conversion.

The implementation passed independent FP32-reference, SGLang-reference,
CUDA Graph, and CUDA memory-check tests.

Both kernels support PyTorch's current CUDA stream and CUDA Graph capture;
warp_decode does not currently use Programmatic Dependent Launch (PDL).

---

## Build & test

### Requirements

- Linux, a CUDA toolkit compatible with the installed PyTorch build, and a
  Hopper GPU (sm_90). The kernels were tested on H20 with CUDA 13.0.
- PyTorch (with matching CUDA build) — the kernels are JIT-compiled via
  `torch.utils.cpp_extension.load`, no separate build step is needed.
- For benchmarks: `sglang` and `sglang-kernel`.

### Install & first run

```bash
# From the repo root — the kernels are JIT-compiled on the first import.
python tests/test_fp8_blockwise.py

# With SGLang and sglang-kernel installed:
python tests/test_benchmark_reference.py
```

The initial import triggers an `nvcc` build under `build/`. Subsequent imports
reuse the cached extension.

Set `WARP_DECODE_VERBOSE=1` to see the `nvcc` invocation on rebuild:

```bash
WARP_DECODE_VERBOSE=1 python tests/test_fp8_blockwise.py
```

### Benchmark

```bash
python benchmark/bench.py --batches 1,2,4,8,16,32,64 --iters 200

# Custom batch set
python benchmark/bench.py --batches 1,8,16
```

The script prints the SGLang version and records selected configurations in
the output JSON. SGLang logs any configuration fallback. Both implementations
use existing FP8 weights and their FP32 block scales directly.

### Python API

```python
from warp_decode import warp_decode as wd

# gate_up + silu_and_mul (fused)
buf = wd.moe_gate_up_3d_batched(
    x,               # [B, hidden]         bf16
    gate_up_w,       # [E, 2*inter, hidden] fp8 E4M3
    gate_up_scale,   # [E, 2*inter/128, hidden/128] fp32
    topk_ids,        # [B, topk]           int32
)                    # -> [B*topk, inter]  bf16

# down + topk-weighted sum-reduce (fused)
out = wd.moe_down_3d_batched(
    buf,             # [B*topk, inter]     bf16
    down_w,          # [E, hidden, inter]  fp8 E4M3
    down_scale,      # [E, hidden/128, inter/128] fp32
    topk_ids,        # [B, topk]           int32
    topk_scale,      # [B, topk]           fp32
)                    # -> [B, hidden]      bf16

# Complete FFN, including both activation quantization steps
out = wd.moe_ffn_3d_batched(
    x, gate_up_w, down_w, gate_up_scale, down_scale, topk_ids, topk_scale,
)
```

Prequantized E4M3 activations are also accepted with their matching FP32
scale as the last argument (`x_scale` for gate/up and FFN, `buffer_scale` for down). `quantize_fp8_blockwise(x)` returns FP8 values
and scales for a BF16 input. Weights and their scales are supplied directly.

### Shape constraints

| kernel   | constraint                                                     |
| -------- | -------------------------------------------------------------- |
| gate_up  | `hidden = 2048`, `inter = 512`, `num_experts = 256`, `topk = 8` |
| down     | `hidden = 2048`, `inter = 512`, `num_experts = 256`, `topk = 8` |

Both kernels assume the standard blockwise FP8 weight layouts used by
sglang's fused_moe: E4M3 weights `[E,N,K]`, FP32 scales `[E,N/128,K/128]`.
All tensors must be contiguous and on the same CUDA device. FP8 data pointers
must be 16-byte aligned; BF16 quantizer inputs need 8-byte alignment. Scales
must be finite and positive; routing IDs must lie in `[0,256)`.

---

## Future work

**Consumer GPUs with multi-precision support (sm_120 and similar).**
Local / on-device deployment is exactly the setting where small-batch,
latency-bound decode kernels are most valuable — batch sizes are often 1–4,
and every millisecond of TTFT/ITL is user-visible. Concrete next steps:

- Extend to more MoE shapes and quantization layouts.
- Autotune the same shape knobs (`WARPS_PER_CTA`, `OUTS_PER_WARP`) for
  smaller SM counts and different L2/HBM ratios.

**Same idea on non-NVIDIA accelerators.**
The roofline argument that motivates warp_decode is architecture-agnostic.
The output-oriented layout could also be explored on AMD, Ascend, and other
accelerators for small-batch decode.

---

## Acknowledgements

- [Cursor's warp decode blog](https://cursor.com/blog/warp-decode) —
  the original idea.
- [sglang](https://github.com/sgl-project/sglang) — the reference
  implementation used for correctness and performance comparisons.
