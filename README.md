# warp_decode

An **unofficial re-implementation** of Cursor's *warp decode* MoE inference
technique ([blog](https://cursor.com/blog/warp-decode)), targeting **BF16
precision on NVIDIA Hopper GPUs (H20 / H100)**.

Cursor's original work targets Blackwell and MXFP8; the public blog describes
the high-level idea (organize the kernel around outputs instead of experts,
give each warp one output neuron, avoid all staging) but not the concrete
kernel shape. This repository is an independent implementation that arrives at
similar conclusions on Hopper by working the design out from a roofline
argument, and adds a few Hopper-specific optimizations that were needed to
actually turn the idea into a measured win.

The two kernels in this repository are:

| Kernel                          | Fused operation                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `moe_gate_up_3d_batched`        | gate + up projections, `silu(gate) * up` activation, all in one launch          |
| `moe_down_3d_batched`           | down projection with topk-weighted sum-reduce, in one launch                    |

Both take unsorted `topk_ids` directly, so **no `moe_align_block_size` /
scatter / permute is needed** — routing overhead is zero.

---

## Performance on Hopper

### Setup

- **GPU:** NVIDIA H20 (Hopper, sm_90).
- **Software:** PyTorch 2.11.0+cu130, CUDA toolkit 13.0.88, SGLang 0.5.13.post1.
- **Shape:** Qwen3-235B-A22B-style MoE FFN under TP=8:
  `hidden=4096`, `inter=192` (1536 before TP split), `num_experts=128`, `topk=8`.
- **Reference:** SGLang's standard Triton configuration lookup and
  `fused_experts` end-to-end path. Configurations are selected using the original
  batch size `M=B`; gate/up and down use their respective standard configurations.
- **Routing:** eight distinct experts per token, with normalized FP32 routing weights.
- **Timing:** CUDA Graph replay, 25 warmup calls and 200 timed calls per run.
  Latencies are medians of three runs; speedups are calculated from those medians.
- **Precision:** BF16 activations and weights, FP32 accumulators.

For this environment, SGLang falls back to its bundled Triton 3.2.0 H20
configuration. There is no dedicated down configuration for this shape, so
SGLang uses its standard gate/up configuration fallback.

### Results

Times are per call in milliseconds. `speedup = sglang_ms / warp_decode_ms`.
SGLang stage timings include expert alignment. Both implementations receive
precomputed top-k IDs and weights, so router computation is excluded. The
end-to-end benchmark measures gate/up, activation, down, and expert combination
in one call. Stage timings are measured separately and should not be added to
estimate end-to-end latency.

**Gate/up + SiLU activation:**

| batch | tokens | warp_decode (ms) | SGLang Triton (ms) | speedup |
| ----: | -----: | ---------------: | -----------------: | ------: |
| 1 | 8 | 0.0075 | 0.0199 | 2.65x |
| 2 | 16 | 0.0148 | 0.0319 | 2.16x |
| 4 | 32 | 0.0315 | 0.0405 | 1.29x |
| 8 | 64 | 0.0528 | 0.0648 | 1.23x |
| 16 | 128 | 0.0849 | 0.0911 | 1.07x |
| 32 | 256 | 0.1451 | 0.1184 | 0.82x |
| 64 | 512 | 0.2760 | 0.1322 | 0.48x |

**Down + top-k weighted sum:**

| batch | tokens | warp_decode (ms) | SGLang Triton (ms) | speedup |
| ----: | -----: | ---------------: | -----------------: | ------: |
| 1 | 8 | 0.0101 | 0.0168 | 1.66x |
| 2 | 16 | 0.0115 | 0.0225 | 1.96x |
| 4 | 32 | 0.0205 | 0.0270 | 1.32x |
| 8 | 64 | 0.0391 | 0.0447 | 1.14x |
| 16 | 128 | 0.0632 | 0.0579 | 0.92x |
| 32 | 256 | 0.1170 | 0.0880 | 0.75x |
| 64 | 512 | 0.2286 | 0.0816 | 0.36x |

**End-to-end MoE FFN:**

| batch | tokens | warp_decode (ms) | SGLang Triton (ms) | speedup |
| ----: | -----: | ---------------: | -----------------: | ------: |
| 1 | 8 | 0.0232 | 0.0348 | 1.50x |
| 2 | 16 | 0.0350 | 0.0510 | 1.46x |
| 4 | 32 | 0.0511 | 0.0633 | 1.24x |
| 8 | 64 | 0.0878 | 0.0953 | 1.09x |
| 16 | 128 | 0.1430 | 0.1475 | 1.03x |
| 32 | 256 | 0.2682 | 0.1983 | 0.74x |
| 64 | 512 | 0.5187 | 0.2047 | 0.39x |

### Interpretation

For this shape on H20, warp_decode is faster at small batch sizes: the
end-to-end speedup is 1.50x at batch 1 and 1.09x at batch 8. Batch 16 is close
to parity (1.03x), while the standard SGLang reference is faster at batches
32 and 64. The gate/up kernel remains slightly faster at batch 16, but the
down kernel is already slower than its reference.

These measurements support using warp_decode for small-batch decode. The
crossover depends on the shape, routing distribution, GPU, and SGLang
configuration; it should be measured for the intended workload.

---

## Design & optimizations

The blog gives one useful headline — *"assign each warp to one output value,
stream weights, avoid staging"* — but leaves out the details that actually
determine performance on Hopper. Direct translation of the headline gives a
kernel that is correct but far from optimal (we measured this — see below).
Two structural changes were needed for each of the two kernels.

### `moe_gate_up_3d_batched`

**Shape.** `X:[B, hidden]` × `W:[E, 2*inter, hidden]` → `[B*topk, inter]`,
where the output is `silu(gate) * up` fused.

**Kernel structure:**
- 2D grid: `(B*topk, inter / WARPS_PER_CTA)`, `blockDim = 256` = 8 warps
- Each CTA handles one `(token, topk_slot)` pair.
- The 8 warps in a CTA together cover 8 adjacent output neurons.
- All 8 warps in a CTA share the **same** `x[token]` row → staged once into
  shared memory, then reused 8× on-chip.
- Each warp streams two weight rows (`gate` and `up` for its neuron) directly
  from HBM using vectorized `uint4` loads (`VEC = 8` bf16 per lane per iter),
  and accumulates the two dot products in registers.
- Final `silu(g) * u` and the bf16 write-out happen at the end in a single
  lane.

**Why this specific layout:**
- The reduction axis is `hidden = 4096`, long enough that a single warp gives
  a sufficiently deep dot-product pipeline.
- 8 warps per CTA is the largest choice that keeps the shared-memory copy of
  `x[token]` small (`hidden * 2B` = 8 KB) while giving every SM enough
  warp-level parallelism to hide the weight-loading latency.
- No cross-warp synchronization on the reduction — each warp is entirely
  independent after the initial `x` load.

### `moe_down_3d_batched`

**Shape.** `buffer:[B, topk, inter]` × `W:[E, hidden, inter]` → `[B, hidden]`,
with per-slot `topk_scale` folded in and a sum-reduce over `topk`.

**This kernel is where the naive translation of Cursor's headline hurts.**
The obvious mirror of the gate_up design — 1 warp per output `h`, stage
`buffer[token, :, :]` in shared memory, sum across topk in a single warp —
turns out to be significantly slower than the sglang triton reference for any
`B > 4`. The reasons are structural:

1. The reduction axis is `inter = 192`, only ~3 lane-loop iterations. A
   single dot product is not enough to hide HBM latency by itself.
2. There is no cross-warp reuse of the input — every warp reads its own set
   of weight rows, and staging the small `buffer` in smem just adds
   `__syncthreads` and register pressure without any real reuse win.

The design that actually works on Hopper is very different:

**Kernel structure (v2):**
- **1D grid** over `(token, h-group)`. Each warp owns `OUTS_PER_WARP = 8`
  adjacent output dims for one token — *not* a single `h`.
- No shared memory at all. The 3 KB per-token buffer fits comfortably in the
  L1/L2 hierarchy on its own; explicit staging just costs.
- For each of the `topk` experts, the warp reads one `bf16x2` element of the
  buffer and reuses it against **8 different weight rows** (one for each of
  its output dims), doing 8 `__hmul2` dot-product updates before consuming
  the next buffer element.
- 8 independent FP32 accumulators (one per output `h`) live in registers per
  lane; the topk-weighted sum folds `topk_scale` in FP32 before the final
  BF16 conversion.
- Both inner loops (over `inter/2` and over the 8 outputs) are fully
  `#pragma unroll`-ed so the compiler can hoist all the loads and overlap
  them with the FMAs.

**Development measurements** (Qwen3 TP=8, batch 16):

| version | down time |
| ------- | --------: |
| v1: 1 warp / output, smem-staged buffer | 0.143 ms |
| v2: `OUTS_PER_WARP = 2`, no smem | 0.097 ms |
| v2: `OUTS_PER_WARP = 8`, fully unrolled | 0.066 ms |

These earlier measurements illustrate the effect of the kernel design changes.
The performance tables above use the current standard SGLang reference.

The `OUTS_PER_WARP` sweep on H20 showed 8 is the sweet spot: 4 leaves
input-side arithmetic intensity on the table, 16 pushes register pressure
past the point where occupancy falls off and the B=1 CTA count halves.

### Numerical accuracy

Both kernels use BF16 activations and weights with FP32 accumulators. The
gate/up kernel converts operands to FP32 before multiplication and applies
SiLU in FP32 before writing BF16 output. The down kernel uses BF16 `__hmul2`
products, then converts those products to FP32 for accumulation. It applies
`topk_scale` and sums across experts in FP32 before the final BF16 conversion;
the BF16 products still introduce rounding before accumulation.

The kernel tests compare against PyTorch BF16 references with
`atol=5e-2, rtol=5e-2`. In the tested Qwen3-style cases, the largest observed
absolute differences were approximately 0.0039 for gate/up and 0.00049 for
down. The standard-reference test also checks exact agreement between the
staged SGLang reference and `fused_experts`, FP32-reference accuracy, and CUDA
Graph replay with changed inputs and routing. All 20 test cases passed in
the environment listed above.

---

## Build & test

### Requirements

- Linux, a CUDA toolkit compatible with the installed PyTorch build, and a
  Hopper GPU (sm_90). The benchmark above was tested on H20 with CUDA 13.0.
- PyTorch (with matching CUDA build) — the kernels are JIT-compiled via
  `torch.utils.cpp_extension.load`, no separate build step is needed.
- For the benchmark and `test_benchmark_reference.py`: `sglang` and `sgl-kernel`.
  The standard reference was tested with SGLang 0.5.13.post1 and requires its
  `moe_runner.triton_utils` API.

### Install & first run

```bash
# From the repo root — the kernels are JIT-compiled on the first import.
python tests/test_gate_up_correctness.py
python tests/test_down_correctness.py

# With SGLang and sgl-kernel installed:
python tests/test_benchmark_reference.py
```

The initial import triggers an `nvcc` build under `build/`. Subsequent imports
reuse the cached extension.

Set `WARP_DECODE_VERBOSE=1` to see the `nvcc` invocation on rebuild:

```bash
WARP_DECODE_VERBOSE=1 python tests/test_gate_up_correctness.py
```

### Benchmark

```bash
# Reproduce the batch set and iteration count used in the tables above.
# Run three times and take the median for each row.
python benchmark/bench.py --mode all --batches 1,2,4,8,16,32,64 --warmup 25 --iters 200

# One mode only, custom batch set
python benchmark/bench.py --mode e2e --batches 1,8,16,32,64

# Custom shape (e.g. TP=4)
python benchmark/bench.py --tp-size 4 --batches 1..32
```

The script prints the SGLang version and selected configurations. SGLang logs
any configuration fallback. The isolated down benchmark includes routing and
uses initialized activations for both implementations. For a down configuration
that requires sorted TMA activations, use `--mode e2e` to run SGLang's standard
end-to-end path.

### Python API

```python
from warp_decode import warp_decode as wd

# gate_up + silu_and_mul (fused)
buf = wd.moe_gate_up_3d_batched(
    x,               # [B, hidden]         bf16
    gate_up_w,       # [E, 2*inter, hidden] bf16
    topk_ids,        # [B, topk]           int32
)                    # -> [B*topk, inter]  bf16

# down + topk-weighted sum-reduce (fused)
out = wd.moe_down_3d_batched(
    buf,             # [B*topk, inter]     bf16
    down_w,          # [E, hidden, inter]  bf16
    topk_ids,        # [B, topk]           int32
    topk_scale,      # [B, topk]           fp32
    B,               # int
)                    # -> [B, hidden]      bf16
```

### Shape constraints

| kernel   | constraint                                                     |
| -------- | -------------------------------------------------------------- |
| gate_up  | `hidden % 256 == 0` (32 lanes × VEC=8), `inter % 8 == 0`       |
| down     | `inter % 64 == 0` (32 lanes × VEC=2), `hidden % 8 == 0`        |

Both kernels assume the standard weight layouts used by sglang's fused_moe.

---

## Future work

**Consumer GPUs with multi-precision support (sm_120 and similar).**
Local / on-device deployment is exactly the setting where small-batch,
latency-bound decode kernels are most valuable — batch sizes are often 1–4,
and every millisecond of TTFT/ITL is user-visible. Extending warp_decode to
GeForce-class Blackwell (sm_120) with the full precision matrix (BF16, FP16,
FP8, MXFP8, INT8, INT4) would let a single-GPU workstation run frontier MoEs
at interactive latency. Concrete next steps:
- On-the-fly dequant inside the warp loop — the parallelism layout does not
  change, only the innermost read+convert.
- Group / per-channel / MX scale handling that stays in registers to avoid
  extra smem traffic.
- Autotune the same shape knobs (`WARPS_PER_CTA`, `OUTS_PER_WARP`) for
  smaller SM counts and different L2/HBM ratios.

**Same idea on non-NVIDIA accelerators.**
The roofline argument that motivates warp_decode is architecture-agnostic:
for `B * topk / E_local < ridge_point`, any expert-centric grouped GEMM is
wasting work, and an output-centric kernel that avoids all routing/staging
should win. Ports worth trying:
- **AMD (CDNA3 / RDNA):** HIP is close enough to CUDA that the two kernels
  should translate almost mechanically; the CDNA3 memory hierarchy has a
  lower ridge point, which extends the useful batch range further.
- **Ascend NPU (Cube / Vector cores):** the equivalent of the "one warp per
  output" mapping is one Vector core per output group, feeding the Cube
  core with weight tiles. The same structural argument — skip routing,
  fold the topk-weighted sum inside the compute unit — should transfer.
- **Other tile-based accelerators (Trainium, TPU, etc.):** any device where
  small-batch decode is memory-bound is a candidate.


---

## Acknowledgements

- [Cursor's warp decode blog](https://cursor.com/blog/warp-decode) —
  the original idea.
- [sglang](https://github.com/sgl-project/sglang) — the reference
  implementation used for correctness and performance comparisons.
