"""Benchmark warp_decode kernels against a sglang triton reference.

Three modes (default: all):
  gate_up : warp_decode.moe_gate_up_3d_batched  vs  triton gate_up + silu_and_mul
  down    : warp_decode.moe_down_3d_batched      vs  triton down + sum-reduce
  e2e     : gate_up + down (warp_decode)         vs  full sglang MoE pipeline

Run from the repo root:
    python benchmark/bench.py --batches 1,2,4,8,16,32,64
"""

import argparse
import sys
from pathlib import Path

# Make the `warp_decode` package importable regardless of CWD.
# Repo layout: <repo>/warp_decode/benchmark/bench.py  ->  parents[2] = <repo>
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import torch
import triton.language as tl

import sglang
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    fused_experts,
    moe_sum_reduce_torch_compile,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.jit_kernel.activation import silu_and_mul
from sgl_kernel import moe_sum_reduce

from warp_decode import warp_decode as wd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--intermediate-size", type=int, default=1536,
                   help="moe_intermediate_size per expert BEFORE TP split")
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--mode", choices=["gate_up", "down", "e2e", "all"], default="all")
    p.add_argument("--batches", type=str, default="1..64",
                   help='e.g. "1..64", "16,32,64,128", "16..512:16"')
    return p.parse_args()


def parse_batches(spec):
    if ":" in spec:
        rng, step = spec.split(":")
        a, b = rng.split("..")
        return list(range(int(a), int(b) + 1, int(step)))
    if ".." in spec:
        a, b = spec.split("..")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_inputs(B, hidden, inter, E_local, topk, device="cuda"):
    dtype = torch.bfloat16
    x = torch.randn(B, hidden, dtype=dtype, device=device)
    gate_up_w  = torch.randn(E_local, 2 * inter, hidden, dtype=dtype, device=device)
    down_w     = torch.randn(E_local, hidden, inter,    dtype=dtype, device=device)
    # Real top-k routing selects distinct experts for each token.
    if not 1 <= topk <= E_local:
        raise ValueError("topk must be between 1 and num_experts")
    topk_ids = torch.rand(B, E_local, device=device).topk(topk, dim=1).indices
    topk_ids = topk_ids.to(torch.int32).contiguous()
    raw        = torch.rand(B, topk, dtype=torch.float32, device=device)
    topk_scale = raw / raw.sum(dim=1, keepdim=True)
    return x, gate_up_w, down_w, topk_ids, topk_scale


# ---------------------------------------------------------------------------
# Bench helper (CUDA graph)
# ---------------------------------------------------------------------------

def bench_cuda_graph(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=stream):
            fn()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def initialize_sglang():
    """Use declared server defaults without loading a model or starting a server."""
    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", moe_runner_backend="triton")
        )


def sglang_bf16_configs(B, gate_up_shape, down_shape, topk):
    """Use SGLang's device/shape config lookup with original batch size M=B."""
    initialize_sglang()
    config, (down_config, _) = try_get_optimal_moe_config(
        gate_up_shape, down_shape, topk, None, B, return_down_config=True,
    )
    return dict(config), dict(down_config or config)


def _invoke_bf16(a, w, out, ids, scale, routing, config, mul_weight, topk):
    invoke_fused_moe_kernel(
        a, w, None, out, None, None, None, scale, ids, *routing,
        mul_weight, topk, config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False,
        per_channel_quant=False, block_shape=None, filter_expert=False,
    )


# ---------------------------------------------------------------------------
# gate_up bench
# ---------------------------------------------------------------------------

def bench_gate_up_wd(x, gate_up_w, topk_ids, warmup, iters):
    def fn():
        wd.moe_gate_up_3d_batched(x, gate_up_w, topk_ids)
    return bench_cuda_graph(fn, warmup, iters)


def make_gate_up_sgl(x, gate_up_w, topk_ids, topk_scale):
    B, hidden = x.shape
    E, two_inter, _ = gate_up_w.shape
    inter = two_inter // 2
    topk = topk_ids.shape[1]
    config, _ = sglang_bf16_configs(
        B, gate_up_w.shape, (E, hidden, inter), topk,
    )
    # Routing has padded indices; activation rows are logical token/slot rows.
    cache1 = torch.empty(B * topk, 2 * inter, dtype=x.dtype, device=x.device)
    cache2 = torch.empty(B * topk, inter, dtype=x.dtype, device=x.device)

    def fn():
        routing = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
        _invoke_bf16(x, gate_up_w, cache1, topk_ids, topk_scale,
                     routing, config, False, topk)
        silu_and_mul(cache1, cache2)
        return cache2

    return fn


def bench_gate_up_sgl(x, gate_up_w, topk_ids, topk_scale, E_local, warmup, iters):
    assert E_local == gate_up_w.shape[0]
    return bench_cuda_graph(
        make_gate_up_sgl(x, gate_up_w, topk_ids, topk_scale), warmup, iters,
    )


# ---------------------------------------------------------------------------
# down bench
# ---------------------------------------------------------------------------

def bench_down_wd(buffer, down_w, topk_ids, topk_scale, warmup, iters):
    B = topk_scale.shape[0]
    def fn():
        wd.moe_down_3d_batched(buffer, down_w, topk_ids, topk_scale, B)
    return bench_cuda_graph(fn, warmup, iters)


def make_down_sgl(buffer, down_w, topk_ids, topk_scale):
    """Standard down config and combine kernels, with routing included in timing."""
    B, topk = topk_ids.shape
    E, hidden, inter = down_w.shape
    config, down_config = sglang_bf16_configs(
        B, (E, 2 * inter, hidden), down_w.shape, topk,
    )
    if down_config.get("USE_TMA", False):
        raise NotImplementedError(
            "Standalone down uses unsorted activations; use --mode e2e "
            "for SGLang's sorted TMA path."
        )
    cache3 = torch.empty(B, topk, hidden, dtype=buffer.dtype, device=buffer.device)
    out = torch.empty(B, hidden, dtype=buffer.dtype, device=buffer.device)

    def fn():
        routing = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
        _invoke_bf16(buffer, down_w, out.unsqueeze(0) if topk == 1 else cache3,
                     topk_ids, topk_scale, routing, down_config, True, 1)
        if topk == 2:
            torch.add(cache3[:, 0], cache3[:, 1], out=out)
        elif topk > 2:
            if B <= 32:
                moe_sum_reduce_torch_compile(cache3, out, 1.0)
            else:
                moe_sum_reduce(cache3, out, 1.0)
        return out

    return fn


def bench_down_sgl(buffer, down_w, topk_ids, topk_scale, E_local, warmup, iters):
    assert E_local == down_w.shape[0]
    return bench_cuda_graph(
        make_down_sgl(buffer, down_w, topk_ids, topk_scale), warmup, iters,
    )


# ---------------------------------------------------------------------------
# e2e bench
# ---------------------------------------------------------------------------

def bench_e2e_wd(x, gate_up_w, down_w, topk_ids, topk_scale, warmup, iters):
    B = x.shape[0]
    def fn():
        buf = wd.moe_gate_up_3d_batched(x, gate_up_w, topk_ids)
        wd.moe_down_3d_batched(buf, down_w, topk_ids, topk_scale, B)
    return bench_cuda_graph(fn, warmup, iters)


def make_e2e_sgl(x, gate_up_w, down_w, topk_ids, topk_scale):
    """Call the standard SGLang Triton FFN, including routing and combine."""
    initialize_sglang()
    E = gate_up_w.shape[0]
    runner_config = MoeRunnerConfig(
        num_experts=E, num_local_experts=E,
        top_k=topk_ids.shape[1], inplace=False,
    )
    topk_output = StandardTopKOutput(topk_scale, topk_ids, None)

    def fn():
        return fused_experts(x, gate_up_w, down_w, topk_output, runner_config)

    return fn


def bench_e2e_sgl(x, gate_up_w, down_w, topk_ids, topk_scale, E_local, warmup, iters):
    assert E_local == gate_up_w.shape[0]
    return bench_cuda_graph(
        make_e2e_sgl(x, gate_up_w, down_w, topk_ids, topk_scale), warmup, iters,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_mode(name, B_list, args, hidden, inter, E_local, topk, wd_call, sgl_call):
    print(f"\n=== {name} ===")
    hdr = (f"{'batch':>6}  {'toks':>6}  "
           f"{'wd(ms)':>10}  {'sgl(ms)':>10}  {'speedup':>8}")
    print(hdr)
    print("-" * len(hdr))

    for B in B_list:
        x, gate_up_w, down_w, ids, scale = make_inputs(
            B, hidden, inter, E_local, topk
        )
        wd_ms  = wd_call (x, gate_up_w, down_w, ids, scale, args.warmup, args.iters)
        sgl_ms = sgl_call(x, gate_up_w, down_w, ids, scale, args.warmup, args.iters)

        print(f"{B:>6}  {B*topk:>6}  "
              f"{wd_ms:>10.4f}  {sgl_ms:>10.4f}  {sgl_ms/wd_ms:>7.2f}x")


def main():
    args = parse_args()
    hidden = args.hidden_size
    if args.tp_size <= 0 or args.intermediate_size % args.tp_size:
        raise ValueError("intermediate-size must be divisible by a positive tp-size")
    inter  = args.intermediate_size // args.tp_size
    E_local = args.num_experts     # matches sglang bench: TP shards only inter
    topk = args.topk
    B_list = parse_batches(args.batches)

    initialize_sglang()
    torch.manual_seed(0)

    dev = torch.cuda.get_device_properties(0)

    print(f"\ndevice        : {dev.name}")
    print(f"reference     : SGLang {sglang.__version__} standard Triton path")
    print(f"config        : hidden={hidden} inter={inter} E_local={E_local} topk={topk}")
    print(f"weights       : gate_up={E_local*2*inter*hidden*2/1e6:.1f} MB, "
          f"down={E_local*hidden*inter*2/1e6:.1f} MB")

    for B in B_list:
        up_config, down_config = sglang_bf16_configs(
            B, (E_local, 2 * inter, hidden), (E_local, hidden, inter), topk,
        )
        print(f"sglang config : B={B} gate_up={up_config} down={down_config}")

    modes = ["gate_up", "down", "e2e"] if args.mode == "all" else [args.mode]

    # signature adapters: (x, gate_up_w, down_w, ids, scale, warmup, iters) -> ms
    def _wd_gu (x, gu, dn, ids, sc, w, i): return bench_gate_up_wd (x, gu, ids, w, i)
    def _sgl_gu(x, gu, dn, ids, sc, w, i): return bench_gate_up_sgl(x, gu, ids, sc, E_local, w, i)

    def _wd_dn (x, gu, dn, ids, sc, w, i):
        # Use a real gate_up output so the L2 state during the down bench is
        # representative of the e2e path.
        buf = wd.moe_gate_up_3d_batched(x, gu, ids)
        return bench_down_wd(buf, dn, ids, sc, w, i)

    def _sgl_dn(x, gu, dn, ids, sc, w, i):
        buf = wd.moe_gate_up_3d_batched(x, gu, ids)
        return bench_down_sgl(buf, dn, ids, sc, E_local, w, i)

    def _wd_e2e (x, gu, dn, ids, sc, w, i): return bench_e2e_wd (x, gu, dn, ids, sc,           w, i)
    def _sgl_e2e(x, gu, dn, ids, sc, w, i): return bench_e2e_sgl(x, gu, dn, ids, sc, E_local, w, i)

    if "gate_up" in modes:
        run_mode("gate_up  (warp_decode vs triton gate_up + silu_and_mul)",
                 B_list, args, hidden, inter, E_local, topk, _wd_gu, _sgl_gu)
    if "down" in modes:
        run_mode("down     (warp_decode vs triton down + sum-reduce)",
                 B_list, args, hidden, inter, E_local, topk, _wd_dn, _sgl_dn)
    if "e2e" in modes:
        run_mode("e2e      (warp_decode gate_up+down vs full sglang pipeline)",
                 B_list, args, hidden, inter, E_local, topk, _wd_e2e, _sgl_e2e)


if __name__ == "__main__":
    main()
