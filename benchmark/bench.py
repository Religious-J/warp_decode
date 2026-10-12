"""Benchmark warp_decode FP8 FFN against the SGLang Triton reference.

Fixed Qwen3.6 TP1 shape: hidden=2048, intermediate=512, experts=256, topk=8.
Both implementations include dynamic activation quantization and use supplied
FP8 weights with FP32 block scales. Weights stay at fixed addresses.

Run from the repo root:
    python benchmark/bench.py --batches 1,2,4,8,16,32,64
"""

import argparse
import hashlib
import importlib.metadata
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

# Make the package and shared test inputs importable regardless of CWD.
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import torch
import sglang
import triton

from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args, publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)

from warp_decode import warp_decode as wd
from fp8_reference import make_weights, make_inputs, ffn_reference, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1,2,4,8,16,32,64")
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--calls-per-graph", type=int, default=16)
    p.add_argument("--output", default="fp8_blockwise.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_e2e_inputs(batch):
    x, ids, route = make_inputs(batch)
    topk = StandardTopKOutput(route, ids, None)
    return x, ids, route, topk


# ---------------------------------------------------------------------------
# Bench helper (CUDA graph)
# ---------------------------------------------------------------------------


def initialize_sglang():
    """Use declared server defaults without loading a model or starting a server."""
    try:
        get_server_args()
    except ValueError:
        publish(
            ServerArgs(model_path="dummy", moe_runner_backend="triton"),
            role="test",
        )
        assert not get_exec().comm.enable_symm_mem
        assert not get_exec().moe.enable_fused_moe_sum_all_reduce
        get_parallel().override_permanently(tp_group=None)


def capture(fn, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(30):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            out = fn()
    torch.cuda.synchronize()
    return graph, out


def measure(graph, calls, replays):
    for _ in range(20):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (calls * replays)


# ---------------------------------------------------------------------------
# e2e bench
# ---------------------------------------------------------------------------

def make_e2e_wd(x, gate_up_w, down_w, gate_up_scale, down_scale, ids, route):
    def fn():
        return wd.moe_ffn_3d_batched(
            x, gate_up_w, down_w, gate_up_scale, down_scale, ids, route,
        )
    return fn


def make_e2e_sgl(x, gate_up_w, down_w, gate_up_scale, down_scale, topk, runner):
    """Call the standard SGLang Triton FP8 FFN, including routing and combine."""
    def fn():
        return fused_experts(
            x, gate_up_w, down_w, topk, runner,
            use_fp8_w8a8=True,
            w1_scale=gate_up_scale, w2_scale=down_scale, block_shape=[128, 128],
        )
    return fn


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if min(args.rounds, args.iters, args.calls_per_graph) < 1:
        raise ValueError("rounds, iters and calls-per-graph must be positive")
    initialize_sglang()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gu, dn, gs, ds = make_weights(args.seed)
    root = Path(__file__).resolve().parents[1]
    result = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        sglang=sglang.__version__,
        sgl_kernel=importlib.metadata.version("sglang-kernel"),
        triton=triton.__version__,
        seed=args.seed,
        shape=dict(hidden=2048, inter=512, experts=256, topk=8, tp=1),
        layout="E4M3 W8A8; activation 1x128; weight 128x128; FP32 scales",
        protocol=dict(
            rounds=args.rounds,
            replays=args.iters,
            calls_per_graph=args.calls_per_graph,
            weights="fixed addresses; no rotation/cache flush",
            clocks="dynamic",
            graph_update_inputs="restored before timing",
            scope="FFN includes activation quantization and combine; excludes weight setup and router",
        ),
        source_hashes={
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                root / "csrc/moe_gate_up_3d_batched.cu",
                root / "csrc/moe_down_3d_batched.cu",
                root / "csrc/moe_3d_batched.h",
                root / "csrc/binding.cpp",
                root / "__init__.py",
                root / "tests/fp8_reference.py",
                Path(__file__),
            ]
        },
        gpu_start=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        cases=[],
    )
    runner = MoeRunnerConfig(
        num_experts=256, num_local_experts=256, top_k=8,
        inplace=False, gate_up_interleaved=False,
    )
    print("Stack:", result["torch"], result["sglang"], result["triton"], flush=True)
    for b in map(int, args.batches.split(",")):
        if b <= 0:
            raise ValueError("benchmark batch must be positive")
        x, ids, route, topk = make_e2e_inputs(b)
        native = make_e2e_wd(x, gu, dn, gs, ds, ids, route)
        sgl = make_e2e_sgl(x, gu, dn, gs, ds, topk, runner)
        expected = ffn_reference(x, gu, dn, gs, ds, ids, route)
        validation = {
            "warp_decode_fp8": metrics(native(), expected),
            "sglang_fp8": metrics(sgl(), expected, .02),
        }
        q, _ = wd.quantize_fp8_blockwise(x)
        sq, _ = sglang_per_token_group_quant_fp8(x, 128)
        validation["sglang_quant_byte_differences"] = (
            (q.view(torch.uint8) != sq.view(torch.uint8)).sum().item()
        )
        config, (down_config, _) = try_get_optimal_moe_config(
            gu.shape, dn.shape, 8, "fp8_w8a8", b,
            block_shape=[128, 128], return_down_config=True,
        )
        fns = {"warp_decode_fp8": native, "sglang_fp8": sgl}
        graphs = {
            name: capture(fn, args.calls_per_graph) for name, fn in fns.items()
        }
        # Verify graph updates, then restore the fixture before timing.
        original_inputs = [(tensor, tensor.clone()) for tensor in (x, gs, ds, ids, route)]
        x.mul_(-1.03)
        gs.mul_(1.001)
        ds.mul_(.999)
        ids.copy_(ids.roll(1, -1))
        route.copy_(route.roll(1, -1))
        # Alter actual FP8 weight bytes in an expert guaranteed to be routed.
        expert = ids[0, 0].item()
        original = gu[expert, 0, :128].clone()
        gu[expert, 0, :128].copy_((-original.float()).to(gu.dtype))
        for name, (graph, out) in graphs.items():
            graph.replay()
            torch.testing.assert_close(out, fns[name](), rtol=0, atol=0)
        gu[expert, 0, :128].copy_(original)
        for tensor, value in original_inputs:
            tensor.copy_(value)
        for name, (graph, out) in graphs.items():
            graph.replay()
            torch.testing.assert_close(out, fns[name](), rtol=0, atol=0)
        validation["graph_update_and_restore"] = True
        # Validate on the exact values used for timing as well.
        expected = ffn_reference(x, gu, dn, gs, ds, ids, route)
        validation["timed_native"] = metrics(native(), expected)
        validation["timed_sglang"] = metrics(sgl(), expected, .02)
        samples = {name: [] for name in fns}
        order = []
        for _ in range(args.rounds):
            names = list(fns)
            rng.shuffle(names)
            order.append(names)
            for name in names:
                samples[name].append(
                    measure(graphs[name][0], args.calls_per_graph, args.iters)
                )
        medians = {
            name: statistics.median(values) for name, values in samples.items()
        }
        result["cases"].append(dict(
            batch=b,
            ids=ids.cpu().tolist(),
            config=config,
            down_config=down_config,
            validation=validation,
            order=order,
            samples_ms=samples,
            median_ms=medians,
            speedup=medians["sglang_fp8"] / medians["warp_decode_fp8"],
        ))
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print("B", b, medians, "speedup", result["cases"][-1]["speedup"], flush=True)


if __name__ == "__main__":
    main()
