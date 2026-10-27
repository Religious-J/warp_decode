"""Benchmark warp_decode FP8 FFN against the SGLang Triton reference.

Defaults to Qwen3.6 TP1: hidden=2048, intermediate=512, experts=256, topk=8.
Use --inter for the intermediate dimension per tensor-parallel rank.
Both implementations include dynamic activation quantization and use supplied
FP8 weights with FP32 block scales. Weights stay at fixed addresses.

Run from the repo root:
    python benchmark/bench.py --case qwen35_36_35b_tp1
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import (
    DEFAULT_BATCHES, add_shape_arguments, resolve_shape_arguments,
    parse_batches, print_cases,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_shape_arguments(p)
    p.add_argument("--batches", default=DEFAULT_BATCHES)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--calls-per-graph", type=int, default=16)
    p.add_argument("--output", default="fp8_blockwise.json")
    args = p.parse_args(argv)
    if args.list_cases:
        print_cases()
        p.exit()
    resolve_shape_arguments(p, args)
    parse_batches(p, args.batches)
    if min(args.rounds, args.iters, args.calls_per_graph) < 1:
        p.error("rounds, iters and calls-per-graph must be positive")
    return args


# Resolve CLI-only requests before importing the GPU software stack.
_cli_args = parse_args() if __name__ == "__main__" else None

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
from fp8_reference import (
    make_weights, make_inputs, gate_reference, down_reference, ffn_reference, metrics,
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_e2e_inputs(batch, *, hidden=2048, experts=256, topk=8):
    x, ids, route = make_inputs(batch, hidden=hidden, experts=experts, topk=topk)
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


def sglang_reference(x, gu, dn, gs, ds, ids, route):
    """Use SGLang quantization with independent FP32 projections.

    SGLang's quantization rounding differs at FP8 halfway values, and its
    down GEMM rounds each weighted expert result to BF16 before combine.
    """
    q, xs = sglang_per_token_group_quant_fp8(x, 128)
    buf = gate_reference(q, gu, xs, gs, ids)
    q2, s2 = sglang_per_token_group_quant_fp8(buf, 128)
    return down_reference(
        q2, dn, s2, ds, ids, route, round_expert_output=True,
    ).bfloat16()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    args = _cli_args or parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    initialize_sglang()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gu, dn, gs, ds = make_weights(
        args.seed, hidden=args.hidden, inter=args.inter, experts=args.experts,
    )
    root = Path(__file__).resolve().parents[1]
    result = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        sglang=sglang.__version__,
        sgl_kernel=importlib.metadata.version("sglang-kernel"),
        triton=triton.__version__,
        seed=args.seed,
        case=args.case,
        shape=dict(
            hidden=args.hidden, inter=args.inter, experts=args.experts,
            topk=args.topk, tp=args.tp,
        ),
        layout="E4M3 W8A8; activation 1x128; weight 128x128; FP32 scales",
        protocol=dict(
            rounds=args.rounds,
            replays=args.iters,
            calls_per_graph=args.calls_per_graph,
            weights="fixed addresses; no rotation/cache flush",
            clocks="dynamic",
            graph_update_inputs="restored before timing",
            scope="FFN includes activation quantization and combine; excludes weight setup and router",
            accuracy="native/common reference <0.006; SGLang/own-quant reference <0.02; "
                     "SGLang/common-reference error and original 0.02 check also recorded; "
                     "timings require own-reference checks; failed checks excluded from cases",
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
                root / "benchmark/cases.py",
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
        validation_failures=[],
    )
    runner = MoeRunnerConfig(
        num_experts=args.experts, num_local_experts=args.experts, top_k=args.topk,
        inplace=False, gate_up_interleaved=False,
    )
    print("Stack:", result["torch"], result["sglang"], result["triton"], flush=True)

    def require_valid(case, names):
        failed = [
            name for name in names
            if not case["validation"][name]["passes_existing_threshold"]
        ]
        if failed:
            result["validation_failures"].append(dict(case, failed_checks=failed))
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            raise AssertionError({name: case["validation"][name] for name in failed})

    for b in batches:
        x, ids, route, topk = make_e2e_inputs(
            b, hidden=args.hidden, experts=args.experts, topk=args.topk,
        )
        native = make_e2e_wd(x, gu, dn, gs, ds, ids, route)
        sgl = make_e2e_sgl(x, gu, dn, gs, ds, topk, runner)
        expected = ffn_reference(x, gu, dn, gs, ds, ids, route)
        sgl_expected = sglang_reference(x, gu, dn, gs, ds, ids, route)
        native_out, sgl_out = native(), sgl()
        validation = {
            "warp_decode_fp8": metrics(native_out, expected, check=False),
            "sglang_fp8": metrics(sgl_out, sgl_expected, .02, check=False),
            "sglang_common_reference": metrics(sgl_out, expected, .02, check=False),
        }
        q, _ = wd.quantize_fp8_blockwise(x)
        sq, _ = sglang_per_token_group_quant_fp8(x, 128)
        validation["sglang_quant_byte_differences"] = (
            (q.view(torch.uint8) != sq.view(torch.uint8)).sum().item()
        )
        config, (down_config, _) = try_get_optimal_moe_config(
            gu.shape, dn.shape, args.topk, "fp8_w8a8", b,
            block_shape=[128, 128], return_down_config=True,
        )
        case = dict(batch=b, ids=ids.cpu().tolist(), config=config,
                    down_config=down_config, validation=validation)
        # Failed checks retain diagnostics separately from completed timings.
        require_valid(case, ("warp_decode_fp8", "sglang_fp8"))
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
        sgl_expected = sglang_reference(x, gu, dn, gs, ds, ids, route)
        native_out, sgl_out = native(), sgl()
        validation["timed_native"] = metrics(native_out, expected, check=False)
        validation["timed_sglang"] = metrics(sgl_out, sgl_expected, .02, check=False)
        validation["timed_sglang_common_reference"] = metrics(
            sgl_out, expected, .02, check=False,
        )
        require_valid(case, ("timed_native", "timed_sglang"))
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
        case.update(
            order=order,
            samples_ms=samples,
            median_ms=medians,
            speedup=medians["sglang_fp8"] / medians["warp_decode_fp8"],
        )
        result["cases"].append(case)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print("B", b, medians, "speedup", result["cases"][-1]["speedup"], flush=True)


if __name__ == "__main__":
    main()
