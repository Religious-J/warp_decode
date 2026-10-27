import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from cases import (
    DEFAULT_BATCHES, add_shape_arguments, resolve_shape_arguments,
    parse_batches, print_cases,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    add_shape_arguments(p)
    p.add_argument('--batches', default=DEFAULT_BATCHES)
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--output', default=None)
    p.add_argument('--sanitizer', action='store_true')
    p.add_argument('--edges-only', action='store_true',
                   help='run only small shape/routing boundary tests')
    args = p.parse_args(argv)
    if args.list_cases:
        print_cases()
        p.exit()
    if args.edges_only and args.case:
        p.error("--edges-only tests generic boundaries; omit --case")
    resolve_shape_arguments(p, args)
    parse_batches(p, args.batches, allow_zero=True)
    return args


# --help, --list-cases and invalid arguments do not import Torch or build CUDA.
_cli_args = parse_args() if __name__ == "__main__" else None

import torch
from warp_decode import warp_decode as wd
from fp8_reference import (
    H, I, E, TOPK, quant_activation, gate_reference,
    down_reference, ffn_reference, metrics, make_weights, make_inputs,
)


def bits(t):
    return t.contiguous().view(torch.uint8)


def quant_tests():
    for rows, cols in [(0, 128), (3, 0), (1, 128), (3, 512), (4, 2048)]:
        x = torch.randn(rows, cols, device='cuda', dtype=torch.bfloat16)
        if rows and cols:
            x[0].zero_()
        q, scale = wd.quantize_fp8_blockwise(x)
        expected, es = quant_activation(x)
        torch.testing.assert_close(scale, es, rtol=1e-6, atol=0)
        assert torch.equal(bits(q), bits(expected)), (rows, cols)
    # Signed zero, tiny values and large activation ranges.
    x = torch.tensor([-1e6, -0., 0., 1e6, 1e-12, -1e-12, 1.0625, 1.1875],
                     device='cuda', dtype=torch.bfloat16).repeat(1, 16)
    q, scale = wd.quantize_fp8_blockwise(x)
    eq, es = quant_activation(x)
    assert torch.equal(bits(q), bits(eq))
    torch.testing.assert_close(scale, es, rtol=1e-6, atol=0)
    print('PASS quantization: independent reference, zeros, tiny/large values, RNE', flush=True)


def graph_test(fn, mutate):
    eager = fn().clone()
    for _ in range(10):
        assert torch.equal(bits(eager), bits(fn()))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fn()
    torch.cuda.synchronize()
    for _ in range(3):
        mutate()
        expected = fn()
        graph.replay()
        assert torch.equal(bits(captured), bits(expected))



def shape_edge_tests(seed):
    """Exercise generic shapes and routing boundaries with small allocations."""
    hidden, inter, experts = 128, 128, 32
    gu, dn, gs, ds = make_weights(
        seed, hidden=hidden, inter=inter, experts=experts,
    )
    records = []
    for topk in (1, 7, 10, 32):
        for batch in (1, 3):
            x, ids, route = make_inputs(
                batch, hidden=hidden, experts=experts, topk=topk,
            )
            actual = wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, ids, route)
            error = metrics(
                actual, ffn_reference(x, gu, dn, gs, ds, ids, route), .01,
            )
            # Check every token, both end slots, and both invalid-ID bounds.
            for token in range(batch):
                valid_tokens = torch.arange(batch, device=x.device) != token
                for slot in sorted({0, topk - 1}):
                    for bad in (-1, experts):
                        bad_ids = ids.clone()
                        bad_ids[token, slot] = bad
                        out = wd.moe_ffn_3d_batched(
                            x, gu, dn, gs, ds, bad_ids, route,
                        )
                        assert torch.isnan(out[token]).all(), (batch, topk, token, slot, bad)
                        assert torch.equal(bits(out[valid_tokens]), bits(actual[valid_tokens]))
            records.append(dict(batch=batch, topk=topk, ffn=error))
        x, ids, route = make_inputs(0, hidden=hidden, experts=experts, topk=topk)
        assert wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, ids, route).shape == (0, hidden)
        xq, xs = wd.quantize_fp8_blockwise(x)
        assert wd.moe_ffn_3d_batched(xq, gu, dn, gs, ds, ids, route, xs).shape == (0, hidden)

    def rejected(fn, message):
        try:
            fn()
        except RuntimeError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError('invalid input accepted: ' + message)

    x, ids, route = make_inputs(1, hidden=hidden, experts=experts, topk=1)
    # Individually valid weights/scales must still agree across projections.
    for shape in ((experts, 256, inter), (experts, hidden, 256), (experts - 1, hidden, inter)):
        other_dn = torch.empty(shape, device=x.device, dtype=dn.dtype)
        other_ds = torch.empty(
            (shape[0], shape[1] // 128, shape[2] // 128),
            device=x.device, dtype=ds.dtype,
        )
        rejected(
            lambda: wd.moe_ffn_3d_batched(x, gu, other_dn, gs, other_ds, ids, route),
            'same MoE',
        )
    for invalid_topk in (0, 33):
        invalid_ids = torch.zeros(1, invalid_topk, device=x.device, dtype=torch.int32)
        invalid_route = torch.zeros(1, invalid_topk, device=x.device)
        rejected(
            lambda: wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, invalid_ids, invalid_route),
            '1 <= topk <= 32',
        )
    # Keep top-k within the kernel limit, but exceed this smaller expert count.
    invalid_ids = torch.zeros(1, 17, device=x.device, dtype=torch.int32)
    invalid_route = torch.zeros(1, 17, device=x.device)
    rejected(
        lambda: wd.moe_ffn_3d_batched(
            x, gu[:16], dn[:16], gs[:16], ds[:16], invalid_ids, invalid_route,
        ),
        'topk must not exceed',
    )
    print('PASS shape edges: topk=1/7/10/32, split/serial, empty, invalid IDs and mixed shapes', flush=True)
    return records


def main():
    args = _cli_args or parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    H, I, E, TOPK = args.hidden, args.inter, args.experts, args.topk
    assert H > 0 and I > 0 and H % 128 == 0 and I % 128 == 0
    assert 1 <= TOPK <= min(E, 32)
    quant_tests()
    edge_records = shape_edge_tests(args.seed)
    if args.edges_only:
        torch.cuda.synchronize()
        if args.output:
            Path(args.output).write_text(json.dumps(edge_records, indent=2) + '\n')
        print('All FP8 shape edge tests passed', flush=True)
        return
    gu, dn, gs, ds = make_weights(args.seed, hidden=H, inter=I, experts=E)
    inputs = lambda b: make_inputs(b, hidden=H, experts=E, topk=TOPK)
    # Deliberately vary N, K, and expert scales to expose indexing mistakes.
    gs *= torch.logspace(-1, 1, H // 128, device='cuda')[None, None, :]
    gs *= torch.linspace(.7, 1.3, 2 * I // 128, device='cuda')[None, :, None]
    ds *= torch.logspace(-1, 1, I // 128, device='cuda')[None, None, :]
    ds *= torch.linspace(.7, 1.3, H // 128, device='cuda')[None, :, None]
    gs[-1] *= 1.1
    ds[-1] *= .9
    records = []
    for b in map(int, args.batches.split(',')):
        if b == 0:  # The empty batch is checked separately below.
            continue
        if b < 0:
            raise ValueError('batch must be non-negative')
        x, ids, route = inputs(b)
        ids[0, 0], ids[-1, -1] = 0, E - 1
        xq, xs = wd.quantize_fp8_blockwise(x)
        ref_gate = gate_reference(xq, gu, xs, gs, ids)
        buf = wd.moe_gate_up_3d_batched(x, gu, gs, ids)
        gate_error = metrics(buf, ref_gate, .004)
        assert torch.equal(bits(buf), bits(wd.moe_gate_up_3d_batched(xq, gu, gs, ids, xs)))
        bq, bs = wd.quantize_fp8_blockwise(buf)
        out = wd.moe_down_3d_batched(buf, dn, ds, ids, route)
        down_error = metrics(out, down_reference(bq, dn, bs, ds, ids, route), .004)
        assert torch.equal(bits(out), bits(wd.moe_down_3d_batched(bq, dn, ds, ids, route, bs)))
        assert torch.equal(bits(out), bits(wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, ids, route)))
        full_error = metrics(out, ffn_reference(x, gu, dn, gs, ds, ids, route), .01)
        for zero in (False, True):
            z = torch.zeros_like(x) if zero else x
            rr = route if zero else torch.zeros_like(route)
            assert torch.count_nonzero(wd.moe_ffn_3d_batched(z, gu, dn, gs, ds, ids, rr)) == 0
        if not args.sanitizer:
            def mutate():
                x.mul_(-1.03)
                ids.copy_(ids.roll(1, -1))
                route.copy_(route.roll(1, -1))
                gs.mul_(1.001)
                ds.mul_(.999)
            fn = lambda: wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, ids, route)
            graph_test(fn, mutate)
            metrics(fn(), ffn_reference(x, gu, dn, gs, ds, ids, route), .01)
            # Also replay a prequantized input with its scales updated in place.
            def mutate_fp8():
                mutate()
                qq, ss = wd.quantize_fp8_blockwise(x)
                xq.copy_(qq); xs.copy_(ss)
            graph_test(lambda: wd.moe_ffn_3d_batched(xq, gu, dn, gs, ds, ids, route, xs), mutate_fp8)
        records.append(dict(
            case=args.case,
            shape=dict(hidden=H, inter=I, experts=E, topk=TOPK, tp=args.tp),
            batch=b, gate=gate_error, down=down_error, ffn=full_error,
        ))
        print('PASS', records[-1], flush=True)
    x, ids, route = inputs(0)
    assert wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, ids, route).shape == (0, H)
    x, ids, route = inputs(1)
    xq, xs = wd.quantize_fp8_blockwise(x)
    misaligned = torch.empty(H + 1, device='cuda', dtype=torch.bfloat16)[1:].view(1,H)
    misaligned_q = torch.empty(H + 1, device='cuda', dtype=xq.dtype)[1:].view(1,H)
    invalid = [
        lambda: wd.moe_gate_up_3d_batched(xq, gu, gs, ids),
        lambda: wd.moe_gate_up_3d_batched(x[:, :H - 128], gu, gs, ids),
        lambda: wd.moe_gate_up_3d_batched(x, gu, gs[:, :, :-1], ids),
        lambda: wd.moe_gate_up_3d_batched(x, gu, gs, ids.long()),
        lambda: wd.moe_gate_up_3d_batched(x, gu[:-1], gs, ids),
        lambda: wd.quantize_fp8_blockwise(x.float()),
        lambda: wd.quantize_fp8_blockwise(x[:, :127].contiguous()),
        lambda: wd.moe_gate_up_3d_batched(xq, gu, gs, ids, xs.cpu()),
        lambda: wd.moe_gate_up_3d_batched(x, gu, gs, ids, xs),
        lambda: wd.quantize_fp8_blockwise(misaligned),
        lambda: wd.moe_gate_up_3d_batched(misaligned_q, gu, gs, ids, xs),
    ]
    for fn in invalid:
        try:
            fn()
        except RuntimeError:
            pass
        else:
            raise AssertionError('invalid input accepted')
    # Invalid IDs must propagate NaNs without an out-of-bounds weight load.
    for bad in (-1, E):
        bad_ids = ids.clone()
        bad_ids[0, -1] = bad
        out = wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, bad_ids, route)
        assert torch.isnan(out).all(), bad
    # Repeated experts and signed, non-normalized routing remain well-defined.
    ids.fill_(E - 1)
    route = torch.linspace(-.5, 1, TOPK, device='cuda').reshape(1, TOPK)
    metrics(wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, ids, route),
            ffn_reference(x, gu, dn, gs, ds, ids, route), .01)
    torch.cuda.synchronize()
    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2) + '\n')
    print('All FP8 blockwise tests passed', flush=True)


if __name__ == '__main__':
    main()
