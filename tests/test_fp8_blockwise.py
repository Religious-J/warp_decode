import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batches', default='1,2,3,4,8,15,16,17,32,64')
    p.add_argument('--output', default=None)
    p.add_argument('--sanitizer', action='store_true')
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261012)
    quant_tests()
    gu, dn, gs, ds = make_weights()
    # Deliberately vary N, K, and expert scales to expose indexing mistakes.
    gs *= torch.logspace(-1, 1, 16, device='cuda')[None, None, :]
    gs *= torch.linspace(.7, 1.3, 8, device='cuda')[None, :, None]
    ds *= torch.logspace(-1, 1, 4, device='cuda')[None, None, :]
    ds *= torch.linspace(.7, 1.3, 16, device='cuda')[None, :, None]
    gs[255] *= 1.1
    ds[255] *= .9
    records = []
    for b in map(int, args.batches.split(',')):
        x, ids, route = make_inputs(b)
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
        records.append(dict(batch=b, gate=gate_error, down=down_error, ffn=full_error))
        print('PASS', records[-1], flush=True)
    x, ids, route = make_inputs(0)
    assert wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, ids, route).shape == (0, H)
    x, ids, route = make_inputs(1)
    xq, xs = wd.quantize_fp8_blockwise(x)
    misaligned = torch.empty(H + 1, device='cuda', dtype=torch.bfloat16)[1:].view(1,H)
    misaligned_q = torch.empty(H + 1, device='cuda', dtype=xq.dtype)[1:].view(1,H)
    invalid = [
        lambda: wd.moe_gate_up_3d_batched(xq, gu, gs, ids),
        lambda: wd.moe_gate_up_3d_batched(x[:, :1024], gu, gs, ids),
        lambda: wd.moe_gate_up_3d_batched(x, gu, gs[:, :, :8], ids),
        lambda: wd.moe_gate_up_3d_batched(x, gu, gs, ids.long()),
        lambda: wd.moe_gate_up_3d_batched(x, gu[:128], gs, ids),
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
    torch.cuda.synchronize()
    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2) + '\n')
    print('All FP8 blockwise tests passed', flush=True)


if __name__ == '__main__':
    main()
