"""Independent PyTorch reference for E4M3 W8A8, 1x128 / 128x128 scales."""
import torch
import torch.nn.functional as F

H, I, E, TOPK, GROUP = 2048, 512, 256, 8, 128


def quant_activation(x):
    blocks = x.float().reshape(x.shape[0], x.shape[1] // GROUP, GROUP)
    # Use FP64 division to independently check CUDA's correctly rounded
    # FP32 scale and reciprocal at E4M3 halfway values.
    scale = (blocks.abs().amax(-1).clamp_min(1e-10).double() / 448).float()
    inv = scale.double().reciprocal().float()
    q = (blocks * inv[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), scale


def dequant_activation(q, s):
    return q.float() * s.repeat_interleave(GROUP, -1)


def dequant_weight(q, s):
    return q.float() * s.repeat_interleave(GROUP, 1).repeat_interleave(GROUP, 2)


def gate_reference(xq, gu, xs, gs, ids):
    x = dequant_activation(xq, xs)
    inter, topk = gu.shape[1] // 2, ids.shape[1]
    result = torch.empty(ids.numel(), inter, device=x.device, dtype=torch.bfloat16)
    # Bound temporary gather memory regardless of batch size.
    for start in range(0, ids.numel(), 8):
        flat = torch.arange(start, min(start + 8, ids.numel()), device=x.device)
        expert = ids.flatten()[flat].long()
        w = dequant_weight(gu[expert], gs[expert])
        dot = torch.bmm(w, x[flat // topk, :, None]).squeeze(-1).bfloat16().float()
        g, u = dot.chunk(2, -1)
        result[flat] = (F.silu(g) * u).bfloat16()
    return result


def down_reference(q, dn, xs, ds, ids, route, *, round_expert_output=False):
    x = dequant_activation(q, xs)
    b, topk = ids.shape
    hidden, inter = dn.shape[1:]
    result = torch.zeros(b, hidden, device=q.device)
    for slot in range(topk):
        for start in range(0, b, 8):
            end = min(start + 8, b)
            expert = ids[start:end, slot].long()
            w = dequant_weight(dn[expert], ds[expert])
            dot = torch.bmm(w, x.reshape(b, topk, inter)[start:end, slot, :, None]).squeeze(-1)
            weighted = dot * route[start:end, slot, None]
            if round_expert_output:
                weighted = weighted.bfloat16().float()
            result[start:end] += weighted
    return result


def ffn_reference(x, gu, dn, gs, ds, ids, route):
    q, xs = quant_activation(x)
    buf = gate_reference(q, gu, xs, gs, ids)
    q2, s2 = quant_activation(buf)
    return down_reference(q2, dn, s2, ds, ids, route)


def metrics(actual, reference, limit=.006, *, check=True):
    a, r = actual.float(), reference.float()
    finite = bool(torch.isfinite(a).all())
    rms = r.square().mean().sqrt().item()
    error = a - r
    nrmse = error.square().mean().sqrt().item() / max(rms, 1e-20)
    maximum = error.abs().max().item()
    passed = finite and nrmse < limit and maximum < max(.15 * rms, 1e-6)
    if check:
        assert finite
        assert nrmse < limit, (nrmse, limit, maximum)
        assert maximum < max(.15 * rms, 1e-6), (maximum, rms)
    return dict(nrmse=nrmse, max_abs=maximum, reference_rms=rms,
                passes_existing_threshold=passed)


def make_weights(seed=20261012, *, hidden=H, inter=I, experts=E):
    """Synthetic, already-quantized E4M3 weights with supplied block scales.

    No BF16 weight tensor or runtime weight quantizer is used. FP32 random
    values are cast only to construct the test's E4M3 bit patterns.
    """
    torch.manual_seed(seed)
    gu = (torch.randn(experts, 2 * inter, hidden, device='cuda') * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    dn = (torch.randn(experts, hidden, inter, device='cuda') * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    gs = torch.rand(experts, 2 * inter // GROUP, hidden // GROUP, device='cuda') * .0005 + .0005
    ds = torch.rand(experts, hidden // GROUP, inter // GROUP, device='cuda') * .0005 + .0005
    return gu, dn, gs, ds


def make_inputs(b, *, hidden=H, experts=E, topk=TOPK):
    x = torch.randn(b, hidden, device='cuda', dtype=torch.bfloat16) * .1
    ids = torch.rand(b, experts, device='cuda').topk(topk, -1).indices.int().contiguous()
    route = torch.rand(b, topk, device='cuda').softmax(-1)
    return x, ids, route
