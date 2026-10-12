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
    result = torch.empty(ids.numel(), I, device=x.device, dtype=torch.bfloat16)
    # Bound temporary gather memory regardless of batch size.
    for start in range(0, ids.numel(), 8):
        flat = torch.arange(start, min(start + 8, ids.numel()), device=x.device)
        expert = ids.flatten()[flat].long()
        w = dequant_weight(gu[expert], gs[expert])
        dot = torch.bmm(w, x[flat // TOPK, :, None]).squeeze(-1).bfloat16().float()
        g, u = dot.chunk(2, -1)
        result[flat] = (F.silu(g) * u).bfloat16()
    return result


def down_reference(q, dn, xs, ds, ids, route):
    x = dequant_activation(q, xs)
    b = ids.shape[0]
    result = torch.zeros(b, H, device=q.device)
    for slot in range(TOPK):
        for start in range(0, b, 8):
            end = min(start + 8, b)
            expert = ids[start:end, slot].long()
            w = dequant_weight(dn[expert], ds[expert])
            dot = torch.bmm(w, x.reshape(b, TOPK, I)[start:end, slot, :, None]).squeeze(-1)
            result[start:end] += dot * route[start:end, slot, None]
    return result


def ffn_reference(x, gu, dn, gs, ds, ids, route):
    q, xs = quant_activation(x)
    buf = gate_reference(q, gu, xs, gs, ids)
    q2, s2 = quant_activation(buf)
    return down_reference(q2, dn, s2, ds, ids, route)


def metrics(actual, reference, limit=.006):
    a, r = actual.float(), reference.float()
    assert torch.isfinite(a).all()
    rms = r.square().mean().sqrt().item()
    error = a - r
    nrmse = error.square().mean().sqrt().item() / max(rms, 1e-20)
    maximum = error.abs().max().item()
    assert nrmse < limit, (nrmse, limit, maximum)
    assert maximum < max(.15 * rms, 1e-6), (maximum, rms)
    return dict(nrmse=nrmse, max_abs=maximum, reference_rms=rms)


def make_weights(seed=20261012):
    """Synthetic, already-quantized E4M3 weights with supplied block scales.

    No BF16 weight tensor or runtime weight quantizer is used. FP32 random
    values are cast only to construct the test's E4M3 bit patterns.
    """
    torch.manual_seed(seed)
    gu = (torch.randn(E, 2 * I, H, device='cuda') * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    dn = (torch.randn(E, H, I, device='cuda') * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    gs = torch.rand(E, 2 * I // GROUP, H // GROUP, device='cuda') * .0005 + .0005
    ds = torch.rand(E, H // GROUP, I // GROUP, device='cuda') * .0005 + .0005
    return gu, dn, gs, ds


def make_inputs(b):
    x = torch.randn(b, H, device='cuda', dtype=torch.bfloat16) * .1
    ids = torch.rand(b, E, device='cuda').topk(TOPK, -1).indices.int().contiguous()
    route = torch.rand(b, TOPK, device='cuda').softmax(-1)
    return x, ids, route
