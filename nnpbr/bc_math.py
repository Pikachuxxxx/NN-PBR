"""Unsigned Mode 10 arithmetic: exact forward values and continuous training math."""

import torch
from bc6h_spec import BC6H_MODE10_WEIGHTS


def half_bits_to_float(y):
    # Exact for all finite nonnegative FP16 bit patterns, continuous between them.
    h = ((y - 1) / 1024).floor().sub(1).clamp_min(0)
    return torch.pow(2.0, h - 14) * (y / 1024 - h)


def unquantize6(e):
    e = e.to(torch.int64)
    middle = ((e << 16) + 32768) >> 6
    return torch.where(e == 0, 0, torch.where(e == 63, 65535, middle))


def decode_palette(endpoint_pairs):
    """[B,2,3] endpoint codes -> [B,8,3] exact unsigned BC6 palette."""
    u = unquantize6(endpoint_pairs)
    w = torch.tensor(BC6H_MODE10_WEIGHTS, device=u.device).view(1, 8, 1)
    interp = (u[:, 0:1] * (64 - w) + u[:, 1:2] * w + 32) >> 6
    bits = (interp * 31) >> 6
    return half_bits_to_float(bits.float())


def decode_blocks(endpoints_q, indices_q, mask):
    """Hardware labels: subset 0 owns endpoints 0/1; subset 1 owns 2/3."""
    p0 = decode_palette(endpoints_q[:, :2])
    p1 = decode_palette(endpoints_q[:, 2:])
    idx = indices_q.long().unsqueeze(-1).expand(-1, -1, 3)
    a = p0.gather(1, idx)
    b = p1.gather(1, idx)
    return torch.where(mask.bool().unsqueeze(-1), b, a)


def continuous_unquantize6(codes):
    # Piecewise linear extension THROUGH the exact endpoint table, including 0/63.
    lo = codes.floor().long().clamp(0, 63)
    hi = (lo + 1).clamp_max(63)
    a = unquantize6(lo).float()
    b = unquantize6(hi).float()
    return a + (b - a) * (codes - lo)
