"""Reference data and clamp-to-edge filtering shared by training and evaluation."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import math
import torch
import torch.nn.functional as F


def build_mip_chain(base_chw: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """Builds [mip0..mipN] from a base [C,H,W]."""
    if levels < 1:
        raise ValueError("levels must be positive")
    mips = [base_chw]
    cur = base_chw
    for _ in range(levels - 1):
        if cur.shape[1:] == (1, 1):
            break
        h = max(1, cur.shape[1] // 2)
        w = max(1, cur.shape[2] // 2)
        cur = F.interpolate(
            cur.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mips.append(cur)
    return mips


def sample_texture_chw(
    tex_chw: torch.Tensor, uv: torch.Tensor, mode: str
) -> torch.Tensor:
    """Samples [C,H,W] at uv [B,2] in [0,1], returns [B,C]."""
    if tex_chw.device.type == "mps":
        raise ValueError(
            "MPS does not support the required sampling contract; use CPU or CUDA/HIP"
        )
    sample_mode = mode
    padding_mode = "border"
    grid = uv.mul(2.0).sub(1.0).view(1, -1, 1, 2)
    out = F.grid_sample(
        tex_chw.unsqueeze(0),
        grid,
        mode=sample_mode,
        align_corners=False,
        padding_mode=padding_mode,
    )
    return out.squeeze(0).squeeze(-1).transpose(0, 1)


def sample_mips_trilinear(
    mips: Sequence[torch.Tensor],
    uv: torch.Tensor,
    lod: torch.Tensor,
    bilinear_mode: str,
) -> torch.Tensor:
    """Trilinear over discrete mip levels; returns [B,C]."""
    if not mips:
        raise ValueError("At least one mip is required")
    if len(mips) == 1:
        return sample_texture_chw(mips[0], uv, bilinear_mode)
    lod = lod.clamp(0.0, float(len(mips) - 1))
    l0 = lod.floor().long()
    l1 = (l0 + 1).clamp_max(len(mips) - 1)
    alpha = lod - l0
    out = uv.new_zeros((len(uv), mips[0].shape[0]))
    for level, mip in enumerate(mips):
        weight = (l0 == level) * (1 - alpha) + (l1 == level) * alpha
        selected = torch.nonzero(weight > 0, as_tuple=True)[0]
        if selected.numel():
            samples = sample_texture_chw(mip, uv[selected], bilinear_mode)
            out = out.index_add(0, selected, samples * weight[selected, None])
    return out


def random_uv_lod(
    batch: int, max_lod: float, device: torch.device, shared_scale: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    uv = torch.rand(batch, 2, device=device)
    lod = (torch.rand(1 if shared_scale else batch, device=device) * max_lod).expand(
        batch
    )
    return uv, lod


def load_reference_mips(
    path: Optional[Path], levels: int, out_channels: int, device: torch.device
) -> List[torch.Tensor]:
    """
    Expects either:
    - .pt file with {"base": [C,H,W]} OR {"mips": List[[C,H,W]]}
    - if path is None, creates synthetic material channels for smoke testing.
    """
    if path is None:
        h = w = 1024
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij"
        )
        base = []
        for c in range(out_channels):
            f = torch.sin((c + 1) * math.pi * xx) * torch.cos((c + 1) * math.pi * yy)
            base.append(f)
        base = torch.stack(base, dim=0).to(torch.float32)
        base[:3] = (base[:3] + 1) * 0.5
        return [m.to(device) for m in build_mip_chain(base, levels)]

    obj = torch.load(path, map_location="cpu", weights_only=True)
    if "mips" in obj:
        mips = [m.float() for m in obj["mips"][:levels]]
    elif "base" in obj:
        mips = build_mip_chain(obj["base"].float(), levels)
    else:
        raise ValueError("Reference .pt must contain key 'mips' or 'base'.")

    if not mips:
        raise ValueError("Reference mip chain is empty")
    if mips[0].shape[0] != out_channels:
        raise ValueError(
            f"Reference channels mismatch: got {mips[0].shape[0]}, expected {out_channels}."
        )
    for level, m in enumerate(mips):
        expected = max(1, mips[0].shape[-1] >> level)
        if m.shape[-2:] != (expected, expected):
            raise ValueError("Invalid reference mip dimensions")
        if m.ndim != 3 or m.shape[0] != out_channels or not torch.isfinite(m).all():
            raise ValueError(
                "Reference mips must be finite CHW tensors with consistent channels"
            )
        if m.shape[1] != m.shape[2]:
            raise ValueError(
                "Current model contract requires square reference textures"
            )
        if (m[:3] < 0).any() or (m[:3] > 1).any() or (m[3:].abs() > 1).any():
            raise ValueError(
                "Dataset channel range mismatch. Rebuild legacy signed-albedo data with python -m nnpbr.prepare"
            )
    return [m.to(device) for m in mips]
