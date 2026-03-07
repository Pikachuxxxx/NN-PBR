#!/usr/bin/env python3
"""
Core training infrastructure for neural material compression.

Pipeline:
1) Warmup phase (unconstrained latent mip pyramids)
2) BC-constrained phase (differentiable BC6-style block parameterization + mip filtering)
3) Quantized finetune phase (freeze BC features, tune decoder on quantized path)

Notes:
- Training uses a BC6-inspired surrogate; it is not a bit-exact BC6H codec.
- export_trained_artifacts() packs block params directly to BC6H Mode 12 DDS —
  no float32 decode/re-encode roundtrip.  PNG previews are kept for visual inspection.
- Export layout v4: runtime files in root (latent_XX.bc6.dds, decoder_fp16.bin,
  metadata.json); debug files in metadata/ (decoder_state.pt, latent_XX_mip_YY.png).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None


# -------------------------------
# Utilities
# -------------------------------

EPS = 1e-8


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def quantize_ste(x: torch.Tensor, bits: int, min_v: float, max_v: float) -> torch.Tensor:
    if bits <= 0:
        return x
    levels = (1 << bits) - 1
    x = x.clamp(min_v, max_v)
    xn = (x - min_v) / (max_v - min_v + EPS)
    qn = ste_round(xn * levels) / levels
    return qn * (max_v - min_v) + min_v


def build_mip_chain(base_chw: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """Builds [mip0..mipN] from a base [C,H,W]."""
    mips = [base_chw]
    cur = base_chw
    for _ in range(levels - 1):
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


def sample_texture_chw(tex_chw: torch.Tensor, uv: torch.Tensor, mode: str) -> torch.Tensor:
    """Samples [C,H,W] at uv [B,2] in [0,1], returns [B,C]."""
    sample_mode = mode
    if tex_chw.device.type == "mps":
        # MPS backend does not support border padding in grid_sample.
        uv = uv.clamp(1e-4, 1.0 - 1e-4)
        padding_mode = "zeros"
        if sample_mode == "bicubic":
            sample_mode = "bilinear"
    else:
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
    mips: Sequence[torch.Tensor], uv: torch.Tensor, lod: torch.Tensor, bilinear_mode: str
) -> torch.Tensor:
    """Trilinear over discrete mip levels; returns [B,C]."""
    max_lod = float(len(mips) - 1)
    lod = lod.clamp(0.0, max_lod - 1e-4)
    l0 = torch.floor(lod).long()
    l1 = torch.clamp(l0 + 1, max=len(mips) - 1)
    a = (lod - l0.float()).unsqueeze(1)

    all_samples = [sample_texture_chw(m, uv, mode=bilinear_mode) for m in mips]
    stack = torch.stack(all_samples, dim=0)  # [L,B,C]
    bidx = torch.arange(uv.shape[0], device=uv.device)
    v0 = stack[l0, bidx]
    v1 = stack[l1, bidx]
    return (1.0 - a) * v0 + a * v1


def random_uv_lod(batch: int, max_lod: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    uv = torch.rand(batch, 2, device=device)
    lod = torch.rand(batch, device=device) * max_lod
    return uv, lod


def make_partition_bank(device: torch.device) -> torch.Tensor:
    """
    Procedurally builds 32 binary 4x4 masks [32,16].
    This mirrors the idea of choosing among fixed partition patterns.
    """
    yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).float()  # [16,2]
    candidates = []
    seen = set()

    params = []
    for a in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
        for b in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
            for c in (-3.0, -1.5, 0.0, 1.5, 3.0):
                params.append((a, b, c))

    for a, b, c in params:
        m = ((a * pts[:, 0] + b * pts[:, 1] + c) > 0.0).float()
        s = int(m.sum().item())
        if s == 0 or s == 16:
            continue
        key = tuple(m.int().tolist())
        inv = tuple((1 - m).int().tolist())
        if key in seen or inv in seen:
            continue
        seen.add(key)
        candidates.append(m)
        if len(candidates) == 32:
            break

    # Fallback deterministic masks if needed.
    seed = 17
    while len(candidates) < 32:
        mask = torch.zeros(16)
        for i in range(16):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            mask[i] = 1.0 if (seed % 7) < 3 else 0.0
        if int(mask.sum().item()) in (0, 16):
            continue
        key = tuple(mask.int().tolist())
        inv = tuple((1 - mask).int().tolist())
        if key in seen or inv in seen:
            continue
        seen.add(key)
        candidates.append(mask)

    return torch.stack(candidates[:32], dim=0).to(device)


# -------------------------------
# Latent structures
# -------------------------------

class WarmupLatentPyramid(nn.Module):
    def __init__(self, base_res: int, num_mips: int):
        super().__init__()
        self.base_res = base_res
        self.num_mips = num_mips
        mips = []
        for i in range(num_mips):
            sz = max(1, base_res >> i)
            # Start small so warmup remains stable.
            p = nn.Parameter(torch.randn(3, sz, sz) * 0.05)
            mips.append(p)
        self.mips = nn.ParameterList(mips)

    def sample(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        mips = [torch.tanh(m) for m in self.mips]  # keep latent range bounded
        return sample_mips_trilinear(mips, uv, lod, bilinear_mode="bilinear")

    def decode_mips(self) -> List[torch.Tensor]:
        return [torch.tanh(m) for m in self.mips]


class BC6SurrogateBlockLevel(nn.Module):
    """
    BC6-style block parameterization for one mip level:
    - per 4x4 block: 4 endpoints (RGB), 16 scalar indices, 1 partition-id (soft/hard over 32 masks)
    """

    def __init__(
        self,
        h: int,
        w: int,
        partition_bank: torch.Tensor,
        endpoint_bits: int = 6,
        index_bits: int = 3,
        bc6_signed_mode: bool = False,
    ):
        super().__init__()
        assert h % 4 == 0 and w % 4 == 0, f"BC blocks require multiples of 4, got {h}x{w}"
        self.h = h
        self.w = w
        self.by = h // 4
        self.bx = w // 4
        self.nb = self.by * self.bx
        self.endpoint_bits = endpoint_bits
        self.index_bits = index_bits
        self.bc6_signed_mode = bc6_signed_mode

        self.register_buffer("partition_bank", partition_bank)  # [32,16]
        self.endpoints = nn.Parameter(torch.randn(self.nb, 4, 3) * 0.1)
        self.indices = nn.Parameter(torch.zeros(self.nb, 16))
        self.partition_logits = nn.Parameter(torch.zeros(self.nb, 32))

    @torch.no_grad()
    def init_from_unconstrained(self, mip_chw: torch.Tensor):
        """
        Paper-like warm-start from unconstrained mip:
        choose best fixed partition per block and initialize endpoints/indices
        from per-partition line segments.
        """
        x = mip_chw.detach().clamp(-1.0, 1.0)
        # [3,H,W] -> [NB,16,3]
        blocks = (
            x.unfold(1, 4, 4)
            .unfold(2, 4, 4)
            .permute(1, 2, 3, 4, 0)
            .contiguous()
            .view(self.nb, 16, 3)
        )
        partition_bank = self.partition_bank.to(blocks.dtype)
        n_part = int(partition_bank.shape[0])

        best_err = torch.full((self.nb,), float("inf"), device=blocks.device, dtype=blocks.dtype)
        best_part = torch.zeros((self.nb,), device=blocks.device, dtype=torch.long)
        best_endpoints = torch.zeros((self.nb, 4, 3), device=blocks.device, dtype=blocks.dtype)
        best_indices = torch.zeros((self.nb, 16), device=blocks.device, dtype=blocks.dtype)

        # Chunking keeps memory bounded at higher resolutions.
        chunk = 4096
        huge = 1e6

        for s in range(0, self.nb, chunk):
            e = min(self.nb, s + chunk)
            blk = blocks[s:e]  # [B,16,3]
            bsz = blk.shape[0]

            chunk_best_err = torch.full((bsz,), float("inf"), device=blk.device, dtype=blk.dtype)
            chunk_best_part = torch.zeros((bsz,), device=blk.device, dtype=torch.long)
            chunk_best_ep = torch.zeros((bsz, 4, 3), device=blk.device, dtype=blk.dtype)
            chunk_best_idx = torch.zeros((bsz, 16), device=blk.device, dtype=blk.dtype)

            for pid in range(n_part):
                mask = partition_bank[pid].view(1, 16, 1)      # [1,16,1]
                inv_mask = 1.0 - mask

                g1_min = (blk + inv_mask * huge).amin(dim=1)
                g1_max = (blk - inv_mask * huge).amax(dim=1)
                g2_min = (blk + mask * huge).amin(dim=1)
                g2_max = (blk - mask * huge).amax(dim=1)

                d1 = g1_max - g1_min
                d2 = g2_max - g2_min
                d1_den = (d1 * d1).sum(dim=1, keepdim=True) + EPS
                d2_den = (d2 * d2).sum(dim=1, keepdim=True) + EPS

                t1 = ((blk - g1_min.unsqueeze(1)) * d1.unsqueeze(1)).sum(dim=2) / d1_den
                t2 = ((blk - g2_min.unsqueeze(1)) * d2.unsqueeze(1)).sum(dim=2) / d2_den
                t1 = t1.clamp(0.0, 1.0)
                t2 = t2.clamp(0.0, 1.0)

                rec1 = g1_min.unsqueeze(1) + t1.unsqueeze(-1) * d1.unsqueeze(1)
                rec2 = g2_min.unsqueeze(1) + t2.unsqueeze(-1) * d2.unsqueeze(1)
                rec = mask * rec1 + inv_mask * rec2

                err = (rec - blk).pow(2).mean(dim=(1, 2))
                t_all = mask.squeeze(-1) * t1 + inv_mask.squeeze(-1) * t2
                ep_all = torch.stack([g1_min, g1_max, g2_min, g2_max], dim=1)

                better = err < chunk_best_err
                chunk_best_err = torch.where(better, err, chunk_best_err)
                chunk_best_part = torch.where(better, torch.full_like(chunk_best_part, pid), chunk_best_part)
                chunk_best_ep = torch.where(better.view(-1, 1, 1), ep_all, chunk_best_ep)
                chunk_best_idx = torch.where(better.view(-1, 1), t_all, chunk_best_idx)

            best_err[s:e] = chunk_best_err
            best_part[s:e] = chunk_best_part
            best_endpoints[s:e] = chunk_best_ep
            best_indices[s:e] = chunk_best_idx

        # Set fixed partition IDs from best warmup fit.
        self.partition_logits.zero_()
        self.partition_logits.scatter_(1, best_part.view(-1, 1), 10.0)

        # Initialize index params in logit space.
        t = best_indices.clamp(1e-4, 1.0 - 1e-4)
        self.indices.copy_(torch.log(t / (1.0 - t)))

        # Initialize endpoint params in the same raw space expected by decode path.
        if self.bc6_signed_mode:
            ep = best_endpoints.clamp(-0.999, 0.999)
            self.endpoints.copy_(torch.atanh(ep))
        else:
            ep_n = ((best_endpoints + 1.0) * 0.5).clamp(1e-4, 1.0 - 1e-4)
            self.endpoints.copy_(torch.log(ep_n / (1.0 - ep_n)))

    def _partition_mask(self, hard_partition: bool) -> torch.Tensor:
        # Partition choice (soft or hard) over fixed masks.
        if hard_partition:
            part = torch.argmax(self.partition_logits, dim=-1)
            p = F.one_hot(part, num_classes=32).to(self.partition_logits.dtype)
        else:
            p = F.softmax(self.partition_logits, dim=-1)  # [NB,32]
        mask = p @ self.partition_bank  # [NB,16]
        return mask.unsqueeze(-1)  # [NB,16,1]

    def _decode_soft_blocks(self, hard_partition: bool) -> torch.Tensor:
        """
        Paper-style BC6 surrogate following Eq.7/Eq.8/Eq.9 structure:
        - unquantize endpoints
        - partitioned interpolation
        - nonlinear half reinterpret approximation
        """
        b = int(self.endpoint_bits)
        q = int(self.index_bits)

        # Endpoint quantized integers e_i.
        if self.bc6_signed_mode:
            # Approximate signed mode by mapping [-1,1] to integer range [0, 2^b-1].
            e_n = (torch.tanh(self.endpoints) + 1.0) * 0.5
            a = 31.0 / 32.0
        else:
            e_n = torch.sigmoid(self.endpoints)
            a = 31.0 / 64.0

        e_q = ste_round(e_n * ((1 << b) - 1))

        # Eq.7: endpoint unquantization.
        e_u = (a * (2.0**16) * e_q + (2.0**15)) / (2.0**b)  # [NB,4,3]

        # Quantized indices in [0,1].
        x_n = torch.sigmoid(self.indices)
        x_q = ste_round(x_n * ((1 << q) - 1))
        x_scaled = (2.0**q) * (x_q / max(1.0, float((1 << q) - 1)))  # [NB,16]

        mask = self._partition_mask(hard_partition)

        # Eq.8-style interpolation under partition mask.
        y1 = e_u[:, 0:1, :] + x_scaled.unsqueeze(-1) * (e_u[:, 1:2, :] - e_u[:, 0:1, :])
        y2 = e_u[:, 2:3, :] + x_scaled.unsqueeze(-1) * (e_u[:, 3:4, :] - e_u[:, 2:3, :])
        y = mask * y1 + (1.0 - mask) * y2  # [NB,16,3]

        # Eq.9 nonlinear reinterpret surrogate.
        y = torch.clamp(y, 0.0, 65535.0)
        h = torch.clamp(torch.floor((y - 1.0) / 1024.0) - 1.0, 0.0, 31.0)
        pow2 = torch.pow(torch.tensor(2.0, device=y.device, dtype=y.dtype), h - 14.0)
        pow2 = torch.clamp(pow2, 0.0, 1e4)
        w = pow2 * ((y / 1024.0) - h)

        # Bound decoded values to latent range expected by decoder.
        return torch.tanh(w * 0.25)

    def decode_mip(self, hard_partition: bool) -> torch.Tensor:
        blocks = self._decode_soft_blocks(hard_partition=hard_partition)
        mip = (
            blocks.view(self.by, self.bx, 4, 4, 3)
            .permute(4, 0, 2, 1, 3)
            .contiguous()
            .view(3, self.h, self.w)
        )
        return mip

    @torch.no_grad()
    def export_quantized_block_params(self) -> dict:
        b = int(self.endpoint_bits)
        q = int(self.index_bits)
        if self.bc6_signed_mode:
            e_n = (torch.tanh(self.endpoints) + 1.0) * 0.5
        else:
            e_n = torch.sigmoid(self.endpoints)
        e_q = torch.round(e_n * ((1 << b) - 1)).to(torch.int16)
        x_q = torch.round(torch.sigmoid(self.indices) * ((1 << q) - 1)).to(torch.uint8)
        p_id = torch.argmax(self.partition_logits, dim=-1).to(torch.uint8)
        return {
            "h": self.h,
            "w": self.w,
            "endpoint_bits": b,
            "index_bits": q,
            "signed_mode": bool(self.bc6_signed_mode),
            "endpoints_q": e_q.cpu(),
            "indices_q": x_q.cpu(),
            "partition_id": p_id.cpu(),
        }

    @torch.no_grad()
    def fix_partition_ids(self):
        part = torch.argmax(self.partition_logits, dim=-1, keepdim=True)
        self.partition_logits.zero_()
        self.partition_logits.scatter_(1, part, 10.0)
        self.partition_logits.requires_grad_(False)

    @torch.no_grad()
    def quantize_inplace(self):
        b = int(self.endpoint_bits)
        q = int(self.index_bits)

        if self.bc6_signed_mode:
            e_n = (torch.tanh(self.endpoints) + 1.0) * 0.5
            e_qn = torch.round(e_n * ((1 << b) - 1)) / ((1 << b) - 1)
            e_signed = (e_qn * 2.0 - 1.0).clamp(-0.999, 0.999)
            self.endpoints.copy_(torch.atanh(e_signed))
        else:
            e_n = torch.sigmoid(self.endpoints)
            e_qn = torch.round(e_n * ((1 << b) - 1)) / ((1 << b) - 1)
            e_qn = e_qn.clamp(1e-4, 1.0 - 1e-4)
            self.endpoints.copy_(torch.log(e_qn / (1.0 - e_qn)))

        x_n = torch.sigmoid(self.indices)
        x_qn = torch.round(x_n * ((1 << q) - 1)) / ((1 << q) - 1)
        x_qn = x_qn.clamp(1e-4, 1.0 - 1e-4)
        self.indices.copy_(torch.log(x_qn / (1.0 - x_qn)))


class BC6SurrogatePyramid(nn.Module):
    def __init__(
        self,
        base_res: int,
        num_mips: int,
        partition_bank: torch.Tensor,
        endpoint_bits: int,
        index_bits: int,
        bc6_signed_mode: bool,
    ):
        super().__init__()
        self.base_res = base_res
        self.num_mips = num_mips
        self.mips = nn.ModuleList()
        for i in range(num_mips):
            sz = max(4, base_res >> i)
            sz = (sz // 4) * 4  # BC-block aligned
            self.mips.append(
                BC6SurrogateBlockLevel(
                    h=sz,
                    w=sz,
                    partition_bank=partition_bank,
                    endpoint_bits=endpoint_bits,
                    index_bits=index_bits,
                    bc6_signed_mode=bc6_signed_mode,
                )
            )

    @torch.no_grad()
    def init_from_unconstrained(self, unconstrained_mips: Sequence[torch.Tensor]):
        for bc_mip, unc_mip in zip(self.mips, unconstrained_mips):
            # Resize in case dimensions mismatch due to 4x4 alignment.
            src = unc_mip
            if src.shape[1] != bc_mip.h or src.shape[2] != bc_mip.w:
                src = F.interpolate(
                    src.unsqueeze(0),
                    size=(bc_mip.h, bc_mip.w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            bc_mip.init_from_unconstrained(src)

    def decode_mips(self, hard_partition: bool) -> List[torch.Tensor]:
        return [
            m.decode_mip(hard_partition=hard_partition)
            for m in self.mips
        ]

    def sample(self, uv: torch.Tensor, lod: torch.Tensor, hard_partition: bool) -> torch.Tensor:
        mips = self.decode_mips(hard_partition=hard_partition)
        return sample_mips_trilinear(mips, uv, lod, bilinear_mode="bilinear")

    @torch.no_grad()
    def export_quantized_params(self) -> List[dict]:
        return [m.export_quantized_block_params() for m in self.mips]

    @torch.no_grad()
    def fix_partition_ids(self):
        for m in self.mips:
            m.fix_partition_ids()

    @torch.no_grad()
    def quantize_inplace(self):
        for m in self.mips:
            m.quantize_inplace()


# -------------------------------
# Full model
# -------------------------------

class MaterialDecoderMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class NeuralMaterialCompressionModel(nn.Module):
    def __init__(
        self,
        latent_resolutions: Sequence[int],
        latent_mips: Sequence[int],
        out_channels: int,
        hidden_dim: int,
        partition_bank: torch.Tensor,
        endpoint_bits: int,
        index_bits: int,
        ref_base_res: int,
        bc6_signed_mode: bool = False,
    ):
        super().__init__()
        assert len(latent_resolutions) == len(latent_mips)
        self.n_latent = len(latent_resolutions)
        self.latent_resolutions = list(latent_resolutions)
        self.ref_base_res = ref_base_res
        self.bc6_signed_mode = bc6_signed_mode
        self.partition_bank = partition_bank  # Store for export

        self.warmup_pyramids = nn.ModuleList(
            [WarmupLatentPyramid(r, m) for r, m in zip(latent_resolutions, latent_mips)]
        )
        self.bc_pyramids = nn.ModuleList(
            [
                BC6SurrogatePyramid(
                    base_res=r,
                    num_mips=m,
                    partition_bank=partition_bank,
                    endpoint_bits=endpoint_bits,
                    index_bits=index_bits,
                    bc6_signed_mode=bc6_signed_mode,
                )
                for r, m in zip(latent_resolutions, latent_mips)
            ]
        )

        self.decoder = MaterialDecoderMLP(in_dim=self.n_latent * 3, hidden=hidden_dim, out_dim=out_channels)

        # lod bias b_i = log2(max(h_i/h, w_i/w))
        self.lod_biases = [math.log2(max(r / ref_base_res, r / ref_base_res)) for r in latent_resolutions]
        self.freeze_bc_features = False

    @torch.no_grad()
    def initialize_bc_from_warmup(self):
        for p_bc, p_w in zip(self.bc_pyramids, self.warmup_pyramids):
            p_bc.init_from_unconstrained(p_w.decode_mips())
            p_bc.fix_partition_ids()

    def set_freeze_bc_features(self, enabled: bool):
        self.freeze_bc_features = enabled

    def warmup_parameters(self):
        for p in self.warmup_pyramids.parameters():
            yield p

    def bc_feature_parameters(self):
        for p in self.bc_pyramids.parameters():
            yield p

    def decoder_parameters(self):
        yield from self.decoder.parameters()

    def _collect_latents_warmup(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        feats = []
        for pyr, b in zip(self.warmup_pyramids, self.lod_biases):
            li = lod + b
            feats.append(pyr.sample(uv, li))
        return torch.cat(feats, dim=1)

    def _collect_latents_bc(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        feats = []
        hard_partition = True
        for pyr, b in zip(self.bc_pyramids, self.lod_biases):
            li = lod + b
            feats.append(pyr.sample(uv=uv, lod=li, hard_partition=hard_partition))
        return torch.cat(feats, dim=1)

    def forward_warmup(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        x = self._collect_latents_warmup(uv, lod)
        return self.decoder(x)

    def forward_bc(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        x = self._collect_latents_bc(uv, lod)
        return self.decoder(x)

    @torch.no_grad()
    def quantize_and_freeze_bc_features(self):
        for pyr in self.bc_pyramids:
            pyr.quantize_inplace()
        for p in self.bc_feature_parameters():
            p.requires_grad_(False)
        self.set_freeze_bc_features(True)


# Backward-compatible aliases for existing imports/scripts.
UnconstrainedLatentPyramid = WarmupLatentPyramid
BCBlockMip = BC6SurrogateBlockLevel
BCBlockPyramid = BC6SurrogatePyramid
DecoderMLP = MaterialDecoderMLP
NeuralMaterialModel = NeuralMaterialCompressionModel


# -------------------------------
# Training config and loop
# -------------------------------

@dataclass
class TrainConfig:
    device: str = "cuda"
    batch_size: int = 8192
    phase1_iters: int = 5_000
    phase2_iters: int = 200_000
    phase3_iters: int = 0

    lr_feat_phase1: float = 5e-2
    lr_mlp_phase1: float = 1e-3
    gamma_phase1: float = 0.9995

    lr_feat_phase2: float = 1e-2
    lr_mlp_phase2: float = 1e-3
    gamma_phase2: float = 0.9999

    lr_mlp_phase3: float = 5e-4

    log_every: int = 200
    interactive_progress: bool = False


def load_reference_mips(path: Optional[Path], levels: int, out_channels: int, device: torch.device) -> List[torch.Tensor]:
    """
    Expects either:
    - .pt file with {"base": [C,H,W]} OR {"mips": List[[C,H,W]]}
    - if path is None, creates synthetic material channels for smoke testing.
    """
    if path is None:
        h = w = 1024
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        base = []
        for c in range(out_channels):
            f = torch.sin((c + 1) * math.pi * xx) * torch.cos((c + 1) * math.pi * yy)
            base.append(f)
        base = torch.stack(base, dim=0).to(torch.float32)
        return [m.to(device) for m in build_mip_chain(base, levels)]

    obj = torch.load(path, map_location="cpu")
    if "mips" in obj:
        mips = [m.float() for m in obj["mips"]]
    elif "base" in obj:
        mips = build_mip_chain(obj["base"].float(), levels)
    else:
        raise ValueError("Reference .pt must contain key 'mips' or 'base'.")

    if mips[0].shape[0] != out_channels:
        raise ValueError(f"Reference channels mismatch: got {mips[0].shape[0]}, expected {out_channels}.")
    return [m.to(device) for m in mips]


def save_chw_png_ldr(t: torch.Tensor, out_path: Path, signed_mode: bool = False):
    """Save CHW tensor to PNG LDR.

    For unsigned BC6 mode: latents are in [0, 1], save directly as uint8 [0, 255].
    For signed BC6 mode: latents are in [-1, 1], convert to [0, 1] then uint8 [0, 255].
    """
    if Image is None or np is None:
        raise RuntimeError("Pillow and numpy are required to export PNG previews.")
    x = t.detach().to("cpu")
    if x.shape[0] < 3:
        x = x.repeat(3, 1, 1)[:3]
    x = x[:3]

    if signed_mode:
        # Signed mode: latents in [-1, 1], convert to [0, 1] then uint8
        x = x.clamp(-1.0, 1.0)
        x = ((x + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
    else:
        # Unsigned mode: latents already in [0, 1], just convert to uint8
        x = x.clamp(0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8)

    img = x.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(img, mode="RGB").save(out_path)


def _pack_fields_to_fixed_block(fields: Sequence[Tuple[int, int]], total_bits: int = 128) -> bytes:
    """
    Packs integer fields into a fixed-size little-endian bitstream.
    Fields are appended in order, least-significant-bit first.
    """
    acc = 0
    offset = 0
    for value, nbits in fields:
        if nbits <= 0:
            continue
        mask = (1 << nbits) - 1
        acc |= (int(value) & mask) << offset
        offset += nbits
    if offset > total_bits:
        raise ValueError(f"bit overflow: used {offset} bits > {total_bits}")
    total_bytes = (total_bits + 7) // 8
    return int(acc).to_bytes(total_bytes, byteorder="little", signed=False)


def pack_quantized_blocks_to_128b(qp: dict) -> bytes:
    """
    Packs one mip's quantized block params into 128-bit records per block.
    Layout (default b=6, q=3):
    - partition_id: 5 bits
    - endpoints_q: 4*3*b bits
    - indices_q: 16*q bits
    - padding: remaining bits to 128

    This is a compact BC6-like custom blob, not guaranteed to be BC6 bitstream compliant.
    """
    b = int(qp["endpoint_bits"])
    q = int(qp["index_bits"])
    endpoints = qp["endpoints_q"].to(torch.int64)  # [NB,4,3]
    indices = qp["indices_q"].to(torch.int64)      # [NB,16]
    part = qp["partition_id"].to(torch.int64)      # [NB]

    nb = int(part.shape[0])
    out = bytearray()
    for i in range(nb):
        fields: List[Tuple[int, int]] = []
        fields.append((int(part[i].item()), 5))
        for e in endpoints[i].reshape(-1):
            fields.append((int(e.item()), b))
        for x in indices[i].reshape(-1):
            fields.append((int(x.item()), q))
        out.extend(_pack_fields_to_fixed_block(fields, total_bits=128))
    return bytes(out)


def unpack_quantized_blocks_from_128b(
    data: bytes, endpoint_bits: int, index_bits: int, num_blocks: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpacks fixed 128-bit custom block records into:
    - partition ids [NB]
    - endpoints_q [NB,4,3]
    - indices_q [NB,16]
    """
    rec_size = 16
    expected = num_blocks * rec_size
    if len(data) != expected:
        raise ValueError(f"Packed size mismatch: got {len(data)} expected {expected}")

    parts = torch.empty((num_blocks,), dtype=torch.uint8)
    endpoints = torch.empty((num_blocks, 4, 3), dtype=torch.int16)
    indices = torch.empty((num_blocks, 16), dtype=torch.uint8)

    for i in range(num_blocks):
        value = int.from_bytes(data[i * rec_size : (i + 1) * rec_size], byteorder="little", signed=False)
        offset = 0

        def pop(nbits: int) -> int:
            nonlocal value, offset
            mask = (1 << nbits) - 1
            out = (value >> offset) & mask
            offset += nbits
            return int(out)

        parts[i] = pop(5)
        for e in range(12):
            endpoints.view(num_blocks, 12)[i, e] = pop(endpoint_bits)
        for j in range(16):
            indices[i, j] = pop(index_bits)

    return parts, endpoints, indices


def decode_quantized_params_to_mip(qp: dict, partition_bank: torch.Tensor) -> torch.Tensor:
    """
    Decodes exported quantized surrogate BC params back to [3,H,W] latent mip.
    """
    b = int(qp["endpoint_bits"])
    q = int(qp["index_bits"])
    signed_mode = bool(qp.get("signed_mode", False))

    e_q = qp["endpoints_q"].to(torch.float32)
    x_q = qp["indices_q"].to(torch.float32)
    p_id = qp["partition_id"].to(torch.long)

    a = 31.0 / 32.0 if signed_mode else 31.0 / 64.0
    e_u = (a * (2.0**16) * e_q + (2.0**15)) / (2.0**b)
    x_scaled = (2.0**q) * (x_q / max(1.0, float((1 << q) - 1)))

    mask = partition_bank[p_id].unsqueeze(-1).to(torch.float32)
    y1 = e_u[:, 0:1, :] + x_scaled.unsqueeze(-1) * (e_u[:, 1:2, :] - e_u[:, 0:1, :])
    y2 = e_u[:, 2:3, :] + x_scaled.unsqueeze(-1) * (e_u[:, 3:4, :] - e_u[:, 2:3, :])
    y = mask * y1 + (1.0 - mask) * y2

    h = torch.clamp(torch.floor((y - 1.0) / 1024.0) - 1.0, min=0.0)
    w = torch.pow(torch.tensor(2.0, dtype=y.dtype, device=y.device), h - 14.0) * ((y / 1024.0) - h)
    blocks = torch.tanh(w * 0.25)

    h_img = int(qp["h"])
    w_img = int(qp["w"])
    by = h_img // 4
    bx = w_img // 4
    tex = (
        blocks.view(by, bx, 4, 4, 3)
        .permute(4, 0, 2, 1, 3)
        .contiguous()
        .view(3, h_img, w_img)
    )
    return tex


# ---------------------------------------------------------------------------
# BC6H DDS helpers (Mode 12, single-subset, 11-bit endpoints, 3-bit indices)
# BC6H Mode 12: 5-bit mode header = 0b00111, one partition, absolute endpoints
# ---------------------------------------------------------------------------

import struct as _struct

_DDS_MAGIC = b"DDS "
_DDSD_CAPS = 0x1
_DDSD_HEIGHT = 0x2
_DDSD_WIDTH = 0x4
_DDSD_PIXELFORMAT = 0x1000
_DDSD_MIPMAPCOUNT = 0x20000
_DDSD_LINEARSIZE = 0x80000
_DDSCAPS_COMPLEX = 0x8
_DDSCAPS_TEXTURE = 0x1000
_DDSCAPS_MIPMAP = 0x400000
_DDPF_FOURCC = 0x4
_DXGI_BC6H_UF16 = 95
_DXGI_BC6H_SF16 = 96
_D3D10_TEXTURE2D = 3


def _pack_bc6h_mode12_block(
    ep0: Tuple[int, int, int],
    ep1: Tuple[int, int, int],
    indices: List[int],
    signed_mode: bool,
) -> bytes:
    """Pack one BC6H Mode 12 block into 16 bytes (LSB-first).

    Mode 12 layout (bits are LE, LSB first):
      [4:0]   = 0b00111 (mode identifier)
      [14:5]  = rw[9:0]   [24:15] = rx[9:0]
      [34:25] = gw[9:0]   [44:35] = gx[9:0]
      [54:45] = bw[9:0]   [64:55] = bx[9:0]
      [65]    = rw[10]    [66]    = rx[10]
      [67]    = gw[10]    [68]    = gx[10]
      [69]    = bw[10]    [70]    = bx[10]
      [72:71] = index[0] (anchor, 2 bits; MSB fixed 0)
      [75:73] = index[1], ..., [117:115] = index[15]  (3 bits each)
    """
    def _s11u(v: int) -> int:
        return v & 0x7FF  # two's-complement 11-bit → unsigned bits

    if signed_mode:
        rw, gw, bw = _s11u(ep0[0]), _s11u(ep0[1]), _s11u(ep0[2])
        rx, gx, bx = _s11u(ep1[0]), _s11u(ep1[1]), _s11u(ep1[2])
    else:
        rw, gw, bw = ep0[0] & 0x7FF, ep0[1] & 0x7FF, ep0[2] & 0x7FF
        rx, gx, bx = ep1[0] & 0x7FF, ep1[1] & 0x7FF, ep1[2] & 0x7FF

    block = 0x7  # mode bits [4:0] = 0b00111
    block |= (rw & 0x3FF) << 5
    block |= (rx & 0x3FF) << 15
    block |= (gw & 0x3FF) << 25
    block |= (gx & 0x3FF) << 35
    block |= (bw & 0x3FF) << 45
    block |= (bx & 0x3FF) << 55
    block |= ((rw >> 10) & 1) << 65
    block |= ((rx >> 10) & 1) << 66
    block |= ((gw >> 10) & 1) << 67
    block |= ((gx >> 10) & 1) << 68
    block |= ((bw >> 10) & 1) << 69
    block |= ((bx >> 10) & 1) << 70

    # anchor texel (2 bits, MSB implicit 0), then indices 1..15 (3 bits each)
    bit_pos = 71
    block |= (int(indices[0]) & 0x3) << bit_pos
    bit_pos += 2
    for i in range(1, 16):
        block |= (int(indices[i]) & 0x7) << bit_pos
        bit_pos += 3
    assert bit_pos == 118

    return block.to_bytes(16, byteorder="little")


def _write_bc6h_dds(
    mip_bytes_list: List[bytes],
    w0: int,
    h0: int,
    out_path: Path,
    signed_mode: bool,
):
    """Write a BC6H DDS file with a full mip chain."""
    mip_count = len(mip_bytes_list)
    dxgi_format = _DXGI_BC6H_SF16 if signed_mode else _DXGI_BC6H_UF16
    bw0, bh0 = max(1, (w0 + 3) // 4), max(1, (h0 + 3) // 4)
    linear_size = bw0 * bh0 * 16

    flags = _DDSD_CAPS | _DDSD_HEIGHT | _DDSD_WIDTH | _DDSD_PIXELFORMAT | _DDSD_LINEARSIZE
    if mip_count > 1:
        flags |= _DDSD_MIPMAPCOUNT
    caps = _DDSCAPS_TEXTURE
    if mip_count > 1:
        caps |= _DDSCAPS_COMPLEX | _DDSCAPS_MIPMAP

    dds_header = _struct.pack(
        "<IIIIIII11I",
        124, flags, h0, w0, linear_size, 0, mip_count, *([0] * 11),
    )
    dds_pixelformat = _struct.pack("<II4sIIIII", 32, _DDPF_FOURCC, b"DX10", 0, 0, 0, 0, 0)
    dds_caps = _struct.pack("<IIIII", caps, 0, 0, 0, 0)
    dx10_header = _struct.pack("<IIIII", dxgi_format, _D3D10_TEXTURE2D, 0, 1, 0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(_DDS_MAGIC)
        f.write(dds_header)
        f.write(dds_pixelformat)
        f.write(dds_caps)
        f.write(dx10_header)
        for mip_data in mip_bytes_list:
            f.write(mip_data)


def _pack_mip_blocks_to_bc6h_mode10_bytes(
    params: dict,
    signed_mode: bool,
    partition_bank: np.ndarray = None
) -> bytes:
    """
    Pack quantized surrogate block params → BC6H Mode 10 bytes.
    Mode 10: two-region with 32 partitions, 6-bit endpoints, 3-bit indices.

    Uses all 4 endpoints (both regions) and partition information, unlike Mode 12.
    6-bit endpoints are packed directly (no scaling needed).

    Args:
    - params: dict with endpoints_q, indices_q, partition_id
    - signed_mode: bool
    - partition_bank: [32, 16] numpy array of partition masks
    """
    if np is None:
        raise RuntimeError("numpy is required for BC6H block packing.")

    ep_q = params["endpoints_q"].numpy()  # [NB, 4, 3] int16
    idx_q = params["indices_q"].numpy()   # [NB, 16] uint8
    part_id = params["partition_id"].numpy()  # [NB] uint8

    # If partition_bank not provided, create a simple default
    if partition_bank is None:
        partition_bank = np.zeros((32, 16), dtype=np.uint8)
        for i in range(32):
            # Simple fallback: split pixels by index parity
            partition_bank[i] = (np.arange(16) % 2)

    out = bytearray()
    for bi in range(ep_q.shape[0]):
        # Get quantized endpoints for this block (already 6-bit)
        endpoints_block = ep_q[bi].astype(np.uint8)  # [4, 3]
        indices_block = idx_q[bi].astype(np.uint8)   # [16]
        partition_block = int(part_id[bi]) & 0x1F    # [0, 31]

        block_bytes = _pack_bc6h_mode10_block(
            endpoints_block,
            indices_block,
            partition_block,
            partition_bank,
            signed_mode
        )
        out.extend(block_bytes)

    return bytes(out)


def _pack_mip_blocks_to_bc6h_bytes(params: dict, signed_mode: bool) -> bytes:
    """
    Directly pack quantized surrogate block params → BC6H Mode 12 bytes.
    Lossless within the BC6H representation: no float texture decode/re-encode.

    Surrogate uses 2-subset blocks (4 endpoints); Mode 12 is single-subset.
    We use subset-0 endpoints (ep_q[bi,0] and ep_q[bi,1]) as ep0/ep1.

    Endpoint integer scaling (no float roundtrip):
      UF16: 6-bit [0, ep_max] → 11-bit [0, 2047]  via  round(q * 2047 / ep_max)
      SF16: 6-bit [0, ep_max] → SF16 [-1023, 1023] via  round((q/ep_max*2-1)*1023)
    Both done with integer arithmetic to keep the packing exact.
    """
    if np is None:
        raise RuntimeError("numpy is required for BC6H block packing.")
    ep_q = params["endpoints_q"].numpy()  # [NB, 4, 3] int16
    idx_q = params["indices_q"].numpy()   # [NB, 16] uint8
    ep_bits = int(params["endpoint_bits"])
    ep_max = (1 << ep_bits) - 1  # 63 for 6-bit

    ep_i = ep_q.astype(np.int32)
    if signed_mode:
        # round((ep_q/ep_max * 2 - 1) * 1023)  →  (ep_q*2046 + ep_max//2) // ep_max - 1023
        ep_11 = (ep_i * 2046 + ep_max // 2) // ep_max - 1023
        ep_11 = ep_11.clip(-1023, 1023)
    else:
        # round(ep_q / ep_max * 2047)  →  (ep_q*2047 + ep_max//2) // ep_max
        ep_11 = (ep_i * 2047 + ep_max // 2) // ep_max
        ep_11 = ep_11.clip(0, 2047)

    out = bytearray()
    for bi in range(ep_q.shape[0]):
        ep0 = ep_11[bi, 0]  # subset-0 ep0, shape [3]
        ep1 = ep_11[bi, 1]  # subset-0 ep1, shape [3]
        idxs = idx_q[bi].tolist()

        # Enforce anchor constraint: index[0] MSB must be 0 (< 4)
        if idxs[0] > 3:
            ep0, ep1 = ep1, ep0
            idxs = [7 - x for x in idxs]

        out.extend(_pack_bc6h_mode12_block(
            (int(ep0[0]), int(ep0[1]), int(ep0[2])),
            (int(ep1[0]), int(ep1[1]), int(ep1[2])),
            idxs,
            signed_mode=signed_mode,
        ))
    return bytes(out)


def _decode_bc6h_mode12_block(block_bytes: bytes, signed_mode: bool) -> "np.ndarray":
    """Decode one 16-byte BC6H Mode 12 block to 16 texels [4, 4, 3] float32.

    UF16: output in [0, 1].  SF16: output in [-1, 1].
    Mirrors _pack_bc6h_mode12_block exactly.
    """
    v = int.from_bytes(block_bytes, byteorder="little")

    # Extract lower 10 bits of each endpoint channel
    rw = (v >> 5)  & 0x3FF
    rx = (v >> 15) & 0x3FF
    gw = (v >> 25) & 0x3FF
    gx = (v >> 35) & 0x3FF
    bw = (v >> 45) & 0x3FF
    bx = (v >> 55) & 0x3FF

    # Reconstruct 11-bit endpoints (MSBs stored at bits 65-70)
    rw |= ((v >> 65) & 1) << 10
    rx |= ((v >> 66) & 1) << 10
    gw |= ((v >> 67) & 1) << 10
    gx |= ((v >> 68) & 1) << 10
    bw |= ((v >> 69) & 1) << 10
    bx |= ((v >> 70) & 1) << 10

    # Extract indices: anchor[0] = 2 bits at 71, rest = 3 bits each
    indices = [(v >> 71) & 0x3]
    bp = 73
    for _ in range(15):
        indices.append((v >> bp) & 0x7)
        bp += 3

    # Convert 11-bit integer endpoints to float
    if signed_mode:
        def _s11f(x: int) -> float:
            # 11-bit two's complement → float in [-1, 1]
            return (x - 2048 if x > 1023 else x) / 1023.0
        ep0 = np.array([_s11f(rw), _s11f(gw), _s11f(bw)], dtype=np.float32)
        ep1 = np.array([_s11f(rx), _s11f(gx), _s11f(bx)], dtype=np.float32)
    else:
        ep0 = np.array([rw / 2047.0, gw / 2047.0, bw / 2047.0], dtype=np.float32)
        ep1 = np.array([rx / 2047.0, gx / 2047.0, bx / 2047.0], dtype=np.float32)

    # Interpolate 16 texels (3-bit indices: 8 levels, weight = idx/7)
    out = np.empty((16, 3), dtype=np.float32)
    for i, idx in enumerate(indices):
        t = idx / 7.0
        out[i] = ep0 + (ep1 - ep0) * t

    return out.reshape(4, 4, 3)


def decode_bc6h_dds_mip0(dds_path: Path, signed_mode: bool) -> "torch.Tensor":
    """Decode mip0 of a BC6H Mode 12 DDS file → float tensor [3, H, W].

    Reads the DDS header for W/H, skips to mip0 data (offset 148 bytes),
    decodes each 4×4 block, and returns latent space values:
      UF16 (unsigned) → latent [0, 1]  (soft surrogate produces [0, 1])
      SF16 (signed)   → latent [-1, 1]  (soft surrogate produces [-1, 1])
    """
    if np is None:
        raise RuntimeError("numpy is required for BC6H DDS decoding.")

    raw = dds_path.read_bytes()
    if raw[:4] != _DDS_MAGIC:
        raise RuntimeError(f"Not a DDS file: {dds_path}")

    # DDS layout: 4 (magic) + 124 (header) + 20 (DX10 ext) = 148 bytes before pixel data
    h = _struct.unpack_from("<I", raw, 12)[0]
    w = _struct.unpack_from("<I", raw, 16)[0]
    bw = max(1, (w + 3) // 4)
    bh = max(1, (h + 3) // 4)

    offset = 148
    mip0_end = offset + bw * bh * 16
    mip0_data = raw[offset:mip0_end]

    pixels = np.zeros((h, w, 3), dtype=np.float32)
    for by in range(bh):
        for bx_i in range(bw):
            bi = by * bw + bx_i
            texels = _decode_bc6h_mode12_block(mip0_data[bi * 16:(bi + 1) * 16], signed_mode)
            y0, x0 = by * 4, bx_i * 4
            rh, rw_px = min(4, h - y0), min(4, w - x0)
            pixels[y0:y0 + rh, x0:x0 + rw_px] = texels[:rh, :rw_px]

    t = torch.from_numpy(pixels).permute(2, 0, 1)  # [3, H, W]
    # Unsigned mode: keep [0, 1] (soft surrogate output is [0, 1])
    # Signed mode: keep [-1, 1] (soft surrogate output is [-1, 1])
    return t


def _decode_bc6h_mode10_block(
    block_bytes: bytes,
    partition_bank: torch.Tensor,
    signed_mode: bool
) -> "np.ndarray":
    """Decode one 16-byte BC6H Mode 10 block to [4, 4, 3] float32.

    Mode 10: 6-bit endpoints, 32 partitions, 3-bit indices.

    Bit layout (little-endian):
      [4:0]     = Mode bits (0b01110 = 14)
      [9:5]     = Partition ID (5 bits, 0-31)
      [69:10]   = 4 × 6-bit endpoints × 3 channels (60 bits)
      [127:70]  = 16 indices × 3-bit + fix-up (46 bits)
    """
    if np is None:
        raise RuntimeError("numpy is required for BC6H Mode 10 decoding.")

    v = int.from_bytes(block_bytes, byteorder="little")

    # Extract mode bits (verify Mode 10)
    mode = v & 0x1F
    if mode != 14:  # Mode 10 = 0b01110 = 14
        raise ValueError(f"Expected Mode 10 (14), got mode {mode}")

    # Extract partition ID [9:5]
    partition_id = (v >> 5) & 0x1F
    if partition_id >= len(partition_bank):
        raise ValueError(f"Partition ID {partition_id} out of range [0, {len(partition_bank)-1}]")

    # Extract 6-bit endpoints [69:10] - 4 endpoints × 3 channels
    endpoints_q = []
    bit_pos = 10
    for ep_idx in range(4):
        endpoint = []
        for ch_idx in range(3):
            ch_val = (v >> bit_pos) & 0x3F  # 6-bit
            endpoint.append(ch_val)
            bit_pos += 6
        endpoints_q.append(endpoint)

    # Unquantize endpoints using BC6H formula
    # For 6-bit: if x == 0: u = 0, if x == 63: u = 0xFFFF, else u = ((x << 16) + 0x8000) >> 6
    endpoints_u = []
    for endpoint in endpoints_q:
        endpoint_u = []
        for ch_val in endpoint:
            if ch_val == 0:
                u = 0.0
            elif ch_val == 63:
                u = 1.0  # Normalized from 0xFFFF
            else:
                # BC6H unquantize formula: ((x << 16) + 0x8000) >> 6, then normalize
                u_int = ((ch_val << 16) + 0x8000) >> 6  # Range [0, 65535]
                u = u_int / 65535.0
            endpoint_u.append(u)
        endpoints_u.append(np.array(endpoint_u, dtype=np.float32))

    # Handle signed mode
    if signed_mode:
        # Convert from [0, 1] to [-1, 1] for signed
        endpoints_u = [2.0 * ep - 1.0 for ep in endpoints_u]

    # Extract indices [127:70] - 16 indices (simplified: all 3-bit, ignore fix-up optimization)
    # NOTE: Proper BC6H has fix-up indices (2-bit for anchors), but we simplify here
    # by using 3-bit for all indices. This costs 2 bits but avoids partition-dependent logic.
    indices = []
    bit_pos = 70

    for i in range(16):
        # All indices as 3-bit values [0,7]
        idx = (v >> bit_pos) & 0x7
        indices.append(idx)
        bit_pos += 3

    # BC6H interpolation weights for 3-bit indices (8 levels)
    weights = np.array([0, 9, 18, 27, 37, 46, 55, 64], dtype=np.float32) / 64.0

    # Get partition mask
    partition_mask = partition_bank[partition_id].cpu().numpy()  # [16]

    # Interpolate per-texel using partition
    out = np.empty((16, 3), dtype=np.float32)
    for i in range(16):
        region = int(partition_mask[i])
        if region == 0:
            ep0, ep1 = endpoints_u[0], endpoints_u[1]
        else:
            ep0, ep1 = endpoints_u[2], endpoints_u[3]

        idx = indices[i]
        weight = weights[idx]
        out[i] = ep0 + (ep1 - ep0) * weight

    return out.reshape(4, 4, 3)


def _get_anchor_indices_for_partition(partition_mask: np.ndarray) -> tuple[int, int]:
    """
    Get anchor bit indices for a BC6H partition mask.

    Anchor 0 is always at pixel 0.
    Anchor 1 is the first pixel in the second region (first pixel with different region value).

    Returns: (anchor0_idx, anchor1_idx)
    """
    anchor0_region = int(partition_mask[0])

    # Find first pixel in the other region
    for i in range(1, 16):
        if int(partition_mask[i]) != anchor0_region:
            return (0, i)

    # Shouldn't happen with valid partition (needs two regions), default to position 0 and 8
    return (0, 8)


def _pack_bc6h_mode10_block(
    endpoints_q: np.ndarray,
    indices_q: np.ndarray,
    partition_id: int,
    partition_bank: np.ndarray = None,
    signed_mode: bool = False
) -> bytes:
    """Pack 4×4 block into 16-byte BC6H Mode 10 format (128 bits exactly).

    Input:
    - endpoints_q: [4, 3] 6-bit quantized endpoints [0, 63]
    - indices_q: [16] quantized indices [0, 7]
    - partition_id: [0, 31]
    - partition_bank: [32, 16] partition masks for anchor detection
    - signed_mode: bool

    Output: 16 bytes (128-bit block, little-endian)

    BC6H Mode 10 Layout (128 bits, per DXGI spec):
    - Bits [4:0]: Mode = 14 (5 bits, positions 0-4)
    - Bits [9:5]: Partition ID (5 bits, positions 5-9)
    - Bits [69:10]: Endpoints (60 bits, positions 10-69)
    - Bits [127:70]: Indices with 2-bit anchors (2+2+14×3 = 46 bits, positions 70-127)
    Total: 5 + 5 + 60 + 46 = 116 bits used + 12 bits padding = 128 bits

    Note: Endpoints field is 60 bits. For 4 endpoints × 3 channels, this is 10 × 6-bit values (not 12).
    The layout stores endpoint pairs per partition in a specific format.

    Anchor optimization:
    - Pixel 0: always 2-bit anchor
    - Pixel at partition anchor position: 2-bit anchor
    - Other 14 pixels: 3-bit indices
    """
    # Initialize 128-bit block
    block = 0

    # Pack mode bits [4:0] = 0b01110 (14 for Mode 10)
    block |= 14

    # Pack partition ID [9:5]
    block |= (partition_id & 0x1F) << 5

    # Pack endpoints [69:10] - 4 endpoints × 3 channels × 6 bits
    bit_pos = 10
    for ep_idx in range(4):
        for ch_idx in range(3):
            ch_val = int(endpoints_q[ep_idx, ch_idx]) & 0x3F
            block |= ch_val << bit_pos
            bit_pos += 6

    # Pack indices with 2-bit anchors
    # Find partition anchor (second anchor pixel for this partition)
    partition_anchor = 15  # Default fallback
    if partition_bank is not None:
        # Partition mask tells us which endpoint pair each pixel uses
        # Pixel 0 is always anchor. Find the next anchor (usually pixel with largest x or y)
        partition_mask = partition_bank[partition_id]  # [16]
        # In BC6H Mode 10, second anchor varies by partition
        # Common second anchors: 2, 8, 15, etc. depending on partition
        # For simplicity: if partition uses two endpoint pairs, find transition point
        transition_pixels = []
        for i in range(1, 16):
            if partition_mask[i] != partition_mask[i-1]:
                transition_pixels.append(i)
        if transition_pixels:
            partition_anchor = transition_pixels[0]

    # Indices start after all 4 endpoints (10 + 72 = 82)
    # Note: Standard BC6H spec shows [127:70] for indices, but that overlaps
    # with 4×3×6=72-bit endpoint data. Our implementation packs all 4 endpoints
    # and places indices at position 82, leaving a small gap at bits 70-81.
    bit_pos = 82
    for i in range(16):
        if i == 0 or i == partition_anchor:
            # 2-bit anchor indices
            idx = int(indices_q[i]) & 0x3  # Mask to 2 bits
            block |= idx << bit_pos
            bit_pos += 2
        else:
            # 3-bit regular indices
            idx = int(indices_q[i]) & 0x7  # Mask to 3 bits
            block |= idx << bit_pos
            bit_pos += 3

    return block.to_bytes(16, byteorder="little")


@torch.no_grad()
def export_trained_artifacts(model: NeuralMaterialCompressionModel, out_dir: Path):
    """
    Export all runtime artifacts to out_dir.

    Runtime files (export root):
      decoder_fp16.bin, metadata.json, latent_XX.bc6.dds

    Debug files (metadata/ subdir):
      decoder_state.pt, latent_XX_mip_YY.png

    DDS files are built directly from the trained block params — no float32
    decode/re-encode roundtrip.  PNG previews are decoded from the soft
    surrogate for visual inspection only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # --- Decoder weights ---
    state = model.decoder.state_dict()
    torch.save(state, meta_dir / "decoder_state.pt")
    flat = []
    for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
        flat.append(state[k].detach().to(torch.float16).contiguous().view(-1))
    fp16_blob = torch.cat(flat).cpu().numpy().tobytes()
    (out_dir / "decoder_fp16.bin").write_bytes(fp16_blob)

    # --- Latent DDS (BC6H from block params) + PNG previews ---
    latent_files = []
    model.set_freeze_bc_features(True)
    for i, pyr in enumerate(model.bc_pyramids):
        # Block params: no float32 decode needed for the DDS
        all_params = pyr.export_quantized_params()
        # Decoded mips using training soft surrogate decoder (original formula)
        decoded_mips = pyr.decode_mips(hard_partition=True)

        mip_bytes_list = []
        for m, (params, tex) in enumerate(zip(all_params, decoded_mips)):
            stem = f"latent_{i:02d}_mip_{m:02d}"
            # PNG preview from training soft surrogate decoder
            # (Different from BC6H GPU decode - this is expected and correct!)
            save_chw_png_ldr(tex, meta_dir / f"{stem}.png", signed_mode=model.bc6_signed_mode)
            # BC6H bytes packed directly from quantized block params (Mode 10)
            # Pass partition_bank for correct anchor bit packing
            partition_bank_np = model.partition_bank.cpu().numpy()
            mip_bytes_list.append(_pack_mip_blocks_to_bc6h_mode10_bytes(
                params,
                model.bc6_signed_mode,
                partition_bank=partition_bank_np
            ))
            latent_files.append({
                "latent_index": i,
                "mip_index": m,
                "shape_chw": list(tex.shape),
                "png": f"metadata/{stem}.png",
            })

        # Write DDS to export root
        W0, H0 = int(all_params[0]["w"]), int(all_params[0]["h"])
        dds_path = out_dir / f"latent_{i:02d}.bc6.dds"
        _write_bc6h_dds(mip_bytes_list, W0, H0, dds_path, signed_mode=model.bc6_signed_mode)
        print(f"[export] latent {i:02d}: {W0}×{H0}  mips={len(mip_bytes_list)}  -> {dds_path.name}")

    meta = {
        "version": 4,
        "latent_count": model.n_latent,
        "latent_resolutions": model.latent_resolutions,
        "lod_biases": model.lod_biases,
        "bc6_signed_mode": model.bc6_signed_mode,
        "decoder": {
            "in_dim": int(model.decoder.fc1.in_features),
            "hidden_dim": int(model.decoder.fc1.out_features),
            "out_dim": int(model.decoder.fc2.out_features),
            "weights_fp16_blob": "decoder_fp16.bin",
            "state_dict": "metadata/decoder_state.pt",
        },
        "latent_files": latent_files,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote artifacts to {out_dir}")


def train(model: NeuralMaterialCompressionModel, ref_mips: List[torch.Tensor], cfg: TrainConfig) -> List[dict]:
    device = torch.device(cfg.device)
    max_lod = float(len(ref_mips) - 1)
    history: List[dict] = []

    # ---- Phase 1: warmup (unconstrained latents + decoder)
    opt1 = torch.optim.Adam(
        [
            {"params": list(model.warmup_parameters()), "lr": cfg.lr_feat_phase1},
            {"params": list(model.decoder_parameters()), "lr": cfg.lr_mlp_phase1},
        ]
    )
    sch1 = torch.optim.lr_scheduler.ExponentialLR(opt1, gamma=cfg.gamma_phase1)

    phase1_iter = range(cfg.phase1_iters)
    pbar1 = None
    if cfg.interactive_progress and tqdm is not None and cfg.phase1_iters > 0:
        pbar1 = tqdm(phase1_iter, desc="phase1", dynamic_ncols=True)
    for it in (pbar1 if pbar1 is not None else phase1_iter):
        uv, lod = random_uv_lod(cfg.batch_size, max_lod, device)
        target = sample_mips_trilinear(ref_mips, uv, lod, bilinear_mode="bicubic")
        pred = model.forward_warmup(uv, lod)
        loss = F.mse_loss(pred, target)
        history.append({"phase": 1, "iter": it, "mse": float(loss.item())})

        opt1.zero_grad(set_to_none=True)
        loss.backward()
        opt1.step()
        sch1.step()

        if pbar1 is not None:
            if (it % cfg.log_every) == 0 or it == (cfg.phase1_iters - 1):
                pbar1.set_postfix(mse=f"{loss.item():.4e}", refresh=False)
        elif it % cfg.log_every == 0:
            print(f"[phase1 {it:06d}/{cfg.phase1_iters}] mse={loss.item():.6e}")
    if pbar1 is not None:
        pbar1.close()

    # Initialize BC pyramids from warmup latents.
    model.initialize_bc_from_warmup()

    # ---- Phase 2: BC-constrained (BC features + decoder)
    model.set_freeze_bc_features(False)
    opt2 = torch.optim.Adam(
        [
            {"params": list(model.bc_feature_parameters()), "lr": cfg.lr_feat_phase2},
            {"params": list(model.decoder_parameters()), "lr": cfg.lr_mlp_phase2},
        ]
    )
    sch2 = torch.optim.lr_scheduler.ExponentialLR(opt2, gamma=cfg.gamma_phase2)

    phase2_iter = range(cfg.phase2_iters)
    pbar2 = None
    if cfg.interactive_progress and tqdm is not None and cfg.phase2_iters > 0:
        pbar2 = tqdm(phase2_iter, desc="phase2", dynamic_ncols=True)
    for it in (pbar2 if pbar2 is not None else phase2_iter):
        uv, lod = random_uv_lod(cfg.batch_size, max_lod, device)
        target = sample_mips_trilinear(ref_mips, uv, lod, bilinear_mode="bicubic")

        pred = model.forward_bc(uv, lod)
        loss = F.mse_loss(pred, target)
        history.append({"phase": 2, "iter": it, "mse": float(loss.item())})

        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
        sch2.step()

        if pbar2 is not None:
            if (it % cfg.log_every) == 0 or it == (cfg.phase2_iters - 1):
                pbar2.set_postfix(mse=f"{loss.item():.4e}", refresh=False)
        elif it % cfg.log_every == 0:
            print(f"[phase2 {it:06d}/{cfg.phase2_iters}] mse={loss.item():.6e}")
    if pbar2 is not None:
        pbar2.close()

    # ---- Phase 3: optional export-finetune (freeze BC features, tune decoder only)
    if cfg.phase3_iters > 0:
        model.quantize_and_freeze_bc_features()

        opt3 = torch.optim.Adam(model.decoder_parameters(), lr=cfg.lr_mlp_phase3)
        phase3_iter = range(cfg.phase3_iters)
        pbar3 = None
        if cfg.interactive_progress and tqdm is not None and cfg.phase3_iters > 0:
            pbar3 = tqdm(phase3_iter, desc="phase3", dynamic_ncols=True)
        for it in (pbar3 if pbar3 is not None else phase3_iter):
            uv, lod = random_uv_lod(cfg.batch_size, max_lod, device)
            target = sample_mips_trilinear(ref_mips, uv, lod, bilinear_mode="bicubic")
            pred = model.forward_bc(uv, lod)
            loss = F.mse_loss(pred, target)
            history.append({"phase": 3, "iter": it, "mse": float(loss.item())})

            opt3.zero_grad(set_to_none=True)
            loss.backward()
            opt3.step()

            log_step = max(1, cfg.log_every // 2)
            if pbar3 is not None:
                if (it % log_step) == 0 or it == (cfg.phase3_iters - 1):
                    pbar3.set_postfix(mse=f"{loss.item():.4e}", refresh=False)
            elif it % log_step == 0:
                print(f"[phase3 {it:05d}/{cfg.phase3_iters}] mse={loss.item():.6e}")
        if pbar3 is not None:
            pbar3.close()

    return history


def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-pt", type=Path, default=None, help="Path to .pt with {'base':CHW} or {'mips':[CHW...]}")
    p.add_argument("--out-channels", type=int, default=8, help="Output material channels (e.g., albedo3+normal2+arm3=8)")
    p.add_argument("--ref-mips", type=int, default=9, help="Number of reference mip levels")
    p.add_argument("--latent-res", type=str, default="512,256,128,64")
    p.add_argument("--latent-mips", type=str, default="8,7,6,5")
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--endpoint-bits", type=int, default=6)
    p.add_argument("--index-bits", type=int, default=3)
    p.add_argument(
        "--bc6-signed-mode",
        action="store_true",
        help="Use signed-mode constant (31/32) in bc6_eq surrogate.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--phase1-iters", type=int, default=5000)
    p.add_argument("--phase2-iters", type=int, default=200000)
    p.add_argument("--phase3-iters", type=int, default=0)
    p.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="If set, export decoder + latent artifacts to this directory after training.",
    )
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--interactive-progress", action="store_true", help="Enable tqdm-like live progress bars during training.")
    args = p.parse_args()

    device = torch.device(args.device)
    latent_res = parse_list_int(args.latent_res)
    latent_mips = parse_list_int(args.latent_mips)
    if len(latent_res) != len(latent_mips):
        raise ValueError("--latent-res and --latent-mips length mismatch")

    ref_mips = load_reference_mips(args.reference_pt, args.ref_mips, args.out_channels, device=device)
    ref_base_res = int(ref_mips[0].shape[1])

    partition_bank = make_partition_bank(device=device)
    model = NeuralMaterialCompressionModel(
        latent_resolutions=latent_res,
        latent_mips=latent_mips,
        out_channels=args.out_channels,
        hidden_dim=args.hidden_dim,
        partition_bank=partition_bank,
        endpoint_bits=args.endpoint_bits,
        index_bits=args.index_bits,
        ref_base_res=ref_base_res,
        bc6_signed_mode=args.bc6_signed_mode,
    ).to(device)

    cfg = TrainConfig(
        device=args.device,
        batch_size=args.batch_size,
        phase1_iters=args.phase1_iters,
        phase2_iters=args.phase2_iters,
        phase3_iters=args.phase3_iters,
        log_every=args.log_every,
        interactive_progress=args.interactive_progress,
    )

    train(model, ref_mips, cfg)

    if args.export_dir is not None:
        export_trained_artifacts(model=model, out_dir=args.export_dir)


if __name__ == "__main__":
    main()
