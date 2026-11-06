"""Differentiable block features and the material decoder."""

from __future__ import annotations
from typing import List, Sequence, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bc6h_spec import (
    BC67_WEIGHT_MAX,
    BC6H_MODE10_ENDPOINT_BITS,
    BC6H_MODE10_ENDPOINT_FLOATS_UF16,
    BC6H_MODE10_INDEX_BITS,
    BC6H_MODE10_PARTITION_TABLE,
    BC6H_MODE10_WEIGHTS,
)
from .sampling import sample_mips_trilinear
from .bc_math import decode_blocks, decode_palette, continuous_unquantize6

EPS = 1e-8


def make_partition_bank(device: torch.device) -> torch.Tensor:
    return torch.tensor(BC6H_MODE10_PARTITION_TABLE, dtype=torch.float32, device=device)


def _decode_mode10_quantized_blocks_torch(
    endpoints_q: torch.Tensor,
    indices_q: torch.Tensor,
    mask: torch.Tensor,
    endpoint_bits: int,
    weight_values: torch.Tensor,
) -> torch.Tensor:
    if endpoint_bits != 6:
        raise ValueError("Mode 10 requires six endpoint bits")
    return decode_blocks(endpoints_q, indices_q, mask)


def _fit_mode10_subset(
    blocks: torch.Tensor,
    subset_mask: torch.Tensor,
    weight_lut_norm: torch.Tensor,
    endpoint_lut: torch.Tensor,
    refine_steps: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Start with subset extrema, then coordinate-descent endpoint codes against
    # the actual nonlinear palette. Never project onto a float-space line.
    valid = subset_mask.bool()
    low = blocks.masked_fill(~valid, float("inf")).amin(1)
    high = blocks.masked_fill(~valid, -float("inf")).amax(1)
    endpoints = torch.stack([low, high], 1)
    codes = (endpoints.unsqueeze(-1) - endpoint_lut).abs().argmin(-1)

    def score(ep):
        palette = decode_palette(ep)
        errors = (blocks[:, :, None, :] - palette[:, None, :, :]).square().sum(-1)
        err, indices = errors.min(-1)
        return (err * subset_mask.squeeze(-1)).sum(-1), indices

    best, indices = score(codes)
    for _ in range(refine_steps):
        for endpoint in range(2):
            for channel in range(3):
                for delta in (-1, 1):
                    candidate = codes.clone()
                    candidate[:, endpoint, channel] = (
                        candidate[:, endpoint, channel] + delta
                    ).clamp(0, 63)
                    error, candidate_indices = score(candidate)
                    better = error < best
                    codes = torch.where(better[:, None, None], candidate, codes)
                    indices = torch.where(better[:, None], candidate_indices, indices)
                    best = torch.minimum(best, error)
    values = endpoint_lut[codes]
    return values[:, 0], values[:, 1], indices, codes


def _search_mode10_initial_block_params(
    blocks: torch.Tensor,
    partition_bank: torch.Tensor,
    endpoint_bits: int,
    index_bits: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        index_bits != BC6H_MODE10_INDEX_BITS
        or endpoint_bits != BC6H_MODE10_ENDPOINT_BITS
    ):
        raise ValueError(
            "The fixed BC6H initialization path currently targets Mode 10 (6-bit endpoints, 3-bit indices)."
        )

    bank = partition_bank.to(device=blocks.device, dtype=blocks.dtype)
    nb = blocks.shape[0]
    endpoint_max = float((1 << endpoint_bits) - 1)
    weight_values = torch.tensor(
        BC6H_MODE10_WEIGHTS,
        device=blocks.device,
        dtype=blocks.dtype,
    )
    weight_lut_norm = weight_values / float(BC67_WEIGHT_MAX)
    endpoint_lut = torch.tensor(
        BC6H_MODE10_ENDPOINT_FLOATS_UF16,
        device=blocks.device,
        dtype=blocks.dtype,
    )

    best_err = torch.full((nb,), float("inf"), device=blocks.device, dtype=blocks.dtype)
    best_part = torch.zeros((nb,), device=blocks.device, dtype=torch.long)
    best_endpoints_q = torch.zeros((nb, 4, 3), device=blocks.device, dtype=torch.int64)
    best_indices_q = torch.zeros((nb, 16), device=blocks.device, dtype=torch.int64)

    for pid in range(int(bank.shape[0])):
        mask = bank[pid].view(1, 16, 1)
        ep0, ep1, idx0, ep_codes0 = _fit_mode10_subset(
            blocks, 1.0 - mask, weight_lut_norm, endpoint_lut
        )
        ep2, ep3, idx1, ep_codes1 = _fit_mode10_subset(
            blocks, mask, weight_lut_norm, endpoint_lut
        )

        endpoints_q = (
            torch.stack(
                [
                    ep_codes0[:, 0],
                    ep_codes0[:, 1],
                    ep_codes1[:, 0],
                    ep_codes1[:, 1],
                ],
                dim=1,
            )
            .to(torch.int64)
            .clamp_(0, int(endpoint_max))
        )
        indices_q = (
            (1.0 - mask.squeeze(-1)).to(torch.int64) * idx0
            + mask.squeeze(-1).to(torch.int64) * idx1
        ).clamp_(0, (1 << index_bits) - 1)

        recon = _decode_mode10_quantized_blocks_torch(
            endpoints_q=endpoints_q,
            indices_q=indices_q,
            mask=mask.squeeze(-1),
            endpoint_bits=endpoint_bits,
            weight_values=weight_values,
        )
        err = (recon - blocks).pow(2).mean(dim=(1, 2))
        better = err < best_err

        best_err = torch.where(better, err, best_err)
        best_part = torch.where(better, torch.full_like(best_part, pid), best_part)
        best_endpoints_q = torch.where(
            better.view(-1, 1, 1), endpoints_q, best_endpoints_q
        )
        best_indices_q = torch.where(better.view(-1, 1), indices_q, best_indices_q)

    return best_part, best_endpoints_q, best_indices_q


class WarmupLatentPyramid(nn.Module):
    def __init__(self, base_res: int, num_mips: int):
        super().__init__()
        self.base_res = base_res
        self.num_mips = num_mips
        mips = []
        for i in range(num_mips):
            sz = max(1, base_res >> i)
            p = nn.Parameter(torch.randn(3, sz, sz) * 0.05 - 3.0)
            mips.append(p)
        self.mips = nn.ParameterList(mips)

    def sample(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        mips = [F.softplus(m) for m in self.mips]
        return sample_mips_trilinear(mips, uv, lod, bilinear_mode="bilinear")

    def decode_mips(self) -> List[torch.Tensor]:
        return [F.softplus(m) for m in self.mips]


class BC6SurrogateBlockLevel(nn.Module):
    """
    BC6-style block parameterization for one mip level:
    - per 4x4 block: 4 endpoints (RGB), 16 scalar indices, 1 fixed partition ID selecting one of 32 masks
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
        assert h % 4 == 0 and w % 4 == 0, (
            f"BC blocks require multiples of 4, got {h}x{w}"
        )
        self.h = h
        self.w = w
        self.by = h // 4
        self.bx = w // 4
        self.nb = self.by * self.bx
        self.endpoint_bits = endpoint_bits
        self.index_bits = index_bits
        self.bc6_signed_mode = bc6_signed_mode

        self.register_buffer("partition_bank", partition_bank)  # [32,16]
        self.register_buffer(
            "mode10_weight_lut",
            torch.tensor(BC6H_MODE10_WEIGHTS, dtype=torch.float32) / 64.0,
        )
        self.endpoints = nn.Parameter(torch.randn(self.nb, 4, 3) * 0.1)
        self.indices = nn.Parameter(torch.zeros(self.nb, 16))
        self.register_buffer("partition_ids", torch.zeros(self.nb, dtype=torch.long))
        self.register_buffer("frozen_mip", torch.empty(0))

    @torch.no_grad()
    def init_from_unconstrained(self, mip_chw: torch.Tensor):
        """
        Paper-like warm-start from unconstrained mip:
        run a fixed Mode 10 search over the official 32 BC6 partitions, then
        initialize endpoints/indices from the best quantized block parameters.
        """
        x = mip_chw.detach().clamp_min(0.0)
        # [3,H,W] -> [NB,16,3]
        blocks = (
            x.unfold(1, 4, 4)
            .unfold(2, 4, 4)
            .permute(1, 2, 3, 4, 0)
            .contiguous()
            .view(self.nb, 16, 3)
        )
        partition_bank = self.partition_bank

        chunk = 4096
        best_part = torch.zeros((self.nb,), device=blocks.device, dtype=torch.long)
        best_endpoints_q = torch.zeros(
            (self.nb, 4, 3), device=blocks.device, dtype=torch.int64
        )
        best_indices_q = torch.zeros(
            (self.nb, 16), device=blocks.device, dtype=torch.int64
        )

        for s in range(0, self.nb, chunk):
            e = min(self.nb, s + chunk)
            blk = blocks[s:e]  # [B,16,3]
            (
                best_part[s:e],
                best_endpoints_q[s:e],
                best_indices_q[s:e],
            ) = _search_mode10_initial_block_params(
                blocks=blk,
                partition_bank=partition_bank,
                endpoint_bits=self.endpoint_bits,
                index_bits=self.index_bits,
            )

        # Set fixed partition IDs from best warmup fit.
        self.partition_ids.copy_(best_part)

        # Initialize params so a round-trip quantize lands on the chosen Mode 10 values.
        idx_levels = float((1 << self.index_bits) - 1)
        ep_levels = float((1 << self.endpoint_bits) - 1)
        idx_n = (best_indices_q.to(blocks.dtype) / idx_levels).clamp(1e-4, 1.0 - 1e-4)
        ep_n = (best_endpoints_q.to(blocks.dtype) / ep_levels).clamp(1e-4, 1.0 - 1e-4)
        self.indices.copy_(torch.log(idx_n / (1.0 - idx_n)))
        self.endpoints.copy_(torch.log(ep_n / (1.0 - ep_n)))

    def _partition_mask(self, hard_partition: bool) -> torch.Tensor:
        if not hard_partition:
            raise ValueError(
                "Partition IDs are fixed; soft partition training is unsupported"
            )
        return self.partition_bank[self.partition_ids].unsqueeze(-1)

    def _decode_soft_blocks(self, hard_partition: bool) -> torch.Tensor:
        b = int(self.endpoint_bits)
        q = int(self.index_bits)
        if self.bc6_signed_mode:
            raise NotImplementedError(
                "Paper-aligned BC6 training path is currently implemented for unsigned BC6H only."
            )

        e_q = torch.sigmoid(self.endpoints) * ((1 << b) - 1)
        e_u = continuous_unquantize6(e_q) * (31.0 / 64.0)

        x_n = torch.sigmoid(self.indices)
        lut = self.mode10_weight_lut.to(
            device=self.indices.device, dtype=self.indices.dtype
        )
        x_scaled = x_n * ((1 << q) - 1)
        x0 = torch.floor(x_scaled).long()
        x1 = torch.clamp(x0 + 1, max=(1 << q) - 1)
        frac = (x_scaled - x0.to(x_scaled.dtype)).clamp(0.0, 1.0)
        weights = lut[x0] * (1.0 - frac) + lut[x1] * frac

        mask = self._partition_mask(hard_partition)
        y1 = e_u[:, 0:1, :] * (1.0 - weights).unsqueeze(-1) + e_u[
            :, 1:2, :
        ] * weights.unsqueeze(-1)
        y2 = e_u[:, 2:3, :] * (1.0 - weights).unsqueeze(-1) + e_u[
            :, 3:4, :
        ] * weights.unsqueeze(-1)
        y = (1.0 - mask) * y1 + mask * y2
        h = torch.clamp(torch.floor((y - 1.0) / 1024.0) - 1.0, 0.0, 31.0)
        pow2 = torch.pow(torch.tensor(2.0, device=y.device, dtype=y.dtype), h - 14.0)
        return pow2 * ((y / 1024.0) - h)

    def decode_mip(self, hard_partition: bool) -> torch.Tensor:
        if self.frozen_mip.numel():
            return self.frozen_mip
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
        p_id = self.partition_ids.to(torch.uint8)
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
        # IDs are buffers from construction, never trainable logits.
        pass

    @torch.no_grad()
    def quantize_inplace(self):
        params = self.export_quantized_block_params()
        blocks = decode_blocks(
            params["endpoints_q"].to(self.endpoints.device),
            params["indices_q"].to(self.endpoints.device),
            self.partition_bank[self.partition_ids],
        )
        self.frozen_mip = (
            blocks.view(self.by, self.bx, 4, 4, 3)
            .permute(4, 0, 2, 1, 3)
            .reshape(3, self.h, self.w)
        )


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
        return [m.decode_mip(hard_partition=hard_partition) for m in self.mips]

    def sample(
        self, uv: torch.Tensor, lod: torch.Tensor, hard_partition: bool
    ) -> torch.Tensor:
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


class MaterialDecoderMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.quantized_forward = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized_forward:
            # Exact FP16 storage values in forward, straight-through gradients for tuning.
            q = lambda p: p + (p.half().float() - p).detach()
            return F.linear(
                F.relu(F.linear(x, q(self.fc1.weight), q(self.fc1.bias))),
                q(self.fc2.weight),
                q(self.fc2.bias),
            )
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
        if len(latent_resolutions) != 4 or len(latent_mips) != 4:
            raise ValueError("Runtime contract requires four feature pyramids")
        if out_channels != 8 or hidden_dim != 16:
            raise ValueError(
                "Current Vulkan contract requires decoder dimensions 12-16-8"
            )
        if endpoint_bits != 6 or index_bits != 3 or bc6_signed_mode:
            raise ValueError("Only unsigned BC6H Mode 10 (6/3 bits) is supported")
        for r, m in zip(latent_resolutions, latent_mips):
            if r < 4 or r & (r - 1) or not 1 <= m <= int(math.log2(r)) - 1:
                raise ValueError(
                    "Feature resolutions must be powers of two and mips must end at or above 4x4"
                )
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

        self.decoder = MaterialDecoderMLP(
            in_dim=self.n_latent * 3, hidden=hidden_dim, out_dim=out_channels
        )

        # lod bias b_i = log2(max(h_i/h, w_i/w))
        self.lod_biases = [
            math.log2(max(r / ref_base_res, r / ref_base_res))
            for r in latent_resolutions
        ]
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

    def _collect_latents_warmup(
        self, uv: torch.Tensor, lod: torch.Tensor
    ) -> torch.Tensor:
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
