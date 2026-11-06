"""Exact BC6H Mode 10 bit packing and decoding. No optimizer or model logic."""

from __future__ import annotations
from pathlib import Path
from typing import List
import math
import numpy as np
import torch
from bc6h_spec import (
    BC67_WEIGHT_MAX,
    BC67_WEIGHT_ROUND,
    BC67_WEIGHT_SHIFT,
    BC6H_MODE10_DESCRIPTOR,
    BC6H_MODE10_ENDPOINT_BITS,
    BC6H_MODE10_FIXUP_INDICES,
    BC6H_MODE10_HEADER_BITS,
    BC6H_MODE10_INDEX_BITS,
    BC6H_MODE10_MODE,
    BC6H_MODE10_PARTITION_TABLE,
    BC6H_MODE10_WEIGHTS,
)
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


_BC6H_MODE10_FIELD_TO_ENDPOINT = {
    "RW": (0, 0),
    "GW": (0, 1),
    "BW": (0, 2),
    "RX": (1, 0),
    "GX": (1, 1),
    "BX": (1, 2),
    "RY": (2, 0),
    "GY": (2, 1),
    "BY": (2, 2),
    "RZ": (3, 0),
    "GZ": (3, 1),
    "BZ": (3, 2),
}


def decode_quantized_params_to_mip(
    qp: dict, partition_bank: torch.Tensor
) -> torch.Tensor:
    """
    Decodes exported Mode 10 quantized block params back to [3,H,W] latent mip.
    """
    if np is None:
        raise RuntimeError("numpy is required for BC6H Mode 10 decoding.")
    mip = _decode_mode10_params_to_mip(
        params=qp,
        partition_bank=partition_bank,
        signed_mode=bool(qp.get("signed_mode", False)),
    )
    return torch.from_numpy(mip).permute(2, 0, 1)


def _write_bc6h_dds(
    mip_bytes_list: List[bytes],
    w0: int,
    h0: int,
    out_path: Path,
    signed_mode: bool,
):
    """Write a BC6H DDS file with a full mip chain."""
    mip_count = len(mip_bytes_list)
    if min(w0, h0) < 1 or not 1 <= mip_count <= int(math.log2(max(w0, h0))) + 1:
        raise ValueError("Invalid DDS dimensions or mip count")
    for level, payload in enumerate(mip_bytes_list):
        expected = (
            ((max(1, w0 >> level) + 3) // 4) * ((max(1, h0 >> level) + 3) // 4) * 16
        )
        if len(payload) != expected:
            raise ValueError("Incorrect BC6H mip payload size")
    dxgi_format = _DXGI_BC6H_SF16 if signed_mode else _DXGI_BC6H_UF16
    bw0, bh0 = max(1, (w0 + 3) // 4), max(1, (h0 + 3) // 4)
    linear_size = bw0 * bh0 * 16

    flags = (
        _DDSD_CAPS | _DDSD_HEIGHT | _DDSD_WIDTH | _DDSD_PIXELFORMAT | _DDSD_LINEARSIZE
    )
    if mip_count > 1:
        flags |= _DDSD_MIPMAPCOUNT
    caps = _DDSCAPS_TEXTURE
    if mip_count > 1:
        caps |= _DDSCAPS_COMPLEX | _DDSCAPS_MIPMAP

    dds_header = _struct.pack(
        "<IIIIIII11I",
        124,
        flags,
        h0,
        w0,
        linear_size,
        0,
        mip_count,
        *([0] * 11),
    )
    dds_pixelformat = _struct.pack(
        "<II4sIIIII", 32, _DDPF_FOURCC, b"DX10", 0, 0, 0, 0, 0
    )
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
    params: dict, signed_mode: bool, partition_bank: np.ndarray = None
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
    idx_q = params["indices_q"].numpy()  # [NB, 16] uint8
    part_id = params["partition_id"].numpy()  # [NB] uint8
    count = ((int(params["w"]) + 3) // 4) * ((int(params["h"]) + 3) // 4)
    if (
        ep_q.shape != (count, 4, 3)
        or idx_q.shape != (count, 16)
        or part_id.shape != (count,)
    ):
        raise ValueError("Block arrays disagree with mip dimensions")
    if np.any(part_id < 0) or np.any(part_id > 31):
        raise ValueError("Invalid partition ID")

    if partition_bank is None:
        partition_bank = np.asarray(BC6H_MODE10_PARTITION_TABLE, dtype=np.uint8)

    out = bytearray()
    for bi in range(ep_q.shape[0]):
        # Get quantized endpoints for this block (already 6-bit)
        endpoints_block = ep_q[bi]
        indices_block = idx_q[bi]
        partition_block = int(part_id[bi])

        block_bytes = _pack_bc6h_mode10_block(
            endpoints_block, indices_block, partition_block, partition_bank, signed_mode
        )
        out.extend(block_bytes)

    return bytes(out)


def _decode_mode10_params_to_mip(
    params: dict,
    partition_bank: np.ndarray = None,
    signed_mode: bool = False,
) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required for BC6H Mode 10 decoding.")
    bank = _bc6h_mode10_partition_bank_np(partition_bank)
    h = int(params["h"])
    w = int(params["w"])
    bw = w // 4
    bh = h // 4
    pixels = np.zeros((h, w, 3), dtype=np.float32)

    endpoints = params["endpoints_q"].numpy()
    indices = params["indices_q"].numpy()
    partition_ids = params["partition_id"].numpy()

    for by in range(bh):
        for bx_i in range(bw):
            bi = by * bw + bx_i
            texels = _bc6h_mode10_decode_from_quantized(
                endpoints_q=endpoints[bi],
                indices_q=indices[bi],
                partition_id=int(partition_ids[bi]),
                partition_bank=bank,
                signed_mode=signed_mode,
            )
            y0, x0 = by * 4, bx_i * 4
            pixels[y0 : y0 + 4, x0 : x0 + 4] = texels
    return pixels


def _decode_mode10_bytes_to_mip(
    block_bytes: bytes,
    w: int,
    h: int,
    partition_bank: np.ndarray = None,
    signed_mode: bool = False,
) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required for BC6H Mode 10 decoding.")
    bank = _bc6h_mode10_partition_bank_np(partition_bank)
    bw = w // 4
    bh = h // 4
    pixels = np.zeros((h, w, 3), dtype=np.float32)

    for by in range(bh):
        for bx_i in range(bw):
            bi = by * bw + bx_i
            texels = _decode_bc6h_mode10_block(
                block_bytes[bi * 16 : (bi + 1) * 16],
                partition_bank=bank,
                signed_mode=signed_mode,
            )
            y0, x0 = by * 4, bx_i * 4
            pixels[y0 : y0 + 4, x0 : x0 + 4] = texels
    return pixels


def _bc6h_mode10_partition_bank_np(
    partition_bank: "np.ndarray | torch.Tensor | None",
) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required for BC6H Mode 10 packing.")
    if partition_bank is None:
        return np.asarray(BC6H_MODE10_PARTITION_TABLE, dtype=np.uint8)
    if isinstance(partition_bank, torch.Tensor):
        return partition_bank.detach().to(torch.uint8).cpu().numpy()
    return np.asarray(partition_bank, dtype=np.uint8)


def _bc6h_sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    value &= mask
    return value - (1 << bits) if value & sign_bit else value


def _bc6h_unquantize(comp: int, bits_per_comp: int, signed_mode: bool) -> int:
    if signed_mode:
        if bits_per_comp >= 16:
            return int(comp)
        sign = 1 if comp < 0 else 0
        comp = -comp if comp < 0 else comp
        if comp == 0:
            out = 0
        elif comp >= ((1 << (bits_per_comp - 1)) - 1):
            out = 0x7FFF
        else:
            out = ((comp << 15) + 0x4000) >> (bits_per_comp - 1)
        return -out if sign else out
    if bits_per_comp >= 15:
        return int(comp)
    if comp == 0:
        return 0
    if comp == ((1 << bits_per_comp) - 1):
        return 0xFFFF
    return ((int(comp) << 16) + 0x8000) >> bits_per_comp


def _bc6h_finish_unquantize(comp: int, signed_mode: bool) -> int:
    if signed_mode:
        return -(((-comp) * 31) >> 5) if comp < 0 else (comp * 31) >> 5
    return (comp * 31) >> 6


def _bc6h_int_to_half_float(comp: int, signed_mode: bool) -> float:
    if np is None:
        raise RuntimeError("numpy is required for BC6H Mode 10 decoding.")
    if signed_mode:
        sign = 0x8000 if comp < 0 else 0
        magnitude = min(abs(int(comp)), 0x7BFF)
        bits = sign | magnitude
    else:
        bits = min(max(int(comp), 0), 0xFFFF)
    return np.asarray([bits], dtype=np.uint16).view(np.float16).astype(np.float32)[0]


def _bc6h_mode10_prepare_block(
    endpoints_q: "np.ndarray",
    indices_q: "np.ndarray",
    partition_id: int,
    partition_bank: "np.ndarray | torch.Tensor | None",
    signed_mode: bool,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    bank = _bc6h_mode10_partition_bank_np(partition_bank)
    if not (0 <= partition_id < len(bank)):
        raise ValueError(
            f"Partition ID {partition_id} out of range [0, {len(bank) - 1}]"
        )

    endpoints = np.asarray(endpoints_q, dtype=np.int32).copy()
    indices_raw = np.asarray(indices_q)
    if endpoints.shape != (4, 3) or indices_raw.shape != (16,):
        raise ValueError("A block requires four RGB endpoints and sixteen indices")
    if np.any(indices_raw < 0) or np.any(indices_raw > 7):
        raise ValueError("Mode 10 indices must be in [0,7]")
    indices = indices_raw.astype(np.uint8).copy()
    mask = bank[partition_id]

    if signed_mode:
        if endpoints.min() < -31 or endpoints.max() > 31:
            raise ValueError("BC6H Mode 10 signed endpoints must be in [-31, 31].")
    else:
        if endpoints.min() < 0 or endpoints.max() > (
            (1 << BC6H_MODE10_ENDPOINT_BITS) - 1
        ):
            raise ValueError("BC6H Mode 10 unsigned endpoints must be in [0, 63].")

    fixup_positions = (0, BC6H_MODE10_FIXUP_INDICES[partition_id])
    for subset, anchor in enumerate(fixup_positions):
        if int(mask[anchor]) != subset:
            raise ValueError(
                f"Partition {partition_id} fix-up anchor mismatch at texel {anchor}."
            )
        if indices[anchor] & (1 << (BC6H_MODE10_INDEX_BITS - 1)):
            a = subset * 2
            b = a + 1
            endpoints[[a, b]] = endpoints[[b, a]]
            subset_mask = mask == subset
            indices[subset_mask] = ((1 << BC6H_MODE10_INDEX_BITS) - 1) - indices[
                subset_mask
            ]

    return endpoints, indices, mask


def _bc6h_mode10_decode_from_quantized(
    endpoints_q: "np.ndarray",
    indices_q: "np.ndarray",
    partition_id: int,
    partition_bank: "np.ndarray | torch.Tensor | None",
    signed_mode: bool,
) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required for BC6H Mode 10 decoding.")
    endpoints, indices, mask = _bc6h_mode10_prepare_block(
        endpoints_q=endpoints_q,
        indices_q=indices_q,
        partition_id=partition_id,
        partition_bank=partition_bank,
        signed_mode=signed_mode,
    )
    out = np.empty((16, 3), dtype=np.float32)
    for texel in range(16):
        subset = int(mask[texel])
        ep0 = endpoints[subset * 2]
        ep1 = endpoints[subset * 2 + 1]
        weight = BC6H_MODE10_WEIGHTS[int(indices[texel])]
        for channel in range(3):
            c0 = _bc6h_unquantize(
                int(ep0[channel]), BC6H_MODE10_ENDPOINT_BITS, signed_mode
            )
            c1 = _bc6h_unquantize(
                int(ep1[channel]), BC6H_MODE10_ENDPOINT_BITS, signed_mode
            )
            interp = (
                c0 * (BC67_WEIGHT_MAX - weight) + c1 * weight + BC67_WEIGHT_ROUND
            ) >> BC67_WEIGHT_SHIFT
            interp = _bc6h_finish_unquantize(interp, signed_mode)
            out[texel, channel] = _bc6h_int_to_half_float(interp, signed_mode)
    return out.reshape(4, 4, 3)


def _decode_bc6h_mode10_block(
    block_bytes: bytes,
    partition_bank: "np.ndarray | torch.Tensor | None",
    signed_mode: bool,
) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required for BC6H Mode 10 decoding.")
    if len(block_bytes) != 16:
        raise ValueError("BC6H blocks must contain exactly 16 bytes")
    value = int.from_bytes(block_bytes, byteorder="little", signed=False)
    partition_id = 0
    endpoints_q = np.zeros((4, 3), dtype=np.int32)

    for bit_pos, (field, field_bit) in enumerate(BC6H_MODE10_DESCRIPTOR):
        bit = (value >> bit_pos) & 0x1
        if field == "M":
            continue
        if field == "D":
            partition_id |= bit << field_bit
            continue
        endpoint_index, channel_index = _BC6H_MODE10_FIELD_TO_ENDPOINT[field]
        endpoints_q[endpoint_index, channel_index] |= bit << field_bit

    mode = 0
    for bit_pos, (field, field_bit) in enumerate(BC6H_MODE10_DESCRIPTOR):
        if field == "M":
            mode |= ((value >> bit_pos) & 0x1) << field_bit
    if mode != BC6H_MODE10_MODE:
        raise ValueError(
            f"Expected BC6H Mode 10 ({BC6H_MODE10_MODE}), got mode {mode}."
        )

    if signed_mode:
        for endpoint_index in range(4):
            for channel_index in range(3):
                endpoints_q[endpoint_index, channel_index] = _bc6h_sign_extend(
                    int(endpoints_q[endpoint_index, channel_index]),
                    BC6H_MODE10_ENDPOINT_BITS,
                )

    bank = _bc6h_mode10_partition_bank_np(partition_bank)
    if not (0 <= partition_id < len(bank)):
        raise ValueError(
            f"Partition ID {partition_id} out of range [0, {len(bank) - 1}]"
        )

    indices = np.zeros((16,), dtype=np.uint8)
    bit_pos = BC6H_MODE10_HEADER_BITS
    fixup_positions = {0, BC6H_MODE10_FIXUP_INDICES[partition_id]}
    for texel in range(16):
        bit_count = (
            BC6H_MODE10_INDEX_BITS - 1
            if texel in fixup_positions
            else BC6H_MODE10_INDEX_BITS
        )
        indices[texel] = (value >> bit_pos) & ((1 << bit_count) - 1)
        bit_pos += bit_count
    if bit_pos != 128:
        raise ValueError(
            f"Invalid BC6H Mode 10 index payload length: ended at bit {bit_pos}."
        )

    return _bc6h_mode10_decode_from_quantized(
        endpoints_q=endpoints_q,
        indices_q=indices,
        partition_id=partition_id,
        partition_bank=bank,
        signed_mode=signed_mode,
    )


def read_dds(path: Path) -> tuple[dict, list[bytes]]:
    raw = Path(path).read_bytes()
    if (
        len(raw) < 148
        or raw[:4] != b"DDS "
        or _struct.unpack_from("<I", raw, 4)[0] != 124
    ):
        raise ValueError(f"Invalid or truncated DDS: {path}")
    if _struct.unpack_from("<I", raw, 76)[0] != 32 or raw[84:88] != b"DX10":
        raise ValueError("Expected DX10 DDS header")
    h, w = _struct.unpack_from("<II", raw, 12)
    count = _struct.unpack_from("<I", raw, 28)[0]
    fmt, dimension, misc, array_size, misc2 = _struct.unpack_from("<5I", raw, 128)
    if (
        fmt not in (95, 96)
        or dimension != 3
        or array_size != 1
        or misc != 0
        or misc2 != 0
    ):
        raise ValueError("Expected a single 2D BC6H DDS texture")
    if min(h, w) < 1 or not 1 <= count <= int(math.log2(max(h, w))) + 1:
        raise ValueError("Invalid DDS dimensions or mip count")
    offset = 148
    payloads = []
    shapes = []
    for i in range(count):
        mh = max(1, h >> i)
        mw = max(1, w >> i)
        size = ((mh + 3) // 4) * ((mw + 3) // 4) * 16
        if offset + size > len(raw):
            raise ValueError("Truncated DDS mip payload")
        payloads.append(raw[offset : offset + size])
        shapes.append((mh, mw))
        offset += size
    if offset != len(raw):
        raise ValueError("Unexpected trailing bytes in DDS")
    return {
        "w": w,
        "h": h,
        "mip_count": count,
        "signed_mode": fmt == 96,
        "shapes": shapes,
    }, payloads


def decode_bc6h_dds(path: Path, signed_mode=None) -> list[torch.Tensor]:
    info, payloads = read_dds(path)
    if signed_mode is not None and bool(signed_mode) != info["signed_mode"]:
        raise ValueError("DDS signedness disagrees with metadata")
    result = []
    for payload, (h, w) in zip(payloads, info["shapes"]):
        pixels = np.zeros((h, w, 3), np.float32)
        bw = (w + 3) // 4
        for by in range((h + 3) // 4):
            for bx in range(bw):
                bi = by * bw + bx
                block = _decode_bc6h_mode10_block(
                    payload[bi * 16 : (bi + 1) * 16], None, info["signed_mode"]
                )
                rh = min(4, h - by * 4)
                rw = min(4, w - bx * 4)
                pixels[by * 4 : by * 4 + rh, bx * 4 : bx * 4 + rw] = block[:rh, :rw]
        result.append(torch.from_numpy(pixels).permute(2, 0, 1).contiguous())
    return result


def decode_bc6h_dds_mip0(dds_path: Path, signed_mode: bool) -> torch.Tensor:
    return decode_bc6h_dds(dds_path, signed_mode)[0]


def _pack_bc6h_mode10_block(
    endpoints_q: np.ndarray,
    indices_q: np.ndarray,
    partition_id: int,
    partition_bank: np.ndarray = None,
    signed_mode: bool = False,
) -> bytes:
    if np is None:
        raise RuntimeError("numpy is required for BC6H Mode 10 packing.")

    endpoints, indices, _ = _bc6h_mode10_prepare_block(
        endpoints_q=endpoints_q,
        indices_q=indices_q,
        partition_id=partition_id,
        partition_bank=partition_bank,
        signed_mode=signed_mode,
    )

    block = 0
    for bit_pos, (field, field_bit) in enumerate(BC6H_MODE10_DESCRIPTOR):
        if field == "M":
            bit_value = (BC6H_MODE10_MODE >> field_bit) & 0x1
        elif field == "D":
            bit_value = (partition_id >> field_bit) & 0x1
        else:
            endpoint_index, channel_index = _BC6H_MODE10_FIELD_TO_ENDPOINT[field]
            bit_value = (
                int(endpoints[endpoint_index, channel_index]) >> field_bit
            ) & 0x1
        block |= bit_value << bit_pos

    bit_pos = BC6H_MODE10_HEADER_BITS
    fixup_positions = {0, BC6H_MODE10_FIXUP_INDICES[partition_id]}
    for texel in range(16):
        bit_count = (
            BC6H_MODE10_INDEX_BITS - 1
            if texel in fixup_positions
            else BC6H_MODE10_INDEX_BITS
        )
        block |= (int(indices[texel]) & ((1 << bit_count) - 1)) << bit_pos
        bit_pos += bit_count
    if bit_pos != 128:
        raise ValueError(
            f"Invalid BC6H Mode 10 packed bit count: ended at bit {bit_pos}."
        )

    return block.to_bytes(16, byteorder="little", signed=False)
