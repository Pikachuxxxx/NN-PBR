#!/usr/bin/env python3
"""
Pure-Python BC6H encoder/decoder for latent texture export.

Encodes exported latent tensors ([3,H,W] float32 in ~[-1,1]) directly into
BC6H_UF16 or BC6H_SF16 DDS files with a full mip chain.

No external binary tools required.

BC6H Mode 11 (single-subset, 11-bit endpoints) is used throughout.

Reference:
  Microsoft DirectX documentation — BC6H format
  https://docs.microsoft.com/en-us/windows/win32/direct3d11/bc6h-format

BC6H Mode 11 block layout (128 bits total):
  bits [4:0]   = mode = 5'b00111  (= 0x07 stored LSB-first)
  bits [14:5]  = RW[9:0]  (endpoint 0, R bits 9..0)
  bits [24:15] = RX[9:0]  (endpoint 1, R bits 9..0)
  bits [34:25] = GW[9:0]  (endpoint 0, G bits 9..0)
  bits [44:35] = GX[9:0]  (endpoint 1, G bits 9..0)
  bits [54:45] = BW[9:0]  (endpoint 0, B bits 9..0)
  bits [64:55] = BX[9:0]  (endpoint 1, B bits 9..0)
  bit  [65]    = RW[10]
  bit  [66]    = RX[10]
  bit  [67]    = GW[10]
  bit  [68]    = GX[10]
  bit  [69]    = BW[10]
  bit  [70]    = BX[10]
  bits [72:71] = index[0]  (anchor texel, 2 bits, MSB is always 0 so only 2 bits stored)
  bits [75:73] = index[1]  (3 bits)
  ...
  bits [127:118] = index[15] (3 bits), padding if needed

  Total endpoint bits: 5 + 6*11 = 5 + 66 = 71 bits
  Index bits: 2 + 15*3 = 2 + 45 = 47 bits
  Total: 71 + 47 = 118 bits → padded to 128 with 10 zero bits

  Interpolation weights for 3-bit indices (BC6H spec Table BC6H_WEIGHTS3):
    index 0 → weight 0    (pure ep0)
    index 1 → weight 9
    index 2 → weight 18
    index 3 → weight 27
    index 4 → weight 37
    index 5 → weight 46
    index 6 → weight 55
    index 7 → weight 64   (pure ep1)

  Interpolated value: (ep0*(64-w) + ep1*w + 32) >> 6
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# DDS constants
# ---------------------------------------------------------------------------
DDS_MAGIC = b"DDS "
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000

DDSCAPS_COMPLEX = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS_MIPMAP = 0x400000

DDPF_FOURCC = 0x4

DXGI_FORMAT_R16G16B16A16_FLOAT = 10
DXGI_FORMAT_BC6H_UF16 = 95
DXGI_FORMAT_BC6H_SF16 = 96
D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3


# ---------------------------------------------------------------------------
# BC6H interpolation constants and helpers
# ---------------------------------------------------------------------------

# BC6H_WEIGHTS3: BC6H interpolation weights for 3-bit indices.
# From DirectX spec table BC6H_WEIGHTS3. Scaled to 64 (0..64).
# index 0 = 0 (pure ep0), index 7 = 64 (pure ep1).
_BC6H_WEIGHTS3 = [0, 9, 18, 27, 37, 46, 55, 64]

# BC6H endpoint ranges:
#   UF16 mode: 11-bit unsigned, endpoints in [0, 2047], dequantized to [0, 1].
#   SF16 mode: 11-bit signed (two's complement), endpoints in [-1023, 1023],
#              dequantized to [-1, 1].
_BC6H_EP_LEVELS = 2047    # 11-bit unsigned max
_BC6H_SF16_EP_MAX = 1023  # 11-bit signed max (spec clamps to ±1023, not ±1024)
_BC6H_SF16_MAX = 32767.0  # kept for reference; not used in encoder directly


def _signed_11bit_to_uint11(val: int) -> int:
    """Convert a signed 11-bit integer (range -1023..1023) to unsigned 11-bit for packing."""
    if val < 0:
        return val & 0x7FF  # two's complement, 11 bits
    return val & 0x7FF


# ---------------------------------------------------------------------------
# BC6H Mode 11 block encoder
# ---------------------------------------------------------------------------

def _pack_bc6h_mode11_block(
    ep0: Tuple[int, int, int],
    ep1: Tuple[int, int, int],
    indices: List[int],
    signed_mode: bool,
) -> bytes:
    """
    Pack a BC6H Mode 11 block into 16 bytes.

    ep0, ep1: (R, G, B) endpoint values.
      - UF16 mode: integers in [0, 2047] (11-bit unsigned)
      - SF16 mode: integers in [-1023, 1023] (11-bit signed)

    indices: list of 16 texel index values (3 bits each, 0..7).
             indices[0] is the anchor texel (must have MSB = 0, i.e., index[0] < 4).

    Returns 16 bytes (little-endian BC6H block).
    """
    # Convert signed endpoints to 11-bit two's-complement unsigned for packing
    if signed_mode:
        rw = _signed_11bit_to_uint11(ep0[0])
        gw = _signed_11bit_to_uint11(ep0[1])
        bw = _signed_11bit_to_uint11(ep0[2])
        rx = _signed_11bit_to_uint11(ep1[0])
        gx = _signed_11bit_to_uint11(ep1[1])
        bx = _signed_11bit_to_uint11(ep1[2])
    else:
        rw = int(ep0[0]) & 0x7FF
        gw = int(ep0[1]) & 0x7FF
        bw = int(ep0[2]) & 0x7FF
        rx = int(ep1[0]) & 0x7FF
        gx = int(ep1[1]) & 0x7FF
        bx = int(ep1[2]) & 0x7FF

    # Build 128-bit block as a Python integer (LSB = bit 0)
    block = 0

    # bits [4:0] = mode 11 = 0b00111 = 7
    block |= 0x7  # bits 0-4

    # bits [14:5]  = RW[9:0]
    block |= ((rw & 0x3FF) << 5)
    # bits [24:15] = RX[9:0]
    block |= ((rx & 0x3FF) << 15)
    # bits [34:25] = GW[9:0]
    block |= ((gw & 0x3FF) << 25)
    # bits [44:35] = GX[9:0]
    block |= ((gx & 0x3FF) << 35)
    # bits [54:45] = BW[9:0]
    block |= ((bw & 0x3FF) << 45)
    # bits [64:55] = BX[9:0]
    block |= ((bx & 0x3FF) << 55)

    # bit [65] = RW[10]
    block |= (((rw >> 10) & 1) << 65)
    # bit [66] = RX[10]
    block |= (((rx >> 10) & 1) << 66)
    # bit [67] = GW[10]
    block |= (((gw >> 10) & 1) << 67)
    # bit [68] = GX[10]
    block |= (((gx >> 10) & 1) << 68)
    # bit [69] = BW[10]
    block |= (((bw >> 10) & 1) << 69)
    # bit [70] = BX[10]
    block |= (((bx >> 10) & 1) << 70)

    # bits [72:71] = index[0] (anchor: 2 bits, MSB implicit 0 so store only bits[1:0])
    # Note: the anchor index must be < 4 (MSB=0). We enforce this before calling.
    bit_pos = 71
    anchor_idx = int(indices[0]) & 0x3  # 2 bits for anchor
    block |= (anchor_idx << bit_pos)
    bit_pos += 2

    # bits [75:73], [78:76], ... = indices 1..15 (3 bits each)
    for i in range(1, 16):
        idx = int(indices[i]) & 0x7
        block |= (idx << bit_pos)
        bit_pos += 3

    # bit_pos should now be 71 + 2 + 45 = 118. Remaining bits 118..127 are zero (padding).
    assert bit_pos == 118, f"Unexpected bit_pos={bit_pos}"

    # Pack to 16 bytes little-endian
    return block.to_bytes(16, byteorder="little")


def _bc6h_interp_uf16(ep0_int: int, ep1_int: int, w: int) -> int:
    """
    BC6H unsigned interpolation in integer space.

    ep0_int, ep1_int: 11-bit unsigned endpoint values [0, 2047].
    w: interpolation weight from _BC6H_WEIGHTS3 (0..64).
    Returns interpolated value in [0, 2047].
    """
    return (ep0_int * (64 - w) + ep1_int * w + 32) >> 6


def _bc6h_interp_sf16(ep0_int: int, ep1_int: int, w: int) -> int:
    """
    BC6H signed interpolation in integer space.

    ep0_int, ep1_int: 11-bit signed endpoint values [-1023, 1023].
    w: interpolation weight from _BC6H_WEIGHTS3 (0..64).
    Returns interpolated value in [-1023, 1023].
    """
    return (ep0_int * (64 - w) + ep1_int * w + 32) >> 6


def _find_best_index_uf16_int(px_int: np.ndarray, ep0_int: np.ndarray, ep1_int: np.ndarray) -> int:
    """
    Find the best 3-bit BC6H index for a pixel given quantized endpoints.

    px_int: [3] int array, quantized pixel value in [0, 2047].
    ep0_int, ep1_int: [3] int arrays, quantized endpoints.

    Returns best index in [0, 7].
    """
    best_idx = 0
    best_err = float("inf")
    for idx in range(8):
        w = _BC6H_WEIGHTS3[idx]
        # Compute interpolated RGB in integer space
        r_rec = _bc6h_interp_uf16(int(ep0_int[0]), int(ep1_int[0]), w)
        g_rec = _bc6h_interp_uf16(int(ep0_int[1]), int(ep1_int[1]), w)
        b_rec = _bc6h_interp_uf16(int(ep0_int[2]), int(ep1_int[2]), w)
        # L2 error in integer space
        dr = r_rec - int(px_int[0])
        dg = g_rec - int(px_int[1])
        db = b_rec - int(px_int[2])
        err = dr * dr + dg * dg + db * db
        if err < best_err:
            best_err = err
            best_idx = idx
    return best_idx


def _find_best_index_sf16_int(px_int: np.ndarray, ep0_int: np.ndarray, ep1_int: np.ndarray) -> int:
    """
    Find the best 3-bit BC6H index for a pixel given signed quantized endpoints.

    px_int: [3] int array, quantized pixel value in [-1023, 1023].
    ep0_int, ep1_int: [3] int arrays, quantized endpoints.

    Returns best index in [0, 7].
    """
    best_idx = 0
    best_err = float("inf")
    for idx in range(8):
        w = _BC6H_WEIGHTS3[idx]
        r_rec = _bc6h_interp_sf16(int(ep0_int[0]), int(ep1_int[0]), w)
        g_rec = _bc6h_interp_sf16(int(ep0_int[1]), int(ep1_int[1]), w)
        b_rec = _bc6h_interp_sf16(int(ep0_int[2]), int(ep1_int[2]), w)
        dr = r_rec - int(px_int[0])
        dg = g_rec - int(px_int[1])
        db = b_rec - int(px_int[2])
        err = dr * dr + dg * dg + db * db
        if err < best_err:
            best_err = err
            best_idx = idx
    return best_idx


def _encode_bc6h_uf16_block(pixels_4x4_rgb_fp32: np.ndarray) -> bytes:
    """
    Encode a 4x4 block of RGB pixels to a BC6H_UF16 Mode 11 block (16 bytes).

    pixels_4x4_rgb_fp32: shape [16, 3], float32, values already in [0, 1] range.

    Returns 16 bytes.
    """
    # Clamp to valid range for UF16 (non-negative, max 1.0 for our normalized range)
    px = np.clip(pixels_4x4_rgb_fp32, 0.0, 1.0).astype(np.float64)

    # Quantize all 16 pixels to 11-bit integers [0, 2047]
    px_q = np.round(px * _BC6H_EP_LEVELS).astype(np.int32).clip(0, _BC6H_EP_LEVELS)  # [16, 3]

    # Find min/max quantized values per channel for endpoints
    ep0_q = px_q.min(axis=0)  # [3]
    ep1_q = px_q.max(axis=0)  # [3]

    # Compute per-texel best indices in integer space
    indices = []
    for tex_i in range(16):
        idx = _find_best_index_uf16_int(px_q[tex_i], ep0_q, ep1_q)
        indices.append(idx)

    # BC6H anchor texel (index 0) must have MSB = 0 (i.e., index[0] <= 3).
    # If index[0] > 3, swap endpoints and invert all indices.
    if indices[0] > 3:
        ep0_q, ep1_q = ep1_q, ep0_q
        indices = [7 - idx for idx in indices]

    # Safety clamp for anchor (shouldn't be needed after swap, but be defensive)
    if indices[0] > 3:
        indices[0] = 3

    ep0_tuple = (int(ep0_q[0]), int(ep0_q[1]), int(ep0_q[2]))
    ep1_tuple = (int(ep1_q[0]), int(ep1_q[1]), int(ep1_q[2]))

    return _pack_bc6h_mode11_block(ep0_tuple, ep1_tuple, indices, signed_mode=False)


def _encode_bc6h_sf16_block(pixels_4x4_rgb_fp32: np.ndarray) -> bytes:
    """
    Encode a 4x4 block of RGB pixels to a BC6H_SF16 Mode 11 block (16 bytes).

    pixels_4x4_rgb_fp32: shape [16, 3], float32, values in [-1, 1] range.

    Returns 16 bytes.
    """
    # Clamp to valid signed range
    px = np.clip(pixels_4x4_rgb_fp32, -1.0, 1.0).astype(np.float64)

    # Quantize all 16 pixels to 11-bit signed integers [-1023, 1023]
    px_q = np.round(px * _BC6H_SF16_EP_MAX).astype(np.int32).clip(-_BC6H_SF16_EP_MAX, _BC6H_SF16_EP_MAX)

    # Find min/max quantized values per channel for endpoints
    ep0_q = px_q.min(axis=0)  # [3]
    ep1_q = px_q.max(axis=0)  # [3]

    # Compute per-texel best indices in integer space
    indices = []
    for tex_i in range(16):
        idx = _find_best_index_sf16_int(px_q[tex_i], ep0_q, ep1_q)
        indices.append(idx)

    # Ensure anchor texel index MSB = 0
    if indices[0] > 3:
        ep0_q, ep1_q = ep1_q, ep0_q
        indices = [7 - idx for idx in indices]

    if indices[0] > 3:
        indices[0] = 3

    ep0_tuple = (int(ep0_q[0]), int(ep0_q[1]), int(ep0_q[2]))
    ep1_tuple = (int(ep1_q[0]), int(ep1_q[1]), int(ep1_q[2]))

    return _pack_bc6h_mode11_block(ep0_tuple, ep1_tuple, indices, signed_mode=True)


# ---------------------------------------------------------------------------
# BC6H Mode 11 block decoder
# ---------------------------------------------------------------------------

def _unpack_bc6h_mode11_block_uf16(block_16bytes: bytes) -> np.ndarray:
    """
    Decode a BC6H_UF16 Mode 11 block (16 bytes) to 4x4 RGB float32 pixels.

    Interpolation is performed in integer space per the BC6H spec:
      result_int = (ep0*(64-w) + ep1*w + 32) >> 6
    then dequantized to [0, 1] as result_int / 2047.0.

    Returns ndarray shape [16, 3], float32, values in [0, 1].
    """
    block = int.from_bytes(block_16bytes, byteorder="little")

    def bits(hi: int, lo: int) -> int:
        mask = (1 << (hi - lo + 1)) - 1
        return (block >> lo) & mask

    # Verify mode bits [4:0] == 7
    mode = bits(4, 0)
    if mode != 7:
        raise ValueError(f"Expected BC6H Mode 11 (mode=7), got mode={mode} (block may not be Mode 11)")

    # Extract low 10 bits of each endpoint component
    rw_lo = bits(14, 5)
    rx_lo = bits(24, 15)
    gw_lo = bits(34, 25)
    gx_lo = bits(44, 35)
    bw_lo = bits(54, 45)
    bx_lo = bits(64, 55)

    # Extract high bit (bit 10) of each endpoint component
    rw_hi = bits(65, 65)
    rx_hi = bits(66, 66)
    gw_hi = bits(67, 67)
    gx_hi = bits(68, 68)
    bw_hi = bits(69, 69)
    bx_hi = bits(70, 70)

    # Reassemble 11-bit unsigned endpoint values [0, 2047]
    rw = (rw_hi << 10) | rw_lo
    rx = (rx_hi << 10) | rx_lo
    gw = (gw_hi << 10) | gw_lo
    gx = (gx_hi << 10) | gx_lo
    bw = (bw_hi << 10) | bw_lo
    bx = (bx_hi << 10) | bx_lo

    ep0_int = (rw, gw, bw)
    ep1_int = (rx, gx, bx)

    # Extract indices
    # Index 0: bits [72:71] (2 bits, MSB implicit 0 — anchor has index < 4)
    idx0 = bits(72, 71)
    texel_indices = [idx0]

    # Indices 1..15: 3 bits each starting at bit 73
    bit_pos = 73
    for _ in range(15):
        texel_indices.append(bits(bit_pos + 2, bit_pos))
        bit_pos += 3

    # Reconstruct pixels using integer interpolation, then dequantize
    pixels = np.zeros((16, 3), dtype=np.float32)
    for i in range(16):
        w = _BC6H_WEIGHTS3[texel_indices[i]]
        r_rec = _bc6h_interp_uf16(ep0_int[0], ep1_int[0], w)
        g_rec = _bc6h_interp_uf16(ep0_int[1], ep1_int[1], w)
        b_rec = _bc6h_interp_uf16(ep0_int[2], ep1_int[2], w)
        pixels[i, 0] = float(r_rec) / _BC6H_EP_LEVELS
        pixels[i, 1] = float(g_rec) / _BC6H_EP_LEVELS
        pixels[i, 2] = float(b_rec) / _BC6H_EP_LEVELS

    return pixels


def _unpack_bc6h_mode11_block_sf16(block_16bytes: bytes) -> np.ndarray:
    """
    Decode a BC6H_SF16 Mode 11 block (16 bytes) to 4x4 RGB float32 pixels.

    Interpolation is performed in integer space per the BC6H spec:
      result_int = (ep0*(64-w) + ep1*w + 32) >> 6
    then dequantized to [-1, 1] as result_int / 1023.0.

    Returns ndarray shape [16, 3], float32, values in [-1, 1].
    """
    block = int.from_bytes(block_16bytes, byteorder="little")

    def bits(hi: int, lo: int) -> int:
        mask = (1 << (hi - lo + 1)) - 1
        return (block >> lo) & mask

    mode = bits(4, 0)
    if mode != 7:
        raise ValueError(f"Expected BC6H Mode 11 (mode=7), got mode={mode}")

    rw_lo = bits(14, 5)
    rx_lo = bits(24, 15)
    gw_lo = bits(34, 25)
    gx_lo = bits(44, 35)
    bw_lo = bits(54, 45)
    bx_lo = bits(64, 55)

    rw_hi = bits(65, 65)
    rx_hi = bits(66, 66)
    gw_hi = bits(67, 67)
    gx_hi = bits(68, 68)
    bw_hi = bits(69, 69)
    bx_hi = bits(70, 70)

    # Reassemble 11-bit values (interpret as 11-bit two's complement for SF16)
    def to_signed11(val: int) -> int:
        # 11-bit two's complement: if bit 10 is set, value is negative
        if val & 0x400:
            return val - 0x800
        return val

    rw = to_signed11((rw_hi << 10) | rw_lo)
    rx = to_signed11((rx_hi << 10) | rx_lo)
    gw = to_signed11((gw_hi << 10) | gw_lo)
    gx = to_signed11((gx_hi << 10) | gx_lo)
    bw = to_signed11((bw_hi << 10) | bw_lo)
    bx = to_signed11((bx_hi << 10) | bx_lo)

    ep0_int = (rw, gw, bw)
    ep1_int = (rx, gx, bx)

    idx0 = bits(72, 71)
    texel_indices = [idx0]
    bit_pos = 73
    for _ in range(15):
        texel_indices.append(bits(bit_pos + 2, bit_pos))
        bit_pos += 3

    # Reconstruct pixels using integer interpolation, then dequantize
    pixels = np.zeros((16, 3), dtype=np.float32)
    for i in range(16):
        w = _BC6H_WEIGHTS3[texel_indices[i]]
        r_rec = _bc6h_interp_sf16(ep0_int[0], ep1_int[0], w)
        g_rec = _bc6h_interp_sf16(ep0_int[1], ep1_int[1], w)
        b_rec = _bc6h_interp_sf16(ep0_int[2], ep1_int[2], w)
        pixels[i, 0] = float(r_rec) / _BC6H_SF16_EP_MAX
        pixels[i, 1] = float(g_rec) / _BC6H_SF16_EP_MAX
        pixels[i, 2] = float(b_rec) / _BC6H_SF16_EP_MAX

    return pixels


# ---------------------------------------------------------------------------
# Mip-level encode / decode
# ---------------------------------------------------------------------------

def _encode_bc6h_mip(tex_chw_fp32: np.ndarray, signed_mode: bool) -> bytes:
    """
    Encode a [3, H, W] float32 CHW image to BC6H Mode 11 bytes.

    tex_chw_fp32: numpy array [3, H, W], float32.
      - UF16 mode: values in [0, 1]  (after remap from [-1,1])
      - SF16 mode: values in [-1, 1]

    Returns bytes for the entire mip level (num_blocks_x * num_blocks_y * 16 bytes).
    """
    _, H, W = tex_chw_fp32.shape
    bw = (W + 3) // 4
    bh = (H + 3) // 4

    # Pad to multiple of 4
    pad_h = bh * 4
    pad_w = bw * 4
    if pad_h != H or pad_w != W:
        padded = np.zeros((3, pad_h, pad_w), dtype=np.float32)
        padded[:, :H, :W] = tex_chw_fp32
        tex_chw_fp32 = padded

    encode_fn = _encode_bc6h_sf16_block if signed_mode else _encode_bc6h_uf16_block

    out = bytearray()
    for by in range(bh):
        for bx in range(bw):
            # Extract 4x4 block: [3, 4, 4]
            block_pixels_chw = tex_chw_fp32[:, by*4:(by+1)*4, bx*4:(bx+1)*4]
            # Reshape to [16, 3] (row-major: top-left to bottom-right)
            block_pixels = block_pixels_chw.transpose(1, 2, 0).reshape(16, 3)
            out.extend(encode_fn(block_pixels))

    return bytes(out)


def _decode_bc6h_mip(data: bytes, W: int, H: int, signed_mode: bool) -> np.ndarray:
    """
    Decode BC6H Mode 11 bytes to a [3, H, W] float32 CHW image.

    data: bytes for the mip level.
    W, H: original texture dimensions (blocks may be padded).
    signed_mode: True for SF16, False for UF16.

    Returns numpy array [3, H, W], float32.
    """
    bw = (W + 3) // 4
    bh = (H + 3) // 4
    expected = bw * bh * 16
    if len(data) < expected:
        raise ValueError(f"BC6H mip data too short: got {len(data)}, expected {expected}")

    decode_fn = _unpack_bc6h_mode11_block_sf16 if signed_mode else _unpack_bc6h_mode11_block_uf16

    result = np.zeros((3, bh * 4, bw * 4), dtype=np.float32)
    offset = 0
    for by in range(bh):
        for bx in range(bw):
            block_data = data[offset : offset + 16]
            offset += 16
            pixels = decode_fn(block_data)  # [16, 3]
            block_chw = pixels.reshape(4, 4, 3).transpose(2, 0, 1)  # [3, 4, 4]
            result[:, by*4:(by+1)*4, bx*4:(bx+1)*4] = block_chw

    # Trim to actual dimensions
    return result[:, :H, :W]


# ---------------------------------------------------------------------------
# DDS file writing
# ---------------------------------------------------------------------------

def _write_bc6h_dds(
    mip_bytes_list: List[bytes],
    w0: int,
    h0: int,
    out_path: Path,
    signed_mode: bool,
):
    """
    Write a DDS file with BC6H_UF16 or BC6H_SF16 data and a full mip chain.

    mip_bytes_list: list of raw BC6H bytes per mip level (mip0 first).
    w0, h0: mip0 dimensions.
    out_path: output .dds path.
    signed_mode: True for BC6H_SF16, False for BC6H_UF16.
    """
    mip_count = len(mip_bytes_list)
    dxgi_format = DXGI_FORMAT_BC6H_SF16 if signed_mode else DXGI_FORMAT_BC6H_UF16

    # For BC6H: linearSize = (max(1,(w+3)/4)) * (max(1,(h+3)/4)) * 16
    bw0 = max(1, (w0 + 3) // 4)
    bh0 = max(1, (h0 + 3) // 4)
    linear_size = bw0 * bh0 * 16

    flags = (
        DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
        | DDSD_LINEARSIZE
    )
    if mip_count > 1:
        flags |= DDSD_MIPMAPCOUNT

    caps = DDSCAPS_TEXTURE
    if mip_count > 1:
        caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP

    # DDS_HEADER: 124 bytes
    dds_header = struct.pack(
        "<IIIIIII11I",
        124,          # dwSize
        flags,        # dwFlags
        h0,           # dwHeight
        w0,           # dwWidth
        linear_size,  # dwPitchOrLinearSize (for compressed = linearSize)
        0,            # dwDepth
        mip_count,    # dwMipMapCount
        *([0] * 11),  # dwReserved1[11]
    )

    # DDPF_PIXELFORMAT: 32 bytes
    dds_pixelformat = struct.pack(
        "<II4sIIIII",
        32,           # dwSize
        DDPF_FOURCC,  # dwFlags
        b"DX10",      # dwFourCC
        0,            # dwRGBBitCount
        0,            # dwRBitMask
        0,            # dwGBitMask
        0,            # dwBBitMask
        0,            # dwABitMask
    )

    # DDS_CAPS: 20 bytes (within DDS_HEADER, but packed separately here for clarity)
    dds_caps = struct.pack(
        "<IIIII",
        caps,  # dwCaps
        0,     # dwCaps2
        0,     # dwCaps3
        0,     # dwCaps4
        0,     # dwReserved2
    )

    # DDS_HEADER_DXT10: 20 bytes
    dx10_header = struct.pack(
        "<IIIII",
        dxgi_format,                      # dxgiFormat
        D3D10_RESOURCE_DIMENSION_TEXTURE2D,  # resourceDimension
        0,                                # miscFlag
        1,                                # arraySize
        0,                                # miscFlags2
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(DDS_MAGIC)
        f.write(dds_header)
        f.write(dds_pixelformat)
        f.write(dds_caps)
        f.write(dx10_header)
        for mip_data in mip_bytes_list:
            f.write(mip_data)


# ---------------------------------------------------------------------------
# DDS metadata reader (kept for compat with existing report logic)
# ---------------------------------------------------------------------------

def _read_dds_metadata(path: Path) -> Dict[str, int]:
    with path.open("rb") as f:
        data = f.read(4 + 124 + 20)
    if len(data) < 4 + 124:
        raise RuntimeError(f"Invalid DDS file (too small): {path}")
    if data[:4] != DDS_MAGIC:
        raise RuntimeError(f"Invalid DDS magic: {path}")
    hdr = data[4 : 4 + 124]
    mip_count = struct.unpack_from("<I", hdr, 24)[0]
    height = struct.unpack_from("<I", hdr, 8)[0]
    width = struct.unpack_from("<I", hdr, 12)[0]
    pf_fourcc = hdr[80:84]
    out = {
        "mip_count": int(max(1, mip_count)),
        "width": int(width),
        "height": int(height),
    }
    if pf_fourcc == b"DX10" and len(data) >= 4 + 124 + 20:
        dx10 = data[4 + 124 : 4 + 124 + 20]
        out["dxgi_format"] = int(struct.unpack_from("<I", dx10, 0)[0])
    return out


# ---------------------------------------------------------------------------
# Latent grouping helper
# ---------------------------------------------------------------------------

def _find_latent_dir(export_dir: Path) -> Path:
    """
    Find where .pt latent files live.  The layout may be flat (export_dir/),
    nested (export_dir/latents/), or in metadata/ (v3+).
    """
    for subdir in ["latents", "metadata"]:
        d = export_dir / subdir
        if d.is_dir():
            return d
    return export_dir


def _group_mips_by_latent(
    meta: Dict,
    export_dir: Path,
    latent_index: Optional[int],
    max_latents: int,
) -> List[Tuple[int, List[Dict]]]:
    """Group latent file entries from metadata by latent index."""
    lat_dir = _find_latent_dir(export_dir)
    grouped: Dict[int, List[Dict]] = collections.defaultdict(list)

    for e in meta.get("latent_files", []):
        li = int(e["latent_index"])
        mi = int(e["mip_index"])
        if latent_index is not None and li != latent_index:
            continue
        pt_name = e.get("pt")
        if not pt_name:
            continue
        # If pt_name contains a path separator, it's relative to export_dir (v3)
        if "/" in pt_name or "\\" in pt_name:
            pt_path = export_dir / pt_name
        else:
            # Legacy: simple filename → search in lat_dir then export_dir
            pt_path = lat_dir / pt_name
            if not pt_path.exists():
                pt_path = export_dir / pt_name
        if not pt_path.exists():
            raise RuntimeError(f"Missing latent tensor file: {pt_name} (searched {lat_dir} and {export_dir})")
        grouped[li].append({
            "mip_index": mi,
            "pt_path": pt_path,
            "shape_chw": e.get("shape_chw"),
        })

    items = sorted(grouped.items(), key=lambda kv: kv[0])
    if max_latents > 0:
        items = items[:max_latents]
    for _, arr in items:
        arr.sort(key=lambda x: int(x["mip_index"]))
    return items


# ---------------------------------------------------------------------------
# Verification / diff
# ---------------------------------------------------------------------------

def _compute_psnr(original: np.ndarray, decoded: np.ndarray) -> float:
    """Compute PSNR in dB between two float arrays (same range)."""
    mse = float(np.mean((original.astype(np.float64) - decoded.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return float("inf")
    # For data in [0,1] range, peak=1.0
    return float(10.0 * np.log10(1.0 / mse))


def _verify_and_diff(
    original_chw: np.ndarray,      # [3, H, W] float32, original pixel values
    decoded_chw: np.ndarray,        # [3, H, W] float32, BC6H decoded pixels
    out_dir: Path,
    stem: str,
    signed_mode: bool,
) -> Dict:
    """
    Compute per-channel PSNR / max diff and save a matplotlib diff figure.

    Returns a dict with verification metrics.
    """
    H, W = original_chw.shape[1], original_chw.shape[2]
    channel_names = ["R", "G", "B"]

    # Normalize both to [0,1] for PSNR calculation
    # In UF16 mode both are already [0,1]; in SF16 mode remap [-1,1]->[0,1].
    if signed_mode:
        orig_norm = (original_chw + 1.0) * 0.5
        dec_norm = (decoded_chw + 1.0) * 0.5
    else:
        orig_norm = original_chw.copy()
        dec_norm = decoded_chw.copy()

    orig_norm = np.clip(orig_norm, 0.0, 1.0)
    dec_norm = np.clip(dec_norm, 0.0, 1.0)

    metrics = {}
    for c, name in enumerate(channel_names):
        psnr = _compute_psnr(orig_norm[c], dec_norm[c])
        max_diff = float(np.max(np.abs(orig_norm[c] - dec_norm[c])))
        mean_diff = float(np.mean(np.abs(orig_norm[c] - dec_norm[c])))
        metrics[f"psnr_{name}"] = psnr
        metrics[f"max_diff_{name}"] = max_diff
        metrics[f"mean_diff_{name}"] = mean_diff

    psnr_all = _compute_psnr(orig_norm, dec_norm)
    max_diff_all = float(np.max(np.abs(orig_norm - dec_norm)))
    metrics["psnr_all"] = psnr_all
    metrics["max_diff_all"] = max_diff_all
    metrics["mean_diff_all"] = float(np.mean(np.abs(orig_norm - dec_norm)))

    # Save diff figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        abs_diff = np.abs(orig_norm - dec_norm)  # [3, H, W]
        # Composite diff: max across channels for heatmap
        diff_heatmap = abs_diff.max(axis=0)      # [H, W]

        # Convert CHW to HWC RGB for display
        orig_hwc = np.clip(orig_norm.transpose(1, 2, 0), 0, 1)
        dec_hwc = np.clip(dec_norm.transpose(1, 2, 0), 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_hwc)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(dec_hwc)
        axes[1].set_title("BC6H Decoded")
        axes[1].axis("off")

        im = axes[2].imshow(diff_heatmap, cmap="hot", vmin=0, vmax=0.05)
        axes[2].set_title(f"Abs Diff (max ch)\nPSNR={psnr_all:.1f}dB  max={max_diff_all:.4f}")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], shrink=0.8)

        fig.suptitle(f"{stem}  ({W}x{H})")
        fig.tight_layout()

        diff_png = out_dir / f"{stem}.diff.png"
        fig.savefig(str(diff_png), dpi=100, bbox_inches="tight")
        plt.close(fig)
        metrics["diff_png"] = str(diff_png)
        print(f"  [diff] saved {diff_png.name}")
    except Exception as e:
        print(f"  [diff] matplotlib not available or error: {e}")
        metrics["diff_png"] = None

    return metrics


# ---------------------------------------------------------------------------
# RGBA16F DDS helper (kept for internal use / debugging)
# ---------------------------------------------------------------------------

def _tensor_to_rgba16f_bytes(tex_chw: torch.Tensor, bc6_signed_mode: bool) -> bytes:
    if tex_chw.ndim != 3 or tex_chw.shape[0] < 3:
        raise ValueError(f"Expected CHW with >=3 channels, got {tuple(tex_chw.shape)}")
    x = tex_chw[:3].detach().cpu().float().clamp(-1.0, 1.0)
    if not bc6_signed_mode:
        x = (x + 1.0) * 0.5
    alpha = torch.ones((1, x.shape[1], x.shape[2]), dtype=x.dtype)
    rgba = torch.cat([x, alpha], dim=0).permute(1, 2, 0).contiguous().to(torch.float16)
    return rgba.numpy().tobytes()


# ---------------------------------------------------------------------------
# Main encoding pipeline
# ---------------------------------------------------------------------------

def _encode_latent_to_bc6h_dds(
    mip_tensors: List[torch.Tensor],
    latent_idx: int,
    out_dir: Path,
    signed_mode: bool,
    verify: bool,
) -> Dict:
    """
    Encode a list of mip-level tensors ([3,H,W] float32 in [-1,1]) to BC6H DDS.

    mip_tensors: list of torch.Tensor, one per mip level (mip0 first).
    latent_idx: integer index for naming.
    out_dir: directory to write the .dds and optional diff PNG.
    signed_mode: True for BC6H_SF16, False for BC6H_UF16.
    verify: if True, decode and compute diff metrics.

    Returns a dict with export metadata.
    """
    if not mip_tensors:
        raise ValueError("No mip tensors provided.")

    mip0 = mip_tensors[0]
    H0 = int(mip0.shape[1])
    W0 = int(mip0.shape[2])
    mip_count = len(mip_tensors)

    mip_bytes_list = []
    for m_idx, tensor in enumerate(mip_tensors):
        chw = tensor.detach().cpu().float().clamp(-1.0, 1.0).numpy()  # [3, H, W] in [-1,1]

        if signed_mode:
            # Use [-1, 1] directly
            encoded_data = _encode_bc6h_mip(chw, signed_mode=True)
        else:
            # Remap [-1, 1] -> [0, 1] for UF16 encoding
            chw_uf16 = (chw + 1.0) * 0.5
            encoded_data = _encode_bc6h_mip(chw_uf16, signed_mode=False)

        mip_bytes_list.append(encoded_data)

    dst_dds = out_dir / f"latent_{latent_idx:02d}.bc6.dds"
    _write_bc6h_dds(
        mip_bytes_list=mip_bytes_list,
        w0=W0,
        h0=H0,
        out_path=dst_dds,
        signed_mode=signed_mode,
    )

    dst_meta = _read_dds_metadata(dst_dds)
    entry = {
        "latent_index": latent_idx,
        "mip_count": mip_count,
        "size": [W0, H0],
        "dds": str(dst_dds),
        "dds_size_bytes": int(dst_dds.stat().st_size),
        "dds_mip_count": int(dst_meta["mip_count"]),
        "dds_dxgi_format": int(dst_meta.get("dxgi_format", -1)),
    }

    if verify:
        verify_metrics_all = []
        for m_idx, (tensor, mip_data) in enumerate(zip(mip_tensors, mip_bytes_list)):
            chw_orig = tensor.detach().cpu().float().clamp(-1.0, 1.0).numpy()
            H_m = int(tensor.shape[1])
            W_m = int(tensor.shape[2])

            decoded_chw = _decode_bc6h_mip(mip_data, W_m, H_m, signed_mode=signed_mode)

            if signed_mode:
                # Both are in [-1, 1]
                orig_for_diff = chw_orig
                dec_for_diff = decoded_chw
            else:
                # Encode stored [0,1]; original is [-1,1] → remap original to [0,1]
                orig_for_diff = (chw_orig + 1.0) * 0.5
                dec_for_diff = decoded_chw  # already [0,1]

            stem = f"latent_{latent_idx:02d}_mip_{m_idx:02d}"
            vm = _verify_and_diff(
                original_chw=orig_for_diff,
                decoded_chw=dec_for_diff,
                out_dir=out_dir / "metadata",
                stem=stem,
                signed_mode=False,  # comparison always in [0,1] space
            )
            vm["mip_index"] = m_idx
            verify_metrics_all.append(vm)

            psnr_str = f"{vm['psnr_all']:.1f}" if vm["psnr_all"] != float("inf") else "inf"
            print(
                f"  [verify] mip {m_idx}: {W_m}x{H_m}  "
                f"PSNR={psnr_str}dB  max_diff={vm['max_diff_all']:.4f}"
            )

        entry["verify"] = verify_metrics_all

    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Pure-Python BC6H encoder: convert exported latent .pt tensors "
            "to BC6H_UF16 or BC6H_SF16 DDS files with full mip chain."
        )
    )
    ap.add_argument(
        "--export-dir", type=Path, required=True,
        help="Directory containing metadata.json and latent .pt files (or latents/ subdir).",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output folder for BC6H DDS files (default: <export-dir>; DDS files written to root, diffs/report to <out-dir>/metadata/).",
    )
    ap.add_argument(
        "--bc6-format", type=str, default=None, choices=["UF16", "SF16"],
        help="BC6H format override: UF16 (unsigned) or SF16 (signed). Default: from metadata.json.",
    )
    ap.add_argument(
        "--latent-index", type=int, default=None,
        help="Encode only this latent index (default: all latents).",
    )
    ap.add_argument(
        "--max-latents", type=int, default=0,
        help="Cap on number of latents to encode (0 = all).",
    )
    ap.add_argument(
        "--no-verify", action="store_true",
        help="Skip decoding verification and diff figure generation.",
    )
    args = ap.parse_args()

    export_dir = args.export_dir.resolve()
    meta_path = export_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text())

    bc6_signed = bool(meta.get("bc6_signed_mode", False))
    if args.bc6_format is not None:
        bc6_signed = args.bc6_format == "SF16"

    out_dir = (args.out_dir or export_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)

    grouped = _group_mips_by_latent(
        meta=meta,
        export_dir=export_dir,
        latent_index=args.latent_index,
        max_latents=args.max_latents,
    )
    if not grouped:
        raise RuntimeError("No latent entries matched filters.")

    verify = not args.no_verify
    bc6_format_name = "BC6H_SF16" if bc6_signed else "BC6H_UF16"
    print(f"[bc6h] Encoding {len(grouped)} latent(s) as {bc6_format_name} (pure-Python Mode 11)")

    exported = []
    for latent_idx, mip_entries in grouped:
        mip_tensors = [
            torch.load(e["pt_path"], map_location="cpu", weights_only=True).float()
            for e in mip_entries
        ]
        H0 = int(mip_tensors[0].shape[1])
        W0 = int(mip_tensors[0].shape[2])
        print(
            f"[bc6h] latent {latent_idx:02d}: {W0}x{H0}  "
            f"mips={len(mip_tensors)}  signed={bc6_signed}"
        )
        entry = _encode_latent_to_bc6h_dds(
            mip_tensors=mip_tensors,
            latent_idx=latent_idx,
            out_dir=out_dir,
            signed_mode=bc6_signed,
            verify=verify,
        )
        entry["mip_indices"] = [int(e["mip_index"]) for e in mip_entries]
        entry["source_pt_mips"] = [str(e["pt_path"]) for e in mip_entries]
        exported.append(entry)
        print(f"  [ok] -> {out_dir / f'latent_{latent_idx:02d}.bc6.dds'}")

    report = {
        "export_dir": str(export_dir),
        "out_dir": str(out_dir),
        "encoder": "pure-python-bc6h-mode11",
        "bc6_format": bc6_format_name,
        "bc6_signed_mode": bc6_signed,
        "expected_dxgi": DXGI_FORMAT_BC6H_SF16 if bc6_signed else DXGI_FORMAT_BC6H_UF16,
        "note": (
            "DDS files are true BC6H Mode 11 bitstreams encoded in pure Python. "
            "Endpoints use 11-bit quantization; indices are 3-bit (8 interpolation steps)."
        ),
        "files": exported,
    }
    rep_path = out_dir / "metadata" / "true_bc6_export_report.json"
    rep_path.write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {len(exported)} DDS file(s) to {out_dir} and report -> {rep_path}")


if __name__ == "__main__":
    main()
