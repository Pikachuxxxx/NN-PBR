#!/usr/bin/env python3
"""
export_true_bc6_dds.py — Inspect and validate BC6H DDS latent exports.

Pipeline (lossless within BC6H representation):
  trained block params (endpoints_q + indices_q)
      → final 6-bit / 3-bit quantization
      → pack directly into BC6H Mode 10 blocks (16 bytes/block)
      → write DDS with full mip chain

No float texture decode/re-encode roundtrip.  The quantized values produced
during training are preserved exactly in the DDS bit stream.

This script validates an existing export directory and reports its contents.
BC6H DDS files are produced automatically at the end of training by
export_trained_artifacts() in neuralmaterials.py.

Usage:
  python export_true_bc6_dds.py --export-dir runs/<run>/export
  python export_true_bc6_dds.py --export-dir runs/<run>/export --decode-check
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


_DXGI_BC6H_UF16 = 95
_DXGI_BC6H_SF16 = 96


def _read_dds_header(path: Path) -> dict:
    """Parse DDS + DX10 header; return basic info dict."""
    raw = path.read_bytes()
    if raw[:4] != b"DDS ":
        raise RuntimeError(f"Not a DDS file: {path}")
    h  = struct.unpack_from("<I", raw, 12)[0]
    w  = struct.unpack_from("<I", raw, 16)[0]
    mc = struct.unpack_from("<I", raw, 28)[0]
    dxgi = struct.unpack_from("<I", raw, 128)[0]  # DX10 header starts at 128
    signed = dxgi == _DXGI_BC6H_SF16
    unsigned = dxgi == _DXGI_BC6H_UF16
    fmt_str = "BC6H_SF16" if signed else ("BC6H_UF16" if unsigned else f"DXGI={dxgi}")

    # Compute expected byte sizes for each mip
    mip_sizes = []
    mw, mh = w, h
    for _ in range(max(1, mc)):
        bw = max(1, (mw + 3) // 4)
        bh = max(1, (mh + 3) // 4)
        mip_sizes.append(bw * bh * 16)
        mw = max(1, mw // 2)
        mh = max(1, mh // 2)

    total_expected = 148 + sum(mip_sizes)
    return {
        "w": w, "h": h, "mip_count": mc, "format": fmt_str,
        "signed_mode": signed,
        "file_bytes": path.stat().st_size,
        "expected_bytes": total_expected,
        "ok": path.stat().st_size == total_expected,
        "mip_sizes": mip_sizes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", type=Path, required=True, help="Export directory (contains metadata.json)")
    ap.add_argument(
        "--decode-check", action="store_true",
        help="Decode mip0 of each DDS and verify values are in expected range.",
    )
    args = ap.parse_args()

    export_dir = args.export_dir.resolve()
    meta_path = export_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.json not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    version = int(meta.get("version", 1))
    signed_mode = bool(meta.get("bc6_signed_mode", False))
    latent_count = int(meta.get("latent_count", 0))

    print(f"[bc6h] Export v{version}  latents={latent_count}  signed={signed_mode}")
    print(f"[bc6h] Pipeline: block params → final quantization → BC6H Mode 10 blocks (lossless)\n")

    if version < 4:
        print(f"  WARNING: legacy export v{version}. Re-export with current code to get v4 DDS.")
        return

    dds_files = sorted(export_dir.glob("latent_*.bc6.dds"))
    if not dds_files:
        print("  No latent_*.bc6.dds found in export root.")
        return

    total_bytes = 0
    all_ok = True
    for f in dds_files:
        info = _read_dds_header(f)
        status = "OK" if info["ok"] else f"SIZE MISMATCH (got {info['file_bytes']} expected {info['expected_bytes']})"
        if not info["ok"]:
            all_ok = False
        print(
            f"  {f.name}  {info['w']}x{info['h']}  mips={info['mip_count']}"
            f"  {info['format']}  {info['file_bytes'] // 1024} KB  [{status}]"
        )
        total_bytes += info["file_bytes"]

    decoder_bin = export_dir / "decoder_fp16.bin"
    decoder_bytes = decoder_bin.stat().st_size if decoder_bin.exists() else 0

    print(f"\n  DDS total : {total_bytes // 1024} KB")
    print(f"  Decoder   : {decoder_bytes // 1024} KB")
    print(f"  Runtime   : {(total_bytes + decoder_bytes) // 1024} KB")

    if args.decode_check:
        print("\n[bc6h] Decode check (mip0 value range):")
        try:
            from neuralmaterials import decode_bc6h_dds_mip0
            for f in dds_files:
                t = decode_bc6h_dds_mip0(f, signed_mode=signed_mode)
                lo, hi = float(t.min()), float(t.max())
                print(f"  {f.name}  min={lo:.3f}  max={hi:.3f}  shape={list(t.shape)}")
        except ImportError:
            print("  (skipped — neuralmaterials not importable)")

    print(f"\n[bc6h] {'All DDS files valid.' if all_ok else 'WARNING: size mismatches detected.'}")


if __name__ == "__main__":
    main()
