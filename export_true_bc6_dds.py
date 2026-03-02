#!/usr/bin/env python3
"""
Export true BC6 DDS files from exported latent tensors using an external encoder.

This creates one mip-chained source DDS per latent (from latent_XX_mip_YY.pt),
then encodes it to an actual BC6 DDS bitstream with a real BC encoder CLI
(e.g. CompressonatorCLI).
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# DDS constants.
DDS_MAGIC = b"DDS "
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000

DDSCAPS_COMPLEX = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS_MIPMAP = 0x400000

DDPF_FOURCC = 0x4

DXGI_FORMAT_R16G16B16A16_FLOAT = 10
DXGI_FORMAT_BC6H_UF16 = 95
DXGI_FORMAT_BC6H_SF16 = 96
D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3


def _pick_cli(explicit_cli: Optional[str]) -> str:
    if explicit_cli:
        path = shutil.which(explicit_cli)
        if path is None:
            raise RuntimeError(f"BC encoder CLI not found in PATH: {explicit_cli}")
        return path

    for candidate in ("compressonatorcli", "CompressonatorCLI", "compressonator"):
        path = shutil.which(candidate)
        if path is not None:
            return path
    raise RuntimeError(
        "No BC6 encoder CLI found. Install Compressonator CLI and ensure it is in PATH, "
        "or pass --bc6-cli <path>."
    )


def _run_cmd(cmd: List[str]) -> Tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode), proc.stdout


def _encode_with_fallbacks(cli: str, src_dds: Path, dst_dds: Path, bc6_format: str):
    attempts = [
        [cli, "-fd", bc6_format, str(src_dds), str(dst_dds)],
        [cli, str(src_dds), str(dst_dds), "-fd", bc6_format],
        [cli, "-fd", bc6_format, "-miplevels", "99", str(src_dds), str(dst_dds)],
    ]
    logs = []
    for cmd in attempts:
        code, out = _run_cmd(cmd)
        logs.append({"cmd": cmd, "code": code, "output": out[-2000:]})
        if code == 0 and dst_dds.exists() and dst_dds.stat().st_size > 0:
            return cmd, out
    raise RuntimeError(
        f"Failed to encode {src_dds.name} -> {dst_dds.name}\n"
        f"Attempts:\n{json.dumps(logs, indent=2)}"
    )


def _decode_smoke(cli: str, src_dds: Path, out_png: Path):
    attempts = [
        [cli, str(src_dds), str(out_png)],
        [cli, "-fd", "ARGB_8888", str(src_dds), str(out_png)],
    ]
    for cmd in attempts:
        code, _ = _run_cmd(cmd)
        if code == 0 and out_png.exists() and out_png.stat().st_size > 0:
            return cmd
    raise RuntimeError(f"Failed to decode smoke PNG for {src_dds}")


def _tensor_to_rgba16f_bytes(tex_chw: torch.Tensor, bc6_signed_mode: bool) -> bytes:
    if tex_chw.ndim != 3 or tex_chw.shape[0] < 3:
        raise ValueError(f"Expected CHW with >=3 channels, got {tuple(tex_chw.shape)}")
    x = tex_chw[:3].detach().cpu().float().clamp(-1.0, 1.0)
    if not bc6_signed_mode:
        x = (x + 1.0) * 0.5
    alpha = torch.ones((1, x.shape[1], x.shape[2]), dtype=x.dtype)
    rgba = torch.cat([x, alpha], dim=0).permute(1, 2, 0).contiguous().to(torch.float16)
    return rgba.numpy().tobytes()


def _write_rgba16f_mipchain_dds(mips: Sequence[torch.Tensor], out_path: Path, bc6_signed_mode: bool):
    if not mips:
        raise ValueError("No mip tensors provided.")
    h0 = int(mips[0].shape[1])
    w0 = int(mips[0].shape[2])
    mip_count = len(mips)

    for i, m in enumerate(mips):
        h_exp = max(1, h0 >> i)
        w_exp = max(1, w0 >> i)
        if int(m.shape[1]) != h_exp or int(m.shape[2]) != w_exp:
            raise ValueError(
                f"Mip shape mismatch at level {i}: got {tuple(m.shape)}, expected [3,{h_exp},{w_exp}]"
            )

    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_PITCH
    if mip_count > 1:
        flags |= DDSD_MIPMAPCOUNT

    caps = DDSCAPS_TEXTURE
    if mip_count > 1:
        caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP

    pitch = w0 * 8  # R16G16B16A16_FLOAT bytes per row.

    dds_header = struct.pack(
        "<IIIIIII11I",
        124,  # dwSize
        flags,
        h0,
        w0,
        pitch,  # dwPitchOrLinearSize
        0,  # dwDepth
        mip_count,
        *([0] * 11),  # dwReserved1[11]
    )
    dds_pixelformat = struct.pack(
        "<II4sIIIII",
        32,  # dwSize
        DDPF_FOURCC,  # dwFlags
        b"DX10",
        0,  # dwRGBBitCount
        0,  # dwRBitMask
        0,  # dwGBitMask
        0,  # dwBBitMask
        0,  # dwABitMask
    )
    dds_caps = struct.pack(
        "<IIIII",
        caps,  # dwCaps
        0,  # dwCaps2
        0,  # dwCaps3
        0,  # dwCaps4
        0,  # dwReserved2
    )
    dx10_header = struct.pack(
        "<IIIII",
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D10_RESOURCE_DIMENSION_TEXTURE2D,
        0,  # miscFlag
        1,  # arraySize
        0,  # miscFlags2
    )

    payload = bytearray()
    for m in mips:
        payload.extend(_tensor_to_rgba16f_bytes(m, bc6_signed_mode=bc6_signed_mode))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(DDS_MAGIC)
        f.write(dds_header)
        f.write(dds_pixelformat)
        f.write(dds_caps)
        f.write(dx10_header)
        f.write(payload)


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


def _group_mips_by_latent(
    meta: Dict,
    lat_dir: Path,
    latent_index: Optional[int],
    mip_index: Optional[int],
    max_latents: int,
) -> List[Tuple[int, List[Dict]]]:
    grouped: Dict[int, List[Dict]] = collections.defaultdict(list)
    for e in meta.get("latent_files", []):
        li = int(e["latent_index"])
        mi = int(e["mip_index"])
        if latent_index is not None and li != latent_index:
            continue
        if mip_index is not None and mi != mip_index:
            continue
        pt_name = e.get("pt")
        if not pt_name:
            continue
        pt_path = lat_dir / pt_name
        if not pt_path.exists():
            raise RuntimeError(f"Missing latent tensor file: {pt_path}")
        grouped[li].append({"mip_index": mi, "pt_path": pt_path, "shape_chw": e.get("shape_chw")})

    items = sorted(grouped.items(), key=lambda kv: kv[0])
    if max_latents > 0:
        items = items[:max_latents]
    for _, arr in items:
        arr.sort(key=lambda x: int(x["mip_index"]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, required=True, help="Directory containing metadata.json and latents/")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output folder for true BC6 DDS files")
    ap.add_argument("--bc6-cli", type=str, default=None, help="External BC6 encoder CLI executable")
    ap.add_argument("--bc6-format", type=str, default=None, help="Encoder format name (default: BC6H or BC6H_SF from metadata)")
    ap.add_argument("--latent-index", type=int, default=None, help="Optional filter")
    ap.add_argument("--mip-index", type=int, default=None, help="Optional filter")
    ap.add_argument("--max-latents", type=int, default=0, help="Optional cap on number of latents to export (0 = all)")
    ap.add_argument("--decode-smoke", action="store_true", help="Decode first exported DDS back to PNG as sanity check")
    ap.add_argument(
        "--keep-source-dds",
        action="store_true",
        help="Keep intermediate mip-chained source DDS files used for BC6 encoding.",
    )
    args = ap.parse_args()

    export_dir = args.export_dir
    meta_path = export_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text())

    cli = _pick_cli(args.bc6_cli)
    bc6_signed = bool(meta.get("bc6_signed_mode", False))
    bc6_format = args.bc6_format or ("BC6H_SF" if bc6_signed else "BC6H")

    lat_dir = export_dir / "latents"
    out_dir = args.out_dir or (export_dir / "true_bc6_dds")
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = _group_mips_by_latent(
        meta=meta,
        lat_dir=lat_dir,
        latent_index=args.latent_index,
        mip_index=args.mip_index,
        max_latents=args.max_latents,
    )
    if not grouped:
        raise RuntimeError("No latent entries matched filters.")

    exported = []
    first_dds: Optional[Path] = None
    temp_ctx = tempfile.TemporaryDirectory(prefix="nnpbr_dds_src_") if not args.keep_source_dds else None
    try:
        if args.keep_source_dds:
            src_root = out_dir / "_source_dds_rgba16f"
            src_root.mkdir(parents=True, exist_ok=True)
        else:
            src_root = Path(temp_ctx.name)

        for latent_idx, mip_entries in grouped:
            mip_tensors = [torch.load(e["pt_path"], map_location="cpu").float() for e in mip_entries]
            src_dds = src_root / f"latent_{latent_idx:02d}.rgba16f.dds"
            _write_rgba16f_mipchain_dds(
                mips=mip_tensors,
                out_path=src_dds,
                bc6_signed_mode=bc6_signed,
            )
            src_meta = _read_dds_metadata(src_dds)

            dst_dds = out_dir / f"latent_{latent_idx:02d}.bc6.dds"
            cmd, _ = _encode_with_fallbacks(cli=cli, src_dds=src_dds, dst_dds=dst_dds, bc6_format=bc6_format)
            dst_meta = _read_dds_metadata(dst_dds)
            if int(dst_meta["mip_count"]) < int(src_meta["mip_count"]):
                raise RuntimeError(
                    f"Encoder output dropped mips for latent {latent_idx:02d}: "
                    f"src mips={src_meta['mip_count']} dst mips={dst_meta['mip_count']}. "
                    "Use a BC6 CLI mode that preserves source mip chains."
                )
            if first_dds is None:
                first_dds = dst_dds

            exported.append(
                {
                    "latent_index": int(latent_idx),
                    "mip_indices": [int(e["mip_index"]) for e in mip_entries],
                    "source_pt_mips": [str(e["pt_path"]) for e in mip_entries],
                    "source_rgba16f_dds": str(src_dds),
                    "source_mip_count": int(src_meta["mip_count"]),
                    "source_size": [int(src_meta["width"]), int(src_meta["height"])],
                    "dds": str(dst_dds),
                    "dds_size_bytes": int(dst_dds.stat().st_size),
                    "dds_mip_count": int(dst_meta["mip_count"]),
                    "dds_size": [int(dst_meta["width"]), int(dst_meta["height"])],
                    "dds_dxgi_format": int(dst_meta.get("dxgi_format", -1)),
                    "encode_cmd": cmd,
                }
            )
            print(
                f"[ok] latent {latent_idx:02d}: "
                f"{src_meta['width']}x{src_meta['height']} mips={src_meta['mip_count']} "
                f"-> {dst_dds.name}"
            )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    smoke_png = None
    smoke_cmd = None
    if args.decode_smoke and first_dds is not None:
        smoke_png = out_dir / f"{first_dds.stem}.decoded_smoke.png"
        smoke_cmd = _decode_smoke(cli=cli, src_dds=first_dds, out_png=smoke_png)
        print(f"[smoke] {first_dds.name} -> {smoke_png.name}")

    report = {
        "export_dir": str(export_dir),
        "encoder_cli": cli,
        "bc6_format": bc6_format,
        "note": "DDS files are true BC6 bitstreams produced by external encoder CLI from latent tensor mips.",
        "source_path": "latent .pt tensors -> intermediate mip-chained RGBA16F DDS -> BC6 DDS",
        "warning": (
            "These BC6 DDS files are engine-ready GPU textures. They are encoded from exported latent tensors, "
            "not from PNG previews."
        ),
        "expected_dxgi": DXGI_FORMAT_BC6H_SF16 if bc6_signed else DXGI_FORMAT_BC6H_UF16,
        "files": exported,
        "decode_smoke_png": str(smoke_png) if smoke_png else None,
        "decode_smoke_cmd": smoke_cmd,
    }
    rep_path = out_dir / "true_bc6_export_report.json"
    rep_path.write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {rep_path}")


if __name__ == "__main__":
    main()
