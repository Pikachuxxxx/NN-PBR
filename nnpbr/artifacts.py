"""Runtime artifact export, loading, and validation."""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import torch
import json
import hashlib
from PIL import Image
from .model import NeuralMaterialCompressionModel, MaterialDecoderMLP
from .codec import decode_bc6h_dds
from bc6h_spec import BC6H_MODE10_ENDPOINT_BITS, BC6H_MODE10_INDEX_BITS
from .codec import (
    _pack_mip_blocks_to_bc6h_mode10_bytes,
    _decode_mode10_params_to_mip,
    _decode_mode10_bytes_to_mip,
    _write_bc6h_dds,
)
from .sampling import sample_mips_trilinear
from .hlsl import write_hlsl


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


@torch.no_grad()
def export_trained_artifacts(model: NeuralMaterialCompressionModel, out_dir: Path):
    """
    Export all runtime artifacts to out_dir.

    Runtime files (export root):
      decoder_fp16.bin, metadata.json, latent_XX.bc6.dds

    Debug files (metadata/ subdir):
      decoder_state.pt, latent_XX_mip_YY.png

    The intended final path is paper-aligned BC6H Mode 10 export from trained
    block params, with final 6-bit endpoint quantization and 3-bit index
    quantization as described in §5.2. PNG previews are decoded from the exported bytes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    if model.bc6_signed_mode:
        raise NotImplementedError(
            "Spec-correct BC6H Mode 10 export currently supports unsigned BC6H only."
        )
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Cannot export nonfinite model parameters")

    # --- Decoder weights ---
    state = model.decoder.state_dict()
    torch.save(state, meta_dir / "decoder_state.pt")
    flat = []
    for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
        flat.append(state[k].detach().to(torch.float16).contiguous().view(-1))
    weights = torch.cat(flat).cpu()
    if not torch.isfinite(weights).all():
        raise ValueError("Decoder overflows FP16")
    fp16_blob = weights.numpy().astype("<f2").tobytes()
    (out_dir / "decoder_fp16.bin").write_bytes(fp16_blob)

    # --- Latent DDS (BC6H from block params) + PNG previews ---
    latent_files = []
    for i, pyr in enumerate(model.bc_pyramids):
        # Block params: no float32 decode needed for the DDS
        all_params = pyr.export_quantized_params()
        # Shapes only; previews below come from exact packed bytes.
        decoded_mips = pyr.decode_mips(hard_partition=True)

        mip_bytes_list = []
        for m, (params, tex) in enumerate(zip(all_params, decoded_mips)):
            stem = f"latent_{i:02d}_mip_{m:02d}"

            # BC6H bytes packed directly from quantized block params (Mode 10)
            # Pass partition_bank for correct anchor bit packing
            partition_bank_np = model.partition_bank.cpu().numpy()
            packed_bytes = _pack_mip_blocks_to_bc6h_mode10_bytes(
                params, model.bc6_signed_mode, partition_bank=partition_bank_np
            )
            expected_pixels = _decode_mode10_params_to_mip(
                params,
                partition_bank=partition_bank_np,
                signed_mode=model.bc6_signed_mode,
            )
            decoded_pixels = _decode_mode10_bytes_to_mip(
                packed_bytes,
                w=int(params["w"]),
                h=int(params["h"]),
                partition_bank=partition_bank_np,
                signed_mode=model.bc6_signed_mode,
            )
            if not np.array_equal(expected_pixels, decoded_pixels):
                max_abs = float(np.max(np.abs(expected_pixels - decoded_pixels)))
                raise RuntimeError(
                    f"BC6H Mode 10 pack/decode mismatch on {stem}: max_abs={max_abs:.6e}"
                )
            save_chw_png_ldr(
                torch.from_numpy(decoded_pixels).permute(2, 0, 1),
                meta_dir / f"{stem}.png",
            )
            mip_bytes_list.append(packed_bytes)
            latent_files.append(
                {
                    "latent_index": i,
                    "mip_index": m,
                    "shape_chw": list(tex.shape),
                    "png": f"metadata/{stem}.png",
                }
            )

        # Write DDS to export root
        W0, H0 = int(all_params[0]["w"]), int(all_params[0]["h"])
        dds_path = out_dir / f"latent_{i:02d}.bc6.dds"
        _write_bc6h_dds(
            mip_bytes_list, W0, H0, dds_path, signed_mode=model.bc6_signed_mode
        )
        print(
            f"[export] latent {i:02d}: {W0}×{H0}  mips={len(mip_bytes_list)}  -> {dds_path.name}"
        )

    meta = {
        "version": 5,
        "reference_resolution": model.ref_base_res,
        "address_mode": "clamp_to_edge",
        "channel_schema": "albedo_srgb01_normal_xy_signed_orm_signed_v1",
        "latent_count": model.n_latent,
        "latent_resolutions": model.latent_resolutions,
        "lod_biases": model.lod_biases,
        "bc6_signed_mode": model.bc6_signed_mode,
        "bc6_format": "DXGI_FORMAT_BC6H_UF16",
        "bc6_mode": 10,
        "endpoint_bits": BC6H_MODE10_ENDPOINT_BITS,
        "index_bits": BC6H_MODE10_INDEX_BITS,
        "decoder": {
            "in_dim": int(model.decoder.fc1.in_features),
            "hidden_dim": int(model.decoder.fc1.out_features),
            "out_dim": int(model.decoder.fc2.out_features),
            "weights_fp16_blob": "decoder_fp16.bin",
            "state_dict": "metadata/decoder_state.pt",
        },
        "latent_files": latent_files,
        "sha256": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in ["decoder_fp16.bin"]
            + [f"latent_{i:02d}.bc6.dds" for i in range(4)]
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    write_hlsl(out_dir)
    print(f"[export] wrote artifacts to {out_dir}")


class ExportedMaterial:
    """Strict runtime reader: only DDS bytes and the FP16 weight blob are inputs."""

    def __init__(self, directory: Path, device="cpu"):
        directory = Path(directory)
        self.meta = meta = json.loads((directory / "metadata.json").read_text())
        if (
            meta.get("version") != 5
            or meta.get("latent_count") != 4
            or meta.get("bc6_format") != "DXGI_FORMAT_BC6H_UF16"
            or meta.get("bc6_signed_mode") is not False
            or meta.get("bc6_mode") != 10
            or meta.get("endpoint_bits") != 6
            or meta.get("index_bits") != 3
            or meta.get("address_mode") != "clamp_to_edge"
            or meta.get("channel_schema")
            != "albedo_srgb01_normal_xy_signed_orm_signed_v1"
        ):
            raise ValueError(
                "Unsupported export contract; re-export with current training code"
            )
        dims = meta["decoder"]
        if [dims.get(k) for k in ("in_dim", "hidden_dim", "out_dim")] != [12, 16, 8]:
            raise ValueError("Runtime decoder must be 12-16-8")
        self.biases = meta["lod_biases"]
        resolutions = meta["latent_resolutions"]
        reference = meta["reference_resolution"]
        if len(self.biases) != 4 or len(resolutions) != 4 or reference < 4:
            raise ValueError("Invalid resolution metadata")
        if any(
            not math.isfinite(b) or abs(b - math.log2(r / reference)) > 1e-6
            for r, b in zip(resolutions, self.biases)
        ):
            raise ValueError("LOD biases disagree with reference resolution")
        expected = {f"latent_{i:02d}.bc6.dds" for i in range(4)}
        if {p.name for p in directory.glob("latent_*.bc6.dds")} != expected:
            raise ValueError("Expected exactly four latent DDS files")
        hashes = meta.get("sha256", {})
        if set(hashes) != expected | {"decoder_fp16.bin"}:
            raise ValueError("Missing runtime file fingerprints")
        for name, digest in hashes.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Runtime file fingerprint mismatch: {name}")
        if any(e["latent_index"] not in range(4) for e in meta["latent_files"]):
            raise ValueError("Unexpected latent inventory entry")
        self.mips = []
        for i, r in enumerate(resolutions):
            mips = decode_bc6h_dds(directory / f"latent_{i:02d}.bc6.dds", False)
            entries = [e for e in meta["latent_files"] if e["latent_index"] == i]
            if len(entries) != len(mips) or list(mips[0].shape) != [3, r, r]:
                raise ValueError("DDS dimensions or mip count disagree with metadata")
            for j, (m, e) in enumerate(zip(mips, entries)):
                if (
                    e["mip_index"] != j
                    or e["shape_chw"] != list(m.shape)
                    or not torch.isfinite(m).all()
                ):
                    raise ValueError("Invalid latent mip")
            self.mips.append([m.to(device) for m in mips])
        raw = (directory / "decoder_fp16.bin").read_bytes()
        if len(raw) != 688:
            raise ValueError(
                "FP16 decoder requires exactly 344 weights/biases (688 bytes)"
            )
        values = torch.from_numpy(
            np.frombuffer(raw, dtype="<f2").astype(np.float32)
        ).to(device)
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite FP16 weights")
        self.decoder = MaterialDecoderMLP(12, 16, 8).to(device)
        offset = 0
        with torch.no_grad():
            for parameter in self.decoder.parameters():
                count = parameter.numel()
                parameter.copy_(values[offset : offset + count].view_as(parameter))
                offset += count
        self.decoder.eval()

    def __call__(self, uv, lod):
        features = torch.cat(
            [
                sample_mips_trilinear(m, uv, lod + b, "bilinear")
                for m, b in zip(self.mips, self.biases)
            ],
            dim=1,
        )
        return self.decoder(features)

    forward_bc = __call__
